# COMMAND UNTUK MENJALANKAN:
# uvicorn clipper:app --host 0.0.0.0 --port 8000 --reload --timeout-keep-alive 600

import os
import subprocess
import torch
import math
import time
import gc
import requests 
from datetime import timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
# --- PERBAIKAN IMPORT ---
from faster_whisper import WhisperModel, BatchedInferencePipeline 
import yt_dlp

# --- KONFIGURASI GLOBAL ---
BASE_DIR = r"D:\applications\clipping-engine\media"
BASE_TEMP_DIR = r"D:\AI_Data"
app = FastAPI()

# --- PATH ---
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
TEMP_DIR = os.path.join(BASE_TEMP_DIR, "temp")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# --- MODELS ---
class DownloadRequest(BaseModel):
    url: str

class ClipRequest(BaseModel):
    filename: str
    start: float
    end: float
    text: str = ""

class TranscribeRequest(BaseModel):
    filename: str
    callback_url: str 

# --- FUNGSI BACKGROUND (WORKER) ---
def process_transcription_background(filename: str, callback_url: str):
    print(f"\n[START] Processing: {filename}")
    file_path = os.path.join(DOWNLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {filename}")
        return

    try:
        # 1. Load Model Dasar
        print("[INFO] Loading Model Large-v3...")
        model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16", download_root=r"D:\AI_Data\huggingface")

        # 2. Bungkus dengan Pipeline Batching (OPTIMASI RTX 4050)
        # Ini yang membuat prosesnya bisa paralel (lebih cepat)
        print("[INFO] Mengaktifkan Batched Inference...")
        batched_model = BatchedInferencePipeline(model=model)

        # 3. Transcribe menggunakan batched_model
        print("[INFO] Transcribing (Batch Mode)...")
        # batch_size=8 aman untuk VRAM 6GB. Kalau crash, turunkan ke 4.
        segments, info = batched_model.transcribe(file_path, batch_size=8)
        
        transcript_data = []
        full_text = ""
        
        # Konversi generator ke list agar bisa diproses
        for segment in segments:
            transcript_data.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
            full_text += f"[{segment.start:.2f}-{segment.end:.2f}] {segment.text.strip()}\n"

        # Siapkan Paket Data
        payload = {
            "status": "success",
            "filename": filename,
            "full_text": full_text,
            "segments": transcript_data
        }

        # --- KIRIM BALIK KE N8N (CALLBACK) ---
        print(f"[CALLBACK] Mengirim hasil ke: {callback_url}")
        requests.post(callback_url, json=payload)
        print("[DONE] Data terkirim ke n8n!")

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        try:
            requests.post(callback_url, json={"status": "error", "message": str(e)})
        except:
            print("[CRITICAL] Gagal mengirim error callback ke n8n")

# --- HELPER FUNCTIONS ---
def cleanup_files(file_paths):
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
                print(f"[CLEANUP] Berhasil menghapus: {path}")
            except:
                pass

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def create_srt_file(text, duration, output_path):
    content = f"1\n00:00:00,000 --> {format_timestamp(duration)}\n{text}\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "online", "mode": "Async Webhook + Batch Processing"}

@app.post("/download")
def download_video(req: DownloadRequest):
    try:
        ydl_opts = {
            'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': f'{DOWNLOAD_DIR}/%(id)s.%(ext)s',
            'noplaylist': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(req.url, download=True)
            filename = ydl.prepare_filename(info)
            basename = os.path.basename(filename)
            
        return {
            "status": "success", 
            "filename": basename, 
            "full_path": filename,
            "duration": info.get('duration'),
            "title": info.get('title')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.post("/transcribe")
def transcribe_video(req: TranscribeRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_transcription_background, req.filename, req.callback_url)
    return {"status": "submitted", "message": "Proses dimulai (Batch Mode), tunggu webhook."}

@app.post("/clip")
def clip_video(req: ClipRequest):
    # 1. CLEANUP MEMORY (PENTING untuk VRAM 6GB)
    # Memastikan sisa-sisa Whisper dibuang dulu sebelum FFmpeg jalan
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    input_path = os.path.join(DOWNLOAD_DIR, req.filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Source video not found")

    output_filename = f"clip_{int(req.start)}_{int(req.end)}_{req.filename}"
    output_path = os.path.join(PROCESSED_DIR, output_filename)
    duration = req.end - req.start
    
    if duration <= 0:
        raise HTTPException(status_code=400, detail="Invalid duration")

    try:
        # Filter Crop Portrait
        vf_filters = ["crop=w=ih*(9/16):h=ih:x=(iw-ow)/2:y=0"]
        
        srt_path = None
        if req.text:
            srt_filename = f"temp_{int(req.start)}.srt"
            srt_path = os.path.join(TEMP_DIR, srt_filename)
            create_srt_file(req.text, duration, srt_path)
            
            # Sanitasi Path untuk FFmpeg Windows
            # Mengubah D:\Folder\File.srt menjadi D\:/Folder/File.srt
            sanitized_srt_path = srt_path.replace("\\", "/").replace(":", "\\:")
            
            # Style Subtitle (Saya sederhanakan Font-nya agar tidak crash mencari Arial)
            # Outline=2 (Tebal), MarginV=70 (Posisi agak naik dikit biar aman)
            style = "FontSize=24,PrimaryColour=&H00FFFF,OutlineColour=&H000000,BorderStyle=3,Outline=2,Shadow=0,MarginV=70,Alignment=2"
            vf_filters.append(f"subtitles='{sanitized_srt_path}':force_style='{style}'")

        vf_string = ",".join(vf_filters)
        
        # Command FFmpeg
        command = [
            "ffmpeg", "-y", 
            "-ss", str(req.start), 
            "-t", str(duration),
            "-i", input_path, 
            "-vf", vf_string,
            
            # --- SETTING CPU ENCODING (Ubah Disini) ---
            "-c:v", "libx264",   # Gunakan CPU, bukan GPU
            "-preset", "fast",   # Kecepatan render (ultrafast, superfast, fast, medium)
            "-crf", "23",        # Kualitas (18=Lossless, 23=Standard, 28=Low)
            
            "-c:a", "aac", 
            "-b:a", "128k", 
            output_path
        ]
        
        print(f"[FFMPEG START] Clipping video: {req.filename}")
        
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if process.returncode != 0:
            # Ambil 10 baris terakhir dari error log (biasanya info penting ada disitu)
            error_log = process.stderr.split('\n')[-20:] 
            clean_error = "\n".join(error_log)
            print(f"[FFMPEG ERROR FULL LOG] \n{process.stderr}") # Print full log ke terminal server
            raise Exception(f"FFmpeg Error: {clean_error}")

        print(f"[FFMPEG SUCCESS] Saved to: {output_filename}")

        return {
            "status": "success",
            "output_file": output_filename,
            "full_path": output_path
        }

    except Exception as e:
        print(f"[ERROR CLIP] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        trash_list = []
        if 'srt_path' in locals() and srt_path: 
            trash_list.append(srt_path)
        cleanup_files(trash_list)