import os
import subprocess
import sys
import torch
import gc
import json
import cv2
import re
import numpy as np
import time
import requests
from collections import deque
from ultralytics import YOLO
from faster_whisper import WhisperModel
from openai import OpenAI
from dotenv import load_dotenv

# Load .env
load_dotenv()

# --- CONFIG STORAGE AMAN ---
AI_DRIVE = r"D:\AI_Data"
os.environ["HF_HOME"] = os.path.join(AI_DRIVE, "huggingface")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# OPEN ROUTER CRED
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY');
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'qwen/qwen3.5-flash-02-23')  # ubah jika mau model spesifik
# Optional: allow overriding URL
OPENROUTER_API_URL = os.getenv('OPENROUTER_API_URL', 'https://openrouter.ai/api/v1/chat/completions')


def cleanup_memory():
   gc.collect()
   if torch.cuda.is_available():
      torch.cuda.empty_cache()


def download_video(url, output_dir):
   print(f"\n[DOWNLOADER] Mengambil video dari: {url}")
   command = [
      sys.executable, "-m", "yt_dlp",
      "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
      "-o", os.path.join(output_dir, "%(title)s.%(ext)s"),
      url
   ]
   try:
      subprocess.run(command, check=True)
      print("[DOWNLOADER] ✅ Selesai!")
      files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.mp4')]
      latest_file = max(files, key=os.path.getctime)
      return latest_file
   except Exception as e:
      print(f"[ERROR] Gagal download: {e}")
      return None

# --- HELPER 1: PEMBUAT ANIMASI KARAOKE (.ASS) ---
def create_karaoke_ass(words_data, clip_start, output_path):
   def format_ass_time(seconds):
      seconds = max(0, seconds)
      hours = int(seconds // 3600)
      minutes = int((seconds % 3600) // 60)
      secs = int(seconds % 60)
      cs = int(round((seconds - int(seconds)) * 100))
      return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"

   ass_content = """[Script Info]
   ScriptType: v4.00+
   PlayResX: 1080
   PlayResY: 1920

   [V4+ Styles]
   Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
   Style: EleganceStyle,Montserrat Medium,80,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,2,20,20,250,1

   [Events]
   Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
   """
   lines = []
   current_line = []
   for w in words_data:
      current_line.append(w)
      if len(current_line) >= 5 or w == words_data[-1]:
            lines.append(current_line)
            current_line = []

   for line in lines:
      line_start = line[0]['start'] - clip_start
      line_end = line[-1]['end'] - clip_start
      if line_end <= 0: continue
            
      karaoke_text = ""
      for w in line:
         duration_cs = int(round((w['end'] - w['start']) * 100))

         word_str = w['word'].strip().lower()

         clean_word = word_str.replace(".", "").replace(",", "")

         karaoke_text += f"{{\\k{duration_cs}}}{clean_word} "

      start_str = format_ass_time(line_start)
      end_str = format_ass_time(line_end)
      ass_content += f"Dialogue: 0,{start_str},{end_str},EleganceStyle,,0,0,0,,{karaoke_text.strip()}\n"

   with open(output_path, "w", encoding="utf-8") as f:
      f.write(ass_content)

# --- HELPER 2: AI CAMERA DIRECTOR (YOLO + MOTION) ---
def process_ai_director_vision(raw_clip_path, temp_vision_path):
   model = YOLO("yolov8n.pt") 
   cap = cv2.VideoCapture(raw_clip_path)
   fps = cap.get(cv2.CAP_PROP_FPS)
   orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

   crop_w = int(orig_h * (9 / 16))
   crop_h = orig_h

   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   out = cv2.VideoWriter(temp_vision_path, fourcc, fps, (crop_w, crop_h))

   total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   frame_count = 0

   current_camera_x = orig_w / 2
   frames_since_cut = 0
   MIN_SHOT_FRAMES = int(fps * 2.0)
   
   prev_gray = None
   motion_history = {}
   active_speaker_id = None

   while cap.isOpened():
      ret, frame = cap.read()
      if not ret or frame is None or frame.size == 0: break

      frame_count += 1
      frames_since_cut += 1
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      gray = cv2.GaussianBlur(gray, (15, 15), 0)

      results = model.track(frame, persist=True, classes=[0], verbose=False)

      if results[0].boxes is None or results[0].boxes.id is None:
         active_speaker_id = None

      if prev_gray is not None and results[0].boxes is not None and results[0].boxes.id is not None:
            frame_diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)
            
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            centers_x = results[0].boxes.xywh[:, 0].cpu().numpy()
            
            current_scores = {}
            for box, track_id, cx in zip(boxes, track_ids, centers_x):
               x1_b, y1_b, x2_b, y2_b = map(int, box)
               x1_b, y1_b = max(0, x1_b), max(0, y1_b)
               x2_b, y2_b = min(orig_w, x2_b), min(orig_h, y2_b)

               y_mid = y1_b + int((y2_b - y1_b) * 0.5)
               roi = thresh[y1_b:y_mid, x1_b:x2_b]
               score = np.mean(roi) / 255.0 if roi.size > 0 else 0

               if track_id not in motion_history:
                  motion_history[track_id] = deque(maxlen=15)
               motion_history[track_id].append(score)

               avg_score = sum(motion_history[track_id]) / len(motion_history[track_id])
               current_scores[track_id] = {'score': avg_score, 'cx': cx}

            if current_scores:
               best_id = max(current_scores, key=lambda k: current_scores[k]['score'])
               if active_speaker_id is None:
                  active_speaker_id = best_id
                  current_camera_x = current_scores[best_id]['cx']
               elif best_id != active_speaker_id:
                  # SABUK PENGAMAN: Cek apakah speaker lama masih terlihat di kamera
                  # Jika hilang (KeyError), anggap skor geraknya 0 agar mudah digantikan
                  active_score = current_scores[active_speaker_id]['score'] if active_speaker_id in current_scores else 0
                  
                  if current_scores[best_id]['score'] > active_score + 0.01:
                     if frames_since_cut > MIN_SHOT_FRAMES:
                           active_speaker_id = best_id
                           current_camera_x = current_scores[best_id]['cx']
                           frames_since_cut = 0

      prev_gray = gray

      if active_speaker_id is None:
         current_camera_x = orig_w / 2

      x1 = int(current_camera_x - (crop_w / 2))
      
      if x1 < 0:
         x1 = 0
      elif x1 + crop_w > orig_w:
         x1 = orig_w - crop_w
      
      x2 = x1 + crop_w

      try:
         cropped_frame = frame[0:orig_h, x1:x2]

         if cropped_frame is None or cropped_frame.size == 0:
            raise Exception("Empty frame")

      except:
         # 🔥 fallback ke tengah
         center_x = orig_w // 2
         x1 = center_x - crop_w // 2
         x2 = center_x + crop_w // 2
         cropped_frame = frame[0:orig_h, x1:x2]


      if cropped_frame is None or cropped_frame.size == 0:
         continueactive_speaker_id is None

      cropped_frame = np.ascontiguousarray(cropped_frame)
      if cropped_frame.shape[0] != crop_h or cropped_frame.shape[1] != crop_w:
            cropped_frame = cv2.resize(cropped_frame, (crop_w, crop_h))

      try: out.write(cropped_frame)
      except Exception: break

      if frame_count % 15 == 0 and total_frames > 0:
         percent = (frame_count / total_frames) * 100
         print(f"\r      -> 🎬 AI Director Merender: {percent:.1f}% berjalan...", end="", flush=True)

   cap.release()
   out.release()


def split_transcript_into_chunks(transcript_text, max_words=2500):
   words = transcript_text.split()
   chunks = []

   for i in range(0, len(words), max_words):
      chunk_words = words[i:i + max_words]
      chunks.append(" ".join(chunk_words))

   print(f"[CHUNKING] Total {len(chunks)} bagian dibuat.")
   return chunks


def clean_json_response(text):
   """Cleans model response by removing markdown/code fences and stray text so json.loads can parse it."""
   if not isinstance(text, str):
      text = str(text)
   t = text.strip()

   # remove common code fences
   if t.startswith("```json"):
      t = t[len("```json"):]
   if t.startswith("```"):
      t = t[3:]
   if t.endswith("```"):
      t = t[:-3]

   # sometimes the model wraps the array in a sentence — try to extract the first [...] block
   if '[' in t and ']' in t:
      first = t.find('[')
      last = t.rfind(']')
      candidate = t[first:last+1]
      # quick sanity check: starts with [ and ends with ]
      if candidate.strip().startswith('[') and candidate.strip().endswith(']'):
         return candidate.strip()

   return t.strip()


def get_best_clips_from_chunks(transcript_text, num_clips):
   chunks = split_transcript_into_chunks(transcript_text)

   all_candidates = []

   for i, chunk in enumerate(chunks, 1):
      print(f"\n[AI] Memproses chunk {i}/{len(chunks)}")

      clips = ask_openrouter_for_clips(
         chunk,
         num_clips=3,  # tiap chunk ambil 3 kandidat
         chunk_index=i
      )

      if clips:
         all_candidates.extend(clips)

   if not all_candidates:
      return []

   print(f"\n[AI] Total kandidat: {len(all_candidates)}")

   # 🔥 Ranking global (pakai AI lagi)
   return rank_global_clips(all_candidates, num_clips)


def ask_openrouter_for_clips(transcript_text, num_clips, chunk_index=1, max_retries=3):
   """
   Mengirim prompt ke OpenRouter (chat completions) dan mengembalikan array JSON
   sesuai format yang diharapkan:
   [
   {"start": 10.5, "end": 45.2, "title": "Judul", "reason": "Kenya viral"},
   ...
   ]
   """
   print(f"\n[AI] Membaca dan menganalisis transkrip untuk mencari {num_clips} momen viral...")

   if not OPENROUTER_API_KEY:
      print("[ERROR] OPENROUTER_API_KEY tidak ditemukan di environment.")
      return []

   prompt = f"""
   Anda adalah editor video viral kelas dunia yang bekerja untuk TikTok, Reels, dan YouTube Shorts.

   Anda tidak hanya memilih bagian menarik — Anda memahami STRUKTUR CERITA dan MOMEN VIRAL seperti editor profesional.

   🎬 TUGAS:
   Dari transkrip berikut, pilih {num_clips} klip terbaik yang memiliki alur cerita lengkap dan berpotensi viral.

   🧠 GUNAKAN STRUKTUR INI:
   Setiap klip HARUS memiliki:
   1. HOOK → kalimat pembuka yang menarik perhatian
   2. SETUP → konteks / penjelasan awal
   3. BUILD-UP → peningkatan emosi atau ketegangan
   4. PAYOFF → punchline / insight / momen klimaks

   ⛔ LARANGAN:
   - JANGAN potong di tengah kalimat
   - JANGAN potong di tengah ide
   - JANGAN ambil bagian yang belum selesai atau menggantung

   ✅ ATURAN:
   - Durasi klip: 60 hingga 120 detik
   - Boleh geser start/end sedikit agar kalimat utuh
   - Ambil hanya bagian yang terasa “complete scene”

   🔥 PRIORITAS:
   - Emosi tinggi (marah, lucu, shock, inspiratif)
   - Pernyataan kontroversial atau kuat
   - Perubahan tone suara
   - Cerita personal atau pengalaman nyata

   Ini adalah bagian ke-{chunk_index} dari video panjang.

   Berikut transkrip:
   {transcript_text}

   🎯 OUTPUT:
   Balas HANYA JSON array valid tanpa markdown:

   [
   {{
      "start": 10.5,
      "end": 95.0,
      "title": "Judul viral yang bikin penasaran",
      "reason": "Hook + payoff kuat dan emosi tinggi"
   }}
   ]
   """

   headers = {
      "Authorization": f"Bearer {OPENROUTER_API_KEY}",
      "Content-Type": "application/json",
      "HTTP-Referer": "http://localhost",
      "X-Title": "AI Video Cutter"
   }

   body = {
      "model": OPENROUTER_MODEL,
      "messages": [
         {"role": "system", "content": "Return ONLY raw JSON array. Do NOT use markdown, code fences, or any extra commentary."},
         {"role": "user", "content": prompt}
      ],
      "temperature": 0.0,
      "max_tokens": 1200
   }

   retry_delay = 10
   for attempt in range(max_retries):
      try:
         resp = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=120)
         if resp.status_code != 200:
            raise Exception(f"Status {resp.status_code}: {resp.text[:500]}")

         if not resp.text.strip():
            raise Exception("Response kosong dari OpenRouter")

         # Try parsing JSON structure first (preferred)
         text = None
         try:
            data = resp.json()
            # Ambil teks dari struktur chat completions (beragam kemungkinan)
            if isinstance(data, dict):
               choices = data.get("choices") or []
               if len(choices) > 0:
                  choice0 = choices[0]
                  msg = choice0.get("message")
                  if isinstance(msg, dict):
                     content = msg.get("content")
                     if isinstance(content, str):
                        text = content
                     else:
                        text = json.dumps(content, ensure_ascii=False)
                  else:
                     if "text" in choice0 and isinstance(choice0["text"], str):
                        text = choice0["text"]
         except Exception:
            # If JSON parsing fails, fall back to raw text
            text = resp.text

         if not text:
            text = resp.text

         # Clean response from markdown/code fences and stray text
         cleaned = clean_json_response(text)

         parsed = json.loads(cleaned)
         if not isinstance(parsed, list):
            raise ValueError("Respons dari model bukan array JSON seperti yang diharapkan.")
         return parsed

      except Exception as e:
         print(f"[ERROR OPENROUTER] Percobaan {attempt+1}/{max_retries} gagal: {e}")
         if attempt < max_retries - 1:
            print(f"   -> Menunggu {retry_delay}s sebelum retry...")
            time.sleep(retry_delay)
            retry_delay *= 2
         else:
            print("[FATAL] Gagal mendapatkan analisis setelah maksimal percobaan.")
            return []


def rank_global_clips(clips, num_clips):
   print("[AI] Melakukan ranking global...")

   prompt = f"""
   Anda adalah editor viral TikTok.

   Dari daftar klip berikut, pilih {num_clips} terbaik secara global.

   Kriteria:
   - paling menarik
   - paling viral
   - paling engaging

   Data:
   {json.dumps(clips, ensure_ascii=False)}

   Balas HANYA JSON array:
   """

   headers = {
      "Authorization": f"Bearer {OPENROUTER_API_KEY}",
      "Content-Type": "application/json"
   }

   body = {
      "model": OPENROUTER_MODEL,
      "messages": [
         {"role": "system", "content": "Return ONLY raw JSON array. Do NOT use markdown, code fences, or any extra commentary."},
         {"role": "user", "content": prompt}
      ],
      "temperature": 0.2
   }

   resp = requests.post(OPENROUTER_API_URL, headers=headers, json=body)

   try:
      data = resp.json()
      text = data["choices"][0]["message"]["content"]
      cleaned = clean_json_response(text)
      return json.loads(cleaned)

   except Exception as e:
      print("[ERROR RANKING]", e)
      return clips[:num_clips]  # fallback


def analyze_and_clip(video_path, output_dir, num_clips=3):
   print("\n[AI BRAIN] Membaca video (Transkripsi via Whisper)...")
   cleanup_memory()

   model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16", download_root=r"D:\AI_Data\huggingface")
   segments, info = model.transcribe(video_path, language="id", word_timestamps=True)
   total_duration = info.duration

   print(f"   🎥 Terdeteksi audio berdurasi: {total_duration / 60:.2f} menit. Memulai transkripsi...")

   transcript_text = ""
   all_words = []

   for segment in segments:
      transcript_text += f"[{segment.start:.1f} - {segment.end:.1f}] {segment.text.strip()}\n"
      for word in segment.words:
         all_words.append({'start': word.start, 'end': word.end, 'word': word.word})

      percent = (segment.end / total_duration) * 100
      print(f"\r   ⏳ Menerjemahkan: {segment.end:.1f}s / {total_duration:.1f}s ({percent:.1f}%) berjalan...", end="", flush=True)

   print("\n   ✅ Transkripsi Selesai!")
   cleanup_memory()

   if not transcript_text.strip():
      print("[ERROR] Tidak ada suara yang terdeteksi.")
      return

   print(f"   📄 Mengirim {len(transcript_text.split())} kata ke OpenRouter...")
   top_clips = get_best_clips_from_chunks(transcript_text, num_clips)

   if not top_clips:
      print("[ERROR] Gagal mendapatkan rekomendasi klip dari OpenRouter.")
      return

   print(f"\n[EDITING] Mulai memotong & merender Auto-Cut Director + efek teks Karaoke...")
   base_name = os.path.splitext(os.path.basename(video_path))[0]
   video_specific_dir = os.path.join(output_dir, base_name)
   os.makedirs(video_specific_dir, exist_ok=True)


   for i, clip in enumerate(top_clips, 1):
      start_t = float(clip["start"])
      end_t = float(clip["end"])
      duration = end_t - start_t
      title = clip.get("title", f"clip_{i}")

      # 🔥 HAPUS semua karakter ilegal
      title = re.sub(r'[\\/*?:"<>|]', '', title)

      # optional: rapihin
      title = title.replace(" ", "_")

      # Penamaan File Temporer dan Final
      raw_clip_path = os.path.join(video_specific_dir, f"temp_raw_{i}.mp4")
      vision_clip_path = os.path.join(video_specific_dir, f"temp_vision_{i}.mp4")
      ass_path = os.path.join(video_specific_dir, f"karaoke_{i}.ass")
      out_file = os.path.join(video_specific_dir, f"{i}_{title}.mp4")

      print(f"\n   -> Memproses Part {i}: {clip.get('title','(no title)')} (Durasi: {duration:.1f}s)")

      # TAHAP 1: Ekstrak Klip Mentah (Super Cepat)
      print("      1. Mengambil potongan video asli...")
      cmd_extract = [
         "ffmpeg", "-y", "-ss", str(start_t), "-t", str(duration),
         "-i", video_path, "-c", "copy", raw_clip_path
      ]
      subprocess.run(cmd_extract, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

      # TAHAP 2: AI Director (YOLO Motion Tracking)
      print("      2. AI Director memotong adegan (Cut-to-Cut)...")
      process_ai_director_vision(raw_clip_path, vision_clip_path)

      # TAHAP 3: Generate Subtitle
      print("      3. Meracik animasi teks Karaoke...")
      clip_words = [w for w in all_words if w['start'] >= start_t - 0.5 and w['end'] <= end_t + 0.5]
      create_karaoke_ass(clip_words, start_t, ass_path)

      # TAHAP 4: Final Muxing (Gabung Visual, Audio, & Subtitle)
      print("      4. Final Rendering (Menyatukan Visual & High-Res Audio)...")
      sanitized_ass_path = ass_path.replace("\\", "/").replace(":", "\\:")

      cmd_final = [
         "ffmpeg", "-y",
         "-i", vision_clip_path,  # Input Video hasil Director
         "-i", raw_clip_path,     # Input Audio dari video mentah
         "-vf", f"ass='{sanitized_ass_path}'",
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
         "-map", "0:v:0", "-map", "1:a:0",
         out_file
      ]
      subprocess.run(cmd_final)

      # TAHAP 5: Bersihkan File Sampah
      for temp_file in [raw_clip_path, vision_clip_path, ass_path]:
         if os.path.exists(temp_file):
               os.remove(temp_file)

   print(f"\n[SELESAI] Semua video viral berhasil dirender di folder: {base_name}/")

if __name__ == "__main__":
   print("=== THE OPUSCLIP KILLER (Ultimate AI Director Edition) ===")
   
   BASE_DIR = r"D:\applications\python-tools\media"
   WORK_DIR = os.path.join(BASE_DIR, "viral_clips")
   os.makedirs(WORK_DIR, exist_ok=True)
   
   mode = input("Pilih Mode:\n1. Download URL lalu Potong\n2. Potong Video Lokal yang sudah ada\n3. Download Video Saja\nPilihan (1/2/3): ")
   
   if mode == "1":
      url = input("Masukkan URL YouTube/Instagram: ")
      num_clips = int(input("Mau dipotong jadi berapa video viral? (misal: 3): "))
      video_path = download_video(url, WORK_DIR)
      if video_path: analyze_and_clip(video_path, WORK_DIR, num_clips=num_clips)
            
   elif mode == "2":
      video_path = input("Masukkan FULL PATH video Anda: ").strip('"\'') 
      if os.path.isfile(video_path):
            num_clips = int(input("Mau dipotong jadi berapa video viral? (misal: 3): "))
            analyze_and_clip(video_path, WORK_DIR, num_clips=num_clips)
      else: print("[ERROR] File tidak ditemukan!")
            
   elif mode == "3":
      url = input("Masukkan URL YouTube/Instagram: ")
      video_path = download_video(url, WORK_DIR)
      if video_path: print(f"\n[SELESAI] File: {video_path}")
