import os
import shutil

# --- [BAGIAN PENTING: PENGAMAN DRIVE C] ---
# Kita paksa semua cache aplikasi lari ke D:\AI_Data
# Ini harus ditaruh paling atas sebelum import library lain
AI_DRIVE = r"D:\AI_Data"

# 1. Paksa DeepFilterNet & Tools Rust ke D:
os.environ["XDG_CACHE_HOME"] = os.path.join(AI_DRIVE, "xdg_cache")
os.environ["XDG_DATA_HOME"] = os.path.join(AI_DRIVE, "xdg_data")
os.environ["DF_CACHE_DIR"] = os.path.join(AI_DRIVE, "DeepFilterNet")

# 2. Paksa PyTorch & HuggingFace ke D:
os.environ["HF_HOME"] = os.path.join(AI_DRIVE, "huggingface")
os.environ["TORCH_HOME"] = os.path.join(AI_DRIVE, "torch")

# 3. Optimasi Memory GPU (Anti-Crash)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import subprocess
import torch
import gc
from ffmpeg_normalize import FFmpegNormalize

# --- KONFIGURASI FOLDER ---
BASE_DIR = r"D:\applications\python-tools\media"
INPUT_FOLDER = os.path.join(BASE_DIR, "input_voice")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "cleaned_voice")
FINAL_FOLDER = os.path.join(BASE_DIR, "final_mastered")

# Buat folder kerja
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FINAL_FOLDER, exist_ok=True)
# Buat folder cache di D
os.makedirs(os.path.join(AI_DRIVE, "xdg_cache"), exist_ok=True)

def cleanup_memory():
    """Fungsi bersih-bersih RAM & VRAM"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def normalize_audio(input_wav, filename):
    """Menyamakan volume suara ke standar -14 LUFS"""
    print(f"   ↳ [MASTERING] Normalisasi volume: {filename} ...")
    output_path = os.path.join(FINAL_FOLDER, filename)
    
    normalizer = FFmpegNormalize(
        target_level=-14,
        print_stats=False,
        sample_rate=48000
    )
    
    try:
        # Tambahkan file ke antrian lalu jalankan
        normalizer.add_media_file(input_wav, output_path)
        normalizer.run_normalization()
        print(f"   ✅ Selesai! File Final: 'final_mastered/{filename}'")
    except Exception as e:
        print(f"   ❌ Gagal Normalisasi: {e}")

def process_audio():
    supported_ext = ('.mp3', '.wav', '.m4a', '.flac', '.ogg')
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(supported_ext)]

    if not files:
        print(f"\n[INFO] Folder '{INPUT_FOLDER}' kosong.")
        print("Silakan taruh file audio rekaman di sana.")
        return

    print(f"\n[INFO] Ditemukan {len(files)} file audio.")

    for i, filename in enumerate(files):
        cleanup_memory() # Bersihkan memori sebelum mulai
        
        input_path = os.path.join(INPUT_FOLDER, filename)
        print(f"\n[{i+1}/{len(files)}] Memproses: {filename} ...")

        # 1. AI CLEANING
        # Output cleaning disimpan sementara di cleaned_voice
        command = [
            "deepFilter",
            input_path,
            "-o", OUTPUT_FOLDER,
            "-m", "DeepFilterNet3" 
        ]

        try:
            result = subprocess.run(command, text=True, capture_output=True)

            if result.returncode == 0:
                print(f"   ✅ AI Cleaning Selesai.")
                
                # Logic mencari nama file output (karena DeepFilterNet suka nambah suffix)
                name_without_ext = os.path.splitext(filename)[0]
                expected_output_name = f"{name_without_ext}_DeepFilterNet3.wav"
                cleaned_path = os.path.join(OUTPUT_FOLDER, expected_output_name)
                
                # Fallback jika inputnya sudah .wav, kadang suffixnya beda
                if not os.path.exists(cleaned_path):
                     # Cek file wav apa saja di output folder yg mengandung nama file asli
                     candidates = [f for f in os.listdir(OUTPUT_FOLDER) if name_without_ext in f and f.endswith('.wav')]
                     if candidates:
                         # Ambil yang terbaru
                         candidates.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_FOLDER, x)), reverse=True)
                         cleaned_path = os.path.join(OUTPUT_FOLDER, candidates[0])

                if os.path.exists(cleaned_path):
                    # 2. MASTERING (Normalization)
                    final_name = f"{name_without_ext}_MASTER.wav"
                    normalize_audio(cleaned_path, final_name)
                else:
                    print(f"   ⚠️ File output cleaning tidak ditemukan.")

            else:
                print(f"   ❌ Gagal Cleaning!")
                if "CUDA out of memory" in result.stderr:
                    print("   ⚠️ ERROR MEMORY: File terlalu panjang atau GPU penuh.")
                else:
                    print("   Log:", result.stderr[:200])

        except Exception as e:
            print(f"   ❌ Error Sistem: {str(e)}")

if __name__ == "__main__":
    cleanup_memory()
    print("--- MEMULAI AUDIO STUDIO (TERMINAL MODE) ---")
    print("Lokasi Cache Aman: D:\\AI_Data")
    process_audio()
    print("\n--- Semua proses selesai ---")