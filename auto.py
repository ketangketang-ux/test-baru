import os
import subprocess
import modal

PORT = 8000

# Buat volume persisten yang akan menyimpan file instalasi agar tidak perlu di-download ulang
vol = modal.Volume.from_name("a1111-cache", create_if_missing=True)

# Pada tahap build image, kita clone dan download ke folder /app/webui
a1111_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "wget",
        "git", 
        "aria2",
        "libgl1",
        "libglib2.0-0",
        "google-perftools",  # For tcmalloc
    )
    .env({"LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"})
    .run_commands(
        # Clone kode ke folder /app/webui (bukan /webui)
        "git clone --depth 1 --branch v1.10.1 https://github.com/AUTOMATIC1111/stable-diffusion-webui /app/webui",
        
        # Install extensions penting untuk LoRA (dengan error handling)
        "cd /app/webui/extensions && git clone --depth 1 https://github.com/kohya-ss/sd-webui-additional-networks || echo 'Additional Networks failed, continuing...'",
        "cd /app/webui/extensions && git clone --depth 1 https://github.com/derrian-distro/LoRA_Easy_Training_Scripts || echo 'LoRA Training failed, continuing...'",
        "cd /app/webui/extensions && git clone --depth 1 https://github.com/civitai/civitai-helper || echo 'CivitAI Helper failed, continuing...'",
        
        # Extensions tambahan
        "cd /app/webui/extensions && git clone --depth 1 https://github.com/BlafKing/sd-civitai-browser-plus || echo 'CivitAI Browser Plus failed, continuing...'",
        "cd /app/webui/extensions && git clone --depth 1 https://github.com/camenduru/stable-diffusion-webui-images-browser || echo 'Image Browser failed, continuing...'",
        "cd /app/webui/extensions && git clone --depth 1 https://github.com/Mikubill/sd-webui-controlnet || echo 'ControlNet failed, continuing...'",
        
        # Download embeddings & buat folder structure
        "mkdir -p /app/webui/embeddings /app/webui/models/Lora /app/webui/models/VAE /app/webui/models/ESRGAN",
        "cd /app/webui/embeddings && git clone --depth 1 https://huggingface.co/embed/negative || echo 'Embeddings failed, continuing...'",
        
        # Download models dengan error handling
        "cd /app/webui/models/ESRGAN && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth -o 4x-UltraSharp.pth || echo 'Upscaler download failed'",
        "cd /app/webui/models/Stable-diffusion && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors -o v1-5-pruned.safetensors || echo 'SD1.5 download failed'",
        "cd /app/webui/models/Stable-diffusion && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/SG161222/Realistic_Vision_V5.1/resolve/main/Realistic_Vision_V5.1.safetensors -o Realistic_Vision_V5.1.safetensors || echo 'Realistic Vision download failed'",
        
        # Download Qwen model (gunakan yang 7B saja, 72B terlalu besar)
        "cd /app/webui/models/Stable-diffusion && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/qwen2-vl-7b-instruct.safetensors -o qwen2-vl-7b-instruct.safetensors || echo 'Qwen download failed'",
        
        # Download VAE untuk SD models (Qwen tidak butuh VAE terpisah)
        "cd /app/webui/models/VAE && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors -o vae-ft-mse-840000-ema-pruned.safetensors || echo 'VAE download failed'",
        
        # Download contoh LoRA untuk testing
        "cd /app/webui/models/Lora && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Linaqruf/character-illustrator/resolve/main/character_illustrator_v10.safetensors -o character_illustrator_v10.safetensors || echo 'LoRA example download failed'",
        
        # Setup Python environment
        "python -m venv /app/webui/venv",
        "cd /app/webui && . venv/bin/activate && pip install --upgrade pip",
        "cd /app/webui && . venv/bin/activate && python -c 'from modules import launch_utils; launch_utils.prepare_environment()' --xformers",
        gpu="a100",
    )
    .run_commands(
        "cd /app/webui && . venv/bin/activate && python -c 'from modules import shared_init, initialize; shared_init.initialize(); initialize.initialize()'",
        gpu="a100",
    )
)

app = modal.App("a1111-webui-lora", image=a1111_image)

# Mount volume persisten ke path /webui agar file instalasi tersimpan antar eksekusi.
@app.function(
    gpu="a100",
    cpu=2,
    memory=1024,
    timeout=3600,
    allow_concurrent_inputs=100,
    keep_warm=1,
    volumes={"/webui": vol}
)
@modal.web_server(port=PORT, startup_timeout=180)
def run():
    # Jika folder /webui (yang dipersist) masih kosong, salin dari /app/webui (baked ke image)
    if not os.path.exists("/webui/launch.py"):
        print("📦 First run - copying webui files to persistent volume...")
        subprocess.run("cp -r /app/webui/* /webui/", shell=True, check=True)
        print("✅ WebUI files copied to persistent volume")
    else:
        print("✅ Using existing WebUI installation from volume")
    
    # Pastikan folder structure lengkap
    os.makedirs("/webui/models/Lora", exist_ok=True)
    os.makedirs("/webui/models/VAE", exist_ok=True)
    os.makedirs("/webui/models/ESRGAN", exist_ok=True)
    os.makedirs("/webui/embeddings", exist_ok=True)
    
    # Ubah file shared_options.py dengan menambahkan opsi "sd_vae" dan "CLIP_stop_at_last_layers"
    print("⚙️ Configuring WebUI settings...")
    os.system(
        r"sed -i -e 's/\[\"sd_model_checkpoint\"\]/\[\"sd_model_checkpoint\",\"sd_vae\",\"CLIP_stop_at_last_layers\"\]/g' /webui/modules/shared_options.py 2>/dev/null || echo 'Config update skipped'"
    )
    
    # Install requirements untuk extensions yang mungkin terlewat
    print("📦 Installing extension requirements...")
    extensions_dir = "/webui/extensions"
    if os.path.exists(extensions_dir):
        for ext in os.listdir(extensions_dir):
            req_file = os.path.join(extensions_dir, ext, "requirements.txt")
            if os.path.exists(req_file):
                print(f"Installing requirements for {ext}...")
                subprocess.run(f"cd /webui && . venv/bin/activate && pip install -r {req_file} --quiet", shell=True)
    
    print("🚀 Starting WebUI server...")
    START_COMMAND = f"""
cd /webui && \
. venv/bin/activate && \
python launch.py \
    --skip-prepare-environment \
    --no-download-sd-model \
    --listen \
    --port {PORT} \
    --api \
    --xformers \
    --enable-insecure-extension-access
"""
    process = subprocess.Popen(START_COMMAND, shell=True)
    
    # Tunggu proses berjalan
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()

# 🔥 FUNCTION BARU UNTUK UPLOAD LoRA
@app.function(volumes={"/webui": vol})
def upload_lora(lora_file_path: str):
    """Upload LoRA file ke instance Auto1111 yang sedang jalan"""
    import shutil
    
    # Extract filename
    lora_filename = os.path.basename(lora_file_path)
    
    # Destination path
    dest_path = f"/webui/models/Lora/{lora_filename}"
    
    # Copy file
    shutil.copy2(lora_file_path, dest_path)
    
    return {"status": "success", "path": dest_path, "filename": lora_filename}

# 🔥 FUNCTION UNTUK LIST LoRA YANG ADA
@app.function(volumes={"/webui": vol}) 
def list_loras():
    """List semua LoRA files yang ada di instance"""
    lora_dir = "/webui/models/Lora"
    lora_files = []
    
    if os.path.exists(lora_dir):
        for file in os.listdir(lora_dir):
            if file.endswith(('.safetensors', '.pt', '.ckpt')):
                file_path = os.path.join(lora_dir, file)
                file_size = os.path.getsize(file_path)
                lora_files.append({
                    "name": file,
                    "size": file_size,
                    "path": file_path
                })
    
    return {"loras": lora_files}

# 🔥 FUNCTION UNTUK CHECK INSTALLED MODELS
@app.function(volumes={"/webui": vol})
def list_models():
    """List semua model yang terinstall"""
    models = {
        "checkpoints": [],
        "loras": [],
        "vaes": []
    }
    
    # Check checkpoints
    checkpoint_dir = "/webui/models/Stable-diffusion"
    if os.path.exists(checkpoint_dir):
        models["checkpoints"] = [f for f in os.listdir(checkpoint_dir) if f.endswith('.safetensors')]
    
    # Check LoRAs
    lora_dir = "/webui/models/Lora"
    if os.path.exists(lora_dir):
        models["loras"] = [f for f in os.listdir(lora_dir) if f.endswith(('.safetensors', '.pt', '.ckpt'))]
    
    # Check VAEs
    vae_dir = "/webui/models/VAE"
    if os.path.exists(vae_dir):
        models["vaes"] = [f for f in os.listdir(vae_dir) if f.endswith('.safetensors')]
    
    return models

if __name__ == "__main__":
    # Print info saat deploy
    print("🎨 Auto1111 WebUI with LoRA Support")
    print("📁 Models included: SD1.5, Realistic Vision, Qwen2-VL")
    print("🔧 Extensions: Additional Networks, LoRA Training, CivitAI Helper")
    print(f"🌐 WebUI will be available at: https://your-username--a1111-webui-lora.modal.run")
