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
        
        # Install extensions penting untuk LoRA
        "git clone --depth 1 https://github.com/kohya-ss/sd-webui-additional-networks /app/webui/extensions/sd-webui-additional-networks",
        "git clone --depth 1 https://github.com/derrian-distro/LoRA_Easy_Training_Scripts /app/webui/extensions/LoRA_Easy_Training_Scripts",
        "git clone --depth 1 https://github.com/civitai/civitai-helper /app/webui/extensions/civitai-helper",
        
        # Extensions tambahan
        "git clone --depth 1 https://github.com/BlafKing/sd-civitai-browser-plus /app/webui/extensions/sd-civitai-browser-plus",
        "git clone --depth 1 https://github.com/camenduru/stable-diffusion-webui-images-browser /app/webui/extensions/stable-diffusion-webui-images-browser",
        "git clone --depth 1 https://github.com/Mikubill/sd-webui-controlnet /app/webui/extensions/sd-webui-controlnet",
        
        # Download embeddings & LoRA contoh
        "git clone --depth 1 https://huggingface.co/embed/negative /app/webui/embeddings/negative",
        "mkdir -p /app/webui/models/Lora",
        
        # Download model ke folder yang sudah ditentukan dalam image
        "aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth -d /app/webui/models/ESRGAN -o 4x-UltraSharp.pth",
        "aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors -d /app/webui/models/Stable-diffusion -o v1-5-pruned.safetensors",
        "aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/SG161222/Realistic_Vision_V5.1/resolve/main/Realistic_Vision_V5.1.safetensors -d /app/webui/models/Stable-diffusion -o Realistic_Vision_V5.1.safetensorssafetensors",
        "aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors -d /app/webui/models/VAE -o vae-ft-mse-840000-ema-pruned.safetensors",
        
        # Download contoh LoRA untuk testing
        "aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Linaqruf/character-illustrator/resolve/main/character_illustrator_v10.safetensors -d /app/webui/models/Lora -o character_illustrator_v10.safetensors",
        
        "python -m venv /app/webui/venv",
        "cd /app/webui && . venv/bin/activate && " +
        "python -c 'from modules import launch_utils; launch_utils.prepare_environment()' --xformers",
        gpu="a100",
    )
    .run_commands(
        "cd /app/webui && . venv/bin/activate && " +
        "python -c 'from modules import shared_init, initialize; shared_init.initialize(); initialize.initialize()'",
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
        subprocess.run("cp -r /app/webui/* /webui/", shell=True, check=True)
    
    # Pastikan folder LoRA ada
    os.makedirs("/webui/models/Lora", exist_ok=True)
    
    # Ubah file shared_options.py dengan menambahkan opsi "sd_vae" dan "CLIP_stop_at_last_layers"
    os.system(
        r"sed -i -e 's/\[\"sd_model_checkpoint\"\]/\[\"sd_model_checkpoint\",\"sd_vae\",\"CLIP_stop_at_last_layers\"\]/g' /webui/modules/shared_options.py"
    )
    
    START_COMMAND = f"""
cd /webui && \
. venv/bin/activate && \
accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=fp16 \
    --dynamo_backend=inductor \
    --num_cpu_threads_per_process=6 \
    launch.py \
        --skip-prepare-environment \
        --no-gradio-queue \
        --listen \
        --port {PORT}
"""
    subprocess.Popen(START_COMMAND, shell=True)

# 🔥 FUNCTION BARU UNTAK UPLOAD LoRA
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
