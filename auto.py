import os
import subprocess
import modal

PORT = 8000

# Buat volume persisten
vol = modal.Volume.from_name("a1111-cache", create_if_missing=True)

# Secret untuk HuggingFace dan CivitAI tokens
HF_SECRET = modal.Secret.from_name("huggingface")
CIVITAI_SECRET = modal.Secret.from_name("civitai")

# Pada tahap build image
a1111_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "wget",
        "git",
        "aria2", 
        "libgl1",
        "libglib2.0-0",
        "google-perftools",
    )
    .env({"LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"})
    .run_commands(
        # Clone Auto1111
        "git clone --depth 1 --branch v1.10.1 https://github.com/AUTOMATIC1111/stable-diffusion-webui /app/webui",
        
        # Install extensions
        "git clone --depth 1 https://github.com/BlafKing/sd-civitai-browser-plus /app/webui/extensions/sd-civitai-browser-plus",
        "git clone --depth 1 https://github.com/camenduru/stable-diffusion-webui-images-browser /app/webui/extensions/stable-diffusion-webui-images-browser",
        "git clone --depth 1 https://github.com/kohya-ss/sd-webui-additional-networks /app/webui/extensions/sd-webui-additional-networks",
        
        # Download embeddings
        "mkdir -p /app/webui/embeddings /app/webui/models/Lora",
        "cd /app/webui/embeddings && git clone --depth 1 https://huggingface.co/embed/negative || true",
        
        # Download models yang tidak butuh auth
        "mkdir -p /app/webui/models/Stable-diffusion /app/webui/models/VAE /app/webui/models/ESRGAN",
        "cd /app/webui/models/ESRGAN && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth -o 4x-UltraSharp.pth || echo 'Upscaler download failed'",
        "cd /app/webui/models/Stable-diffusion && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors -o v1-5-pruned-emaonly.safetensors || echo 'SD1.5 download failed'",
        "cd /app/webui/models/VAE && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors -o vae-ft-mse-840000-ema-pruned.safetensors || echo 'VAE download failed'",
        
        # Setup Python environment
        "python -m venv /app/webui/venv",
        "cd /app/webui && . venv/bin/activate && pip install --upgrade pip",
        "cd /app/webui && . venv/bin/activate && python -c 'from modules import launch_utils; launch_utils.prepare_environment()' --xformers",
        gpu="a100",
    )
)

app = modal.App("a1111-webui-pro", image=a1111_image)

@app.function(
    gpu="a100",
    cpu=2,
    memory=1024,
    timeout=3600,
    allow_concurrent_inputs=100,
    keep_warm=1,
    volumes={"/webui": vol},
    secrets=[HF_SECRET, CIVITAI_SECRET]  # Tambahkan secrets di sini
)
@modal.web_server(port=PORT, startup_timeout=180)
def run():
    # Copy ke volume jika pertama kali
    if not os.path.exists("/webui/launch.py"):
        subprocess.run("cp -r /app/webui/* /webui/", shell=True, check=True)
    
    # Setup HuggingFace token
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        print("🔑 Setting up HuggingFace token...")
        # Simpan token untuk huggingface-cli
        subprocess.run(f"huggingface-cli login --token {hf_token} --add-to-git-credential", 
                      shell=True, capture_output=True)
        
        # Simpan di config Auto1111
        hf_config = f"""
{{
    "huggingface_token": "{hf_token}",
    "huggingface_models": true
}}
"""
        with open("/webui/config.json", "a") as f:
            f.write(hf_config)
    
    # Setup CivitAI token
    civitai_token = os.environ.get("CIVITAI_TOKEN")
    if civitai_token:
        print("🔑 Setting up CivitAI token...")
        # Simpan config untuk CivitAI browser
        civitai_config_dir = "/webui/extensions/sd-civitai-browser-plus"
        os.makedirs(civitai_config_dir, exist_ok=True)
        
        civitai_config = f"""
{{
    "civitaiAccessToken": "{civitai_token}",
    "settings": {{
        "useCivitaiLink": true,
        "useMultiThreadedDownload": true,
        "useModelPreview": true
    }}
}}
"""
        with open(f"{civitai_config_dir}/config.json", "w") as f:
            f.write(civitai_config)
    
    # Ubah file shared_options.py
    os.system(
        r"sed -i -e 's/\[\"sd_model_checkpoint\"\]/\[\"sd_model_checkpoint\",\"sd_vae\",\"CLIP_stop_at_last_layers\"\]/g' /webui/modules/shared_options.py 2>/dev/null || true"
    )
    
    print("🚀 Starting WebUI with token authentication...")
    
    START_COMMAND = f"""
cd /webui && \
. venv/bin/activate && \
python launch.py \
    --skip-prepare-environment \
    --no-gradio-queue \
    --listen \
    --port {PORT} \
    --xformers \
    --api
"""
    subprocess.Popen(START_COMMAND, shell=True).wait()

# Function untuk download model dengan HF token
@app.function(
    volumes={"/webui": vol},
    secrets=[HF_SECRET]
)
def download_model(model_url: str, model_name: str):
    """Download model menggunakan HF token"""
    import requests
    
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    headers = {}
    
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    
    model_path = f"/webui/models/Stable-diffusion/{model_name}"
    
    print(f"📥 Downloading {model_name} with authentication...")
    
    try:
        response = requests.get(model_url, headers=headers, stream=True)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return {"status": "success", "path": model_path, "size": os.path.getsize(model_path)}
    
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Function untuk upload LoRA
@app.function(volumes={"/webui": vol})
def upload_lora(lora_file_path: str):
    import shutil
    lora_filename = os.path.basename(lora_file_path)
    dest_path = f"/webui/models/Lora/{lora_filename}"
    shutil.copy2(lora_file_path, dest_path)
    return {"status": "success", "path": dest_path}

# Function untuk list models
@app.function(volumes={"/webui": vol}) 
def list_models():
    models = {
        "checkpoints": [],
        "loras": [],
        "vaes": []
    }
    
    checkpoint_dir = "/webui/models/Stable-diffusion"
    if os.path.exists(checkpoint_dir):
        models["checkpoints"] = [f for f in os.listdir(checkpoint_dir) if f.endswith('.safetensors')]
    
    lora_dir = "/webui/models/Lora"
    if os.path.exists(lora_dir):
        models["loras"] = [f for f in os.listdir(lora_dir) if f.endswith('.safetensors')]
    
    return models

if __name__ == "__main__":
    print("🚀 Auto1111 WebUI Pro - With Token Authentication")
    print("🔑 Supports: HuggingFace Token + CivitAI Token")
    print("🌐 Access at: https://your-username--a1111-webui-pro.modal.run")
