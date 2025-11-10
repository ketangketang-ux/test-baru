import os
import subprocess
import modal

PORT = 8000

# Buat volume persisten yang akan menyimpan file instalasi agar tidak perlu di-download ulang
vol = modal.Volume.from_name("a1111-cache", create_if_missing=True)

# Secret untuk HuggingFace dan CivitAI tokens
HF_SECRET = modal.Secret.from_name("huggingface-secret")
CIVITAI_SECRET = modal.Secret.from_name("civitai-token")

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
        # Tambahan: clone extension penting
        "git clone --depth 1 https://github.com/Gourieff/sd-webui-reactor /app/webui/extensions/sd-webui-reactor",
        "git clone --depth 1 https://github.com/Mikubill/sd-webui-controlnet /app/webui/extensions/sd-webui-controlnet",
        "git clone --depth 1 https://github.com/BlafKing/sd-civitai-browser-plus /app/webui/extensions/sd-civitai-browser-plus",
        "git clone --depth 1 https://huggingface.co/embed/negative /app/webui/embeddings/negative",
        "git clone --depth 1 https://huggingface.co/embed/lora /app/webui/models/Lora/positive",
        "git clone --depth 1 https://github.com/camenduru/stable-diffusion-webui-images-browser /app/webui/extensions/stable-diffusion-webui-images-browser",
        # Download model ke folder yang sudah ditentukan dalam image
        "aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth -d /app/webui/models/ESRGAN -o 4x-UltraSharp.pth",
        "aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/artmozai/duchaiten-aiart-xl/resolve/main/duchaitenAiartSDXL_v33515.safetensors -d /app/webui/models/Stable-diffusion -o duchaitenAiartSDXL_v33515.safetensors",
        "aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/sdxl_vae/resolve/main/sdxl_vae.safetensors -d /app/webui/models/VAE -o sdxl_vae.safetensors",
        # Setup virtual environment dan dependencies
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

app = modal.App("a1111-webui", image=a1111_image)

# Mount volume persisten ke path /webui agar file instalasi tersimpan antar eksekusi.
@app.function(
    gpu="a100",
    cpu=2,
    memory=1024,
    timeout=3600,
    allow_concurrent_inputs=100,
    keep_warm=1,
    volumes={"/webui": vol},
    secrets=[HF_SECRET, CIVITAI_SECRET]  # TAMBAHKAN SECRETS DI SINI
)
@modal.web_server(port=PORT, startup_timeout=180)
def run():
    # Jika folder /webui (yang dipersist) masih kosong, salin dari /app/webui (baked ke image)
    if not os.path.exists("/webui/launch.py"):
        subprocess.run("cp -r /app/webui/* /webui/", shell=True, check=True)
    
    # Setup HuggingFace token untuk CivitAI browser
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        print("🔑 Setting up HuggingFace token for model downloads...")
        # Simpan token di config CivitAI browser
        civitai_config_dir = "/webui/extensions/sd-civitai-browser-plus"
        os.makedirs(civitai_config_dir, exist_ok=True)
        
        civitai_config = f"""
{{
    "huggingface_token": "{hf_token}",
    "civitaiAccessToken": "{os.environ.get('CIVITAI_TOKEN', '')}",
    "settings": {{
        "useCivitaiLink": true,
        "useMultiThreadedDownload": true,
        "useModelPreview": true
    }}
}}
"""
        with open(f"{civitai_config_dir}/config.json", "w") as f:
            f.write(civitai_config)
    
    # Ubah file shared_options.py dengan menambahkan opsi "sd_vae" dan "CLIP_stop_at_last_layers"
    os.system(
        r"sed -i -e 's/\[\"sd_model_checkpoint\"\]/\[\"sd_model_checkpoint\",\"sd_vae\",\"CLIP_stop_at_last_layers\"\]/g' /webui/modules/shared_options.py"
    )
    
    # ✅ Fix: aktifkan akses extension
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
        --enable-insecure-extension-access \
        --port {PORT}
"""
    subprocess.Popen(START_COMMAND, shell=True)

# FUNCTION BARU: Download Qwen model menggunakan token HF
@app.function(
    volumes={"/webui": vol},
    secrets=[HF_SECRET]
)
def download_qwen():
    """Download Qwen model menggunakan HF token"""
    import requests
    
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        return {"status": "error", "message": "No HuggingFace token found"}
    
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Download Qwen2-VL-7B model parts
    model_parts = [
        "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/model-00001-of-00003.safetensors",
        "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/model-00002-of-00003.safetensors",
        "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/model-00003-of-00003.safetensors"
    ]
    
    results = []
    for i, url in enumerate(model_parts, 1):
        try:
            filename = f"qwen2-vl-7b-part-{i}.safetensors"
            filepath = f"/webui/models/Stable-diffusion/{filename}"
            
            print(f"📥 Downloading part {i}/3...")
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = os.path.getsize(filepath)
            results.append({"part": i, "status": "success", "size": file_size})
            print(f"✅ Part {i} downloaded: {file_size} bytes")
            
        except Exception as e:
            results.append({"part": i, "status": "error", "error": str(e)})
            print(f"❌ Part {i} failed: {e}")
    
    return {"results": results}

# FUNCTION BARU: Download model dari CivitAI
@app.function(
    volumes={"/webui": vol},
    secrets=[CIVITAI_SECRET]
)
def download_from_civitai(model_id: str):
    """Download model dari CivitAI menggunakan token"""
    import requests
    
    civitai_token = os.environ.get("CIVITAI_TOKEN")
    if not civitai_token:
        return {"status": "error", "message": "No CivitAI token found"}
    
    headers = {"Authorization": f"Bearer {civitai_token}"}
    
    try:
        # Dapatkan model info dulu
        model_info_url = f"https://civitai.com/api/v1/models/{model_id}"
        model_info = requests.get(model_info_url, headers=headers).json()
        
        # Download model file
        download_url = model_info["modelVersions"][0]["downloadUrl"]
        filename = model_info["modelVersions"][0]["files"][0]["name"]
        filepath = f"/webui/models/Stable-diffusion/{filename}"
        
        print(f"📥 Downloading {filename} from CivitAI...")
        response = requests.get(download_url, headers=headers, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size = os.path.getsize(filepath)
        return {"status": "success", "filename": filename, "size": file_size}
        
    except Exception as e:
        return {"status": "error", "error": str(e)}
