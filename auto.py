import os
import subprocess
import modal

PORT = 8000

# Buat volume persisten
vol = modal.Volume.from_name("a1111-cache", create_if_missing=True)

# Image yang lebih simple dan reliable
a1111_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "wget",
        "git", 
        "aria2",
        "libgl1",
        "libglib2.0-0",
    )
    .run_commands(
        # Clone Auto1111
        "git clone --depth 1 https://github.com/AUTOMATIC1111/stable-diffusion-webui /app/webui",
        
        # Extensions penting saja
        "cd /app/webui/extensions && git clone --depth 1 https://github.com/kohya-ss/sd-webui-additional-networks",
        "cd /app/webui/extensions && git clone --depth 1 https://github.com/derrian-distro/LoRA_Easy_Training_Scripts",
        
        # Buat folder structure
        "mkdir -p /app/webui/models/Stable-diffusion /app/webui/models/Lora",
        
        # Download model utama saja (SD 1.5)
        "cd /app/webui/models/Stable-diffusion && wget -q https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors",
        
        # Setup environment yang lebih simple
        "cd /app/webui && python -m venv venv",
        "cd /app/webui && . venv/bin/activate && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118",
        "cd /app/webui && . venv/bin/activate && pip install -r requirements_versions.txt",
    )
)

app = modal.App("a1111-simple", image=a1111_image)

@app.function(
    gpu="a100",
    timeout=3600,
    keep_warm=1,
    volumes={"/webui": vol}
)
@modal.web_server(port=PORT, startup_timeout=300)
def run():
    # Copy ke volume jika pertama kali
    if not os.path.exists("/webui/launch.py"):
        print("📦 Copying WebUI to persistent volume...")
        subprocess.run("cp -r /app/webui/* /webui/", shell=True, check=True)
    
    # Pastikan folder ada
    os.makedirs("/webui/models/Lora", exist_ok=True)
    
    print("🚀 Starting WebUI...")
    
    # Command yang lebih simple
    START_COMMAND = f"""
cd /webui && \
. venv/bin/activate && \
python launch.py \
    --listen \
    --port {PORT} \
    --skip-torch-cuda-test \
    --no-download-sd-model
"""
    
    process = subprocess.Popen(START_COMMAND, shell=True)
    process.wait()

# Function untuk upload LoRA
@app.function(volumes={"/webui": vol})
def upload_lora(lora_file_path: str):
    """Upload LoRA file"""
    import shutil
    lora_filename = os.path.basename(lora_file_path)
    dest_path = f"/webui/models/Lora/{lora_filename}"
    shutil.copy2(lora_file_path, dest_path)
    return {"status": "success", "path": dest_path}

@app.function(volumes={"/webui": vol}) 
def list_loras():
    """List LoRA files"""
    lora_dir = "/webui/models/Lora"
    lora_files = []
    if os.path.exists(lora_dir):
        for file in os.listdir(lora_dir):
            if file.endswith('.safetensors'):
                file_path = os.path.join(lora_dir, file)
                file_size = os.path.getsize(file_path)
                lora_files.append({
                    "name": file,
                    "size": file_size,
                })
    return {"loras": lora_files}

if __name__ == "__main__":
    print("🚀 Auto1111 WebUI - Simple Version")
    print("📁 Includes: SD1.5 + LoRA extensions")
    print("🌐 Access at: https://your-username--a1111-simple.modal.run")
