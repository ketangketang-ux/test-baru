import os
import subprocess
import modal

PORT = 8000

vol = modal.Volume.from_name("a1111-cache", create_if_missing=True)

# Pakai image yang lebih compatible
a1111_image = (
    modal.Image.debian_slim(python_version="3.10")  # Python 3.10 lebih stable
    .apt_install(
        "wget",
        "git", 
        "aria2",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
    )
    .run_commands(
        # Clone Auto1111 dengan branch yang stable
        "git clone --depth 1 --branch v1.6.0 https://github.com/AUTOMATIC1111/stable-diffusion-webui /app/webui",
        
        # Install torch yang compatible dulu
        "pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118",
        
        # Setup WebUI
        "cd /app/webui && python -m venv venv",
        "cd /app/webui && . venv/bin/activate && pip install --upgrade pip",
        
        # Install requirements dengan version fixing
        "cd /app/webui && . venv/bin/activate && pip install -r requirements.txt",
        
        # Install pytorch-lightning version yang compatible
        "cd /app/webui && . venv/bin/activate && pip install pytorch_lightning==1.9.4",
        
        # Install xformers untuk performance
        "cd /app/webui && . venv/bin/activate && pip install xformers==0.0.20",
        
        # Extensions penting
        "cd /app/webui/extensions && git clone --depth 1 https://github.com/kohya-ss/sd-webui-additional-networks",
        
        # Buat folder models
        "mkdir -p /app/webui/models/Stable-diffusion /app/webui/models/Lora",
        
        # Download model kecil dulu untuk testing
        "cd /app/webui/models/Stable-diffusion && wget -q https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors -O v1-5-pruned-emaonly.safetensors",
    )
)

app = modal.App("a1111-fixed", image=a1111_image)

@app.function(
    gpu="a100",
    timeout=3600,
    keep_warm=1,
    volumes={"/webui": vol}
)
@modal.web_server(port=PORT, startup_timeout=300)
def run():
    if not os.path.exists("/webui/launch.py"):
        print("📦 Copying WebUI to persistent volume...")
        subprocess.run("cp -r /app/webui/* /webui/", shell=True, check=True)
    
    os.makedirs("/webui/models/Lora", exist_ok=True)
    
    print("🚀 Starting WebUI...")
    
    START_COMMAND = f"""
cd /webui && \
. venv/bin/activate && \
python launch.py \
    --listen \
    --port {PORT} \
    --skip-torch-cuda-test \
    --no-download-sd-model \
    --xformers \
    --disable-safe-unpickle
"""
    
    process = subprocess.Popen(START_COMMAND, shell=True)
    process.wait()

@app.function(volumes={"/webui": vol})
def upload_lora(lora_file_path: str):
    import shutil
    lora_filename = os.path.basename(lora_file_path)
    dest_path = f"/webui/models/Lora/{lora_filename}"
    shutil.copy2(lora_file_path, dest_path)
    return {"status": "success", "path": dest_path}

@app.function(volumes={"/webui": vol}) 
def list_loras():
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
    print("🚀 Auto1111 WebUI - Fixed Version")
    print("🔧 Fixed pytorch_lightning dependency")
    print("🌐 Access at: https://your-username--a1111-fixed.modal.run")
