import os
import subprocess
import modal

PORT = 8000

vol = modal.Volume.from_name("a1111-cache", create_if_missing=True)

# Pakai base image Modal yang reliable + install CUDA manual
a1111_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "wget",
        "git", 
        "libgl1",
        "libglib2.0-0", 
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "python3-venv",
        "python3-pip",
        "build-essential",
        "software-properties-common",
    )
    .run_commands(
        # Add NVIDIA repository dan install CUDA
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb",
        "dpkg -i cuda-keyring_1.0-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-11-8",
        
        # Install torch dengan CUDA support
        "pip3 install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118",
        "pip3 install xformers==0.0.21",
        
        # Clone Auto1111
        "git clone --depth 1 https://github.com/AUTOMATIC1111/stable-diffusion-webui /app/webui",
        
        # Setup venv
        "cd /app/webui && python3 -m venv venv",
        "cd /app/webui && . venv/bin/activate && pip install --upgrade pip",
        
        # Install torch di venv
        "cd /app/webui && . venv/bin/activate && pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118",
        "cd /app/webui && . venv/bin/activate && pip install xformers==0.0.21",
        
        # Install requirements tanpa torch (karena udah diinstall)
        "cd /app/webui && . venv/bin/activate && grep -v '^torch' requirements.txt > requirements_no_torch.txt && pip install -r requirements_no_torch.txt",
        
        # Install packages penting
        "cd /app/webui && . venv/bin/activate && pip install pytorch_lightning==1.9.4 transformers accelerate safetensors opencv-python",
        
        # Extensions
        "mkdir -p /app/webui/extensions",
        "cd /app/webui/extensions && git clone --depth 1 https://github.com/kohya-ss/sd-webui-additional-networks",
        
        # Models
        "mkdir -p /app/webui/models/Stable-diffusion /app/webui/models/Lora",
        "cd /app/webui/models/Stable-diffusion && wget -q https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors -O v1-5-pruned-emaonly.safetensors",
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
    --skip-prepare-environment \
    --skip-torch-cuda-test \
    --no-download-sd-model \
    --xformers \
    --api
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
    print("🚀 Auto1111 WebUI - Simple & Working")
    print("🌐 Access at: https://your-username--a1111-simple.modal.run")
