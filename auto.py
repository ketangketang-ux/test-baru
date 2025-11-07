import os
import subprocess
import modal

PORT = 8000

vol = modal.Volume.from_name("a1111-cache", create_if_missing=True)

# Pakai base image yang udah include CUDA
a1111_image = (
    modal.Image.from_registry("nvidia/cuda:11.8-devel-ubuntu20.04", add_python="3.10")
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
    )
    .run_commands(
        # Install torch di system Python dulu
        "pip3 install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118",
        "pip3 install xformers==0.0.21",
        
        # Clone Auto1111
        "git clone --depth 1 https://github.com/AUTOMATIC1111/stable-diffusion-webui /app/webui",
        
        # Buat venv dan install packages manual
        "cd /app/webui && python3 -m venv venv",
        "cd /app/webui && . venv/bin/activate && pip install --upgrade pip",
        
        # Install torch di venv juga
        "cd /app/webui && . venv/bin/activate && pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118",
        "cd /app/webui && . venv/bin/activate && pip install xformers==0.0.21",
        
        # Install requirements lainnya
        "cd /app/webui && . venv/bin/activate && pip install -r requirements.txt",
        
        # Install packages penting manual
        "cd /app/webui && . venv/bin/activate && pip install pytorch_lightning==1.9.4 transformers accelerate safetensors opencv-python",
        
        # Extensions
        "mkdir -p /app/webui/extensions",
        "cd /app/webui/extensions && git clone --depth 1 https://github.com/kohya-ss/sd-webui-additional-networks",
        
        # Models
        "mkdir -p /app/webui/models/Stable-diffusion /app/webui/models/Lora",
        "cd /app/webui/models/Stable-diffusion && wget -q https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors -O v1-5-pruned-emaonly.safetensors",
    )
)

app = modal.App("a1111-final", image=a1111_image)

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
    
    # Test torch dulu
    test_cmd = "cd /webui && . venv/bin/activate && python -c 'import torch; print(f\"Torch version: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")'"
    subprocess.run(test_cmd, shell=True)
    
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

# Function untuk test environment
@app.function(volumes={"/webui": vol})
def test_environment():
    """Test jika environment work"""
    test_script = """
cd /webui && . venv/bin/activate && python -c "
import torch
import torchvision
import xformers
print('✅ Torch version:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
print('✅ Xformers version:', xformers.__version__)
print('✅ Environment ready!')
"
"""
    result = subprocess.run(test_script, shell=True, capture_output=True, text=True)
    return {"output": result.stdout, "error": result.stderr}

if __name__ == "__main__":
    print("🚀 Auto1111 WebUI - Final Fixed Version")
    print("🔧 Using CUDA 11.8 base image")
    print("🌐 Access at: https://your-username--a1111-final.modal.run")
