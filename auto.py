# cell 3: Create fixed version
%%writefile /content/auto_fixed.py
import os
import subprocess
import modal

PORT = 8000

vol = modal.Volume.from_name("a1111-cache", create_if_missing=True)

# FIXED: Remove CUDA manual install, use Modal's built-in CUDA
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
    )
    .run_commands(
        # FIXED: Install torch without specifying CUDA version (Modal sudah include CUDA)
        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118",
        "pip install xformers",
        
        # Clone Auto1111
        "git clone --depth 1 https://github.com/AUTOMATIC1111/stable-diffusion-webui /app/webui",
        
        # Setup venv
        "cd /app/webui && python3 -m venv venv",
        "cd /app/webui && . venv/bin/activate && pip install --upgrade pip",
        
        # FIXED: Install requirements directly
        "cd /app/webui && . venv/bin/activate && pip install -r requirements.txt",
        
        # Install additional packages
        "cd /app/webui && . venv/bin/activate && pip install accelerate safetensors",
        
        # Extensions (with error handling)
        "mkdir -p /app/webui/extensions",
        "cd /app/webui/extensions && git clone --depth 1 https://github.com/kohya-ss/sd-webui-additional-networks || echo 'Extension install failed, continuing...'",
        
        # Models
        "mkdir -p /app/webui/models/Stable-diffusion /app/webui/models/Lora",
        "cd /app/webui/models/Stable-diffusion && wget -q https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors -O v1-5-pruned-emaonly.safetensors || echo 'Model download failed, continuing...'",
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
cd /webui && . venv/bin/activate && python launch.py \
    --listen \
    --port {PORT} \
    --skip-prepare-environment \
    --skip-torch-cuda-test \
    --no-download-sd-model \
    --xformers \
    --api \
    --enable-insecure-extension-access
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
    print("🔧 Removed manual CUDA install")
    print("🌐 Access at: https://your-username--a1111-fixed.modal.run")
