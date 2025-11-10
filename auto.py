import os
import subprocess
import modal

PORT = 8000

vol = modal.Volume.from_name("a1111-cache", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("huggingface-secret")
CIVITAI_SECRET = modal.Secret.from_name("civitai-token")

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
        "git clone --depth 1 --branch v1.10.1 https://github.com/AUTOMATIC1111/stable-diffusion-webui /app/webui",
        "git clone --depth 1 https://github.com/BlafKing/sd-civitai-browser-plus /app/webui/extensions/sd-civitai-browser-plus",
        "git clone --depth 1 https://huggingface.co/embed/negative /app/webui/embeddings/negative",
        "git clone --depth 1 https://huggingface.co/embed/lora /app/webui/models/Lora/positive",
        "git clone --depth 1 https://github.com/camenduru/stable-diffusion-webui-images-browser /app/webui/extensions/stable-diffusion-webui-images-browser",
        "aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth -d /app/webui/models/ESRGAN -o 4x-UltraSharp.pth",
        "aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/artmozai/duchaiten-aiart-xl/resolve/main/duchaitenAiartSDXL_v33515.safetensors -d /app/webui/models/Stable-diffusion -o duchaitenAiartSDXL_v33515.safetensors",
        "aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/sdxl_vae/resolve/main/sdxl_vae.safetensors -d /app/webui/models/VAE -o sdxl_vae.safetensors",
        "python -m venv /app/webui/venv",
        "cd /app/webui && . venv/bin/activate && " +
        "python -c 'from modules import launch_utils; launch_utils.prepare_environment()' --xformers",
        gpu="a100",
    )
)

app = modal.App("a1111-webui", image=a1111_image)

@app.function(
    gpu="a100",
    cpu=2,
    memory=1024,
    timeout=3600,
    allow_concurrent_inputs=100,
    keep_warm=1,
    volumes={"/webui": vol},
    secrets=[HF_SECRET, CIVITAI_SECRET]
)
@modal.web_server(port=PORT, startup_timeout=180)
def run():
    # Copy image ke volume persisten
    if not os.path.exists("/webui/launch.py"):
        subprocess.run("cp -r /app/webui/* /webui/", shell=True, check=True)

    # 🔧 Install extensions at runtime (since GitHub is blocked during build)
    EXT_DIR = "/webui/extensions"
    os.makedirs(EXT_DIR, exist_ok=True)
    if not os.path.exists(f"{EXT_DIR}/sd-webui-reactor"):
        subprocess.run(
            "git clone --depth 1 https://github.com/Gourieff/sd-webui-reactor /webui/extensions/sd-webui-reactor",
            shell=True,
            check=False,
        )
    if not os.path.exists(f"{EXT_DIR}/sd-webui-controlnet"):
        subprocess.run(
            "git clone --depth 1 https://github.com/Mikubill/sd-webui-controlnet /webui/extensions/sd-webui-controlnet",
            shell=True,
            check=False,
        )

    # setup CivitAI config
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        print("🔑 Setting up HuggingFace token...")
        civitai_dir = "/webui/extensions/sd-civitai-browser-plus"
        os.makedirs(civitai_dir, exist_ok=True)
        with open(f"{civitai_dir}/config.json", "w") as f:
            f.write(f"""
{{
    "huggingface_token": "{hf_token}",
    "civitaiAccessToken": "{os.environ.get('CIVITAI_TOKEN', '')}",
    "settings": {{
        "useCivitaiLink": true,
        "useMultiThreadedDownload": true,
        "useModelPreview": true
    }}
}}
""")

    # patch shared_options
    os.system(
        r"sed -i -e 's/\[\"sd_model_checkpoint\"\]/\[\"sd_model_checkpoint\",\"sd_vae\",\"CLIP_stop_at_last_layers\"\]/g' /webui/modules/shared_options.py"
    )

    # ✅ enable extension access
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

# (fungsi download_qwen & download_from_civitai tetap sama kayak sebelumnya)
