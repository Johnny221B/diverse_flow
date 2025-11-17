from huggingface_hub import snapshot_download

model_dir = snapshot_download(
    "black-forest-labs/FLUX.1-schnell",
    local_dir="./FLUX.1-schnell",
    local_dir_use_symlinks=False,
)
print(model_dir)
