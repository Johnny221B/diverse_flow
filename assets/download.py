#Model Download
from modelscope import snapshot_download
model_dir = snapshot_download(
    'stabilityai/stable-diffusion-3.5-medium',
    cache_dir='./models/stable-diffusion-3.5-medium'
                              )