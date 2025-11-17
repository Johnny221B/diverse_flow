from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch

model_name = "facebook/dinov2-base"

# 这两行会自动从 Hugging Face 下载模型和预处理器到本地缓存 (~/.cache/huggingface)
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.eval()