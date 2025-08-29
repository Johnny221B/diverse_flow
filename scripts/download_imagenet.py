from datasets import load_dataset
import os
out_dir = "flow_grpo/flow_base/datasets/ImageNet"
ds = load_dataset("benjamin-paine/imagenet-1k-256x256", split="validation")
for i, ex in enumerate(ds):
    label = ds.features["label"].int2str(ex["label"])
    out = os.path.join(out_dir, label)
    os.makedirs(out, exist_ok=True)
    ex["image"].save(os.path.join(out, f"{i:06d}.jpg"))