import os, sys, argparse, time, traceback
from typing import Optional, List

import torch
import torch.nn.functional as F

# 让脚本能 import 同项目的 diverse_flow
CUR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CUR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diverse_flow.clip_wrapper import CLIPWrapper
from diverse_flow.volume_objective import VolumeObjective
from diverse_flow.config import DiversityConfig

def gb(x): return f"{x/1024**3:.2f} GB"

def print_mem(tag: str, dev: torch.device):
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info(dev)
        used = total - free
        print(f"[{tag}] {dev}: used={gb(used)}, free={gb(free)}", flush=True)

def load_images_as_tensor(paths: Optional[List[str]], batch: int, H: int, W: int, device: torch.device):
    # 随机图或加载本地图；都转到 [0,1] float32
    if paths and len(paths) > 0:
        from PIL import Image
        import numpy as np
        imgs = []
        for p in paths[:batch]:
            img = Image.open(p).convert("RGB").resize((W, H))
            arr = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
            imgs.append(arr)
        while len(imgs) < batch:
            imgs.append(imgs[-1].clone())
        x = torch.stack(imgs, dim=0)
    else:
        torch.manual_seed(0)
        x = torch.rand(batch, 3, H, W)
    return x.to(device).clamp(0, 1)

def main():
    ap = argparse.ArgumentParser("Only-CLIP+Volume probe")
    ap.add_argument("--impl", choices=["openai_clip","open_clip"], default="openai_clip")
    ap.add_argument("--arch", type=str, default="ViT-B-32")
    ap.add_argument("--jit-path", type=str, default=os.path.expanduser("~/.cache/clip/ViT-B-32.pt"))
    ap.add_argument("--checkpoint-path", type=str, default=None)  # for open_clip
    ap.add_argument("--device-clip", type=str, default="cuda:0")
    ap.add_argument("--images", nargs="*", default=None, help="paths to images; if empty, use random")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--cudnn-off", action="store_true", help="disable cuDNN for CLIP forward (probe workspace)")
    # 配置项（与你的 DiversityConfig 对齐）
    ap.add_argument("--whiten", action="store_true", help="enable whitening in features (建议先别开)")
    ap.add_argument("--center", action="store_true", help="feature_center")
    ap.add_argument("--l2norm", action="store_true", help="feature_l2norm")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--eps-logdet", type=float, default=1e-3)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    dev_clip = torch.device(args.device_clip)
    if dev_clip.type == "cuda":
        assert torch.cuda.is_available(), "CUDA not available"
        torch.cuda.synchronize(dev_clip)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev_clip)

    print(f"[setup] impl={args.impl}, device_clip={dev_clip}, cudnn_off={args.cudnn_off}, "
          f"batch={args.batch}, res={args.res}", flush=True)

    # 1) 构建 CLIP（仅放到 device-clip 上）
    clip = CLIPWrapper(
        impl=args.impl,
        arch=args.arch,
        checkpoint_path=args.checkpoint_path,
        jit_path=args.jit_path,
        device=dev_clip if dev_clip.type=="cuda" else torch.device("cpu"),
    )
    print(f"[clip] wrapper device={clip.device}, impl={clip.impl}", flush=True)

    # 2) 构建体积目标（只用到 CLIP）
    cfg = DiversityConfig(
        tau=args.tau,
        eps_logdet=args.eps_logdet,
        feature_center=args.center,
        feature_l2norm=args.l2norm,
        whiten=args.whiten,
        clip_image_size=args.res,
    )
    vol = VolumeObjective(clip, cfg)

    # 3) 准备输入图片张量（直接在 CLIP 设备上）
    x = load_images_as_tensor(args.images, args.batch, args.res, args.res, device=dev_clip if dev_clip.type=="cuda" else torch.device("cpu"))

    # 4) 跑几次前向/求梯度，打印显存
    for i in range(args.runs):
        if dev_clip.type == "cuda":
            torch.cuda.synchronize(dev_clip)
        print_mem(f"iter{i}-before", dev_clip)

        t0 = time.time()
        try:
            # 只围绕 CLIP 的前向禁用 cuDNN（帮助定位 1GiB workspace）
            if args.cudnn_off and dev_clip.type == "cuda":
                import torch.backends.cudnn as cudnn
                with cudnn.flags(enabled=False, benchmark=False, deterministic=False):
                    # 这里调用 volume 的接口即可（内部会用 CLIP）
                    loss, grad_x, logs = vol.volume_loss_and_grad(x)
            else:
                loss, grad_x, logs = vol.volume_loss_and_grad(x)
        except Exception:
            traceback.print_exc()
            break

        if dev_clip.type == "cuda":
            torch.cuda.synchronize(dev_clip)
        dt = (time.time() - t0) * 1000
        print_mem(f"iter{i}-after ", dev_clip)
        print(f"[iter{i}] elapsed={dt:.1f} ms | loss={float(loss.detach().item()):.4f} | "
              f"logdet={logs.get('logdet',0):.4f} | min/mean-angle={logs.get('min_angle_deg',0):.2f}/{logs.get('mean_angle_deg',0):.2f}",
              flush=True)

        # 清理这一轮的中间量（模拟你回调里的清缓存）
        del loss, grad_x
        if dev_clip.type == "cuda":
            torch.cuda.synchronize(dev_clip)
            with torch.cuda.device(dev_clip):
                torch.cuda.empty_cache()
        print_mem(f"iter{i}-after-empty", dev_clip)
        print("-"*60, flush=True)

    # 打印总结
    if dev_clip.type == "cuda":
        print(torch.cuda.memory_summary(device=dev_clip), flush=True)

if __name__ == "__main__":
    main()