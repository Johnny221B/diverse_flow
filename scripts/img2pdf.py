#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_grid_2x2_no_crop.py

把 4 张图片按 2 列 x 2 排紧密拼接到一页 PDF，优先不裁剪、不拉伸：
- 若 4 张图尺寸相同（如 512x512），MODE="auto" 会直接按原像素拼接（无缩放/裁剪/留白）。
- 否则可以选择 "cover"（裁剪填满）、"stretch"（拉伸到单元格）、"contain-pad"（等比缩放并填充背景）。

修改 IMAGE_PATHS 列表和 MODE 后运行。
依赖: pip install pillow
"""

from pathlib import Path
from PIL import Image, ImageOps
import sys

# ----------------- 在这里直接编辑你的 4 张图片路径 -----------------
IMAGE_PATHS = [
    "/data2/toby/OSCAR/outputs/try3_bowl/imgs/a_photo_of_a_bowl_seed1111_g5.0_s30/img_000.png",
    "/data2/toby/OSCAR/outputs/try3_bowl/imgs/a_photo_of_a_bowl_seed1111_g5.0_s30/img_004.png",
    "/data2/toby/OSCAR/outputs/try3_bowl/imgs/a_photo_of_a_bowl_seed1111_g5.0_s30/img_005.png",
    "/data2/toby/OSCAR/outputs/try3_bowl/imgs/a_photo_of_a_bowl_seed3333_g5.0_s30/img_000.png",
]
# -----------------------------------------------------------------
# /mnt/data6t/yyz/flow_grpo/flow_base/outputs/dpp_dining_table/imgs/a_dining_table_seed1111_g3.0_s30/01.png
# /mnt/data6t/yyz/flow_grpo/flow_base/outputs/dpp_dining_table/imgs/a_dining_table_seed1111_g3.0_s30/08.png
OUT_PDF = "results/ablation/ourmethod.pdf"

# MODE: "auto" / "cover" / "stretch" / "contain-pad"
MODE = "auto"
BG_COLOR = (255, 255, 255)

# Grid
COLS = 2
ROWS = 2
DPI = 300
A4_MM = (210, 297)
A4_PAGE_W = int(A4_MM[0] / 25.4 * DPI)
A4_PAGE_H = int(A4_MM[1] / 25.4 * DPI)


def load_images(paths):
    imgs = []
    sizes = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"图片不存在: {p}")
        im = Image.open(p)
        if im.mode != "RGB":
            im = im.convert("RGB")
        imgs.append(im)
        sizes.append(im.size)
    return imgs, sizes


def all_sizes_equal(sizes):
    return all(s == sizes[0] for s in sizes)


def make_2x2_tiled_pdf(image_paths, out_pdf, mode="auto", dpi=DPI, bg=(255, 255, 255)):
    imgs, sizes = load_images(image_paths)

    # 补足或截取到 4 张
    if len(imgs) < 4:
        while len(imgs) < 4:
            imgs.append(Image.new("RGB", (100, 100), bg))
            sizes.append((100, 100))
    elif len(imgs) > 4:
        imgs = imgs[:4]
        sizes = sizes[:4]

    # 若尺寸一致且 mode 是 auto -> 原像素拼接
    if mode == "auto" and all_sizes_equal(sizes):
        single_w, single_h = sizes[0]
        page_w = single_w * COLS
        page_h = single_h * ROWS
        page = Image.new("RGB", (page_w, page_h), bg)
        for idx, im in enumerate(imgs):
            r = idx // COLS
            c = idx % COLS
            x0 = c * single_w
            y0 = r * single_h
            page.paste(im, (x0, y0))
        out_pdf = Path(out_pdf)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        page.save(out_pdf, "PDF", resolution=dpi)
        print(
            f"[OK] Saved exact-pixel tiled PDF: {out_pdf} (page {page_w}x{page_h} px)")
        return

    # 否则按 A4 页来分配单元格
    cell_w = A4_PAGE_W // COLS
    cell_h = A4_PAGE_H // ROWS
    page_w, page_h = A4_PAGE_W, A4_PAGE_H

    processed = []
    for im in imgs:
        if mode == "stretch":
            proc = im.resize((cell_w, cell_h), Image.LANCZOS)
        elif mode == "contain-pad":
            iw, ih = im.size
            scale = min(cell_w / iw, cell_h / ih)
            new_w = max(1, int(iw * scale))
            new_h = max(1, int(ih * scale))
            resized = im.resize((new_w, new_h), Image.LANCZOS)
            canvas = Image.new("RGB", (cell_w, cell_h), bg)
            left = (cell_w - new_w) // 2
            top = (cell_h - new_h) // 2
            canvas.paste(resized, (left, top))
            proc = canvas
        else:
            proc = ImageOps.fit(im, (cell_w, cell_h),
                                method=Image.LANCZOS, centering=(0.5, 0.5))
        processed.append(proc)

    page = Image.new("RGB", (page_w, page_h), bg)
    for idx, im in enumerate(processed[:4]):
        r = idx // COLS
        c = idx % COLS
        x0 = c * cell_w
        y0 = r * cell_h
        page.paste(im, (x0, y0))

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    page.save(out_pdf, "PDF", resolution=dpi)
    print(
        f"[OK] Saved tiled PDF: {out_pdf} (page {page_w}x{page_h} px, mode={mode})")


if __name__ == "__main__":
    if not isinstance(IMAGE_PATHS, (list, tuple)):
        print("请将 IMAGE_PATHS 设为包含图片路径的列表。")
        sys.exit(2)

    if len(IMAGE_PATHS) < 4:
        print(
            f"WARNING: IMAGE_PATHS 长度为 {len(IMAGE_PATHS)} (<4)。脚本会用占位图补足到 4 张。")
    elif len(IMAGE_PATHS) > 4:
        print(f"WARNING: IMAGE_PATHS 长度为 {len(IMAGE_PATHS)} (>4)。脚本将只使用前 4 张。")

    try:
        make_2x2_tiled_pdf(IMAGE_PATHS, OUT_PDF,
                           mode=MODE, dpi=DPI, bg=BG_COLOR)
    except Exception as e:
        print("错误：", e)
        raise

# from pathlib import Path
# from PIL import Image, ImageOps
# import sys

# # ----------------- 在这里直接编辑你的 8 张图片路径 -----------------
# IMAGE_PATHS = [
#     "/mnt/data6t/yyz/flow_grpo/flow_base/outputs/ourmethod_truck/imgs/a_photo_of_a_truck_seed4444_g3.0_s30/img_001.png",
#     "/mnt/data6t/yyz/flow_grpo/flow_base/outputs/ourmethod_truck/imgs/a_photo_of_a_truck_seed4444_g3.0_s30/img_007.png",
#     "/mnt/data6t/yyz/flow_grpo/flow_base/outputs/ourmethod_truck/imgs/a_photo_of_a_truck_seed4444_g3.0_s30/img_016.png",
#     "/mnt/data6t/yyz/flow_grpo/flow_base/outputs/ourmethod_truck/imgs/a_photo_of_a_truck_seed4444_g3.0_s30/img_018.png",
#     "/mnt/data6t/yyz/flow_grpo/flow_base/outputs/ourmethod_truck/imgs/a_photo_of_a_truck_seed3333_g3.0_s30/img_010.png",
#     "/mnt/data6t/yyz/flow_grpo/flow_base/outputs/ourmethod_truck/imgs/a_photo_of_a_truck_seed3333_g3.0_s30/img_022.png",
#     "/mnt/data6t/yyz/flow_grpo/flow_base/outputs/ourmethod_truck/imgs/a_photo_of_a_truck_seed4444_g3.0_s30/img_054.png",
#     "/mnt/data6t/yyz/flow_grpo/flow_base/outputs/ourmethod_truck/imgs/a_photo_of_a_truck_seed4444_g3.0_s30/img_060.png",
# ]
# # -----------------------------------------------------------------

# OUT_PDF = "results/tmlt.pdf"
# # dpp /mnt/data6t/yyz/flow_grpo/flow_base/outputs/dpp_dog/imgs/a_dog_seed1111_g3.0_s30/04.png
# # our 
# # dpp /mnt/data6t/yyz/flow_grpo/flow_base/outputs/dpp_dog/imgs/a_dog_seed1111_g5.0_s30/09.png
# # dpp /mnt/data6t/yyz/flow_grpo/flow_base/outputs/dpp_dog/imgs/a_dog_seed1111_g5.0_s30/29.png

# # MODE:
# #   "auto" - 如果所有图尺寸相同则原像素拼接；否则默认使用 COVER（无留白但可能裁剪）
# #   "cover" - 使用 cover（ImageOps.fit），等比放大并裁剪以填满单元格（无留白）
# #   "stretch" - 直接拉伸每张图到单元格尺寸（无裁剪、无留白，但会变形）
# #   "contain-pad" - 等比缩放并在单元格内居中，周围用 bg 填充（会有留白）
# MODE = "auto"  # 可改为 "cover" / "stretch" / "contain-pad"

# # 如果 MODE 中用到填充颜色，指定背景色
# BG_COLOR = (255, 255, 255)  # 白色（仅在 contain-pad 或默认 page background 时有意义）

# # 网格参数（固定 2 列 x 4 排）
# COLS = 4
# ROWS = 2

# # 默认 A4 输出（如果采用 cover/stretch/contain-pad，会把每单元按 A4 页分割）
# # 但如果所有图相同尺寸并使用 "原像素拼接" 模式，会以图像尺寸为准，不使用 A4
# DPI = 300
# A4_MM = (210, 297)  # mm
# A4_PAGE_W = int(A4_MM[0] / 25.4 * DPI)
# A4_PAGE_H = int(A4_MM[1] / 25.4 * DPI)


# def load_images(paths):
#     imgs = []
#     sizes = []
#     for p in paths:
#         p = Path(p)
#         if not p.exists():
#             raise FileNotFoundError(f"图片不存在: {p}")
#         im = Image.open(p)
#         if im.mode != "RGB":
#             im = im.convert("RGB")
#         imgs.append(im)
#         sizes.append(im.size)
#     return imgs, sizes


# def all_sizes_equal(sizes):
#     return all(s == sizes[0] for s in sizes)


# def make_tiled_pdf_no_crop(image_paths, out_pdf, mode="auto", dpi=DPI, bg=(255,255,255)):
#     imgs, sizes = load_images(image_paths)

#     # 补足或截取到 8 张
#     if len(imgs) < 8:
#         while len(imgs) < 8:
#             imgs.append(Image.new("RGB", (100, 100), bg))
#             sizes.append((100, 100))
#     elif len(imgs) > 8:
#         imgs = imgs[:8]
#         sizes = sizes[:8]

#     # 若所有尺寸相同，并且 mode 为 auto 或者想要原像素拼接，则直接按原像素拼接（无裁剪、无拉伸）
#     if mode == "auto" and all_sizes_equal(sizes):
#         single_w, single_h = sizes[0]
#         page_w = single_w * COLS
#         page_h = single_h * ROWS
#         page = Image.new("RGB", (page_w, page_h), bg)
#         for idx, im in enumerate(imgs):
#             r = idx // COLS
#             c = idx % COLS
#             x0 = c * single_w
#             y0 = r * single_h
#             page.paste(im, (x0, y0))
#         out_pdf = Path(out_pdf)
#         out_pdf.parent.mkdir(parents=True, exist_ok=True)
#         page.save(out_pdf, "PDF", resolution=dpi)
#         print(f"[OK] Saved exact-pixel tiled PDF: {out_pdf} (page {page_w}x{page_h} px)")
#         return

#     # 否则我们需要决定单元格尺寸。这里默认使用 A4 页面（纵向），把页面均分为 COLS x ROWS 单元格，
#     # 你也可以改成按最大图片尺寸决定单元格大小（如需我可以改）。
#     cell_w = A4_PAGE_W // COLS
#     cell_h = A4_PAGE_H // ROWS
#     page_w, page_h = A4_PAGE_W, A4_PAGE_H

#     # 根据模式处理每张图以填充单元格
#     processed = []
#     for im in imgs:
#         if mode == "stretch":
#             # 直接拉伸到单元格大小（会变形，但无留白、无裁剪）
#             proc = im.resize((cell_w, cell_h), Image.LANCZOS)
#         elif mode == "contain-pad":
#             # 等比缩放到单元格内并用背景填充（会有留白）
#             iw, ih = im.size
#             scale = min(cell_w / iw, cell_h / ih)
#             new_w = max(1, int(iw * scale))
#             new_h = max(1, int(ih * scale))
#             resized = im.resize((new_w, new_h), Image.LANCZOS)
#             canvas = Image.new("RGB", (cell_w, cell_h), bg)
#             left = (cell_w - new_w) // 2
#             top = (cell_h - new_h) // 2
#             canvas.paste(resized, (left, top))
#             proc = canvas
#         else:
#             # cover（默认）：等比放大并居中裁剪以完全填满单元格（无留白）
#             proc = ImageOps.fit(im, (cell_w, cell_h), method=Image.LANCZOS, centering=(0.5,0.5))
#         processed.append(proc)

#     # 拼接到 A4 页（单张）
#     page = Image.new("RGB", (page_w, page_h), bg)
#     for idx, im in enumerate(processed[:8]):
#         r = idx // COLS
#         c = idx % COLS
#         x0 = c * cell_w
#         y0 = r * cell_h
#         page.paste(im, (x0, y0))

#     out_pdf = Path(out_pdf)
#     out_pdf.parent.mkdir(parents=True, exist_ok=True)
#     page.save(out_pdf, "PDF", resolution=dpi)
#     print(f"[OK] Saved tiled PDF: {out_pdf} (page {page_w}x{page_h} px, mode={mode})")


# if __name__ == "__main__":
#     if not isinstance(IMAGE_PATHS, (list, tuple)):
#         print("请将 IMAGE_PATHS 设为包含图片路径的列表。")
#         sys.exit(2)

#     if len(IMAGE_PATHS) < 8:
#         print(f"WARNING: IMAGE_PATHS 长度为 {len(IMAGE_PATHS)} (<8)。脚本会用占位图补足到 8 张。")
#     elif len(IMAGE_PATHS) > 8:
#         print(f"WARNING: IMAGE_PATHS 长度为 {len(IMAGE_PATHS)} (>8)。脚本将只使用前 8 张。")

#     # 如果你明确想要不裁剪且不失真，请把 MODE 改为 "stretch"（会变形）或先保证所有图片相同尺寸（512x512），MODE="auto" 会直接按原像素拼接
#     try:
#         make_tiled_pdf_no_crop(IMAGE_PATHS, OUT_PDF, mode=MODE, dpi=DPI, bg=BG_COLOR)
#     except Exception as e:
#         print("错误：", e)
#         raise