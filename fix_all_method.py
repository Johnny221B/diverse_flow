from PIL import Image, ImageOps, ImageDraw, ImageFont
from pathlib import Path
from difflib import SequenceMatcher
import re
import textwrap


def make_method_grid_auto(
    ref_path,
    methods,
    method_to_indices,
    output_prefix,
    source_method="ourmethod",
    task_suffix="t2_color",
    imgs_dir_name="imgs",
    prefix_len=20,
    outer_padding=20,
    inner_padding=8,
    bg_color="white",
    image_ext=".png",
    grid_rows=4,
    grid_cols=2,
    prompt_text=None,                 # 新增：可手动指定 prompt
    method_title_ratio=0.16,          # 标题区高度占单张图高度比例
    bottom_caption_ratio=0.30,        # 底部 prompt 区高度占单张图高度比例
    title_top_bottom_pad=12,
    caption_pad=18,
    max_method_font_size=80,
    min_method_font_size=18,
    max_prompt_font_size=75,
    min_prompt_font_size=20,
):
    ref_path = Path(ref_path)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_path}")

    if source_method not in methods:
        raise ValueError(f"{source_method} must be included in methods")

    expected_num = grid_rows * grid_cols

    for method in methods:
        if method not in method_to_indices:
            raise ValueError(f"Missing indices for method: {method}")
        if len(method_to_indices[method]) != expected_num:
            raise ValueError(f"{method} must have exactly {expected_num} indices")

    def method_dir_name(method):
        return f"{method}_{task_suffix}"

    def normalize_text(s):
        return re.sub(r'[^a-z0-9]+', '', s.lower())

    def prettify_prompt(s):
        # 文件夹名字恢复成人可读 prompt
        s = s.replace("_", " ").strip()
        s = re.sub(r"\s+", " ", s)
        return s

    def prettify_method_name(name):
        # 你可以按需要自定义显示名字
        mapping = {
            "ourmethod": "OSCAR",
            "pg": "PG",
            "cads": "CADS",
            "dpp": "DPP",
            "base": "Base Model",
            "mix": "Mix ODE/SDE",
            "apg": "APG",
        }
        return mapping.get(name, name)

    def find_base_root_and_prompt_dir(p):
        parts = p.parts
        target = method_dir_name(source_method)
        for i, part in enumerate(parts):
            if part == target:
                if i + 2 >= len(parts):
                    raise ValueError(f"Invalid reference path: {p}")
                if parts[i + 1] != imgs_dir_name:
                    raise ValueError(f"Invalid reference path: {p}")
                base_root = Path(*parts[:i])
                prompt_dir = parts[i + 2]
                return base_root, prompt_dir
        raise ValueError(f"Cannot find source method folder in path: {p}")

    def resolve_prompt_dir(method, query_prompt, base_root):
        imgs_root = base_root / method_dir_name(method) / imgs_dir_name
        if not imgs_root.exists():
            raise FileNotFoundError(f"Directory not found: {imgs_root}")

        if method == source_method:
            exact_dir = imgs_root / query_prompt
            if exact_dir.exists() and exact_dir.is_dir():
                return exact_dir

        candidates = [x for x in imgs_root.iterdir() if x.is_dir()]

        # exact normalized full-name match (handles case differences like dpp lowercase)
        for c in candidates:
            if normalize_text(c.name) == normalize_text(query_prompt):
                return c
        if len(candidates) == 0:
            raise FileNotFoundError(f"No prompt folders found under {imgs_root}")

        qn = normalize_text(query_prompt)
        qprefix = qn[:prefix_len]

        exact_prefix_matches = []
        contained_matches = []
        fuzzy_matches = []

        for c in candidates:
            cn = normalize_text(c.name)

            if cn.startswith(qprefix):
                exact_prefix_matches.append(c)
            elif qprefix.startswith(cn) or qprefix in cn or cn in qn:
                contained_matches.append(c)
            else:
                sim = max(
                    SequenceMatcher(None, qn, cn).ratio(),
                    SequenceMatcher(None, qprefix, cn[:len(qprefix)]).ratio() if cn else 0.0
                )
                fuzzy_matches.append((sim, c))

        if exact_prefix_matches:
            exact_prefix_matches.sort(key=lambda x: (len(x.name), x.name))
            return exact_prefix_matches[0]

        if contained_matches:
            contained_matches.sort(key=lambda x: (-len(normalize_text(x.name)), len(x.name), x.name))
            return contained_matches[0]

        if fuzzy_matches:
            fuzzy_matches.sort(key=lambda x: x[0], reverse=True)
            best_sim, best_path = fuzzy_matches[0]
            if best_sim >= 0.55:
                return best_path

        raise FileNotFoundError(f"No matched prompt folder for '{query_prompt}' under {imgs_root}")

    def get_font(size):
        # 尽量用常见 TrueType 字体；找不到就退回默认字体
        candidate_fonts = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/Library/Fonts/Arial.ttf",
            "arial.ttf",
        ]
        for fp in candidate_fonts:
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                continue
        return ImageFont.load_default()

    def text_bbox(draw, text, font):
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        return (x1 - x0, y1 - y0)

    def fit_single_line_font(draw, text, max_width, max_size, min_size):
        # 自动找一个单行字体大小，宽度不超过 max_width
        for size in range(max_size, min_size - 1, -2):
            font = get_font(size)
            w, h = text_bbox(draw, text, font)
            if w <= max_width:
                return font, w, h
        font = get_font(min_size)
        w, h = text_bbox(draw, text, font)
        return font, w, h

    def wrap_text_by_pixel(draw, text, font, max_width):
        words = text.split()
        if not words:
            return [""]

        lines = []
        current = words[0]
        for w in words[1:]:
            trial = current + " " + w
            tw, _ = text_bbox(draw, trial, font)
            if tw <= max_width:
                current = trial
            else:
                lines.append(current)
                current = w
        lines.append(current)
        return lines

    def fit_multiline_font(draw, text, max_width, max_height, max_size, min_size, line_spacing=6):
        for size in range(max_size, min_size - 1, -2):
            font = get_font(size)
            lines = wrap_text_by_pixel(draw, text, font, max_width)
            line_heights = [text_bbox(draw, ln, font)[1] for ln in lines]
            total_h = sum(line_heights) + line_spacing * (len(lines) - 1)
            max_line_w = max(text_bbox(draw, ln, font)[0] for ln in lines) if lines else 0
            if total_h <= max_height and max_line_w <= max_width:
                return font, lines, total_h
        font = get_font(min_size)
        lines = wrap_text_by_pixel(draw, text, font, max_width)
        line_heights = [text_bbox(draw, ln, font)[1] for ln in lines]
        total_h = sum(line_heights) + line_spacing * (len(lines) - 1)
        return font, lines, total_h

    base_root, query_prompt_dir = find_base_root_and_prompt_dir(ref_path)
    method_to_image_paths = {}

    for method in methods:
        prompt_dir = resolve_prompt_dir(method, query_prompt_dir, base_root)
        indices = method_to_indices[method]
        paths = []

        for idx in indices:
            idx_str = str(idx)
            if not idx_str.endswith(image_ext):
                idx_str = idx_str + image_ext
            img_path = prompt_dir / idx_str
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            paths.append(img_path)

        method_to_image_paths[method] = paths

    loaded = {}
    for method, paths in method_to_image_paths.items():
        loaded[method] = [Image.open(p).convert("RGB") for p in paths]

    # 所有图统一到最小尺寸
    cell_w = min(img.size[0] for imgs in loaded.values() for img in imgs)
    cell_h = min(img.size[1] for imgs in loaded.values() for img in imgs)

    title_h = max(36, int(cell_h * method_title_ratio))
    caption_h = max(50, int(cell_h * bottom_caption_ratio))

    # 先生成每个 method block（标题 + 2x4 图片）
    method_blocks = []
    dummy_img = Image.new("RGB", (10, 10), bg_color)
    dummy_draw = ImageDraw.Draw(dummy_img)

    for method in methods:
        imgs = [ImageOps.fit(img, (cell_w, cell_h)) for img in loaded[method]]

        grid_w = cell_w * grid_cols + inner_padding * (grid_cols - 1)
        grid_h = cell_h * grid_rows + inner_padding * (grid_rows - 1)

        block_w = grid_w
        block_h = title_h + title_top_bottom_pad + grid_h

        block = Image.new("RGB", (block_w, block_h), bg_color)
        draw = ImageDraw.Draw(block)

        # 方法名：自动调大，但不超过 block 宽度
        method_title = prettify_method_name(method)
        title_font, text_w, text_h = fit_single_line_font(
            draw,
            method_title,
            max_width=block_w - 12,
            max_size=max_method_font_size,
            min_size=min_method_font_size
        )
        tx = (block_w - text_w) // 2
        ty = (title_h - text_h) // 2
        draw.text((tx, ty), method_title, fill="black", font=title_font)

        # 图片网格
        grid_y0 = title_h + title_top_bottom_pad
        for i, img in enumerate(imgs):
            r = i // grid_cols
            c = i % grid_cols
            x = c * (cell_w + inner_padding)
            y = grid_y0 + r * (cell_h + inner_padding)
            block.paste(img, (x, y))

        method_blocks.append(block)

    total_w = sum(block.size[0] for block in method_blocks) + outer_padding * (len(method_blocks) - 1)
    total_h = max(block.size[1] for block in method_blocks) + caption_h + caption_pad

    canvas = Image.new("RGB", (total_w, total_h), bg_color)

    x = 0
    max_block_h = max(block.size[1] for block in method_blocks)
    for block in method_blocks:
        canvas.paste(block, (x, 0))
        x += block.size[0] + outer_padding

    # 底部 prompt
    draw = ImageDraw.Draw(canvas)
    if prompt_text is None:
        prompt_text = prettify_prompt(query_prompt_dir)

    caption_y0 = max_block_h
    caption_box_x0 = caption_pad
    caption_box_y0 = caption_y0 + 4
    caption_box_w = total_w - 2 * caption_pad
    caption_box_h = caption_h

    prompt_font, prompt_lines, total_text_h = fit_multiline_font(
        draw,
        text=f"Prompt: {prompt_text}",
        max_width=caption_box_w,
        max_height=caption_box_h - 8,
        max_size=max_prompt_font_size,
        min_size=min_prompt_font_size,
        line_spacing=6
    )

    current_y = caption_box_y0 + max(0, (caption_box_h - total_text_h) // 2)
    for line in prompt_lines:
        line_w, line_h = text_bbox(draw, line, prompt_font)
        line_x = caption_box_x0 + (caption_box_w - line_w) // 2
        draw.text((line_x, current_y), line, fill="black", font=prompt_font)
        current_y += line_h + 6

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")

    canvas.save(png_path)
    canvas.save(pdf_path, "PDF", resolution=300.0)

    return canvas, str(png_path), str(pdf_path)


ref_path = "/data2/toby/OSCAR/outputs/ourmethod_human/imgs/A_portrait_of_an_artist_in_a_paint-splattered_apron_standing_in_front_of_a_large_abstract_canvas_holding_a_brush_messy_h/000.png"

methods = ["ourmethod", "pg", "cads", "dpp", "base", "mix", "apg"]

# method_to_indices = {
#     "ourmethod": ["014", "009", "004", "003", "010", "013", "011", "008"],
#     "pg": ["001", "002", "003", "013", "004", "014", "007", "011"],
#     "cads": ["001", "002", "003", "004", "005", "000", "006", "015"],
#     "dpp": ["009", "002", "006", "010", "005", "014", "007", "008"],
#     "base": ["001", "002", "003", "010", "011", "014", "006", "015"],
#     "mix": ["001", "002", "006", "015", "005", "014", "013", "011"],
#     "apg": ["001", "002", "006", "015", "005", "014", "013", "011"],
# }
method_to_indices = {
    "ourmethod": ["000", "002", "015", "031"],
    "pg": ["000", "008", "018", "025"],
    "cads": ["000", "008", "018", "025"],
    "dpp": ["000", "008", "018", "025"],
    "base": ["000", "008", "018", "025"],
    "mix": ["000", "008", "018", "025"],
    "apg": ["000", "008", "014", "015"],
}

canvas, png_path, pdf_path = make_method_grid_auto(
    ref_path=ref_path,
    methods=methods,
    method_to_indices=method_to_indices,
    output_prefix="/data2/toby/OSCAR/results/human/artist",
    source_method="ourmethod",
    task_suffix="human",
    grid_rows=2,
    grid_cols=2,
    prompt_text="A portrait of an artist in a paint-splattered apron standing in front of a large abstract canvas, holding a brush, messy hair, warm studio light."   # 也可以不写，默认从文件夹名恢复
)

# The_black_camera_was_mounted_on_the_silver_tripod
# a blue bench and a green boat
print(png_path)
print(pdf_path)