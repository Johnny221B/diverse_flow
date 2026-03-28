from PIL import Image, ImageOps
from pathlib import Path
from difflib import SequenceMatcher
import re

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
    grid_cols=2
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

    cell_w = min(img.size[0] for imgs in loaded.values() for img in imgs)
    cell_h = min(img.size[1] for imgs in loaded.values() for img in imgs)

    method_blocks = []
    for method in methods:
        imgs = [ImageOps.fit(img, (cell_w, cell_h)) for img in loaded[method]]

        block_w = cell_w * grid_cols + inner_padding * (grid_cols - 1)
        block_h = cell_h * grid_rows + inner_padding * (grid_rows - 1)

        block = Image.new("RGB", (block_w, block_h), bg_color)

        for i, img in enumerate(imgs):
            r = i // grid_cols
            c = i % grid_cols
            x = c * (cell_w + inner_padding)
            y = r * (cell_h + inner_padding)
            block.paste(img, (x, y))

        method_blocks.append(block)

    total_w = sum(block.size[0] for block in method_blocks) + outer_padding * (len(method_blocks) - 1)
    total_h = max(block.size[1] for block in method_blocks)

    canvas = Image.new("RGB", (total_w, total_h), bg_color)

    x = 0
    for block in method_blocks:
        canvas.paste(block, (x, 0))
        x += block.size[0] + outer_padding

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")

    canvas.save(png_path)
    canvas.save(pdf_path, "PDF", resolution=300.0)

    return canvas, str(png_path), str(pdf_path)

ref_path = "/data2/toby/OSCAR/outputs/ourmethod_t2i_spatial/imgs/a_bicycle_on_the_left_of_a_bird/000.png"

methods = ["ourmethod", "pg", "cads", "dpp", "base", "mix", "apg"]

method_to_indices = {
    "ourmethod": ["000", "001", "004", "005", "010", "013", "012", "008"],
    "pg": ["001", "002", "003", "013", "004", "014", "007", "011"],
    "cads": ["001", "002", "003", "010", "005", "014", "007", "011"],
    "dpp": ["001", "002", "006", "010", "005", "014", "007", "008"],
    "base": ["001", "002", "003", "010", "011", "014", "007", "015"],
    "mix": ["001", "002", "003", "010", "011", "014", "007", "015"],
    "apg": ["001", "002", "003", "010", "011", "014", "007", "015"],
}

canvas, png_path, pdf_path = make_method_grid_auto(
    ref_path=ref_path,
    methods=methods,
    method_to_indices=method_to_indices,
    output_prefix="/data2/toby/OSCAR/results/complex/spatial1",
    source_method="ourmethod",
    task_suffix="t2i_spatial",
    grid_rows=4,
    grid_cols=2
)