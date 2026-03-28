from PIL import Image, ImageOps, ImageDraw
import os


def make_concept_grid(
    image_paths,
    output_path="merged.png",
    concept_names=None,
    images_per_concept=8,
    concept_cols=2,
    concept_rows=4,
    num_concepts=3,
    cell_size=(512, 512),
    padding=10,
    concept_gap=30,
    add_concept_title=False,
    title_height=50,
    bg_color=(255, 255, 255),
):
    """
    image_paths: 长度必须为 num_concepts * images_per_concept
    默认:
        3 个 concept
        每个 concept 8 张图
        每个 concept 2列4行
        总图 6列4行
    """

    expected = num_concepts * images_per_concept
    if len(image_paths) != expected:
        raise ValueError(f"需要 {expected} 张图，但你给了 {len(image_paths)} 张。")

    if concept_names is None:
        concept_names = [f"Concept {i+1}" for i in range(num_concepts)]

    if len(concept_names) != num_concepts:
        raise ValueError("concept_names 的长度必须等于 num_concepts。")

    cell_w, cell_h = cell_size

    # 每个 concept 的宽高
    one_concept_width = concept_cols * cell_w + (concept_cols - 1) * padding
    one_concept_height = concept_rows * cell_h + (concept_rows - 1) * padding

    extra_title_h = title_height if add_concept_title else 0

    canvas_width = (
        num_concepts * one_concept_width
        + (num_concepts - 1) * concept_gap
    )
    canvas_height = one_concept_height + extra_title_h

    canvas = Image.new("RGB", (canvas_width, canvas_height), bg_color)
    draw = ImageDraw.Draw(canvas)

    for concept_idx in range(num_concepts):
        start = concept_idx * images_per_concept
        end = start + images_per_concept
        concept_paths = image_paths[start:end]

        x_offset = concept_idx * (one_concept_width + concept_gap)
        y_offset = extra_title_h

        if add_concept_title:
            title = concept_names[concept_idx]
            draw.text((x_offset + 10, 10), title, fill=(0, 0, 0))

        for i, img_path in enumerate(concept_paths):
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"找不到图片: {img_path}")

            img = Image.open(img_path).convert("RGB")
            img = ImageOps.fit(img, (cell_w, cell_h), method=Image.Resampling.LANCZOS)

            row = i // concept_cols
            col = i % concept_cols

            paste_x = x_offset + col * (cell_w + padding)
            paste_y = y_offset + row * (cell_h + padding)

            canvas.paste(img, (paste_x, paste_y))

    canvas.save(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    image_paths = [
        # concept 1: 8 张
        "/data2/toby/OSCAR/outputs/mix_t2i_color/imgs/a_blue_bench_and_a_green_boat/000.png",
        "/data2/toby/OSCAR/outputs/mix_t2i_color/imgs/a_blue_bench_and_a_green_boat/001.png",
        "/data2/toby/OSCAR/outputs/mix_t2i_color/imgs/a_blue_bench_and_a_green_boat/002.png",
        "/data2/toby/OSCAR/outputs/mix_t2i_color/imgs/a_blue_bench_and_a_green_boat/003.png",
        "/data2/toby/OSCAR/outputs/mix_t2i_color/imgs/a_blue_bench_and_a_green_boat/004.png",
        "/data2/toby/OSCAR/outputs/mix_t2i_color/imgs/a_blue_bench_and_a_green_boat/005.png",
        "/data2/toby/OSCAR/outputs/mix_t2i_color/imgs/a_blue_bench_and_a_green_boat/006.png",
        "/data2/toby/OSCAR/outputs/mix_t2i_color/imgs/a_blue_bench_and_a_green_boat/007.png",

        # concept 2: 8 张
        "/data2/toby/OSCAR/outputs/mix_t2i_complex/imgs/The_black_camera_was_mounted_on_the_silver_tripod/000.png",
        "/data2/toby/OSCAR/outputs/mix_t2i_complex/imgs/The_black_camera_was_mounted_on_the_silver_tripod/001.png",
        "/data2/toby/OSCAR/outputs/mix_t2i_complex/imgs/The_black_camera_was_mounted_on_the_silver_tripod/002.png",
        "/data2/toby/OSCAR/outputs/mix_t2i_complex/imgs/The_black_camera_was_mounted_on_the_silver_tripod/003.png",
        "/data2/toby/OSCAR/outputs/mix_t2i_complex/imgs/The_black_camera_was_mounted_on_the_silver_tripod/010.png",
        "/data2/toby/OSCAR/outputs/mix_t2i_complex/imgs/The_black_camera_was_mounted_on_the_silver_tripod/005.png",
        "/data2/toby/OSCAR/outputs/mix_t2i_complex/imgs/The_black_camera_was_mounted_on_the_silver_tripod/006.png",
        "/data2/toby/OSCAR/outputs/mix_t2i_complex/imgs/The_black_camera_was_mounted_on_the_silver_tripod/007.png",

        # concept 3: 8 张
        "/data2/toby/OSCAR/outputs/mix_t2i_spatial/imgs/a_bicycle_on_the_left_of_a_bird/000.png",
        "/data2/toby/OSCAR/outputs/mix_t2i_spatial/imgs/a_bicycle_on_the_left_of_a_bird/001.png",
        "/data2/toby/OSCAR/outputs/mix_t2i_spatial/imgs/a_bicycle_on_the_left_of_a_bird/002.png",
        "/data2/toby/OSCAR/outputs/mix_t2i_spatial/imgs/a_bicycle_on_the_left_of_a_bird/003.png",
        "/data2/toby/OSCAR/outputs/mix_t2i_spatial/imgs/a_bicycle_on_the_left_of_a_bird/004.png",
        "/data2/toby/OSCAR/outputs/mix_t2i_spatial/imgs/a_bicycle_on_the_left_of_a_bird/010.png",
        "/data2/toby/OSCAR/outputs/mix_t2i_spatial/imgs/a_bicycle_on_the_left_of_a_bird/006.png",
        "/data2/toby/OSCAR/outputs/mix_t2i_spatial/imgs/a_bicycle_on_the_left_of_a_bird/007.png",
    ]

    concept_names = ["truck", "bus", "bicycle"]

    make_concept_grid(
        image_paths=image_paths,
        output_path="mix.png",
        concept_names=concept_names,
        cell_size=(512,512),   # 可以改成 (256,256) 或 (512,512)
        padding=8,
        concept_gap=24,
        add_concept_title=False,
    )