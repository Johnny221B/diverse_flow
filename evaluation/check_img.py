# import cv2
# import os
# from pathlib import Path
# import argparse

# parser = argparse.ArgumentParser(description="Find corrupted or small images in folders.")
# parser.add_argument('folders', nargs='+', type=str, help='Path to one or more image folders to check.')
# parser.add_argument('--min-size', type=int, default=11, help='Minimum allowed width and height.')
# args = parser.parse_args()

# def check_images(root_folder, min_size):
#     print(f"\n--- Checking folder: {root_folder} ---")
#     image_paths = list(Path(root_folder).rglob("*.jpg")) + list(Path(root_folder).rglob("*.png"))
   
#     bad_files = []
#     for image_path in image_paths:
#         try:
#             img = cv2.imread(str(image_path))
#             if img is None:
#                 print(f"[CORRUPTED] Cannot read file: {image_path}")
#                 bad_files.append(image_path)
#                 continue
           
#             h, w, _ = img.shape
#             if h < min_size or w < min_size:
#                 print(f"[TOO SMALL] Image dimensions are {w}x{h}: {image_path}")
#                 bad_files.append(image_path)
#         except Exception as e:
#             print(f"[ERROR] Exception '{e}' while processing file: {image_path}")
#             bad_files.append(image_path)
   
#     if not bad_files:
#         print("No problematic images found.")
#     else:
#         print(f"\nFound {len(bad_files)} problematic images in {root_folder}")
#     return bad_files

# if __name__ == '__main__':
#     all_bad_files = []
#     for folder in args.folders:
#         all_bad_files.extend(check_images(folder, args.min_size))
   
#     if all_bad_files:
#         print("\n--- Summary of all problematic files ---")
#         for f in all_bad_files:
#             print(f)

import sys
import numpy as np

print("--- Starting BRISQUE Sanity Check ---")

# 1. 打印当前使用的Python解释器路径，确认环境是否正确
print(f"Python Executable: {sys.executable}")

# 2. 导入并打印scikit-image的版本号
try:
    import skimage
    print(f"Found scikit-image version: {skimage.__version__}")
except ImportError:
    print("Error: scikit-image is not installed.")
    sys.exit()

# 3. 导入brisque库
try:
    import imquality.brisque as brisque
    import cv2
    print("Successfully imported imquality.brisque and cv2.")
except ImportError as e:
    print(f"Error importing libraries: {e}")
    sys.exit()

# 4. 创建一个虚拟的彩色图片，并尝试计算BRISQUE分数
try:
    print("\nAttempting to calculate BRISQUE score on a dummy image...")
    # 创建一个 100x100 的3通道随机彩色图片
    dummy_image_rgb = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
   
    # 调用有问题的函数
    score = brisque.score(dummy_image_rgb)
   
    print("\n--- S U C C E S S ---")
    print(f"Successfully calculated BRISQUE score: {score}")
    print("This means the libraries are working correctly together.")

except TypeError as e:
    print("\n--- F A I L U R E ---")
    print("Caught the exact TypeError we were expecting.")
    print(f"Error Message: {e}")
    print("This confirms a deep version conflict in your environment.")
except Exception as e:
    print("\n--- UNEXPECTED ERROR ---")
    print(f"Caught an unexpected error: {e}")