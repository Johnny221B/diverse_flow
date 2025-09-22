import os

def find_folders_with_not_64_images(root_path, target_count=64):
    """
    查找根目录下所有图片数量不等于target_count的文件夹名称。
    
    参数:
        root_path (str): 要检查的根目录路径。
        target_count (int): 目标图片数量，默认为64。
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')  # 常见图片扩展名[5,7](@ref)
    
    for root, dirs, files in os.walk(root_path):
        image_count = 0
        for file in files:
            if file.lower().endswith(image_extensions):
                image_count += 1
        
        # 如果当前文件夹的图片数量不等于目标值，则打印文件夹路径
        if image_count != target_count:
            # 获取当前文件夹相对于根目录的路径，或直接使用绝对路径
            folder_name = os.path.basename(root)  # 当前文件夹名称
            full_path = os.path.abspath(root)      # 当前文件夹绝对路径
            print(f"文件夹: {folder_name}, 路径: {full_path}, 图片数量: {image_count}")

# 使用示例
root_path = "/mnt/data6t/yyz/flow_grpo/flow_base/outputs/ourmethod_bus/imgs"  # 替换为您的实际路径
find_folders_with_not_64_images(root_path)