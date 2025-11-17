from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PIL import Image
import os

def images_to_pdf_1x4(image_paths, output_pdf, page_size=A4, margin=20):
    """
    将四张图片按1排4列布局拼接成PDF
    
    参数:
        image_paths: 包含4个图片路径的列表
        output_pdf: 输出的PDF文件路径
        page_size: 页面尺寸，默认为A4
        margin: 页面边距，默认为20
    """
    if len(image_paths) != 4:
        raise ValueError("必须提供恰好4张图片路径")
    
    # 验证图片路径是否存在
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"图片路径不存在: {path}")
    
    # 计算布局参数
    usable_width = page_size[0] - 2 * margin  # 可用宽度
    column_width = usable_width / 4  # 每列宽度
    row_height = column_width * 0.75  # 行高（假设图片宽高比为4:3）
    
    # 创建PDF画布
    c = canvas.Canvas(output_pdf, pagesize=page_size)
    
    # 计算起始Y坐标（垂直居中）
    y_start = (page_size[1] - row_height) / 2
    
    # 添加每张图片到PDF
    for i, img_path in enumerate(image_paths):
        # 计算当前图片的X坐标
        x = margin + i * column_width
        
        # 在指定位置绘制图片（等分宽度，保持宽高比）
        c.drawImage(
            img_path,
            x, y_start,
            width=column_width,
            height=row_height,
            preserveAspectRatio=True,  # 保持宽高比
            anchor='c'  # 居中锚点
        )
    
    # 保存PDF
    c.save()

# 使用示例
if __name__ == "__main__":
    image_paths = [
        "/mnt/data6t/yyz/flow_grpo/flow_base/outputs/ablation/wo_OP/imgs/a_photo_of_a_bowl_seed1111_g5.0_s30/img_000.png", 
        "/mnt/data6t/yyz/flow_grpo/flow_base/outputs/ablation/wo_OP/imgs/a_photo_of_a_bowl_seed1111_g5.0_s30/img_004.png", 
        "/mnt/data6t/yyz/flow_grpo/flow_base/outputs/ablation/wo_OP/imgs/a_photo_of_a_bowl_seed1111_g5.0_s30/img_005.png", 
        "/mnt/data6t/yyz/flow_grpo/flow_base/outputs/ablation/wo_OP/imgs/a_photo_of_a_bowl_seed3333_g5.0_s30/img_000.png"
        ]
    output_pdf = "wo_OP.pdf"
    
    images_to_pdf_1x4(image_paths, output_pdf)
    print(f"PDF已生成: {output_pdf}")