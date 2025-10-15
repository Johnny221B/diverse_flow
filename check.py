from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

pipe = pipeline(Tasks.text_to_image_synthesis,
                model='AI-ModelScope/stable-diffusion-3.5-medium')

# 常见位置 1：有些封装直接暴露 scheduler
print('scheduler on pipe:', getattr(pipe, 'scheduler', None))

# 常见位置 2：scheduler 挂在内部的 diffusers 管线/模型上
for attr in ['pipeline', 'diffusion_pipeline', 'sd_pipeline', 'model']:
    inner = getattr(pipe, attr, None)
    if inner is not None:
        sch = getattr(inner, 'scheduler', None)
        if sch is not None:
            print('found on', attr, '->', sch.__class__.__name__)
            try:
                print(sch.config)   # 打印完整配置
            except Exception:
                pass
