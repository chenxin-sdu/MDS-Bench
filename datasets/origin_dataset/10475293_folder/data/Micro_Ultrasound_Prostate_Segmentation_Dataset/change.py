import os
import nibabel as nib
import numpy as np
from PIL import Image

# 设置输入和输出目录
input_root = "/data1/chenxin/dataset/1-20/10475293_folder"
output_root = "/data1/chenxin/dataset/1-20/10475293_folder_new"

# 遍历 input_root 目录下的所有文件
for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith(".nii"):  # 检查文件是否为.nii格式
            input_path = os.path.join(root, file)

            # 生成相对路径，用于输出目录
            rel_path = os.path.relpath(root, input_root)
            output_dir = os.path.join(output_root, rel_path)

            # 保证输出目录存在
            os.makedirs(output_dir, exist_ok=True)

            # 输出文件路径，替换后缀为.png
            output_file = os.path.splitext(file)[0] + ".png"
            output_path = os.path.join(output_dir, output_file)

            try:
                # 读取 NIfTI 文件 (.nii)
                nii_data = nib.load(input_path)
                img_data = nii_data.get_fdata()

                # 处理 NIfTI 文件数据，假设每个文件是一个3D图像，选取其中一层 (例如，选取第一个slice)
                slice_data = img_data[:, :, 0]  # 这里选择了第一个切片，如果需要其它切片可修改

                # 将 numpy 数据转换为 PIL 图像
                img = Image.fromarray(slice_data.astype(np.uint8))

                # 保存为 PNG 格式
                img.save(output_path, "PNG")
                print(f"✅ 转换完成: {input_path} -> {output_path}")
            except Exception as e:
                print(f"❌ 转换失败: {input_path}，错误信息: {e}")
