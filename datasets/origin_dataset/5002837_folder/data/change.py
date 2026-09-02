import os
from PIL import Image

def convert_tif_to_png(input_folder, output_folder):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.tif', '.tiff')):
            # 输入输出文件路径
            tif_path = os.path.join(input_folder, filename)
            png_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')

            try:
                # 打开并转换为PNG
                with Image.open(tif_path) as img:
                    img.convert('RGB').save(png_path, 'PNG')
                print(f"✅ 已转换: {filename} → {os.path.basename(png_path)}")
            except Exception as e:
                print(f"❌ 转换失败: {filename}, 错误信息: {e}")

    print("🎉 全部转换完成！")

if __name__ == "__main__":
    # 输入与输出路径（可根据需要修改）
    input_dir = "/data1/chenxin/dataset/1-20/5002837_folder"     # 放置.tif文件的文件夹
    output_dir = "/data1/chenxin/dataset/1-20/5002837_folder_new"   # 保存.png文件的文件夹

    convert_tif_to_png(input_dir, output_dir)
