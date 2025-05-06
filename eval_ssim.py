from skimage.metrics import structural_similarity as ssim
import cv2
import os
import re

origin_folder = "/home/cy/Gra_design/dataset/spine_phantom/left2/images"
train_folder = "/home/cy/Gra_design/us_nerf_pro/logs/spine_phantom_left2_/output_maps_left2_model_012000_0/output"

def get_numeric_key(filename):
    # 提取文件名中的数字部分，默认取第一个出现的数字
    nums = re.findall(r'\d+', filename)
    return int(nums[0]) if nums else -1  # 没有数字则返回 -1，排在前面

files1 = sorted([f for f in os.listdir(origin_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
    key=get_numeric_key)
files2 = sorted([f for f in os.listdir(train_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
    key=get_numeric_key)

# 检查数量一致性
if len(files1) != len(files2):
    raise ValueError("两个文件夹中的图片数量不一致")

ssim_score = 0

# 遍历所有图片，计算 SSIM
for f1, f2 in zip(files1, files2):
    path1 = os.path.join(origin_folder, f1)
    path2 = os.path.join(train_folder, f2)

    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

    if img1.shape != img2.shape:
        raise ValueError(f"图像尺寸不匹配：{f1} vs {f2}")

    score, _ = ssim(img1, img2, full=True)
    print(f"{f1} vs {f2}: SSIM = {score:.4f}")

    ssim_score += score
    
average_ssim = ssim_score / len(files1)

print(f"Average SSIM = {average_ssim:.4f}")
    


