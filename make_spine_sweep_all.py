import os 
from load_us import load_us_data
import tensorflow as tf
import numpy as np
import shutil

data_dir = ['left1', 'left1_1', 'left2', 'left3', 'left3_2', 'right1', 'right1_1', 'right3', 'right3_2']
data_base = "/home/cy/Gra_design/dataset/spine_phantom/"

result_images = []
result_poses = []
    
for item in data_dir:
    data_path = os.path.join(data_base, item)
    images, poses, i_test = load_us_data(data_path)
    
    if item == "left1":
        result_images = images
        result_poses = poses
    else:
        result_images = tf.concat([result_images, images], axis=0)
        result_poses = tf.concat([result_poses, poses], axis=0)


np.save("/home/cy/Gra_design/dataset/spine_phantom/all_sweeps/images.npy", result_images)
np.save("/home/cy/Gra_design/dataset/spine_phantom/all_sweeps/poses.npy", result_poses)


def consolidate_images(folder_paths, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    current_index = 0  
    

    for folder_path in folder_paths:
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        
        # 遍历每个图片文件
        for image_file in image_files:
            src_path = os.path.join(folder_path, image_file)
            dst_path = os.path.join(output_folder, f"{current_index}.png")
            
            shutil.copy(src_path, dst_path)
            current_index += 1 
            
    print(f"所有图片已整合到 {output_folder}，总计 {current_index} 张图片。")

images_path = []
for item in data_dir:
    images_path.append(os.path.join(data_base, item, "images"))

images_path = images_path
output_path = "/home/cy/Gra_design/dataset/spine_phantom/all_sweeps/images"
consolidate_images(images_path, output_path)
