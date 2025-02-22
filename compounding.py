#!/usr/bin/env python
# coding: utf-8
import os
import open3d as o3d

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import numpy as np
import imageio
import pprint
import pathlib

import matplotlib.pyplot as plt

import pretrain as run_nerf_ultrasound
from load_us import load_us_data

basedir = './logs'
expname = 'spine_phantom_left2'

config = os.path.join(basedir, expname, 'config.txt')
print('Args:')
print(open(config, 'r').read())
parser = run_nerf_ultrasound.config_parser()
model_no = 'model_012000'

args = parser.parse_args('--config {} --ft_path {}'.format(config, os.path.join(basedir, expname, model_no + ".npy")))
print('loaded args')
model_name = args.datadir.split("/")[-1]
images, poses, i_test = load_us_data(args.datadir)
H, W = images[0].shape

H = int(H)
W = int(W)

images = images.astype(np.float32)
poses = poses.astype(np.float32)

near = 0.
far = args.probe_depth * 0.001

# In[3]:


# Create nerf model
_, render_kwargs_test, start, grad_vars, models = run_nerf_ultrasound.create_nerf(args)
render_kwargs_test["args"] = args
bds_dict = {
    'near': tf.cast(near, tf.float32),
    'far': tf.cast(far, tf.float32),
}
render_kwargs_test.update(bds_dict)

print('Render kwargs:')
pprint.pprint(render_kwargs_test)
sw = args.probe_width * 0.001 / float(W)
sh = args.probe_depth * 0.001 / float(H)

down = 4
render_kwargs_fast = {k: render_kwargs_test[k] for k in render_kwargs_test}

frames = []
impedance_map = []
map_number = 0
output_dir = "{}/{}/output_maps_{}_{}_{}/".format(basedir, expname, model_name, model_no, map_number)
output_dir_params = "{}/params/".format(output_dir)
output_dir_output = "{}/output/".format(output_dir, expname, model_name, model_no)
# os.mkdir(output_dir)
# os.mkdir(output_dir_params)
# os.mkdir(output_dir_output)


def show_colorbar(image, name=None, cmap='rainbow', np_a=False):
    figure = plt.figure()
    if np_a:
        image_out = plt.imshow(image, cmap=cmap)
    else:
        image_out = plt.imshow(image.numpy(), cmap=cmap, vmin=0, vmax=1)
    plt.tick_params(top=False, bottom=False, left=False, right=False,
                    labelleft=False, labelbottom=False)
    m = plt.cm.ScalarMappable(cmap=cmap)
    m.set_clim(0., 1.)
    plt.colorbar(m)
    figure.savefig(name)
    plt.close(figure)
    return image_out
save_it = 300

rendering_params_save = None


# create empty cloud data
point_cloud_tf = []
point_cloud = o3d.geometry.PointCloud()

for i, c2w in enumerate(poses):
    print(i)

    # run_out = output_dir + str(i) + "/"
    # os.mkdir(run_out)
    rendering_params = run_nerf_ultrasound.render_us_compounding(H, W, sw, sh, c2w=c2w[:3, :4], **render_kwargs_fast)
    
    position = rendering_params['position']
    prob_border = tf.expand_dims(rendering_params['prob_border'], axis=-1)
    border_indicator = tf.expand_dims(rendering_params['border_sample'], axis=-1)
    
    cloud_raw = tf.concat([position, prob_border, border_indicator], axis=-1)

    cloud_filtered = cloud_raw[cloud_raw[..., 3] > 0.688]
    cloud_filtered = cloud_filtered[..., 0:3]
    
    # print(cloud_filtered)
    
    sample_ratio = 100
    sample_size = int(cloud_filtered.shape[0] / sample_ratio)
    indices = np.random.choice(cloud_filtered.shape[0], size=sample_size, replace=False)
    # print(indices)
    cloud_filtered_sampled = tf.gather(cloud_filtered, indices)
    
    if i == 0:
        point_cloud_tf = cloud_filtered_sampled
    else:
        point_cloud_tf = tf.concat([point_cloud_tf, cloud_filtered_sampled], axis=0)
    


point_cloud.points = o3d.utility.Vector3dVector(point_cloud_tf)
# point_cloud.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([point_cloud])
    
a = 10
    
# # 保存
# new_folder_path = ply_path = "/home/cy/Gra_design/us_nerf_pro/compounding_result_ply/" + expname
# os.makedirs(new_folder_path, exist_ok=True)
# ply_path = new_folder_path + "/" + model_no + ".ply"
# o3d.io.write_point_cloud(ply_path, point_cloud)
# print(f"Point cloud saved to {ply_path}")