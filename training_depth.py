#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

# In[2]:


import os
import sys
import logging
import time
import shutil
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from model.depth_network import DepthNetwork
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import dataloading as dl
import model as mdl
import open3d
import open3d as o3d
from open3d import *
from train_depth import GenerateRays
from model.render_utils import vis_pc
import trimesh
from matplotlib import pyplot as plt
from render import render_mesh
from model.render_utils import normalize_mesh
import pyrender
import numpy as np
from train_depth import Loss
import torch

logger_py = logging.getLogger(__name__)

# Fix seeds
np.random.seed(42)
torch.manual_seed(42)

cfg = dl.load_config("configs/DTU/scan0106.yaml", 'configs/default.yaml')
is_cuda = True
device = torch.device("cuda" if is_cuda else "cpu")

# params
out_dir = cfg['training']['out_dir']
backup_every = cfg['training']['backup_every']
batch_size = cfg['training']['batch_size']
n_workers = cfg['dataloading']['n_workers']
lr = 0.001

# In[3]:


gr = GenerateRays("armadillo.obj")
gr.generate()
data = gr.generate_data(1024)
pc = trimesh.points.PointCloud(vertices=data["points"])

model_cfg = cfg['model']
model = DepthNetwork(model_cfg)
model.cuda()

# In[4]:


# model.load_state_dict(torch.load("model.pth"))


# In[5]:


from train_depth import Loss

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = Loss()

# In[ ]:


from tqdm import tqdm

losses = []
for i in range(1000000):
    optimizer.zero_grad()
    data = gr.generate_data(4096 * 2)
    out = model(data["origins"], data["vectors"])

    loss = loss_func.forward(out.view(1, -1), data["depths"])

    #     loss = torch.abs(out.view(1, -1)[0, 0:512] - data["depths"][0, 0:512]).mean() \
    #     + torch.abs(out.view(1, -1)[0, 512:] - data["depths"][0, 512:]).mean()

    depths = []
    new_vectors = []
    close_camera_centers = []

    for j in range(3):
        depth, new_vector, close_camera_center, _ = gr.generate_depths_from_other_views(out,
                                                                                        data["origins"],
                                                                                        data["vectors"],
                                                                                        data["index"],
                                                                                        closest_index=j)
        depths.append(depth)
        new_vectors.append(new_vector)
        close_camera_centers.append(close_camera_center)

    depths = torch.cat(depths, 1)
    close_camera_centers = torch.cat(close_camera_centers, 1)
    new_vectors = torch.cat(new_vectors, 1)

    loss_consistency = torch.abs(model(close_camera_centers, new_vectors) - 1.0 / depths).mean()
    loss = loss + loss_consistency

    loss.backward()
    optimizer.step()
    losses.append(loss.item() ** 2)

    if i % 50 == 0:
        print(" {} Loss :".format(i), loss.item())

# In[7]:


# torch.save(model.state_dict(), "model_wo_mv_consistency.pth")


# In[8]:


pred_points = []
for i in range(100):
    with torch.no_grad():
        data = gr.generate_data(1024, set_type="test")
        out = model(data["origins"], data["vectors"])
        indices = out[0, :, 0] >= 0.3
        #         indices = np.arange(512)
        points = data["origins"] + 1 / out.reshape(-1, 1) * data["vectors"]
        points = points[0][indices].data.cpu().numpy()
        pred_points.append(points)
        print(indices.sum(), data["depths"][0, 0:512].min())


#         color_gt = np.ones(data["points"].shape)
#         color_gt[:, 0:2] = 0
#         color_pred = np.ones(points.shape)
#         color_pred[:, 1:3] = 0
#         points = transform_points(points, gr.camera_poses[data["index"]])
#         gt_points = transform_points(data["points"], gr.camera_poses[data["index"]])
#         pc = vis_pc(points, colors=color_pred, show=False)
#         pc_gt = vis_pc(gt_points, colors=color_gt, show=False)
#         scene = pc.scene()
#         scene.add_geometry(pc_gt)
#         scene.show("gl")


# In[9]:


def transform_points(points, trans):
    points = np.concatenate([points, np.ones((points.shape[0], 1))], -1)
    points = np.linalg.inv(trans) @ points.T
    return points.T[:, 0:3]


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')

# In[11]:


from train_depth import smoothplot

# In[12]:


plt.plot(np.arange(len(losses)), np.log(smoothplot(losses, 1000)))
# plt.plot(np.log(smoothplot(l2losses)), "r")


# In[13]:


vis_pc(np.concatenate(pred_points))

