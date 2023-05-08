#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
import pandas as pd
import pickle
import scipy.io
import seaborn as sns
from timeit import default_timer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torchvision.transforms import GaussianBlur

from utilities import *
from nn_conv import NNConv, NNConv_old

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', '..'))
from foundation_pdes.pytorch_net.util import record_data, to_cpu, to_np_array


# In[ ]:


class KernelNN3(torch.nn.Module):
    def __init__(self, width_node, width_kernel, depth, ker_in, in_width=1, out_width=1):
        super(KernelNN3, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, width_node)

        kernel = DenseNet([ker_in, width_kernel // 2, width_kernel, width_node**2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width_node, width_node, kernel, aggr='mean')

        self.fc2 = torch.nn.Linear(width_node, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(self.depth):
            x = self.conv1(x, edge_index, edge_attr)
            if k != self.depth - 1:
                x = F.relu(x)

        x = self.fc2(x)
        return x

def plot(pred, y, a_ori, resolution, diff_v_limit=0.01, title=""):
    vmin = min(pred.min().item(), y.min().item())
    vmax = max(pred.max().item(), y.max().item())
    fontsize = 15
    add_plot = 0
    if a_ori is not None:
        add_plot += 1
    L2 = myloss(pred.view(1,-1), y.view(1, -1)).item()
    mae = nn.L1Loss()(pred.view(1,-1), y.view(1, -1)).item()
    fig = plt.figure(figsize=(25,4.5))
    plt.subplot(1,3+add_plot,1)
    sns.heatmap(to_np_array(pred).reshape(resolution,resolution), square=True, xticklabels=False, vmin=vmin, vmax=vmax, yticklabels=False)
    plt.title(f"{title}: pred, L2={L2:.6f}", fontsize=fontsize)
    plt.subplot(1,3+add_plot,2)
    sns.heatmap(to_np_array(y).reshape(resolution,resolution), square=True, xticklabels=False, vmin=vmin, vmax=vmax, yticklabels=False)
    plt.title(f"gt, MAE={mae:.6f}", fontsize=fontsize)
    plt.subplot(1,3+add_plot,3)
    sns.heatmap(to_np_array(pred - y).reshape(resolution,resolution), square=True, xticklabels=False, yticklabels=False, vmin=-diff_v_limit, vmax=diff_v_limit, cmap="PiYG")
    plt.title("pred - gt", fontsize=fontsize)
    if add_plot > 0:
        plt.subplot(1,3+add_plot,4)
        sns.heatmap(to_np_array(a_ori).reshape(resolution,resolution), square=True, xticklabels=False, yticklabels=False)
        plt.title("f", fontsize=fontsize)
    plt.show()
    return fig


# In[ ]:


parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--dataset_type', default="poisson1.0-32", type=str,
                    help='dataset type')
parser.add_argument('--epochs', default=1000, type=int,
                    help='Epochs')
parser.add_argument('--lr', default=0.0001, type=float,
                    help='learning rate')
parser.add_argument('--inspect_interval', default=100, type=int,
                    help='inspect interval')
parser.add_argument('--id', default="0", type=str,
                    help='ID')
try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    is_jupyter = True
    args = parser.parse_args([])
    args.dataset_type = "poisson1.0-64"
except:
    args = parser.parse_args()


# In[ ]:


dataset_type = args.dataset_type
resolution = eval(dataset_type.split("-")[1])

TRAIN_PATH = 'data/piececonst_r241_N1024_smooth1.mat'
TEST_PATH = 'data/piececonst_r241_N1024_smooth2.mat'
DATA_PATH = f"../../data/Poisson1.0_{resolution}/files/"

ms = [200]
case = 0
r = 1
m = ms[case]
k = 1

radius_train = 0.2
radius_test = 0.2

batch_size = 1
batch_size2 = 1
width = 64
ker_width = 256
depth = 4
edge_features = 6
node_features = 6

epochs = args.epochs
learning_rate = args.lr
scheduler_step = 50
scheduler_gamma = 0.5
inspect_interval = args.inspect_interval


runtime = np.zeros(2, )
t1 = default_timer()

if dataset_type == "darcy":
    s = int(((241 - 1)/r) + 1)
    n = s**2
    print('resolution', s)
    ntrain = 100
    ntest = 100
    path = 'neurips1_GKN_s'+str(s)+'_ntrain'+str(ntrain)+'_kerwidth'+str(ker_width) + '_m0' + str(m) + "_id_" + args.id
    path_model = 'model/' + path
    path_train_err = 'results/' + path + 'train.txt'
    path_test_err = 'results/' + path + 'test.txt'
    path_runtime = 'results/' + path + 'time.txt'
    path_image = 'results/' + path

    reader = MatReader(TRAIN_PATH)
    train_a = reader.read_field('coeff')[:ntrain,::r,::r].reshape(ntrain,-1)
    train_a_smooth = reader.read_field('Kcoeff')[:ntrain,::r,::r].reshape(ntrain,-1)
    train_a_gradx = reader.read_field('Kcoeff_x')[:ntrain,::r,::r].reshape(ntrain,-1)
    train_a_grady = reader.read_field('Kcoeff_y')[:ntrain,::r,::r].reshape(ntrain,-1)
    train_u = reader.read_field('sol')[:ntrain,::r,::r].reshape(ntrain,-1)

    reader.load_file(TEST_PATH)
    test_a = reader.read_field('coeff')[:ntest,::r,::r].reshape(ntest,-1)
    test_a_smooth = reader.read_field('Kcoeff')[:ntest,::r,::r].reshape(ntest,-1)
    test_a_gradx = reader.read_field('Kcoeff_x')[:ntest,::r,::r].reshape(ntest,-1)
    test_a_grady = reader.read_field('Kcoeff_y')[:ntest,::r,::r].reshape(ntest,-1)
    test_u = reader.read_field('sol')[:ntest,::r,::r].reshape(ntest,-1)
elif dataset_type.startswith("poisson1.0"):
    s = int(((resolution - 1)/r) + 1)
    n = s**2
    print('resolution', s)

    ntrain = 900
    ntest = 100
    path = 'poisson_s'+str(s)+'_dataset_' + dataset_type + '_ntrain'+str(ntrain)+'_kerwidth'+str(ker_width) + '_m0' + str(m)
    path_model = 'model/' + path
    path_train_err = 'results/' + path + 'train.txt'
    path_test_err = 'results/' + path + 'test.txt'
    path_runtime = 'results/' + path + 'time.txt'
    path_image = 'results/' + path

    f_all = np.load(DATA_PATH + "RHS_all.npy")
    sol_all = np.load(DATA_PATH + "SOL_all.npy")

    f_all = np.load(DATA_PATH + "RHS_all.npy")
    sol_all = np.load(DATA_PATH + "SOL_all.npy")
    gblur = GaussianBlur(kernel_size=5, sigma=5)

    all_a = f_all[:,:,-1]
    all_a_smooth = to_np_array(gblur(torch.tensor(all_a.reshape(all_a.shape[0], resolution, resolution))).flatten(start_dim=1))
    all_a_reshape = all_a_smooth.reshape(-1, resolution, resolution)
    all_a_gradx = np.concatenate([
        all_a_reshape[:,1:2] - all_a_reshape[:,0:1],
        (all_a_reshape[:,2:] - all_a_reshape[:,:-2]) / 2,
        all_a_reshape[:,-1:] - all_a_reshape[:,-2:-1],
    ], 1)
    all_a_gradx = all_a_gradx.reshape(-1, n)
    all_a_grady = np.concatenate([
        all_a_reshape[:,:,1:2] - all_a_reshape[:,:,0:1],
        (all_a_reshape[:,:,2:] - all_a_reshape[:,:,:-2]) / 2,
        all_a_reshape[:,:,-1:] - all_a_reshape[:,:,-2:-1],
    ], 2)
    all_a_grady = all_a_grady.reshape(-1, n)
    all_u = sol_all[:,:,0]

    train_a = torch.FloatTensor(all_a[:ntrain])
    train_a_smooth = torch.FloatTensor(all_a_smooth[:ntrain])
    train_a_gradx = torch.FloatTensor(all_a_gradx[:ntrain])
    train_a_grady = torch.FloatTensor(all_a_grady[:ntrain])
    train_u = torch.FloatTensor(all_u[:ntrain])

    test_a = torch.FloatTensor(all_a[ntrain:])
    test_a_smooth = torch.FloatTensor(all_a_smooth[ntrain:])
    test_a_gradx = torch.FloatTensor(all_a_gradx[ntrain:])
    test_a_grady = torch.FloatTensor(all_a_grady[ntrain:])
    test_u = torch.FloatTensor(all_u[ntrain:])

else:
    raise


a_normalizer = GaussianNormalizer(train_a)
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)
as_normalizer = GaussianNormalizer(train_a_smooth)
train_a_smooth = as_normalizer.encode(train_a_smooth)
test_a_smooth = as_normalizer.encode(test_a_smooth)
agx_normalizer = GaussianNormalizer(train_a_gradx)
train_a_gradx = agx_normalizer.encode(train_a_gradx)
test_a_gradx = agx_normalizer.encode(test_a_gradx)
agy_normalizer = GaussianNormalizer(train_a_grady)
train_a_grady = agy_normalizer.encode(train_a_grady)
test_a_grady = agy_normalizer.encode(test_a_grady)

u_normalizer = UnitGaussianNormalizer(train_u)
train_u = u_normalizer.encode(train_u)
# test_u = y_normalizer.encode(test_u)



meshgenerator = RandomMeshGenerator([[0,1],[0,1]],[s,s], sample_size=m)
data_train = []
for j in range(ntrain):
    for i in range(k):
        idx = meshgenerator.sample()
        grid = meshgenerator.get_grid()
        edge_index = meshgenerator.ball_connectivity(radius_train)
        edge_attr = meshgenerator.attributes(theta=train_a[j,:])
        #data_train.append(Data(x=init_point.clone().view(-1,1), y=train_y[j,:], edge_index=edge_index, edge_attr=edge_attr))
        data_train.append(Data(x=torch.cat([grid, train_a[j, idx].reshape(-1, 1),
                                            train_a_smooth[j, idx].reshape(-1, 1), train_a_gradx[j, idx].reshape(-1, 1),
                                            train_a_grady[j, idx].reshape(-1, 1)
                                            ], dim=1),
                               y=train_u[j, idx], edge_index=edge_index, edge_attr=edge_attr, sample_idx=idx
                               ))


meshgenerator = RandomMeshGenerator([[0,1],[0,1]],[s,s], sample_size=m)
data_test = []
for j in range(ntest):
    idx = meshgenerator.sample(is_random=False)
    grid = meshgenerator.get_grid()
    edge_index = meshgenerator.ball_connectivity(radius_test)
    edge_attr = meshgenerator.attributes(theta=test_a[j,:])
    data_test.append(Data(x=torch.cat([grid, test_a[j, idx].reshape(-1, 1),
                                       test_a_smooth[j, idx].reshape(-1, 1), test_a_gradx[j, idx].reshape(-1, 1),
                                       test_a_grady[j, idx].reshape(-1, 1)
                                       ], dim=1),
                          y=test_u[j, idx], edge_index=edge_index, edge_attr=edge_attr, sample_idx=idx
                          ))

train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(data_test, batch_size=batch_size2, shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')

model = KernelNN3(width, ker_width,depth,edge_features,in_width=node_features).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)
u_normalizer.cuda()
ttrain = np.zeros((epochs, ))
ttest = np.zeros((epochs,))
model.train()

# Load model:
filename = f"poisson_s{resolution}_dataset_{dataset_type}_ntrain900_kerwidth256_m0200"
data_record = pickle.load(open(f"model/{filename}", "rb"))
model.load_state_dict(data_record["state_dict"][-1])


# In[ ]:


isplot = True
if isplot:
    pdf = matplotlib.backends.backend_pdf.PdfPages(f"model/analysis_{filename}.pdf")
analysis_record = {}
model.eval()
with torch.no_grad():
    for ii, data in enumerate(test_loader):
        data = data.to(device)
        out = model(data)
        out = u_normalizer.decode(out.view(batch_size2,-1), sample_idx=data.sample_idx.view(batch_size2,-1))
        a_ori = a_normalizer.decode(data.x[:,2].view(1,-1))
        l2_item = myloss(out, data.y.view(batch_size2, -1)).item()
        mae_item = nn.L1Loss()(out, data.y.view(batch_size2, -1)).item()
        record_data(analysis_record, [l2_item, mae_item], ["L2", "MAE"])
        if isplot:
            fig = plot(out, data.y, a_ori=a_ori, resolution=resolution, diff_v_limit=0.0075, title=f"{901+ii}")
            pdf.savefig(fig)
if isplot:
    pdf.close()


# In[ ]:


fontsize = 14
plt.figure(figsize=(8,6))
plt.scatter(analysis_record["MAE"], analysis_record["L2"], s=6)
plt.xlabel("MAE", fontsize=fontsize)
plt.ylabel("L2", fontsize=fontsize)
plt.tick_params(labelsize=fontsize)
plt.title(f"L2 = {np.mean(analysis_record['L2']):.6f}" + r" $\pm$ " + f"{np.std(analysis_record['L2']):.6f},   "
          f"MAE = {np.mean(analysis_record['MAE']):.6f}" + r" $\pm$ " + f"{np.std(analysis_record['MAE']):.6f}")
plt.savefig(f"model/plot_{filename}.pdf", bbox_inches="tight")
plt.show()


# In[ ]:


# 32:
fontsize = 14
plt.figure(figsize=(8,6))
plt.scatter(analysis_record["MAE"], analysis_record["L2"], s=6)
plt.xlabel("MAE", fontsize=fontsize)
plt.ylabel("L2", fontsize=fontsize)
plt.tick_params(labelsize=fontsize)
plt.title(f"L2 = {np.mean(analysis_record['L2']):.6f}" + r" $\pm$ " + f"{np.std(analysis_record['L2']):.6f},   "
          f"MAE = {np.mean(analysis_record['MAE']):.6f}" + r" $\pm$ " + f"{np.std(analysis_record['MAE']):.6f}")
plt.savefig(f"model/plot_{filename}.pdf", bbox_inches="tight")
plt.show()

