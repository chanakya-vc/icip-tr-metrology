import os

import numpy as np

root_dir = r"\\wsl.localhost\Ubuntu-24.04\home\vc\research\SEM-Dataset\Dataset-generation\baseline\data"
cross_train_datasets = [r"4_29_energy1000_sigma5_dose500_h[30_50]_a[50_90]_w[20_50]_SS",
                        r"4_29_energy1000_sigma5_dose500_h[30_50]_a[50_90]_w[20_50]_SS"]

cross_test_datasets= [r"5_19_mis_Si_PMMA_energy950_sigma5_dose500_h[30_50]_a[50_90]_w[20_50]_SS",
                     r"5_19_mis_Si_PMMA_energy1000_sigma4-5_dose500_h[30_50]_a[50_90]_w[20_50]_SS"]

datasets=[r"4_29_energy1000_sigma5_dose500_h[30_50]_a[50_90]_w[20_50]_SS",
          r"5_18_energy1000_sigma3_dose500_h[30_50]_a[50_90]_w[20_50]_SS",
          r"5_14_energy1000_sigma1_dose500_h[30_50]_a[50_90]_w[20_50]_SS",
          r"5_20_energy1000_sigma5_dose100_h[30_50]_a[50_90]_w[20_50]_SS",
          r"5_14_energy500_sigma5_dose500_h[30_50]_a[50_90]_w[20_50]_SS"]


save_root = r"..\..\results_new"



validation_frac = 0.2
pixel_1_idx = 20
pixel_2_idx = 50


resume = False
ckpt="weight_best.pt"

lr = 3e-3
batch_size = 64
weight_decay = 1e-4
hidden_dim = 128
epochs = 125
save_interval = 20
# Change this for datasets with different image sizes, make is 101 for everything else except dataset5
pixel_ranges= np.linspace(0,101,100,dtype='int').tolist()

# [alpha_min, h_min, w_min]
target_mins = [50.0, 30.0, 20.0]

# [alpha_max, h_max, w_max]
target_maxs = [90.0, 50.0, 50.0]
min_lr= 1e-6
dose = 500
# model="qm_sam_var"
model="conv_eta"
# model="qm"




