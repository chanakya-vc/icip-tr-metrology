import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from einops import rearrange


class SEMSimulatorDataset(Dataset):
    """
    Custom Dataset to load SEM simulation parameters and compute SE count targets.
    """

    def __init__(self, csv_file, npy_dir, pixel_1_idx, pixel_2_idx,aggregation_fn=np.mean, transform=None):
        """
        Args:
            csv_file (str): Path to the csv file with h, alpha, w parameters.
            npy_dir (str): Directory with all the {sim_num}.npy simulation files.
            pixel_1_idx (int/tuple): Index for the flat background pixel.
            pixel_2_idx (int/tuple): Index for the edge bloom pixel.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.params_df = pd.read_csv(csv_file)
        self.npy_dir = npy_dir

        # Pixel locations for eta_1 (flat) and eta_2 (edge)
        self.pixel_1_idx = pixel_1_idx
        self.pixel_2_idx = pixel_2_idx
        self.transform = transform
        self.aggregation_fn = aggregation_fn

    def __len__(self):
        return len(self.params_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Read parameters from CSV
        row = self.params_df.iloc[idx]
        sim_num_name = row['sim-num']

        # Extract inputs (a, h, w)
        a = float(row['a'])
        h = float(row['h'])
        w = float(row['w'])

        # Create input tensor [alpha, h]
        # x_inputs = torch.tensor([a, h,w], dtype=torch.float32)
        y_targets = torch.tensor([a, h, w], dtype=torch.float32)
        # 2. Read simulation data from .npy
        npy_path = os.path.join(self.npy_dir, sim_num_name)

        # Assuming the .npy file shape is (N_frames, num_pixels) or similar,
        # where the first axis represents independent noisy shots.
        sim_data = np.load(npy_path)
        # change the data to the correct shape
        sim_data = np.transpose(sim_data, (0, 2, 1))
        sim_data = rearrange(sim_data, '(new_h chunks) w d -> new_h (chunks w)  d', chunks=np.shape(sim_data)[0])


        # 3. Compute sample means (eta_1, eta_2) at the chosen pixels
        # np.mean is taken across the frame axis (axis=0) to compute the expected rate
        eta_1 = self.aggregation_fn(sim_data[0, :, self.pixel_1_idx])
        eta_2 = self.aggregation_fn(sim_data[0, :, self.pixel_2_idx])
        # Create target tensor [eta_1, eta_2]
        x_inputs = torch.tensor([eta_1, eta_2], dtype=torch.float32)

        sample = {'inputs': x_inputs, 'targets': y_targets, 'sim_num': sim_num_name}

        if self.transform:
            sample = self.transform(sample)

        return sample


# --- Multiple-Pixel-window ---
# class SEMSimulatorDataset_multipixel(Dataset):
#     """
#     Custom Dataset to load SEM simulation parameters and compute SE count targets.
#     """
#
#     def __init__(self, csv_file, npy_dir, pixel_ranges,aggregation_fn=np.nanmean,var_fn=None,dose=None, transform=None):
#         """
#         Args:
#             csv_file (str): Path to the csv file with h, alpha, w parameters.
#             npy_dir (str): Directory with all the {sim_num}.npy simulation files.
#             pixel_1_idx (int/tuple): Index for the flat background pixel.
#             pixel_2_idx (int/tuple): Index for the edge bloom pixel.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.params_df = pd.read_csv(csv_file)
#         self.npy_dir = npy_dir
#
#         # Pixel locations for eta_1 (flat) and eta_2 (edge)
#         self.pixel_ranges = pixel_ranges
#         self.transform = transform
#         self.aggregation_fn = aggregation_fn
#         self.dose=dose
#         self.var_flag=var_fn
#     def __len__(self):
#         return len(self.params_df)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         # 1. Read parameters from CSV
#         row = self.params_df.iloc[idx]
#         sim_num_name = row['sim-num']
#
#         # Extract inputs (a, h, w)
#         a = float(row['a'])
#         h = float(row['h'])
#         w = float(row['w'])
#
#         # Create input tensor [alpha, h]
#         # x_inputs = torch.tensor([a, h,w], dtype=torch.float32)
#         y_targets = torch.tensor([a, h, w], dtype=torch.float64)
#         # 2. Read simulation data from .npy
#         npy_path = os.path.join(self.npy_dir, sim_num_name)
#
#         # Assuming the .npy file shape is (N_frames, num_pixels) or similar,
#         # where the first axis represents independent noisy shots.
#         sim_data = np.load(npy_path)
#         # change the data to the correct shape
#         sim_data = np.transpose(sim_data, (0, 2, 1))
#         height = np.shape(sim_data)[0]
#         sim_data = rearrange(sim_data, '(new_h chunks) w d -> new_h (chunks w)  d', chunks=np.shape(sim_data)[0])
#
#
#         # 3. Compute sample means (eta_1, eta_2) at the chosen pixels
#         # np.mean is taken across the frame axis (axis=0) to compute the expected rate
#         # eta_1 = self.aggregation_fn(sim_data[0, :, self.pixel_1_idx])
#         # eta_2 = self.aggregation_fn(sim_data[0, :, self.pixel_2_idx])
#         if self.dose is not None and self.aggregation_fn.__name__ == 'conv_estimator':
#             eta_pixel_window=[self.aggregation_fn(sim_data[0, :, pixels],self.dose*height) for pixels in self.pixel_ranges]
#         elif self.dose is not None and  self.aggregation_fn.__name__ == 'ctml_estimator':
#             eta_pixel_window = self.aggregation_fn(sim_data[0][:,self.pixel_ranges],self.dose*height).tolist()
#         else:
#             eta_pixel_window=[self.aggregation_fn(sim_data[0, :, pixels]) for pixels in self.pixel_ranges]
#
#         if self.var_flag=="sample_var" or self.var_flag is None:
#             variance_array = np.nanvar(sim_data[0][:,self.pixel_ranges], axis=0, ddof=1)
#         elif self.var_flag=="random_var":
#             # variance_array = np.random.normal(loc=eta_pixel_window, scale=5.0)
#             variance_array=np.random.normal(loc=0.0, scale=5.0, size=len(self.pixel_ranges))
#         if self.var_flag is not None:
#             combined_input = np.vstack((eta_pixel_window, variance_array))
#         else:
#             combined_input = eta_pixel_window
#         # Create target tensor [eta_1, eta_2]
#         x_inputs = torch.tensor(combined_input, dtype=torch.float64)
#
#         sample = {'inputs': x_inputs, 'targets': y_targets, 'sim_num': sim_num_name}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample


# --- Multiple-Pixel-window (Full Spatial Dimension) ---
class SEMSimulatorDataset_multipixel(Dataset):
    """
    Custom Dataset to load SEM simulation parameters and compute SE count targets
    across the entire spatial dimension of the data.
    """

    def __init__(self, csv_file, npy_dir, aggregation_fn=np.nanmean, var_fn=None, dose=None, transform=None):
        """
        Args:
            csv_file (str): Path to the csv file with h, alpha, w parameters.
            npy_dir (str): Directory with all the {sim_num}.npy simulation files.
            aggregation_fn (callable): Function to compute the mean/expected rate.
            var_fn (str): Flag for variance computation ('sample_var', 'random_var', or None).
            dose (float): Dose parameter.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.params_df = pd.read_csv(csv_file)
        self.npy_dir = npy_dir
        self.transform = transform
        self.aggregation_fn = aggregation_fn
        self.dose = dose
        self.var_flag = var_fn

    def __len__(self):
        return len(self.params_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Read parameters from CSV
        row = self.params_df.iloc[idx]
        sim_num_name = row['sim-num']

        # Extract targets (a, h, w)
        a = float(row['a'])
        h = float(row['h'])
        w = float(row['w'])
        y_targets = torch.tensor([a, h, w], dtype=torch.float64)

        # 2. Read simulation data from .npy
        npy_path = os.path.join(self.npy_dir, sim_num_name)

        sim_data = np.load(npy_path)
        # change the data to the correct shape
        sim_data = np.transpose(sim_data, (0, 2, 1))
        height = np.shape(sim_data)[0]
        sim_data = rearrange(sim_data, '(new_h chunks) w d -> new_h (chunks w) d', chunks=height)

        # Isolate the data slice. sim_data[0] shape is (num_frames, num_pixels)
        data_slice = sim_data[0]
        num_pixels = data_slice.shape[1]

        # 3. Compute sample means across the entire spatial dimension
        if self.dose is not None and self.aggregation_fn.__name__ == 'conv_estimator':
            eta_pixel_window = [self.aggregation_fn(data_slice[:, p], self.dose * height) for p in range(num_pixels)]
        elif self.dose is not None and self.aggregation_fn.__name__ == 'ctml_estimator':
            eta_pixel_window = self.aggregation_fn(data_slice, self.dose * height).tolist()
        else:
            # Optimized vectorization if using standard np.nanmean
            if self.aggregation_fn == np.nanmean:
                eta_pixel_window = np.nanmean(data_slice, axis=0).tolist()
            else:
                eta_pixel_window = [self.aggregation_fn(data_slice[:, p]) for p in range(num_pixels)]

        # 4. Compute variance across the entire spatial dimension
        if self.var_flag == "sample_var" or self.var_flag is None:
            variance_array = np.nanvar(data_slice, axis=0, ddof=1)
        elif self.var_flag == "random_var":
            variance_array = np.random.normal(loc=0.0, scale=5.0, size=num_pixels)

        # 5. Combine and create input tensor
        if self.var_flag is not None:
            combined_input = np.vstack((eta_pixel_window, variance_array))
        else:
            combined_input = eta_pixel_window

        x_inputs = torch.tensor(combined_input, dtype=torch.float64)

        sample = {'inputs': x_inputs, 'targets': y_targets, 'sim_num': sim_num_name}

        if self.transform:
            sample = self.transform(sample)

        return sample