import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import preprocess

class CustomDataset(Dataset):
    def __init__(self, df, seed=None):
        self.data = torch.tensor(df.to_numpy(dtype=np.float32), dtype=torch.float32)
        self.columns = df.columns
        self.preproc = preprocess.Preprocessor(in_data_size=195,n_bins=15)
        self.preproc.load_state("preproc_state.pkl")
        if seed is not None:
            torch.manual_seed(seed)

    def __len__(self):
        return len(self.data)
      
    def __getitem__(self, idx):
        sample = self.data[idx].clone()  # clone to avoid in-place modifications

        # Columns to set to zero
        cols_to_zero = ['AFine', 'BFine', 'CFine', 'DFine']
        for col in cols_to_zero:
            if col in self.columns and torch.rand(1) > 0.5:  # 50% chance
                sample[self.columns.get_loc(col)] = 0

        # Add noise to FilterFreq and subtract from FeAmount
        # if 'FilterFreq' in self.columns and 'FeAmount' in self.columns and torch.rand(1) > 0.5:  # 50% chance
        #     noise = torch.normal(mean=0., std=0.1)  # adjust std for the amount of noise
        #     sample[self.columns.get_loc('FilterFreq')] += noise
        #     sample[self.columns.get_loc('FeAmount')] -= noise

        # Add noise to any column with "Decay" in its name
        # for col in self.columns:
        #     if 'Decay' in col:
        #         sample[self.columns.get_loc(col)] += torch.normal(mean=0., std=0.1)  # adjust std for the amount of noise

        # Mixup augmentation
        if torch.rand(1) > 0.7:  # 90% chance of applying Mixup
            # Choose another random sample
            other_idx = torch.randint(len(self.data), size=(1,)).item()
            other_sample = self.data[other_idx]
        
            # Compute a convex combination of the sample and the other sample
            alpha = torch.rand(1).item() * 0.3  # random alpha between 0 and 0.3
            sample = (1 - alpha) * sample + alpha * other_sample

        ## One hot encoding
        out_arr = self.preproc.one_hot_encode(sample.numpy())
        
        sample_onehot = torch.from_numpy(out_arr)

        return sample_onehot
