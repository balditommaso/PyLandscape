import os
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset, random_split
from .telescope_loss import normalize
from benchmarks.noise import Noise



ARRANGE = torch.tensor(
    [
        28, 29, 30, 31, 0,  4,  8,  12,
        24, 25, 26, 27, 1,  5,  9,  13,
        20, 21, 22, 23, 2,  6,  10, 14,
        16, 17, 18, 19, 3,  7,  11, 15,
        47, 43, 39, 35, 35, 34, 33, 32,
        46, 42, 38, 34, 39, 38, 37, 36,
        45, 41, 37, 33, 43, 42, 41, 40,
        44, 40, 36, 32, 47, 46, 45, 44,
    ]
)

ARRANGE_MASK = torch.tensor(
    [
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 0, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0,
    ]
)


class AutoEncoderDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            data_path: str, 
            processed_data: Optional[str] = None,
            batch_size: int = 500, 
            train_size: float = 0.8,
            val_size: float = 0.1,
            test_size: float = 0.1,
            num_workers: int = 8, 
            noise_type: Optional[str] = None,
            noise_module: float = 0.0,
            seed: int = 42,
            **kwargs) -> None:
        super().__init__()
        self.raw_data = data_path
        self.processed_data = processed_data
        self.batch_size = batch_size   
        self.num_workers = num_workers
        self.seed = seed
        self.noise_type = noise_type
        self.noise_module = noise_module
        self.calq_cols = [f"CALQ_{i}" for i in range(48)]
        
        # test the split of the dataset
        self.train_size = train_size
        self.validation_size = val_size
        self.test_size = test_size
        
        # pre/post processing metrics
        self.max_data = None
        self.sum_data = None
        self.val_max = None
        self.val_sum = None
        self.train_data = None
        self.val_data = None
        
        # add noise 
        self.noise_adder = None
        if noise_type == 'random':
            self.noise_adder = Noise.add_random_perturbation
        elif noise_type == 'salt_pepper':
            self.noise_adder = Noise.add_salt_and_pepper_noise
        elif noise_type == 'gaussian':
            self.noise_adder = Noise.add_gaussian_noise
        else:
            print('No noise will be added!')
            
        if self.processed_data is None:
            self.processed_data = f"{self.raw_data}processed.npy"
        else:
            self.processed_data = f"{self.raw_data}{self.processed_data}.npy"
            
        self.setup(0)
        self.summary()


    def mask_data(self, data):
        """
        Mask rows where occupancy is zero
        """
        return data[data[self.calq_cols].astype("float32").sum(axis=1) != 0]


    def load_data_dir(self):
        """
        Read and concat all csv files in the data directory into a single
        dataframe
        """
        files = os.listdir(self.raw_data)
        csv_files = [file for file in files if file.endswith('.csv')]
        data = pd.concat(
            [
                pd.read_csv(os.path.join(self.raw_data, file), encoding='utf-8', engine='python')
                for file in csv_files
            ]
        )
        data = self.mask_data(data)
        data = data[self.calq_cols].astype("float32")

        return data


    def prep_input(self, norm_data, shape=(1, 8, 8)):
        """
        Prepare the input data for the model
        """
        input_data = norm_data[:, ARRANGE]
        input_data[:, ARRANGE_MASK == 0] = 0  # zero out repeated entries
        shaped_data = input_data.reshape(len(input_data), shape[0], shape[1], shape[2])
        return shaped_data


    def get_val_max_and_sum(self):
        if self.max_data is None or self.sum_data is None:
            _, self.max_data, self.sum_data = self.process_data(save=False)
            
        max_data = self.max_data / 35.0  # normalize to units of transverse MIPs
        sum_data = self.sum_data / 35.0  # normalize to units of transverse MIPs

        self.val_max = max_data[self.test_indices]
        self.val_sum = sum_data[self.test_indices]
        return self.val_max, self.val_sum


    def process_data(self, save=True):
        """
        Only need to run once to prepare the data and pickle it
        """
        # load data 
        data = self.load_data_dir()
        # normalize
        norm_data, max_data, sum_data = normalize(data.values.copy())
        # reshape
        shaped_data = self.prep_input(norm_data)
        # save on file if required
        if save:
            np.save(self.processed_data, shaped_data)
        return shaped_data, max_data, sum_data


    def setup(self, stage):
        """
        Load data from provided npy data_file
        """
        # process the data if not done yet
        if not os.path.exists(self.processed_data):
            shaped_data, self.max_data, self.sum_data = self.process_data()
        else:
            shaped_data = np.load(self.processed_data)
        
        data = torch.tensor(shaped_data, dtype=torch.float32)
        dataset = None
        if self.noise_adder is not None:
            # add noise to the data
            Noise.set_seed(self.seed)
            noise_data = torch.tensor(
                self.noise_adder(shaped_data, percentage=self.noise_module)
                )
            noise_data = torch.clamp(noise_data, 0 , noise_data.max())
            noise_data = torch.tensor(noise_data, dtype=torch.float32)
            dataset = TensorDataset(noise_data, data)
        else:
            dataset = TensorDataset(data, data)
        
        # calculate the number of samples for each set
        self.train_dataset, self.val_dataset, self.test_dataset = \
                                        random_split(dataset, 
                                        [self.train_size, self.validation_size, self.test_size],
                                        torch.Generator().manual_seed(self.seed))
        self.test_indices = self.test_dataset.indices
        

    def train_dataloader(self):
        """
        Return the training dataloader
        """
        return DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=self.num_workers,
                pin_memory=True
            )


    def val_dataloader(self):
        """
        Return the validation dataloader
        """
        return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=self.num_workers,
                drop_last=True,
                pin_memory=True
            )


    def test_dataloader(self):
        """
        Return the test dataloader
        """
        return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=self.num_workers,
                drop_last=True,
                pin_memory=True
            )
        
        
    def summary(self):
        msg = f"** ECON-T dataset: **\n" \
              f"train dataset:\t{len(self.train_dataset)}\n" \
              f"val dataset:\t{len(self.val_dataset)}\n" \
              f"test dataset:\t{len(self.test_dataset)}\n"
              
        if self.noise_type is not None:
            msg += f"noise type:\t{self.noise_type}\n" \
                   f"noise magnitude:\t{self.noise_module}\n"
        print(msg)
