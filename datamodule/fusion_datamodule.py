import os
import sys
import h5py
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch import tensor
from torchvision import datasets, transforms
from torch.utils.data import Subset, random_split, DataLoader
from typing import Optional, Callable, List, Tuple, Any
from benchmarks.noise import Noise


# preprocessing
Fusion_image_norm = 4095.0  # 12 bits image

Fusion_image_mean = [0.066927]
Fusion_image_std = [0.030739]
Fusion_label_mean = [0, 0]
Fusion_label_std = [20, 20]



class FusionData(datasets.VisionDataset):
    
    H5FILE = 'db-220629-v0.h5'
    
    def __init__(self, 
            root: str = None, 
            train: bool = True,
            standardize: bool = True,
            resize: bool = True,
            cache: bool = True,
            noise_adder: Optional[Callable] = None,
            noise_module: float = 0.0,
            target_transform: Optional[Callable] = None) -> None:
        
        self.root = root
        self.data = None
        self.targets = None
        self.noise_adder = noise_adder
        self.noise_module = noise_module
        
        data_path = os.path.join(root, f"data_{int(train)}.npy")
        targets_path = os.path.join(root, f"targets_{int(train)}.npy")
        # check if dataset already processed
        if cache and os.path.exists(data_path):
            # get the data from the cache
            self.data = np.load(data_path)
            self.targets = np.load(targets_path)
        else:
            # load the dataset
            metadata = self.fetch_metadata()
            shotno_list = metadata.loc[(metadata['shot_style'].isin([1]) & 
                                    (metadata['group'].isin([0])))]['shotno'].to_numpy()
            # divide the data in train and test set
            shotno_list_train = shotno_list
            shotno_list_test = [114464, 114467, 114468, 114472, 114473]
            # remove the test pictures from the train list
            for shotno in shotno_list_test:
                shotno_list_train = np.delete(
                        shotno_list_train,  # list
                        np.where(shotno_list_train == shotno)   # indices
                    )
                
            if train:
                self.data = self.batch_fetch_tiff_sliding_window(shotno_list_train)
                labels = self.batch_fetch_label_sliding_window(shotno_list_train)
            else:
                self.data = self.batch_fetch_tiff_sliding_window(shotno_list_test)
                labels = self.batch_fetch_label_sliding_window(shotno_list_test)
            # convert to components in sin & cos phase
            sin_comp, cos_comp = self.ampl_phase_to_comp(labels[:,0], labels[:,1])
            self.targets = np.stack((sin_comp, cos_comp), axis=-1, dtype=np.float32)
            self.data = self.data.astype(np.float32)    # NumPy default is float64
            # normalize the labels
            self.targets = self.normalize_label(self.targets, Fusion_label_mean, Fusion_label_std)
            # save in cache
            if cache:
                np.save(data_path, self.data)
                np.save(targets_path, self.targets)
            
        
        # convert targets to tensor
        self.targets = torch.tensor(self.targets)
        
        mean, std = [0.], [1.]
        if standardize:
            mean, std = Fusion_image_mean, Fusion_image_std
        
        # add normalization step if required
        transf_list = [
            transforms.ToTensor(),  # NOTE: not divided by 255.
            transforms.Lambda(lambda x: x / Fusion_image_norm), # Normalize to [0, 1]
            transforms.Normalize(mean, std) # normalize if needed
        ]
        # resize to 32 x 32 if needed
        if resize:
            transf_list.append(transforms.Resize((32, 32))) # resize if needed

        transform = transforms.Compose(transf_list)
        
        super().__init__(root, transform=transform, target_transform=target_transform)

    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        if self.noise_adder is not None:
            img = self.noise_adder(img, percentage=self.noise_module)
            img = torch.clamp(img, 0, 1,)
            img = torch.tensor(img, dtype=torch.float32)

        return img, target
    
    
    def __len__(self) -> int:
        return len(self.data)
    
    
    def compute_data_stats(self) -> Tuple[float, float]:
        """
        Computes the mean and standard deviation of the dataset.

        Returns:
            Tuple[float, float]: The mean and standard deviation of the dataset.
        """
        # Convert data to a torch tensor
        data_tensor = torch.tensor(self.data, dtype=torch.float32)

        # Normalize the data
        data_tensor = data_tensor / Fusion_image_norm  # Assuming Fusion_image_norm is predefined

        # Compute mean and standard deviation
        mean = data_tensor.mean(dim=[0, 1, 2])  # Mean across batch, height, and width
        std = data_tensor.std(dim=[0, 1, 2])    # Std across batch, height, and width

        return mean.tolist(), std.tolist()
        
        
    def fetch_metadata(self) -> np.array:
        '''
        Return metadata
        '''
        file = os.path.join(self.root, self.H5FILE)
        metadata = pd.read_hdf(file, key='metadata')
        
        return metadata
    
    
    @staticmethod
    def ampl_phase_to_comp(ampl: np.array, phase: np.array) -> Tuple[np.array, np.array]:
        '''
        Convert ampl & phase (-pi, pi) to sin & cos components
        - sin_comp = ampl * sin(phase)
        - cos_comp = ampl * cos(phase)
        Returns sin_comp, cos_comp
        '''
        sin_comp = ampl * np.sin(phase)
        cos_comp = ampl * np.cos(phase)
    
        return sin_comp, cos_comp
    
    
    @staticmethod
    def normalize_label(label: np.array, mean: List[float], stdev: List[float]) -> np.array:
        '''
        label[:,i] = (label[:,i] - mean[i])/stdev[i]
        Return label
        '''
        for i in range(len(mean)):
            label[:,i] = (label[:,i] - mean[i]) / stdev[i]
            
        return label
    
    
    @staticmethod
    def comp_to_ampl_phase(sin_comp: tensor, cos_comp: tensor) -> tensor:
        '''
        Convert sin_comp & cos_comp to ampl & phase (-pi, pi)
        Returns ampl, phase
        '''
        ampl = torch.sqrt(sin_comp**2 + cos_comp**2)
        phase = torch.arctan2(sin_comp, cos_comp)
        
        return ampl, phase


    @staticmethod
    def rescale_label(label: tensor, mean: List[float], stdev: List[float]) -> tensor:
        '''
        label[:,i] = label[:,i]*stdev[i] + mean[i]
        Return label
        '''
        for i in range(len(mean)):
            label[:,i] = label[:,i] * stdev[i] + mean[i]
            
        return label
    
    
    def fetch_shot_cam(self, shotno: np.array, cam: int, suffix: str) -> np.array:
        '''
        Return shotno-cam-tiff
        '''
        file = os.path.join(self.root, self.H5FILE)
        h5f = h5py.File(file, 'r')
        if cam == 26730:
            tiff = h5f[str(shotno)+'-26730-tiff'+suffix][:]
        elif cam == 26731:
            tiff = h5f[str(shotno)+'-26731-tiff'+suffix][:]
        h5f.close()    
        
        return tiff
    
    
    def fetch_shot_info(self, shotno: np.array) -> np.array:
        '''
        Return shotno-info dataframe
        '''   
        file = os.path.join(self.root, self.H5FILE)
        shotno_info = pd.read_hdf(file, key=str(shotno)+'-info')
        
        return shotno_info
    
    
    def process_tiff_sliding_window(self, tiff: np.array, timesteps: int) -> np.array:
        '''
        Convert tiff (N, width, height) to tiff_sw (N-timesteps+1, width, height, timesteps)
        The lastest frame of each stack is (_, :, :, -1)
        Return tiff_sw
        '''
        shape = tiff.shape
        tiff_sw = np.zeros((shape[0]-timesteps+1, shape[1], shape[2], timesteps))
        
        for i in range(shape[0]-timesteps+1):
            for j in range(timesteps):
                tiff_sw[i, :, :, j] = tiff[i+j, :, :]
        
        return tiff_sw
    
    
    def batch_fetch_tiff_sliding_window(self,
                                        shotno_list: np.array, 
                                        cam: int = 26730, 
                                        suffix: str = '', 
                                        timesteps: int = 1) -> np.array:
        '''
        Fetch tiff of all shots in shotno_list, convert each shot to a 
        4D array (N-timesteps+1, width, height, timesteps), then concat to a single 
        np array.
        Returns batch_tiff_sw
        '''
        # fetch 1st shot to determine array shape
        shotno_tiff = self.fetch_shot_cam(shotno_list[0], cam, suffix)
        batch_tiff_sw = self.process_tiff_sliding_window(shotno_tiff, timesteps)
        
        # fetch the rest
        for shotno in shotno_list[1:]:
            shotno_tiff = self.fetch_shot_cam(shotno, cam, suffix)
            shotno_tiff_sw = self.process_tiff_sliding_window(shotno_tiff, timesteps)
            batch_tiff_sw = np.concatenate((batch_tiff_sw, shotno_tiff_sw), axis=0) 
        
        return batch_tiff_sw
    
    
    def process_label_sliding_window(self, label: np.array, timesteps: int) -> np.array:
        '''
        Complementary method for fetch labels using sliding window method
        Fetch the label that corresponds to the latest frame of each stack ([_, :, :, -1])
        Returns label_sw (shape=[N-timesteps+1, n_label])
        '''
        label_sw = label[timesteps-1:, :]
        
        return label_sw
        
    
    def batch_fetch_label_sliding_window(self,
            shotno_list: np.array, 
            column: List[str] = ['n1_ampl_10k', 'n1_phase_raw'], 
            timesteps: int = 1) -> np.array:
        '''
        Fetch column of all shots in shotno_list, select [timesteps-1:, :], 
        then concat to a single (N, len(column)) np array.
        
        Returns batch_label_sw
        '''
        # fetch 1st shot to determine array shape
        shotno_info = self.fetch_shot_info(shotno_list[0])
        shotno_label = shotno_info[column].to_numpy()
        batch_label_sw = self.process_label_sliding_window(shotno_label, timesteps)
        
        # fetch the rest
        for shotno in shotno_list[1:]:
            shotno_info = self.fetch_shot_info(shotno)
            shotno_label = self.process_label_sliding_window(shotno_info[column].to_numpy(), timesteps)
            batch_label_sw = np.concatenate((batch_label_sw, shotno_label), axis=0)
        
        return batch_label_sw
            


class FusionDataModule(pl.LightningDataModule):
    
    CAM = 26730
    CAM_SUFFIX = ''
    TIMESTEPS = 1
    LABEL = ['n1_ampl_10k', 'n1_phase_raw']
    
    def __init__(
            self,
            data_path: str,
            batch_size: int = 32, 
            val_size: float = 0.1,
            num_workers: int = 8, 
            seed: int = 42,
            noise_type: Optional[str] = None,
            noise_module: float = 0.0,
            standardize: bool = False,
            cache: bool = True,
            **kwargs) -> None:
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size   
        self.val_size = val_size
        self.num_workers = num_workers
        self.seed = seed
        self.noise_type = noise_type
        self.noise_module = noise_module
        self.standardize = standardize
        self.cache = cache
        
        # add noise 
        Noise.set_seed(self.seed)
        self.noise_adder = None
        if noise_type == 'random':
            self.noise_adder = Noise.add_random_perturbation
        elif noise_type == 'salt_pepper':
            self.noise_adder = Noise.add_salt_and_pepper_noise
        elif noise_type == 'gaussian':
            self.noise_adder = Noise.add_gaussian_noise
        else:
            print('No noise will be added!')
        
        self.setup(0)

    
    def setup(self, stage: str) -> None:
        pl.seed_everything(self.seed, workers=True)
        self.train_dataset = FusionData(root=self.data_path, 
                                        train=True,
                                        standardize=self.standardize,
                                        cache=self.cache,
                                        noise_adder=self.noise_adder,
                                        noise_module=self.noise_module)
        self.test_dataset = FusionData(root=self.data_path, 
                                       train=False,
                                       standardize=self.standardize,
                                       cache=self.cache,
                                       noise_adder=self.noise_adder,
                                       noise_module=self.noise_module)
        
        if self.val_size > 0.0:
            self.val_dataset = FusionData(root=self.data_path, 
                                          train=True,
                                          standardize=self.standardize,
                                          cache=self.cache,
                                          noise_adder=self.noise_adder,
                                          noise_module=self.noise_module)
            train_part, val_part = random_split(self.train_dataset, [1 - self.val_size, self.val_size])
            self.train_dataset = Subset(self.train_dataset, train_part.indices)
            self.val_dataset = Subset(self.val_dataset, val_part.indices)
            
        self.summary()
            
            
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    
    def val_dataloader(self) -> DataLoader:
        if self.val_size <= 0.0:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
    
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        
        
    def raw_dataloader(self) -> DataLoader:
        return self.test_dataloader()
            
            
    def summary(self):
        msg = f"** FUSION dataset: **\n" \
              f"train dataset:\t{len(self.train_dataset)}\n" \
              f"val dataset:\t{len(self.val_dataset if self.val_size > 0.0 else 0)}\n" \
              f"test dataset:\t{len(self.test_dataset)}\n" \
              f"using cache:\t{int(self.cache)}\n" \
              f"standardize:\t{int(self.standardize)}"
        
        if self.noise_type is not None:
            msg += f"\nnoise type:\t{self.noise_type}\n" \
                   f"magnitude:\t{self.noise_module}\n"
                   
        print(msg)
     
     