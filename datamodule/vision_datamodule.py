import os
import numpy as np
import torch
from torch import tensor
import pytorch_lightning as pl
from typing import Optional, Callable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from .utils import extract_data


# ---------------------------------------------------------------------------- #
#                                   CIFAR-10                                   #
# ---------------------------------------------------------------------------- #
C_CIFAR10_extract_path ="./data/VISION/CIFAR-10-C"

CIFAR10_mean = [0.49139968, 0.48215841, 0.44653091]
CIFAR10_std = [0.2023, 0.1994, 0.2010]

class CIFAR10(datasets.CIFAR10):
    def __init__(
            self,
            root: str, 
            download: bool = True,
            train: bool = True, 
            center: bool = True, 
            rescale: bool = True, 
            augment: bool = True
        ) -> None:
        
        mean = CIFAR10_mean if center else [0., 0., 0.]
        std = CIFAR10_std if rescale else [1., 1., 1.]
        
        transf_list = [transforms.ToTensor(), transforms.Normalize(mean, std)]
        if train and augment:
            transf_list.append(transforms.RandomCrop(32, padding=4))
            transf_list.append(transforms.RandomHorizontalFlip())

        transform = transforms.Compose(transf_list)
        super().__init__(root=root, train=train, transform=transform, download=download)


class CIFAR10C(Dataset):
    
    C_CIFAR10_url = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
    C_CIFAR10_tar_path = "CIFAR-10-C.tar"
    
    def __init__(self, root: str, corruption: str) -> None:
        self.root = root
        # normalize the images
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_mean, CIFAR10_std)
            ]) 
        
        self.data, self.labels = self.load_data(corruption)
        
        
    def load_data(self, corruption: str):
        # check if the C-CIFAR-10 is downloaded
        if not os.path.exists(self.root):
            extract_data(self.C_CIFAR10_url, self.C_CIFAR10_tar_path, self.root)
        
        # path to the selected corruption
        data_path = os.path.join(self.root, f"{corruption}.npy")
        if not os.path.exists(data_path):
            raise ValueError(f"Not valid corruption type selected: {corruption}")
        
        images = np.load(data_path)
        labels = np.load(os.path.join(self.root, "labels.npy"))
        
        return images, labels
        
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
   
    
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
            self, 
            data_path: str,
            batch_size: int=1024,
            val_size: float=0.2,
            num_workers: int=8,
            seed: int = 20000605,
            **kwargs
        ) -> None:
        super().__init__()
        
        self.data_path = data_path
        self.batch_size = batch_size
        self.val_size = val_size
        self.num_workers = num_workers
        self.seed = seed
        self.setup(0)
       
        
    @property
    def dataset_mean():
        return CIFAR10_mean
    
    
    @property
    def dataset_std():
        return CIFAR10_std
        
        
    def setup(self, stage: str) -> None:
        torch.manual_seed(self.seed)
        self.train_dataset = CIFAR10(
            root=self.data_path,
            download=True,
            train=True,
            center=True,
            rescale=True,
            augment=True
        )
        self.val_dataset = CIFAR10(
            root=self.data_path,
            download=True,
            train=True,
            center=True,
            rescale=True,
            augment=False
        )
        self.test_dataset = CIFAR10(
            root=self.data_path,
            download=True,
            train=False,
            center=True,
            rescale=True,
            augment=False
        )
        
        # split the dataset
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
            pin_memory=True
        )
    
    
    def val_dataloader(self) -> DataLoader:
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
        
        
    def corrupted_dataloader(self, path: str, corruption: str, num_samples: Optional[int] = None) -> DataLoader: 
        c_dataset = CIFAR10C(path, corruption)
        assert len(c_dataset) >= num_samples, "There are no enough samples!"
        c_dataset, _ = random_split(c_dataset, [num_samples, len(c_dataset) - num_samples])
        print(f"C-CIFAR-10 ({corruption}):\t{len(c_dataset)} sameples")

        return DataLoader(
            c_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        
        
    def raw_dataloader(self, train: bool = False, batch_size: Optional[int] = None) -> DataLoader:
        '''
        Dataloader without pre-processing used to apply adversarial attack
        '''
        raw_dataset = CIFAR10(self.data_path, 
                              download=False, 
                              train=train, 
                              center=False, 
                              rescale=False, 
                              augment=False)
        if batch_size is None:
            batch_size = self.batch_size
            
        return DataLoader(
            raw_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        ) 
        
        
    def summary(self):
        print(f"** CIFAR-10 dataset: **\n" \
              f"train dataset:\t{len(self.train_dataset)}\n" \
              f"val dataset:\t{len(self.val_dataset)}\n" \
              f"test dataset:\t{len(self.test_dataset)}\n")
        
        
# ---------------------------------------------------------------------------- #
#                                  DATA MODULE                                 #
# ---------------------------------------------------------------------------- #
ImageNet1k_mean = [0.485, 0.456, 0.406]
ImageNet1k_std = [0.229, 0.224, 0.225]

class ImageNet1k(datasets.ImageNet):
    def __init__(
        self, 
        root: str,
        train: bool = True, 
        rescale: bool = True,
        center: bool = True,
        rand_aug: Optional[int] = None
    ) -> None:
        split = 'train' if train else 'val'
        mean = ImageNet1k_mean if center else [0., 0., 0.]
        std = ImageNet1k_std if rescale else [1., 1., 1.]
        
        transf_list = []
        if rand_aug is not None:
            transf_list.append(transforms.RandAugment(num_ops=2, magnitude=rand_aug))
        transf_list.append(transforms.ToTensor())
        transf_list.append(transforms.Normalize(mean, std))
        transf_list.append(transforms.Resize(size=(224, 224), antialias=None))
        
        transform = transforms.Compose(transf_list)
        super().__init__(root=root, split=split, transform=transform)


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_path: str,
        batch_size: int = 128,
        val_size: float = 0.2,
        num_workers: int = 8,
        seed: int = 20000605,
        **kwargs
    ) -> None:
        super().__init__()
        
        self.data_path = data_path
        self.batch_size = batch_size
        self.val_size = val_size
        self.num_workers = num_workers
        self.seed = seed
 
        self.setup(0)
        
        
    def setup(self, stage: str) -> None:
        torch.manual_seed(self.seed)
        self.train_dataset = ImageNet1k(
            root=self.data_path,
            train=True,
            rescale=True,
            center=True,
            rand_aug=9
        )
        
        self.val_dataset = ImageNet1k(
            root=self.data_path,
            train=True,
            rescale=True,
            center=True,
            rand_aug=None
        )
        
        self.test_dataset = ImageNet1k(
            root=self.data_path,
            train=False,
            rescale=True,
            center=True,
            rand_aug=None
        )
        
        # we are taking the validation from the training without applying augmentation,
        # the seed is used to be sure to have the same partitioning among all the trained models
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
            pin_memory=True
        )
    
    
    def val_dataloader(self) -> DataLoader:
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
        
        
    def summary(self):
        print(f"** ImageNet dataset: **\n" \
              f"train dataset:\t{len(self.train_dataset)}\n" \
              f"val dataset:\t{len(self.val_dataset)}\n" \
              f"test dataset:\t{len(self.test_dataset)}\n")
        