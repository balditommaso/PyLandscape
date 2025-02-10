import yaml
import os
import pytorch_lightning as pl
from typing import List, Union, Dict, Any
from torch.types import _device
from torch.utils.data import DataLoader



def save_on_file(path: str, res: Union[float, Dict[str, float]]) -> None:
    '''
    Store on text file the results
    '''
    directory, base_name = os.path.split(path)
    file_name, extension = os.path.splitext(base_name)
    # create the directory if it does not exist
    dir_name = os.path.dirname(directory)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    version = 1
    while os.path.exists(path):
        # Create a new file path with version number
        path = os.path.join(directory, f"{file_name}_v{version}{extension}")
        version += 1
        
    with open(path, "w+") as f:
        f.write(str(res))
        f.close()


def evaluate_model(
        trainer: pl.Trainer, 
        model: pl.LightningModule, 
        dataloader: DataLoader, 
        save_path: str = None) -> None:
    '''
    Evaluate the model and if required store the results.
    '''
    
    trainer.test_loop._results.clear()
    test_results = trainer.test(model, dataloaders=dataloader)

    if save_path is not None:
        save_on_file(save_path, test_results)


def yaml_load(config):
    '''
    Read a YAML/YML file
    '''
    with open(config) as stream:
        param = yaml.safe_load(stream)
    return param
    
    
def load_models(
        model: pl.LightningModule, 
        dir_path: str, 
        tag: str, 
        num_models: int = -1,
        device: _device = 'cpu') -> List[pl.LightningModule]:
    '''
    Load models starting with a certain tag
    '''
    files = os.listdir(dir_path)
    models_ckpt = sorted([file for file in files if file.startswith(tag) and file.endswith('.ckpt')])
    models = []
    for ckpt in models_ckpt:
        model_dict = model.load_from_checkpoint(os.path.join(dir_path, ckpt), map_location=device)
        models.append(model_dict)
    assert len(models) > 0, "No models found!"
    assert num_models <= len(models), "Not enough models available!"
    num_models = len(models) if num_models < 1 else num_models
    return models[:num_models]


def make_iterable(value: Any):
    if isinstance(value, list):
        return value
    return [value]


