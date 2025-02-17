import os
import pytorch_lightning as pl
import torch
import warnings
from argparse import ArgumentParser
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import datamodule as dm
import models
from models.utils import *


warnings.filterwarnings("ignore")


def argument_parser():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yml", help="Path to the configuration file.")
    parser.add_argument("--saving_folder", type=str, help="Path to the saving directory.")
    parser.add_argument("--precision", type=int, default=8, help="Bit width of the model.")
    parser.add_argument("--lr", type=float, default=0.0015625, help="Batch size.")
    parser.add_argument("--batch_size", type=int, default=512, help="Learning rate.")
    parser.add_argument("--experiment", type=int, default=1, help="Number of the experiment running.")
    parser.add_argument("--pretrained", action="store_true", default=False, help="Start from a pretrained version of the model.")
    parser.add_argument("--no_train", action="store_true", default=False, help="Skip the training part.")
    parser.add_argument("--recover", action="store_true", default=False, help="Skip the training part.")

    # NOTE: use it only for debugging
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    
    args = parser.parse_args()
    
    return args
    

def main():
    args = argument_parser()
    
    # load the configuration file
    config = yaml_load(args.config)
    save_dir = os.path.join(
        args.saving_folder, 
        f"{config['save_dir']}_{args.precision}b/"
    )
    experiment = f"{config['save_dir'].lower()}_{args.experiment}"
    print(f"Saving to dir:\t\t{save_dir}")
    print(f"Running experiment:\t{experiment}")
    
    # device -> NOTE: designed for single GPU train
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")
        torch.cuda.empty_cache()
        
    # ---------------------------------------------------------------------------- #
    #                                  Data Module                                 #
    # ---------------------------------------------------------------------------- #
    data_cfg = config["data"]
    if not hasattr(dm, data_cfg["name"]):
        raise ValueError(f"Not Valid data module: {data_cfg['name']} ")
    # get the instance of the required data module
    data_module = getattr(dm, data_cfg["name"])(batch_size=args.batch_size, **data_cfg)

    # ---------------------------------------------------------------------------- #
    #                                Lightning model                               #
    # ---------------------------------------------------------------------------- #
    # starting from a pretrained model
    model_cfg = config["model"]
    model_args = dict(
        config=args.config,
        quantized=args.precision < 32,
        bit_width=args.precision,
        learning_rate=args.lr
    )
    if not hasattr(models, model_cfg["name"]):
        raise ValueError(f"Not Valid model: {model_cfg['name']} ")
    # get the instance of the required model
    architecture = getattr(models, model_cfg["name"])
    
    # load the data from the full precision version
    if args.pretrained:
        # checkpoint path
        full_precision_ckpt = config["save_dir"].split("_")[0]
        model_ckpt = os.path.join(
            args.saving_folder, 
            f"{full_precision_ckpt}_32b/"
            f"{full_precision_ckpt.lower()}_" \
            f"{args.experiment}_best.ckpt"
        )
        print(f"Loading the model from:\n\t{model_ckpt}")
        if not os.path.exists(model_ckpt):
            raise ValueError("Warning: pretrained version of the model not found!")
        
        # load the weights of the pretrained model
        pl_model = architecture.load_from_checkpoint(
            model_ckpt, 
            map_location=device, 
            **model_args
        ) 
        
    elif args.recover:
        # checkpoint path
        model_ckpt = os.path.join(save_dir, f"{experiment}_best.ckpt")
        print(f"Loading the model from:\n\t{model_ckpt}")
        if not os.path.exists(model_ckpt):
            raise ValueError("Warning: recover version of the model not found!")
        
        # recover the model from the checkpoint
        pl_model = architecture.load_from_checkpoint(
            model_ckpt, 
            map_location=device, 
            **model_args
        ) 
    else:
        pl_model = architecture(**model_args)

    # ---------------------------------------------------------------------------- #
    #                                    Trainer                                   #
    # ---------------------------------------------------------------------------- #
    # tensorboard directory
    tb_logger = pl_loggers.TensorBoardLogger(save_dir, name=f"{experiment}_logs")
    
    # monitor the learning rate and the scheduler
    callbacks = [LearningRateMonitor("epoch")]
    
    # save top checkpoints based on val_loss
    top_checkpoint_callback = ModelCheckpoint(
        **config["fit"]["model_checkpoint"],
        dirpath=save_dir,
        filename=f"{experiment}_best"
    )
    callbacks.append(top_checkpoint_callback)
            
    # stop training when model converges
    if hasattr(config["fit"], "early_stopping"):
        callbacks.append[EarlyStopping(**config["fit"]["early_stopping"])]
    
    trainer = pl.Trainer(
        **config["fit"]["trainer"],
        devices="auto",
        logger=tb_logger,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run    
    )
    
    # ---------------------------------------------------------------------------- #
    #                                  Train model                                 #
    # ---------------------------------------------------------------------------- #
    if not args.no_train:
        trainer.fit(
            model=pl_model, 
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader()
        )        
        
        pl_model = getattr(models, model_cfg["name"]).load_from_checkpoint(
            top_checkpoint_callback.best_model_path,
            strict=True
        )
    
    # ---------------------------------------------------------------------------- #
    #                                Evaluate model                                #
    # ---------------------------------------------------------------------------- #
    test_results_log = os.path.join(save_dir, f"accuracy_{args.experiment}.txt")
    
    # preprocessing for the ECON model
    if isinstance(data_module, dm.AutoEncoderDataModule):
        _, val_sum = data_module.get_val_max_and_sum()
        pl_model.set_val_sum(val_sum)
    
    evaluate_model(trainer, pl_model, data_module.test_dataloader(), test_results_log)



if __name__ == "__main__":
    main()