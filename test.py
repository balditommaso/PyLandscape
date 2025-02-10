import os
import torch
import warnings
import torch.utils
import models as archs
import datamodule as dm
from argparse import ArgumentParser
from itertools import combinations
from torch.utils.data import DataLoader
from pylandscape import * 
from models.utils import *
from benchmarks.bit_flip import BitFlip


warnings.filterwarnings("ignore")



def argument_parser():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yml")
    parser.add_argument("--saving_folder", type=str)
    parser.add_argument("--precision", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_models", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=None)    

    return parser.parse_args()


def main():
    args = argument_parser()
    # load the config file
    config = yaml_load(args.config)
    save_dir = os.path.join(
        args.saving_folder, 
        f"{config['save_dir']}_{args.precision}b/"
    )
    experiment = f"{config['save_dir'].lower()}"
    
    # set the seed for the tests
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)
        
    # detect the device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision('high')
        torch.cuda.empty_cache()
        
    # ---------------------------------------------------------------------------- #
    #                                  Data Module                                 #
    # ---------------------------------------------------------------------------- #
    data_cfg = config['data']
    if not hasattr(dm, data_cfg['name']):
        raise ValueError(f"Not Valid data module: {data_cfg['name']} ")
    # get the instance of the required data module
    data_module = getattr(dm, data_cfg['name'])(batch_size=args.batch_size, **data_cfg)
    
    # ---------------------------------------------------------------------------- #
    #                                     Model                                    #
    # ---------------------------------------------------------------------------- #
    # starting from a pretrained model
    model_cfg = config['model']

    if not hasattr(archs, model_cfg['name']):
        raise ValueError(f"Not Valid model: {model_cfg['name']} ")
    # get the instance of the required model
    architecture = getattr(archs, model_cfg['name'])
    models = load_models(architecture, save_dir, experiment, args.num_models, device)
    print(f"Number of model tested:\t{len(models)}")

    # post processing required for the ECON-T model
    if isinstance(data_module, dm.AutoEncoderDataModule):
        _, val_sum = data_module.get_val_max_and_sum()
        for model in models:
            model.set_val_sum(val_sum)
        
    # trainer used for evaluation under noise/attacks
    trainer = pl.Trainer(**config['fit']['trainer'], logger=False)
    
    # ---------------------------------------------------------------------------- #
    #                                  BENCHMARKS                                  #
    # ---------------------------------------------------------------------------- #        
    # ------------------------------------ noise --------------------------------- #
    if 'noise' in config['test']:
        noise_cfg = config['test']['noise'] 
        for type in noise_cfg['type']:
            for percentage in noise_cfg['percentage']:
                # get the data_module with noisy data
                noisy_data_module = getattr(dm, data_cfg['name'])(
                    batch_size=args.batch_size,
                    processed_data=f"{type}_{percentage}",
                    noise_type=type,
                    noise_module=percentage,
                    **config['data']
                )
                if isinstance(noisy_data_module, dm.AutoEncoderDataModule):
                    _, val_sum = noisy_data_module.get_val_max_and_sum()
                for idx, model in enumerate(models, 1):
                    #required pre-processing for ECON-T model
                    if isinstance(model, archs.AutoEncoder):
                        model.set_val_sum(val_sum)
                    # evaluate the performances
                    test_path = os.path.join(
                        save_dir, 
                        f"{type}_{percentage}_{idx}.txt"
                    )
                    evaluate_model(
                        trainer, 
                        model.to(device), 
                        noisy_data_module.test_dataloader(), 
                        test_path
                    )

    # ---------------------------------- bit-flip -------------------------------- #
    if 'bit_flip' in config['test']:
        cfg = config['test']['bit_flip']
        for idx, model in enumerate(models, 1):
            bit_flip = BitFlip(model, data_module.test_dataloader(), device)
            for strategy in cfg['strategy']:
                for n_bits in cfg['n_bits']:
                    perturbed_model = bit_flip.attack(n_bits, strategy)
                    test_path = os.path.join(
                        save_dir, 
                        f"{strategy}_bit_flip_{n_bits}_{idx}.txt"
                    )
                    evaluate_model(
                        trainer, 
                        perturbed_model.to(device), 
                        data_module.test_dataloader(), 
                        test_path
                    )    
    
    # ---------------------------------------------------------------------------- #
    #                                    METRICS                                   #
    # ---------------------------------------------------------------------------- #
    # ----------------------------------- hessian -------------------------------- #
    if "hessian" in config["test"]:
        cfg = config["test"]["hessian"]
        n_iter = cfg["n_iter"]
        top_n = cfg["top_n"]
        for idx, model in enumerate(models, 1):
            h = Hessian(
                model=model, 
                criterion=model.criterion, 
                dataloader=data_module.test_dataloader(), 
                name=f"hessian_{n_iter}_{idx}"
            )
            h.compute_trace(n_iter)
            h.compute_eigenvalues(n_iter, top_n=top_n)

            h.save_on_file(save_dir)
    
    # ------------------------------------ cka ----------------------------------- #
    if 'cka' in config['test']:
        assert len(models) > 1, "Attention: you need more then one model!"
        
        score = []
        cfg = config['test']['cka']
        num_outputs = make_iterable(cfg['num_outputs'])
        # get the dataloader
        dataloader = data_module.test_dataloader()
        # add noise to the data if necessary
        if 'noise_type' in cfg and 'noise_module' in cfg and cfg['noise_module'] > 0:
            # get the data_module with noisy data
            noisy_data_module = getattr(dm, data_cfg['name'])(
                processed_data=f"{cfg['noise_type']}_{cfg['noise_module']}",
                **config['data'], 
                **cfg
            ) 
            dataloader = DataLoader(
                noisy_data_module.test_dataset, 
                batch_size=cfg['batch_size'],
                shuffle=True,
                num_workers=config['data']['num_workers']
            )
        for m in num_outputs:    
            # test the distance among all the combinations of models
            for model1, model2 in combinations(models, 2):
                model1.eval()
                model2.eval()
                cka = CKA(device=device)
                score.append(
                    cka.output_similarity(
                        model1, 
                        model2, 
                        dataloader,
                        num_outputs=m,
                        num_runs=cfg["num_runs"]
                    )
                )
                del cka
                
            # store the result
            name = f"CKA_similarity_{m}"
            if cfg and 'noise_module' in cfg and cfg['noise_module'] > 0:
                name += f"_{cfg['noise_module']}"
            cka = CKA(name=name)
            cka.results['CKA_similarity'] = score
            cka.save_on_file(save_dir)
  
    # ---------------------------- mode connectivity ----------------------------- #
    if 'mode_connectivity' in config['test']:
        assert len(models) > 1, "Attention: you need more then one model!"
        
        score = []
        cfg = config['test']['mode_connectivity']
        curve = cfg['curve']
        num_bends = make_iterable(cfg['num_bends'])
        epochs = make_iterable(cfg['max_epochs'])
        num_points = cfg['num_points']
        for n_bend in num_bends:
            for max_epochs in epochs:
                # test the distance among all the combinations of models
                for model1, model2 in combinations(models, 2):
                    model1.eval()
                    model2.eval()
                    mc = ModeConnectivity(device)
                    score.append(
                        mc.compute(
                            model1, 
                            model2, 
                            criterion=model1.criterion,
                            train_dataloader=data_module.train_dataloader(),
                            test_dataloader=data_module.test_dataloader(),
                            learning_rate=args.lr,
                            curve=curve,
                            num_bends=n_bend,
                            num_points=num_points,
                            max_epochs=max_epochs,
                            init_linear=True,
                            device=device
                        )
                    )
                    del mc
                # store the result
                mc = ModeConnectivity(name=f"{curve}_bends_{n_bend}_epochs_{max_epochs}")
                mc.results["mode_connectivity"] = score
                mc.save_on_file(save_dir)
        
    # ------------------------------------ plot ---------------------------------- #
    if 'plot' in config['test']:
        cfg = config['test']['plot']
        lams = (cfg["min_lam"], cfg["max_lam"])
        steps = cfg["steps"]
        for idx, model in enumerate(models, 1):
            plot = Surface(
                model, 
                model.criterion, 
                data_module.test_dataloader(), 
                device, seed=args.seed, 
                name=f"plot_{steps}_{idx}"
            )
            
            plot.random_line(lams, steps)
            plot.hessian_line(lams, steps)
            plot.random_surface(lams, steps)
            plot.hessian_surface(lams, steps)
  
            plot.save_on_file(save_dir)
        
        
    print("\nTest completed!")



if __name__ == "__main__":
    main()
    