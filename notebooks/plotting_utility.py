import os
import ast
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

# ---------------------------------------------------------------------------- #
#                                   CONSTANT                                   #
# ---------------------------------------------------------------------------- #
DATA_PATH = '../checkpoint/'

performance_metric = {
    "ECON": "AVG_EMD",
    "FUSION": "ampl_mae",
    "VGG16": "test_top1_acc",
    "MobileNet": "test_top1_acc"
}

precisions = [4, 5, 6, 7, 8, 9, 10, 11, 12]
econ_noise_tags = ["gaussian", "salt_pepper"]
vision_noise_tags = ["gaussian_noise", "impulse_noise"]
vision_noise_module = [1, 2, 3, 4, 5]
flip_strategy = ["random_bit_flip", "fkeras_bit_flip"]

# plot styling
FIG_SIZE = (7, 5)
LINE_WIDTH = 2
LEGEND_SIZE = 14
LABEL_SIZE = 20
TICK_SIZE = 18

labels = {
    "JREG_0.1": "Jacobian (δ=1e-1)",
    "JREG_0.01": "Jacobian (δ=1e-2)",
    "JREG_0.001": "Jacobian (δ=1e-3)",
    "JREG_0.005": "Jacobian (δ=5e-3)",
    "JREG_0.0001": "Jacobian (δ=1e-4)",
    "JREG_0.00001": "Jacobian (δ=1e-5)",
    "JREG_0.000001": "Jacobian (δ=1e-6)",
    "JREG_0.0000001": "Jacobian (δ=1e-7)",

    "LIP_0.1": "Orthogonality (δ=1e-1)",
    "LIP_0.01": "Orthogonality (δ=1e-2)",
    "LIP_0.001": "Orthogonality (δ=1e-3)",
    "LIP_0.0001": "Orthogonality (δ=1e-4)",
    "LIP_0.0005": "Orthogonality (δ=5e-4)",
    "LIP_0.00001": "Orthogonality (δ=1e-5)",
    "LIP_0.000001": "Orthogonality (δ=1e-6)",
    "LIP_0.0000001": "Orthogonality (δ=1e-7)",

    "baseline": "Baseline"
}

# ---------------------------------------------------------------------------- #
#                                   PLOTTING                                   #
# ---------------------------------------------------------------------------- #
def plot_precision_vs_performace(
    values: pd.DataFrame, 
    group_by: str, 
    performance_tag: str = "AVG_EMD",
    std: bool = False, 
    log_scale: bool = False, 
    plot_legend: bool = False,
    ylim: Optional[Tuple[float, float]] = None
) -> None:
    plt.figure(figsize=FIG_SIZE)

    # Group data and plot each group with mean and std shading
    for label, df_group in values.groupby(group_by):
        # Plot mean line for the group
        plt.plot(df_group["precision"], df_group[performance_tag], marker='o', linewidth=LINE_WIDTH, label=label)
        
        # Plot shaded area for standard deviation
        if std:
            plt.fill_between(
                df_group["precision"],
                df_group[performance_tag] - df_group[f"{performance_tag} std"],
                df_group[performance_tag] + df_group[f"{performance_tag} std"],
                alpha=0.2  # Adjust transparency as needed
            )
    if log_scale:
        plt.yscale('log')
    if ylim is not None:
        plt.ylim(ylim)
    
    plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE) 
    plt.xlabel("Precision", fontsize=LABEL_SIZE)
    plt.ylabel(performance_tag.replace("_", " "), fontsize=LABEL_SIZE)

    # plt.title(title, fontsize=16)
    if plot_legend:
        legend = plt.legend(title=group_by, fontsize=LABEL_SIZE-2, ncol=1)
        legend.set_title(group_by, prop={'size': LABEL_SIZE, 'weight': 'bold'}) 
        
    
    plt.grid(True)
    plt.show()


def plot_precision_vs_metrics(
    values: pd.DataFrame, 
    group_by: str, 
    x: str, 
    y: str, 
    y_std: Optional[str] = None, 
    log_scale: bool = False, 
    plot_legend: bool = False,
    ylim: Optional[Tuple[float, float]] = None
) -> None:
    plt.figure(figsize=FIG_SIZE)

    # group data and plot each group with mean and std shading
    for label, df_group in values.groupby(group_by):
        # Plot mean line for the group
        plt.plot(df_group[x], df_group[y], marker='o', linewidth=LINE_WIDTH, label=label)
        
        # plot shaded area for standard deviation
        if y_std is not None:
            plt.fill_between(
                df_group[x],
                df_group[y] - df_group[y_std],
                df_group[y] + df_group[y_std],
                alpha=0.1  
            )

    if log_scale:
        plt.yscale('log')
    if ylim is not None:
        plt.ylim(ylim)
        
    plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)  # Increase major tick label size
    plt.xlabel("Precision", fontsize=LABEL_SIZE)
    plt.ylabel(y, fontsize=LABEL_SIZE)
    
    if plot_legend:
        legend = plt.legend(title=group_by, fontsize=LABEL_SIZE-2)
        legend.set_title(group_by, prop={'size': LABEL_SIZE, 'weight': 'bold'}) 

    plt.grid(True)
    plt.show()
    
# ---------------------------------------------------------------------------- #
#                                    UTILITY                                   #
# ---------------------------------------------------------------------------- #
def load_from_pickle(dir_path: str, file: str):
    full_file_path = os.path.join(dir_path, file)
    # Ensure the file has a .pkl or .pickle extension before loading
    if file.endswith('.pkl') or file.endswith('.pickle'):
        with open(full_file_path, 'rb') as f:
            return pickle.load(f)


def get_results(
    path: str, 
    tag: str = "accuracy", 
    aggregate: str = "mean", 
    key: str = "test_top1_acc",  # Default for vision results
    verbose: bool = False
) -> np.ndarray:
    """
    Generalized function to retrieve and aggregate results from files in a directory.

    Args:
        path (str): Directory containing result files.
        tag (str): Keyword to filter relevant files.
        aggregate (str): Aggregation method (e.g., "mean", "median").
        key (str): Key to extract data from the file.
        verbose (bool): If True, prints messages when errors occur.

    Returns:
        np.ndarray: Aggregated results or NaN if no results are found.
    """
    
    if os.path.exists(path) and os.path.isdir(path):
        files = os.listdir(path)
        result_files = [file for file in files if tag in file]
        results = []

        for file in result_files:
            with open(os.path.join(path, file)) as f:
                file_txt = f.read()
                data = ast.literal_eval(file_txt)
                if isinstance(data, list) and len(data) > 0 and key in data[0]:
                    results.append(data[0][key])
                else:
                    if verbose:
                        print(f"Key '{key}' not found in file: {file}")
    else:
        if verbose:
            print(f"Directory not found!\n\tpath: {path}")
        return np.NaN  

    if not results:
        if verbose:
            print(f"Results not found!\n\tpath: {path}/{result_files}\n\ttag: {tag}")
        return np.NaN

    return getattr(np, aggregate)(results)


def get_metrics_results(dir_path: str, file_tag: str, key: str, aggregate: str = 'mean') -> np.ndarray:
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        files = os.listdir(dir_path)
        result_files = [file for file in files if file_tag in file and file.endswith(".pkl")]
        results = []
        # we have many files with the same tag
        if len(result_files) == 0:
            print(f"Warning: File not found!\n\tpath: {dir_path}")
            return np.NaN
        if len(result_files) > 1:
            for file in result_files:
                data = load_from_pickle(dir_path, file)
                res = data[key]
                if isinstance(res, list) and len(res) > 1:
                    # aggregate the results
                    results.extend(res)
                else:
                    results.append(res)
            # aggregate the results
            return getattr(np, aggregate)(results)
        if len(result_files) == 1:
            # just one file found
            data = load_from_pickle(dir_path, result_files[0])
            res = data[key]
            if isinstance(res, list) and len(res) > 1:
                # aggregate the results
                return getattr(np, aggregate)(res)
        
        print(f"Aggregation not used for {file_tag} - {key}")
        return res
    # error directory not found
    else:
        print(f"Directory not found!\n\tpath: {dir_path}")
        return np.NaN



def create_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

    

def load_metrics(
    model_type: str,
    tags: List[str],
    batch_sizes: List[int] = None,
    learning_rates: List[float] = None,
    cka_samples: int = 5,
    mc_epochs: int = 100,
    mc_bends: int = 3,
    verbose: bool = False
) -> None:
    # Default parameters based on model type
    if isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes]
    if isinstance(learning_rates, float):
        learning_rates = [learning_rates]

    # Determine which function to use
    if model_type not in performance_metric:
        raise ValueError("Not available model!")
    
    # store the results
    records = []
    for p in precisions:
        for x, bs in enumerate(batch_sizes):
            for y, lr in enumerate(learning_rates):
                for tag in tags:
                    # build the path
                    base_path = f"bs{bs}_lr{lr}/"
                    path = os.path.join(DATA_PATH, f"{base_path}{model_type}_{p}b")

                    if tag != "baseline":
                        path = os.path.join(DATA_PATH, f"{base_path}{model_type}_{tag}_{p}b")

                    mc_max = get_metrics_results(path, f"Bezier_bends_{mc_bends}_epochs_{mc_epochs}", "mode_connectivity", "max")
                    mc_min = get_metrics_results(path, f"Bezier_bends_{mc_bends}_epochs_{mc_epochs}", "mode_connectivity", "min")
                    max_dev = mc_max if abs(mc_max) > abs(mc_min) else mc_min
                            
                    records.append({
                        "batch_size": bs,
                        "learning_rate": lr,
                        "precision": p,
                        "regularizer": labels[tag],
                        "CKA": get_metrics_results(path, f"CKA_similarity_{cka_samples}", "CKA_similarity", "mean"),
                        "CKA_median": get_metrics_results(path, f"CKA_similarity_{cka_samples}", "CKA_similarity", "median"),
                        "CKA_std": get_metrics_results(path, f"CKA_similarity_{cka_samples}", "CKA_similarity", "std"),
                        "CKA_max": get_metrics_results(path, f"CKA_similarity_{cka_samples}", "CKA_similarity", "max"),
                        "CKA_min": get_metrics_results(path, f"CKA_similarity_{cka_samples}", "CKA_similarity", "min"),
                        "Hessian trace": get_metrics_results(path, "hessian", "trace", "mean"),
                        "h_trace_max": get_metrics_results(path, "hessian", "trace", "max"),
                        "h_trace_std": get_metrics_results(path, "hessian", "trace", "std"),
                        "Top eigenvalue": get_metrics_results(path, "hessian", "eigenvalue", "mean"),
                        "top_eigen_max": get_metrics_results(path, "hessian", "eigenvalue", "max"),
                        "top_eigen_std": get_metrics_results(path, "hessian", "eigenvalue", "std"),
                        "mc_median": get_metrics_results(path, f"Bezier_bends_{mc_bends}_epochs_{mc_epochs}", "mode_connectivity", "median"),
                        "mc_std": get_metrics_results(path, f"Bezier_bends_{mc_bends}_epochs_{mc_epochs}", "mode_connectivity", "std"),
                        "max mc": max_dev
                    })
                    
    df = pd.DataFrame(records)
    save_path = f"./results/{model_type}/"
    create_dir(save_path)
    df.to_csv(os.path.join(save_path, "metrics.csv"), index=False)
    
    

def load_benchmarks(
    model_type: str,  # "econ", "mobilenet",
    tags: List[str], 
    batch_sizes: List[int] = None,
    learning_rates: List[float] = None,
    noise_modules: List[str] = None,
    num_bits: Optional[List[int]] = None,
    flip_strategies: Optional[List[str]] = None,
    verbose: bool = False
) -> None:
    # Default parameters based on model type
    if isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes]
    if isinstance(learning_rates, float):
        learning_rates = [learning_rates]

    # Determine which function to use
    if model_type not in performance_metric:
        raise ValueError("Not available model!")
    
    result_key = performance_metric[model_type]

    # Define noise tags based on model type
    noise_tags = (
        econ_noise_tags if model_type in ["ECON", "FUSION"] else vision_noise_tags
    )

    # Store results
    records = []
    for p in precisions:
        for bs in batch_sizes:
            for lr in learning_rates:
                for tag in tags:
                    # Construct the base path
                    base_path = f"bs{bs}_lr{lr}/"
                    path = os.path.join(DATA_PATH, f"{base_path}{model_type}_{p}b")

                    if tag != "baseline":
                        path = os.path.join(DATA_PATH, f"{base_path}{model_type}_{tag}_{p}b")

                    # handle different result types
                    if not flip_strategies:
                        file_tag = "noise.csv"
                        records.append({
                            "batch_size": bs,
                            "learning_rate": lr,
                            "precision": p,
                            "regularizer": labels[tag],
                            "noise_type": "clean",
                            result_key: get_results(path, aggregate="mean", key=result_key, verbose=verbose),
                            f"{result_key} std": get_results(path, aggregate="std", key=result_key, verbose=verbose),
                            f"max {result_key}": get_results(path, aggregate="max", key=result_key, verbose=verbose),
                            f"min {result_key}": get_results(path, aggregate="min", key=result_key, verbose=verbose),
                        })

                        # add noisy cases
                        for noise_tag in noise_tags:
                            if noise_modules and model_type in ["ECON", "FUSION"]:
                                for module in noise_modules:
                                    records.append({
                                        "batch_size": bs,
                                        "learning_rate": lr,
                                        "precision": p,
                                        "regularizer": labels[tag],
                                        "noise_type": noise_tag,
                                        "Noise module (%)": module,
                                        result_key: get_results(path, f"{noise_tag}_{module}", aggregate="mean", key=result_key, verbose=verbose),
                                        f"{result_key} std": get_results(path, f"{noise_tag}_{module}", aggregate="std", key=result_key, verbose=verbose),
                                        f"max {result_key}": get_results(path, f"{noise_tag}_{module}", aggregate="max", key=result_key, verbose=verbose),
                                        f"min {result_key}": get_results(path, f"{noise_tag}_{module}", aggregate="min", key=result_key, verbose=verbose),
                                    })
                            else:
                                for severity in vision_noise_module:
                                    records.append({
                                        "batch_size": bs,
                                        "learning_rate": lr,
                                        "precision": p,
                                        "regularizer": labels[tag],
                                        "noise_type": noise_tag,
                                        "Noise Severity": severity,
                                        result_key: get_results(path, f"{noise_tag}_{severity}", aggregate="mean", key=result_key, verbose=verbose),
                                        f"{result_key} std": get_results(path, f"{noise_tag}_{severity}", aggregate="std", key=result_key, verbose=verbose),
                                        f"max {result_key}": get_results(path, f"{noise_tag}_{severity}", aggregate="max", key=result_key, verbose=verbose),
                                        f"min {result_key}": get_results(path, f"{noise_tag}_{severity}", aggregate="min", key=result_key, verbose=verbose),
                                    })
                                    
                    # bit flipping
                    elif num_bits and flip_strategies:
                        file_tag = "bit_flip.csv"
                        records.append({
                            "batch_size": bs,
                            "learning_rate": lr,
                            "precision": p,
                            "regularizer": labels[tag],
                            "flip_strategy": "clean",
                            "# bits flipped": 0,
                            result_key: get_results(path, aggregate="mean", key=result_key),
                            f"{result_key} std": get_results(path, aggregate="std", key=result_key),
                            f"min {result_key}": get_results(path, aggregate="min", key=result_key),
                            f"max {result_key}": get_results(path, aggregate="max", key=result_key),
                        })

                        # Add bit-flip variations
                        for strategy in flip_strategies:
                            for bit in num_bits:
                                records.append({
                                    "batch_size": bs,
                                    "learning_rate": lr,
                                    "precision": p,
                                    "regularizer": labels[tag],
                                    "flip_strategy": strategy,
                                    "# bits flipped": bit,
                                    result_key: get_results(path, f"{strategy}_{bit}", aggregate="mean", key=result_key),
                                    f"{result_key} std": get_results(path, f"{strategy}_{bit}", aggregate="std", key=result_key),
                                    f"min {result_key}": get_results(path, f"{strategy}_{bit}", aggregate="min", key=result_key),
                                    f"max {result_key}": get_results(path, f"{strategy}_{bit}", aggregate="max", key=result_key),
                                })

    # Convert to DataFrame and save
    df = pd.DataFrame(records)
    save_path = f"./results/{model_type}/"
    create_dir(save_path)
    df.to_csv(os.path.join(save_path, file_tag), index=False)
    