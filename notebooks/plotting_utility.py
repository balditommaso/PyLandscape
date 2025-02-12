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
batch_sizes = [1024]
learning_rates = [0.0015625]
precisions = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
noise_tags = ["gaussian", "salt_pepper"]
flip_strategy = ["random_bit_flip", "fkeras_bit_flip"]

FIG_SIZE = (7, 5)
LINE_WIDTH = 2
LEGEND_SIZE = 14
LABEL_SIZE = 20
TICK_SIZE = 18


labels = {
    "JREG_0.1": "Jacobian (δ=1e-1)",
    "LIP_0.00001": "Orthogonality (δ=1e-5)",
    "JREG_0.01": "Jacobian (δ=1e-2)",
    "LIP_0.0001": "Orthogonality (δ=1e-4)",
    "JREG_0.001": "Jacobian (δ=1e-3)",
    "LIP_0.000001": "Orthogonality (δ=1e-6)",
    "baseline": "Baseline"
}

# ---------------------------------------------------------------------------- #
#                                   PLOTTING                                   #
# ---------------------------------------------------------------------------- #
def plot_precision_vs_emd(
    values: pd.DataFrame, 
    group_by: str, 
    std: bool = False, 
    log_scale: bool = False, 
    plot_legend: bool = False,
    ylim: Optional[Tuple[float, float]] = None
) -> None:
    plt.figure(figsize=FIG_SIZE)

    # Group data and plot each group with mean and std shading
    for label, df_group in values.groupby(group_by):
        # Plot mean line for the group
        plt.plot(df_group["precision"], df_group["EMD"], marker='o', linewidth=LINE_WIDTH, label=label)
        
        # Plot shaded area for standard deviation
        if std:
            plt.fill_between(
                df_group["precision"],
                df_group["EMD"] - df_group["EMD std"],
                df_group["EMD"] + df_group["EMD std"],
                alpha=0.2  # Adjust transparency as needed
            )
    if log_scale:
        plt.yscale('log')
    if ylim is not None:
        plt.ylim(ylim)
    
    plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE) 
    plt.xlabel("Precision", fontsize=LABEL_SIZE)
    plt.ylabel("EMD", fontsize=LABEL_SIZE)

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


def get_econ_results(
    path: str, 
    tag: str = "accuracy", 
    aggregate: str = "mean",
    verbose: bool = False
) -> np.ndarray:
    if os.path.exists(path) and os.path.isdir(path):
        files = os.listdir(path)
        result_files = [file for file in files if tag in file]
        results = []    
        for file in result_files:
            with open(os.path.join(path, file)) as f:
                file_txt = f.read()
                data = ast.literal_eval(file_txt)
                data = data[0]['AVG_EMD']
                results.append(data)  
    else:
        if verbose:
            print(f"Directory not found!\n\tpath: {path}")
        return np.NaN     
     
    if len(results) == 0:
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


def load_econ_performance(tags: List[str], noise_modules: List[str], verbose: bool = False) -> None:
    # store the results
    records = []
    for p in precisions:
        for x, bs in enumerate(batch_sizes):
            for y, lr in enumerate(learning_rates):
                for tag in tags:
                    # build the path
                    path = os.path.join(DATA_PATH, f"bs{bs}_lr{lr}/ECON_{p}b")
                    if tag != "baseline":
                        path = os.path.join(DATA_PATH, f"bs{bs}_lr{lr}/ECON_{tag}_{p}b")
                    
                    records.append({
                        "batch_size": bs,
                        "learning_rate": lr,
                        "precision": p,
                        "regularizer": labels[tag],
                        "noise_type": "clean",
                        "Noise module (%)": 0,
                        "EMD": get_econ_results(path, aggregate="mean", verbose=verbose),
                        "EMD std": get_econ_results(path, aggregate="std", verbose=verbose),
                        "max EMD": get_econ_results(path, aggregate="max", verbose=verbose),
                        "min EMD": get_econ_results(path, aggregate="min", verbose=verbose),
                    })
                    
                    for noise_tag in noise_tags:
                        for module in noise_modules:
                            records.append({
                                "batch_size": bs,
                                "learning_rate": lr,
                                "precision": p,
                                "regularizer": labels[tag],
                                "noise_type": noise_tag,
                                "Noise module (%)": module,
                                "EMD": get_econ_results(path, f"{noise_tag}_{module}", aggregate="mean", verbose=verbose),
                                "EMD std": get_econ_results(path, f"{noise_tag}_{module}", aggregate="std", verbose=verbose),
                                "max EMD": get_econ_results(path, f"{noise_tag}_{module}", aggregate="max", verbose=verbose),
                                "min EMD": get_econ_results(path, f"{noise_tag}_{module}", aggregate="min", verbose=verbose),
                            })
                    

    df = pd.DataFrame(records)
    create_dir("./results/econ")
    df.to_csv("./results/econ/noise.csv", index=False)
    

def load_econ_bit_flip(tags: List[str], num_bits: List[int]) -> None:
    # store the results
    records = []
    for p in precisions:
        for x, bs in enumerate(batch_sizes):
            for y, lr in enumerate(learning_rates):
                for tag in tags:
                    # build the path
                    path = os.path.join(DATA_PATH, f"bs{bs}_lr{lr}/ECON_{p}b")
                    if tag != "baseline":
                        path = os.path.join(DATA_PATH, f"bs{bs}_lr{lr}/ECON_{tag}_{p}b")
                    
                    records.append({
                        "batch_size": bs,
                        "learning_rate": lr,
                        "precision": p,
                        "regularizer": labels[tag],
                        "flip_strategy": "clean",
                        "# bits flipped": 0,
                        "EMD": get_econ_results(path, aggregate="mean"),
                        "EMD std": get_econ_results(path, aggregate="std"),
                        "min EMD": get_econ_results(path, aggregate="min"),
                        "max EMD": get_econ_results(path, aggregate="max"),
                    })
                    
                    for strategy in flip_strategy:
                        for bit in num_bits:
                            
                            records.append({
                                "batch_size": bs,
                                "learning_rate": lr,
                                "precision": p,
                                "regularizer": labels[tag],
                                "flip_strategy": strategy,
                                "# bits flipped": bit,
                                "EMD": get_econ_results(path, f"{strategy}_{bit}", aggregate="mean"),
                                "EMD std": get_econ_results(path, f"{strategy}_{bit}", aggregate="std"),
                                "min EMD": get_econ_results(path, f"{strategy}_{bit}", aggregate="min"),
                                "max EMD": get_econ_results(path, f"{strategy}_{bit}", aggregate="max"),
                            })
                    
                    
    df = pd.DataFrame(records)
    create_dir("./results/econ")
    df.to_csv("./results/econ/bit_flip.csv", index=False)
    
    
def load_econ_metrics(tags: List[str]) -> None:
    # store the results
    records = []
    for p in precisions:
        for x, bs in enumerate(batch_sizes):
            for y, lr in enumerate(learning_rates):
                for tag in tags:
                    # build the path
                    path = None
                    if tag == "baseline":
                        path = os.path.join(DATA_PATH, f"bs{bs}_lr{lr}/ECON_{p}b")
                    else:
                        path = os.path.join(DATA_PATH, f"bs{bs}_lr{lr}/ECON_{tag}_{p}b")    
                    
                    mc_max = get_metrics_results(path, "Bezier", "mode_connectivity", "max")
                    mc_min = get_metrics_results(path, "Bezier", "mode_connectivity", "min")
                    max_dev = mc_max if abs(mc_max) > abs(mc_min) else mc_min
                            
                    records.append({
                        "batch_size": bs,
                        "learning_rate": lr,
                        "precision": p,
                        "regularizer": labels[tag],
                        "CKA": get_metrics_results(path, "CKA", "CKA_similarity", "mean"),
                        "CKA_median": get_metrics_results(path, "CKA", "CKA_similarity", "median"),
                        "CKA_std": get_metrics_results(path, "CKA", "CKA_similarity", "std"),
                        "CKA_max": get_metrics_results(path, "CKA", "CKA_similarity", "max"),
                        "CKA_min": get_metrics_results(path, "CKA", "CKA_similarity", "min"),
                        "Hessian trace": get_metrics_results(path, "hessian", "trace", "mean"),
                        "h_trace_max": get_metrics_results(path, "hessian", "trace", "max"),
                        "h_trace_std": get_metrics_results(path, "hessian", "trace", "std"),
                        "Top eigenvalue": get_metrics_results(path, "hessian", "eigenvalue", "mean"),
                        "top_eigen_max": get_metrics_results(path, "hessian", "eigenvalue", "max"),
                        "top_eigen_std": get_metrics_results(path, "Bezier", "mode_connectivity", "mean"),
                        "mc_median": get_metrics_results(path, "Bezier", "mode_connectivity", "median"),
                        "mc_std": get_metrics_results(path, "Bezier", "mode_connectivity", "std"),
                        "max mc": max_dev
                    })
                    
                    
    df = pd.DataFrame(records)
    create_dir("./results/econ")
    df.to_csv("./results/econ/metrics.csv", index=False)
    