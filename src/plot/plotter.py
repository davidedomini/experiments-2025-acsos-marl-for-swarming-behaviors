import glob
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

def load_data_from_csv(path, experiemnt):
    files = glob.glob(f'{path}/experiment_{experiemnt}-seed*.csv')
    print(f'For experiment {experiemnt} found {len(files)} files')
    dataframes = []
    for file in files:
        df = pd.read_csv(file)
        dataframes.append(df)
    return dataframes

def compute_mean_variance(dfs):
    stacked = pd.concat(dfs, axis=0).groupby(level=0)
    mean_df = stacked.mean()
    variance_df = stacked.var(ddof=0)
    return mean_df, variance_df

def plot(mean, variance, experiment, metrics, charts_dir):
    colors = sns.color_palette("viridis", n_colors=1)
    plt.figure(figsize=(10, 6))
    
    for metric in metrics:
        sns.lineplot(
            data = mean,
            x = 'Episode',
            y = metric,
            # label = algorithm,
            color = colors[0],
        )
        lower_bound = mean - np.sqrt(variance)
        upper_bound = mean + np.sqrt(variance)
        plt.fill_between(mean['Episode'], lower_bound[metric], upper_bound[metric], color=colors[0], alpha=0.2)
        plt.title(f'Experiment {experiment}')
        plt.xlabel('Episode')
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(f'{charts_dir}/experiment-{experiment}_metric-{metric}.pdf', dpi=300)
        plt.close()

if __name__ == '__main__':
    
    stats_path  = 'stats'
    charts_path = 'charts'
    experiments = ['GoTo', 'Flocking', 'ObstacleAvoidance']

    Path(charts_path).mkdir(parents=True, exist_ok=True)

    matplotlib.rcParams.update({'axes.titlesize': 20})
    matplotlib.rcParams.update({'axes.labelsize': 18})
    matplotlib.rcParams.update({'xtick.labelsize': 15})
    matplotlib.rcParams.update({'ytick.labelsize': 15})
    plt.rcParams.update({"text.usetex": True})
    plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}')

    for experiment in experiments:
        data = load_data_from_csv(stats_path, experiment)
        mean, variance = compute_mean_variance(data)
        plot(mean, variance, experiment, ['Reward', 'Loss'], charts_path)