import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


where = "test_stats"
experiments = ['go_to', 'obstacle_avoidance']
seeds = range(10)
agents = range(5, 13)
episodes = range(8)


def load_results(experiment, seed, agent):
    return pd.read_csv(f"{where}/{experiment}/seed_{seed}/agents_{agent}/result.csv")


def plot_boxplot(data, ylabel, title, filename):
    plt.figure(figsize=(4, 4))
    for agent, df in data.items():
        plt.boxplot(df, positions=[agent], showfliers=False, widths=0.6)
    plt.ylabel(ylabel)
    plt.xlabel("Agents")
    plt.title(title)
    print("Handling", filename)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def generate_boxplots():
    for experiment in experiments:
        for seed in seeds:
            agents = range(6, 13)
            agents_data = {agent: load_results(experiment, seed, agent) for agent in agents}

            plot_boxplot(
                {agent: df['Reward'] / agent for agent, df in agents_data.items()},
                "Reward per agent", f"Experiment {experiment}",
                f"charts/{experiment}_seed_{seed}_boxplot.pdf"
            )
            plot_boxplot(
                {agent: df['Distance (end)'] for agent, df in agents_data.items()},
                "Distance End", f"Experiment {experiment}",
                f"charts/{experiment}_seed_{seed}_boxplot_distance_end.pdf"
            )
            if experiment == 'obstacle_avoidance':
                plot_boxplot(
                    {agent: df['Collisions'] for agent, df in agents_data.items()},
                    "Collisions", f"Experiment {experiment}",
                    f"charts/{experiment}_seed_{seed}_boxplot_collisions.pdf"
                )


def load_episode_data(experiment, seed, agent, episode):
    return pd.read_csv(f"{where}/{experiment}/seed_{seed}/agents_{agent}/data/distances_episode_{episode}.csv")


def plot_time_series(data, y_col, title, filename):
    plt.figure(figsize=(5, 4))
    sns.lineplot(data=data, x='Tick', y=y_col, errorbar='sd')
    plt.title(title)
    print("Handling", filename)
    plt.savefig(filename)
    plt.close()


def generate_time_series():
    for experiment in experiments:
        for seed in seeds:
            for agent in agents:
                data = pd.concat([load_episode_data(experiment, seed, agent, ep) for ep in episodes])
                plot_time_series(data, 'Distance', f"Experiment {experiment} Agent {agent}",
                                 f"charts/{experiment}_seed_{seed}_agent_{agent}_distance.pdf")
                plot_time_series(data, 'Hits', f"Experiment {experiment} Agent {agent}",
                                 f"charts/{experiment}_seed_{seed}_agent_{agent}_hits.pdf")


def load_positions(experiment, seed, agent, episode):
    x = pd.read_csv(f"{where}/{experiment}/seed_{seed}/agents_{agent}/positions/positions_episode_{episode}_x.csv")
    y = pd.read_csv(f"{where}/{experiment}/seed_{seed}/agents_{agent}/positions/positions_episode_{episode}_y.csv")
    return pd.concat([x, y], axis=1)


def generate_kde_plots():
    for experiment in experiments:
        for seed in seeds:
            for agent in agents:
                all_x, all_y = [], []
                for episode in episodes:
                    x_y = load_positions(experiment, seed, agent, episode)
                    for i in range(agent):
                        all_x.extend(x_y[f'X{i}'])
                        all_y.extend(x_y[f'Y{i}'])
                data = pd.DataFrame({'x': all_x, 'y': all_y})

                plt.figure(figsize=(3, 3))
                sns.kdeplot(data=data, x='x', y='y', fill=True, cmap='viridis')
                plt.scatter(-0.8, 0.8, color='orange', s=300, marker='*')
                if experiment == 'obstacle_avoidance':
                    plt.scatter(-0.1, 0.1, color='red', s=200)
                plt.title(f"Agents: {agent}")
                plt.tight_layout()
                filename = f"charts/{experiment}_seed_{seed}_agents_{agent}_kde.pdf"
                print("Handling", filename)
                plt.savefig(filename)
                plt.close()


# Run all processes
generate_boxplots()
generate_time_series()
generate_kde_plots()