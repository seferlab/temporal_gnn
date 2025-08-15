import matplotlib.pyplot as plt
import numpy as np

models = ["LSTM", "MTGNN", "StemGNN", "FourierGNN", "ARIMA", "VAR"]
x = np.arange(len(models))

# Forex metrics: (mean, std)
forex_metrics = {
    "MAPE (%)": (
        [1.707, 1.309, 1.850, 1.756, 1.038, 2.582],
        [0.173, 0.111, 0.165, 0.203, 0.0, 0.0],
    ),
    "MAE": (
        [0.017, 0.013, 0.021, 0.019, 0.010, 0.026],
        [0.002, 0.001, 0.002, 0.002, 0.0, 0.0],
    ),
    "RMSE": (
        [0.154, 0.021, 0.223, 0.159, 0.015, 0.076],
        [0.0004, 0.002, 0.001, 0.003, 0.0, 0.0],
    ),
    "A20": ([1.0, 1.0, 0.998, 0.994, 1.0, 0.966], [0.0, 0.0, 0.003, 0.003, 0.0, 0.0]),
}

# Crypto metrics: (mean, std)
crypto_metrics = {
    "MAPE (%)": (
        [10.982, 6.174, 8.050, 9.924, 11.703, 15.521],
        [0.250, 0.187, 0.224, 0.174, 0.0, 0.0],
    ),
    "MAE": (
        [0.104, 0.061, 0.076, 0.095, 0.115, 0.154],
        [0.002, 0.002, 0.002, 0.002, 0.0, 0.0],
    ),
    "RMSE": (
        [0.153, 0.084, 0.108, 0.147, 0.180, 0.210],
        [0.002, 0.002, 0.003, 0.001, 0.0, 0.0],
    ),
    "A20": (
        [0.874, 0.967, 0.937, 0.900, 0.843, 0.718],
        [0.002, 0.002, 0.004, 0.003, 0.0, 0.0],
    ),
}


def plot_compact(metrics_dict, title):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    metric_names = list(metrics_dict.keys())

    for i, ax in enumerate(axs.flat):
        metric = metric_names[i]
        means, stds = metrics_dict[metric]

        # Determine best model index
        if metric == "A20":
            best_idx = np.argmax(means)
        else:
            best_idx = np.argmin(means)

        # Colors: highlight best model
        colors = ["steelblue"] * len(models)
        colors[best_idx] = "darkorange"

        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black")
        ax.set_title(metric, fontsize=16)
        ax.set_xticks(x)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=16)
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{title}.png", dpi=300)


# Plot
plot_compact(forex_metrics, "fxPredictionMetrics")
plot_compact(crypto_metrics, "cryptoPredictionMetrics")
