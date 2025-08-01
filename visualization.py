"""
These are functions for visualizing data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import shap
#%% plot SOH and RUL predictions comparison and errors 
color_palette = {
    "With P*": "#551F33", 
    "Without P*": "#CBBBC1"
}

color_palette = {
    "With Physical Features": "#551F33", 
    "Without Physical Features": "#CBBBC1"
}

def plot_grouped_violin(data, ylabel):
    if data.empty:
        print(f"⚠️ Data is empty, cannot plot {ylabel}.")
        return  # 跳过空数据

    fig, ax = plt.subplots(figsize=(5, 3))  

    sns.violinplot(
        x="train_ratio", 
        y="value", 
        hue="physical_features",  
        data=data,
        palette=color_palette, 
        scale="width",
        linewidth=2,
        inner="box", 
        dodge=True  
    )

    legend_patches = [
        mpatches.Patch(color=color_palette["With Physical Features"], label="With P*"),
        mpatches.Patch(color=color_palette["Without Physical Features"], label="Without P*")
    ]
    
    if ylabel in ["SOH R²", "RUL R²"]:
        ax.legend(
            handles=legend_patches,
            fontsize=12,
            loc="lower right",
            bbox_to_anchor=(1, 0),
            ncol=2,
            frameon=False
        )
    else:
        ax.legend(
            handles=legend_patches,
            fontsize=12,
            loc="upper right",  
            bbox_to_anchor=(1, 1),
            ncol=2,
            frameon=False
        )

    ax.set_xlabel("Train Ratio", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    plt.xlabel("Train Ratio", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)   

    plt.tight_layout()
    plt.show()
#%% add split cycle judgement and opt plot
def find_elbow_point_stable_soft(mae_means, cycles, threshold=0.003, stable_window=2, max_fluctuation=0.006):
    """
    A more robust way to identify the convergence turning point
    """
    import numpy as np
    mae_means = np.array(mae_means)

    for i in range(len(mae_means) - stable_window):
        window = mae_means[i:i + stable_window + 1]
        if np.max(window) - np.min(window) <= max_fluctuation:
            return cycles[i + stable_window]
    return cycles[-1]  

# find the smallest one
def find_elbow_point_min_value(mae_means, cycles):
    """
    Find the cycle corresponding to the minimum MAE and use it as the dividing point
    """
    import numpy as np
    min_index = np.argmin(mae_means)
    return cycles[min_index]

def plot_with_error_bars_and_trend_with_split(df, metric_filter, ylabel, title,
                                              threshold=0.003, stable_window=2, max_fluctuation=0.006):
    plt.figure(figsize=(8, 4.5))
    x_labels = sorted(df["cycle"].unique())
    bar_width = 0.35
    x_indices = np.arange(len(x_labels))

    palette = {
        "With Physical Features": "#1f77b4",
        "Without Physical Features": "#ff7f0e"
    }

    label_map = {
        "With Physical Features": "With P* Features",
        "Without Physical Features": "Without P* Features",
        "With Physical Features Trend": "With P* Trend",
        "Without Physical Features Trend": "Without P* Trend"
    }

    for idx, physical_feature in enumerate(df["physical_features"].unique()):
        filtered_df = df[(df["metric"] == metric_filter) & (df["physical_features"] == physical_feature)]

        if filtered_df.empty:
            continue

        filtered_df = filtered_df.sort_values(by="cycle")
        mean_values = filtered_df["error_mean"].values
        std_values = filtered_df["error_std"].values
        cycles = filtered_df["cycle"].values      

        if "MAPE" in metric_filter:
            mean_values *= 100
            std_values *= 100
            ylabel = metric_filter.replace("MAE", "MAPE") + " (%)"
        elif "SOH MAE" in metric_filter:
            mean_values *= 100
            std_values *= 100
            ylabel = "SOH MAE [%]"

        offset = -bar_width / 2 if idx == 0 else bar_width / 2
        color = palette[physical_feature]

        plt.bar(
            x_indices + offset, mean_values, bar_width,
            label=label_map[physical_feature], yerr=std_values, capsize=5,
            alpha=0.8, color=color
        )

        plt.plot(
            x_indices + offset, mean_values,
            marker="o", linestyle="--", linewidth=2,
            label=label_map[physical_feature + " Trend"], color=color
        )

        elbow = find_elbow_point_min_value(mean_values, cycles)
        
        elbow_index = list(cycles).index(elbow)
        split_x = x_indices[elbow_index] + offset
        
        text_y = max(mean_values) + 0.05 * max(std_values) * (1.5 - idx)

        plt.axvline(x=split_x, linestyle=":", color=color, linewidth=2)
        plt.text(
            split_x + 0.1, text_y+ 0.001 * (idx + 1),
            f"Split at cycle {elbow}", fontsize=14, color=color,
            verticalalignment='bottom', 
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8) 
        )

    plt.xticks(x_indices, x_labels, rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Time of cycle", fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    legend_loc = "upper right" if "SOH" in metric_filter else "lower right"
    plt.legend(fontsize=15, loc=legend_loc)

    plt.grid(False)
    plt.tight_layout() 
    plt.show()

def plot_heatmap(shap_df, title="SHAP Heatmap"):
    plt.figure(figsize=(12, 7))
    ax = sns.heatmap(
        shap_df,
        annot=False,
        cmap="YlGnBu",
        cbar_kws={"label": "Mean |SHAP|", "shrink": 0.95}
    )
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Random Seed", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=11)
    colorbar.set_label("Mean |SHAP|", fontsize=12)

    plt.tight_layout()      
    plt.show()

# === SHAP Summary Plot===
def plot_shap_beeswarm(
    shap_values,
    X_df,
    max_display=12,
    figsize=(7, 3.5),
    cmap="coolwarm",
    title=None,
    feature_names=None,
    font_size=11,
    label_fontsize=11,
    value_fontsize=10,
    title_fontsize=12,
):

    plt.figure(figsize=figsize)

    if feature_names is not None:
        X_df.columns = feature_names

    shap.summary_plot(
        shap_values,
        X_df,
        plot_type="dot",
        max_display=max_display,
        cmap=cmap,
        show=False
    )
    
    plt.gcf().set_size_inches(figsize)

    if title:
        plt.title(title, fontsize=title_fontsize, pad=8)

    plt.xticks(fontsize=value_fontsize)
    plt.yticks(fontsize=label_fontsize)
    plt.tight_layout(pad=0.5)
    plt.show()
#%% radar plot
model_order = ["Linear Regression", "SVR", "XGBoost", "MLP", "Random Forest"]
setting_map = {
    "NoPhysParam_NoPhysFeat": "NoPhysParam_NoP*",
    "NoPhysParam_WithPhysFeat": "NoPhysParam_WithP*",
    "WithPhysParam_WithPhysFeat": "WithPhysParam_WithP*"
}

def prepare_radar_data(df, metric_name="Test SOH MAE"):
    df_filtered = df[df["metric"] == metric_name].copy()
    df_filtered["setting"] = df_filtered["setting"].map(setting_map)
    df_filtered["model"] = pd.Categorical(df_filtered["model"], categories=model_order, ordered=True)
    radar_df = df_filtered.pivot(index="setting", columns="model", values="mean")
    return radar_df

def plot_radar_custom(
    data_df,
    color_list=None,
    legend_labels=None,
    fill_alpha=0.1,
    line_alpha=0.9,
    label_fontsize=9,
    value_fontsize=8,
    legend_fontsize=8,  
    title_text=None,
    title_fontsize=10,
    save_fig=False,
    save_dir="."
):
    labels = data_df.columns.tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    if color_list is None:
        color_list = ['#1f77b4', '#ff7f0e', '#2ca02c']
    if legend_labels is None:
        legend_labels = data_df.index.tolist()

    linestyles = ['-', '--', ':']
    markers = ['o', 's', 'd']

    fig, ax = plt.subplots(figsize=(5, 4), subplot_kw=dict(polar=True))

    for i, (setting, row) in enumerate(data_df.iterrows()):
        values = row.tolist() + [row.tolist()[0]]
        ax.plot(
            angles, values,
            color=color_list[i % len(color_list)],
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            label=legend_labels[i],
            linewidth=1.5,
            alpha=line_alpha
        )
        ax.fill(
            angles, values,
            color=color_list[i % len(color_list)],
            alpha=fill_alpha
        )

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=label_fontsize)
    ax.tick_params(axis='y', labelsize=value_fontsize)
    ax.grid(True)
    if title_text:
        ax.set_title(title_text, fontsize=title_fontsize, pad=5)

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.06),
        fontsize=legend_fontsize,
        ncol=1,
        frameon=False
    )

    plt.subplots_adjust(bottom=0.25)     
    plt.show()
    
def plot_voltage_mae_heatmap(
    error_dict,
    metric_key="RUL_MAE",
    title="RUL MAE Without P*",
    cmap="plasma"
):
    """
    Plot a heatmap of MAE under different voltage ranges

    parameters：
    - error_dict: Voltage range to error dictionary, formatted as: {(v_max, v_min): {"SOH_MAE": ..., "RUL_MAE": ...}, ...}
    - metric_key: The name of the metric to be plotted, e.g., "RUL_MAE" or "SOH_MAE"
    - title: str
    - cmap: Colormap (default: 'plasma')
    """

    x_vals, y_vals, metric_vals = [], [], []
    for (v_max, v_min), errors in error_dict.items():
        x_vals.append(v_min)
        y_vals.append(v_max)
        metric_vals.append(errors[metric_key])

    if len(metric_vals) == 0:
        print(f"⚠️ No available data found for '{metric_key}', plot not generated.")
        return

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    metric_vals = np.array(metric_vals)

    # Point size is scaled based on normalized error
    sizes = (metric_vals - metric_vals.min()) / (metric_vals.max() - metric_vals.min()) * 300 + 50

    # plot
    plt.figure(figsize=(6, 4))
    sc = plt.scatter(
        x_vals, y_vals,
        c=metric_vals,
        s=sizes,
        cmap=cmap,
        alpha=0.9,
        edgecolors=None
    )

    # add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label(metric_key, fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    # set axis labels and title
    plt.xlabel("Lower Voltage [V]", fontsize=15)
    plt.ylabel("Upper Voltage [V]", fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(False)
    plt.tight_layout()
    plt.show()