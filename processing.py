import subprocess
import sys

try:
    import pandas as pd
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd

try:
    from scipy.stats import entropy
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    from scipy.stats import entropy

try:
    import powerlaw
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "powerlaw"])
    import powerlaw

try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

try:
    import numpy as np
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy as np



modularity_class_col = "modularity_class"
degree_col = "Degree"
betweenness_col = "betweenesscentrality"
closeness_col = "closnesscentrality"

# Flags
SPLIT = False
COLOR = False

EVENESS = False
DEGREE = True
BETWEENNESS = False
CLOSENESS = False

# Show everything
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Load the node data with modularity classes
df = pd.read_csv("facebook_pa.csv")
if "d0" in df.columns:
    page_types = df["d0"].unique()

# Check what columns you have
print(df.columns)

if EVENESS:
    # Group by Modularity Class, then get distribution of d0 in each group
    grouped = df.groupby(modularity_class_col)["d0"].value_counts().unstack(fill_value=0)

    # Add a percentage column per page_type
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
    percentages = percentages.round(2)  # Optional: round to 2 decimals

    # Combine counts and percentages into one table (multi-index columns)
    combined = pd.concat([grouped, percentages], axis=1, keys=["Count", "Percent"])
    print(combined)


    # Function to compute normalized entropy
    def normalized_entropy(counts):
        probs = counts / counts.sum()
        ent = entropy(probs, base=2)  # base-2 for bits
        max_ent = np.log2(len(probs)) if len(probs) > 1 else 1
        return ent / max_ent if max_ent != 0 else 0


    # Compute spread (entropy) per modularity class
    spread_results = []

    for mod_class, group in df.groupby(modularity_class_col):
        d0_counts = group['d0'].value_counts()
        ent = normalized_entropy(d0_counts)
        spread_results.append({
            modularity_class_col: mod_class,
            "Entropy": ent,
            "Total Nodes": len(group),
            "Num Page Types": d0_counts.count()
        })

    # Create DataFrame
    spread_df = pd.DataFrame(spread_results).sort_values(by="Entropy", ascending=False)

    # Calculate global evenness (weighted average entropy)
    total_nodes_graph = spread_df["Total Nodes"].sum()
    spread_df["Weight"] = spread_df["Total Nodes"] / total_nodes_graph
    spread_df["Weighted Entropy"] = spread_df["Weight"] * spread_df["Entropy"]

    global_evenness = spread_df["Weighted Entropy"].sum()

    print(spread_df[[modularity_class_col, "Entropy", "Total Nodes", "Weight", "Weighted Entropy"]])
    print(f"\n🌐 Global Graph Evenness (Weighted Entropy): {global_evenness:.4f}")


def normalize_column(df, entry):
    min_val = df[entry].min()
    max_val = df[entry].max()
    norm_col = (df[entry] - min_val) / (max_val - min_val + 1e-9)
    norm_name = entry + "_normalized"
    df[norm_name] = norm_col
    return norm_name


def get_dominant_d0_colors(df, entry_for_plot):
    dominant_d0 = df.groupby(entry_for_plot)["d0"].agg(lambda x: x.mode().iloc[0])
    color_map = {d0: c for d0, c in zip(dominant_d0.unique(), plt.cm.tab10.colors)}
    return dominant_d0, color_map


def plot_log_distribution(df, entry_for_plot, entry_label, log, split=False, color=False):
    x_min = max(df[entry_for_plot].min(), 1e-3) if log else df[entry_for_plot].min()
    x_max = df[entry_for_plot].max()

    if split:
        for d0_value in df["d0"].unique():
            plt.figure()
            subset = df[df["d0"] == d0_value]
            value_counts = subset[entry_for_plot].value_counts().sort_index()

            if log:
                plt.loglog(value_counts.index, value_counts.values, marker='o', linestyle='None')
            else:
                plt.plot(value_counts.index, value_counts.values, marker='o', linestyle='None')

            plt.xlim(x_min, x_max)
            plt.xlabel(entry_label)
            plt.ylabel("Frequency")
            plt.title(f"{entry_for_plot} Distribution for Page Type: {d0_value}")
            # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            plt.show()

    elif color:
        dominant_d0, d0_colors = get_dominant_d0_colors(df, entry_for_plot)
        value_counts = df[entry_for_plot].value_counts().sort_index()

        plt.figure()
        for val, freq in value_counts.items():
            d0 = dominant_d0.get(val, None)
            color_val = d0_colors.get(d0, "gray")
            if log:
                plt.loglog(val, freq, marker='o', linestyle='None', color=color_val, label=d0)
            else:
                plt.plot(val, freq, marker='o', linestyle='None', color=color_val, label=d0)

        plt.xlim(x_min, x_max)
        plt.xlabel(entry_label)
        plt.ylabel("Frequency")
        plt.title(f"{entry_for_plot} Distribution")
        # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), title="Page Type (d0)")
        plt.tight_layout()
        plt.show()

    else:
        value_counts = df[entry_for_plot].value_counts().sort_index()

        plt.figure()
        if log:
            plt.loglog(value_counts.index, value_counts.values, marker='o', linestyle='None')
        else:
            plt.plot(value_counts.index, value_counts.values, marker='o', linestyle='None')

        plt.xlim(x_min, x_max)
        plt.xlabel(entry_label)
        plt.ylabel("Frequency")
        plt.title(f"{entry_for_plot} Distribution")
        # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()


def plot_powerlaw_fit(values, entry):
    filtered = values[values > 0]
    fit = powerlaw.Fit(filtered, verbose=False,
                       xmin=min(values),
                       # xmax=max(values)
                       )
    plt.figure()
    fit.plot_pdf(color='b', label='Empirical')
    fit.power_law.plot_pdf(color='r', linestyle='--', label='Power Law Fit')
    plt.legend()
    plt.title(f"{entry} Power Law Fit (Raw Values)")
    # plt.grid(True, which='both', linestyle='', linewidth=0.5)
    print(fit.data)
    print(filtered)
    plt.tight_layout()

    print(f"Power Law α = {fit.power_law.alpha}, xmin = {fit.power_law.xmin:.4f}")
    R, p = fit.distribution_compare('power_law', 'lognormal')
    print(f"Log-likelihood ratio (R): {R:.4f}, p-value: {p:.4f}")
    if p < 0.05:
        better = "Power Law" if R > 0 else "Lognormal"
        print(f"✅ {better} is the better fit.")
    else:
        print("❓ No significant difference between Power Law and Lognormal.")

    plt.show()


def process_by_data(entry, log=False, normalize=False, check_powerlaw=False):
    raw_values = df[entry].astype(float)
    df[entry] = raw_values  # Ensure it's clean

    entry_for_plot = entry
    if normalize:
        entry_for_plot = normalize_column(df, entry)

    entry_label = entry + (" (Normalized)" if normalize else "")

    # Plot log/log distribution
    # if log:
    plot_log_distribution(df, entry_for_plot, entry_label, log=log, split=SPLIT, color=COLOR)

    # Plot power law on raw values
    if check_powerlaw:
        plot_powerlaw_fit(raw_values, entry)


if DEGREE:
    process_by_data(degree_col, log=True, normalize=False, check_powerlaw=True)

if BETWEENNESS:
    process_by_data(betweenness_col, log=True, normalize=True, check_powerlaw=True)

if CLOSENESS:
    process_by_data(closeness_col, log=True, normalize=True, check_powerlaw=True)
