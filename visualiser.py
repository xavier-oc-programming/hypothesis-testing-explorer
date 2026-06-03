import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats as scipy_stats


def plot_distributions(
    group1: list[float],
    group2: list[float],
    group1_name: str,
    group2_name: str,
    test_result: dict,
    output_path: str,
) -> str:
    """
    Generate a distribution comparison plot for two groups.

    Two-panel figure:
    Left panel: overlapping KDE plots + histograms for both groups.
      - Group 1: blue, Group 2: orange
      - Vertical dashed lines at group means
      - Legend showing mean values
    Right panel: box plots side by side showing median, IQR, outliers.

    Add annotation showing the verdict in large text at the top of the figure.
    Add p-value and test name as subtitle.

    Save to output_path as PNG (dpi=150).
    Return output_path.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    g1 = np.array(group1, dtype=float)
    g2 = np.array(group2, dtype=float)
    mean1 = float(np.mean(g1))
    mean2 = float(np.mean(g2))

    significant = test_result.get("significant", False)
    p_value = test_result.get("p_value", None)
    test_name = test_result.get("test_name", "")

    verdict_text = "SIGNIFICANT DIFFERENCE" if significant else "NO SIGNIFICANT DIFFERENCE"
    verdict_color = "#16a34a" if significant else "#dc2626"

    axes[0].hist(g1, bins=30, alpha=0.35, color="#3b82f6", density=True, label=group1_name)
    axes[0].hist(g2, bins=30, alpha=0.35, color="#f97316", density=True, label=group2_name)

    if len(g1) > 1:
        kde1 = scipy_stats.gaussian_kde(g1)
        x1 = np.linspace(g1.min(), g1.max(), 300)
        axes[0].plot(x1, kde1(x1), color="#2563eb", linewidth=2)

    if len(g2) > 1:
        kde2 = scipy_stats.gaussian_kde(g2)
        x2 = np.linspace(g2.min(), g2.max(), 300)
        axes[0].plot(x2, kde2(x2), color="#ea580c", linewidth=2)

    axes[0].axvline(mean1, color="#2563eb", linestyle="--", linewidth=1.5,
                    label=f"{group1_name} mean = {mean1:.2f}")
    axes[0].axvline(mean2, color="#ea580c", linestyle="--", linewidth=1.5,
                    label=f"{group2_name} mean = {mean2:.2f}")

    axes[0].set_title("Distribution Comparison", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=9)

    plot_data = [g1, g2]
    plot_labels = [group1_name, group2_name]
    bp = axes[1].boxplot(plot_data, tick_labels=plot_labels, patch_artist=True,
                          medianprops=dict(color="white", linewidth=2))

    bp["boxes"][0].set_facecolor("#3b82f6")
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("#f97316")
    bp["boxes"][1].set_alpha(0.7)

    axes[1].set_title("Box Plot Comparison", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Value")

    test_label = test_name.replace("_", " ").title()
    p_text = f"p = {p_value:.4f}" if p_value is not None else ""
    fig.suptitle(
        f"{verdict_text}   |   {test_label}   {p_text}",
        fontsize=14,
        fontweight="bold",
        color=verdict_color,
        y=1.02,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_qq(data: list[float], group_name: str, output_path: str) -> str:
    """
    Q-Q plot for normality visual inspection.
    Used when Shapiro-Wilk is skipped for large samples.
    Save to output_path. Return output_path.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 6))

    arr = np.array(data, dtype=float)
    (osm, osr), (slope, intercept, r) = scipy_stats.probplot(arr, dist="norm")

    ax.scatter(osm, osr, alpha=0.4, color="#4338ca", s=10, label="Data")
    x_line = np.linspace(min(osm), max(osm), 100)
    ax.plot(x_line, slope * x_line + intercept, color="#dc2626", linewidth=2, label="Normal line")

    ax.set_title(f"Q-Q Plot: {group_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
