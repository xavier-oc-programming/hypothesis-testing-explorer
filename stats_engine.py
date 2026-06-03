from scipy import stats
from config import ALPHA, NORMALITY_THRESHOLD, LARGE_SAMPLE_NORMALITY


def check_normality(data: list[float]) -> dict:
    """
    Test normality using Shapiro-Wilk (n <= LARGE_SAMPLE_NORMALITY)
    or skip with a warning for larger samples.

    Returns:
        {
          "is_normal": bool,
          "statistic": float,
          "p_value": float,
          "test_used": "shapiro-wilk" or "skipped_large_sample",
          "interpretation": str
        }
    """
    n = len(data)
    if n > LARGE_SAMPLE_NORMALITY:
        return {
            "is_normal": True,
            "statistic": None,
            "p_value": None,
            "test_used": "skipped_large_sample",
            "interpretation": (
                f"Sample size ({n}) exceeds {LARGE_SAMPLE_NORMALITY}. "
                "Shapiro-Wilk is unreliable on large samples — it detects "
                "trivial deviations from normality. Normality assumed; use "
                "a Q-Q plot for visual inspection."
            ),
        }

    stat, p = stats.shapiro(data)
    is_normal = bool(p >= NORMALITY_THRESHOLD)
    return {
        "is_normal": is_normal,
        "statistic": float(stat),
        "p_value": float(p),
        "test_used": "shapiro-wilk",
        "interpretation": (
            f"Shapiro-Wilk test: W={stat:.4f}, p={p:.4f}. "
            + (
                "The data appears normally distributed."
                if is_normal
                else "The data does NOT appear normally distributed. "
                "Consider a non-parametric test."
            )
        ),
    }


def check_equal_variance(group1: list[float], group2: list[float]) -> dict:
    """
    Test equal variance using Levene's test.

    Returns:
        {
          "equal_variance": bool,
          "statistic": float,
          "p_value": float,
          "interpretation": str
        }
    """
    stat, p = stats.levene(group1, group2)
    equal_variance = bool(p >= NORMALITY_THRESHOLD)
    return {
        "equal_variance": equal_variance,
        "statistic": float(stat),
        "p_value": float(p),
        "interpretation": (
            f"Levene's test: W={stat:.4f}, p={p:.4f}. "
            + (
                "The variances of the two groups are approximately equal."
                if equal_variance
                else "The variances of the two groups are significantly different. "
                "Welch's t-test is recommended over the standard independent t-test."
            )
        ),
    }


def run_independent_ttest(group1: list[float], group2: list[float]) -> dict:
    """
    Independent samples t-test (scipy.stats.ttest_ind).
    Assumes equal variances and normality.
    Checks assumptions first — recommends Welch's if variance unequal,
    Mann-Whitney if non-normal.

    Returns full result dict (see run_test() return format).
    """
    norm1 = check_normality(group1)
    norm2 = check_normality(group2)
    var_check = check_equal_variance(group1, group2)

    recommendation = None
    if not norm1["is_normal"] or not norm2["is_normal"]:
        recommendation = (
            "One or both groups are not normally distributed. "
            "Mann-Whitney U test is recommended as a non-parametric alternative."
        )
    elif not var_check["equal_variance"]:
        recommendation = (
            "The group variances are unequal. "
            "Welch's t-test is recommended — it does not assume equal variances."
        )

    stat, p = stats.ttest_ind(group1, group2, equal_var=True)
    significant = bool(p < ALPHA)
    n1, n2 = len(group1), len(group2)

    verdict = _build_verdict(significant, p)
    interpretation = (
        f"The independent t-test compares the average values of two separate groups. "
        f"A significant result means the difference between the averages is unlikely "
        f"to be due to random chance alone. With {n1} and {n2} observations, the test "
        f"had sufficient data to detect a meaningful difference if one exists."
    )

    return {
        "test_name": "independent_ttest",
        "statistic": float(stat),
        "p_value": float(p),
        "significant": significant,
        "alpha": ALPHA,
        "verdict": verdict,
        "interpretation": interpretation,
        "assumption_checks": {
            "normality_group1": norm1,
            "normality_group2": norm2,
            "equal_variance": var_check,
        },
        "recommendation": recommendation,
        "engine": "pandas/scipy",
        "n_group1": n1,
        "n_group2": n2,
    }


def run_welch_ttest(group1: list[float], group2: list[float]) -> dict:
    """
    Welch's t-test (scipy.stats.ttest_ind with equal_var=False).
    Does not assume equal variances. Preferred over Student's t-test
    when variances differ.
    """
    norm1 = check_normality(group1)
    norm2 = check_normality(group2)
    var_check = check_equal_variance(group1, group2)

    recommendation = None
    if not norm1["is_normal"] or not norm2["is_normal"]:
        recommendation = (
            "One or both groups are not normally distributed. "
            "Mann-Whitney U test is recommended as a non-parametric alternative."
        )

    stat, p = stats.ttest_ind(group1, group2, equal_var=False)
    significant = bool(p < ALPHA)
    n1, n2 = len(group1), len(group2)

    verdict = _build_verdict(significant, p)
    interpretation = (
        f"Welch's t-test compares the means of two groups without assuming equal variances. "
        f"It is the safer choice when the spread of data differs between groups. "
        f"With {n1} and {n2} observations, the test examined whether the observed mean "
        f"difference is larger than what would be expected from random variation alone."
    )

    return {
        "test_name": "welch_ttest",
        "statistic": float(stat),
        "p_value": float(p),
        "significant": significant,
        "alpha": ALPHA,
        "verdict": verdict,
        "interpretation": interpretation,
        "assumption_checks": {
            "normality_group1": norm1,
            "normality_group2": norm2,
            "equal_variance": var_check,
        },
        "recommendation": recommendation,
        "engine": "pandas/scipy",
        "n_group1": n1,
        "n_group2": n2,
    }


def run_paired_ttest(group1: list[float], group2: list[float]) -> dict:
    """
    Paired samples t-test (scipy.stats.ttest_rel).
    Requires equal-length arrays — the same subjects measured twice.
    """
    if len(group1) != len(group2):
        raise ValueError(
            f"Paired t-test requires equal-length groups. "
            f"Got {len(group1)} and {len(group2)}."
        )

    differences = [a - b for a, b in zip(group1, group2)]
    norm_diff = check_normality(differences)

    recommendation = None
    if not norm_diff["is_normal"]:
        recommendation = (
            "The differences between paired observations are not normally distributed. "
            "Wilcoxon signed-rank test is the non-parametric alternative for paired data."
        )

    stat, p = stats.ttest_rel(group1, group2)
    significant = bool(p < ALPHA)
    n = len(group1)

    verdict = _build_verdict(significant, p)
    interpretation = (
        f"The paired t-test compares two measurements from the same subjects — "
        f"for example, before and after a treatment. By comparing each pair directly, "
        f"it removes individual variation and focuses on the change. "
        f"With {n} paired observations, the test assessed whether the average change "
        f"is distinguishable from zero."
    )

    return {
        "test_name": "paired_ttest",
        "statistic": float(stat),
        "p_value": float(p),
        "significant": significant,
        "alpha": ALPHA,
        "verdict": verdict,
        "interpretation": interpretation,
        "assumption_checks": {
            "normality_of_differences": norm_diff,
        },
        "recommendation": recommendation,
        "engine": "pandas/scipy",
        "n_group1": n,
        "n_group2": n,
    }


def run_mannwhitney(group1: list[float], group2: list[float]) -> dict:
    """
    Mann-Whitney U test (scipy.stats.mannwhitneyu).
    Non-parametric — does not assume normality.
    Tests whether one distribution is stochastically greater than the other.
    """
    stat, p = stats.mannwhitneyu(group1, group2, alternative="two-sided")
    significant = bool(p < ALPHA)
    n1, n2 = len(group1), len(group2)

    verdict = _build_verdict(significant, p)
    interpretation = (
        f"The Mann-Whitney U test is a non-parametric test that compares two groups "
        f"without assuming the data follows a normal distribution. It works by ranking "
        f"all values together and checking whether one group tends to have higher ranks "
        f"than the other. With {n1} and {n2} observations, a significant result means "
        f"the two groups come from different distributions."
    )

    return {
        "test_name": "mannwhitney",
        "statistic": float(stat),
        "p_value": float(p),
        "significant": significant,
        "alpha": ALPHA,
        "verdict": verdict,
        "interpretation": interpretation,
        "assumption_checks": {},
        "recommendation": None,
        "engine": "pandas/scipy",
        "n_group1": n1,
        "n_group2": n2,
    }


def run_chisquare(observed: list[int], expected: list[int] = None) -> dict:
    """
    Chi-square test (scipy.stats.chisquare or chi2_contingency).
    For independence between categorical variables.
    """
    if expected is not None:
        stat, p = stats.chisquare(observed, f_exp=expected)
    else:
        stat, p = stats.chisquare(observed)

    significant = bool(p < ALPHA)
    n = sum(observed)

    verdict = _build_verdict(significant, p)
    interpretation = (
        f"The chi-square test checks whether the observed frequencies differ "
        f"significantly from what we would expect if there were no relationship "
        f"between the variables. With {n} total observations, "
        + (
            "the observed pattern is unlikely to have occurred by chance alone."
            if significant
            else "the observed pattern is consistent with random variation."
        )
    )

    return {
        "test_name": "chisquare",
        "statistic": float(stat),
        "p_value": float(p),
        "significant": significant,
        "alpha": ALPHA,
        "verdict": verdict,
        "interpretation": interpretation,
        "assumption_checks": {},
        "recommendation": None,
        "engine": "pandas/scipy",
        "n_group1": n,
        "n_group2": None,
    }


def run_test(
    test_name: str,
    group1: list[float],
    group2: list[float] = None,
) -> dict:
    """
    Dispatcher — routes to the correct test function.

    Returns standard result format:
        {
          "test_name": str,
          "statistic": float,
          "p_value": float,
          "significant": bool,
          "alpha": float,
          "verdict": str,
          "interpretation": str,
          "assumption_checks": dict,
          "recommendation": str or None,
          "engine": "pandas/scipy",
          "n_group1": int,
          "n_group2": int or None
        }

    verdict format:
    If significant:
      "There IS a statistically significant difference between the two groups
       (p = {p_value:.4f}, which is below the significance threshold of {ALPHA})."
    If not significant:
      "There IS NOT a statistically significant difference between the two groups
       (p = {p_value:.4f}, which is above the significance threshold of {ALPHA})."
    """
    supported = {
        "independent_ttest",
        "paired_ttest",
        "welch_ttest",
        "mannwhitney",
        "chisquare",
    }
    if test_name not in supported:
        raise ValueError(
            f"Unsupported test '{test_name}'. "
            f"Supported tests: {', '.join(sorted(supported))}."
        )

    if test_name == "independent_ttest":
        return run_independent_ttest(group1, group2)
    elif test_name == "welch_ttest":
        return run_welch_ttest(group1, group2)
    elif test_name == "paired_ttest":
        return run_paired_ttest(group1, group2)
    elif test_name == "mannwhitney":
        return run_mannwhitney(group1, group2)
    elif test_name == "chisquare":
        return run_chisquare(group1, group2)


def _build_verdict(significant: bool, p_value: float) -> str:
    if significant:
        return (
            f"There IS a statistically significant difference between the two groups "
            f"(p = {p_value:.4f}, which is below the significance threshold of {ALPHA})."
        )
    return (
        f"There IS NOT a statistically significant difference between the two groups "
        f"(p = {p_value:.4f}, which is above the significance threshold of {ALPHA})."
    )
