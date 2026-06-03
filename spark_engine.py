import os
import tempfile
from config import SPARK_APP_NAME, ALPHA, NORMALITY_THRESHOLD
from stats_engine import run_test as run_test_scipy, _build_verdict

_spark = None


def get_spark():
    """
    Lazy-load SparkSession on first call. Cached globally.

    In production on Databricks or EMR, the master URL would
    point to a cluster. Here we use local[*] — all available CPU cores —
    which still parallelises computation on a single machine and demonstrates
    the Spark API pattern identical to what would run on a cluster.
    """
    global _spark
    if _spark is None:
        from pyspark.sql import SparkSession

        _spark = (
            SparkSession.builder.appName(SPARK_APP_NAME)
            .master("local[*]")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )
        _spark.sparkContext.setLogLevel("ERROR")
    return _spark


def get_spark_summary(df_path: str, col: str) -> dict:
    """
    Compute descriptive statistics for one column using PySpark.
    Returns: {mean, std, count, min, max, median (approx)}
    """
    from pyspark.sql import functions as F

    spark = get_spark()
    df = spark.read.csv(df_path, header=True, inferSchema=True)
    df = df.select(F.col(col).cast("double").alias("value")).dropna()

    agg = df.agg(
        F.mean("value").alias("mean"),
        F.stddev("value").alias("std"),
        F.count("value").alias("count"),
        F.min("value").alias("min"),
        F.max("value").alias("max"),
        F.percentile_approx("value", 0.5).alias("median"),
    ).collect()[0]

    return {
        "mean": float(agg["mean"]) if agg["mean"] is not None else None,
        "std": float(agg["std"]) if agg["std"] is not None else None,
        "count": int(agg["count"]),
        "min": float(agg["min"]) if agg["min"] is not None else None,
        "max": float(agg["max"]) if agg["max"] is not None else None,
        "median": float(agg["median"]) if agg["median"] is not None else None,
    }


def run_test_spark(
    test_name: str,
    group1_path: str,
    group2_path: str = None,
    col1: str = None,
    col2: str = None,
) -> dict:
    """
    Run a statistical test on a large dataset using PySpark.

    Reads data from temporary CSV files written by the upload handler.
    Uses PySpark for descriptive statistics and mean/variance computation.
    Falls back to scipy for the actual test statistic (Spark does not have
    native hypothesis testing) — but all data preparation and summary
    statistics use PySpark DataFrames.

    This is the honest framing: PySpark handles the data scale,
    scipy computes the test statistic. In a true enterprise pipeline,
    a distributed statistics library (e.g. Spark MLlib) would be used.
    For this portfolio project, the split demonstrates PySpark DataFrame
    operations on large data.

    Steps:
    1. Read CSV into Spark DataFrame
    2. Compute descriptive stats via df.describe()
    3. Compute group means, variances, counts via df.groupBy().agg()
    4. Sample down to 10,000 rows for the scipy test statistic computation
       (with a note in the result that sampling was applied)
    5. Run scipy test on the sample
    6. Return result with engine = "PySpark + scipy" and row_count

    # PySpark handles data at scale — reading, filtering, aggregating
    # millions of rows in parallel. The actual hypothesis test statistic
    # is computed by scipy on a representative sample. This is a common
    # pattern in production: use distributed computing for data preparation,
    # use established statistical libraries for the test itself.
    """
    from pyspark.sql import functions as F

    SAMPLE_SIZE = 10_000

    spark = get_spark()

    df1 = spark.read.csv(group1_path, header=True, inferSchema=True)
    col1_name = col1 or df1.columns[0]
    df1 = df1.select(F.col(col1_name).cast("double").alias("value")).dropna()
    total_rows_1 = df1.count()

    stats1 = df1.agg(
        F.mean("value").alias("mean"),
        F.stddev("value").alias("std"),
        F.count("value").alias("count"),
        F.min("value").alias("min"),
        F.max("value").alias("max"),
    ).collect()[0]

    if group2_path:
        df2 = spark.read.csv(group2_path, header=True, inferSchema=True)
        col2_name = col2 or df2.columns[0]
        df2 = df2.select(F.col(col2_name).cast("double").alias("value")).dropna()
        total_rows_2 = df2.count()

        stats2 = df2.agg(
            F.mean("value").alias("mean"),
            F.stddev("value").alias("std"),
            F.count("value").alias("count"),
            F.min("value").alias("min"),
            F.max("value").alias("max"),
        ).collect()[0]
    else:
        total_rows_2 = 0
        stats2 = None

    total_rows = total_rows_1 + (total_rows_2 if group2_path else 0)

    fraction1 = min(1.0, SAMPLE_SIZE / max(total_rows_1, 1))
    sample1 = [
        float(row["value"])
        for row in df1.sample(fraction=fraction1, seed=42).limit(SAMPLE_SIZE).collect()
    ]

    if group2_path and stats2 is not None:
        fraction2 = min(1.0, SAMPLE_SIZE / max(total_rows_2, 1))
        sample2 = [
            float(row["value"])
            for row in df2.sample(fraction=fraction2, seed=42).limit(SAMPLE_SIZE).collect()
        ]
    else:
        sample2 = None

    scipy_result = run_test_scipy(test_name, sample1, sample2)

    spark_stats = {
        "group1": {
            "mean": float(stats1["mean"]) if stats1["mean"] is not None else None,
            "std": float(stats1["std"]) if stats1["std"] is not None else None,
            "count": int(stats1["count"]),
            "min": float(stats1["min"]) if stats1["min"] is not None else None,
            "max": float(stats1["max"]) if stats1["max"] is not None else None,
        }
    }
    if stats2 is not None:
        spark_stats["group2"] = {
            "mean": float(stats2["mean"]) if stats2["mean"] is not None else None,
            "std": float(stats2["std"]) if stats2["std"] is not None else None,
            "count": int(stats2["count"]),
            "min": float(stats2["min"]) if stats2["min"] is not None else None,
            "max": float(stats2["max"]) if stats2["max"] is not None else None,
        }

    sampled_for_test = len(sample1) + (len(sample2) if sample2 else 0)

    result = {**scipy_result}
    result["engine"] = "PySpark + scipy"
    result["total_rows"] = total_rows
    result["sampled_for_test"] = sampled_for_test
    result["spark_stats"] = spark_stats
    result["n_group1"] = int(stats1["count"])
    result["n_group2"] = int(stats2["count"]) if stats2 is not None else None
    result["interpretation"] = (
        result["interpretation"]
        + f" Note: PySpark computed descriptive statistics on all {total_rows:,} rows. "
        f"The test statistic was computed by scipy on a representative sample of "
        f"{sampled_for_test:,} rows."
    )

    return result
