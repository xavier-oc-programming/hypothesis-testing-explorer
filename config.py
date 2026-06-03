# Routing threshold
PYSPARK_THRESHOLD = 50000
# Datasets above this row count route to PySpark.
# Below this threshold, pandas + scipy is faster due to Spark's
# initialisation overhead (~3-5 seconds). Above it, Spark's
# parallelism pays off. This threshold is documented in the UI
# so users understand why the engine switches.

# Statistical thresholds
ALPHA = 0.05
# The significance level. A p-value below ALPHA means
# we reject the null hypothesis. 0.05 is the conventional threshold
# in most fields — there is a 5% chance of a false positive.

NORMALITY_THRESHOLD = 0.05
# Shapiro-Wilk p-value threshold for normality assumption.
# If p < NORMALITY_THRESHOLD, the data is not normally distributed
# and a non-parametric test should be recommended.

LARGE_SAMPLE_NORMALITY = 5000
# Shapiro-Wilk is unreliable on very large samples —
# it will flag almost any large dataset as non-normal due to
# sensitivity to minor deviations. Above this size, use visual
# inspection (Q-Q plot) rather than Shapiro-Wilk.

# File limits
MAX_FILE_SIZE_MB = 50
MAX_COLUMNS = 100

# PySpark
SPARK_APP_NAME = "hypothesis-testing-explorer"

# Azure
AZURE_APP_NAME = "hypothesis-testing-xoc"
AZURE_RESOURCE_GROUP = "hypothesis-testing-rg"
AZURE_LOCATION = "westeurope"
