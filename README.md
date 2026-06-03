---
title: Hypothesis Testing Explorer
emoji: 📊
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Hypothesis tests with plain-English verdicts + PySpark
---

# hypothesis-testing-explorer

Upload two CSV columns, select a statistical test, get a plain-English answer. Runs five hypothesis tests (independent t-test, paired t-test, Welch's t-test, Mann-Whitney U, chi-square), checks assumptions automatically, and routes large datasets (50,000+ rows) through PySpark.

**Live demo → [hypothesis-testing-xoc.azurewebsites.net](https://hypothesis-testing-xoc.azurewebsites.net)**
&nbsp;&nbsp;·&nbsp;&nbsp;
**API docs → [/docs](https://hypothesis-testing-xoc.azurewebsites.net/docs)**
&nbsp;&nbsp;·&nbsp;&nbsp;
**Notebook → [notebook.ipynb](notebook.ipynb)**

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![scipy](https://img.shields.io/badge/scipy-1.11+-green)
![PySpark](https://img.shields.io/badge/PySpark-3.5+-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-teal)
![Azure App Service](https://img.shields.io/badge/Azure-App_Service-0078D4)

---

## Prerequisites

- Python 3.11+
- Java (OpenJDK 11+) — required for PySpark
- pip

## Quick Start

```bash
git clone https://github.com/xavier-oc-programming/hypothesis-testing-explorer
cd hypothesis-testing-explorer

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

uvicorn main:app --reload
# → http://localhost:8000
```

Run the tests:

```bash
pytest tests/ -v
```

## Project Structure

```
hypothesis-testing-explorer/
├── config.py           # All constants — ALPHA, PYSPARK_THRESHOLD, Azure settings
├── stats_engine.py     # Statistical tests via pandas + scipy (small path, <50k rows)
├── spark_engine.py     # Statistical tests via PySpark (large path, ≥50k rows)
├── visualiser.py       # Distribution plots — KDE, histogram, box plot
├── main.py             # FastAPI app — routes, Pydantic models, upload handling
├── Dockerfile          # python:3.11-slim + OpenJDK (required for PySpark)
├── startup.txt         # Gunicorn startup command for Azure App Service
├── notebook.ipynb      # Full walkthrough — theory, all five tests, PySpark demo
├── requirements.txt
├── portfolio.yaml
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml      # GitHub Actions CI — installs Java, runs pytest
├── templates/
│   └── index.html      # Single-page demo frontend
├── tests/
│   └── test_api.py     # pytest API tests via TestClient
├── uploads/            # gitignored — runtime CSV storage
└── plots/              # gitignored — generated plot images
```

## Supported Tests

| Test | Use when | Parametric | Assumes equal variance |
|------|----------|------------|----------------------|
| Independent t-test | Two separate groups, continuous, normal | Yes | Yes |
| Welch's t-test | Two separate groups, continuous, unequal variance | Yes | No |
| Paired t-test | Same subjects measured twice | Yes | N/A |
| Mann-Whitney U | Non-normal data or ordinal data | No | No |
| Chi-square | Categorical variables, frequency data | No | N/A |

## Assumption Checking

Before running any parametric test, the app checks:

1. **Normality** — Shapiro-Wilk test on each group. If `p < 0.05`, the data is not normally distributed. Skipped on samples larger than 5,000 rows (Shapiro-Wilk is unreliable at large n — it flags trivial deviations that are statistically significant but practically irrelevant).
2. **Equal variance** — Levene's test. If `p < 0.05`, the variances differ. Welch's t-test is recommended over the standard independent t-test.

If assumptions are violated, the API response includes a `recommendation` field naming the appropriate alternative test.

## PySpark Routing

Datasets below 50,000 rows use `pandas` + `scipy`. Datasets at or above 50,000 rows route through PySpark (`local[*]` mode — all available CPU cores).

**The honest limitation:** PySpark handles data preparation and descriptive statistics. The actual test statistic is computed by `scipy` on a representative 10,000-row sample. This is the standard production pattern: distributed computing for data scale, established statistical libraries for the test itself. Spark does not have native hypothesis testing equivalent to `scipy.stats`.

The 50,000-row threshold is the crossover point where Spark's initialisation overhead (~3-5 seconds) is justified by the parallelism gains. Below it, pandas is faster; above it, Spark wins.

## API Reference

### `POST /test/upload`

Upload two CSV files and run a statistical test.

**Form fields:**
- `group1_file` / `group2_file` — CSV files
- `test_name` — one of `independent_ttest`, `paired_ttest`, `welch_ttest`, `mannwhitney`, `chisquare`
- `col1` / `col2` — column names (optional, defaults to first column)
- `alpha` — significance level (default `0.05`)
- `group1_name` / `group2_name` — display labels

### `POST /test/text`

Run a test on raw JSON arrays. Useful for testing without file uploads.

```json
{
  "group1": [1.2, 3.4, 5.6],
  "group2": [2.1, 4.5, 6.7],
  "test_name": "independent_ttest",
  "alpha": 0.05,
  "group1_name": "Group A",
  "group2_name": "Group B"
}
```

### `GET /api/tests`

Returns all supported tests with descriptions and `use_when` guidance.

### `GET /api/example-data`

Returns three pre-loaded example datasets (numpy seed 42 for reproducibility).

### `GET /health`

Returns API status, PySpark availability, routing threshold, and supported test names.

**Full interactive docs:** `/docs` (Swagger UI) · `/redoc` (ReDoc)

## Deployment — Azure App Service

```bash
# Package (exclude dev/runtime artifacts)
zip -r deploy.zip . \
  -x "*.git*" -x "venv/*" -x "__pycache__/*" \
  -x "*.ipynb_checkpoints*" -x "uploads/*" -x "plots/*"

az webapp deployment source config-zip \
  --name hypothesis-testing-xoc \
  --resource-group hypothesis-testing-rg \
  --src deploy.zip
```

**Java requirement:** PySpark requires the JVM. For Azure deployment without Docker, ensure the app service plan supports custom build steps or use the Dockerfile-based deployment path. The `Dockerfile` installs `default-jdk` via `apt-get`.

Startup command (set in Azure App Service configuration):

```
gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 600
```

## CI/CD

GitHub Actions runs on every push and pull request to `main`. The workflow:
1. Checks out code
2. Sets up Python 3.11
3. **Installs Java** — same requirement as the Dockerfile; PySpark will not initialise without it
4. Installs dependencies
5. Runs `pytest tests/ -v`

## Design Decisions

**Why plain-English output over raw statistics.** A p-value of 0.034 tells a statistician something precise but tells most stakeholders nothing actionable. The verdict — "There IS a statistically significant difference between the two groups (p = 0.034, which is below the significance threshold of 0.05)" — is immediately useful to a product manager, a doctor, or a student. The raw statistics are still shown for anyone who needs them, but the plain-English conclusion is the primary output.

**Why automatic assumption checking.** Running a t-test on non-normal data or data with unequal variances produces unreliable p-values. Most statistics tools run the test regardless and leave assumption checking to the user. This tool checks automatically and recommends an alternative when assumptions are violated — which is what a careful statistician does before reporting results.

**Why 50,000 rows as the PySpark threshold.** PySpark initialises in 3-5 seconds on a single machine due to JVM startup overhead. For datasets smaller than ~50,000 rows, this overhead makes it slower than pandas — a library that can load 50,000 rows in milliseconds. Above 50,000 rows, Spark's parallelism across all CPU cores (`local[*]`) pays off. At production scale on a cluster, the threshold would be higher, but the principle is the same: choose the right tool for the data size.

**Why scipy for the test statistic even in the PySpark path.** Spark does not have native hypothesis testing equivalent to `scipy.stats`. Spark MLlib has some statistical utilities but not the full suite of tests scipy provides. The correct production pattern is: use PySpark for data ingestion, filtering, and descriptive statistics at scale; use `scipy` on a representative sample for the test statistic. This is documented explicitly in `spark_engine.py` and in the notebook so the limitation is clear.

**Why five tests and not more.** These five tests cover the most common statistical questions in practice: comparing two group means (t-tests), handling non-normal data (Mann-Whitney), and testing categorical relationships (chi-square). Adding more tests — Kruskal-Wallis for multiple groups, ANOVA, Wilcoxon signed-rank — would broaden the scope without adding to the core demonstration of plain-English output and PySpark routing.

## Tool Comparison

| Concern | pandas + scipy | PySpark + scipy |
|---------|---------------|----------------|
| Dataset size | Up to ~1M rows in memory | Unlimited (distributed) |
| Speed on small data | Fast | Slow (3-5s init overhead) |
| Speed on large data | Slow (single core) | Fast (all cores / cluster) |
| Setup complexity | None | Requires Java + Spark |
| Production pattern | Single-machine analysis | Enterprise data pipelines |

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.0 | CSV parsing, small-path data handling |
| numpy | ≥1.24,<2.0 | Numerical operations, example data generation |
| scipy | ≥1.11 | Statistical tests — all five test implementations |
| pyspark | ≥3.5 | Large-dataset routing — descriptive stats, data prep |
| matplotlib | ≥3.7 | Distribution plots |
| seaborn | ≥0.12 | Plot styling |
| fastapi | ≥0.110 | REST API framework |
| uvicorn | ≥0.27 | ASGI server |
| gunicorn | ≥21.0 | Production WSGI server for Azure |
| pydantic | ≥2.0 | Request/response validation |
| pillow | ≥10.0 | Image handling |
| pytest | ≥7.0 | Test suite |
| httpx | ≥0.27 | FastAPI TestClient dependency |
| python-multipart | ≥0.0.9 | File upload form parsing |
