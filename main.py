import os
import uuid
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from config import PYSPARK_THRESHOLD, ALPHA
import stats_engine
import spark_engine
import visualiser

UPLOAD_DIR = Path("uploads")
PLOTS_DIR = Path("plots")
UPLOAD_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

SPARK_AVAILABLE = True

SUPPORTED_TESTS = [
    {
        "name": "independent_ttest",
        "label": "Independent t-test",
        "description": "Compare the means of two independent groups. Assumes normality and equal variance.",
        "use_when": "Two separate groups, continuous data, approximately normal distribution.",
        "parametric": True,
    },
    {
        "name": "welch_ttest",
        "label": "Welch's t-test",
        "description": "t-test that does not assume equal variances. More robust than the standard independent t-test.",
        "use_when": "Two separate groups, continuous data, normal distribution, but unequal variances.",
        "parametric": True,
    },
    {
        "name": "paired_ttest",
        "label": "Paired t-test",
        "description": "Compare two measurements from the same subjects. Removes individual variation.",
        "use_when": "Same subjects measured twice (e.g. before/after treatment), continuous data.",
        "parametric": True,
    },
    {
        "name": "mannwhitney",
        "label": "Mann-Whitney U",
        "description": "Non-parametric alternative to the independent t-test. Does not assume normality.",
        "use_when": "Two independent groups, non-normal data or ordinal data.",
        "parametric": False,
    },
    {
        "name": "chisquare",
        "label": "Chi-square test",
        "description": "Test whether observed frequencies differ from expected frequencies.",
        "use_when": "Categorical variables, frequency/count data.",
        "parametric": False,
    },
]

SUPPORTED_TEST_NAMES = {t["name"] for t in SUPPORTED_TESTS}

app = FastAPI(
    title="Hypothesis Testing Explorer",
    description=(
        f"Upload two CSV columns and select a statistical test. Returns a plain-English verdict, "
        f"test statistic, p-value, and distribution visualisation. Routes large datasets "
        f"(>{PYSPARK_THRESHOLD} rows) through PySpark automatically."
    ),
    version="1.0.0",
)

class TestRequest(BaseModel):
    test_name: str = Field(
        ...,
        description="One of: independent_ttest, paired_ttest, welch_ttest, mannwhitney, chisquare",
    )
    alpha: float = Field(0.05, ge=0.001, le=0.1, description="Significance level (default 0.05)")
    group1_name: str = Field("Group 1", description="Label for group 1")
    group2_name: str = Field("Group 2", description="Label for group 2")
    group1: list[float]
    group2: list[float] | None = None


class TestResult(BaseModel):
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    verdict: str
    interpretation: str
    assumption_checks: dict
    recommendation: str | None = None
    engine: str
    n_group1: int
    n_group2: int | None = None
    plot_url: str | None = None
    spark_stats: dict | None = None
    total_rows: int | None = None
    sampled_for_test: int | None = None


class HealthResponse(BaseModel):
    status: str
    spark_available: bool
    pyspark_threshold: int
    supported_tests: list[str]


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html") as f:
        return HTMLResponse(f.read())


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        spark_available=SPARK_AVAILABLE,
        pyspark_threshold=PYSPARK_THRESHOLD,
        supported_tests=[t["name"] for t in SUPPORTED_TESTS],
    )


@app.post("/test/upload", response_model=TestResult)
async def test_upload(
    group1_file: UploadFile = File(...),
    group2_file: UploadFile = File(...),
    test_name: str = Form(...),
    alpha: float = Form(0.05),
    group1_name: str = Form("Group 1"),
    group2_name: str = Form("Group 2"),
    col1: str = Form(None),
    col2: str = Form(None),
):
    if test_name not in SUPPORTED_TEST_NAMES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported test '{test_name}'. Supported: {', '.join(sorted(SUPPORTED_TEST_NAMES))}.",
        )

    for f in [group1_file, group2_file]:
        if not f.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail=f"File '{f.filename}' must be a CSV.")

    uid = uuid.uuid4().hex
    path1 = UPLOAD_DIR / f"{uid}_group1.csv"
    path2 = UPLOAD_DIR / f"{uid}_group2.csv"

    path1.write_bytes(await group1_file.read())
    path2.write_bytes(await group2_file.read())

    try:
        df1 = pd.read_csv(path1)
        df2 = pd.read_csv(path2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    col1 = col1 or df1.columns[0]
    col2 = col2 or df2.columns[0]

    if col1 not in df1.columns:
        raise HTTPException(status_code=400, detail=f"Column '{col1}' not found in group 1 file.")
    if col2 not in df2.columns:
        raise HTTPException(status_code=400, detail=f"Column '{col2}' not found in group 2 file.")

    try:
        group1 = pd.to_numeric(df1[col1], errors="raise").dropna().tolist()
        group2 = pd.to_numeric(df2[col2], errors="raise").dropna().tolist()
    except Exception:
        raise HTTPException(status_code=400, detail="Columns must contain numeric data.")

    total_rows = len(group1) + len(group2)

    # Route large datasets through PySpark — see config.PYSPARK_THRESHOLD for rationale.
    if total_rows >= PYSPARK_THRESHOLD:
        result = spark_engine.run_test_spark(
            test_name=test_name,
            group1_path=str(path1),
            group2_path=str(path2),
            col1=col1,
            col2=col2,
        )
    else:
        result = stats_engine.run_test(test_name, group1, group2, alpha=alpha)

    plot_filename = f"{uid}_plot.png"
    plot_path = PLOTS_DIR / plot_filename
    visualiser.plot_distributions(
        group1=group1[:5000],
        group2=group2[:5000],
        group1_name=group1_name,
        group2_name=group2_name,
        test_result=result,
        output_path=str(plot_path),
    )

    result["plot_url"] = f"/plots/{plot_filename}"
    return TestResult(**result)


@app.post("/test/text", response_model=TestResult)
async def test_text(body: TestRequest):
    if body.test_name not in SUPPORTED_TEST_NAMES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported test '{body.test_name}'. Supported: {', '.join(sorted(SUPPORTED_TEST_NAMES))}.",
        )

    try:
        result = stats_engine.run_test(body.test_name, body.group1, body.group2, alpha=body.alpha)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    uid = uuid.uuid4().hex
    plot_filename = f"{uid}_plot.png"
    plot_path = PLOTS_DIR / plot_filename

    group2 = body.group2 or []
    if group2:
        visualiser.plot_distributions(
            group1=body.group1,
            group2=group2,
            group1_name=body.group1_name,
            group2_name=body.group2_name,
            test_result=result,
            output_path=str(plot_path),
        )
        result["plot_url"] = f"/plots/{plot_filename}"

    return TestResult(**result)


@app.get("/plots/{filename}")
async def serve_plot(filename: str):
    path = PLOTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Plot not found.")
    return FileResponse(str(path), media_type="image/png")


@app.get("/api/tests")
async def list_tests():
    return SUPPORTED_TESTS


@app.get("/api/example-data")
async def example_data():
    rng = np.random.default_rng(42)
    return {
        "exam_scores": {
            "group1": rng.normal(72, 12, 80).round(1).tolist(),
            "group2": rng.normal(78, 11, 80).round(1).tolist(),
            "description": "Exam scores: students taught with Method A vs Method B",
            "test_name": "independent_ttest",
            "group1_name": "Method A",
            "group2_name": "Method B",
        },
        "sales_by_region": {
            "group1": rng.normal(42000, 8000, 60).round(0).tolist(),
            "group2": rng.normal(45000, 9500, 60).round(0).tolist(),
            "description": "Monthly sales: North region vs South region",
            "test_name": "welch_ttest",
            "group1_name": "North",
            "group2_name": "South",
        },
        "treatment_effect": {
            "group1": rng.normal(140, 15, 50).round(1).tolist(),
            "group2": (rng.normal(140, 15, 50) - rng.normal(8, 4, 50)).round(1).tolist(),
            "description": "Blood pressure: before and after treatment (paired)",
            "test_name": "paired_ttest",
            "group1_name": "Before",
            "group2_name": "After",
        },
        "response_times": {
            "group1": np.clip(rng.exponential(scale=200, size=80), 50, 2000).round(1).tolist(),
            "group2": np.clip(rng.exponential(scale=350, size=80), 50, 2000).round(1).tolist(),
            "description": "API response times (ms): optimised vs legacy service — skewed, non-normal",
            "test_name": "mannwhitney",
            "group1_name": "Optimised",
            "group2_name": "Legacy",
        },
        "survey_ratings": {
            "group1": (rng.choice([1,2,3,4,5], size=120, p=[0.05,0.10,0.25,0.40,0.20])).tolist(),
            "group2": (rng.choice([1,2,3,4,5], size=120, p=[0.15,0.25,0.30,0.20,0.10])).tolist(),
            "description": "Product ratings 1-5: new design vs old design — ordinal/categorical",
            "test_name": "chisquare",
            "group1_name": "New design",
            "group2_name": "Old design",
        },
    }
