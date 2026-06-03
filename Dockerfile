FROM python:3.11-slim
WORKDIR /app

# PySpark requires Java. default-jdk installs OpenJDK.
# This adds ~200MB to the image but is required for Spark to initialise.
# JAVA_HOME is set automatically by the package.
RUN apt-get update && apt-get install -y default-jdk && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p uploads plots

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
