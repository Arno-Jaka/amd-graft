FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if needed (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata
COPY pyproject.toml README.md /app/

# Copy source code
COPY src /app/src
COPY examples /app/examples

# Install the package
RUN pip install --no-cache-dir .

# Set up entrypoint
ENTRYPOINT ["graft"]
CMD ["--help"]
