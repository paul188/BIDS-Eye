FROM python:3.12-slim

WORKDIR /app

# Copy packages in dependency order (backend/ includes llm/ and vocabulary/)
COPY BIDS-SQL/                         /app/BIDS-SQL/
COPY backend/                          /app/backend/

# The commit this image is built from (read by benchmarks/demand/record_run.py
# to label its recordings). Either pass
# --build-arg BIDSEYE_COMMIT=<hash>, or have a file DEPLOYED_COMMIT in the
# build context (the deploy writes it before building). The file is optional:
# constraints.txt is named only so that the COPY has a source without it.
ARG BIDSEYE_COMMIT=""
ENV BIDSEYE_COMMIT=${BIDSEYE_COMMIT}
COPY constraints.txt DEPLOYED_COMMIT* /app/

# Every pip call is constrained to the pinned versions in constraints.txt so
# that a rebuild yields the same environment (see the comment in that file).
ENV PIP_CONSTRAINT=/app/constraints.txt

# Install BIDS-SQL dependencies and package
RUN pip install --no-cache-dir -r /app/BIDS-SQL/requirements.txt && \
    pip install --no-cache-dir -e /app/BIDS-SQL

# backend/vocabulary/ and backend/llm/, which the backend imports by module name
# (their pyproject.toml lists the modules)
RUN pip install --no-cache-dir -e /app/backend/vocabulary -e /app/backend/llm

# Install Backend dependencies from pyproject.toml
RUN pip install --no-cache-dir -e /app/backend

WORKDIR /app/backend

# Run as an unprivileged user. It can read /app but not change it; its home
# directory is the only place it can write.
RUN useradd --create-home --uid 10001 app
USER app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
