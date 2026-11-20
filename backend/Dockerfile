FROM python:3.12-slim

WORKDIR /app

# Copy packages in dependency order
COPY BIDS-SQL/                         /app/BIDS-SQL/
COPY backend/                          /app/backend/
COPY LLM_preprocessor/                 /app/LLM_preprocessor/
COPY RAG/                              /app/RAG/

# The commit this image is built from (read by benchmarks/demand/record_run.py
# to label its recordings). Either pass
# --build-arg BIDSEYE_COMMIT=<hash>, or have a file DEPLOYED_COMMIT in the
# build context (the deploy writes it before building). The file is optional:
# constraints.txt is named only so that the COPY has a source without it.
ARG BIDSEYE_COMMIT=""
ENV BIDSEYE_COMMIT=${BIDSEYE_COMMIT}
COPY backend/constraints.txt DEPLOYED_COMMIT* /app/

# Every pip call is constrained to the pinned versions in constraints.txt so
# that a rebuild yields the same environment (see the comment in that file).
ENV PIP_CONSTRAINT=/app/backend/constraints.txt

# Install BIDS-SQL dependencies and package
RUN pip install --no-cache-dir -r /app/BIDS-SQL/requirements.txt && \
    pip install --no-cache-dir -e /app/BIDS-SQL

# RAG/ and LLM_preprocessor/, which the backend imports by module name
# (their pyproject.toml lists the modules)
RUN pip install --no-cache-dir -e /app/RAG -e /app/LLM_preprocessor

# Install Backend dependencies from pyproject.toml
RUN pip install --no-cache-dir -e /app/backend

WORKDIR /app/backend

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
