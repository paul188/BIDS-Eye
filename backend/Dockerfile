FROM python:3.12-slim

WORKDIR /app

# Copy packages in dependency order
COPY BIDS-SQL/                         /app/BIDS-SQL/
COPY backend/                          /app/backend/
COPY LLM_preprocessor/                 /app/LLM_preprocessor/
COPY RAG/                              /app/RAG/

# Install BIDS-SQL dependencies and package
RUN pip install --no-cache-dir -r /app/BIDS-SQL/requirements.txt && \
    pip install --no-cache-dir -e /app/BIDS-SQL

# Install Backend dependencies from pyproject.toml
RUN pip install --no-cache-dir -e /app/backend

WORKDIR /app/backend

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
