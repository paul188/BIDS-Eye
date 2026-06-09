FROM python:3.12-slim

WORKDIR /app

# Copy packages in dependency order
COPY BIDS-SQL/                         /app/BIDS-SQL/
COPY backend/                          /app/backend/
COPY LLM_preprocessor/                 /app/LLM_preprocessor/
COPY RAG/                              /app/RAG/
COPY modal_app/few_shot_examples.json  /app/backend/few_shot_examples.json

RUN pip install --no-cache-dir -e /app/BIDS-SQL && \
    pip install --no-cache-dir -e /app/backend

WORKDIR /app/backend

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
