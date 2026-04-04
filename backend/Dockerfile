FROM python:3.12-slim

WORKDIR /app

# Install BIDS-SQL (editable) then the backend itself
COPY BIDS-SQL/ /app/BIDS-SQL/
COPY backend/   /app/backend/
COPY Text-To-SQL/ /app/Text-To-SQL/

RUN pip install --no-cache-dir -e /app/BIDS-SQL && \
    pip install --no-cache-dir -e /app/backend

WORKDIR /app/backend

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
