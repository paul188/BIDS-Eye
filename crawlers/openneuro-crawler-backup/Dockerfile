FROM python:3.12-slim

WORKDIR /app

COPY BIDS-SQL/  /app/BIDS-SQL/
COPY crawlers/openneuro-crawler-backup/ /app/crawler/

RUN pip install --no-cache-dir -e /app/BIDS-SQL && \
    pip install --no-cache-dir -e /app/crawler

WORKDIR /app/crawler

CMD ["python", "crawler_service.py"]
