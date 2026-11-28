FROM python:3.12-slim

WORKDIR /app

COPY BIDS-SQL/  /app/BIDS-SQL/
COPY ingest/openneuro-crawler/ /app/crawler/
# For the post-crawl steps (post_crawl.py): dataset descriptions, README
# profiles and word forms import the backend's code, the shared LLM client and
# the vocabulary (also read by remote/participant_codes.py). All of backend/:
# a module missing here would only be logged and its step skipped
# (backend/tests/ingest/test_crawler_image.py). Editable installs, as in the
# backend image: the modules read the YAML and JSON files next to them.
COPY backend/ /app/backend/
COPY constraints.txt /app/constraints.txt

RUN pip install --no-cache-dir -e /app/BIDS-SQL && \
    pip install --no-cache-dir -c /app/constraints.txt -e /app/backend/vocabulary && \
    pip install --no-cache-dir -e /app/crawler && \
    pip install --no-cache-dir -c /app/constraints.txt google-genai

WORKDIR /app/crawler

CMD ["python", "crawler_service.py"]
