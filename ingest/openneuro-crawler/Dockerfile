FROM python:3.12-slim

WORKDIR /app

COPY BIDS-SQL/  /app/BIDS-SQL/
COPY ingest/openneuro-crawler/ /app/crawler/
# For the post-crawl dataset descriptions (post_crawl.py): their code and the
# shared LLM client, with the Gemini SDK pinned like the backend's.
COPY backend/services/ /app/backend/services/
COPY backend/exceptions/ /app/backend/exceptions/
COPY constraints.txt /app/constraints.txt
COPY backend/llm/ /app/backend/llm/
# The vocabulary: its standard codes for participant sex and handedness
# (remote/participant_codes.py), and the diagnosis resolver that codes the
# conditions of the README profiles (backend/services/indexing/
# readme_profile_runner.py). Editable, as in the backend image: the modules
# read the YAML and JSON files next to them.
COPY backend/vocabulary/ /app/backend/vocabulary/

RUN pip install --no-cache-dir -e /app/BIDS-SQL && \
    pip install --no-cache-dir -c /app/constraints.txt -e /app/backend/vocabulary && \
    pip install --no-cache-dir -e /app/crawler && \
    pip install --no-cache-dir -c /app/constraints.txt google-genai

WORKDIR /app/crawler

CMD ["python", "crawler_service.py"]
