FROM python:3.12-slim

WORKDIR /app

COPY BIDS-SQL/  /app/BIDS-SQL/
COPY crawlers/openneuro-crawler/ /app/crawler/
# For the post-crawl dataset descriptions (post_crawl.py): their code and the
# shared LLM client, with the Gemini SDK pinned like the backend's.
COPY backend/services/ /app/backend/services/
COPY backend/constraints.txt /app/backend/constraints.txt
COPY LLM_preprocessor/ /app/LLM_preprocessor/
# The vocabulary whose standard codes the post-crawl step writes for
# participant sex and handedness (remote/participant_codes.py).
COPY RAG/value_mappings.yaml /app/RAG/value_mappings.yaml

RUN pip install --no-cache-dir -e /app/BIDS-SQL && \
    pip install --no-cache-dir -e /app/crawler && \
    pip install --no-cache-dir -c /app/backend/constraints.txt google-genai

WORKDIR /app/crawler

CMD ["python", "crawler_service.py"]
