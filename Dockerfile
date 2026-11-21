# Scored on linux/amd64 with no network. Same base image the tests were labelled in.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

ARG SQLGLOT_COMMIT=eb11aec4be8cb2122afe713a4816662c3f0fb2fa

WORKDIR /app
COPY src/ /app/
COPY data/map.bin /app/map.bin

# Pinned sqlglot sources (from sqlglot's own repository at the pinned commit) let the selector
# check that a patched module still compiles and imports. The map itself is committed in data/.
RUN python /app/fetch_sources.py "$SQLGLOT_COMMIT" /app/sqlglot-src \
    && python -S -B /app/import_baseline.py /app/sqlglot-src /app/import_baseline.txt \
    && python -m compileall -q /app/blast_select.py \
    && chmod -R a+rX /app

CMD ["python", "-S", "-B", "/app/run.py"]
