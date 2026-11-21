# Scored on linux/amd64 with no network. Same base image the tests were labelled in.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

WORKDIR /app
COPY src/ /app/
COPY data/map.bin /app/map.bin
COPY data/sqlglot-src.tar.xz /tmp/sqlglot-src.tar.xz

# The pinned sqlglot sources (sqlglot at eb11aec: its Python files and the TPC-DS fixtures, MIT)
# are committed in data/, so the build needs no network. The selector verifies every file against
# the map's digests and uses them to check that a patched module still compiles and imports, and
# to replay single TPC-DS queries.
RUN python -c "import tarfile; tarfile.open('/tmp/sqlglot-src.tar.xz').extractall('/app/sqlglot-src', filter='data')" \
    && rm /tmp/sqlglot-src.tar.xz \
    && python -m compileall -q /app/sqlglot-src/sqlglot \
    && python -S -B /app/import_baseline.py /app/sqlglot-src /app/import_baseline.txt \
    && python -m compileall -q /app/blast_select.py \
    && chmod -R a+rX /app

CMD ["python", "-S", "-B", "/app/run.py"]
