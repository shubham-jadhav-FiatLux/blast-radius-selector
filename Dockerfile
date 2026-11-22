# Scored on linux/amd64 with no network. Same base image the tests were labelled in.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

WORKDIR /app
# Fixed so the TPC-DS plan digests computed below match the ones computed while scoring.
ENV PYTHONHASHSEED=0
# The packages sqlglot's tests import, at the versions the tests are labelled with, so the
# selector can run cheap candidate tests itself.
RUN pip install --no-cache-dir duckdb==2.0.0 numpy==2.5.5 pandas==3.1.1 python-dateutil==4.0.0 pytz==2026.6 tzdata==2026.6
COPY src/ /app/
COPY data/map.bin /app/map.bin
COPY data/sqlglot-src.tar.xz /tmp/sqlglot-src.tar.xz

# The pinned sqlglot sources (sqlglot at eb11aec: its Python files and the TPC-DS fixtures, MIT)
# are committed in data/, so the build needs no network. The selector verifies every file against
# the map's digests and uses them to check that a patched module still compiles and imports, and
# to replay single TPC-DS queries. The digests of each TPC-DS query's plan are computed here, in
# the image that will use them (a missing file only disables comparing plans).
RUN python -c "import tarfile; tarfile.open('/tmp/sqlglot-src.tar.xz').extractall('/app/sqlglot-src', filter='data')" \
    && rm /tmp/sqlglot-src.tar.xz \
    && python -m compileall -q /app/sqlglot-src/sqlglot \
    && python -S -B /app/import_baseline.py /app/sqlglot-src /app/import_baseline.txt \
    && python -S -B /app/tpcds_check.py --plans /app/sqlglot-src /app/tpcds_plans.json \
    && python -S -B /app/max_nodes_check.py --reference /app/sqlglot-src /app/max_nodes_ref.json \
    && python -m compileall -q /app/blast_select.py \
    && chmod -R a+rX /app

CMD ["python", "-S", "-B", "/app/run.py"]
