"""Fetches, at build time, the test fixtures the committed sqlglot sources leave out.

    python fetch_fixtures.py <sqlglot sources>

data/sqlglot-src.tar.xz holds sqlglot's Python files, its tests and most fixtures, but not the
TPC-H tables or the JSONPath compliance suite. TestExecutor.setUpClass loads the TPC-H tables,
so without them every test of that class errors when the selector runs it, and is kept for
that. They are fetched from sqlglot's repository at the pinned commit and each file is checked
against its SHA-256. A file that cannot be fetched or does not match is left out: the build
still succeeds, and the selector keeps those tests as before.
"""

import hashlib
import os
import sys
import time
import urllib.request

COMMIT = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
URL = "https://raw.githubusercontent.com/tobymao/sqlglot/%s/%s"
FILES = (
    ("tests/fixtures/optimizer/tpc-h/customer.csv.gz", "45f17dd88099ce6093248c572487c7696f7d46855d59f0689afb1765170c6e04"),
    ("tests/fixtures/optimizer/tpc-h/lineitem.csv.gz", "55cbce7fc868fd95faf2acbbcda785456a0753ef40cd4efa460d4db7ae60e43e"),
    ("tests/fixtures/optimizer/tpc-h/nation.csv.gz", "c62e3f18f882364cf851ac364273eefdeec3f63316ce07abb3e5c0d82e7e7a5e"),
    ("tests/fixtures/optimizer/tpc-h/orders.csv.gz", "db196ef1d3ac09ad8853937118e729adb92160c40e3fae6cf35851f845640e0c"),
    ("tests/fixtures/optimizer/tpc-h/part.csv.gz", "62c9f8c5334d0b743c776f570acafe70fd7d11f4f73230f16c75ee771ca41b27"),
    ("tests/fixtures/optimizer/tpc-h/partsupp.csv.gz", "c74a7c4b6446ade0594ddee0152ba4ff6ca58ea5265ddca4cb029a6ffff8160b"),
    ("tests/fixtures/optimizer/tpc-h/region.csv.gz", "99f2b9d882955a66ab56844dc04153e83cef00d48a494dece48b80b9e1ff6353"),
    ("tests/fixtures/optimizer/tpc-h/supplier.csv.gz", "9c0c8355499d6ca08a4b51e84c33bb24da8bd1e887aa4b30f0439b4e4dd8b467"),
    ("tests/fixtures/jsonpath/cts.json", "4c968ee17d39f1fa73964c82068130bd94e91b55e14482ecce3d864980494133"),
    ("tests/fixtures/jsonpath/LICENSE", "0a76d5e15eeff92346a8783de64d5164c4d527a163f8599733e4e0ab941b59c0"),
)
MARKER = ".fixtures-complete"  # written when every file is present and verified


def fetch(rel):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(URL % (COMMIT, rel), timeout=30) as response:
                return response.read()
        except Exception as error:  # noqa: BLE001
            print("fetch_fixtures: %s attempt %d failed: %r" % (rel, attempt + 1, error), file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    return None


def main():
    root = sys.argv[1]
    ok = 0
    for rel, digest in FILES:
        path = os.path.join(root, rel)
        if os.path.exists(path):
            with open(path, "rb") as f:
                if hashlib.sha256(f.read()).hexdigest() == digest:
                    ok += 1
                    continue
        data = fetch(rel)
        if data is None or hashlib.sha256(data).hexdigest() != digest:
            print("fetch_fixtures: %s not available; left out" % rel, file=sys.stderr)
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        ok += 1
    if ok == len(FILES):
        with open(os.path.join(root, "tests", "fixtures", MARKER), "w") as f:
            f.write(COMMIT + "\n")
    print("fetch_fixtures: %d of %d fixture files in place" % (ok, len(FILES)), file=sys.stderr)


if __name__ == "__main__":
    main()
