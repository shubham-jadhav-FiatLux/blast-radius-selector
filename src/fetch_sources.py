"""Fetches the pinned sqlglot sources at build time.

    python fetch_sources.py <commit> <target dir>

Downloads sqlglot's own repository archive at the pinned commit and keeps its Python files.
The selector uses them to check that a patched module still compiles. If the download fails
the build still succeeds: the selector then skips that check and relies on the map alone.
"""

import io
import os
import sys
import tarfile
import time
import urllib.request

KEEP = ("sqlglot/", "tests/", "sqlglotc/")


def download(commit):
    url = "https://codeload.github.com/tobymao/sqlglot/tar.gz/%s" % commit
    # At most about three minutes in all: the build has ten, and a missing archive only costs
    # the compile and import checks, never the build.
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except Exception as error:  # noqa: BLE001
            print("fetch_sources: attempt %d failed: %r" % (attempt + 1, error), file=sys.stderr)
            time.sleep(3 * (attempt + 1))
    return None


def main():
    commit, target = sys.argv[1], sys.argv[2]
    data = download(commit)
    if data is None:
        print("fetch_sources: giving up; the selector will run without sources", file=sys.stderr)
        return
    kept = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            _, _, rel = member.name.partition("/")
            if not rel.endswith(".py") or ".." in rel.split("/") or rel.startswith("/"):
                continue
            if not (rel.startswith(KEEP) or "/" not in rel):
                continue
            path = target.rstrip("/") + "/" + rel
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as out:
                out.write(archive.extractfile(member).read())
            kept += 1
    print("fetch_sources: kept %d Python files from %s" % (kept, commit), file=sys.stderr)


if __name__ == "__main__":
    main()
