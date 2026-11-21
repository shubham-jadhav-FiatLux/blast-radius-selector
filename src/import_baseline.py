"""Build step: records the sqlglot modules that fail to import in this image without any patch.

    python -S -B import_baseline.py <pinned sources> <output file>

The selector's import check runs under the same interpreter flags and skips these modules, so a
module that needs an optional dependency is not mistaken for one a patch broke.
"""

import importlib
import os
import pkgutil
import sys


def main():
    root, out = sys.argv[1], sys.argv[2]
    failed = []
    if os.path.isdir(os.path.join(root, "sqlglot")):
        sys.path.insert(0, root)
        import sqlglot

        for info in pkgutil.walk_packages(sqlglot.__path__, "sqlglot.", onerror=failed.append):
            if info.name.endswith(".__main__"):
                continue
            try:
                importlib.import_module(info.name)
            except BaseException:  # noqa: BLE001
                failed.append(info.name)
    with open(out, "w") as f:
        f.write(",".join(sorted(set(failed))))
    print("import_baseline: %d modules fail without a patch: %s" % (len(failed), failed[:5]), file=sys.stderr)


if __name__ == "__main__":
    main()
