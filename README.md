# blast-radius-selector

Predictive test selection for [sqlglot](https://github.com/tobymao/sqlglot) at commit
`eb11aec4be8cb2122afe713a4816662c3f0fb2fa`, built for the Blast Radius track of the Builders
Buildathon. For every patch it prints the tests that can fail and how to split them over at
most 8 workers.

The guiding idea comes from mutation testing: a test can only fail if it **reaches** the
edited code, the edit actually **changes** the program's state, and that change **reaches** an
assertion. Plain coverage stops at the first step. The selector answers each edit with the
narrowest test set it can justify from the pinned commit, and falls back to something larger
whenever it cannot justify less.

## How it works

Everything expensive happens offline, against the pinned commit, in the same image the track
labelled its commits in (`profiling/`):

1. **Profile the suite.** A pytest plugin (`trace_plugin.py`) runs all 1,620 tests and records,
   per test, the lines it executes (via `sys.monitoring`, so the overhead stays near 5%), the
   files it opens, and which expression arguments were missing whenever a node was validated.
   A few details make the attribution trustworthy:
   - every sqlglot module is imported before collection, so lazily loaded dialects do not
     charge their import-time work to whichever test happened to touch them first;
   - class- and module-scoped fixtures (`setUpClass`) are recorded separately and charged to
     every test in their scope;
   - the optimizer and executor suites run cases in process pools, so pools are replaced by an
     in-process executor that still round-trips arguments and results through pickle, because
     pickling expressions runs sqlglot's own serialisation code.
2. **Index the sources** (`source_index.py`). For every line: whether it runs at import time or
   inside a function, the span of the statement around it, whether it sits inside a string
   literal, and the `arg_types` entries it defines.
3. **Build the map** (`build_map.py`) into `data/map.bin`, about 0.4 MB: test sets stored as
   integer bitsets, deduplicated.

At run time `src/blast_select.py` (standard library only) parses each patch and classifies every
edited block:

| Edit | Tests charged |
| --- | --- |
| comment or blank line outside a string literal (Python files only) | none |
| inside a function body | every test that ran any line of the enclosing statement; with `and`/`or` short-circuiting, a test can depend on a statement without running the edited line |
| inside a lambda that an import-time table holds | the tests that called it |
| `arg_types` flag flipped | tests that validated a node of an affected class (itself, subclasses, classes splicing its dict) while that argument was missing, because `arg_types` values only feed `required_args` |
| other module or class level code in `sqlglot/` | all tests |
| module level code in `tests/` | tests in that module and in every module importing it |
| data file (SQL fixtures) | the tests that opened it |
| anything that no longer compiles, config files, deletions and renames | all tests |

The chosen tests are then packed: longest-processing-time first for every worker count from 1
to 8, followed by a little local search, minimising the busiest worker plus 2 s per worker.
The whole batch takes about 0.2 s of CPU.

## Results on the sample

On the kit's 30 sample commits: **259.8 s simulated, 0 misses** (running everything costs
1,104 s, and an oracle that knows the failing tests costs 94.4 s). Most of the remaining gap is
module-level edits outside `arg_types`, which still run everything.

## Running it

```sh
make sample KIT=path/to/kit          # build, run under the scorer's limits, score, per-commit table
make profile SQLGLOT=path/to/sqlglot # re-profile the suite at the pinned commit
make map SQLGLOT=path/to/sqlglot     # rebuild data/map.bin from that profile
```

`KIT` is the unpacked starter kit (it holds `sample/` and `score.py`). `SQLGLOT` is a checkout
of sqlglot at the pinned commit.

## Layout

```
Dockerfile              submission image: selector, map, pinned sources for compile checks
src/blast_select.py     the selector
src/fetch_sources.py    build-time download of the pinned sqlglot sources
data/map.bin            the map built from the pinned commit
profiling/              plugins, source indexer and map builder
eval/                   local scoring helpers
```

## License

MIT
