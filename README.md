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
   per test:
   - the lines it executes (via `sys.monitoring`, so the overhead stays near 5%);
   - the files it opens;
   - which expression arguments were missing whenever a node was validated;
   - which class attributes it reads, keyed by the class that defines them. Every data
     attribute of every sqlglot class is swapped for a descriptor that returns the same object
     and notes the read. Dict attributes are served as copies that also note which keys are
     looked up; any use that can touch every key (iteration, `items()`, copies, `**` spreads,
     comparisons) counts as reading them all.

   A second pass (`weak_rewrite.py`) observes the "changes the state" step for the most common
   edits. Every comparison and condition inside a sqlglot function is rewritten so that it
   evaluates the swapped form next to the real one: `a <= b` next to `a < b`, `a == b` next to
   `a is b`, `n + 1` and `n - 1` next to a constant `n`, and `x or y` next to `x and y` where
   only the truth value is used. For a chain like `a and b and c`, swapping any one operator
   leaves the result alone only when every operand is truthy, which is what each operand
   records; conditions split across lines are matched on the line holding the operator. Swaps that always change the result (`==` and `!=`, `in` and
   `not in`, `<` and `>=`, dropping a `not`) only record whether the expression was evaluated
   at all, because `a and b == c` can stop before it. The pass records, per test, which swaps would have
   produced a different result. It runs under several hash seeds, and the runs are merged by
   site. The program still sees only the original results, and all 1,620 tests
   still pass under the rewrite.

   A few details make the attribution trustworthy:
   - every sqlglot module is imported before collection, so lazily loaded dialects do not
     charge their import-time work to whichever test happened to touch them first;
   - while importing, each function line is tied to the module or class statement it ran for.
     sqlglot parses SQL templates and builds tables while its classes are created, so a parser
     line can matter at import time too;
   - class- and module-scoped fixtures (`setUpClass`) are recorded separately and charged to
     every test in their scope;
   - the optimizer and executor suites run cases in process pools, so pools are replaced by an
     in-process executor that still round-trips arguments and results through pickle, because
     pickling expressions runs sqlglot's own serialisation code.
2. **Index the sources** (`source_index.py`). For every line, it records:
   - whether the line runs at import time or inside a function;
   - the span of the statement around it;
   - whether it sits inside a string literal;
   - the `arg_types` entries, class attributes and default values it defines.

   It also records what each import-time statement and class-creation hook derives from what:
   `SHOW_TRIE` from `SHOW_PARSERS`, a dialect's `ESCAPED_SEQUENCES` from its tokenizer's
   `NUMERIC_ESCAPES`.
3. **Build the map** (`build_map.py`, `import_effects.py`) into `data/map.bin`, about 0.6 MB.
   Test sets are stored as integer bitsets and deduplicated.

At run time `src/blast_select.py` (standard library only) parses each patch and classifies every
edited block:

| Edit | Tests charged |
| --- | --- |
| comment or blank line outside a string literal (Python files only) | none |
| a comparison, constant in a comparison, condition or `not` swapped for one the weak pass observed | the tests in which that swap would have changed a result; an `is` that can never differ from `==` for the values actually seen costs nothing |
| any other edit inside a function body | every test that ran any line of the enclosing statement (with `and`/`or` short-circuiting, a test can depend on a statement without running the edited line), plus the readers of anything that code computed while sqlglot was being imported |
| inside a lambda that an import-time table holds | the tests that called it |
| `arg_types` flag flipped | tests that validated a node of an affected class while that argument was missing (`arg_types` values only feed `required_args`); affected classes are the class itself, its subclasses, and classes splicing in its dict |
| the value of one entry of a class-level dict | tests that looked up that key or used the whole dict, on the class and on dicts spreading it; tables built only from its keys (`SHOW_TRIE` from `SHOW_PARSERS`, `"x" in NUMERIC_ESCAPES` in a hook) are unaffected |
| other class attribute or module constant (including whole entries added to or removed from a table) | tests that read it, or anything derived from it, on the defining class, its subclasses, or the dialects built from them; `{**Parser.FUNCTIONS}` resolves to `Parser`, not to whichever parser happens to be nearby |
| default value of a function argument | tests that ran the function |
| module level code in `tests/` | tests in that module and in every module importing it |
| data file (SQL fixtures) | the tests that opened it |
| anything that no longer compiles, config files, deletions and renames | all tests |

When an edited value is computed at import time by code that could fail on the new value, the
patched modules and their importers are imported again over the pinned sources, in a forked
child of the selector, and a failure is confirmed in a fresh interpreter. If they no longer
import, all tests run. One test imports sqlglot in a subprocess, out of the profile's sight; it
rides along whenever such an edit touches a module that `import sqlglot` itself loads.

The chosen tests are then packed: longest-processing-time first for every worker count from 1
to 8, followed by a little local search, minimising the busiest worker plus 2 s per worker.

## Results on the sample

On the kit's 30 sample commits: **102.1 s simulated, 0 misses**, using well under 1 s of CPU
for the whole batch. For comparison, running everything costs 1,104 s, and an oracle that knows the
failing tests costs 94.4 s.

`eval/mutants.py` generates synthetic edits with the same operators, and `eval/label.py` labels
them by running the full suite on each, which gives a test bed far beyond the 30 samples. A
failure counts only if it happens again when the failing tests are re-run alone, because a
parallel run on a loaded machine can lose a process-pool worker. `eval/check_labels.py` scores
a selector output against those labels.

`tests/` holds unit tests for the selector's building blocks: `python -m unittest discover tests`.

## Running it

```sh
make sample KIT=path/to/kit          # build, run under the scorer's limits, score, per-commit table
make profile SQLGLOT=path/to/sqlglot # re-profile the suite at the pinned commit (both passes)
make map SQLGLOT=path/to/sqlglot     # rebuild data/map.bin from those profiles
```

`KIT` is the unpacked starter kit (it holds `sample/` and `score.py`). `SQLGLOT` is a checkout
of sqlglot at the pinned commit.

## Layout

```
Dockerfile              submission image: selector, map, pinned sources for compile checks
src/blast_select.py     the selector
src/fetch_sources.py    build-time download of the pinned sqlglot sources
src/import_baseline.py  build-time list of modules that fail to import even unpatched
data/map.bin            the map built from the pinned commit
profiling/              plugins (tracing, weak mutation), source indexer and map builder
eval/                   local scoring, synthetic edits and labelling
```

## License

MIT
