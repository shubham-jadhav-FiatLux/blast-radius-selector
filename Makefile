IMAGE ?= blast-radius-selector
PROFILE_IMAGE ?= blast-radius-profile
KIT ?= ../blast-radius-kit
SQLGLOT ?= ../sqlglot
OUT ?= out
SEED ?= 0

.PHONY: build sample profile map reference sources stages

# The submission image.
build:
	docker build --platform linux/amd64 --tag $(IMAGE) .

# Runs the image under the scorer's limits on the kit's sample commits and scores it.
sample: build
	mkdir -p $(OUT)
	eval/run_image.sh $(IMAGE) $(KIT)/sample $(OUT)/sample.jsonl $(KIT)/score.py
	python3 eval/breakdown.py $(KIT)/sample $(KIT)/sample/answers.json $(OUT)/sample.jsonl

# Runs the sqlglot suite at the pinned commit under the tracing plugin: a coverage pass and a
# weak mutation pass. Repeat with another SEED into another OUT and list both runs in PROFILES
# and WEAK to union runs made under different hash seeds.
profile:
	docker build --platform linux/amd64 --tag $(PROFILE_IMAGE) profiling
	profiling/run_profile.sh $(PROFILE_IMAGE) $(SQLGLOT) $(OUT)/profile $(SEED)

PROFILES ?= $(OUT)/profile/trace_main.txt.gz $(OUT)/profile/trace_max_nodes.txt.gz
WEAK ?= $(OUT)/profile/weak_main.txt.gz $(OUT)/profile/weak_max_nodes.txt.gz
WEAK_OUTCOMES ?= $(OUT)/profile/weak_main.json $(OUT)/profile/weak_max_nodes.json
OUTCOMES ?= $(OUT)/profile/trace_main.json $(OUT)/profile/trace_max_nodes.json

# Reference results of each TPC-DS query of test_execute_tpcds, for single-query re-checks.
reference:
	docker run --rm --user $$(id -u):$$(id -g) -v $(abspath $(SQLGLOT)):/src:ro -v $(CURDIR):/repo -w /repo $(PROFILE_IMAGE) \
		python -S src/tpcds_check.py /src data/tpcds_reference.json

# Per TPC-DS query of test_execute_tpcds: the lines run before and after its plan, and in which step.
stages:
	docker run --rm --user $$(id -u):$$(id -g) -e PYTHONHASHSEED=0 -v $(abspath $(SQLGLOT)):/src:ro -v $(CURDIR):/repo -w /repo $(PROFILE_IMAGE) \
		python profiling/tpcds_stages.py /src data/tpcds_stages.json

# Turns the profiles and the pinned sources into data/map.bin.
map:
	docker run --rm --user $$(id -u):$$(id -g) -v $(abspath $(SQLGLOT)):/src:ro -v $(CURDIR):/repo -w /repo $(PROFILE_IMAGE) \
		python profiling/build_map.py --src /src --out data/map.bin \
		--profile $(PROFILES) --keys $(PROFILES) --weak $(WEAK) --weak-outcomes $(WEAK_OUTCOMES) --outcomes $(OUTCOMES) \
		--tpcds data/tpcds_reference.json --tpcds-stages data/tpcds_stages.json

# The pinned sources the image checks patches against (sqlglot's Python files, its SQL fixtures and
# the TPC-DS data).
sources:
	cd $(SQLGLOT) && git ls-files '*.py' LICENSE tests/fixtures/optimizer/tpc-ds 'tests/fixtures/*.sql' | sort -u | \
		tar cJf $(CURDIR)/data/sqlglot-src.tar.xz --owner=0 --group=0 --mtime=2026-01-01 -T -
