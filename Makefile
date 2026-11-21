IMAGE ?= blast-radius-selector
PROFILE_IMAGE ?= blast-radius-profile
KIT ?= ../blast-radius-kit
SQLGLOT ?= ../sqlglot
OUT ?= out
SEED ?= 0

.PHONY: build sample profile map

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
OUTCOMES ?= $(OUT)/profile/trace_main.json $(OUT)/profile/trace_max_nodes.json

# Turns the profiles and the pinned sources into data/map.bin.
map:
	docker run --rm --user $$(id -u):$$(id -g) -v $(abspath $(SQLGLOT)):/src:ro -v $(CURDIR):/repo -w /repo $(PROFILE_IMAGE) \
		python profiling/build_map.py --src /src --out data/map.bin \
		--profile $(PROFILES) --weak $(WEAK) --outcomes $(OUTCOMES)
