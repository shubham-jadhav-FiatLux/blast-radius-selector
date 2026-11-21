IMAGE ?= blast-radius-selector
PROFILE_IMAGE ?= blast-radius-profile
KIT ?= ../blast-radius-kit
SQLGLOT ?= ../sqlglot
OUT ?= out

.PHONY: build sample profile map

# The submission image.
build:
	docker build --platform linux/amd64 --tag $(IMAGE) .

# Runs the image under the scorer's limits on the kit's sample commits and scores it.
sample: build
	mkdir -p $(OUT)
	eval/run_image.sh $(IMAGE) $(KIT)/sample $(OUT)/sample.jsonl $(KIT)/score.py
	python3 eval/breakdown.py $(KIT)/sample $(KIT)/sample/answers.json $(OUT)/sample.jsonl

# Runs the sqlglot suite at the pinned commit under the tracing plugins (about 3 minutes).
profile:
	docker build --platform linux/amd64 --tag $(PROFILE_IMAGE) profiling
	profiling/run_profile.sh $(PROFILE_IMAGE) $(SQLGLOT) $(OUT)/profile

# Turns the profile and the pinned sources into data/map.bin.
map:
	docker run --rm --user $$(id -u):$$(id -g) -v $(abspath $(SQLGLOT)):/src:ro -v $(CURDIR):/repo -w /repo $(PROFILE_IMAGE) \
		python profiling/build_map.py --src /src --out data/map.bin \
		--profile $(OUT)/profile/main.txt.gz $(OUT)/profile/max_nodes.txt.gz \
		--outcomes $(OUT)/profile/main.json $(OUT)/profile/max_nodes.json
