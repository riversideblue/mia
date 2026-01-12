ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

ifneq ($(filter zeek-logs,$(MAKECMDGOALS)),)
ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
PCAP ?= $(firstword $(ARGS))
endif

.PHONY: run
run:
	@cd "$(ROOT_DIR)/src/main" && python3 Run.py

.PHONY: zeek-logs
zeek-logs:
	@[ -n "$(PCAP)" ] || { echo "PCAP path required (usage: make zeek-logs /path/to/file.pcap or make zeek-logs PCAP=/path/to/file.pcap)"; exit 1; }
	@PCAP_REAL="$$(realpath "$(PCAP)")"; \
	OUT_DIR="$(ROOT_DIR)/data/logs/$$(basename "$$PCAP_REAL" .pcap)"; \
	mkdir -p "$$OUT_DIR"; \
	OUT_DIR="$$(realpath "$$OUT_DIR")"; \
	cd "$$OUT_DIR" && zeek -r "$$PCAP_REAL" LogAscii::use_json=T

.PHONY: log-to-csv
log-to-csv:
	@[ -n "$(LOG_DIR)" ] || { echo "LOG_DIR required (usage: make log-to-csv LOG_DIR=/path/to/logs)"; exit 1; }
	@[ -n "$(OUT_CSV)" ] || { echo "OUT_CSV required (usage: make log-to-csv LOG_DIR=/path/to/logs OUT_CSV=/path/to/out.csv)"; exit 1; }
	@python3 "$(ROOT_DIR)/src/util/LogToCsvExtractor.py" "$(LOG_DIR)" "$(OUT_CSV)" \
		--pattern "$(PATTERN)" \
		$(if $(NETWORK_KEY),--network-key "$(NETWORK_KEY)",)
