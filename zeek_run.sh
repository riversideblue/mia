#!/usr/bin/env bash
set -eu

PCAP="$(realpath ${1:? "pcap file required"})"

OUT_DIR="$(realpath ./data/logs/$(basename "$PCAP" .pcap))"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

zeek -r "$PCAP" LogAscii::use_json=T