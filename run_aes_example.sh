#!/bin/bash

PROFILER='time'
#PROFILER='flamegraph --no-inline -F 1500 -- '
#PROFILER='valgrind --tool=callgrind'

cargo build --release --example aes

killall aes

target/release/examples/aes -m 1 -t 2 -p dependencies/parties.txt -r &> /dev/null &
target/release/examples/aes -m 2 -t 2 -p dependencies/parties.txt -r &> /dev/null &
time target/release/examples/aes -m 3 -t 2 -p dependencies/parties.txt -r
