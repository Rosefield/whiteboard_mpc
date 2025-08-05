#!/bin/bash

killall benchmarks &> /dev/null

set -e

cargo build --release --example benchmarks

N=3
T=2

for ((i = 2; i <= $N; i++))
do
#    target/release/examples/benchmarks -m $i -t $T -p dependencies/parties_$N.txt 1> /dev/null &
#    target/release/examples/benchmarks -m $i -t $T -p dependencies/parties_$N.txt -s tmp/state_$i.json 1> /dev/null &
    target/release/examples/benchmarks -m $i -t $T -p dependencies/parties_$N.txt --use-generic 1> /dev/null &
done

#PERF=/usr/lib/linux-tools/5.4.0-153-generic/perf flamegraph --no-inline -F 1500 --  target/release/examples/benchmarks -m 1 -t $T -p dependencies/parties_$N.txt
#target/release/examples/benchmarks -m 1 -t $T -p dependencies/parties_$N.txt 
#target/release/examples/benchmarks -m 1 -t $T -p dependencies/parties_$N.txt -s tmp/state_1.json
target/release/examples/benchmarks -m 1 -t $T -p dependencies/parties_$N.txt --use-generic
