#!/bin/bash

# make sure all containers are reset 
echo "stopping all containers"
seq 105 112 | xargs -n 1 pct stop 
echo "starting all containers"
seq 105 112 | xargs -n 1 pct start 

# make sure the expected files are available

echo "copying required files"
seq 105 112 | xargs -i pct push {} ./benchmarks ~/benchmarks
seq 105 112 | xargs -i pct push {} ./mpc_runner_3 ~/dependencies/mpc_runner_3
seq 105 112 | xargs -i pct push {} ./mpc_runner_4 ~/dependencies/mpc_runner_4
seq 105 112 | xargs -i pct push {} ./mpc_runner_5 ~/dependencies/mpc_runner_5
seq 105 112 | xargs -i pct push {} ./mpc_runner_8 ~/dependencies/mpc_runner_8
seq 105 112 | xargs -i pct push {} ./parties_8.txt ~/parties_8.txt
