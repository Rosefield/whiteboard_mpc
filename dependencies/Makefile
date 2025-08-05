
all: mpc_runner_2

mpc_runner_2: mpc_runner.cpp
	cat mpc_runner.cpp | sed 's/NUM_PARTIES = 3/NUM_PARTIES = 3/' > mpc_runner_3.cpp
	clang++ -std=c++17 -pthread -lcrypto -O2 -march=native mpc_runner_3.cpp -o mpc_runner_3

test: all
	./mpc_runner_3 1 "parties.txt" "inputs_1.txt" "outputs_1.txt" "circuit.txt" &
	./mpc_runner_3 2 "parties.txt" "inputs_2.txt" "outputs_2.txt" "circuit.txt" &
	./mpc_runner_3 3 "parties.txt" "inputs_3.txt" "outputs_3.txt" "circuit.txt"

test2: all
	./mpc_runner_3 3 ../tmp/parties_3_0.txt ../tmp/input_3_0.txt ../tmp/output_3_0.txt ../tmp/circuit_3_0.txt &
	./mpc_runner_3 2 ../tmp/parties_2_0.txt ../tmp/input_2_0.txt ../tmp/output_2_0.txt ../tmp/circuit_2_0.txt &
	gdb --args ./mpc_runner_3 1 ../tmp/parties_1_0.txt ../tmp/input_1_0.txt ../tmp/output_1_0.txt ../tmp/circuit_1_0.txt 

clean:
	rm mpc_runner_2 mpc_runner_3
