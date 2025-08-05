#include <emp-tool/emp-tool.h>
#include "emp_agmpc/emp-agmpc.h"
#include "emp_agmpc/flexible_input_output.h"
#include "whiteboard_mpc/dependencies/mpc_ffi.h"

#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <atomic>

using namespace emp;

const int SSP = 80;

void call(std::unique_ptr<Function> f)
{
    f->f();
}

class RsPool {
    public:
    template<typename F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type> {
        return rs_schedule(f, args...);
    }
};

std::shared_ptr<Network> make_network(uint16_t my_id,
    rust::Slice<const PartyInfo> party_info,
    uint16_t port_offset
) {
    std::unordered_map<int32_t, std::unique_ptr<NetIO>> ios;
    std::unordered_map<int32_t, std::unique_ptr<NetIO>> ios2;

    const auto& [i, ip_i, port_i] = party_info[my_id-1];
    const auto pi = port_i + port_offset;
    assert(i == my_id);


    for(const auto& [j, ip_j, port_j]: party_info) {
        if (i == j) { continue; }

        const auto pj = port_j + port_offset;
        std::string ip{ip_j};
        if(my_id < j) {
            usleep(1000);
            ios[j] = std::make_unique<NetIO>(ip.c_str(), pj+2*i, true);
            ios[j]->set_nodelay();  
            usleep(1000);
            ios2[j] = std::make_unique<NetIO>(nullptr, pi+2*j+1, true);
            ios2[j]->set_nodelay();  
        } else {
            usleep(1000);
            ios[j] = std::make_unique<NetIO>(nullptr, pi+2*(j), true);
            ios[j]->set_nodelay();  
            usleep(1000);
            ios2[j] = std::make_unique<NetIO>(ip.c_str(), pj+2*i+1, true);
            ios2[j]->set_nodelay();  

            usleep(1000);
        }
    }

    return std::make_shared<Network>(std::move(ios), std::move(ios2));
}

std::shared_ptr<IknpOte> make_ot_player(uint16_t my_id,
    uint16_t other_id,
    std::shared_ptr<Network> net,
    bool is_sender,
    std::array<uint8_t, 16> delta
) {

    auto first = my_id < other_id ^ is_sender;
    // 7 == FuncId::Fcote
    return std::make_shared<IknpOte>(std::make_unique<IoView>(other_id, *net, false), is_sender, delta);
}

std::shared_ptr<EmpAbit> make_abit(int32_t my_id, 
    const std::vector<int32_t>& parties,
    std::unordered_map<int32_t, int32_t>& id_remap,
    std::shared_ptr<Network> net,
    rust::Slice<const bool> delta) {

    std::unordered_map<int32_t, IoView> ios;
    std::unordered_map<int32_t, IoView> ios2;

    const uint32_t N = parties.size();

    for(const auto p: parties) {
        if (p != my_id) {
            ios.insert({id_remap[p], IoView(p, *net, true)});
            ios2.insert({id_remap[p], IoView(p, *net, false)});
        }
    }

    my_id = id_remap.at(my_id);
    
    // UNSAFE: going to pretend that this can handle more than 1 copy at a time
    auto io = std::make_shared<NetIOMP<IoView>>(my_id, ios, ios2);
    //auto pool = std::make_shared<RsPool>();
    auto pool = std::make_shared<ThreadPool>(2*N);
    // Largely taken from emp-agmpc/test_in_out
    auto abit = std::make_shared<ABitMP<ThreadPool, IoView>>(io.get(), pool.get(), N, my_id, (bool *)delta.data(), SSP);

    std::function<CreateFn> create { [my_id = my_id, abit = std::move(abit), pool = std::move(pool), io = std::move(io)](std::unordered_map<int, block*>& macs, std::unordered_map<int, block*>& keys, const bool* data, int len) {
        abit->compute(macs, keys, data, len);
        auto f = abit->check(macs, keys, data, len);
        f.get();
        return io->count();
    } }; 

    return std::make_shared<EmpAbit>(my_id, N, SSP, std::move(create));
}


std::shared_ptr<EmpAbit> make_abit_player(uint16_t my_id,
    rust::Slice<const uint16_t> parties,
    std::shared_ptr<Network> net,
    rust::Slice<const bool> delta) {
    std::unordered_map<int, int> id_remap {};
    std::vector<int32_t> ids;
    for(uint16_t i = 1; i <= parties.size(); i++) {
        const auto& id = parties[i-1];
        id_remap[id] = i;
        ids.push_back(id);
    }

    return make_abit(my_id, ids, id_remap, std::move(net), delta);
}

std::tuple<std::vector<bool>, std::vector<uint8_t>, uint64_t> run_n_mpc(int32_t my_id, 
    const std::vector<int32_t>& party_info,
    std::unordered_map<int32_t, int32_t>& id_remap,
    std::shared_ptr<Network> net,
    bool * delta,
    const std::vector<int32_t>& input_assignment,
    const std::vector<bool>& input,
    const std::vector<Abit>& auth_input,
    const std::vector<int32_t>& output_assignment,
    BristolFormat& circuit
) {

    auto num_inputs = circuit.n1 + circuit.n2;
    auto num_outputs = circuit.n3;

    std::unordered_map<int32_t, IoView> is;
    std::unordered_map<int32_t, IoView> is2;

    const int32_t N = party_info.size();

    for(const auto p: party_info) {
        if (p != my_id) {
            is.insert({id_remap[p], IoView(p, *net, true)});
            is2.insert({id_remap[p], IoView(p, *net, false)});
        }
    }

    my_id = id_remap[my_id];

    NetIOMP<IoView> io(my_id, is, is2);
    NetIOMP<IoView> io2(my_id, is, is2);
    
    NetIOMP<IoView> *ios[2] = {&io, &io2};
    //RsPool pool{};
    ThreadPool pool(2*N);


    // Largely taken from emp-agmpc/test_in_out
    CMPC<ThreadPool, IoView> mpc(ios, &pool, N, my_id, &circuit, delta, SSP);

    ios[0]->flush();
    ios[1]->flush();
    
    mpc.function_independent();
    ios[0]->flush();
    ios[1]->flush();
    
    mpc.function_dependent();
    ios[0]->flush();
    ios[1]->flush();
    
    FlexIn<ThreadPool, IoView> flexIn(num_inputs, N, my_id);
    auto j = 0, k = 0;
    for (auto i = 0; i < num_inputs; i++) {
        const auto& p = input_assignment[i];
        flexIn.assign_party(i, p);

        // my / public input
        if (my_id == p || p == 0) {
            flexIn.assign_plaintext_bit(i, input[j]);
            j += 1;
        }

        // auth input 
        if (p == -1) {
            AuthBitShare abit {};
            abit.bit_share = auth_input[k].bit;
            abit.key.resize(N); 
            abit.mac.resize(N); 
            for (auto x = 1, y = 0; x <= N; x++) {
                if (x == my_id) { continue; }
                abit.key[x-1] = *(block*) &auth_input[k].keys[y];
                abit.mac[x-1] = *(block*) &auth_input[k].macs[y];
                y++;
            }
        
            flexIn.assign_authenticated_bitshare(i, abit);
            k += 1;
        }
    }
    
    FlexOut<ThreadPool, IoView> flexOut(num_outputs, N, my_id);
    auto auth_outs = 0;
    for (auto i = 0; i < num_outputs; i++) {
        const auto& p = output_assignment[i];
        flexOut.assign_party(i, p);
        if (i == -1) {
            auth_outs += 1;
        }
    }
    
    mpc.online(&flexIn, &flexOut);
    ios[0]->flush();
    ios[1]->flush();

    std::vector<bool> output;
    std::vector<uint8_t> abit_bytes;


    // abit can be represented in size 1 + 32*N bytes
    // but with alignment it is 16*(2n + 1) bytes
    auto sbytes = 1 + 32*N;
    abit_bytes.reserve(auth_outs * sbytes);

    for (auto i = 0; i < num_outputs; i++) {
        const auto& p = output_assignment[i];
        if (p == my_id || p == 0) {
            auto bit = flexOut.get_plaintext_bit(i);
            output.push_back(bit);
        }
        if (p == -1) {
            auto abit = flexOut.get_authenticated_bitshare(i);
            abit_bytes.push_back(abit.bit_share);

            auto pkey = (uint8_t *)abit.key.data();
            abit_bytes.insert(abit_bytes.end(), pkey, pkey+(N*sizeof(block)));

            auto pmac = (uint8_t *)abit.mac.data();
            abit_bytes.insert(abit_bytes.end(), pmac, pmac+(N*sizeof(block)));
        }
    }

    auto net_bytes = ios[0]->count() + ios[1]->count();

    return { output, abit_bytes, net_bytes };
}


MpcOut run_mpc(uint16_t my_id,
    rust::Slice<const uint16_t> parties,
    std::shared_ptr<Network> net,
    rust::Vec<bool> delta,
    rust::Vec<int32_t> input_assignments,
    rust::Vec<bool> inputs,
    rust::Vec<Abit> auth_inputs,
    rust::Vec<int32_t> output_assignments,
    std::unique_ptr<BristolFormat> circuit
) {

    // need to remap the party ids of the full protocol to sequential
    // ids for emp-tool
    std::unordered_map<int, int> id_remap {};
    std::vector<int32_t> party_info;
    for(uint16_t i = 1; i <= parties.size(); i++) {
        const auto& id = parties[i-1];
        id_remap[id] = i;
        party_info.emplace_back(id);
    }

    std::vector<int> input_assignment;
    for (auto id: input_assignments) {
        if (id > 0) { id = id_remap[id]; }
        input_assignment.push_back(id);
    }

    std::vector<bool> input;
    std::copy(inputs.begin(), inputs.end(), std::back_inserter(input));

    std::vector<Abit> auth_input;
    std::copy(auth_inputs.begin(), auth_inputs.end(), std::back_inserter(auth_input));

    std::vector<int> output_assignment;
    for (auto id: output_assignments) {
        if (id > 0) { id = id_remap[id]; }
        output_assignment.push_back(id);
    }

    std::vector<bool> out;
    std::vector<uint8_t> abit_out;
    uint64_t net_bytes;

    bool * pdelta = delta.data();
    if (delta.size() == 0) {
        pdelta = nullptr;
    }

    std::tie(out, abit_out, net_bytes) = run_n_mpc(my_id, party_info, id_remap, std::move(net), pdelta, input_assignment, input, auth_input, output_assignment, *circuit);

    MpcOut output;
    std::copy(out.begin(), out.end(), std::back_inserter(output.outs));

    std::copy(abit_out.begin(), abit_out.end(), std::back_inserter(output.auth_outs));

    output.bytes_sent = net_bytes;
    
    return output;
}

std::unique_ptr<BristolFormat> make_bristol_circuit(int32_t num_inputs,
    int32_t num_outputs,
    rust::Vec<Gate> gates
) {
    auto num_gate = gates.size();
    auto num_wire = num_gate + num_inputs;
    return std::make_unique<BristolFormat>(num_gate, num_wire, num_inputs, 0, num_outputs, (int *)gates.data());
}


/*

// Read the list of (party_id, ip) network configuration
std::vector<std::tuple<int, std::string, uint16_t>> parse_party_file(char * file_name) {
    std::string buf;
    std::vector<std::tuple<int, std::string, uint16_t>> res{};
    std::ifstream file(file_name);

    while(std::getline(file, buf)) {
        const auto split1 = buf.find(",");
        const auto split2 = buf.find(",", split1);
        const auto id = std::stoi(buf.substr(0, split1));
        const auto ip = buf.substr(split1+1, split2);
        const auto port = std::stoi(buf.substr(split2+1));
        res.emplace_back(id, ip, port);
    }
    
    return res;
}

// Read the split of input wires to each party, and the input for the current party
std::tuple<int, std::vector<int>, std::vector<bool>> parse_inputs_file(char * file_name, std::unordered_map<int,int>& id_remap) {
    std::string buf;
    std::vector<int> party_wires;
    std::ifstream file(file_name);

    // first line is the total number of wires
    int count;
    file >> count >> std::ws;

    // second line is the assignment of each input wire to a party
    // space separated
    std::getline(file, buf);
    std::istringstream inputs(buf);

    std::string party;
    while(std::getline(inputs, party, ' ')) {
        // convert any party assignments to the emp-tool expected party ids
        auto id = std::stoi(party);
        if (id > 0) { id = id_remap[id]; };
        party_wires.emplace_back(id);
    }

    // final line is the input for the party, just a series of 0,1s
    std::getline(file, buf);
    std::vector<bool> party_input;
    for (auto& c: buf) {
        party_input.push_back(c == '1');
    }
    
    return {count, party_wires, party_input};
}

// Read the number of output wires, and the assignment of output wires to each party
std::tuple<int, std::vector<int>> parse_outputs_file(char * file_name, std::unordered_map<int,int>& id_remap) {
    std::string buf;
    std::vector<int> party_wires;
    std::ifstream file(file_name);

    // first line is the total number of wires
    int count;
    file >> count >> std::ws;

    // second line is the assignment of each input wire to a party
    // space separated
    std::getline(file, buf);
    std::istringstream inputs(buf);

    std::string party;
    while(std::getline(inputs, party, ' ')) {
        auto id = std::stoi(party);
        if (id > 0) { id = id_remap[id]; };
        party_wires.emplace_back(id);
    }

    return {count, party_wires};
}

// emp-tool is parameterized by the number of parties so can't be dynamic
const int NUM_PARTIES = 3;


int mpc_main(int argc, char** argv) {
    //program my_id, party_file, input_file, output_file, circuit_file
    assert(argc == 6);

    auto party_info = parse_party_file(argv[2]);
    assert(party_info.size() == NUM_PARTIES);

    // need to remap the party ids of the full protocol to sequential
    // ids for emp-tool
    std::unordered_map<int, int> id_remap {};
    for(uint16_t i = 1; i <= party_info.size(); i++) {
        const auto& [id, ip, port] = party_info[i-1];
        id_remap[id] = i;
        party_info[i-1] = {i, ip, port};
    }

    const auto original_id = std::stoi(argv[1]);
    const auto my_id = id_remap[original_id];
    assert(my_id > 0 && my_id <= NUM_PARTIES);

    const auto [num_inputs, input_assignments, input] = parse_inputs_file(argv[3], id_remap);
    assert((int)input_assignments.size() == num_inputs);

    const auto [num_outputs, output_assignments] = parse_outputs_file(argv[4], id_remap);
    assert((int)output_assignments.size() == num_outputs);

    BristolFormat circuit(argv[5]);
    assert(circuit.n1 + circuit.n2 == num_inputs);
    assert(circuit.n3 == num_outputs);

    std::vector<Abit> auth_input;

    const auto [ output, abits, net_bytes ] = run_n_mpc(my_id, 3, party_info, nullptr, input_assignments, input, auth_input, output_assignments, circuit);
    
    for (const auto bit: output) {
        //std::cout << my_id << " open " << w << " " << bit << std::endl;
        std::cout << bit;
    }

    return 0;
}

*/
