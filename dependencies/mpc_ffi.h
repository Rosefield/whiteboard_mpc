
#pragma once
#include "rust/cxx.h"
#include <emp-tool/emp-tool.h>
#include <emp-ot/emp-ot.h>
#include <unordered_map>

using emp::BristolFormat;
using emp::block;



using BlockSlice = rust::Slice<std::array<uint8_t, 16>>;

typedef uint64_t CreateFn(std::unordered_map<int, block*>&, std::unordered_map<int, block*>&, const bool*, int);

struct EmpAbit;
struct IknpOte;
struct Function;
struct Network;

#include "whiteboard_mpc/src/ffi.rs.h"

struct Function {
    std::function<void(void)> f;

    Function(std::function<void(void)>&& _f): f(std::move(_f)) {}
};

template<typename F, class... Args>
auto rs_schedule(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
    
    using return_type = typename std::result_of<F(Args...)>::type;

    // packaged_task is move-only and std::function requires copy-constructable
    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
    std::future<return_type> res = task->get_future();

    std::function<void(void)> type_erased {[t = std::move(task)](){ (*t)(); }};
    
    auto fn = std::make_unique<Function>(std::move(type_erased));
    
    spawn(std::move(fn));

    return res;
}

typedef std::unordered_map<int, std::unique_ptr<NetIO>> IOS;

struct Network {
    Network(IOS&& ios, IOS&& ios2): _ios(std::move(ios)), _ios2(std::move(ios2)) {}

    IOS _ios;
    IOS _ios2;
};

struct EmpAbit {
    public:
        EmpAbit(uint16_t id, int n, int ssp,
            std::function<CreateFn>&& create): _id(id), _n(n), _ssp(ssp), _create(create) {} 

        uint64_t create_abits(rust::Slice<const bool> bits, rust::Slice<BlockSlice> out_macs, rust::Slice<BlockSlice> out_keys) const {
            std::unordered_map<int, block *> macs;
            std::unordered_map<int, block *> keys;

            auto nbits = bits.size();

            // abitmp is expecting an extra 3*SSP bits 
            uint16_t np = _n -1;
            assert(nbits > 3*_ssp);
            assert(out_macs.size() == np);
            assert(out_macs[0].size() == nbits);
            assert(out_keys.size() == np);
            assert(out_keys[0].size() == nbits);
            assert(sizeof(block) == sizeof(std::array<uint8_t, 16>));

            auto j = 0;
            for(auto i = 1; i <= _n; i++) {
                if (i != _id) {
                    // We are already violating all kinds of type safety here.
                    // The actual input is a Vec<Vec<T>> for some field F that is of size 16 bytes
                    macs[i] = ((block *) out_macs[j].data());
                    keys[i] = ((block *) out_keys[j].data());
                    j += 1;
                }
            }

            // store a function so that we don't need to reference emp_agmpc here itself
            // to prevent multiple-definition linker errors with cxx that I can't figure out
            return _create(macs, keys, bits.data(), nbits);
        }

        uint16_t _id;
        uint16_t _n;
        uint32_t _ssp;
        std::function<CreateFn> _create; 
};

class IoView {
    public: 
    IoView(int32_t dst, const Network& net, bool first = true) 
    { 
        if (first) {
            this->_net = net._ios.at(dst).get();
        } else {
            this->_net = net._ios2.at(dst).get();
        }
    }

    void send_block(const block * blocks, size_t len) {
        this->send_data(blocks, len*sizeof(block));
    }

    void send_data(const void * data, size_t len) {
        this->counter += len;
        
        this->_net->send_data(data, len);
    }

    void recv_block(block * blocks, size_t len) {
        this->recv_data(blocks, len*sizeof(block));
    }

    void recv_data(void * data, size_t len) {
        this->counter += len;

        this->_net->recv_data(data, len);
    }

    // because this is part of the io channel apparently used by OT
    void send_pt(Point *A, size_t num_pts = 1) {
            for(size_t i = 0; i < num_pts; ++i) {
                    size_t len = A[i].size();
                    A[i].group->resize_scratch(len);
                    unsigned char * tmp = A[i].group->scratch;
                    send_data(&len, 4);
                    A[i].to_bin(tmp, len);
                    send_data(tmp, len);
            }
    }

    void recv_pt(Group * g, Point *A, size_t num_pts = 1) {
            size_t len = 0;
            for(size_t i = 0; i < num_pts; ++i) {
                    // could probably remove this since it is duplicative with the func_net networking
                    recv_data(&len, 4);
                    assert(len <= 2048);
                    g->resize_scratch(len);
                    unsigned char * tmp = g->scratch;
                    recv_data(tmp, len);
                    A[i].from_bin(g, tmp, len);
            }
    }	

    void flush() {
        this->_net->flush();

    }
    // not used in theory
    void sync() {
        assert(false);
    }

    int64_t counter = 0;

    private:
        NetIO * _net;
};

struct IknpOte {
    IknpOte(std::unique_ptr<IoView>&& io, bool isSender, std::array<uint8_t, 16> delta): io(std::move(io))
    {

        this->cot = std::make_unique<IKNP<IoView>>(this->io.get(), true);
        
        if (isSender) {
            std::array<bool, 128> delta_b;
            for (int i = 0; i < 128; i++) {
                auto j = i / 8;
                auto k = i % 8;
                delta_b[i] = ((delta[j] >> k) & 1) == 1;
            }
            this->cot->setup_send(delta_b.data());
            this->io->flush();
        } else {
            this->cot->setup_recv();
            this->io->flush();
        }
    }

    void ote_extend_send_rand(BlockSlice out_corr) const {
        this->cot->send_cot((block*) out_corr.data(), out_corr.size());
        this->io->flush();
    }

    void ote_extend_recv_rand(rust::Slice<const bool> selection, BlockSlice out_block) const {
        this->cot->recv_cot((block*) out_block.data(), selection.data(), out_block.size());
        this->io->flush();
    }

    uint64_t net_stat() const {
        return this->io->counter;
    }
    
    private:
        // Members are pointers instead of the actual objects so that we can have
        // const function signatures for CXX with none of the thread safety #JustC++Things
        std::unique_ptr<IoView> io;
        std::unique_ptr<IKNP<IoView>> cot;

};

void call(std::unique_ptr<Function> f);

std::shared_ptr<Network> make_network(uint16_t my_id,
    rust::Slice<const PartyInfo> parties,
    uint16_t port_offset
);

std::shared_ptr<IknpOte> make_ot_player(uint16_t my_id,
    uint16_t other_id,
    std::shared_ptr<Network> net,
    bool is_sender,
    std::array<uint8_t, 16> delta
);

std::shared_ptr<EmpAbit> make_abit_player(uint16_t my_id,
    rust::Slice<const uint16_t> parties,
    std::shared_ptr<Network> net,
    rust::Slice<const bool> delta
);


MpcOut run_mpc(uint16_t my_id,
    rust::Slice<const uint16_t> parties,
    std::shared_ptr<Network> net,
    rust::Vec<bool> delta,
    rust::Vec<int32_t> input_assignments,
    rust::Vec<bool> input,
    rust::Vec<Abit> auth_input,
    rust::Vec<int32_t> output_assignments,
    std::unique_ptr<BristolFormat> circuit
);

std::unique_ptr<BristolFormat> make_bristol_circuit(int32_t num_inputs,
    int32_t num_outputs,
    rust::Vec<Gate> gates
);

