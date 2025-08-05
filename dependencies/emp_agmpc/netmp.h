#ifndef NETIOMP_H__
#define NETIOMP_H__
#include <emp-tool/emp-tool.h>
#include "cmpc_config.h"
#include <unordered_map>
using namespace emp;


template<typename IO>
class NetIOMP { public:
    std::unordered_map<int, IO> ios;
    std::unordered_map<int, IO> ios2;
    int party;
    int nP;
    std::unordered_map<int, bool> sent;

    NetIOMP(int party, std::unordered_map<int, IO> ios, std::unordered_map<int, IO> ios2) {
        this->party = party;
        this->nP = ios.size()+1;
        this->ios = ios;
        this->ios2 = ios2;
        for(const auto& [k,v]: ios) {
            sent[k] = false;
        }
    }
    int64_t count() {
        int64_t res = 0;
        for(int i = 1; i <= nP; ++i) if(i != party) {
            res += ios.at(i).counter;
            res += ios2.at(i).counter;
        }
        return res;
    }

    void send_data(int dst, const void * data, size_t len) {
        if(dst != 0 and dst!= party) {
            if(party < dst)
                ios.at(dst).send_data(data, len);
            else
                ios2.at(dst).send_data(data, len);
            sent[dst] = true;
        }
#ifdef __MORE_FLUSH
        flush(dst);
#endif
    }
    void recv_data(int src, void * data, size_t len) {
        if(src != 0 and src!= party) {
            if(sent[src])flush(src);
            if(src < party)
                ios.at(src).recv_data(data, len);
            else
                ios2.at(src).recv_data(data, len);
        }
    }
    IO* get(size_t idx, bool b = false){
        if (b)
            return &ios.at(idx);
        else return &ios2.at(idx);
    }
    void flush(int idx = 0) {
        if(idx == 0) {
            for(int i = 1; i <= nP; ++i)
                if(i != party) {
                    ios.at(i).flush();
                    ios2.at(i).flush();
                }
        } else {
            if(party < idx)
                ios.at(idx).flush();
            else
                ios2.at(idx).flush();
        }
    }
    void sync() {
        for(int i = 1; i <= nP; ++i) for(int j = 1; j <= nP; ++j) if(i < j) {
            if(i == party) {
                ios.at(j).sync();
                ios2.at(j).sync();
            } else if(j == party) {
                ios.at(i).sync();
                ios2.at(i).sync();
            }
        }
    }
};
#endif //NETIOMP_H__
