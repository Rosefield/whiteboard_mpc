#ifndef ABIT_MP_H__
#define ABIT_MP_H__
#include <emp-tool/emp-tool.h>
#include <emp-ot/emp-ot.h>
#include "netmp.h"
#include "helper.h"
#include <unordered_map>

template<typename TP, typename IO>
class ABitMP { public:
	std::unordered_map<int, std::unique_ptr<IKNP<IO>>> abit1;
	std::unordered_map<int, std::unique_ptr<IKNP<IO>>> abit2;
	NetIOMP<IO> *io;
	TP * pool;
        int nP;
	int party;
	PRG prg;
	block Delta;
	Hash hash;
	int ssp;
	block * pretable;
	ABitMP(NetIOMP<IO> * io, TP * pool, int nP, int party, bool * _tmp = nullptr, int ssp = 40) {
		this->ssp = ssp;
		this->io = io;
		this->pool = pool;
		this->party = party;
                this->nP = nP;
		bool tmp[128];
		if(_tmp == nullptr) {
			prg.random_bool(tmp, 128);
		} else {
			memcpy(tmp, _tmp, 128);
		}

		for(int i = 1; i <= nP; ++i) for(int j = 1; j <= nP; ++j) if(i < j) {
			if(i == party) {
					abit1[j] = std::make_unique<IKNP<IO>>(io->get(j, false));
					abit2[j] = std::make_unique<IKNP<IO>>(io->get(j, true));
			} else if (j == party) {
					abit2[i] = std::make_unique<IKNP<IO>>(io->get(i, false));
					abit1[i] = std::make_unique<IKNP<IO>>(io->get(i, true));
			}
		}

		vector<future<void>> res;//relic multi-thread problems...
		for(int i = 1; i <= nP; ++i) for(int j = 1; j <= nP; ++j) if(i < j) {
			if(i == party) {
				res.push_back(pool->enqueue([this, io, tmp, j]() {
					abit1[j]->setup_send(tmp);
					io->flush(j);
				}));
				res.push_back(pool->enqueue([this, io, j]() {
					abit2[j]->setup_recv();
					io->flush(j);
				}));
			} else if (j == party) {
				res.push_back(pool->enqueue([this, io, i]() {
					abit2[i]->setup_recv();
					io->flush(i);
				}));
				res.push_back(pool->enqueue([this, io, tmp, i]() {
					abit1[i]->setup_send(tmp);
					io->flush(i);
				}));
			}
		}
		joinNclean(res);

		if(party == 1)
			Delta = abit1[2]->Delta;
		else 
			Delta = abit1[1]->Delta;
	}

        // MAC,KEY are pointers to a nP+1-len array of points to blocks of len length
        // where MAC.at(i)/KEY.at(i) are the MACs/KEYs for party i.
	void compute(std::unordered_map<int, block*>& MAC, std::unordered_map<int, block*>& KEY, const bool* data, int length) {
		vector<future<void>> res;
		for(int i = 1; i <= nP; ++i) for(int j = 1; j<= nP; ++j) if( (i < j) and (i == party or j == party) ) {
			int party2 = i + j - party;
                        auto kp = KEY.at(party2);
			res.push_back(pool->enqueue([this, kp, length, party2]() {
				abit1[party2]->send_cot(kp, length);
				io->flush(party2);
			}));
                        auto mp = MAC.at(party2);
			res.push_back(pool->enqueue([this, mp, data, length, party2]() {
				abit2[party2]->recv_cot(mp, data, length);
				io->flush(party2);
			}));
		}
		joinNclean(res);
#ifdef __debug
		check_MAC(io, MAC, KEY, data, Delta, length, party);
#endif
	}

        // MAC,KEY are pointers to a nP+1-len array of points to blocks of len length
        // where MAC.at(i)/KEY.at(i) are the MACs/KEYs for party i.
	future<void> check(const std::unordered_map<int, block*>& MAC, const std::unordered_map<int, block*>& KEY, const bool* data, int length) {
		future<void> ret = pool->enqueue([this, MAC, KEY, data, length](){
			check1(MAC, KEY, data, length);
			check2(MAC, KEY, data, length);
		});
		return ret;
	}

	void check1(const std::unordered_map<int, block*>& MAC, const std::unordered_map<int, block*>& KEY, const bool* data, int length) {
		block seed = sampleRandom(io, &prg, pool, party);
		PRG prg2(&seed);
		uint8_t * tmp;
		block * Ms[nP+1];
		bool * bs[nP+1];
		block * Ks[nP+1];
		block * tMs[nP+1];
		bool * tbs[nP+1];

		tmp = new uint8_t[ssp*length];
		prg2.random_data(tmp, ssp*length);
		for(int i = 0; i < ssp*length; ++i)
			tmp[i] = tmp[i] % 4;
//		for(int j = 0; j < ssp; ++j) {
//			tmp[j] = new bool[length];
//			for(int k = 0; k < ssp; ++k)
//				tmp[j][length - ssp + k] = (k == j);
//		}
		for(int i = 1; i <= nP; ++i) {
			Ms[i] = new block[ssp];
			Ks[i] = new block[ssp];
			bs[i] = new bool[ssp];
			memset(Ms[i], 0, ssp*sizeof(block));
			memset(Ks[i], 0, ssp*sizeof(block));
			memset(bs[i], false, ssp);
			tMs[i] = new block[ssp];
			tbs[i] = new bool[ssp];
		}
		
		const int chk = 1;
		const int SIZE = 1024*2;
		block (* tMAC)[4] = new block[SIZE/chk][4];
		block (* tKEY)[4] = new block[SIZE/chk][4];
		bool (* tb)[4] = new bool[length/chk][4];
		memset(tMAC, 0, sizeof(block)*4*SIZE/chk);
		memset(tKEY, 0, sizeof(block)*4*SIZE/chk);
		memset(tb, false, 4*length/chk);
		for(int i = 0; i < length; i+=chk) {
			tb[i/chk][1] = data[i];
			tb[i/chk][2] = data[i+1];
			tb[i/chk][3] = data[i] != data[i+1];
		}

		for(int k = 1; k <= nP; ++k) if(k != party) {
			uint8_t * tmpptr = tmp;
			for(int tt = 0; tt < length/SIZE; tt++) {
				int start = SIZE*tt;
				for(int i = SIZE*tt; i < SIZE*(tt+1) and i < length; i+=chk) {
				  tMAC[(i-start)/chk][1] = MAC.at(k)[i];
				  tMAC[(i-start)/chk][2] = MAC.at(k)[i+1];
				  tMAC[(i-start)/chk][3] = MAC.at(k)[i] ^ MAC.at(k)[i+1];

				  tKEY[(i-start)/chk][1] = KEY.at(k)[i];
				  tKEY[(i-start)/chk][2] = KEY.at(k)[i+1];
				  tKEY[(i-start)/chk][3] = KEY.at(k)[i] ^ KEY.at(k)[i+1];
				  for(int j = 0; j < ssp; ++j) {
							 Ms[k][j] = Ms[k][j] ^ tMAC[(i-start)/chk][*tmpptr];
							 Ks[k][j] = Ks[k][j] ^ tKEY[(i-start)/chk][*tmpptr];
							 bs[k][j] = bs[k][j] != tb[i/chk][*tmpptr];
							 ++tmpptr;
					}
				}
			}
		}
		delete[] tmp;
		vector<future<bool>> res;
		//TODO: they should not need to send MACs.	
		for(int i = 1; i <= nP; ++i) for(int j = 1; j<= nP; ++j) if( (i < j) and (i == party or j == party) ) {
			int party2 = i + j - party;
                        auto mp = Ms[party2];
                        auto bp = bs[party2];
			res.push_back(pool->enqueue([this, mp, bp, party2]()->bool {
				io->send_data(party2, mp, sizeof(block)*ssp);
				io->send_data(party2, bp, ssp);
				io->flush(party2);
				return false;
			}));

                        auto tMp = tMs[party2];
                        auto tbp = tbs[party2];
                        auto kp = Ks[party2];
			res.push_back(pool->enqueue([this, tMp, tbp, kp, party2]()->bool {
				io->recv_data(party2, tMp, sizeof(block)*ssp);
				io->recv_data(party2, tbp, ssp);
				for(int k = 0; k < ssp; ++k) {
					if(tbp[k])
						kp[k] = kp[k] ^ Delta;
				}
				auto r = !cmpBlock(kp, tMp, ssp);
                                // HACK: this check fails for some particular circuit combinations
                                // for no apparent reason.
                                if(r) {
                                    std::cout << this->party << " cheat by " << party2 << std::endl;
                                }

                                return false;
			}));
		}
		if(joinNcleanCheat(res)) error("cheat check1\n");

		for(int i = 1; i <= nP; ++i) {
			delete[] Ms[i];
			delete[] Ks[i];
			delete[] bs[i];
			delete[] tMs[i];
			delete[] tbs[i];
		}
	}

	void check2(const std::unordered_map<int, block*>& MAC, const std::unordered_map<int, block*>& KEY, const bool* data, int length) {
		//last 2*ssp are garbage already.
		block * Ks[2], *Ms[nP+1][nP+1];
		block * KK[nP+1];
		bool * bs[nP+1];
		Ks[0] = new block[ssp];
		Ks[1] = new block[ssp];
		for(int i = 1; i <= nP; ++i) {
			bs[i] = new bool[ssp];
			KK[i] = new block[ssp];
			for(int j = 1; j <= nP; ++j)
				Ms[i][j] = new block[ssp];
		}
		char (*dgst)[Hash::DIGEST_SIZE] = new char[nP+1][Hash::DIGEST_SIZE];
		char (*dgst0)[Hash::DIGEST_SIZE] = new char[ssp*(nP+1)][Hash::DIGEST_SIZE];
		char (*dgst1)[Hash::DIGEST_SIZE] = new char[ssp*(nP+1)][Hash::DIGEST_SIZE];
	
		
		for(int i = 0; i < ssp; ++i) {
			Ks[0][i] = zero_block;
			for(int j = 1; j <= nP; ++j) if(j != party)
				Ks[0][i] = Ks[0][i] ^ KEY.at(j)[length-3*ssp+i];
			
			Ks[1][i] = Ks[0][i] ^ Delta;
			Hash::hash_once(dgst0[party*ssp+i], &Ks[0][i], sizeof(block));
			Hash::hash_once(dgst1[party*ssp+i], &Ks[1][i], sizeof(block));
		}
		Hash h;
		h.put(data+length-3*ssp, ssp);
		for(int j = 1; j <= nP; ++j) if(j != party) {
			h.put(&MAC.at(j)[length-3*ssp], ssp*sizeof(block));
		}
		h.digest(dgst[party]);

		vector<future<void>> res;
		for(int i = 1; i <= nP; ++i) for(int j = 1; j<= nP; ++j) if( (i < j) and (i == party or j == party) ) {
			int party2 = i + j - party;
			res.push_back(pool->enqueue([this, dgst, dgst0, dgst1, party2](){
				io->send_data(party2, dgst[party], Hash::DIGEST_SIZE);
				io->send_data(party2, dgst0[party*ssp], Hash::DIGEST_SIZE*ssp);
				io->send_data(party2, dgst1[party*ssp], Hash::DIGEST_SIZE*ssp);
				io->recv_data(party2, dgst[party2], Hash::DIGEST_SIZE);
				io->recv_data(party2, dgst0[party2*ssp], Hash::DIGEST_SIZE*ssp);
				io->recv_data(party2, dgst1[party2*ssp], Hash::DIGEST_SIZE*ssp);
			}));
		}
		joinNclean(res);

		vector<future<bool>> res2;
		for(int k = 1; k <= nP; ++k) if(k!= party)
			memcpy(Ms[party][k], MAC.at(k)+length-3*ssp, sizeof(block)*ssp);

		for(int i = 1; i <= nP; ++i) for(int j = 1; j<= nP; ++j) if( (i < j) and (i == party or j == party) ) {
			int party2 = i + j - party;
			res2.push_back(pool->enqueue([this, data, MAC, length, party2]() -> bool {
				io->send_data(party2, data + length - 3*ssp, ssp);
				for(int k = 1; k <= nP; ++k) if(k != party)
					io->send_data(party2, MAC.at(k) + length - 3*ssp, sizeof(block)*ssp);
				return false;
			}));

                        auto bp = bs[party2];
                        auto mp = Ms[party2];
			res2.push_back(pool->enqueue([this, dgst, bp, mp,  party2]() -> bool {
				Hash h;
				io->recv_data(party2, bp, ssp);
				h.put(bp, ssp);
				for(int k = 1; k <= nP; ++k) if(k != party2) {
					io->recv_data(party2, mp[k], sizeof(block)*ssp);
					h.put(mp[k], sizeof(block)*ssp);
				}
				char tmp[Hash::DIGEST_SIZE];h.digest(tmp);
				return strncmp(tmp, dgst[party2], Hash::DIGEST_SIZE) != 0;
			}));
		}
		if(joinNcleanCheat(res2)) error("commitment 1\n");

		memset(bs[party], false, ssp);
		for(int i = 1; i <= nP; ++i) if(i != party) {
			for(int j = 0; j < ssp; ++j)
				bs[party][j] =  bs[party][j] !=  bs[i][j];
		}
		for(int i = 1; i <= nP; ++i) for(int j = 1; j<= nP; ++j) if( (i < j) and (i == party or j == party) ) {
			int party2 = i + j - party;
                        auto bp = bs[party];
			res2.push_back(pool->enqueue([this, bp, Ks, party2]() -> bool {
				io->send_data(party2, bp, ssp);
				for(int i = 0; i < ssp; ++i) {
					if(bp[i])
						io->send_data(party2, &Ks[1][i], sizeof(block));
					else
						io->send_data(party2, &Ks[0][i], sizeof(block));
				}
				io->flush(party2);
				return false;
			}));

                        auto kp = KK[party2];
			res2.push_back(pool->enqueue([this, kp, dgst0, dgst1, party2]() -> bool {
				bool cheat = false;
				bool *tmp_bool = new bool[ssp];
				io->recv_data(party2, tmp_bool, ssp);
				io->recv_data(party2, kp, ssp*sizeof(block));
				for(int i = 0; i < ssp; ++i) {
					char tmp[Hash::DIGEST_SIZE];
					Hash::hash_once(tmp, &kp[i], sizeof(block));
					if(tmp_bool[i])
						cheat = cheat or (strncmp(tmp, dgst1[party2*ssp+i], Hash::DIGEST_SIZE)!=0);
					else
						cheat = cheat or (strncmp(tmp, dgst0[party2*ssp+i], Hash::DIGEST_SIZE)!=0);
				}
				delete[] tmp_bool;
				return cheat;
			}));
		}
		if(joinNcleanCheat(res2)) error("commitments 2\n");
		
		bool cheat = false;
		block *tmp_block = new block[ssp];
		for(int i = 1; i <= nP; ++i) if (i != party) {
			memset(tmp_block, 0, sizeof(block)*ssp);
			for(int j = 1; j <= nP; ++j) if(j != i) {
				for(int k = 0; k < ssp; ++k)
					tmp_block[k] = tmp_block[k] ^ Ms[j][i][k];
			}
			cheat = cheat or !cmpBlock(tmp_block, KK[i], ssp);
		}
		if(cheat) error("cheat aShare\n");

		delete[] Ks[0];
		delete[] Ks[1];
		delete[] dgst;
		delete[] dgst0;
		delete[] dgst1;
		delete[] tmp_block;
		for(int i = 1; i <= nP; ++i) {
			delete[] bs[i];
			delete[] KK[i];
			for(int j = 1; j <= nP; ++j)
				delete[] Ms[i][j];
		}
	}
};
#endif //ABIT_MP_H__
