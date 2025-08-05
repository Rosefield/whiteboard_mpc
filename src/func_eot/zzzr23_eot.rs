use crate::{
    func_eot::AsyncEot,
    base_func::{BaseFunc, FuncId, SessionId, UnexpectedError},
    ecgroup::EcGroup,
    field::{ToFromBytes, FWrap, RandElement},
    func_net::AsyncNet,
    party::PartyId,
    ro::RO,
};

use std::{
    sync::Arc,
    marker::PhantomData,
};

use rand::Rng;

use anyhow::Context;
//use log::{info, trace};

#[derive(Debug)]
pub struct ZzzrEotPlayer<FN, E> {
    party_id: PartyId,
    net: Arc<FN>,
    _curve: PhantomData<E>,
}

impl<FN, E> BaseFunc for ZzzrEotPlayer<FN, E> {
    const FUNC_ID: FuncId = FuncId::Feot;
    const REQUIRED_FUNCS: &'static [FuncId] = &[FuncId::Fnet];

    fn party(&self) -> PartyId {
        self.party_id
    }
}

impl<FN: AsyncNet, E: EcGroup + RandElement> AsyncEot for ZzzrEotPlayer<FN, E>
    where E::Ford: RandElement
{
    /// Runs the setup to create the base OTs and correlation
    async fn init(
        &self,
        _sid: SessionId,
        _other: PartyId,
        _is_sender: bool,
    ) -> Result<(), UnexpectedError> {
        Ok(())
    }

    async fn send<T: RandElement>(
        &self,
        sid: SessionId,
        other: PartyId,
        num: usize
    ) -> Result<Vec<[T;2]>, UnexpectedError> {

        let e_len = E::BYTES;
        let mut send_msg = vec![0u8; num*(16+e_len)];

        let commits: Vec<_> = send_msg.chunks_exact_mut(16+e_len)
            .enumerate()
            .map(|(i, send_msg)| {
                let mut rng = rand::thread_rng();
                let mut seed = [0u8;16];
                rng.fill(&mut seed);

                let FWrap((g,h)): FWrap<(E, E)> = RO::new()
                                .add_context(&sid.as_bytes())
                                .add_context(&i.to_le_bytes())
                                .add_context("s_1")
                                .generate(seed);

                let FWrap((r,s)): FWrap<(E::Ford, E::Ford)> = RandElement::rand(&mut rng);

                // calculate and send (seed, z = g^r h^s)
                let z = g.exp(&r) + h.exp(&s);

                send_msg[..16].copy_from_slice(&seed);
                z.to_bytes(&mut send_msg[16..]);
                (r,s)
            }).collect();

        let _ = self.net.send_to_local(other, FuncId::Feot, sid, &send_msg).await
                    .with_context(|| self.err(sid, "failed to send OT sender messages"))?;

        let mut recv_msg = vec![0u8; num*(16 + 2*e_len)];
        let _ = self.net.recv_from_local(other, FuncId::Feot, sid, &mut recv_msg).await
                    .with_context(|| self.err(sid, "failed to receive OT receiver messages"))?;

        let messages = recv_msg.chunks_exact(16 + 2*e_len)
            .enumerate()
            .zip(commits.iter())
            .map(|((i, recv_msg), (r,s))| {
                // receive seed, B1, B2
                let seed2 = &recv_msg[..16];
                let FWrap((B1, B2)) = FWrap::<(E,E)>::from_bytes(&recv_msg[16..]);

                // calculate group elements (G,H) using RO on seed2
                let FWrap((G, H)): FWrap<(E,E)> = RO::new()
                                                            .add_context(&sid.as_bytes())
                                                            .add_context(&i.to_le_bytes())
                                                            .add_context("r_1")
                                                            .generate(&seed2);


                let ro2 = RO::new()
                            .add_context(&sid.as_bytes())
                            .add_context(&i.to_le_bytes())
                            .add_context("s_2");

                // m_0 = RO2(sid, s, B_1^r B_2^s)
                // m_1 = RO2(sid, s, (B_1/G)^r (B_2/H)^s)


                let mut bytes = vec![0u8; e_len];

                let el1 = B1.exp(r) + B2.exp(s);
                el1.to_bytes(&mut bytes);
                let m0 = ro2.generate(&bytes);

                let mut bytes2 = bytes;
                let el2 = (B1-G).exp(r) + (B2-H).exp(s);
                el2.to_bytes(&mut bytes2);
                let m1 = ro2.generate(&bytes2);

                [m0, m1]
            }).collect();
            
        return Ok(messages);
    }

    async fn recv<T: RandElement>(
        &self,
        sid: SessionId,
        other: PartyId,
        selections: &[bool],
    ) -> Result<Vec<T>, UnexpectedError> {
        let e_len = E::BYTES;
        let num = selections.len();
        let mut recv_msg = vec![0u8; num*(16+e_len)];

        let _ = self.net.recv_from_local(other, FuncId::Feot, sid, &mut recv_msg).await
                    .with_context(|| self.err(sid, "failed to receive OT sender messages"))?;

        let mut send_msg = vec![0u8; num*(16+2*e_len)];
        let messages = recv_msg.chunks_exact(16 + e_len)
            .enumerate()
            .zip(selections.into_iter().cloned())
            .zip(send_msg.chunks_exact_mut(16+2*e_len))
            .map(|(((i, recv_msg), b), send_msg)| {
                // receive (seed, z)
                let seed = &recv_msg[..16];
                let z = E::from_bytes(&recv_msg[16..]);

                // calculate (g,h) using RO on seed
                let FWrap((g,h)): FWrap<(E, E)> = RO::new()
                                .add_context(&sid.as_bytes())
                                .add_context(&i.to_le_bytes())
                                .add_context("s_1")
                                .generate(seed);

                let mut rng = rand::thread_rng();
                // sample seed2
                let mut seed2 = [0u8;16];
                rng.fill(&mut seed2);

                // calculate group elements (G,H) using RO on seed2
                let FWrap((G, H)): FWrap<(E,E)> = RO::new()
                                                            .add_context(&sid.as_bytes())
                                                            .add_context(&i.to_le_bytes())
                                                            .add_context("r_1")
                                                            .generate(&seed2);

                // sample x, calculate (B_1 = g^x G^b, B_2 = h^x H^b)

                let x: E::Ford = RandElement::rand(&mut rng);

                let B1 = g.exp(&x) + G.exp(&(b as u64).into());
                let B2 = h.exp(&x) + H.exp(&(b as u64).into());
                // send (seed2, B_1, B_2)
                send_msg[..16].copy_from_slice(&seed2);
                B1.to_bytes(&mut send_msg[16..]);
                B2.to_bytes(&mut send_msg[16+e_len..]);

                let ro2 = RO::new()
                            .add_context(&sid.as_bytes())
                            .add_context(&i.to_le_bytes())
                            .add_context("s_2");

                // output m_b = RO2(sid, s, z^x)
                let mut bytes = vec![0u8; e_len];

                let el1 = z.exp(&x);
                el1.to_bytes(&mut bytes);
                let mb = ro2.generate(&bytes);

                mb
            }).collect();

        let _ = self.net.send_to_local(other, FuncId::Feot, sid, &send_msg).await
                    .with_context(|| self.err(sid, "failed to send OT receiver messages"))?;

        return Ok(messages);
    }

}


impl<FN, E> ZzzrEotPlayer<FN, E> {
    pub fn new(id: PartyId, net: Arc<FN>) -> Self {
        Self {
            party_id: id,
            net,
            _curve: PhantomData

        }
    }
}


#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::{
        p256::P256,
        func_net::tests::{build_test_nets, get_test_party_infos},
    };
    use std::sync::Arc;
    use tokio::task::JoinHandle;
    use test::Bencher;

    pub fn build_test_eots<FN: AsyncNet>(nets: &[Arc<FN>]) -> Vec<Arc<ZzzrEotPlayer<FN, P256>>> {
        let num = nets.len() as PartyId;
        (1..=num)
            .map(|i| Arc::new(ZzzrEotPlayer::new(i, nets[(i-1) as usize].clone())))
            .collect()
    }

    enum SendOrRecv {
        Send(Vec<[[u8; 32]; 2]>),
        Recv(Vec<bool>, Vec<[u8; 32]>)
    }
    async fn run_eots(num: usize) -> (Result<SendOrRecv, &'static str>, Result<SendOrRecv, &'static str>) {
        let party_info = get_test_party_infos(2);
        let funcs = vec![FuncId::Feot];
        let nets = build_test_nets(&party_info, funcs).await;

        let eots = build_test_eots(&nets);



        let sid = SessionId::new(FuncId::Ftest);


        let t1: JoinHandle<Result<_, &'static str>> = tokio::spawn({
            let eot = eots[0].clone();
            async move {
                let _ = eot.init(sid, 2, true).await.map_err(|_| "failed to init")?;
                let msgs: Vec<[[u8;32]; 2]> = eot.send(sid, 2, num).await.map_err(|_| "failed to send")?;

                Ok(SendOrRecv::Send(msgs))
            }
        });

        let t2: JoinHandle<Result<_, &'static str>> = tokio::spawn({
            let eot = eots[1].clone();
            async move {
                let _ = eot.init(sid, 1, false).await.map_err(|_| "failed to init")?;
                let selections = {
                    let mut selections = vec![true; num];
                    let mut rng = rand::thread_rng();
                    rng.fill(&mut selections[..]);
                    selections
                };
                let msgs: Vec<[u8;32]> = eot.recv(sid, 1, &selections).await.map_err(|_| "failed to recv")?;

                Ok(SendOrRecv::Recv(selections, msgs))
            }
        });

        tokio::try_join!(t1, t2).expect("Error running eots")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_eot() {
        match run_eots(10).await {
            (Ok(SendOrRecv::Send(send)), Ok(SendOrRecv::Recv(selections, recv))) => {
                send.iter().zip(selections.iter()).zip(recv.iter())
                    .all(|((send, bit), recv)| {
                        assert_eq!(if *bit { send[1] } else { send[0]}, *recv);
                        true
                    });
            },
            _ => { panic!("unexpected") }
        };
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .build()
            .unwrap()
    }


    #[bench]
    fn bench_eot(b: &mut Bencher) {
        let rt = rt();


        for i in 1..=2 {
            if let Ok(Some(summary)) = b.bench(|b| {
                b.iter(|| rt.block_on(run_eots(10usize.pow(i))));
                Ok(())
            }) {
                println!("Creating {} eots results: {:?}", 10usize.pow(i), summary);
            }

        }
    }

}

