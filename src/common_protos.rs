use crate::{
    auth_bits::Abits,
    base_func::{
        CheatDetectedError, CheatOrUnexpectedError, FuncContext, FuncId, SessionId, UnexpectedError,
    },
    field::{Field, RandElement, Ring},
    func_com::AsyncCom,
    func_net::AsyncNet,
    party::PartyId,
    polynomial::{FixedPolynomial, Polynomial},
};

use std::{
    sync::Arc,
    collections::HashMap,
};

use anyhow::Context;

use futures::stream::FuturesUnordered;
use futures::StreamExt;
use tokio::task::JoinSet;

pub async fn random_shares<T: Field + RandElement, FN: AsyncNet>(
    num: usize,
    my_id: PartyId,
    parties: &[PartyId],
    t: usize,
    caller: FuncId,
    sid: SessionId,
    net: Arc<FN>,
) -> Result<Vec<T>, UnexpectedError> {
    let polys: Vec<_> = {
        let mut rng = rand::thread_rng();
        (0..num)
            .map(|_| FixedPolynomial::rand_polynomial(&mut rng, t - 1))
            .collect()
    };

    // TODO: this really doesn't need to be spawned
    let mut send_set = JoinSet::new();
    let mut recv_set = JoinSet::new();
    for &p in parties.iter().filter(|&p| *p != my_id) {
        let mut sbuf = unsafe { Arc::<[u8]>::new_zeroed_slice(num * T::BYTES).assume_init() };
        let p_point = T::from(p.into());
        Arc::get_mut(&mut sbuf)
            .unwrap()
            .chunks_exact_mut(T::BYTES)
            .zip(polys.iter())
            .for_each(|(c, s)| {
                let y = s.evaluate(&p_point);
                y.to_bytes(c);
            });

        let net2 = net.clone();
        send_set.spawn(async move { net2.send_to(p, caller, sid, sbuf).await });

        let rbuf = unsafe { Arc::<[u8]>::new_zeroed_slice(num * T::BYTES).assume_init() };
        let net3 = net.clone();
        recv_set.spawn(async move { net3.recv_from(p, caller, sid, rbuf).await });
    }

    while let Some(r) = send_set.join_next().await {
        let _ = r.unwrap().context("Failed to send shares")?;
    }

    let my_point = T::from(my_id.into());
    let mut my_shares: Vec<T> = polys.iter().map(|s| s.evaluate(&my_point)).collect();

    while let Some(r) = recv_set.join_next().await {
        let (buf, count) = r.unwrap().context("Failed to receive shares")?;

        assert!(count == num * T::BYTES);

        buf.chunks_exact(T::BYTES)
            .zip(my_shares.iter_mut())
            .for_each(|(c, s)| {
                *s += T::from_bytes(c);
            });
    }

    Ok(my_shares)
}

pub async fn broadcast_opportunistic<FN: AsyncNet, D: AsRef<[u8]>>(
    caller: FuncId,
    sid: SessionId,
    data: D,
    party_id: PartyId,
    parties: &[PartyId],
    net: Arc<FN>
) -> Result<HashMap<PartyId, Vec<u8>>, UnexpectedError> {

    let mut futs = FuturesUnordered::new();

    for p in parties.iter().filter(|p| **p != party_id) {
        futs.push({
            let p = p.clone();
            let data = &data;
            let net = &net;
            async move {
                let recv = data.as_ref().to_vec();
                let sendf = net.send_to_local(p, caller, sid, data.as_ref());
                let recvf = net.recv_from_local(p, caller, sid, recv);
                (p, tokio::join!(sendf, recvf))
            }
        });
    }

    let mut results = HashMap::new();

    while let Some((other, (r1, r2))) = futs.next().await {
        let (_, (buf, _)) = (r1.context("Failed to send to {other}")?, r2.context("failed to recv from {other}")?);
        results.insert(other, buf);
    }

    Ok(results)
}

pub async fn broadcast_commit_open<FC: AsyncCom>(
    sid: SessionId,
    data: &[u8],
    party_id: PartyId,
    parties: &[PartyId],
    com: Arc<FC>,
) -> Result<Vec<Vec<u8>>, CheatOrUnexpectedError> {


    
    let mut commit_futs = FuturesUnordered::new();

    for &party in parties.iter().filter(|&&x| x != party_id) {
        commit_futs.push({
            let party = party.clone();
            let com = &com;
            async move {
                tokio::join!(com.commit_to(sid, party, data), com.expect_from(sid, party))
            }
        });
    }

    while let Some((r1, r2)) = commit_futs.next().await {
        let _ = (r1?, r2?);
    }

    let mut decom_futs = FuturesUnordered::new();

    for &party in parties.iter().filter(|&&x| x != party_id) {
        decom_futs.push({
            let party = party.clone();
            let com = &com;
            async move {
                (party, tokio::join!(com.decommit_to(sid, party, data), com.value_from(sid, party, data.len())))
            }
        });
    }

    let mut other_vals = Vec::new();

    while let Some((_other, (r1, r2))) = decom_futs.next().await {
        let (_, val) = (r1?, r2?);

        other_vals.push(val);
    }

    Ok(other_vals)
}

pub async fn synchronize<FN: AsyncNet>(
    party_id: PartyId,
    parties: &[PartyId],
    caller: FuncId,
    sid: SessionId,
    net: Arc<FN>,
) -> Result<(), UnexpectedError> {
    let futs = FuturesUnordered::new();

    for &p in parties.iter().filter(|&&p| p != party_id) {
        let net = &net;
        futs.push(async move {
            net.send_to_local(p, caller, sid, [0])
                .await
                .context("Failed to send to {p}")?;
            net.recv_from_local(p, caller, sid, [0])
                .await
                .context("Failed to receive from {p}")
                .map(|_| ())
        });
    }

    tokio::pin!(futs);
    while let Some(f) = futs.next().await {
        let _ = f?;
    }

    Ok(())
}

pub async fn open_abits<F: Ring, FN: AsyncNet>(
    abits: &Abits<F>,
    net: Arc<FN>,
    delta: &F,
    my_id: PartyId,
    parties: &[PartyId],
    sid: SessionId,
    dst: FuncId,
) -> Result<Vec<bool>, CheatOrUnexpectedError> {
    let mut send_set = JoinSet::new();
    let mut recv_set = JoinSet::new();

    let nbits = abits.len();

    for (j, &p) in parties.iter().filter(|&p| *p != my_id).enumerate() {
        let mut sbuf =
            unsafe { Arc::<[u8]>::new_zeroed_slice(nbits * (F::BYTES + 1)).assume_init() };

        {
            let b = Arc::get_mut(&mut sbuf).unwrap();

            for (x, &y) in b[..nbits].iter_mut().zip(abits.bits.iter()) {
                *x = y.into();
            }

            // if we wanted to be unsafe we could probably cast abits.macs[j] as [u8] and just
            // copy_from_slice
            b[nbits..]
                .chunks_exact_mut(F::BYTES)
                .zip(abits.macs[j].iter())
                .for_each(|(c, m)| {
                    m.to_bytes(c);
                });
        }

        let net2 = net.clone();
        send_set.spawn(async move {
            net2.send_to(p, dst, sid, sbuf)
                .await
                .context("Failed to send abits to {p}")
        });

        let rbuf =
            unsafe { Arc::<[u8]>::new_zeroed_slice(abits.len() * (F::BYTES + 1)).assume_init() };
        let net3 = net.clone();
        recv_set.spawn(async move {
            (
                j,
                p,
                net3.recv_from(p, dst, sid, rbuf)
                    .await
                    .context("Failed to receive abits from {p}"),
            )
        });
    }

    while let Some(s) = send_set.join_next().await {
        let _ = s.unwrap()?;
    }

    let mut bits: Vec<bool> = abits.bits.clone();

    while let Some(r) = recv_set.join_next().await {
        let r = r.unwrap();
        let j = r.0;
        let p = r.1;
        let (buf, count) = r.2?;

        assert!(count == abits.len() * (F::BYTES + 1));
        assert!(buf.len() >= count);

        let mut i = 0;

        buf[..nbits]
            .iter()
            .zip(bits.iter_mut())
            .zip(abits.keys[j].iter())
            .zip(buf[nbits..].chunks_exact(F::BYTES))
            .try_for_each(|(((&nb, b), k), m)| {
                let pb = nb == 1;
                let m = F::from_bytes(m);
                let mut emac = k.clone();

                if pb {
                    emac += delta;
                }

                *b ^= pb;

                if m != emac {
                    // TODO: include an actual sid
                    let sid = SessionId::new(dst);
                    let ctx = FuncContext {
                        party: my_id,
                        func: dst,
                        sid: sid,
                    };
                    Err(CheatDetectedError::new(
                        ctx,
                        Some(p),
                        format!("check mac {} failed ({:?} != {:?})", i, m, emac),
                    ))
                } else {
                    i += 1;
                    Ok(())
                }
            })?;
    }

    Ok(bits)
}
