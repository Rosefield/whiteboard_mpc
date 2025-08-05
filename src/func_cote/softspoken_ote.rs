use crate::{
    base_func::{
        BaseFunc, CheatDetectedError, CheatOrUnexpectedError, FuncId, SessionId, UnexpectedError,
    },
    field::{
        ConstInt, Extension, FWrap, Group, Module, RandElement, Ring, RingExtension, ToFromBytes,
    },
    func_cote::AsyncOt,
    func_eot::AsyncEot,
    func_net::AsyncNet,
    linalg::{Matrix, Vector},
    party::PartyId,
    prg::Prg,
    punctured_prf::*,
    ro::RO,
    small_fields::{Rr2Ell, Rr4Ell, FF2, FF4},
};

use anyhow::Context;
use rand::Rng;
use std::sync::Arc;

// TODO: TCR hashes, AES-based PRG?

pub struct SoftspokenOtePlayer<FOT, FN> {
    party_id: PartyId,
    ot: Arc<FOT>,
    net: Arc<FN>,
}

impl<FOT, FN> BaseFunc for SoftspokenOtePlayer<FOT, FN> {
    const FUNC_ID: FuncId = FuncId::Fot;
    const REQUIRED_FUNCS: &'static [FuncId] = &[FuncId::Fot, FuncId::Fnet];

    fn party(&self) -> PartyId {
        self.party_id
    }
}

impl<FOT: AsyncEot, FN: AsyncNet> AsyncOt for SoftspokenOtePlayer<FOT, FN> {
    async fn init(
        &self,
        sid: SessionId,
        other: PartyId,
        is_sender: bool,
    ) -> Result<(), UnexpectedError> {
        // our sender is the base ot receiver
        self.ot
            .init(sid.derive_ssid(FuncId::Fot), other, !is_sender)
            .await?;

        Ok(())
    }

    async fn rand_recv<T: RandElement, const ELL: usize>(
        &self,
        sid: SessionId,
        other: PartyId,
    ) -> Result<(Vector<bool, ELL>, Vector<T, ELL>), CheatOrUnexpectedError> {
        self.ot_extension_recv(sid, other).await
    }

    async fn rand_send<T: RandElement, const ELL: usize>(
        &self,
        sid: SessionId,
        other: PartyId,
    ) -> Result<(Vector<T, ELL>, Vector<T, ELL>), CheatOrUnexpectedError> {
        self.ot_extension_send(sid, other).await
    }
}

impl<FOT: AsyncEot, FN: AsyncNet> SoftspokenOtePlayer<FOT, FN> {
    async fn nm1_ot_send(
        &self,
        sid: SessionId,
        other: PartyId,
        num: usize,
    ) -> Result<Vec<[Rr2Ell<128>; 4]>, CheatOrUnexpectedError> {
        const K: usize = 2;
        const _NC: usize = 2;

        let otsid = sid.derive_ssid(FuncId::Fot);

        let ms = self.ot.send::<Rr2Ell<128>>(otsid, other, num).await?; // 128 = K*NC

        // probably should avoid double-allocating on the messages to send
        let (nm1_ots, msgs): (Vec<_>, Vec<_>) = ms
            .chunks_exact(K)
            .map(|c| {
                let (leaves, trace, s, t) = internal::all_but_one_ot_send(c);
                (
                    <[_; 4]>::try_from(leaves).unwrap(),
                    FWrap((trace[0].clone(), s, t)),
                )
            })
            .unzip();

        let msize = msgs[0].num_bytes();
        let mut buffer = vec![0u8; msize * msgs.len()];
        buffer
            .chunks_exact_mut(msize)
            .zip(msgs.into_iter())
            .for_each(|(b, m)| {
                m.to_bytes(b);
            });

        let _ = self
            .net
            .send_to_local(other, FuncId::Fot, sid, &mut buffer)
            .await
            .with_context(|| {
                self.err(
                    sid,
                    format!("Failed to send n-minus-1 OT messages to {other}"),
                )
            })?;

        Ok(nm1_ots)
    }

    async fn nm1_ot_recv(
        &self,
        sid: SessionId,
        other: PartyId,
        bits: &[bool],
    ) -> Result<Vec<(usize, [Rr2Ell<128>; 3])>, CheatOrUnexpectedError> {
        const K: usize = 2;
        const NC: usize = 64;

        let otsid = sid.derive_ssid(FuncId::Fot);

        let ms = self.ot.recv::<Rr2Ell<128>>(otsid, other, bits).await?;

        type AB1Recv = FWrap<([Rr2Ell<128>; 2], [u8; 32], [u8; 32])>;

        let mut buffer = vec![0u8; AB1Recv::BYTES * NC];
        let _ = self
            .net
            .recv_from_local(other, FuncId::Fot, sid, &mut buffer)
            .await
            .with_context(|| {
                self.err(
                    sid,
                    format!("Failed to recv n-minus-1 OT messages from {other}"),
                )
            })?;

        // probably should avoid double-allocating on the messages to send
        let nm1_ots = buffer
            .chunks_exact(AB1Recv::BYTES)
            .zip(ms.chunks_exact(K))
            .zip(bits.chunks_exact(K))
            .map(|((buf, m), selections)| {
                let FWrap((trace, s, t)) = AB1Recv::from_bytes(buf);
                let missing = [1 - (selections[0] as usize), 1 - (selections[1] as usize)];
                let (ms, _) = m.as_chunks::<1>();
                let (idx, prfs, sp) = internal::all_but_one_ot_recv(&ms, &missing, &[trace], t);

                if s != sp {
                    log::error!("{}: received invalid pprf proof", self.party_id);
                    return Err(self.cheat(sid, Some(other), "invalid pprf proof".to_string()));
                }

                let prfs: [Rr2Ell<128>; 3] = prfs.try_into().unwrap();

                Ok((idx, prfs))
            })
            .collect::<Result<Vec<_>, CheatDetectedError>>()?;

        Ok(nm1_ots)
    }

    async fn vole_send<const ELL: usize>(
        &self,
        sid: SessionId,
        other: PartyId,
    ) -> Result<(Matrix<FF2, ELL, 64>, Matrix<FF4, ELL, 64>), CheatOrUnexpectedError> {
        const _K: usize = 2;
        const NC: usize = 64;

        let nm1_ots = self.nm1_ot_send(sid, other, 128).await?;

        // receive U', V
        // probably need to transpose U' since right now it is in the form of \FF_p^{nc \times \ell}
        // paper mentions calculating this using Eklundh's algorithm
        let voles: (Vec<_>, Vec<_>) = nm1_ots
            .into_iter()
            .map(|prfs| internal::small_vole_send::<_, Rr2Ell<ELL>, Rr4Ell<ELL>>(&prfs))
            .unzip();
        let u_transpose_vec: Vec<_> = voles.0.into_iter().map(|row| row.as_vector()).collect();
        let u_transpose: Matrix<FF2, NC, ELL> = u_transpose_vec.try_into().unwrap();
        let uprime = u_transpose.transpose();

        let v_transpose_vec: Vec<_> = voles.1.into_iter().map(|row| row.as_vector()).collect();
        let v_transpose: Matrix<FF4, NC, ELL> = v_transpose_vec.try_into().unwrap();
        let v: Matrix<FF4, ELL, NC> = v_transpose.transpose();

        Ok((uprime, v))
    }

    async fn vole_recv<const ELL: usize>(
        &self,
        sid: SessionId,
        other: PartyId,
    ) -> Result<(Vector<FF4, 64>, Matrix<FF4, ELL, 64>), CheatOrUnexpectedError> {
        const K: usize = 2;
        const NC: usize = 64;

        let bits = {
            let mut bits = [false; K * NC];
            let mut rng = rand::thread_rng();
            rng.fill(bits.as_mut_slice());
            bits
        };

        let nm1_ots = self.nm1_ot_recv(sid, other, &bits).await?;

        // receive \Delta, W
        // probably need to transpose U' since right now it is in the form of \FF_p^{nc \times \ell}
        // paper mentions calculating this using Eklundh's algorithm
        let (delta, wprime): (Vec<_>, Vec<_>) = nm1_ots
            .into_iter()
            .map(|(idx, prfs)| internal::small_vole_recv::<_, Rr2Ell<ELL>, Rr4Ell<ELL>>(idx, &prfs))
            .unzip();

        let delta: Vector<FF4, NC> = delta.try_into().unwrap();
        let wprime_t_v: Vec<_> = wprime.into_iter().map(|r| r.as_vector()).collect();
        let wprime_t: Matrix<FF4, NC, ELL> = wprime_t_v.try_into().unwrap();
        let wprime = wprime_t.transpose();

        Ok((delta, wprime))
    }

    async fn rep_subspace_vole_send<const ELL: usize>(
        &self,
        sid: SessionId,
        other: PartyId,
    ) -> Result<(Matrix<FF2, ELL, 1>, Matrix<FF4, ELL, 64>), CheatOrUnexpectedError> {
        // k * n_c instances of 1-of-2 (interpretted as n-1 ot)
        // gives n_c instances of (2^k-1 - of - 2^k) ot
        // gives (p,q)-subspace VOLE of length \ell
        // gives (p,q)-subspace VOLE of length \ell where elements lie in some
        // linear code C [n_c, k_c, d_c] which is a k_c-dimensional subspace of F_p^{n_c}
        // gives \ell instances of 1-of-(p^{k_c}) OT (e.g. OT extension)
        const _K: usize = 2;
        const NC: usize = 64;
        const KC: usize = 1;

        let (uprime, v) = self.vole_send(sid, other).await?;

        // TODO: select the right code
        // TODO: since we don't use a generic code, we can specialize this multiplication
        let tcinv: Matrix<FF2, NC, NC> = internal::rep_code_basis_inv();

        let t = uprime.mm(&tcinv);
        // [U C] = t
        let (u, c): (Matrix<_, _, KC>, Matrix<_, _, { NC - KC }>) = t.split_col();

        let mut buf = vec![0u8; c.num_bytes()];
        c.to_bytes(&mut buf[..]);
        let _ = self
            .net
            .send_to_local(other, FuncId::Fot, sid, &buf[..])
            .await
            .with_context(|| self.err(sid, "failed to send code correction message"))?;

        // TODO: What is this supposed to be?
        const M: usize = 100;

        buf.resize(Matrix::<FF4, M, ELL>::BYTES, 0);
        let _ = self
            .net
            .recv_from_local(other, FuncId::Fot, sid, &mut buf[..])
            .await
            .with_context(|| self.err(sid, "failed to received R matrix"))?;

        let r: Matrix<FF4, M, ELL> = Matrix::from_bytes(&buf[..]);

        let u4 = Matrix::embed(&u);
        let ut = r.mm(&u4);
        // paper says \tilde{V} should be m \times k_c, but this seems like a typo and should be m
        // \times n_c?
        let vt = r.mm(&v);

        let msg = FWrap((ut, vt));
        buf.resize(msg.num_bytes(), 0);
        msg.to_bytes(&mut buf[..]);
        let _ = self
            .net
            .send_to_local(other, FuncId::Fot, sid, &buf[..msg.num_bytes()])
            .await
            .with_context(|| self.err(sid, "failed to send check message"))?;

        // TODO: supposed to truncate to h rows, not sure what h is
        Ok((u, v))
    }

    async fn rep_subspace_vole_recv<const ELL: usize>(
        &self,
        sid: SessionId,
        other: PartyId,
    ) -> Result<(Vector<FF4, 64>, Matrix<FF4, ELL, 64>), CheatOrUnexpectedError> {
        const _K: usize = 2;
        const NC: usize = 64;
        const KC: usize = 1;

        let (delta, wprime) = self.vole_recv(sid, other).await?;

        let mut buf = vec![0u8; Matrix::<FF2, ELL, { NC - KC }>::BYTES];
        let _ = self
            .net
            .recv_from_local(other, FuncId::Fot, sid, &mut buf[..])
            .await
            .with_context(|| self.err(sid, "failed to recv code correction message"))?;

        let c = Matrix::<FF2, ELL, 63>::from_bytes(&buf);
        let mut ca = Matrix::<FF2, ELL, NC>::zero();
        // TODO: Slice/view set without allocation
        ca.rows_mut().zip(c.rows()).for_each(|(l, r)| {
            let v: Vector<FF2, 1> = Vector::zero();
            let v: Vector<FF2, NC> = v.concat(r);
            *l = v;
        });

        // TODO: different/generic codes
        let tc: Matrix<FF4, NC, NC> = internal::rep_code_basis();
        let diag = Matrix::diag(&delta);
        let tcd = tc.mm(&diag);

        let diff = Matrix::embed(&ca).mm(&tcd);

        let w = wprime - diff;

        // TODO: What is this supposed to be?
        const M: usize = 100;

        // TODO: send seed instead of full matrix
        let r = {
            let mut rng = rand::thread_rng();
            Matrix::<FF4, M, ELL>::rand(&mut rng)
        };

        buf.resize(Matrix::<FF4, M, ELL>::BYTES, 0);
        r.to_bytes(&mut buf);
        let _ = self
            .net
            .send_to_local(other, FuncId::Fot, sid, &buf[..])
            .await
            .with_context(|| self.err(sid, "failed to send universal hash"))?;

        type CheckMsg = FWrap<(Matrix<FF4, M, KC>, Matrix<FF4, M, NC>)>;
        buf.resize(CheckMsg::BYTES, 0);
        let _ = self
            .net
            .recv_from_local(other, FuncId::Fot, sid, &mut buf[..])
            .await
            .with_context(|| self.err(sid, "failed to send code correction message"))?;

        // paper on eprint says that vt should be M x K_c, but I am fairly certain this is a typo
        // for N_c
        let FWrap((ut, vt)) = CheckMsg::from_bytes(&buf);

        // in this case G_C * Diag(Delta) = Delta
        let code_gen: Matrix<FF4, KC, NC> = internal::rep_code_generator();
        let testv = r.mm(&w) - ut.mm(&code_gen.mm(&diag));
        if testv != vt {
            return Err(self
                .cheat(sid, Some(other), "Invalid vole check message".to_string())
                .into());
        }

        // TODO: truncate to h rows?
        Ok((delta, w))
    }

    async fn ot_extension_send<R: RandElement, const ELL: usize>(
        &self,
        sid: SessionId,
        other: PartyId,
    ) -> Result<(Vector<R, ELL>, Vector<R, ELL>), CheatOrUnexpectedError> {
        // ot extension sender is the vole receiver

        let (delta, w) = self.rep_subspace_vole_recv::<ELL>(sid, other).await?;

        let encode = |_i: usize| {
            // encode i into a random row
            Vector::<FF4, 64>::one()
        };

        let (m0s, m1s): (Vec<_>, Vec<_>) = w
            .rows()
            .enumerate()
            .map(|(i, w)| {
                let ri = encode(i);
                let y0 = ri + w;
                let y1 = y0.clone() - &delta;
                // TODO: proper TCR hash here
                let ro = RO::new()
                    .add_context("Fvole.ot_extension")
                    .add_context(i.to_le_bytes());
                let m0: R = ro.generate_read(&y0);
                let m1: R = ro.generate_read(&y1);

                (m0, m1)
            })
            .unzip();

        Ok((
            m0s.try_into().map_err(|_| "unexpected size").unwrap(),
            m1s.try_into().map_err(|_| "unexpected size").unwrap(),
        ))
    }

    async fn ot_extension_recv<R: RandElement, const ELL: usize>(
        &self,
        sid: SessionId,
        other: PartyId,
    ) -> Result<(Vector<bool, ELL>, Vector<R, ELL>), CheatOrUnexpectedError> {
        // ot extension sender is the vole receiver

        let (u, v) = self.rep_subspace_vole_send::<ELL>(sid, other).await?;
        let bs = std::array::from_fn(|i| !u.get_row(i).is_zero());

        // TODO: get from sender
        let encode = |_i: usize| {
            // encode i into a random row
            Vector::<FF4, 64>::one()
        };

        let ms = std::array::from_fn(|i| {
            let v = v.get_row(i);
            let ri = encode(i);
            let y = ri + v;
            // TODO: proper TCR hash here
            let ro = RO::new()
                .add_context("Fvole.ot_extension")
                .add_context(i.to_le_bytes());
            let m: R = ro.generate_read(&y);

            m
        });

        Ok((bs.into(), ms.into()))
    }
}

impl<FOT, FN> SoftspokenOtePlayer<FOT, FN> {
    pub fn new(party_id: PartyId, ot: Arc<FOT>, net: Arc<FN>) -> Self {
        Self { party_id, net, ot }
    }
}

pub mod internal {
    use super::*;

    pub fn trivial_code_generator<R: Ring, const NC: usize>() -> Matrix<R, NC, NC> {
        Matrix::one()
    }

    pub fn trivial_code_basis<R: Ring, const NC: usize>() -> Matrix<R, NC, NC> {
        Matrix::one()
    }

    pub fn trivial_code_basis_inv<R: Ring, const NC: usize>() -> Matrix<R, NC, NC> {
        Matrix::one()
    }

    pub fn rep_code_generator<R: Ring, const NC: usize>() -> Matrix<R, 1, NC> {
        Matrix::broadcast(R::one())
    }

    pub fn rep_code_basis<R: Ring, const NC: usize>() -> Matrix<R, NC, NC> {
        let g = rep_code_generator();
        // TODO: I feel like I shouldn't have to clone here
        let g = g.rows.i.0[0].clone();
        let mut b = Matrix::one();
        // [1 1 1 .. ]
        // [0 1 0 .. ]
        // [ ..      ]
        // [ ..   0 1]
        b.rows.i.0[0] = g;

        b
    }
    pub fn rep_code_basis_inv<R: Ring, const NC: usize>() -> Matrix<R, NC, NC> {
        // TODO: handle rings without characteristic two
        // conveniently in fields of characteristic 2 the above basis is the inverse
        assert!(R::one() + R::one() == R::zero());
        rep_code_basis()
    }

    pub fn all_but_one_ot_send<G: Group + RandElement>(
        prfs: &[[G; 2]],
    ) -> (Vec<G>, Vec<[G; 2]>, [u8; 32], [u8; 32]) {
        let (mut leaves, trace) = build_pprf(prfs);
        let (s, t) = prove_modify_pprf(&mut leaves);

        (leaves, trace, s, t)
    }

    pub fn all_but_one_ot_recv<G: Group + RandElement>(
        prfs: &[[G; 1]],
        missing_idxs: &[usize],
        traces: &[[G; 2]],
        t: [u8; 32],
    ) -> (usize, Vec<G>, [u8; 32]) {
        let (delta, mut leaves) = eval_pprf(prfs, missing_idxs, traces);
        let sp = verify_modify_pprf(&mut leaves, delta, t);
        (delta, leaves, sp)
    }

    /// F_{VOLE}^{p,q,\FF_p, \ell}
    pub fn small_vole_send<T: AsRef<[u8]>, R2, R4>(prfs: &[T; 4]) -> (R2, R4)
    where
        R2: Ring + RandElement,
        R4: Ring + RingExtension<R2> + Module<FF4> + RandElement,
    {
        // need prg: \bit^\secparam \rightarrow \FF_p^\ell

        let prg = Prg::new();

        let mut u = R2::zero();
        let mut v = R4::zero();

        // TODO: the paper notes a more computationally efficient way to calculate v
        // but I will ignore that for now.
        prfs.iter().enumerate().for_each(|(x, p)| {
            let rx: R2 = prg.generate(p);
            u += &rx;
            v -= R4::embed(&rx) * FF4::new(x as u8);
        });

        (u, v)
    }

    pub fn small_vole_recv<T: AsRef<[u8]>, R2, R4>(missing: usize, prfs: &[T; 3]) -> (FF4, R4)
    where
        R2: Ring + RandElement,
        R4: Ring + RingExtension<R2> + Module<FF4> + RandElement,
    {
        let prg = Prg::new();
        let delta = FF4::new(missing as u8);
        let w = prfs
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let x = if i < missing {
                    FF4::new(i as u8)
                } else {
                    FF4::new((i + 1) as u8)
                };
                let dx = delta - x;
                let rx: R2 = prg.generate(p);
                R4::embed(&rx) * dx
            })
            .sum();
        (delta, w)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::{
        func_eot::zzzr23_eot::tests::build_test_eots,
        func_net::tests::{build_test_nets, get_test_party_infos},
    };
    use std::sync::Arc;
    //use test::Bencher;
    use tokio::task::JoinHandle;

    #[test]
    fn test_small_vole_send_recv() {
        let mut rng = rand::thread_rng();
        let prfs: [Rr2Ell<128>; 4] = FWrap::rand(&mut rng).0;

        let (u, v): (Rr2Ell<1024>, Rr4Ell<1024>) = internal::small_vole_send(&prfs);

        let missing = 2usize;
        let ab1prf = [prfs[0].clone(), prfs[1].clone(), prfs[3].clone()];

        let (d, w): (_, Rr4Ell<1024>) = internal::small_vole_recv(missing, &ab1prf);

        assert_eq!(w - v, Rr4Ell::embed(&u) * d);
    }

    pub fn build_test_ots<FOT: AsyncEot, FN: AsyncNet>(
        eots: &[Arc<FOT>],
        nets: &[Arc<FN>],
    ) -> Vec<Arc<SoftspokenOtePlayer<FOT, FN>>> {
        let num = eots.len() as PartyId;
        (1..=num)
            .map(|i| {
                Arc::new(SoftspokenOtePlayer::new(
                    i,
                    eots[(i - 1) as usize].clone(),
                    nets[(i - 1) as usize].clone(),
                ))
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_nm1_ots() {
        let party_info = get_test_party_infos(2);
        let funcs = vec![FuncId::Feot, FuncId::Fot];
        let nets = build_test_nets(&party_info, funcs).await;
        let eots = build_test_eots(&nets);
        let ots = build_test_ots(&eots, &nets);

        let sid = SessionId::new(FuncId::Ftest);

        let t1: JoinHandle<Result<_, CheatOrUnexpectedError>> = tokio::spawn({
            let ot = ots[0].clone();
            async move {
                let _ = ot.init(sid, 2, false).await?;
                ot.nm1_ot_send(sid, 2, 128).await
            }
        });

        let t2: JoinHandle<Result<_, CheatOrUnexpectedError>> = tokio::spawn({
            let ot = ots[1].clone();
            async move {
                let _ = ot.init(sid, 1, true).await?;
                let bits = [false; 128];
                ot.nm1_ot_recv(sid, 1, &bits).await
            }
        });

        let (r1, r2) = tokio::try_join!(t1, t2).expect("Error running eots");
        let msgs = r1.unwrap();
        let ab1s = r2.unwrap();

        msgs.into_iter()
            .zip(ab1s.into_iter())
            .for_each(|(ms, (missing, rest))| {
                for (r, l) in (0..4).into_iter().filter(|i| *i != missing).enumerate() {
                    assert_eq!(ms[l], rest[r]);
                }
            });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_small_voles() {
        let party_info = get_test_party_infos(2);
        let funcs = vec![FuncId::Feot, FuncId::Fot];
        let nets = build_test_nets(&party_info, funcs).await;
        let eots = build_test_eots(&nets);
        let ots = build_test_ots(&eots, &nets);

        let sid = SessionId::new(FuncId::Ftest);

        // Should create the correlation w - v = u * diag(Delta)
        let t1: JoinHandle<Result<_, CheatOrUnexpectedError>> = tokio::spawn({
            let ot = ots[0].clone();
            async move {
                let _ = ot.init(sid, 2, false).await?;
                ot.vole_send::<128>(sid, 2).await
            }
        });

        let t2: JoinHandle<Result<_, CheatOrUnexpectedError>> = tokio::spawn({
            let ot = ots[1].clone();
            async move {
                let _ = ot.init(sid, 1, true).await?;
                ot.vole_recv::<128>(sid, 1).await
            }
        });

        let (r1, r2) = tokio::try_join!(t1, t2).expect("Error running eots");
        let (u, v) = r1.unwrap();
        let (delta, w) = r2.unwrap();

        let ud = Matrix::embed(&u).mm(&Matrix::diag(&delta));

        let wv = w - v;
        let diff = ud.clone() - &wv;
        diff.rows()
            .enumerate()
            .for_each(|(i, r)| println!("{}: {}", i, r.is_zero()));

        assert_eq!(ud, wv);
    }

    enum SendOrRecv<const ELL: usize> {
        Send(Vector<[u8; 16], ELL>, Vector<[u8; 16], ELL>),
        Recv(Vector<bool, ELL>, Vector<[u8; 16], ELL>),
    }
    async fn run_ots<const ELL: usize>() -> (
        Result<SendOrRecv<ELL>, CheatOrUnexpectedError>,
        Result<SendOrRecv<ELL>, CheatOrUnexpectedError>,
    ) {
        let party_info = get_test_party_infos(2);
        let funcs = vec![FuncId::Feot, FuncId::Fot];
        let nets = build_test_nets(&party_info, funcs).await;
        let eots = build_test_eots(&nets);
        let ots = build_test_ots(&eots, &nets);

        let sid = SessionId::new(FuncId::Ftest);

        let t1: JoinHandle<Result<_, _>> = tokio::spawn({
            let ot = ots[0].clone();
            async move {
                let now = std::time::Instant::now();
                let _ = ot.init(sid, 2, true).await?;
                let (m0, m1) = ot.rand_send(sid, 2).await?;

                log::info!(
                    "OT extension send of len {ELL} took {}ms",
                    now.elapsed().as_millis()
                );

                Ok(SendOrRecv::Send(m0, m1))
            }
        });

        let t2: JoinHandle<Result<_, _>> = tokio::spawn({
            let ot = ots[1].clone();
            async move {
                let _ = ot.init(sid, 1, false).await?;
                let (bs, ms) = ot.rand_recv(sid, 1).await?;

                Ok(SendOrRecv::Recv(bs, ms))
            }
        });

        tokio::try_join!(t1, t2).expect("Error running eots")
    }

    #[test]
    fn test_softspoken_ote() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_stack_size(1 << 24)
            .build()
            .expect("couldn't build runtime");

        rt.block_on(Box::pin(test_softspoken_ote_inner()));
    }

    async fn test_softspoken_ote_inner() {
        let _ = env_logger::builder().is_test(true).try_init();

        match Box::pin(run_ots::<4000>()).await {
            (Ok(SendOrRecv::Send(m0, m1)), Ok(SendOrRecv::Recv(selections, recv))) => {
                m0.into_iter()
                    .zip(m1.into_iter())
                    .zip(selections.into_iter())
                    .zip(recv.into_iter())
                    .all(|(((m0, m1), bit), recv)| {
                        assert_eq!(if bit { m1 } else { m0 }, recv);
                        true
                    });
            }
            (Err(e), _) | (_, Err(e)) => {
                assert!(false, "{:?}", e);
            }
            _ => {
                assert!(false);
            }
        };
    }
}
