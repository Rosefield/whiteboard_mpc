use crate::{
    base_func::{BaseFunc, CheatOrUnexpectedError, FuncId, SessionId, UnexpectedError},
    field::{FWrap, RandElement, Ring, ToFromBytes},
    func_cote::AsyncOt,
    func_net::AsyncNet,
    func_vole::AsyncVole,
    linalg::{Matrix, Vector},
    party::PartyId,
    ro::RO,
};

use std::sync::Arc;

use anyhow::Context;

use log::trace;

#[derive(Debug)]
pub struct Dkls23VolePlayer<FN, FOte> {
    party_id: PartyId,
    net: Arc<FN>,
    ote: Arc<FOte>,
}

impl<FN, FOte> BaseFunc for Dkls23VolePlayer<FN, FOte> {
    const FUNC_ID: FuncId = FuncId::Fvole;
    const REQUIRED_FUNCS: &'static [FuncId] = &[FuncId::Fnet, FuncId::Fote];

    fn party(&self) -> PartyId {
        self.party_id
    }
}

#[allow(non_upper_case_globals)]
const statParam: usize = 80;
#[allow(non_upper_case_globals)]
const _secParam: usize = 128;

type AliceMsg<const ELL: usize, const KSI: usize, const RHO: usize, T: Ring> = FWrap<(
    Matrix<T, KSI, { ELL + RHO }>, 
    Vector<T,  RHO>,
    [u8; 32],
)>;

impl<FN: AsyncNet, FOte: AsyncOt> AsyncVole for Dkls23VolePlayer<FN, FOte> {
    //const KSI<T: Ring>: usize = T::BYTES * 8 + 2 * statParam;
    //const RHO<T: Ring>: usize = (T::BYTES*8).div_ceil(secParam);
    /// Start a new instance with `sid`
    async fn init(&self, _sid: SessionId, _other: PartyId) -> Result<(), UnexpectedError> {
        Ok(())
    }

    /// calculate $\sum_i a_i * \sum_i b_i$ and give an additive share of the output
    /// to each party.
    async fn input<T: Ring + RandElement, const ELL: usize>(
        &self,
        sid: SessionId,
        other: PartyId,
        b: T,
    ) -> Result<Vector<T, ELL>, CheatOrUnexpectedError>
    where [u8; ELL + 1]: ,
          [u8; ELL + 2]: ,
    {

        //const RHO<T: Ring>: usize = (T::BYTES*8).div_ceil(secParam);
        //const KSI<T: Ring>: usize = T::BYTES*8 + 2*statParam;
        //self.input_internal::<T, ELL, {KSI::<T>}, {RHO::<T>}>(sid, other, b).await

        // const generics don't let me actually be as generic as I want, so instead I have a
        // constant number of generics
        match T::BYTES {
            16 => {
                // Box the future since with the arrays things sometimes cause stack overflows.
                Box::pin(self.input_internal::<T, ELL, {128+2*statParam}, 1>(sid, other, b)).await
            },
            32 => {
                Box::pin(self.input_internal::<T, ELL, {256+2*statParam}, 2>(sid, other, b)).await
            },
            _ => {
                panic!("unsupported field size {}", T::BYTES);
            }
        }
    }

    async fn multiply<T: Ring + RandElement, const ELL: usize>(
        &self,
        sid: SessionId,
        other: PartyId,
        a: &Vector<T, ELL>,
    ) -> Result<Vector<T, ELL>, CheatOrUnexpectedError>
    where [u8; ELL + 1]: ,
          [u8; ELL + 2]: ,
    {
        match T::BYTES {
            16 => {
                Box::pin(self.multiply_internal::<T, ELL, {128+2*statParam}, 1>(sid, other, a)).await
            },
            32 => {
                Box::pin(self.multiply_internal::<T, ELL, {256+2*statParam}, 2>(sid, other, a)).await
            },
            _ => {
                panic!("unsupported field size {}", T::BYTES);
            }
        }
    }
}

impl<FN: AsyncNet, FOte: AsyncOt> Dkls23VolePlayer<FN, FOte> {
    async fn input_internal<T: Ring + RandElement, const ELL: usize, const KSI: usize, const RHO: usize>(
        &self,
        sid: SessionId,
        other: PartyId,
        b: T,
    ) -> Result<Vector<T, ELL>, CheatOrUnexpectedError>
    where [u8; ELL + RHO]: ,
    {
        trace!("vole input ({}, {other}, {sid:?})", self.party_id);

        let otsid = sid.derive_ssid(FuncId::Fvole);

        let now = std::time::Instant::now();

        let (bits, gammas): (_,Vector<Vector<T, { ELL + RHO }>, KSI>) = self
            .ote
            .rand_recv(otsid, other)
            .await
            .with_context(|| self.err(sid, "failed to receive OT messages"))?;

        log::info!("Vole Input generating {KSI} ots of {} field elements took {}ms", ELL+RHO, now.elapsed().as_millis()); 
        let gammas = gammas.into();

        let mut buf = vec![0u8; AliceMsg::<ELL, KSI, RHO, T>::BYTES];
        let _ = self
            .net
            .recv_from_local(other, FuncId::Fvole, sid, &mut buf[..])
            .await
            .with_context(|| self.err(sid, "failed to receive alice correction message"))?;

        let FWrap((at, eta, mu)) = AliceMsg::from_bytes(&buf);

        let gadget: Vector<T, KSI> = internal::gadget(sid);
        let brand: T = gadget
            .into_iter()
            .zip(bits.clone().into_iter())
            .map(|(g, b)| if b { g } else { T::zero() })
            .sum();

        let bdiff = b - brand;
        bdiff.to_bytes(&mut buf);
        // TODO: this can be done in parallel
        let _ = self
            .net
            .send_to_local(other, FuncId::Fvole, sid, &buf[..T::BYTES])
            .await
            .with_context(|| self.err(sid, "failed to send input derandomize message"))?;

        let d = internal::bob_multiply(sid, &at, &eta, mu, &bits, &gammas).with_context(|| {
            self.cheat(
                sid,
                Some(other),
                "Alice cheated in vole correction".to_string(),
            )
        })?;

        log::info!("Vole Input size {ELL} took {}ms", now.elapsed().as_millis()); 

        Ok(d)

    }

    async fn multiply_internal<T: Ring + RandElement, const ELL: usize, const KSI: usize, const RHO: usize>(
        &self,
        sid: SessionId,
        other: PartyId,
        a: &Vector<T, ELL>,
    ) -> Result<Vector<T, ELL>, CheatOrUnexpectedError>
    where [u8; ELL + RHO]: ,
    {
        let otsid = sid.derive_ssid(FuncId::Fvole);

        let now = std::time::Instant::now();
        let (alpha_0, alpha_1): (Vector<Vector<T, {ELL + RHO}>, KSI>, _) =
            self
            .ote
            .rand_send(otsid, other)
            .await
            .with_context(|| self.err(sid, "failed to receive OT messages"))?;
        log::info!("Vole Multiply generating {KSI} ots of {} field elements took {}ms", ELL+RHO, now.elapsed().as_millis()); 

        let alpha_0: Matrix<T, KSI, _> = alpha_0.into();
        let alpha_1: Matrix<T, KSI, _> = alpha_1.into();

        let (mut c, at, eta, mu) = internal::alice_choice(sid, a, &alpha_0, &alpha_1);
        let msg: AliceMsg<ELL, KSI, RHO, T> = FWrap((at, eta, mu));

        let mut buf = vec![0u8; AliceMsg::<ELL, KSI, RHO, T>::BYTES];
        msg.to_bytes(&mut buf[..]);
        // TODO: this can be done in parallel
        let _ = self
            .net
            .send_to_local(other, FuncId::Fvole, sid, &buf)
            .await
            .with_context(|| self.err(sid, "failed to send correction message"))?;

        let _ = self
            .net
            .recv_from_local(other, FuncId::Fvole, sid, &mut buf[..T::BYTES])
            .await
            .with_context(|| self.err(sid, "failed to receive derandomize message from Bob"))?;

        let diff = T::from_bytes(&buf[..T::BYTES]);
        c += a * diff;

        log::info!("Vole Multiply size {ELL} took {}ms", now.elapsed().as_millis()); 

        Ok(c)
    }
}


pub(crate) mod internal {
    use super::*;

    pub fn gadget<R: Ring + RandElement, const KSI: usize>(sid: SessionId) -> Vector<R, KSI> {
        let gadget = RO::new()
            .add_context("Dkls23Vole Gadget")
            .add_context(&sid.as_bytes())
            .generate(&[]);

        gadget
    }

    pub fn theta<
        R: RandElement,
        T: ToFromBytes,
        const KSI: usize,
        const ELL: usize,
        const RHO: usize,
    >(
        sid: SessionId,
        at: &T,
    ) -> Matrix<R, ELL, RHO> {
        let theta = RO::new()
            .add_context("Dkls23Vole Theta")
            .add_context(&sid.as_bytes())
            .generate_read(at);
        theta
    }

    pub fn alice_choice<
        R: Ring + RandElement,
        const KSI: usize,
        const ELL: usize,
        const RHO: usize,
    >(
        sid: SessionId,
        a: &Vector<R, ELL>,
        alpha_0: &Matrix<R, KSI, { ELL + RHO }>,
        alpha_1: &Matrix<R, KSI, { ELL + RHO }>,
    ) -> (
        Vector<R, ELL>,
        Matrix<R, KSI, { ELL + RHO }>,
        Vector<R, RHO>,
        [u8; 32],
    ) 
        where [u8; ELL + RHO]: ,
    {
        let gadget = gadget::<R, KSI>(sid);

        let (a0_ell, a0_rho): (Matrix<R, _, ELL>, Matrix<R, _, RHO>) = alpha_0.split_col();

        let c = -gadget.vm(&a0_ell);

        let ahat: Vector<R, RHO> = {
            let mut rng = rand::thread_rng();
            RandElement::rand(&mut rng)
        };
        let av: Vector<R, ELL> = a.clone().into();
        let a_ahat: Vector<R, { ELL + RHO }> = av.concat(&ahat);

        let mut at = alpha_0 - alpha_1;
        // TODO: syntax for broadcasting a row add?
        at.rows_mut().for_each(|r| {
            *r += &a_ahat;
        });

        let theta = theta::<R, _, KSI, ELL, RHO>(sid, &at);

        let eta = ahat + av.vm(&theta);
        // (Ksi x Rho) + (Ksi x Ell)*(Ell x Rho)
        let muv: Matrix<R, KSI, RHO> = a0_rho + a0_ell.mm(&theta);

        let mu = RO::new()
            .add_context("Dkls23Vole mu")
            .add_context(&sid.as_bytes())
            .generate_read(&muv);

        (c, at, eta, mu)
    }

    pub fn bob_multiply<
        R: Ring + RandElement,
        const KSI: usize,
        const ELL: usize,
        const RHO: usize,
    >(
        sid: SessionId,
        alpha_t: &Matrix<R, KSI, { ELL + RHO }>,
        eta: &Vector<R, RHO>,
        mu: [u8; 32],
        beta: &Vector<bool, KSI>, 
        gamma: &Matrix<R, KSI, { ELL + RHO }>,
    ) -> Option<Vector<R, ELL>>
        where [u8; ELL + RHO]: ,
    {

        let beta = beta.clone().as_array();
        // gamma + \beta*\alpha
        let mut d = gamma.clone();
        d.rows_mut()
            .zip(beta.iter())
            .zip(alpha_t.rows())
            .for_each(|((di, b), a)| {
                if *b {
                    *di += a;
                }
            });

        let (d_ell, d_rho) = d.split_col();

        let theta = theta::<R, _, KSI, ELL, RHO>(sid, alpha_t);
        let mut muv: Matrix<R, KSI, RHO> = d_rho + d_ell.mm(&theta);

        // TODO: syntax for broadcasting a row add?
        muv.rows_mut().zip(beta.iter()).for_each(|(r, &b)| {
            if b {
                *r -= eta;
            }
        });

        let mu_prime: [u8; 32] = RO::new()
            .add_context("Dkls23Vole mu")
            .add_context(&sid.as_bytes())
            .generate_read(&muv);

        if mu != mu_prime {
            return None;
        }

        let gadget: Vector<R, KSI> = gadget(sid);
        let d = gadget.vm(&d_ell);

        Some(d)
    }
}

impl<FN, FOte> Dkls23VolePlayer<FN, FOte> {
    pub fn new(id: PartyId, net: Arc<FN>, ote: Arc<FOte>) -> Self {
        Self {
            party_id: id,
            net,
            ote
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::{
        func_net::tests::{get_test_party_infos, build_test_nets},
        func_eot::zzzr23_eot::tests::build_test_eots,
        func_cote::softspoken_ote::tests::build_test_ots,
        //ff2_128::FF2_128,
        p256::P256Scalar,
    };
    use std::sync::Arc;
    use tokio::task::JoinHandle;
    //use test::Bencher;

    pub fn build_test_voles<FN, FOTE>(
        nets: &[Arc<FN>],
        otes: &[Arc<FOTE>],
    ) -> Vec<Arc<Dkls23VolePlayer<FN, FOTE>>> {
        let num = nets.len() as PartyId;
        (1..=num)
            .map(|i| Arc::new(Dkls23VolePlayer::new(i, nets[(i-1) as usize].clone(), otes[(i-1) as usize].clone())))
            .collect()
    }


    enum SendOrRecv<T, const ELL: usize> {
        Send(T, Vector<T, ELL>),
        Recv(Vector<T, ELL>, Vector<T, ELL>)
    }

    async fn run_voles() -> (Result<SendOrRecv<P256Scalar, 10>, &'static str>, Result<SendOrRecv<P256Scalar, 10>, &'static str>) {
        let party_info = get_test_party_infos(2);
        let funcs = vec![FuncId::Fvole, FuncId::Fot, FuncId::Feot];
        let nets = build_test_nets(&party_info, funcs).await;
        let eots = build_test_eots(&nets);
        let ots = build_test_ots(&eots, &nets);
        let voles = build_test_voles(&nets, &ots);

        let sid = SessionId::new(FuncId::Ftest);


        let t1: JoinHandle<Result<_, &'static str>> = tokio::spawn({
            let vole = voles[0].clone();
            async move {
                let _ = vole.init(sid, 2).await.map_err(|_| "failed to init")?;
                let delta = {
                    let mut rng = rand::thread_rng();
                    P256Scalar::rand(&mut rng)
                };
                let msgs  = vole.input::<_, 10>(sid, 2, delta.clone()).await.map_err(|_| "failed to send")?;

                Ok(SendOrRecv::Send(delta, msgs))
            }
        });

        let t2: JoinHandle<Result<_, &'static str>> = tokio::spawn({
            let vole = voles[1].clone();
            async move {
                let _ = vole.init(sid, 1).await.map_err(|_| "failed to init")?;
                let bs = {
                    let mut rng = rand::thread_rng();
                    Vector::rand(&mut rng)
                };
                let msgs  = vole.multiply::<_, 10>(sid, 1, &bs).await.map_err(|_| "failed to multiply")?;
                Ok(SendOrRecv::Recv(bs, msgs))
            }
        });

        tokio::try_join!(t1, t2).expect("Error running voles")
    }

    #[test]
    fn test_vole_2() {
        let rt = tokio::runtime::Builder::new_multi_thread()
                            .enable_all()
                            .thread_stack_size(1 << 24)
                            .build()
                            .expect("couldn't build runtime");

        rt.block_on(Box::pin(test_vole()));
    }

    //#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_vole() {

        //todo!("Compiler ICE when trying to compile const generics");
        let _ = env_logger::builder().is_test(true).try_init();

        match run_voles().await {
            (Ok(SendOrRecv::Send(b, cs)), Ok(SendOrRecv::Recv(a_s, ds))) => {
                let (a_s, b, cs, ds) = dbg!(a_s, b, cs, ds);

                assert_eq!(a_s*b, cs+ds);
            },
            _ => { panic!("unexpected") }
        };
    }

}
