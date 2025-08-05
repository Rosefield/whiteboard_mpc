use crate::{
    field::{FWrap, Group, RandElement},
    prg::Prg,
    ro::RO,
};

/// Builds a punctured PRF according to the description in [Roy22] https://eprint.iacr.org/2022/192.pdf Figure 13
/// Returns the leaves of the tree, and the traces for each level of the tree
pub fn build_pprf<G: Group + RandElement, const N: usize>(
    prfs: &[[G; N]],
) -> (Vec<G>, Vec<[G; N]>) {
    let depth = prfs.len();
    assert!(N > 0 && depth > 0);

    // try to allocate all of the capacity needed up front
    // In theory collect will attempt to reuse allocations for vecs
    let mut parents = Vec::with_capacity(N.pow(depth as u32));
    #[allow(unused_assignments)]
    let mut children = Vec::with_capacity(N.pow((depth - 1) as u32));
    let mut traces = Vec::with_capacity(depth);

    // first layer of seeds is just the original PRF evaluations
    parents.extend_from_slice(&prfs[0]);

    let prg = Prg::new();

    for i in 1..depth {
        // for each subsequent layer of the tree the children of a node are the PRG expansion of
        // the parent seed.

        let mut bytes = vec![0u8; G::BYTES];

        children = parents
            .iter()
            .map(|p| {
                p.to_bytes(&mut bytes);

                let c = prg.generate::<_, FWrap<[G; N]>>(&bytes);
                c
            })
            .collect();

        let mut trace = FWrap(prfs[i].clone());
        trace += children.iter().sum::<FWrap<[G; N]>>();
        traces.push(trace.0);

        parents = children.into_iter().flat_map(|c| c.0).collect();
    }

    (parents, traces)
}

/// Evaluates the punctured PRF given at all points except that given by the N_ary missing_idxs
/// Returns the missing leaf index, and the leaves in order with no hole at the missing index
pub fn eval_pprf<G: Group + RandElement, const N: usize>(
    prfs: &[[G; N - 1]],
    missing_idxs: &[usize],
    traces: &[[G; N]],
) -> (usize, Vec<G>) {
    let depth = prfs.len();
    assert!(N > 0 && depth > 0);
    assert!(missing_idxs.len() == depth && traces.len() == depth - 1);
    assert!(missing_idxs.iter().all(|i| *i < N));

    // try to allocate all of the capacity needed up front
    // In theory collect will attempt to reuse allocations for vecs
    let mut parents = Vec::with_capacity(N.pow(depth as u32));
    #[allow(unused_assignments)]
    let mut children = Vec::with_capacity(N.pow((depth - 1) as u32));

    // first layer of seeds is just the original PRF evaluations
    parents.extend_from_slice(&prfs[0]);

    let prg = Prg::new();

    let mut missing_leaf = missing_idxs[0];

    for i in 1..depth {
        // for each subsequent layer of the tree the children of a node are the PRG expansion of
        // the parent seed.
        let mut bytes = vec![0u8; G::BYTES];

        children = parents
            .iter()
            .map(|p| {
                p.to_bytes(&mut bytes);

                let c = prg.generate::<_, FWrap<[G; N]>>(&bytes);
                c
            })
            .collect();

        // recover all but one of the children seeds using the provided trace
        let mut missing_seeds = FWrap(traces[i - 1].clone());
        missing_seeds += children.iter().sum::<FWrap<[G; N]>>();

        let remaining_seeds = (0..N)
            .into_iter()
            .filter(|j| *j != missing_idxs[i])
            .enumerate()
            .map(|(orig, target)| missing_seeds.0[target].clone() + &prfs[i][orig]);

        // collect the children that we do know
        parents = children.into_iter().flat_map(|c| c.0).collect();

        let row_idx = N * missing_leaf;
        // and insert in order
        parents.splice(row_idx..row_idx, remaining_seeds);

        missing_leaf = N * missing_leaf + missing_idxs[i];
    }

    (missing_leaf, parents)
}

pub fn prove_modify_pprf<G: Group + RandElement>(leaves: &mut [G]) -> ([u8; 32], [u8; 32]) {
    // In Section 6.1. it is specified that the PRG must be collision resistant,
    // so I will explicitly use the RO/hash based primitive instead of the stream-cipher based PRG
    // even though it might itself be collision resistant.
    let ro = RO::new().add_context("prove/verify pprf elements");
    let mut hasher = RO::new().add_context("prove/verify hash");

    let mut bytes = vec![0u8; G::BYTES];

    let mut t = FWrap([0u8; 32]);

    leaves.iter_mut().for_each(|l| {
        l.to_bytes(&mut bytes);

        let FWrap((sy, ly)): FWrap<([u8; 32], G)> = ro.generate(&bytes);

        *l = ly;
        t ^= FWrap(sy);

        hasher.update_context(&sy);
    });

    // generate final hash
    let s = hasher.generate(&[]);

    (s, t.0)
}

pub fn verify_modify_pprf<G: Group + RandElement>(
    leaves: &mut [G],
    missing_leaf: usize,
    proof: [u8; 32],
) -> [u8; 32] {
    let ro = RO::new().add_context("prove/verify pprf elements");

    let mut bytes = vec![0u8; G::BYTES];

    let mut t = FWrap(proof);

    let mut seeds = vec![FWrap([0u8; 32]); leaves.len() + 1];

    let mut update_leaf = |(s, l): (&mut _, &mut G)| {
        l.to_bytes(&mut bytes);

        let FWrap((sy, ly)) = ro.generate(&bytes);

        *l = ly;
        t ^= FWrap(sy);

        *s = FWrap(sy);
    };

    // generate known seeds and update known leaves
    seeds[..missing_leaf]
        .iter_mut()
        .zip(leaves[..missing_leaf].iter_mut())
        .for_each(&mut update_leaf);
    seeds[missing_leaf + 1..]
        .iter_mut()
        .zip(leaves[missing_leaf..].iter_mut())
        .for_each(&mut update_leaf);

    // fill in missing seed using the proof
    seeds[missing_leaf] = seeds.iter().fold(FWrap(proof), |acc, n| acc ^ n);

    // now that we know all of the seeds calculate the final hash
    let mut hasher = RO::new().add_context("prove/verify hash");
    for s in seeds {
        hasher.update_context(&s.0);
    }

    // generate final hash
    let s = hasher.generate(&[]);

    s
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::ff2_128::FF2_128 as F;

    #[test]
    fn test_pprf_evals_match() {
        let mut rng = rand::thread_rng();

        let depth = 3;

        let prfs: Vec<[F; 2]> = (0..depth)
            .into_iter()
            .map(|_| std::array::from_fn(|_| F::rand(&mut rng)))
            .collect();

        let (mut leaves, trace) = build_pprf(&prfs);

        let missing = [0, 1, 0];

        let pprfs: Vec<[F; 1]> = prfs
            .into_iter()
            .enumerate()
            .map(|(i, f)| [f[1 - missing[i]]])
            .collect();

        let (missing_leaf, mut punctured_leaves) = eval_pprf(&pprfs, &missing, &trace);

        assert_eq!(missing_leaf, 2);
        let all_known_leaves_equal = leaves
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 2)
            .zip(punctured_leaves.iter())
            .all(|((_, l), r)| *l == *r);
        assert!(all_known_leaves_equal);

        let (s, t) = prove_modify_pprf(&mut leaves);

        let sprime = verify_modify_pprf(&mut punctured_leaves, missing_leaf, t);

        assert_eq!(s, sprime);

        let updated_known_leaves_equal = leaves
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 2)
            .zip(punctured_leaves.iter())
            .all(|((_, l), r)| *l == *r);
        assert!(updated_known_leaves_equal);
    }
}
