//! L'attention de la génération, qui lit ses clés dans le cache.

use super::{Operation, Reference, operands};
use crate::backend::{Float, Primitives};

/// L'attention de nouveaux tokens sur le passé de leurs séquences, tel que
/// le cache de génération le garde.
///
/// Le cache range les clés et les valeurs de toutes les séquences en cours
/// dans deux tables, une ligne par position : `keys` `[V, Hk, Dk]` et
/// `values` `[V, Hk, Dv]`. Une séquence n'y occupe pas des lignes voisines,
/// mais celles que le cache lui a données au fil de la génération. La liste
/// de ses lignes, dans l'ordre des positions, est la ligne `s` de `slots`
/// `[S, L]`, complétée au-delà de sa longueur par des valeurs quelconques.
///
/// Les requêtes `q` `[R, H, Dk]` sont celles de `R` nouveaux tokens, de
/// séquences quelconques : un token par séquence quand on génère par lots,
/// tous les tokens d'une amorce quand on la lit. La ligne `r` de `rows`
/// `[R, 2]` dit à quelle séquence appartient la requête `r`, et combien de
/// positions de cette séquence elle voit : les siennes et celles qui la
/// précèdent. La sortie est `[R, H, Dv]`.
///
/// Les numéros de séquences et de lignes sont des entiers rangés dans des
/// tenseurs ordinaires, comme pour
/// [`Primitives::write_rows`](crate::backend::Primitives::write_rows).
///
/// Il peut y avoir moins de têtes de clés que de têtes de requêtes : la
/// tête de requête `h` lit alors la tête de clé `h / (H / Hk)`.
///
/// Avec `value_dim`, seules les `value_dim` premières composantes de
/// chaque ligne de `values` servent de valeur. L'attention latente range
/// ainsi dans une seule table son latent, qui sert de valeur, suivi de la
/// partie tournée de sa clé : elle passe la même table comme clés et comme
/// valeurs.
///
/// L'opération ne sert qu'à la génération, qui ne calcule pas de gradient :
/// elle n'a pas de rétropropagation.
pub struct CachedAttention {
    pub scale: f64,
    pub value_dim: Option<usize>,
}

impl Operation for CachedAttention {
    const NAME: &'static str = "cached_attention";
}

/// Les dimensions d'une attention sur le cache.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CachedAttentionDims {
    /// Nombre de requêtes, `R`.
    pub queries: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    /// Largeur des lignes de la table des valeurs : `value_dim`, ou plus.
    pub value_stride: usize,
    /// Nombre de séquences de la table `slots`, `S`.
    pub sequences: usize,
    /// Longueur des lignes de la table `slots`, `L`.
    pub max_len: usize,
}

impl CachedAttention {
    /// Vérifie les formes des entrées et rend leurs dimensions.
    pub fn dims<B: Primitives>(&self, backend: &B, inputs: &[&B::Tensor]) -> CachedAttentionDims {
        let [q, keys, values, slots, rows] = operands(Self::NAME, inputs);
        let (q, keys, values) = (backend.shape(q), backend.shape(keys), backend.shape(values));
        let (slots, rows) = (backend.shape(slots), backend.shape(rows));
        assert!(
            q.len() == 3 && keys.len() == 3 && values.len() == 3,
            "cached_attention : q {q:?}, clés {keys:?} et valeurs {values:?} doivent être \
             à trois dimensions",
        );
        assert!(
            keys[0] == values[0] && keys[1] == values[1] && q[2] == keys[2],
            "cached_attention : q {q:?}, clés {keys:?} et valeurs {values:?} incompatibles",
        );
        assert!(
            keys[1] > 0 && q[1].is_multiple_of(keys[1]),
            "cached_attention : {} têtes de requêtes pour {} têtes de clés",
            q[1],
            keys[1],
        );
        assert!(
            slots.len() == 2 && rows == [q[0], 2],
            "cached_attention : table {slots:?} et lignes {rows:?} pour {} requêtes",
            q[0],
        );
        let value_dim = self.value_dim.unwrap_or(values[2]);
        assert!(
            value_dim <= values[2],
            "cached_attention : {value_dim} composantes de valeur dans des lignes de {}",
            values[2],
        );
        CachedAttentionDims {
            queries: q[0],
            heads: q[1],
            kv_heads: keys[1],
            key_dim: q[2],
            value_dim,
            value_stride: values[2],
            sequences: slots[0],
            max_len: slots[1],
        }
    }
}

impl Reference for CachedAttention {
    /// Chaque requête, l'une après l'autre : les lignes de sa séquence sont
    /// rassemblées, puis l'attention est celle du softmax ordinaire.
    fn reference<B: Primitives>(&self, backend: &B, inputs: &[&B::Tensor]) -> B::Tensor {
        let dims = self.dims(backend, inputs);
        let [q, keys, values, slots, rows] = operands(Self::NAME, inputs);
        let integers = |tensor: &B::Tensor| -> Vec<usize> {
            let values = backend.to_vec(tensor);
            values.iter().map(|value| value.to_f64() as usize).collect()
        };
        let (slots, rows) = (integers(slots), integers(rows));
        let group = dims.heads / dims.kv_heads;
        let scale = backend.scalar(B::Scalar::from_f64(self.scale));
        // `[S, Hk, D]` → `[H, S, D]` : chaque tête de clé sert `group`
        // têtes de requêtes.
        let per_head = |table: &B::Tensor, lines: &[u32], dim: usize| {
            let count = lines.len();
            let gathered = backend.gather_rows(table, lines);
            let spread = backend.expand(
                &backend.reshape(&gathered, &[count, dims.kv_heads, 1, dim]),
                &[count, dims.kv_heads, group, dim],
            );
            backend.transpose(&backend.reshape(&spread, &[count, dims.heads, dim]), 0, 1)
        };
        // Les valeurs, réduites à leurs `value_dim` premières composantes.
        let values = if dims.value_dim == dims.value_stride {
            values.clone()
        } else {
            let data = backend.to_vec(values);
            let kept = data
                .chunks(dims.value_stride)
                .flat_map(|line| line[..dims.value_dim].to_vec())
                .collect();
            let lines = backend.shape(values)[0];
            backend.from_vec(kept, &[lines, dims.kv_heads, dims.value_dim])
        };
        let q = backend.reshape(q, &[dims.queries, dims.heads * dims.key_dim]);
        let mut out: Option<B::Tensor> = None;
        for row in 0..dims.queries {
            let (sequence, count) = (rows[2 * row], rows[2 * row + 1]);
            assert!(
                sequence < dims.sequences && (1..=dims.max_len).contains(&count),
                "cached_attention : la requête {row} voit {count} positions de la séquence \
                 {sequence}",
            );
            let lines: Vec<u32> = slots[sequence * dims.max_len..][..count]
                .iter()
                .map(|&line| line as u32)
                .collect();
            let keys = per_head(keys, &lines, dims.key_dim);
            let values = per_head(&values, &lines, dims.value_dim);
            let query = backend.reshape(
                &backend.gather_rows(&q, &[row as u32]),
                &[dims.heads, 1, dims.key_dim],
            );
            let scores = backend.mul(
                &backend.matmul(&query, &backend.transpose(&keys, 1, 2)),
                &scale,
            );
            let mixed = backend.matmul(&backend.softmax(&scores), &values);
            let mixed = backend.reshape(&mixed, &[1, dims.heads * dims.value_dim]);
            out = Some(match out {
                Some(done) => backend.concat_rows(&done, &mixed),
                None => mixed,
            });
        }
        let out = out.expect("cached_attention : aucune requête");
        backend.reshape(&out, &[dims.queries, dims.heads, dims.value_dim])
    }

    fn reference_backward<B: Primitives>(
        &self,
        _: &B,
        _: &[&B::Tensor],
        _: &B::Tensor,
        _: &B::Tensor,
    ) -> Vec<Option<B::Tensor>> {
        panic!("cached_attention ne sert qu'à la génération : elle n'a pas de gradient")
    }
}
