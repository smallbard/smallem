//! Les opérations par ligne du dernier axe : les réductions, un warp par
//! ligne (voir `row_stats`), et les kernels élément par élément qui s'en
//! servent, softmax, normalisations, entropie croisée.

use super::*;

#[cuda_module]
pub mod device {
    use super::*;

    /// Somme le long d'un axe. Le tenseur est vu comme `[outer, n, inner]`
    /// et la sortie comme `[outer, inner]` : chaque thread additionne les
    /// `n` éléments de sa colonne.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn sum_axis(a: &[f32], n: u32, inner: u32, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let flat = idx.get() as u32;
        if let Some(out_elem) = out.get_mut(idx) {
            let o = flat / inner;
            let i = flat % inner;
            let mut total = 0.0;
            for k in 0..n {
                total += a[((o * n + k) * inner + i) as usize];
            }
            *out_elem = total;
        }
    }

    /// Somme par paquets : le thread `t` additionne les éléments
    /// `[t × chunk, (t + 1) × chunk)`. Répété jusqu'à ce qu'il ne reste
    /// qu'un élément, cela donne la somme de tout un tenseur.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn sum_chunks(a: &[f32], chunk: u32, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let start = idx.get() * chunk as usize;
        if let Some(out_elem) = out.get_mut(idx) {
            // Le dernier paquet peut être incomplet.
            let mut end = start + chunk as usize;
            if end > a.len() {
                end = a.len();
            }
            let mut total = 0.0;
            for i in start..end {
                total += a[i];
            }
            *out_elem = total;
        }
    }

    /// Les deux réductions dont le softmax a besoin, pour chaque ligne de
    /// largeur `width` : son maximum, et `ln(Σ exp(x - max))`.
    ///
    /// Un warp par ligne, comme toutes les réductions par ligne de ce
    /// fichier : le thread de rang `l` dans son warp parcourt les éléments
    /// `l`, `l + 32`, `l + 64`… de la ligne, puis les 32 résultats partiels
    /// sont réunis par [`warp_sum`] ou [`warp_max`]. Les 32 threads lisent à
    /// chaque tour 32 éléments voisins, ce que la mémoire sert en une seule
    /// transaction. Avec un seul thread par ligne, chacun lirait seul dans
    /// son coin, et la ligne serait parcourue 32 fois plus lentement.
    ///
    /// Le lancement doit donc créer 32 threads par ligne.
    ///
    /// Les deux valeurs sont gardées séparées. Leur somme est `ln(Σ exp(x))`,
    /// mais l'additionner tout de suite noierait le second terme, petit,
    /// dans le maximum, qui peut être grand : on perdrait des chiffres.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn row_stats(
        a: &[f32],
        width: u32,
        mut row_max: DisjointSlice<f32>,
        mut row_log_sum: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get();
        let (row, lane) = (index / WARP, index % WARP);
        // Vrai ou faux pour tout le warp à la fois.
        if row < row_max.len() {
            let start = row * width as usize;
            let end = start + width as usize;
            let (max, total) = warp_row_max_and_exp_sum(start, end, lane, |i| a[i]);
            write_warp_row(&mut row_max, max);
            write_warp_row(&mut row_log_sum, total.ln());
        }
    }

    /// Softmax ou log-softmax d'un élément à partir des réductions de sa
    /// ligne : `ln softmax(x) = (x - max) - ln(Σ exp(x - max))`. Soustraire
    /// le maximum d'abord garde la précision, quelle que soit la taille des
    /// valeurs.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn softmax_rows(
        mode: SoftmaxMode,
        a: &[f32],
        row_max: &[f32],
        row_log_sum: &[f32],
        width: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let row = flat / width as usize;
            let log_probability = (a[flat] - row_max[row]) - row_log_sum[row];
            *out_elem = match mode {
                SoftmaxMode::Log => log_probability,
                SoftmaxMode::Probability => log_probability.exp(),
            };
        }
    }

    /// Les deux réductions de la normalisation, pour chaque ligne de
    /// largeur `width` : sa moyenne, et `1 / √(variance + epsilon)`. Un
    /// warp par ligne.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn row_moments(
        a: &[f32],
        width: u32,
        epsilon: f32,
        mut mean: DisjointSlice<f32>,
        mut inverse_std: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get();
        let (row, lane) = (index / WARP, index % WARP);
        if row < mean.len() {
            let start = row * width as usize;
            let end = start + width as usize;
            let row_mean = warp_row_sum(start, end, lane, |i| a[i]) / width as f32;
            let squares = warp_row_sum(start, end, lane, |i| {
                let centered = a[i] - row_mean;
                centered * centered
            });
            write_warp_row(&mut mean, row_mean);
            write_warp_row(
                &mut inverse_std,
                1.0 / (squares / width as f32 + epsilon).sqrt(),
            );
        }
    }

    /// `1 / √(moyenne(x²) + epsilon)` pour chaque ligne de largeur `width` :
    /// la réduction de RMSNorm. Un warp par ligne.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn row_inverse_rms(a: &[f32], width: u32, epsilon: f32, mut out: DisjointSlice<f32>) {
        let index = thread::index_1d().get();
        let (row, lane) = (index / WARP, index % WARP);
        if row < out.len() {
            let start = row * width as usize;
            let end = start + width as usize;
            let squares = warp_row_sum(start, end, lane, |i| a[i] * a[i]);
            write_warp_row(&mut out, 1.0 / (squares / width as f32 + epsilon).sqrt());
        }
    }

    /// Multiplie chaque élément par le facteur de sa ligne.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn scale_rows(a: &[f32], row_scale: &[f32], width: u32, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = a[flat] * row_scale[flat / width as usize];
        }
    }

    /// Rétropropagation de RMSNorm, un thread par élément :
    /// `s × (g - x̂ × moyenne(g × x̂))`. `grad_dot` est la somme par ligne
    /// de `g × x̂`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn rms_grad_rows(
        normalized: &[f32],
        grad: &[f32],
        grad_dot: &[f32],
        inverse_rms: &[f32],
        width: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let row = flat / width as usize;
            let mean_projection = grad_dot[row] / width as f32;
            *out_elem = inverse_rms[row] * (grad[flat] - normalized[flat] * mean_projection);
        }
    }

    /// Normalisation d'un élément à partir des réductions de sa ligne :
    /// `(x - moyenne) × écart-type inverse`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn normalize_rows(
        a: &[f32],
        mean: &[f32],
        inverse_std: &[f32],
        width: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let row = flat / width as usize;
            *out_elem = (a[flat] - mean[row]) * inverse_std[row];
        }
    }

    /// Produit scalaire de chaque ligne de `a` par la ligne correspondante
    /// de `b` : `Σ a × b`. Un warp par ligne.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn row_dot(a: &[f32], b: &[f32], width: u32, mut out: DisjointSlice<f32>) {
        let index = thread::index_1d().get();
        let (row, lane) = (index / WARP, index % WARP);
        if row < out.len() {
            let start = row * width as usize;
            let end = start + width as usize;
            write_warp_row(&mut out, warp_row_sum(start, end, lane, |i| a[i] * b[i]));
        }
    }

    /// Somme de chaque ligne de largeur `width`. Un warp par ligne.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn row_sum(a: &[f32], width: u32, mut out: DisjointSlice<f32>) {
        let index = thread::index_1d().get();
        let (row, lane) = (index / WARP, index % WARP);
        if row < out.len() {
            let start = row * width as usize;
            let end = start + width as usize;
            write_warp_row(&mut out, warp_row_sum(start, end, lane, |i| a[i]));
        }
    }

    /// Rétropropagation de la normalisation, un thread par élément :
    ///
    /// ```text
    /// s × (g - moyenne(g) - x̂ × moyenne(g × x̂))
    /// ```
    ///
    /// `grad_sum` et `grad_dot` sont les sommes par ligne de `g` et de
    /// `g × x̂` ; on les divise ici par `width` pour en faire des moyennes.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    #[allow(clippy::too_many_arguments)]
    pub fn normalize_grad_rows(
        normalized: &[f32],
        grad: &[f32],
        grad_sum: &[f32],
        grad_dot: &[f32],
        inverse_std: &[f32],
        width: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let row = flat / width as usize;
            let mean_grad = grad_sum[row] / width as f32;
            let mean_projection = grad_dot[row] / width as f32;
            *out_elem =
                inverse_std[row] * (grad[flat] - mean_grad - normalized[flat] * mean_projection);
        }
    }

    /// Comme `row_stats`, pour le softmax causal de scores `[…, N, T]` :
    /// `queries` requêtes face à `width` positions, dont elles sont les
    /// dernières. La ligne `i` d'une matrice est à la position
    /// `width - queries + i` et ne tient compte que des colonnes jusqu'à la
    /// sienne, les suivantes étant son futur. Les scores sont multipliés par
    /// `scale` avant tout calcul.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn causal_row_stats(
        a: &[f32],
        width: u32,
        queries: u32,
        scale: f32,
        mut row_max: DisjointSlice<f32>,
        mut row_log_sum: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get();
        let (row, lane) = (index / WARP, index % WARP);
        if row < row_max.len() {
            let start = row * width as usize;
            // Position de cette ligne parmi les `width` positions : c'est
            // aussi le nombre de colonnes visibles, moins une.
            let position = (width - queries) as usize + row % queries as usize;
            let end = start + position + 1;
            let (max, total) = warp_row_max_and_exp_sum(start, end, lane, |i| a[i] * scale);
            write_warp_row(&mut row_max, max);
            write_warp_row(&mut row_log_sum, total.ln());
        }
    }

    /// Softmax causal d'un élément à partir des réductions de sa ligne. Les
    /// colonnes du futur reçoivent exactement 0.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn causal_softmax_rows(
        a: &[f32],
        row_max: &[f32],
        row_log_sum: &[f32],
        width: u32,
        queries: u32,
        scale: f32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let (width, queries) = (width as usize, queries as usize);
            let row = flat / width;
            let col = flat % width;
            let position = width - queries + row % queries;
            *out_elem = if col > position {
                0.0
            } else {
                ((a[flat] * scale - row_max[row]) - row_log_sum[row]).exp()
            };
        }
    }

    /// Rétropropagation d'un softmax, un thread par élément :
    /// `scale × p × (g - Σ g × p)`. `row_dot` contient `Σ g × p` par ligne.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn softmax_grad_rows(
        probabilities: &[f32],
        grad: &[f32],
        row_dot: &[f32],
        width: u32,
        scale: f32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let row = flat / width as usize;
            *out_elem = scale * probabilities[flat] * (grad[flat] - row_dot[row]);
        }
    }

    /// Perte de chaque ligne de logits : `-ln softmax(x)[cible]`, soit
    /// `ln(Σ exp(x - max)) - (x[cible] - max)`. Un warp par ligne.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn cross_entropy_rows(
        logits: &[f32],
        targets: &[u32],
        width: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get();
        let (row, lane) = (index / WARP, index % WARP);
        if row < out.len() {
            let start = row * width as usize;
            let end = start + width as usize;
            let (max, total) = warp_row_max_and_exp_sum(start, end, lane, |i| logits[i]);
            let target = logits[start + targets[row] as usize];
            write_warp_row(&mut out, total.ln() - (target - max));
        }
    }

    /// Gradient de l'entropie croisée moyenne par rapport aux logits :
    /// `(softmax(x) - onehot(cible)) × scale`, avec `scale = 1 / N`.
    /// Un thread par logit ; `row_max` et `row_log_sum` viennent de
    /// `row_stats`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn cross_entropy_grad(
        logits: &[f32],
        row_max: &[f32],
        row_log_sum: &[f32],
        targets: &[u32],
        width: u32,
        scale: f32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let row = flat / width as usize;
            let col = (flat % width as usize) as u32;
            let probability = ((logits[flat] - row_max[row]) - row_log_sum[row]).exp();
            let expected = if col == targets[row] { 1.0 } else { 0.0 };
            *out_elem = (probability - expected) * scale;
        }
    }
}
