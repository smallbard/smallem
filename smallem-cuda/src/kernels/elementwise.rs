//! Les kernels élément par élément : un thread par élément de sortie. Avec
//! eux, les copies et les additions de lignes de tables, et les pas des
//! optimiseurs.

use super::*;

#[cuda_module]
pub mod device {
    use super::*;

    /// `out[i] = op(a[…], b[…])`, avec broadcasting des deux opérandes vers
    /// la forme de sortie.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn binary(
        op: BinaryOp,
        param: f32,
        a: &[f32],
        layout_a: Layout,
        b: &[f32],
        layout_b: Layout,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get() as u32;
        // Le dernier bloc contient des threads en trop : ils n'ont pas
        // d'élément de sortie et ne font rien.
        if let Some(out_elem) = out.get_mut(idx) {
            let x = a[layout_a.source_index(flat)];
            let y = b[layout_b.source_index(flat)];
            *out_elem = match op {
                BinaryOp::Add => x + y,
                BinaryOp::Sub => x - y,
                BinaryOp::Mul => x * y,
                BinaryOp::Div => x / y,
                BinaryOp::MaskedFill => {
                    if y != 0.0 {
                        param
                    } else {
                        x
                    }
                }
                BinaryOp::GeluBackward => y * float::gelu_slope(x),
                BinaryOp::SiluBackward => y * float::silu_slope(x),
            };
        }
    }

    /// `out[i] = op(a[i])`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn unary(op: UnaryOp, a: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let x = a[i];
            *out_elem = match op {
                UnaryOp::Neg => -x,
                UnaryOp::Exp => x.exp(),
                UnaryOp::Ln => x.ln(),
                UnaryOp::Sqrt => x.sqrt(),
                UnaryOp::Tanh => x.tanh(),
                UnaryOp::Gelu => float::gelu(x),
                UnaryOp::Silu => float::silu(x),
            };
        }
    }

    /// Réarrange les éléments : `out[i] = a[layout.source_index(i)]`.
    /// Avec deux pas échangés, c'est une transposition.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn permute(a: &[f32], layout: Layout, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let flat = idx.get() as u32;
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = a[layout.source_index(flat)];
        }
    }

    /// `out[i] = value` : un tenseur rempli d'une même valeur, sans rien
    /// envoyer depuis l'hôte.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn fill(value: f32, mut out: DisjointSlice<f32>) {
        if let Some(out_elem) = out.get_mut(thread::index_1d()) {
            *out_elem = value;
        }
    }

    /// Met `b` à la suite de `a` : `out = a ++ b`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn concat(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = if i < a.len() { a[i] } else { b[i - a.len()] };
        }
    }

    /// Concaténation le long du dernier axe : chaque ligne de la sortie est
    /// une ligne de `a`, de largeur `width_a`, suivie de la ligne de même
    /// rang de `b`, de largeur `width_b`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn concat_last(
        a: &[f32],
        b: &[f32],
        width_a: u32,
        width_b: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let (width_a, width_b) = (width_a as usize, width_b as usize);
            let row = flat / (width_a + width_b);
            let column = flat % (width_a + width_b);
            *out_elem = if column < width_a {
                a[row * width_a + column]
            } else {
                b[row * width_b + column - width_a]
            };
        }
    }

    /// L'inverse de `concat_last` : de chaque ligne de `a`, de largeur
    /// `total`, garde les `width` composantes qui commencent à `offset`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn slice_last(a: &[f32], total: u32, offset: u32, width: u32, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let (total, offset, width) = (total as usize, offset as usize, width as usize);
            let row = flat / width;
            let column = flat % width;
            *out_elem = a[row * total + offset + column];
        }
    }

    /// La rotation de RoPE, un thread par élément. `a` est vu comme des
    /// lignes de `width` composantes, qui se suivent position après
    /// position sur `length` positions ; `cos` et `sin` sont
    /// `[length, width / 2]`. `sign` vaut 1, ou -1 pour la rotation inverse.
    ///
    /// Chaque thread lit les deux composantes de sa paire et écrit la
    /// sienne : la première reçoit `x₀·cos - x₁·sin`, la seconde
    /// `x₀·sin + x₁·cos`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn rotate_pairs(
        a: &[f32],
        cos: &[f32],
        sin: &[f32],
        width: u32,
        length: u32,
        sign: f32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let (width, length) = (width as usize, length as usize);
            let component = flat % width;
            let position = (flat / width) % length;
            let angle = position * (width / 2) + component / 2;
            let (c, s) = (cos[angle], sin[angle] * sign);
            // Le premier élément de la paire, que le thread soit le premier
            // ou le second.
            let first = flat - component % 2;
            let (x0, x1) = (a[first], a[first + 1]);
            *out_elem = if component % 2 == 0 {
                x0 * c - x1 * s
            } else {
                x0 * s + x1 * c
            };
        }
    }

    /// Recherche d'embedding : la ligne `r` de la sortie est la ligne
    /// `indices[r]` de `table`. Un thread par élément de sortie.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn gather_rows(table: &[f32], indices: &[u32], row_len: u32, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let flat = idx.get() as u32;
        if let Some(out_elem) = out.get_mut(idx) {
            let row = indices[(flat / row_len) as usize];
            *out_elem = table[(row * row_len + flat % row_len) as usize];
        }
    }

    /// Écrit la ligne `r` de `rows` à la ligne `indices[r]` de `table`, sur
    /// place. Les indices sont des entiers rangés en `f32`. Un thread par
    /// élément écrit.
    ///
    /// Un indice hors de la table, `table_rows` lignes, n'écrit rien : rien
    /// ne sort du tampon, quels que soient les indices. Deux indices égaux
    /// visent la même ligne, et la carte garde alors l'une des deux.
    ///
    /// # Safety
    ///
    /// `table` doit pointer sur un tampon GPU de `table_rows × row_len`
    /// éléments, que rien d'autre n'utilise pendant le kernel.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub unsafe fn write_rows(
        table: *mut f32,
        table_rows: u32,
        indices: &[f32],
        rows: &[f32],
        row_len: u32,
    ) {
        let flat = thread::index_1d().get();
        if flat < rows.len() {
            let row_len = row_len as usize;
            // Un indice négatif ou NaN devient 0, un indice trop grand
            // dépasse `table_rows` : la conversion ne sort jamais du test.
            let index = indices[flat / row_len] as usize;
            if index < table_rows as usize {
                // SAFETY: la ligne `index` est dans la table, d'après le
                // contrat de `table` et le test ci-dessus.
                unsafe {
                    *table.add(index * row_len + flat % row_len) = rows[flat];
                }
            }
        }
    }

    /// Inverse de `gather_rows` : la ligne `r` de `src` est ajoutée à la
    /// ligne `indices[r]` de `out`, qui doit être nul au départ. Un thread
    /// par élément de `src`.
    ///
    /// C'est le seul kernel où deux threads peuvent viser la même case :
    /// deux lignes de `src` qui portent le même indice s'additionnent au
    /// même endroit. Une addition ordinaire (lire, ajouter, écrire) perdrait
    /// des contributions quand deux threads lisent la même ancienne valeur.
    /// L'addition atomique fait les trois étapes d'un bloc.
    ///
    /// `out` est un pointeur brut et non un `DisjointSlice`, qui garantit
    /// justement qu'aucune case n'est partagée.
    ///
    /// # Safety
    ///
    /// `out` doit pointer sur un tampon GPU assez grand pour toutes les
    /// lignes désignées par `indices`, et rien d'autre ne doit y accéder
    /// pendant le lancement.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub unsafe fn scatter_add_rows(src: &[f32], indices: &[u32], row_len: u32, out: *mut f32) {
        let flat = thread::index_1d().get();
        if flat < src.len() {
            let row_len = row_len as usize;
            let row = indices[flat / row_len] as usize;
            let target = row * row_len + flat % row_len;
            // SAFETY: l'hôte a vérifié que chaque indice désigne une ligne
            // de `out`, et `out` n'est lu ou écrit que par des opérations
            // atomiques pendant ce lancement.
            let cell = unsafe { DeviceAtomicF32::from_ptr(out.add(target)) };
            // `const { … }` : le compilateur de kernels exige que l'ordre
            // soit une constante au point d'appel, ce qu'une simple valeur
            // n'est pas en compilation non optimisée.
            cell.fetch_add(src[flat], const { AtomicOrdering::Relaxed });
        }
    }

    /// Descente de gradient sur place : `param[i] -= learning_rate × grad[i]`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn sgd_step(mut param: DisjointSlice<f32>, grad: &[f32], learning_rate: f32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(p) = param.get_mut(idx) {
            *p -= learning_rate * grad[i];
        }
    }

    /// Un pas d'AdamW sur place : chaque thread met à jour un élément du
    /// paramètre et les deux moments qui lui correspondent, par la formule
    /// que partagent tous les backends (`smallem::backend::float`). Tout
    /// l'optimiseur tient dans ce seul lancement par paramètre.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn adamw_step(
        mut param: DisjointSlice<f32>,
        grad: &[f32],
        mut first_moment: DisjointSlice<f32>,
        mut second_moment: DisjointSlice<f32>,
        step: float::AdamWScalars<f32>,
    ) {
        let i = thread::index_1d().get();
        if let Some(p) = param.get_mut(thread::index_1d())
            && let Some(m) = first_moment.get_mut(thread::index_1d())
            && let Some(v) = second_moment.get_mut(thread::index_1d())
        {
            (*p, *m, *v) = float::adamw_update(*p, grad[i], *m, *v, &step);
        }
    }
}
