//! Les produits matriciels : en tuiles, sur les Tensor Cores, et pour les
//! matrices étroites de la génération, avec la somme de leurs tranches.

use super::*;

#[cuda_module]
pub mod device {
    use super::*;

    /// Produit matriciel par lots, sur les deux dernières dimensions.
    ///
    /// Les `Layout` décrivent les dimensions de lot, et leurs pas comptent
    /// en matrices entières : un pas nul fait relire la même matrice à tous
    /// les lots, comme les poids d'une couche linéaire.
    ///
    /// Un bloc de `16 × 16` threads calcule une tuile `64 × 64` de la
    /// sortie, et ses threads coopèrent. Si chacun calculait seul un élément,
    /// il relirait en mémoire globale toute une ligne de `a` et toute une
    /// colonne de `b`, que ses voisins relisent aussi. Ici le bloc avance le
    /// long de `inner` par tranches de 16 : à chaque tranche, il charge
    /// ensemble une tuile `64 × 16` de `a` et une tuile `16 × 64` de `b`
    /// dans sa mémoire partagée, bien plus rapide. Chaque valeur n'est lue
    /// qu'une fois en mémoire globale.
    ///
    /// Chaque thread calcule un pavé `4 × 4` de la sortie, et non un seul
    /// élément. Pour une tranche, il lit 4 valeurs de `a` et 4 de `b` par
    /// indice `k`, et en tire 16 produits : une lecture pour deux produits,
    /// là où un thread par élément en ferait deux par produit. Les 16
    /// accumulateurs restent dans des registres.
    ///
    /// La grille est à deux dimensions : `x` parcourt les colonnes, `y` les
    /// lignes de toutes les matrices du lot, mises bout à bout. Quand il y a
    /// trop peu de tuiles pour occuper la carte, la dimension commune est en
    /// plus découpée en tranches, que `x` enchaîne aussi : chacune écrit sa
    /// somme partielle, et `sum_splits` les additionne.
    ///
    /// # Safety
    ///
    /// Le lancement doit couvrir exactement `tranches × cols` en `x` et
    /// `lots × rows` en `y`, par blocs de `16 × 16` threads pour `64 × 64`
    /// éléments, où `tranches` vaut `inner.div_ceil(depth)` et `depth` est un
    /// multiple de 16. `out` doit pointer sur un tampon GPU de
    /// `tranches × lots × rows × cols` éléments : chaque tranche de la
    /// dimension commune écrit sa somme partielle dans sa propre copie de
    /// la sortie. `out` est un pointeur brut parce que
    /// chaque thread écrit plusieurs éléments, ce que `DisjointSlice` ne
    /// sait pas vérifier.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (16, 16, 1))]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_tiled(
        a: &[f32],
        batch_a: Layout,
        b: &[f32],
        batch_b: Layout,
        rows: u32,
        inner: u32,
        cols: u32,
        depth: u32,
        out: *mut f32,
    ) {
        // Mémoire partagée par les threads d'un même bloc : `a` en
        // `[64, 16]`, `b` en `[16, 64]`.
        static mut TILE_A: SharedArray<f32, 1024> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 1024> = SharedArray::UNINIT;

        const TILE: usize = MATMUL_TILE;
        const DEPTH: usize = MATMUL_DEPTH;
        const PATCH: usize = MATMUL_PATCH;

        let (rows, inner, cols) = (rows as usize, inner as usize, cols as usize);
        // Position du thread dans son bloc.
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let slot = ty * 16 + tx;

        // L'axe `y` de la grille enchaîne les matrices du lot : on retrouve
        // d'abord la matrice, puis la ligne dans cette matrice.
        let tiles_per_matrix = rows.div_ceil(TILE);
        let block_y = thread::blockIdx_y() as usize;
        let batch = block_y / tiles_per_matrix;
        let tile_row = (block_y % tiles_per_matrix) * TILE;
        // L'axe `x` enchaîne les tranches de la dimension commune, chacune
        // sur toutes les tuiles de colonnes.
        let column_tiles = cols.div_ceil(TILE);
        let block_x = thread::blockIdx_x() as usize;
        let (split, tile_col) = (block_x / column_tiles, (block_x % column_tiles) * TILE);
        let depth = depth as usize;
        let begin = split * depth;
        let end = if begin + depth < inner {
            begin + depth
        } else {
            inner
        };
        let matrices = thread::gridDim_y() as usize / tiles_per_matrix;

        let start_a = batch_a.source_index(batch as u32) * rows * inner;
        let start_b = batch_b.source_index(batch as u32) * inner * cols;

        // Les 16 sommes du pavé de ce thread, `cij` pour la ligne `i` et la
        // colonne `j`. Ce sont seize variables et non un tableau : le
        // compilateur de kernels garde les variables dans des registres,
        // mais range un tableau en mémoire locale, bien plus lente.
        let (mut c00, mut c01, mut c02, mut c03) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut c10, mut c11, mut c12, mut c13) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut c20, mut c21, mut c22, mut c23) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut c30, mut c31, mut c32, mut c33) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut tile_start = begin;
        while tile_start < end {
            // 1. Chargement : chaque thread apporte 4 éléments voisins de
            //    chaque tuile. Hors des matrices, on met 0, qui ne change
            //    pas les sommes.
            let mut i = 0;
            while i < PATCH {
                let cell = slot * PATCH + i;
                let (a_row, a_col) = (tile_row + cell / DEPTH, tile_start + cell % DEPTH);
                let (b_row, b_col) = (tile_start + cell / TILE, tile_col + cell % TILE);
                // SAFETY: `cell` est propre à ce thread, donc deux threads
                // n'écrivent jamais la même case.
                unsafe {
                    TILE_A[cell] = if a_row < rows && a_col < end {
                        a[start_a + a_row * inner + a_col]
                    } else {
                        0.0
                    };
                    TILE_B[cell] = if b_row < end && b_col < cols {
                        b[start_b + b_row * cols + b_col]
                    } else {
                        0.0
                    };
                }
                i += 1;
            }
            // 2. Barrière : personne ne lit avant que tout le bloc ait écrit.
            thread::sync_threads();

            // 3. Calcul : 4 lignes de la tuile de `a` par 4 colonnes de la
            //    tuile de `b`.
            // SAFETY: après la barrière, les tuiles ne sont plus que lues.
            unsafe {
                let mut k = 0;
                while k < DEPTH {
                    let a0 = TILE_A[(ty * PATCH) * DEPTH + k];
                    let a1 = TILE_A[(ty * PATCH + 1) * DEPTH + k];
                    let a2 = TILE_A[(ty * PATCH + 2) * DEPTH + k];
                    let a3 = TILE_A[(ty * PATCH + 3) * DEPTH + k];
                    let b0 = TILE_B[k * TILE + tx * PATCH];
                    let b1 = TILE_B[k * TILE + tx * PATCH + 1];
                    let b2 = TILE_B[k * TILE + tx * PATCH + 2];
                    let b3 = TILE_B[k * TILE + tx * PATCH + 3];
                    c00 += a0 * b0;
                    c01 += a0 * b1;
                    c02 += a0 * b2;
                    c03 += a0 * b3;
                    c10 += a1 * b0;
                    c11 += a1 * b1;
                    c12 += a1 * b2;
                    c13 += a1 * b3;
                    c20 += a2 * b0;
                    c21 += a2 * b1;
                    c22 += a2 * b2;
                    c23 += a2 * b3;
                    c30 += a3 * b0;
                    c31 += a3 * b1;
                    c32 += a3 * b2;
                    c33 += a3 * b3;
                    k += 1;
                }
            }
            // 4. Barrière : personne ne recharge avant que tout le bloc ait lu.
            thread::sync_threads();

            tile_start += DEPTH;
        }

        // Les éléments du pavé qui dépassent de la matrice ne sont pas écrits.
        let (row, col) = (tile_row + ty * PATCH, tile_col + tx * PATCH);
        let values = [
            c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33,
        ];
        let mut i = 0;
        while i < PATCH * PATCH {
            let (row, col) = (row + i / PATCH, col + i % PATCH);
            if row < rows && col < cols {
                // SAFETY: `(batch, row, col)` n'appartient qu'au pavé de ce
                // thread et désigne un élément de `out`, dont l'hôte
                // garantit la taille.
                unsafe {
                    *out.add(((split * matrices + batch) * rows + row) * cols + col) = values[i];
                }
            }
            i += 1;
        }
    }

    /// Le même produit matriciel sur les Tensor Cores, en précision mixte :
    /// les opérandes sont arrondis en `bf16` (16 bits, dont 8 de mantisse),
    /// les sommes restent en `f32`.
    ///
    /// Un Tensor Core multiplie d'un coup une tuile `16 × 16` par une tuile
    /// `16 × 8` (instruction `mma.sync`). L'instruction est exécutée par un
    /// warp entier, c'est-à-dire 32 threads consécutifs : chacun fournit dans
    /// ses registres un fragment des deux tuiles et reçoit un fragment du
    /// résultat. Quel thread tient quel élément est fixé par l'instruction ;
    /// avec `groupe = thread / 4` et `t = thread % 4` dans le warp :
    ///
    /// ```text
    /// a : lignes groupe et groupe + 8, colonnes 2t, 2t+1, 2t+8, 2t+9
    /// b : colonne groupe, lignes 2t, 2t+1, 2t+8, 2t+9
    /// c : lignes groupe et groupe + 8, colonnes 2t et 2t+1
    /// ```
    ///
    /// Les `bf16` voyagent par paires dans un `u32`. Les tuiles en mémoire
    /// partagée sont rangées pour que chaque registre soit une seule
    /// lecture : `a` par paires le long de `k`, ligne par ligne ; `b` par
    /// paires le long de `k`, colonne par colonne.
    ///
    /// Un bloc de `16 × 16` threads, soit 8 warps, calcule une tuile de
    /// sortie `128 × 128`. Chaque warp en calcule une région `32 × 64`, en
    /// seize multiplications par tranche de 16 le long de `inner`.
    ///
    /// Les multiplications elles-mêmes sont si rapides que c'est le
    /// chargement des tuiles qui limite ce kernel. Deux précautions le
    /// rendent supportable : une grande tuile de sortie, qui demande moins
    /// de lectures par produit, et un chemin rapide, sans aucun test, pour
    /// les tranches entièrement à l'intérieur des matrices.
    ///
    /// Le `bf16` garde les exposants du `f32` : contrairement au `f16`, il
    /// ne déborde ni ne s'annule sur les petits gradients, et la perte n'a
    /// pas besoin d'être mise à l'échelle.
    ///
    /// # Safety
    ///
    /// Mêmes exigences que `matmul_tiled`, avec des tuiles de
    /// [`MATMUL_BF16_TILE`].
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (16, 16, 1))]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_bf16(
        a: &[f32],
        batch_a: Layout,
        b: &[f32],
        batch_b: Layout,
        rows: u32,
        inner: u32,
        cols: u32,
        depth: u32,
        out: *mut f32,
    ) {
        // `a` : 128 lignes de 8 paires. `b` : 128 colonnes de 8 paires.
        static mut TILE_A: SharedArray<u32, 1024> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<u32, 1024> = SharedArray::UNINIT;

        const TILE: u32 = MATMUL_BF16_TILE as u32;
        const DEPTH: u32 = MATMUL_DEPTH as u32;
        /// Paires de `bf16` par ligne de `a` ou colonne de `b`.
        const PAIRS: u32 = DEPTH / 2;
        /// Paires que chaque thread charge dans chaque tuile.
        const LOADS: u32 = TILE * PAIRS / 256;

        // Rang du thread dans son bloc : les warps sont formés de 32 threads
        // consécutifs dans cet ordre.
        let slot = thread::threadIdx_y() * 16 + thread::threadIdx_x();
        let warp = slot / 32;
        let group = (slot % 32) / 4;
        let t = slot % 4;
        // Coin de la région `32 × 64` de ce warp dans la tuile de sortie.
        let warp_row = (warp / 2) * 32;
        let warp_col = (warp % 2) * 64;

        let tiles_per_matrix = rows.div_ceil(TILE);
        let block_y = thread::blockIdx_y();
        let batch = block_y / tiles_per_matrix;
        let tile_row = (block_y % tiles_per_matrix) * TILE;
        // L'axe `x` enchaîne les tranches de la dimension commune, chacune
        // sur toutes les tuiles de colonnes.
        let column_tiles = cols.div_ceil(TILE);
        let (split, tile_col) = (
            thread::blockIdx_x() / column_tiles,
            (thread::blockIdx_x() % column_tiles) * TILE,
        );
        let begin = split * depth;
        let end = if begin + depth < inner {
            begin + depth
        } else {
            inner
        };
        let matrices = thread::gridDim_y() / tiles_per_matrix;

        let start_a = batch_a.source_index(batch) * (rows * inner) as usize;
        let start_b = batch_b.source_index(batch) * (inner * cols) as usize;
        // La tuile de sortie tient-elle entièrement dans la matrice ? La
        // réponse est la même pour tous les threads du bloc.
        let inside = tile_row + TILE <= rows && tile_col + TILE <= cols;

        // Un accumulateur par multiplication : seize tuiles `16 × 8`, deux
        // rangées de huit. Ce sont des variables séparées, que le
        // compilateur garde dans des registres.
        let (mut c00, mut c01, mut c02, mut c03) =
            ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut c04, mut c05, mut c06, mut c07) =
            ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut c10, mut c11, mut c12, mut c13) =
            ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut c14, mut c15, mut c16, mut c17) =
            ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);

        let mut tile_start = begin;
        while tile_start < end {
            // 1. Chargement : chaque thread convertit et range quatre paires
            //    de chaque tuile.
            if inside && tile_start + DEPTH <= end {
                // Chemin rapide : tout ce qu'on lit existe.
                let mut i = 0;
                while i < LOADS {
                    let cell = slot * LOADS + i;
                    let index_a = start_a
                        + ((tile_row + cell / PAIRS) * inner + tile_start + 2 * (cell % PAIRS))
                            as usize;
                    // Les threads voisins lisent des colonnes voisines de `b`.
                    let index_b = start_b
                        + ((tile_start + 2 * (cell / TILE)) * cols + tile_col + cell % TILE)
                            as usize;
                    // SAFETY: les indices sont dans les matrices, d'après le
                    // test ci-dessus et les tailles garanties par l'hôte ;
                    // les deux cases écrites sont propres à ce thread.
                    unsafe {
                        let pa = a.as_ptr().add(index_a);
                        let pb = b.as_ptr().add(index_b);
                        TILE_A[cell as usize] = bf16_pair(*pa, *pa.add(1));
                        TILE_B[((cell % TILE) * PAIRS + cell / TILE) as usize] =
                            bf16_pair(*pb, *pb.add(cols as usize));
                    }
                    i += 1;
                }
            } else {
                // Au bord : hors des matrices, on met 0, qui ne change pas
                // les sommes.
                let mut i = 0;
                while i < LOADS {
                    let cell = slot * LOADS + i;

                    let a_row = tile_row + cell / PAIRS;
                    let a_col = tile_start + 2 * (cell % PAIRS);
                    let (mut lo, mut hi) = (0.0, 0.0);
                    if a_row < rows {
                        let base = start_a + (a_row * inner + a_col) as usize;
                        if a_col < end {
                            lo = a[base];
                        }
                        if a_col + 1 < end {
                            hi = a[base + 1];
                        }
                    }
                    let pair_a = bf16_pair(lo, hi);

                    let b_col = tile_col + cell % TILE;
                    let b_row = tile_start + 2 * (cell / TILE);
                    let (mut lo, mut hi) = (0.0, 0.0);
                    if b_col < cols {
                        let base = start_b + (b_row * cols + b_col) as usize;
                        if b_row < end {
                            lo = b[base];
                        }
                        if b_row + 1 < end {
                            hi = b[base + cols as usize];
                        }
                    }
                    let pair_b = bf16_pair(lo, hi);

                    // SAFETY: les deux cases sont propres à ce thread.
                    unsafe {
                        TILE_A[cell as usize] = pair_a;
                        TILE_B[((cell % TILE) * PAIRS + cell / TILE) as usize] = pair_b;
                    }
                    i += 1;
                }
            }
            // 2. Barrière : personne ne lit avant que tout le bloc ait écrit.
            thread::sync_threads();

            // 3. Calcul. Tous les threads du warp exécutent les mêmes seize
            //    instructions, sans branchement.
            // SAFETY: après la barrière, les tuiles ne sont plus que lues,
            // et les fragments suivent la disposition décrite plus haut.
            unsafe {
                let col_b = ((warp_col + group) * PAIRS + t) as usize;
                // Les huit fragments de `b`, communs aux deux rangées.
                macro_rules! frag_b {
                    ($n:expr) => {
                        [
                            TILE_B[col_b + $n * 8 * PAIRS as usize],
                            TILE_B[col_b + $n * 8 * PAIRS as usize + 4],
                        ]
                    };
                }
                let (b0, b1, b2, b3) = (frag_b!(0), frag_b!(1), frag_b!(2), frag_b!(3));
                let (b4, b5, b6, b7) = (frag_b!(4), frag_b!(5), frag_b!(6), frag_b!(7));

                let row_a = ((warp_row + group) * PAIRS + t) as usize;
                macro_rules! frag_a {
                    ($down:expr) => {{
                        let start = row_a + $down * 16 * PAIRS as usize;
                        [
                            TILE_A[start],
                            TILE_A[start + 8 * PAIRS as usize],
                            TILE_A[start + 4],
                            TILE_A[start + 8 * PAIRS as usize + 4],
                        ]
                    }};
                }
                let a0 = frag_a!(0);
                c00 = wmma::mma_m16n8k16_f32_bf16(c00, a0, b0);
                c01 = wmma::mma_m16n8k16_f32_bf16(c01, a0, b1);
                c02 = wmma::mma_m16n8k16_f32_bf16(c02, a0, b2);
                c03 = wmma::mma_m16n8k16_f32_bf16(c03, a0, b3);
                c04 = wmma::mma_m16n8k16_f32_bf16(c04, a0, b4);
                c05 = wmma::mma_m16n8k16_f32_bf16(c05, a0, b5);
                c06 = wmma::mma_m16n8k16_f32_bf16(c06, a0, b6);
                c07 = wmma::mma_m16n8k16_f32_bf16(c07, a0, b7);
                let a1 = frag_a!(1);
                c10 = wmma::mma_m16n8k16_f32_bf16(c10, a1, b0);
                c11 = wmma::mma_m16n8k16_f32_bf16(c11, a1, b1);
                c12 = wmma::mma_m16n8k16_f32_bf16(c12, a1, b2);
                c13 = wmma::mma_m16n8k16_f32_bf16(c13, a1, b3);
                c14 = wmma::mma_m16n8k16_f32_bf16(c14, a1, b4);
                c15 = wmma::mma_m16n8k16_f32_bf16(c15, a1, b5);
                c16 = wmma::mma_m16n8k16_f32_bf16(c16, a1, b6);
                c17 = wmma::mma_m16n8k16_f32_bf16(c17, a1, b7);
            }
            // 4. Barrière : personne ne recharge avant que tout le bloc ait lu.
            thread::sync_threads();

            tile_start += DEPTH;
        }

        // Chaque thread tient quatre éléments de chacune des seize tuiles.
        let tiles = [
            c00, c01, c02, c03, c04, c05, c06, c07, c10, c11, c12, c13, c14, c15, c16, c17,
        ];
        let mut n = 0;
        while n < 16 {
            let mut j = 0;
            while j < 4 {
                let row = tile_row + warp_row + 16 * (n / 8) + group + 8 * (j / 2);
                let col = tile_col + warp_col + 8 * (n % 8) + 2 * t + j % 2;
                if row < rows && col < cols {
                    // SAFETY: `(batch, row, col)` n'appartient qu'à ce thread
                    // et désigne un élément de `out`.
                    unsafe {
                        *out.add(
                            (((split * matrices + batch) * rows + row) * cols + col) as usize,
                        ) = tiles[n as usize][j as usize];
                    }
                }
                j += 1;
            }
            n += 1;
        }
    }

    /// Le produit matriciel quand `a` n'a que quelques lignes : celui de la
    /// génération, où chaque séquence n'apporte qu'un vecteur par token.
    ///
    /// Les tuiles de `matmul_tiled` y gaspillent presque tout. Une tuile de
    /// 64 lignes n'en a qu'une d'utile, et la grille, qui compte un bloc par
    /// tuile, n'a que 8 blocs pour une sortie de 512 colonnes : 8 des 82
    /// multiprocesseurs de la carte travaillent. Chaque poids sert à une
    /// seule multiplication, si bien que le temps se passe à attendre la
    /// mémoire, avec trop peu de lectures en cours pour l'occuper.
    ///
    /// Ici, le travail est découpé finement. Un bloc de 8 warps couvre 32
    /// colonnes et une tranche de `depth` indices le long de `inner`. Les 32
    /// threads d'un warp lisent 32 colonnes voisines d'une même ligne de
    /// `b`, une seule transaction. Les 8 warps se partagent la tranche, un
    /// indice sur huit chacun, quatre indices à la fois pour garder
    /// plusieurs lectures en cours. Chaque poids lu sert aux lignes de `a`
    /// par paquets de 8, et les sommes des 8 warps sont réunies en mémoire
    /// partagée.
    ///
    /// Quand une tranche ne couvre pas tout `inner`, chacune écrit sa somme
    /// partielle dans sa propre copie de la sortie, et `sum_splits` les
    /// additionne ensuite. Pas d'addition atomique : le résultat ne dépend
    /// pas de l'ordre dans lequel les blocs s'exécutent.
    ///
    /// Avec `has_bias`, le biais `bias` `[cols]` est ajouté à chaque ligne :
    /// c'est la couche affine `x · W + b` en un seul passage.
    ///
    /// La grille est à deux dimensions : `x` parcourt les colonnes par 32,
    /// `y` les matrices du lot, chacune découpée en tranches.
    ///
    /// # Safety
    ///
    /// Le lancement doit couvrir `cols` en `x` par blocs de `32 × 8`
    /// threads, et `lots × tranches` en `y`, où `tranches` vaut
    /// `inner.div_ceil(depth)`. `out` doit pointer sur un tampon GPU de
    /// `lots × tranches × rows × cols` éléments, rangés dans cet ordre.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (32, 8, 1))]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_narrow(
        a: &[f32],
        batch_a: Layout,
        b: &[f32],
        batch_b: Layout,
        rows: u32,
        inner: u32,
        cols: u32,
        depth: u32,
        bias: &[f32],
        has_bias: u32,
        out: *mut f32,
    ) {
        // La somme partielle de chaque warp, pour 8 lignes et 32 colonnes.
        static mut SUMS: SharedArray<f32, 2048> = SharedArray::UNINIT;

        let lane = thread::threadIdx_x() as usize;
        let warp = thread::threadIdx_y() as usize;
        let (rows, inner, cols) = (rows as usize, inner as usize, cols as usize);
        let depth = depth as usize;
        let splits = inner.div_ceil(depth);
        let block_y = thread::blockIdx_y() as usize;
        let (batch, split) = (block_y / splits, block_y % splits);
        let col = thread::blockIdx_x() as usize * WARP + lane;
        let begin = split * depth;
        let end = if begin + depth < inner {
            begin + depth
        } else {
            inner
        };
        let start_a = batch_a.source_index(batch as u32) * rows * inner;
        let start_b = batch_b.source_index(batch as u32) * inner * cols;

        let mut first_row = 0;
        while first_row < rows {
            let mut sums = [0.0f32; 8];
            if col < cols {
                let mut k = begin + warp;
                while k < end {
                    // Les quatre poids d'abord : leurs lectures partent
                    // ensemble, au lieu d'attendre chacune la précédente.
                    let mut weights = [0.0f32; 4];
                    unroll!(i in [0, 1, 2, 3] {
                        if k + 8 * i < end {
                            // SAFETY: `k + 8i < inner` et `col < cols`.
                            weights[i] = unsafe {
                                *b.as_ptr().add(start_b + (k + 8 * i) * cols + col)
                            };
                        }
                    });
                    unroll!(r in [0, 1, 2, 3, 4, 5, 6, 7] {
                        if first_row + r < rows {
                            let row = start_a + (first_row + r) * inner + k;
                            unroll!(i in [0, 1, 2, 3] {
                                if k + 8 * i < end {
                                    // SAFETY: la ligne existe, `k + 8i < inner`.
                                    sums[r] += unsafe { *a.as_ptr().add(row + 8 * i) }
                                        * weights[i];
                                }
                            });
                        }
                    });
                    k += 32;
                }
            }
            // SAFETY: chaque thread écrit ses propres cases.
            unroll!(r in [0, 1, 2, 3, 4, 5, 6, 7] {
                unsafe { SUMS[(warp * 8 + r) * WARP + lane] = sums[r] };
            });
            thread::sync_threads();

            // Le warp `w` réunit la ligne `w` du paquet.
            let row = first_row + warp;
            if row < rows && col < cols {
                // Le biais, une seule fois : par la seule tranche quand il
                // n'y en a qu'une, sinon par la réunion des tranches.
                let mut total = if has_bias != 0 && splits == 1 {
                    bias[col]
                } else {
                    0.0
                };
                // SAFETY: après la barrière, les sommes ne sont plus que lues.
                unroll!(w in [0, 1, 2, 3, 4, 5, 6, 7] {
                    total += unsafe { SUMS[(w * 8 + warp) * WARP + lane] };
                });
                // SAFETY: `(batch, split, row, col)` n'appartient qu'à ce
                // thread et désigne un élément de `out`.
                unsafe {
                    *out.add(((batch * splits + split) * rows + row) * cols + col) = total;
                }
            }
            // Personne ne réécrit les sommes avant qu'elles aient été lues.
            thread::sync_threads();
            first_row += 8;
        }
    }

    /// `a · bᵀ` quand `a` n'a que quelques lignes : la sortie du modèle en
    /// génération, où `b` est la table d'embedding `[N, K]`.
    ///
    /// Chaque élément de sortie est le produit scalaire d'une ligne de `a`
    /// et d'une ligne de `b`. Un warp s'occupe d'une ligne de `b` : ses 32
    /// threads la lisent par morceaux voisins, quatre à la fois, puis
    /// réunissent leurs sommes partielles par échanges entre registres. La
    /// ligne de `b` sert aux lignes de `a` par paquets de 8. Un bloc de 8
    /// warps couvre 8 lignes de `b`.
    ///
    /// # Safety
    ///
    /// Le lancement doit couvrir `n` lignes de `b` en `x`, par blocs de
    /// `32 × 8` threads, et `out` doit pointer sur un tampon GPU de
    /// `rows × n` éléments.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (32, 8, 1))]
    pub unsafe fn transposed_matmul_narrow(
        a: &[f32],
        b: &[f32],
        rows: u32,
        inner: u32,
        n: u32,
        out: *mut f32,
    ) {
        let lane = thread::threadIdx_x() as usize;
        let (rows, inner, n) = (rows as usize, inner as usize, n as usize);
        let line = thread::blockIdx_x() as usize * 8 + thread::threadIdx_y() as usize;
        if line < n {
            let start_b = line * inner;
            let mut first_row = 0;
            while first_row < rows {
                let mut sums = [0.0f32; 8];
                let mut k = lane;
                while k < inner {
                    let mut weights = [0.0f32; 4];
                    unroll!(i in [0, 1, 2, 3] {
                        if k + 32 * i < inner {
                            // SAFETY: `line < n` et `k + 32i < inner`.
                            weights[i] = unsafe { *b.as_ptr().add(start_b + k + 32 * i) };
                        }
                    });
                    unroll!(r in [0, 1, 2, 3, 4, 5, 6, 7] {
                        if first_row + r < rows {
                            let row = (first_row + r) * inner + k;
                            unroll!(i in [0, 1, 2, 3] {
                                if k + 32 * i < inner {
                                    // SAFETY: la ligne existe, `k + 32i < inner`.
                                    sums[r] += unsafe { *a.as_ptr().add(row + 32 * i) }
                                        * weights[i];
                                }
                            });
                        }
                    });
                    k += 128;
                }
                unroll!(r in [0, 1, 2, 3, 4, 5, 6, 7] {
                    let total = warp_sum(sums[r]);
                    if lane == 0 && first_row + r < rows {
                        // SAFETY: `(row, line)` n'appartient qu'à ce warp.
                        unsafe { *out.add((first_row + r) * n + line) = total };
                    }
                });
                first_row += 8;
            }
        }
    }

    /// Réunit les tranches de `matmul_narrow` : `parts` est
    /// `[lots, tranches, inner]`, la sortie `[lots, inner]`, et chaque
    /// élément est la somme de ses tranches, plus le biais de sa colonne
    /// avec `has_bias`. Un thread par élément de sortie.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn sum_splits(
        parts: &[f32],
        splits: u32,
        inner: u32,
        bias: &[f32],
        has_bias: u32,
        cols: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let (splits, inner) = (splits as usize, inner as usize);
            let (batch, rest) = (flat / inner, flat % inner);
            let mut total = if has_bias != 0 {
                bias[rest % cols as usize]
            } else {
                0.0
            };
            for split in 0..splits {
                total += parts[(batch * splits + split) * inner + rest];
            }
            *out_elem = total;
        }
    }
}
