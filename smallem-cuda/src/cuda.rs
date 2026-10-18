//! Backend GPU : les mêmes opérations que le CPU, exécutées par les kernels
//! de [`crate::kernels`].
//!
//! Les données d'un tenseur restent sur la carte d'une opération à l'autre,
//! et l'hôte ne garde que les formes. Les lancements sont asynchrones : une
//! opération rend la main dès que son kernel est mis en file, pas quand il a
//! fini.
//!
//! Trois sortes d'appels font traverser des données, et attendent pour cela
//! que la carte ait vidé sa file : `to_vec` (un résultat vers l'hôte),
//! `from_vec` (des valeurs vers la carte, donc aussi chaque constante créée
//! par `scalar` ou `ones`) et l'envoi des indices de `gather_rows`,
//! `scatter_add_rows` et `cross_entropy`. Un pas d'entraînement en fait une
//! dizaine. Les envois pourraient être mis en file sans attendre ; mesuré,
//! cela ne change pas le débit, parce que c'est la carte et non l'hôte qui
//! limite : l'hôte attendrait de toute façon un peu plus loin.
//!
//! Tout est en `f32`, et limité à [`MAX_RANK`] dimensions.
//!
//! La mémoire GPU n'est jamais rendue au pilote pendant le calcul : un
//! tampon devenu inutile retourne dans une réserve (`Pool`) et sert au
//! prochain tenseur de même taille.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::sync::Arc;

use cuda_core::{
    CudaContext, CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig1D, LaunchConfig2D,
};

use smallem::backend::ops::{CachedAttentionDims, concat_widths};
use smallem::backend::shape::{broadcast_shapes, broadcast_strides, strides};
use smallem::backend::{AdamWStep, Primitives};
use smallem::quantize::{Format, Quantized};

use crate::kernels::{
    ATTENTION_MAX_DIM, ATTENTION_TILE, AdamWParams, BLOCK_SIZE, BinaryOp, CachedDims, Layout,
    MATMUL_BF16_TILE, MATMUL_DEPTH, MATMUL_PATCH, MATMUL_TILE, MAX_RANK, QuantizedDims,
    SoftmaxMode, UnaryOp, WARP, WIDE_TILE, device,
};

/// Une erreur du pilote CUDA n'est pas rattrapable ici : on panique, comme
/// pour une erreur de forme.
const CUDA_ERROR: &str = "erreur CUDA";

/// Taille des paquets de la somme globale.
const SUM_CHUNK: usize = 256;

/// Nombre maximal de blocs d'une grille de lancement le long d'une
/// dimension : une limite de CUDA.
const MAX_GRID_BLOCKS: u32 = 65_535;

pub struct Cuda {
    stream: Arc<CudaStream>,
    module: device::LoadedModule,
    /// Tampons des tenseurs.
    pool: Rc<Pool<f32>>,
    /// Tampons d'indices (tokens, cibles), envoyés par l'hôte à chaque pas.
    index_pool: Rc<Pool<u32>>,
    /// Vrai si les produits matriciels passent par les Tensor Cores.
    mixed_precision: bool,
    /// Faux pour calculer l'attention avec les primitives, comme avant
    /// l'attention fusionnée : sert aux comparaisons.
    fused_attention: bool,
    /// Nombre maximal de lignes des produits confiés au kernel des
    /// matrices étroites : [`NARROW_ROWS`], sauf pour les comparaisons.
    narrow_rows: usize,
}

impl Drop for Cuda {
    fn drop(&mut self) {
        // Des kernels et des envois peuvent encore être en file. Il faut
        // les laisser finir avant que le module de kernels soit déchargé
        // et que les données de l'hôte en cours d'envoi soient libérées.
        let _ = self.stream.synchronize();
    }
}

/// Réserve de tampons GPU inutilisés, rangés par taille.
///
/// Allouer et libérer de la mémoire GPU coûte cher, et libérer oblige en
/// plus à attendre la fin des kernels en file, puisque l'un d'eux pourrait
/// encore utiliser cette mémoire. Un entraînement redemande les mêmes
/// tailles à chaque pas : on garde donc les tampons au lieu de les rendre.
///
/// Réutiliser un tampon qu'un kernel en file lit encore est sans danger,
/// parce que tout passe par le même stream : le kernel qui écrira dans le
/// tampon réutilisé est mis en file après celui qui le lit, donc exécuté
/// après lui.
///
/// La réserve garde un tampon par tenseur vivant au moment où la mémoire
/// était la plus sollicitée. Elle ne se vide que si la mémoire vient à
/// manquer : cela arrive quand les tailles demandées changent sans cesse,
/// comme en génération, où la séquence s'allonge d'un token à chaque tour.
struct Pool<T> {
    stream: Arc<CudaStream>,
    free: RefCell<HashMap<usize, Vec<DeviceBuffer<T>>>>,
}

impl<T: DeviceCopy> Pool<T> {
    fn new(stream: Arc<CudaStream>) -> Rc<Self> {
        Rc::new(Self {
            stream,
            free: RefCell::new(HashMap::new()),
        })
    }

    /// Fournit un tampon de `len` éléments. Son contenu est quelconque :
    /// l'appelant doit écrire chaque élément avant de le lire.
    fn take(self: &Rc<Self>, len: usize) -> Pooled<T> {
        let reused = self.free.borrow_mut().get_mut(&len).and_then(Vec::pop);
        let buffer = reused.unwrap_or_else(|| {
            DeviceBuffer::zeroed(&self.stream, len).unwrap_or_else(|_| {
                // Plus de mémoire : on rend au pilote tout ce qui dort dans
                // la réserve, puis on réessaie une fois.
                self.release();
                DeviceBuffer::zeroed(&self.stream, len).expect(CUDA_ERROR)
            })
        });
        Pooled {
            buffer: Some(buffer),
            pool: self.clone(),
        }
    }
}

impl<T> Pool<T> {
    /// Libère tous les tampons inutilisés.
    fn release(&self) {
        // Un kernel en file peut encore lire un tampon rendu à la réserve :
        // il faut attendre qu'ils aient tous fini avant de libérer.
        self.stream.synchronize().expect(CUDA_ERROR);
        self.free.borrow_mut().clear();
    }
}

impl<T> Drop for Pool<T> {
    fn drop(&mut self) {
        // Les tampons vont réellement être libérés : cette fois il faut
        // attendre que plus aucun kernel ne les utilise.
        let _ = self.stream.synchronize();
    }
}

/// Un tampon emprunté à une réserve, qui y retourne quand il est abandonné.
struct Pooled<T> {
    /// Toujours `Some`, sauf pendant `drop`.
    buffer: Option<DeviceBuffer<T>>,
    pool: Rc<Pool<T>>,
}

impl<T> Deref for Pooled<T> {
    type Target = DeviceBuffer<T>;

    fn deref(&self) -> &DeviceBuffer<T> {
        self.buffer.as_ref().expect("tampon déjà rendu")
    }
}

impl<T> DerefMut for Pooled<T> {
    fn deref_mut(&mut self) -> &mut DeviceBuffer<T> {
        self.buffer.as_mut().expect("tampon déjà rendu")
    }
}

impl<T> Drop for Pooled<T> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool
                .free
                .borrow_mut()
                .entry(buffer.len())
                .or_default()
                .push(buffer);
        }
    }
}

/// Tenseur dont les données sont en mémoire GPU.
///
/// Le stockage est partagé par comptage de références : `reshape` renvoie
/// un tenseur qui pointe sur les mêmes données, sans copie. C'est sans
/// danger parce qu'aucune opération ne modifie un tenseur existant.
#[derive(Clone)]
pub struct CudaTensor {
    storage: Rc<Storage>,
    shape: Vec<usize>,
}

struct Storage {
    buffer: Pooled<f32>,
}

/// Lance un kernel sur le stream du backend : prépare la configuration de
/// lancement, que le contrat du kernel vérifie, puis met le kernel en file.
///
/// Le nom du kernel est donné deux fois, nu puis précédé de `prepare_` :
/// une macro ne sait pas fabriquer le second à partir du premier.
macro_rules! launch {
    ($backend:ident, $kernel:ident, $prepare:ident, $config:expr, $($argument:expr),+ $(,)?) => {{
        let prepared = $backend.module.$prepare($config).expect(CUDA_ERROR);
        $backend
            .module
            .$kernel(&$backend.stream, &prepared, $($argument),+)
            .expect(CUDA_ERROR)
    }};
}

/// Nombre maximal de lignes de `a` pour lequel le produit matriciel passe
/// par le kernel des matrices étroites, `matmul_narrow`.
pub const NARROW_ROWS: usize = 16;

/// Nombre de blocs que vise la grille de `matmul_narrow` : de quoi occuper
/// les 82 multiprocesseurs de la carte plusieurs fois.
const NARROW_BLOCKS: usize = 256;

/// Longueur minimale d'une tranche de `matmul_narrow` le long de `inner`.
const NARROW_MIN_DEPTH: usize = 64;

/// Nombre maximal de lignes de `a` pour lequel les kernels en tuiles
/// découpent la dimension commune en tranches : les produits de la
/// génération par lots. Ceux de l'entraînement, bien plus grands, gardent
/// exactement leurs kernels et leurs nombres.
pub const SPLIT_ROWS: usize = 256;

/// Longueur minimale d'une tranche des kernels en tuiles.
const SPLIT_MIN_DEPTH: usize = 128;

/// Taille maximale des têtes de l'attention sur le cache : au-delà de
/// [`ATTENTION_MAX_DIM`], un warp entier lit chaque clé.
pub const CACHED_WIDE_DIM: usize = 512;

/// Nombre minimal de mots d'une tranche de `matmul_narrow_quantized`.
const NARROW_MIN_WORDS: usize = 16;

/// Longueur à partir de laquelle la somme le long de l'avant-dernier axe
/// passe par une transposition (voir `sum`).
const LONG_AXIS: usize = 128;

/// Configuration de lancement pour `len` éléments de sortie : assez de blocs
/// de [`BLOCK_SIZE`] threads pour qu'il y ait un thread par élément.
fn launch_config(len: usize) -> LaunchConfig1D {
    // Les threads sont numérotés sur 32 bits. Au-delà, le nombre de blocs
    // serait tronqué sans erreur et une partie de la sortie ne serait
    // jamais calculée.
    let len = u32::try_from(len)
        .unwrap_or_else(|_| panic!("lancement trop grand pour des indices u32 : {len} threads"));
    LaunchConfig1D::new(len.div_ceil(BLOCK_SIZE).max(1), BLOCK_SIZE, 0)
}

/// Configuration de lancement des réductions par ligne : un warp, soit 32
/// threads, par ligne.
fn row_launch_config(rows: usize) -> LaunchConfig1D {
    launch_config(rows * WARP)
}

/// Décrit la lecture d'un tenseur dont les pas sont `source_strides` à
/// travers la forme `target`, complétée à gauche jusqu'à [`MAX_RANK`]
/// dimensions.
fn layout(target: &[usize], source_strides: &[usize]) -> Layout {
    assert!(
        target.len() <= MAX_RANK,
        "le backend CUDA gère au plus {MAX_RANK} dimensions, reçu {target:?}",
    );
    let pad = MAX_RANK - target.len();
    let mut layout = Layout {
        shape: [1; MAX_RANK],
        strides: [0; MAX_RANK],
    };
    for dim in 0..target.len() {
        layout.shape[pad + dim] = target[dim] as u32;
        layout.strides[pad + dim] = source_strides[dim] as u32;
    }
    layout
}

impl Cuda {
    /// Ouvre le GPU numéro `ordinal` et y charge les kernels.
    pub fn new(ordinal: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let context = CudaContext::new(ordinal)?;
        let stream = context.default_stream();
        // SAFETY: le module chargé est celui que ce paquet a compilé à
        // partir de `kernels::device` et embarqué dans le binaire.
        let module = unsafe { device::load(&context)? };
        Ok(Self {
            pool: Pool::new(stream.clone()),
            index_pool: Pool::new(stream.clone()),
            stream,
            module,
            mixed_precision: false,
            fused_attention: true,
            narrow_rows: NARROW_ROWS,
        })
    }

    /// Fait passer les produits matriciels par les Tensor Cores : leurs
    /// opérandes sont arrondis en `bf16`, soit environ trois chiffres
    /// significatifs, et les sommes restent en `f32`. Toutes les autres
    /// opérations, et les tenseurs eux-mêmes, restent en `f32`.
    pub fn with_mixed_precision(mut self) -> Self {
        self.mixed_precision = true;
        self
    }

    /// Calcule l'attention avec les primitives, en rangeant scores et
    /// poids, au lieu des kernels d'attention fusionnée.
    pub fn without_fused_attention(mut self) -> Self {
        self.fused_attention = false;
        self
    }

    /// Confie au kernel des matrices étroites les produits d'au plus `rows`
    /// lignes, au lieu de [`NARROW_ROWS`] : pour mesurer où passe la
    /// frontière entre les deux kernels.
    pub fn with_narrow_rows(mut self, rows: usize) -> Self {
        self.narrow_rows = rows;
        self
    }

    /// Nombre maximal de lignes des produits du kernel des matrices
    /// étroites.
    pub fn narrow_rows(&self) -> usize {
        self.narrow_rows
    }

    /// Rend au pilote tous les tampons qui dorment dans la réserve. Après
    /// une génération par lots, dont les tailles changent à chaque pas, la
    /// réserve peut garder des gigaoctets dont plus rien ne se servira ; un
    /// autre programme sur la même carte pourrait en manquer.
    pub fn release_memory(&self) {
        self.pool.release();
        self.index_pool.release();
    }

    /// Attend la fin de tous les kernels en file. À appeler avant de lire
    /// une horloge : sans cela, on mesure la mise en file, pas le calcul.
    pub fn synchronize(&self) {
        self.stream.synchronize().expect(CUDA_ERROR);
    }

    /// Tampon pour le résultat d'un kernel. Son contenu est quelconque :
    /// tous les kernels écrivent chaque élément de leur sortie.
    fn alloc(&self, len: usize) -> Pooled<f32> {
        assert!(
            len <= u32::MAX as usize,
            "tenseur trop grand pour des indices u32 : {len} éléments",
        );
        // Le backend CPU sait rendre un tenseur vide ; ici, rien ne
        // dimensionne correctement un lancement sans élément. Mieux vaut un
        // refus clair qu'une erreur obscure du pilote.
        assert!(len > 0, "le backend CUDA ne gère pas les tenseurs vides");
        self.pool.take(len)
    }

    /// Envoie des indices sur le GPU.
    fn upload_indices(&self, indices: &[u32]) -> Pooled<u32> {
        assert!(
            !indices.is_empty(),
            "le backend CUDA ne gère pas les tenseurs vides"
        );
        let mut buffer = self.index_pool.take(indices.len());
        buffer
            .copy_from_host(&self.stream, indices)
            .expect(CUDA_ERROR);
        buffer
    }

    /// Le produit matriciel `a · b`, plus `bias` `[cols]` sur chaque ligne
    /// s'il est donné : la primitive `matmul`, et la couche affine.
    ///
    /// Trois chemins, selon le nombre de lignes de `a` :
    ///
    /// - au plus `narrow_rows` : le kernel des matrices étroites, celui de
    ///   la génération token par token ;
    /// - au plus [`SPLIT_ROWS`] : les kernels en tuiles, avec la dimension
    ///   commune découpée en tranches si la grille est trop petite pour
    ///   occuper la carte, comme en génération par lots ;
    /// - au-delà, les kernels en tuiles tels quels, puis l'addition du
    ///   biais : les produits de l'entraînement, qui gardent ainsi
    ///   exactement leurs nombres.
    pub fn product(&self, a: &CudaTensor, b: &CudaTensor, bias: Option<&CudaTensor>) -> CudaTensor {
        assert!(
            a.shape.len() >= 2 && b.shape.len() >= 2,
            "matmul demande au moins 2 dimensions : {:?} et {:?}",
            a.shape,
            b.shape,
        );
        let (batch_a, matrix_a) = a.shape.split_at(a.shape.len() - 2);
        let (batch_b, matrix_b) = b.shape.split_at(b.shape.len() - 2);
        let (rows, inner) = (matrix_a[0], matrix_a[1]);
        let cols = matrix_b[1];
        assert_eq!(
            inner, matrix_b[0],
            "matmul : formes incompatibles {:?} et {:?}",
            a.shape, b.shape,
        );
        if let Some(bias) = bias {
            assert_eq!(bias.shape, [cols], "biais de forme inattendue");
        }

        let batch = broadcast_shapes(batch_a, batch_b);
        let layout_a = layout(&batch, &broadcast_strides(batch_a, &batch));
        let layout_b = layout(&batch, &broadcast_strides(batch_b, &batch));

        let mut shape = batch;
        shape.extend([rows, cols]);
        assert!(
            shape.len() <= MAX_RANK,
            "le backend CUDA gère au plus {MAX_RANK} dimensions, reçu {shape:?}",
        );
        let batch_len: usize = shape[..shape.len() - 2].iter().product();
        if rows <= self.narrow_rows {
            return self.matmul_narrow(a, layout_a, b, layout_b, batch_len, shape, inner, bias);
        }

        // Une grille à deux dimensions de blocs `16 × 16` threads, chacun
        // pour une tuile carrée de sortie : les colonnes en `x`, les lignes de
        // toutes les matrices du lot en `y`. Les tranches de la dimension
        // commune, s'il y en a, s'enchaînent aussi en `x`.
        let tile = if self.mixed_precision {
            MATMUL_BF16_TILE
        } else {
            MATMUL_TILE
        };
        let threads = (MATMUL_TILE / MATMUL_PATCH) as u32;
        let tiles = cols.div_ceil(tile) * batch_len * rows.div_ceil(tile);
        let splits = if rows <= SPLIT_ROWS {
            NARROW_BLOCKS
                .div_ceil(tiles)
                .clamp(1, inner.div_ceil(SPLIT_MIN_DEPTH).max(1))
        } else {
            1
        };
        let depth = inner.div_ceil(splits).next_multiple_of(MATMUL_DEPTH);
        let splits = inner.div_ceil(depth).max(1);
        let grid = (
            (splits * cols.div_ceil(tile)) as u32,
            (batch_len * rows.div_ceil(tile)) as u32,
        );
        assert!(
            grid.0 <= MAX_GRID_BLOCKS && grid.1 <= MAX_GRID_BLOCKS,
            "matmul trop grande pour une seule grille : {} × {} blocs, {MAX_GRID_BLOCKS} au plus \
             par dimension",
            grid.0,
            grid.1,
        );
        let len = batch_len * rows * cols;
        let out = self.alloc(splits * len);
        let config = LaunchConfig2D::new(grid, (threads, threads), 0);
        // Adresse en mémoire GPU : à ne jamais déréférencer sur l'hôte.
        let out_ptr = out.cu_deviceptr() as *mut f32;
        let (rows_u, inner_u, cols_u, depth_u) =
            (rows as u32, inner as u32, cols as u32, depth as u32);
        let (buffer_a, buffer_b) = (&a.storage.buffer, &b.storage.buffer);
        // SAFETY: dans les deux cas, la grille couvre exactement
        // `tranches × cols` en x et `batch_len × rows` en y, par tuiles de la
        // taille que le kernel attend, et `out` a `tranches × batch_len ×
        // rows × cols` éléments.
        unsafe {
            if self.mixed_precision {
                launch!(
                    self,
                    matmul_bf16,
                    prepare_matmul_bf16,
                    config,
                    buffer_a,
                    layout_a,
                    buffer_b,
                    layout_b,
                    rows_u,
                    inner_u,
                    cols_u,
                    depth_u,
                    out_ptr
                );
            } else {
                launch!(
                    self,
                    matmul_tiled,
                    prepare_matmul_tiled,
                    config,
                    buffer_a,
                    layout_a,
                    buffer_b,
                    layout_b,
                    rows_u,
                    inner_u,
                    cols_u,
                    depth_u,
                    out_ptr
                );
            }
        }
        if splits > 1 {
            // Les tranches sont rangées `[tranches, toute la sortie]`.
            let total = self.sum_splits(&out, 1, splits, len, bias, cols);
            return self.wrap(total, shape);
        }
        let product = self.wrap(out, shape);
        match bias {
            Some(bias) => self.add(&product, bias),
            None => product,
        }
    }

    /// `x · W (+ b)` pour des poids quantifiés au format `format` : `words`
    /// et `scales` leur forme compacte, `dequantized` `[K, N]` les mêmes
    /// poids en `f32` (voir `smallem::backend::ops::QuantizedMatmul`).
    ///
    /// Les produits de la génération lisent la forme compacte : le kernel
    /// des matrices étroites jusqu'à `narrow_rows` lignes, les tuiles
    /// découpées en tranches jusqu'à [`SPLIT_ROWS`]. Au-delà, les produits
    /// d'une évaluation lisent la forme déquantifiée, par [`Cuda::product`].
    #[allow(clippy::too_many_arguments)]
    pub fn quantized_matmul(
        &self,
        x: &CudaTensor,
        words: &CudaTensor,
        scales: &CudaTensor,
        dequantized: &CudaTensor,
        bias: Option<&CudaTensor>,
        format: Format,
    ) -> CudaTensor {
        let (inner, cols) = (dequantized.shape[0], dequantized.shape[1]);
        assert!(
            dequantized.shape.len() == 2 && x.shape.last() == Some(&inner),
            "quantized_matmul : {:?} · {:?}",
            x.shape,
            dequantized.shape,
        );
        let rows = x.storage.buffer.len() / inner;
        let mut shape = x.shape.clone();
        *shape.last_mut().expect("au moins une dimension") = cols;
        let flat = self.reshape(x, &[rows, inner]);
        if rows > SPLIT_ROWS {
            let out = self.product(&flat, dequantized, bias);
            return self.reshape(&out, &shape);
        }
        let (group, groups, per_group) = Quantized::layout(format, inner);
        assert_eq!(
            words.shape,
            [groups * per_group, cols],
            "mots de forme inattendue"
        );
        let dims = QuantizedDims {
            cols: cols as u32,
            bits: format.bits(),
            group: group as u32,
            codes_per_word: format.codes_per_word() as u32,
            words_per_group: per_group as u32,
        };
        let (buffer_x, buffer_w, buffer_s) = (
            &flat.storage.buffer,
            &words.storage.buffer,
            &scales.storage.buffer,
        );
        let (out, splits) = if rows <= self.narrow_rows {
            // Les tranches comptent des mots.
            let total_words = groups * per_group;
            let column_blocks = cols.div_ceil(WARP);
            let splits = NARROW_BLOCKS
                .div_ceil(column_blocks)
                .clamp(1, total_words.div_ceil(NARROW_MIN_WORDS).max(1));
            let depth = total_words.div_ceil(splits);
            let splits = total_words.div_ceil(depth);
            let out = self.alloc(splits * rows * cols);
            let (bias_buffer, has_bias) = match bias {
                Some(bias) if splits == 1 => (&*bias.storage.buffer, 1u32),
                _ => (&**buffer_x, 0u32),
            };
            let config =
                LaunchConfig2D::new((column_blocks as u32, splits as u32), (WARP as u32, 8), 0);
            // SAFETY: la grille couvre `cols` par 32 en x et les tranches en
            // y, et `out` a `tranches × rows × cols` éléments.
            unsafe {
                launch!(
                    self,
                    matmul_narrow_quantized,
                    prepare_matmul_narrow_quantized,
                    config,
                    buffer_x,
                    buffer_w,
                    buffer_s,
                    dims,
                    rows as u32,
                    inner as u32,
                    depth as u32,
                    bias_buffer,
                    has_bias,
                    out.cu_deviceptr() as *mut f32
                );
            }
            if splits == 1 {
                return self.wrap(out, shape);
            }
            (out, splits)
        } else {
            let column_tiles = cols.div_ceil(MATMUL_TILE);
            let tiles = column_tiles * rows.div_ceil(MATMUL_TILE);
            let splits = NARROW_BLOCKS
                .div_ceil(tiles)
                .clamp(1, inner.div_ceil(SPLIT_MIN_DEPTH).max(1));
            let depth = inner.div_ceil(splits).next_multiple_of(MATMUL_DEPTH);
            let splits = inner.div_ceil(depth);
            let out = self.alloc(splits * rows * cols);
            let threads = (MATMUL_TILE / MATMUL_PATCH) as u32;
            let grid = (
                (splits * column_tiles) as u32,
                rows.div_ceil(MATMUL_TILE) as u32,
            );
            let config = LaunchConfig2D::new(grid, (threads, threads), 0);
            // SAFETY: la grille couvre `tranches × cols` en x et `rows` en y,
            // par tuiles de 64, et `out` a `tranches × rows × cols` éléments.
            unsafe {
                launch!(
                    self,
                    matmul_tiled_quantized,
                    prepare_matmul_tiled_quantized,
                    config,
                    buffer_x,
                    buffer_w,
                    buffer_s,
                    dims,
                    rows as u32,
                    inner as u32,
                    depth as u32,
                    out.cu_deviceptr() as *mut f32
                );
            }
            if splits == 1 {
                let product = self.wrap(out, shape);
                return match bias {
                    Some(bias) => self.add(&product, bias),
                    None => product,
                };
            }
            (out, splits)
        };
        let total = self.sum_splits(&out, 1, splits, rows * cols, bias, cols);
        self.wrap(total, shape)
    }

    /// Additionne les tranches d'un produit, `[lots, tranches, inner]`, en
    /// `[lots, inner]`, plus `bias` `[cols]` s'il est donné.
    fn sum_splits(
        &self,
        parts: &Pooled<f32>,
        batches: usize,
        splits: usize,
        inner: usize,
        bias: Option<&CudaTensor>,
        cols: usize,
    ) -> Pooled<f32> {
        let len = batches * inner;
        let mut total = self.alloc(len);
        // Sans biais, le kernel reçoit un tableau qu'il ne lira pas.
        let (bias_buffer, has_bias) = match bias {
            Some(bias) => (&*bias.storage.buffer, 1u32),
            None => (&**parts, 0u32),
        };
        launch!(
            self,
            sum_splits,
            prepare_sum_splits,
            launch_config(len),
            parts,
            splits as u32,
            inner as u32,
            bias_buffer,
            has_bias,
            cols as u32,
            &mut total
        );
        total
    }

    /// Le produit matriciel quand `a` a au plus [`NARROW_ROWS`] lignes,
    /// par le kernel `matmul_narrow`. `shape` est celle de la sortie. Avec
    /// `bias`, `[cols]`, le biais est ajouté à chaque ligne.
    ///
    /// Le nombre de tranches le long de `inner` est choisi pour que la
    /// grille compte au moins [`NARROW_BLOCKS`] blocs, sans tranche plus
    /// courte que [`NARROW_MIN_DEPTH`]. Les sommes partielles des tranches
    /// sont ensuite additionnées.
    #[allow(clippy::too_many_arguments)]
    fn matmul_narrow(
        &self,
        a: &CudaTensor,
        layout_a: Layout,
        b: &CudaTensor,
        layout_b: Layout,
        batch_len: usize,
        shape: Vec<usize>,
        inner: usize,
        bias: Option<&CudaTensor>,
    ) -> CudaTensor {
        let (rows, cols) = (shape[shape.len() - 2], shape[shape.len() - 1]);
        let column_blocks = cols.div_ceil(WARP);
        let wanted = NARROW_BLOCKS.div_ceil(column_blocks * batch_len);
        let splits = wanted.clamp(1, inner.div_ceil(NARROW_MIN_DEPTH).max(1));
        // Une tranche est un multiple de 32 : quatre indices par warp, et
        // huit warps.
        let depth = inner.div_ceil(splits).next_multiple_of(WARP);
        let splits = inner.div_ceil(depth);
        let grid = (column_blocks as u32, (batch_len * splits) as u32);
        assert!(
            grid.0 <= MAX_GRID_BLOCKS && grid.1 <= MAX_GRID_BLOCKS,
            "matmul trop grande pour une seule grille : {} × {} blocs",
            grid.0,
            grid.1,
        );
        let out = self.alloc(batch_len * splits * rows * cols);
        let out_ptr = out.cu_deviceptr() as *mut f32;
        // Sans biais, le kernel reçoit un tableau qu'il ne lira pas.
        let (bias_buffer, has_bias) = match bias {
            Some(bias) => (&bias.storage.buffer, 1u32),
            None => (&a.storage.buffer, 0u32),
        };
        let config = LaunchConfig2D::new(grid, (WARP as u32, 8), 0);
        // SAFETY: la grille couvre `cols` par 32 en x et `lots × tranches`
        // en y, et `out` a `lots × tranches × rows × cols` éléments.
        unsafe {
            launch!(
                self,
                matmul_narrow,
                prepare_matmul_narrow,
                config,
                &a.storage.buffer,
                layout_a,
                &b.storage.buffer,
                layout_b,
                rows as u32,
                inner as u32,
                cols as u32,
                depth as u32,
                bias_buffer,
                has_bias,
                out_ptr
            );
        }
        if splits == 1 {
            return self.wrap(out, shape);
        }
        let total = self.sum_splits(&out, batch_len, splits, rows * cols, bias, cols);
        self.wrap(total, shape)
    }

    /// `x · W + b` pour `x` `[…, K]`, une matrice `W` et un biais `b` : la
    /// couche affine, par [`Cuda::product`].
    pub fn affine(&self, x: &CudaTensor, weight: &CudaTensor, bias: &CudaTensor) -> CudaTensor {
        let inner = weight.shape[0];
        assert!(
            weight.shape.len() == 2 && x.shape.last() == Some(&inner),
            "affine : {:?} · {:?}",
            x.shape,
            weight.shape,
        );
        let rows = x.storage.buffer.len() / inner;
        let flat = self.reshape(x, &[rows, inner]);
        let out = self.product(&flat, weight, Some(bias));
        let mut shape = x.shape.clone();
        *shape.last_mut().expect("au moins une dimension") = weight.shape[1];
        self.reshape(&out, &shape)
    }

    fn wrap(&self, buffer: Pooled<f32>, shape: Vec<usize>) -> CudaTensor {
        CudaTensor {
            storage: Rc::new(Storage { buffer }),
            shape,
        }
    }

    fn binary(&self, op: BinaryOp, param: f32, a: &CudaTensor, b: &CudaTensor) -> CudaTensor {
        let shape = broadcast_shapes(&a.shape, &b.shape);
        let layout_a = layout(&shape, &broadcast_strides(&a.shape, &shape));
        let layout_b = layout(&shape, &broadcast_strides(&b.shape, &shape));
        let len = shape.iter().product();
        let mut out = self.alloc(len);

        launch!(
            self,
            binary,
            prepare_binary,
            launch_config(len),
            op,
            param,
            &a.storage.buffer,
            layout_a,
            &b.storage.buffer,
            layout_b,
            &mut out
        );
        self.wrap(out, shape)
    }

    fn unary(&self, op: UnaryOp, a: &CudaTensor) -> CudaTensor {
        let len = a.storage.buffer.len();
        let mut out = self.alloc(len);
        launch!(
            self,
            unary,
            prepare_unary,
            launch_config(len),
            op,
            &a.storage.buffer,
            &mut out
        );
        self.wrap(out, a.shape.clone())
    }

    /// Maximum et `ln(Σ exp(x - max))` de chaque ligne du dernier axe.
    fn row_stats(&self, a: &CudaTensor) -> (Pooled<f32>, Pooled<f32>) {
        let width = *a
            .shape
            .last()
            .expect("softmax demande au moins une dimension");
        let rows = a.storage.buffer.len() / width;
        let mut row_max = self.alloc(rows);
        let mut row_log_sum = self.alloc(rows);
        launch!(
            self,
            row_stats,
            prepare_row_stats,
            row_launch_config(rows),
            &a.storage.buffer,
            width as u32,
            &mut row_max,
            &mut row_log_sum
        );
        (row_max, row_log_sum)
    }

    /// Deux lancements : les réductions par ligne, puis un thread par
    /// élément qui s'en sert.
    fn softmax_rows(&self, mode: SoftmaxMode, a: &CudaTensor) -> CudaTensor {
        let (row_max, row_log_sum) = self.row_stats(a);
        let width = *a.shape.last().unwrap();
        let len = a.storage.buffer.len();
        let mut out = self.alloc(len);
        launch!(
            self,
            softmax_rows,
            prepare_softmax_rows,
            launch_config(len),
            mode,
            &a.storage.buffer,
            &row_max,
            &row_log_sum,
            width as u32,
            &mut out
        );
        self.wrap(out, a.shape.clone())
    }

    /// Nombre de lignes et largeur du dernier axe.
    fn rows_and_width(a: &CudaTensor) -> (usize, usize) {
        let width = *a
            .shape
            .last()
            .expect("opération par ligne sur un tenseur sans dimension");
        (a.storage.buffer.len() / width, width)
    }

    /// Moyenne et écart-type inverse de chaque ligne du dernier axe.
    fn row_moments(&self, a: &CudaTensor, epsilon: f32) -> (Pooled<f32>, Pooled<f32>) {
        let (rows, width) = Self::rows_and_width(a);
        let mut mean = self.alloc(rows);
        let mut inverse_std = self.alloc(rows);
        launch!(
            self,
            row_moments,
            prepare_row_moments,
            row_launch_config(rows),
            &a.storage.buffer,
            width as u32,
            epsilon,
            &mut mean,
            &mut inverse_std
        );
        (mean, inverse_std)
    }

    /// `1 / √(moyenne(x²) + epsilon)` de chaque ligne du dernier axe.
    fn row_inverse_rms(&self, a: &CudaTensor, epsilon: f32) -> Pooled<f32> {
        let (rows, width) = Self::rows_and_width(a);
        let mut out = self.alloc(rows);
        launch!(
            self,
            row_inverse_rms,
            prepare_row_inverse_rms,
            row_launch_config(rows),
            &a.storage.buffer,
            width as u32,
            epsilon,
            &mut out
        );
        out
    }

    /// `Σ a × b` sur chaque ligne du dernier axe.
    fn row_dot(&self, a: &CudaTensor, b: &CudaTensor) -> Pooled<f32> {
        assert_eq!(a.shape, b.shape, "row_dot : formes différentes");
        let (rows, width) = Self::rows_and_width(a);
        let mut out = self.alloc(rows);
        launch!(
            self,
            row_dot,
            prepare_row_dot,
            row_launch_config(rows),
            &a.storage.buffer,
            &b.storage.buffer,
            width as u32,
            &mut out
        );
        out
    }
}

/// Les opérations qui ont leurs propres kernels. Chacune est le pendant GPU
/// d'une structure de [`super::ops`], qui l'appelle.
impl Cuda {
    pub(crate) fn gelu(&self, a: &CudaTensor) -> CudaTensor {
        self.unary(UnaryOp::Gelu, a)
    }

    pub(crate) fn gelu_backward(&self, a: &CudaTensor, grad: &CudaTensor) -> CudaTensor {
        assert_eq!(a.shape, grad.shape, "gradient de forme inattendue");
        self.binary(BinaryOp::GeluBackward, 0.0, a, grad)
    }

    pub(crate) fn silu(&self, a: &CudaTensor) -> CudaTensor {
        self.unary(UnaryOp::Silu, a)
    }

    pub(crate) fn silu_backward(&self, a: &CudaTensor, grad: &CudaTensor) -> CudaTensor {
        assert_eq!(a.shape, grad.shape, "gradient de forme inattendue");
        self.binary(BinaryOp::SiluBackward, 0.0, a, grad)
    }

    /// Deux lancements : les réductions par ligne, puis un thread par
    /// élément qui s'en sert.
    pub(crate) fn normalize(&self, a: &CudaTensor, epsilon: f32) -> CudaTensor {
        let (mean, inverse_std) = self.row_moments(a, epsilon);
        let (_, width) = Self::rows_and_width(a);
        let len = a.storage.buffer.len();
        let mut out = self.alloc(len);
        launch!(
            self,
            normalize_rows,
            prepare_normalize_rows,
            launch_config(len),
            &a.storage.buffer,
            &mean,
            &inverse_std,
            width as u32,
            &mut out
        );
        self.wrap(out, a.shape.clone())
    }

    pub(crate) fn normalize_backward(
        &self,
        a: &CudaTensor,
        normalized: &CudaTensor,
        grad: &CudaTensor,
        epsilon: f32,
    ) -> CudaTensor {
        assert_eq!(a.shape, grad.shape, "gradient de forme inattendue");
        let (_, inverse_std) = self.row_moments(a, epsilon);
        let grad_sum = self.sum(grad, grad.shape.len() - 1, false);
        let grad_dot = self.row_dot(grad, normalized);

        let (_, width) = Self::rows_and_width(a);
        let len = a.storage.buffer.len();
        let mut out = self.alloc(len);
        launch!(
            self,
            normalize_grad_rows,
            prepare_normalize_grad_rows,
            launch_config(len),
            &normalized.storage.buffer,
            &grad.storage.buffer,
            &grad_sum.storage.buffer,
            &grad_dot,
            &inverse_std,
            width as u32,
            &mut out
        );
        self.wrap(out, a.shape.clone())
    }

    /// Deux lancements : la réduction par ligne, puis un thread par élément.
    pub(crate) fn rms_normalize(&self, a: &CudaTensor, epsilon: f32) -> CudaTensor {
        let inverse_rms = self.row_inverse_rms(a, epsilon);
        let (_, width) = Self::rows_and_width(a);
        let len = a.storage.buffer.len();
        let mut out = self.alloc(len);
        launch!(
            self,
            scale_rows,
            prepare_scale_rows,
            launch_config(len),
            &a.storage.buffer,
            &inverse_rms,
            width as u32,
            &mut out
        );
        self.wrap(out, a.shape.clone())
    }

    pub(crate) fn rms_normalize_backward(
        &self,
        a: &CudaTensor,
        normalized: &CudaTensor,
        grad: &CudaTensor,
        epsilon: f32,
    ) -> CudaTensor {
        assert_eq!(a.shape, grad.shape, "gradient de forme inattendue");
        let inverse_rms = self.row_inverse_rms(a, epsilon);
        let grad_dot = self.row_dot(grad, normalized);
        let (_, width) = Self::rows_and_width(a);
        let len = a.storage.buffer.len();
        let mut out = self.alloc(len);
        launch!(
            self,
            rms_grad_rows,
            prepare_rms_grad_rows,
            launch_config(len),
            &normalized.storage.buffer,
            &grad.storage.buffer,
            &grad_dot,
            &inverse_rms,
            width as u32,
            &mut out
        );
        self.wrap(out, a.shape.clone())
    }

    /// Le masque est calculé par les kernels à partir de la position :
    /// aucun tenseur de masque n'est construit ni envoyé.
    pub(crate) fn causal_softmax(&self, a: &CudaTensor, scale: f32) -> CudaTensor {
        let rank = a.shape.len();
        assert!(
            rank >= 2 && a.shape[rank - 2] <= a.shape[rank - 1],
            "causal_softmax attend des scores […, N, T] avec N ≤ T, reçu {:?}",
            a.shape,
        );
        let queries = a.shape[rank - 2] as u32;
        let (rows, width) = Self::rows_and_width(a);
        let mut row_max = self.alloc(rows);
        let mut row_log_sum = self.alloc(rows);
        launch!(
            self,
            causal_row_stats,
            prepare_causal_row_stats,
            row_launch_config(rows),
            &a.storage.buffer,
            width as u32,
            queries,
            scale,
            &mut row_max,
            &mut row_log_sum
        );

        let len = a.storage.buffer.len();
        let mut out = self.alloc(len);
        launch!(
            self,
            causal_softmax_rows,
            prepare_causal_softmax_rows,
            launch_config(len),
            &a.storage.buffer,
            &row_max,
            &row_log_sum,
            width as u32,
            queries,
            scale,
            &mut out
        );
        self.wrap(out, a.shape.clone())
    }

    /// Les dimensions d'une attention `q [H…, N, Dk]`, `k [H…, T, Dk]`,
    /// `v [H…, T, Dv]` que les kernels fusionnés savent traiter :
    /// `(têtes, N, T, Dk, Dv)`. `None` si elles dépassent leurs limites, ou
    /// si l'attention fusionnée est désactivée.
    pub(crate) fn fused_attention_dims(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
    ) -> Option<(usize, usize, usize, usize, usize)> {
        let rank = q.shape.len();
        assert!(
            rank >= 2 && k.shape.len() == rank && v.shape.len() == rank,
            "attention : formes {:?}, {:?}, {:?}",
            q.shape,
            k.shape,
            v.shape,
        );
        let (queries, key_dim) = (q.shape[rank - 2], q.shape[rank - 1]);
        let (positions, value_dim) = (k.shape[rank - 2], v.shape[rank - 1]);
        assert!(
            q.shape[..rank - 2] == k.shape[..rank - 2]
                && k.shape[..rank - 2] == v.shape[..rank - 2]
                && k.shape[rank - 1] == key_dim
                && v.shape[rank - 2] == positions
                && queries <= positions,
            "attention : formes incompatibles {:?}, {:?}, {:?}",
            q.shape,
            k.shape,
            v.shape,
        );
        let heads: usize = q.shape[..rank - 2].iter().product();
        let fits = key_dim <= ATTENTION_MAX_DIM
            && value_dim <= ATTENTION_MAX_DIM
            && heads <= MAX_GRID_BLOCKS as usize;
        (self.fused_attention && fits).then_some((heads, queries, positions, key_dim, value_dim))
    }

    /// `a · bᵀ` pour `a` `[…, K]` d'au plus [`NARROW_ROWS`] lignes et une
    /// matrice `b` `[N, K]`, par `transposed_matmul_narrow`.
    pub fn transposed_matmul_narrow(&self, a: &CudaTensor, b: &CudaTensor) -> CudaTensor {
        let (n, inner) = (b.shape[0], b.shape[1]);
        let rows = a.storage.buffer.len() / inner;
        assert!(
            b.shape.len() == 2 && a.shape.last() == Some(&inner) && rows <= NARROW_ROWS,
            "transposed_matmul : {:?} · {:?}ᵀ",
            a.shape,
            b.shape,
        );
        let out = self.alloc(rows * n);
        let config = LaunchConfig2D::new((n.div_ceil(8) as u32, 1), (WARP as u32, 8), 0);
        // SAFETY: la grille couvre les `n` lignes de `b` par 8, et `out` a
        // `rows × n` éléments.
        unsafe {
            launch!(
                self,
                transposed_matmul_narrow,
                prepare_transposed_matmul_narrow,
                config,
                &a.storage.buffer,
                &b.storage.buffer,
                rows as u32,
                inner as u32,
                n as u32,
                out.cu_deviceptr() as *mut f32
            );
        }
        let mut shape = a.shape.clone();
        *shape.last_mut().expect("au moins une dimension") = n;
        self.wrap(out, shape)
    }

    /// L'attention sur le cache de génération (voir
    /// `smallem::backend::ops::CachedAttention`), par `attention_cached`, ou
    /// par `attention_cached_wide` pour les têtes de plus de
    /// [`ATTENTION_MAX_DIM`] composantes. Elles doivent en faire au plus
    /// [`CACHED_WIDE_DIM`].
    ///
    /// Les clés de chaque requête sont découpées en tranches pour que la
    /// grille compte au moins `NARROW_BLOCKS` (256) blocs, sans tranche de moins
    /// de 32 positions. Avec plusieurs tranches, un second kernel les réunit.
    #[allow(clippy::too_many_arguments)]
    pub fn cached_attention(
        &self,
        q: &CudaTensor,
        keys: &CudaTensor,
        values: &CudaTensor,
        slots: &CudaTensor,
        rows: &CudaTensor,
        dims: CachedAttentionDims,
        scale: f32,
    ) -> CudaTensor {
        assert!(
            dims.key_dim <= CACHED_WIDE_DIM && dims.value_dim <= CACHED_WIDE_DIM,
            "attention sur le cache : têtes de {} et {} composantes, {CACHED_WIDE_DIM} au plus",
            dims.key_dim,
            dims.value_dim,
        );
        let wide = dims.key_dim > ATTENTION_MAX_DIM || dims.value_dim > ATTENTION_MAX_DIM;
        let lines = dims.queries * dims.heads;
        let wanted = NARROW_BLOCKS.div_ceil(lines);
        let splits = wanted.clamp(1, dims.max_len.div_ceil(WARP).max(1));
        let chunk = dims.max_len.div_ceil(splits).next_multiple_of(16);
        let splits = dims.max_len.div_ceil(chunk);
        let grid = (lines as u32, splits as u32);
        let len = lines * dims.value_dim;
        let shape = vec![dims.queries, dims.heads, dims.value_dim];
        let kernel_dims = CachedDims {
            heads: dims.heads as u32,
            kv_heads: dims.kv_heads as u32,
            key_dim: dims.key_dim as u32,
            value_dim: dims.value_dim as u32,
            value_stride: dims.value_stride as u32,
            max_len: dims.max_len as u32,
        };
        let parts = self.alloc(splits * len);
        let stats = self.alloc(splits * lines * 2);
        let config = LaunchConfig2D::new(grid, (WARP as u32, 4), 0);
        // SAFETY: la grille couvre les tranches et les lignes, `parts` et
        // `stats` ont les tailles qu'attend le kernel, et les têtes tiennent
        // dans ses registres (vérifié ci-dessus).
        unsafe {
            if wide {
                launch!(
                    self,
                    attention_cached_wide,
                    prepare_attention_cached_wide,
                    config,
                    &q.storage.buffer,
                    &keys.storage.buffer,
                    &values.storage.buffer,
                    &slots.storage.buffer,
                    &rows.storage.buffer,
                    kernel_dims,
                    chunk as u32,
                    scale,
                    parts.cu_deviceptr() as *mut f32,
                    stats.cu_deviceptr() as *mut f32
                );
            } else {
                launch!(
                    self,
                    attention_cached,
                    prepare_attention_cached,
                    config,
                    &q.storage.buffer,
                    &keys.storage.buffer,
                    &values.storage.buffer,
                    &slots.storage.buffer,
                    &rows.storage.buffer,
                    kernel_dims,
                    chunk as u32,
                    scale,
                    parts.cu_deviceptr() as *mut f32,
                    stats.cu_deviceptr() as *mut f32
                );
            }
        }
        if splits == 1 {
            return self.wrap(parts, shape);
        }
        let mut out = self.alloc(len);
        launch!(
            self,
            attention_cached_combine,
            prepare_attention_cached_combine,
            launch_config(len),
            &parts,
            &stats,
            splits as u32,
            dims.value_dim as u32,
            &mut out
        );
        self.wrap(out, shape)
    }

    /// Côté des tuiles des kernels d'attention, et forme de leurs blocs :
    /// 64 lignes et quatre warps sur les Tensor Cores, 32 lignes et
    /// `16 × 16` threads sinon.
    fn attention_tiles(&self) -> (u32, (u32, u32)) {
        if self.mixed_precision {
            (WIDE_TILE as u32, (32, 4))
        } else {
            (ATTENTION_TILE as u32, (16, 16))
        }
    }

    /// L'attention fusionnée : la sortie `[H…, N, Dv]`, et `log_sum`
    /// `[H…, N]` pour la passe arrière.
    pub(crate) fn attention(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
        scale: f32,
        (heads, queries, positions, key_dim, value_dim): (usize, usize, usize, usize, usize),
    ) -> (CudaTensor, CudaTensor) {
        let out = self.alloc(heads * queries * value_dim);
        let log_sum = self.alloc(heads * queries);
        let (tile, block) = self.attention_tiles();
        let grid = ((queries as u32).div_ceil(tile), heads as u32);
        let config = LaunchConfig2D::new(grid, block, 0);
        let dims = (
            queries as u32,
            positions as u32,
            key_dim as u32,
            value_dim as u32,
        );
        let (out_ptr, log_sum_ptr) = (
            out.cu_deviceptr() as *mut f32,
            log_sum.cu_deviceptr() as *mut f32,
        );
        // Les deux versions ont les mêmes arguments.
        macro_rules! forward {
            ($kernel:ident, $prepare:ident) => {
                launch!(
                    self,
                    $kernel,
                    $prepare,
                    config,
                    &q.storage.buffer,
                    &k.storage.buffer,
                    &v.storage.buffer,
                    dims.0,
                    dims.1,
                    dims.2,
                    dims.3,
                    scale,
                    out_ptr,
                    log_sum_ptr
                )
            };
        }
        // SAFETY: une tuile de requêtes par bloc en x, une tête en y, des
        // blocs de la forme qu'attend chaque kernel ; les sorties ont les
        // tailles voulues, et les largeurs ont été vérifiées par
        // `fused_attention_dims`.
        unsafe {
            if self.mixed_precision {
                forward!(attention_forward_bf16, prepare_attention_forward_bf16);
            } else {
                forward!(attention_forward, prepare_attention_forward);
            }
        }
        let mut shape = q.shape.clone();
        *shape.last_mut().expect("rang vérifié") = value_dim;
        let rows = shape[..shape.len() - 1].to_vec();
        (self.wrap(out, shape), self.wrap(log_sum, rows))
    }

    /// Les gradients de `q`, `k` et `v`, à partir de la sortie, de son
    /// gradient et de ce que la passe avant a gardé.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_backward(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
        output: &CudaTensor,
        log_sum: &CudaTensor,
        grad: &CudaTensor,
        scale: f32,
        (heads, queries, positions, key_dim, value_dim): (usize, usize, usize, usize, usize),
    ) -> [CudaTensor; 3] {
        // `Σ dO × O` par ligne : le terme de la règle du softmax.
        let delta = self.row_dot(grad, output);
        let grad_q = self.alloc(q.storage.buffer.len());
        let grad_k = self.alloc(k.storage.buffer.len());
        let grad_v = self.alloc(v.storage.buffer.len());
        let (tile, block) = self.attention_tiles();
        let config =
            |len: usize| LaunchConfig2D::new(((len as u32).div_ceil(tile), heads as u32), block, 0);
        let dims = (
            queries as u32,
            positions as u32,
            key_dim as u32,
            value_dim as u32,
        );
        let (grad_q_ptr, grad_k_ptr, grad_v_ptr) = (
            grad_q.cu_deviceptr() as *mut f32,
            grad_k.cu_deviceptr() as *mut f32,
            grad_v.cu_deviceptr() as *mut f32,
        );
        // Les quatre kernels ont les mêmes arguments, à leurs sorties près.
        macro_rules! backward {
            ($kernel:ident, $prepare:ident, $len:expr, $($out:expr),+) => {
                launch!(
                    self,
                    $kernel,
                    $prepare,
                    config($len),
                    &q.storage.buffer,
                    &k.storage.buffer,
                    &v.storage.buffer,
                    &grad.storage.buffer,
                    &log_sum.storage.buffer,
                    &delta,
                    dims.0,
                    dims.1,
                    dims.2,
                    dims.3,
                    scale,
                    $($out),+
                )
            };
        }
        // SAFETY: grilles et tailles comme pour `attention`, une tuile de
        // clés par bloc pour le premier kernel de chaque paire ; les
        // gradients ont la taille de leurs entrées.
        unsafe {
            if self.mixed_precision {
                backward!(
                    attention_backward_keys_bf16,
                    prepare_attention_backward_keys_bf16,
                    positions,
                    grad_k_ptr,
                    grad_v_ptr
                );
                backward!(
                    attention_backward_queries_bf16,
                    prepare_attention_backward_queries_bf16,
                    queries,
                    grad_q_ptr
                );
            } else {
                backward!(
                    attention_backward_keys,
                    prepare_attention_backward_keys,
                    positions,
                    grad_k_ptr,
                    grad_v_ptr
                );
                backward!(
                    attention_backward_queries,
                    prepare_attention_backward_queries,
                    queries,
                    grad_q_ptr
                );
            }
        }
        [
            self.wrap(grad_q, q.shape.clone()),
            self.wrap(grad_k, k.shape.clone()),
            self.wrap(grad_v, v.shape.clone()),
        ]
    }

    /// L'attention sans fusion, telle qu'elle était avant l'étape 12 : les
    /// scores, le softmax causal, le mélange des valeurs. Rend aussi les
    /// poids, que la passe arrière relira.
    pub(crate) fn attention_unfused(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
        scale: f32,
    ) -> (CudaTensor, CudaTensor) {
        let rank = k.shape.len();
        let scores = self.matmul(q, &self.transpose(k, rank - 2, rank - 1));
        let weights = self.causal_softmax(&scores, scale);
        (self.matmul(&weights, v), weights)
    }

    /// Les gradients de [`Cuda::attention_unfused`], dans l'ordre où la
    /// tape les calculait.
    pub(crate) fn attention_unfused_backward(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
        weights: &CudaTensor,
        grad: &CudaTensor,
        scale: f32,
    ) -> [CudaTensor; 3] {
        let rank = k.shape.len();
        let transposed = |t: &CudaTensor| self.transpose(t, rank - 2, rank - 1);
        let grad_v = self.matmul(&transposed(weights), grad);
        let grad_weights = self.matmul(grad, &transposed(v));
        let grad_scores = self.softmax_backward(weights, &grad_weights, scale);
        let grad_q = self.matmul(&grad_scores, k);
        let grad_k = self.matmul(&transposed(&grad_scores), q);
        [grad_q, grad_k, grad_v]
    }

    pub(crate) fn softmax_backward(
        &self,
        probabilities: &CudaTensor,
        grad: &CudaTensor,
        scale: f32,
    ) -> CudaTensor {
        let row_dot = self.row_dot(grad, probabilities);
        let (_, width) = Self::rows_and_width(probabilities);
        let len = probabilities.storage.buffer.len();
        let mut out = self.alloc(len);
        launch!(
            self,
            softmax_grad_rows,
            prepare_softmax_grad_rows,
            launch_config(len),
            &probabilities.storage.buffer,
            &grad.storage.buffer,
            &row_dot,
            width as u32,
            scale,
            &mut out
        );
        self.wrap(out, probabilities.shape.clone())
    }

    pub(crate) fn rotate_pairs(
        &self,
        a: &CudaTensor,
        cos: &CudaTensor,
        sin: &CudaTensor,
        inverse: bool,
    ) -> CudaTensor {
        let rank = a.shape.len();
        assert!(
            rank >= 2 && a.shape[rank - 1].is_multiple_of(2),
            "rotate_pairs attend […, T, D] avec D pair, reçu {:?}",
            a.shape,
        );
        let (length, width) = (a.shape[rank - 2], a.shape[rank - 1]);
        let half = width / 2;
        assert!(
            cos.shape == [length, half] && sin.shape == [length, half],
            "rotate_pairs : angles {:?} et {:?} pour un tenseur {:?}, [{length}, {half}] attendu",
            cos.shape,
            sin.shape,
            a.shape,
        );
        let len = a.storage.buffer.len();
        let mut out = self.alloc(len);
        launch!(
            self,
            rotate_pairs,
            prepare_rotate_pairs,
            launch_config(len),
            &a.storage.buffer,
            &cos.storage.buffer,
            &sin.storage.buffer,
            width as u32,
            length as u32,
            if inverse { -1.0 } else { 1.0 },
            &mut out
        );
        self.wrap(out, a.shape.clone())
    }

    pub(crate) fn concat_last(&self, a: &CudaTensor, b: &CudaTensor) -> CudaTensor {
        let (width_a, width_b) = concat_widths(&a.shape, &b.shape);
        let mut shape = a.shape.clone();
        *shape.last_mut().expect("au moins une dimension") = width_a + width_b;
        let len = shape.iter().product();
        let mut out = self.alloc(len);
        launch!(
            self,
            concat_last,
            prepare_concat_last,
            launch_config(len),
            &a.storage.buffer,
            &b.storage.buffer,
            width_a as u32,
            width_b as u32,
            &mut out
        );
        self.wrap(out, shape)
    }

    /// Les `width` composantes du dernier axe de `a` qui commencent à
    /// `offset`.
    pub(crate) fn slice_last(&self, a: &CudaTensor, offset: usize, width: usize) -> CudaTensor {
        let total = *a.shape.last().expect("au moins une dimension");
        assert!(
            offset + width <= total,
            "slice_last : {offset}..{} dans un axe de {total}",
            offset + width,
        );
        let mut shape = a.shape.clone();
        *shape.last_mut().expect("au moins une dimension") = width;
        let len = shape.iter().product();
        let mut out = self.alloc(len);
        launch!(
            self,
            slice_last,
            prepare_slice_last,
            launch_config(len),
            &a.storage.buffer,
            total as u32,
            offset as u32,
            width as u32,
            &mut out
        );
        self.wrap(out, shape)
    }

    pub(crate) fn cross_entropy(&self, logits: &CudaTensor, targets: &[u32]) -> CudaTensor {
        let (n, vocab) = cross_entropy_dims(logits, targets);
        let device_targets = self.upload_indices(targets);
        let mut losses = self.alloc(n);

        launch!(
            self,
            cross_entropy_rows,
            prepare_cross_entropy_rows,
            row_launch_config(n),
            &logits.storage.buffer,
            &device_targets,
            vocab as u32,
            &mut losses
        );

        let total = self.sum_all(&self.wrap(losses, vec![n]));
        self.mul(&total, &self.scalar(1.0 / n as f32))
    }

    pub(crate) fn cross_entropy_grad(&self, logits: &CudaTensor, targets: &[u32]) -> CudaTensor {
        let (n, vocab) = cross_entropy_dims(logits, targets);
        let (row_max, row_log_sum) = self.row_stats(logits);
        let device_targets = self.upload_indices(targets);
        let len = n * vocab;
        let mut out = self.alloc(len);

        launch!(
            self,
            cross_entropy_grad,
            prepare_cross_entropy_grad,
            launch_config(len),
            &logits.storage.buffer,
            &row_max,
            &row_log_sum,
            &device_targets,
            vocab as u32,
            1.0 / n as f32,
            &mut out
        );

        self.wrap(out, logits.shape.clone())
    }
}

impl Primitives for Cuda {
    type Scalar = f32;
    type Tensor = CudaTensor;

    /// Rempli sur la carte, sans envoi : le cache de génération crée ainsi
    /// des tables de plusieurs centaines de mégaoctets.
    fn full(&self, shape: &[usize], value: f32) -> CudaTensor {
        assert!(
            shape.len() <= MAX_RANK,
            "le backend CUDA gère au plus {MAX_RANK} dimensions, reçu {shape:?}",
        );
        let len: usize = shape.iter().product();
        if len == 0 {
            return self.from_vec(Vec::new(), shape);
        }
        let mut out = self.alloc(len);
        launch!(
            self,
            fill,
            prepare_fill,
            launch_config(len),
            value,
            &mut out
        );
        self.wrap(out, shape.to_vec())
    }

    fn from_vec(&self, data: Vec<f32>, shape: &[usize]) -> CudaTensor {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "{} éléments fournis pour la forme {shape:?}",
            data.len(),
        );
        assert!(
            shape.len() <= MAX_RANK,
            "le backend CUDA gère au plus {MAX_RANK} dimensions, reçu {shape:?}",
        );
        let mut buffer = self.alloc(data.len());
        buffer
            .copy_from_host(&self.stream, &data)
            .expect(CUDA_ERROR);
        self.wrap(buffer, shape.to_vec())
    }

    fn to_vec(&self, tensor: &CudaTensor) -> Vec<f32> {
        tensor
            .storage
            .buffer
            .to_host_vec(&self.stream)
            .expect(CUDA_ERROR)
    }

    fn shape<'a>(&self, tensor: &'a CudaTensor) -> &'a [usize] {
        &tensor.shape
    }

    fn add(&self, a: &CudaTensor, b: &CudaTensor) -> CudaTensor {
        self.binary(BinaryOp::Add, 0.0, a, b)
    }

    fn sub(&self, a: &CudaTensor, b: &CudaTensor) -> CudaTensor {
        self.binary(BinaryOp::Sub, 0.0, a, b)
    }

    fn mul(&self, a: &CudaTensor, b: &CudaTensor) -> CudaTensor {
        self.binary(BinaryOp::Mul, 0.0, a, b)
    }

    fn div(&self, a: &CudaTensor, b: &CudaTensor) -> CudaTensor {
        self.binary(BinaryOp::Div, 0.0, a, b)
    }

    fn neg(&self, a: &CudaTensor) -> CudaTensor {
        self.unary(UnaryOp::Neg, a)
    }

    fn exp(&self, a: &CudaTensor) -> CudaTensor {
        self.unary(UnaryOp::Exp, a)
    }

    fn ln(&self, a: &CudaTensor) -> CudaTensor {
        self.unary(UnaryOp::Ln, a)
    }

    fn sqrt(&self, a: &CudaTensor) -> CudaTensor {
        self.unary(UnaryOp::Sqrt, a)
    }

    fn tanh(&self, a: &CudaTensor) -> CudaTensor {
        self.unary(UnaryOp::Tanh, a)
    }

    fn sum(&self, a: &CudaTensor, axis: usize, keepdim: bool) -> CudaTensor {
        assert!(
            axis < a.shape.len(),
            "axe {axis} hors de la forme {:?}",
            a.shape,
        );
        let n = a.shape[axis];
        // Somme le long de l'avant-dernier axe quand il est long, comme
        // celle des lignes d'une matrice `[N, C]` : le kernel général y
        // consacre un seul thread par colonne, qui lit `N` cases éloignées
        // l'une après l'autre. Amener cet axe en dernier donne à chaque
        // colonne un warp et des lectures voisines.
        if axis + 2 == a.shape.len() && n >= LONG_AXIS {
            let last = axis + 1;
            let summed = self.sum(&self.transpose(a, axis, last), last, false);
            return if keepdim {
                let mut shape = a.shape.clone();
                shape[axis] = 1;
                self.reshape(&summed, &shape)
            } else {
                summed
            };
        }
        let outer: usize = a.shape[..axis].iter().product();
        let inner: usize = a.shape[axis + 1..].iter().product();
        let len = outer * inner;
        let mut out = self.alloc(len);

        if inner == 1 {
            // Somme le long du dernier axe : une réduction par ligne.
            launch!(
                self,
                row_sum,
                prepare_row_sum,
                row_launch_config(len),
                &a.storage.buffer,
                n as u32,
                &mut out
            );
        } else {
            launch!(
                self,
                sum_axis,
                prepare_sum_axis,
                launch_config(len),
                &a.storage.buffer,
                n as u32,
                inner as u32,
                &mut out
            );
        }

        let mut shape = a.shape.clone();
        if keepdim {
            shape[axis] = 1;
        } else {
            shape.remove(axis);
        }
        self.wrap(out, shape)
    }

    /// Réduction en arbre : chaque passe divise le nombre d'éléments par
    /// `SUM_CHUNK`, jusqu'à ce qu'il n'en reste qu'un. Un million
    /// d'éléments demande trois passes.
    fn sum_all(&self, a: &CudaTensor) -> CudaTensor {
        let mut current = a.storage.clone();
        while current.buffer.len() > 1 {
            let len = current.buffer.len().div_ceil(SUM_CHUNK);
            let mut out = self.alloc(len);
            launch!(
                self,
                sum_chunks,
                prepare_sum_chunks,
                launch_config(len),
                &current.buffer,
                SUM_CHUNK as u32,
                &mut out
            );
            current = self.wrap(out, vec![len]).storage;
        }
        CudaTensor {
            storage: current,
            shape: Vec::new(),
        }
    }

    /// Gratuit : le nouveau tenseur partage les données de l'ancien, seule
    /// la forme change.
    fn reshape(&self, a: &CudaTensor, shape: &[usize]) -> CudaTensor {
        let expected: usize = shape.iter().product();
        assert_eq!(
            a.storage.buffer.len(),
            expected,
            "{} éléments fournis pour la forme {shape:?}",
            a.storage.buffer.len(),
        );
        assert!(
            shape.len() <= MAX_RANK,
            "le backend CUDA gère au plus {MAX_RANK} dimensions, reçu {shape:?}",
        );
        CudaTensor {
            storage: a.storage.clone(),
            shape: shape.to_vec(),
        }
    }

    fn transpose(&self, a: &CudaTensor, axis_a: usize, axis_b: usize) -> CudaTensor {
        let mut shape = a.shape.clone();
        shape.swap(axis_a, axis_b);
        let mut source_strides = strides(&a.shape);
        source_strides.swap(axis_a, axis_b);

        let len = a.storage.buffer.len();
        let mut out = self.alloc(len);
        launch!(
            self,
            permute,
            prepare_permute,
            launch_config(len),
            &a.storage.buffer,
            layout(&shape, &source_strides),
            &mut out
        );
        self.wrap(out, shape)
    }

    /// Le kernel de transposition sert aussi ici : avec des pas nuls sur
    /// les dimensions étirées, il relit le même élément.
    fn expand(&self, a: &CudaTensor, shape: &[usize]) -> CudaTensor {
        assert_eq!(
            broadcast_shapes(&a.shape, shape),
            shape,
            "la forme {:?} ne s'étire pas à {shape:?}",
            a.shape,
        );
        let len = shape.iter().product();
        let mut out = self.alloc(len);
        launch!(
            self,
            permute,
            prepare_permute,
            launch_config(len),
            &a.storage.buffer,
            layout(shape, &broadcast_strides(&a.shape, shape)),
            &mut out
        );
        self.wrap(out, shape.to_vec())
    }

    fn matmul(&self, a: &CudaTensor, b: &CudaTensor) -> CudaTensor {
        self.product(a, b, None)
    }

    fn softmax(&self, a: &CudaTensor) -> CudaTensor {
        self.softmax_rows(SoftmaxMode::Probability, a)
    }

    fn log_softmax(&self, a: &CudaTensor) -> CudaTensor {
        self.softmax_rows(SoftmaxMode::Log, a)
    }

    fn masked_fill(&self, a: &CudaTensor, mask: &CudaTensor, value: f32) -> CudaTensor {
        assert_eq!(
            broadcast_shapes(&a.shape, &mask.shape),
            a.shape,
            "le masque {:?} ne s'étire pas à la forme {:?}",
            mask.shape,
            a.shape,
        );
        self.binary(BinaryOp::MaskedFill, value, a, mask)
    }

    fn gather_rows(&self, a: &CudaTensor, indices: &[u32]) -> CudaTensor {
        assert!(
            !a.shape.is_empty(),
            "gather_rows demande au moins une dimension",
        );
        // Vérifié ici plutôt que dans le kernel : sur l'hôte, l'erreur peut
        // nommer l'indice fautif.
        for &index in indices {
            assert!(
                (index as usize) < a.shape[0],
                "indice {index} hors de la première dimension ({})",
                a.shape[0],
            );
        }
        let row_len: usize = a.shape[1..].iter().product();
        let len = indices.len() * row_len;
        let mut out = self.alloc(len);
        let device_indices = self.upload_indices(indices);

        launch!(
            self,
            gather_rows,
            prepare_gather_rows,
            launch_config(len),
            &a.storage.buffer,
            &device_indices,
            row_len as u32,
            &mut out
        );

        let mut shape = a.shape.clone();
        shape[0] = indices.len();
        self.wrap(out, shape)
    }

    fn scatter_add_rows(&self, a: &CudaTensor, indices: &[u32], num_rows: usize) -> CudaTensor {
        assert!(
            !a.shape.is_empty() && a.shape[0] == indices.len(),
            "scatter_add_rows : {} indices pour la forme {:?}",
            indices.len(),
            a.shape,
        );
        for &index in indices {
            assert!(
                (index as usize) < num_rows,
                "indice {index} hors de la première dimension ({num_rows})",
            );
        }
        let row_len: usize = a.shape[1..].iter().product();
        // Le kernel additionne dans `out` : il doit partir de zéro, alors que
        // la réserve rend des tampons au contenu quelconque.
        let mut out = self.alloc(num_rows * row_len);
        out.zero_async(&self.stream).expect(CUDA_ERROR);
        let device_indices = self.upload_indices(indices);

        let len = a.storage.buffer.len();
        // Adresse en mémoire GPU : à ne jamais déréférencer sur l'hôte.
        let out_ptr = out.cu_deviceptr() as *mut f32;
        // SAFETY: `out` vient d'être alloué avec `num_rows × row_len`
        // éléments, tous les indices ont été vérifiés ci-dessus, et les
        // kernels qui liront ce tampon sont mis en file après celui-ci sur
        // le même stream, donc exécutés après lui.
        unsafe {
            launch!(
                self,
                scatter_add_rows,
                prepare_scatter_add_rows,
                launch_config(len),
                &a.storage.buffer,
                &device_indices,
                row_len as u32,
                out_ptr
            );
        }

        let mut shape = a.shape.clone();
        shape[0] = num_rows;
        self.wrap(out, shape)
    }

    fn concat_rows(&self, a: &CudaTensor, b: &CudaTensor) -> CudaTensor {
        assert!(
            !a.shape.is_empty() && a.shape[1..] == b.shape[1..],
            "concat_rows : formes incompatibles {:?} et {:?}",
            a.shape,
            b.shape,
        );
        let len = a.storage.buffer.len() + b.storage.buffer.len();
        let mut out = self.alloc(len);
        launch!(
            self,
            concat,
            prepare_concat,
            launch_config(len),
            &a.storage.buffer,
            &b.storage.buffer,
            &mut out
        );
        let mut shape = a.shape.clone();
        shape[0] += b.shape[0];
        self.wrap(out, shape)
    }

    fn write_rows(&self, table: &mut CudaTensor, indices: &CudaTensor, rows: &CudaTensor) {
        assert!(
            !table.shape.is_empty() && !rows.shape.is_empty(),
            "write_rows demande au moins une dimension",
        );
        assert_eq!(
            table.shape[1..],
            rows.shape[1..],
            "write_rows : lignes {:?} pour une table {:?}",
            rows.shape,
            table.shape,
        );
        assert_eq!(
            indices.shape,
            [rows.shape[0]],
            "write_rows : indices {:?} pour {} lignes",
            indices.shape,
            rows.shape[0],
        );
        let row_len: usize = table.shape[1..].iter().product();
        let len = rows.storage.buffer.len();
        let table_ptr = unique_buffer(table).cu_deviceptr() as *mut f32;
        // SAFETY: la table n'est partagée avec aucun autre tenseur
        // (`unique_buffer`). Les indices viennent du cache, qui ne donne
        // que des lignes de la table, et jamais deux fois la même dans une
        // écriture.
        unsafe {
            launch!(
                self,
                write_rows,
                prepare_write_rows,
                launch_config(len),
                table_ptr,
                &indices.storage.buffer,
                &rows.storage.buffer,
                row_len as u32
            );
        }
    }

    fn sgd_step(&self, param: &mut CudaTensor, grad: &CudaTensor, learning_rate: f64) {
        assert_eq!(param.shape, grad.shape, "gradient de forme inattendue");
        let len = grad.storage.buffer.len();
        launch!(
            self,
            sgd_step,
            prepare_sgd_step,
            launch_config(len),
            unique_buffer(param),
            &grad.storage.buffer,
            learning_rate as f32
        );
    }

    fn adamw_step(
        &self,
        param: &mut CudaTensor,
        grad: &CudaTensor,
        first_moment: &mut CudaTensor,
        second_moment: &mut CudaTensor,
        step: &AdamWStep,
    ) {
        assert_eq!(param.shape, grad.shape, "gradient de forme inattendue");
        assert_eq!(
            param.shape, first_moment.shape,
            "moment de forme inattendue"
        );
        assert_eq!(
            param.shape, second_moment.shape,
            "moment de forme inattendue"
        );
        let params = AdamWParams {
            learning_rate: step.learning_rate as f32,
            beta1: step.beta1 as f32,
            beta2: step.beta2 as f32,
            epsilon: step.epsilon as f32,
            weight_decay: step.weight_decay as f32,
            bias_correction1: step.bias_correction1 as f32,
            bias_correction2: step.bias_correction2 as f32,
            grad_scale: step.grad_scale as f32,
        };
        let len = grad.storage.buffer.len();
        launch!(
            self,
            adamw_step,
            prepare_adamw_step,
            launch_config(len),
            unique_buffer(param),
            &grad.storage.buffer,
            unique_buffer(first_moment),
            unique_buffer(second_moment),
            params
        );
    }
}

/// Accès en écriture au tampon d'un tenseur, pour une mise à jour sur place.
/// Il n'est accordé que si ce tenseur est le seul à référencer ses données :
/// modifier un tampon partagé changerait d'autres tenseurs dans leur dos.
fn unique_buffer(tensor: &mut CudaTensor) -> &mut DeviceBuffer<f32> {
    let storage = Rc::get_mut(&mut tensor.storage).expect(
        "mise à jour sur place d'un tenseur encore partagé : \
         la tape qui l'utilise doit être libérée avant le pas d'optimiseur",
    );
    &mut storage.buffer
}

/// Vérifie les formes de l'entropie croisée et renvoie `(N, V)`.
fn cross_entropy_dims(logits: &CudaTensor, targets: &[u32]) -> (usize, usize) {
    assert!(
        logits.shape.len() == 2 && logits.shape[0] == targets.len(),
        "cross_entropy : logits {:?} pour {} cibles, [N, V] attendu",
        logits.shape,
        targets.len(),
    );
    let vocab = logits.shape[1];
    for &target in targets {
        assert!(
            (target as usize) < vocab,
            "cible {target} hors du vocabulaire ({vocab})",
        );
    }
    (logits.shape[0], vocab)
}
