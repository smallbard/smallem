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
//!
//! Ce fichier tient le backend lui-même : les kernels chargés, la réserve
//! de tampons, les tenseurs. Les lancements sont rangés par famille :
//! `primitives` (le trait `Primitives`), `matmul`, `rows` (les opérations
//! par ligne), `elementwise` et `attention`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::sync::Arc;

use cuda_core::{
    CudaContext, CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig1D, LaunchConfig2D,
};

use smallem::backend::ops::{CachedAttentionDims, concat_widths};
use smallem::backend::shape::{
    broadcast_shapes, broadcast_strides, check_expand, check_mask, check_step, concat_rows_shape,
    cross_entropy_dims, gather_rows_dims, scatter_add_rows_dims, strides, write_rows_dims,
};
use smallem::backend::{AdamWStep, Primitives};
use smallem::quantize::{Format, Quantized};

use crate::kernels;
use crate::kernels::{
    ATTENTION_MAX_DIM, ATTENTION_TILE, BLOCK_SIZE, BinaryOp, CachedDims, Layout, MATMUL_BF16_TILE,
    MATMUL_DEPTH, MATMUL_PATCH, MATMUL_TILE, MAX_RANK, QuantizedDims, SoftmaxMode, UnaryOp, WARP,
    WIDE_TILE,
};

/// Une erreur du pilote CUDA n'est pas rattrapable ici : on panique, comme
/// pour une erreur de forme.
const CUDA_ERROR: &str = "erreur CUDA";

/// Taille des paquets de la somme globale.
const SUM_CHUNK: usize = 256;

/// Nombre maximal de blocs d'une grille de lancement le long d'une
/// dimension : une limite de CUDA.
const MAX_GRID_BLOCKS: u32 = 65_535;

/// Les kernels, une famille par fichier de [`crate::kernels`]. Chaque
/// fichier a son `#[cuda_module]`, mais tous les kernels du paquet sont
/// compilés dans un même PTX : il est chargé une fois, et chaque famille y
/// trouve les siens.
struct Kernels {
    elementwise: kernels::elementwise::device::LoadedModule,
    rows: kernels::rows::device::LoadedModule,
    matmul: kernels::matmul::device::LoadedModule,
    quantized: kernels::quantized::device::LoadedModule,
    attention: kernels::attention::device::LoadedModule,
    attention_bf16: kernels::attention_bf16::device::LoadedModule,
    cached: kernels::cached::device::LoadedModule,
}

impl Kernels {
    /// # Safety
    ///
    /// Le PTX embarqué dans le binaire doit être celui que ce paquet a
    /// compilé à partir de `crate::kernels`.
    unsafe fn load(context: &Arc<CudaContext>) -> Result<Self, Box<dyn std::error::Error>> {
        // SAFETY: d'après le contrat de cette fonction.
        let elementwise = unsafe { kernels::elementwise::device::load(context)? };
        let module = elementwise.as_cuda_module();
        // SAFETY: le même PTX contient les kernels de toutes les familles,
        // avec les signatures que déclarent leurs modules.
        unsafe {
            Ok(Self {
                rows: kernels::rows::device::from_module(module.clone())?,
                matmul: kernels::matmul::device::from_module(module.clone())?,
                quantized: kernels::quantized::device::from_module(module.clone())?,
                attention: kernels::attention::device::from_module(module.clone())?,
                attention_bf16: kernels::attention_bf16::device::from_module(module.clone())?,
                cached: kernels::cached::device::from_module(module.clone())?,
                elementwise,
            })
        }
    }
}

pub struct Cuda {
    stream: Arc<CudaStream>,
    kernels: Kernels,
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
    ($backend:ident, $family:ident, $kernel:ident, $prepare:ident, $config:expr, $($argument:expr),+ $(,)?) => {{
        let module = &$backend.kernels.$family;
        let prepared = module.$prepare($config).expect(CUDA_ERROR);
        module
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
        // SAFETY: le PTX chargé est celui que ce paquet a compilé à partir
        // de `crate::kernels` et embarqué dans le binaire.
        let kernels = unsafe { Kernels::load(&context)? };
        Ok(Self {
            pool: Pool::new(stream.clone()),
            index_pool: Pool::new(stream.clone()),
            stream,
            kernels,
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
            elementwise,
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
            elementwise,
            unary,
            prepare_unary,
            launch_config(len),
            op,
            &a.storage.buffer,
            &mut out
        );
        self.wrap(out, a.shape.clone())
    }
}

/// Le découpage en tranches de la dimension `len` le long de laquelle un
/// kernel additionne : `(tranches, longueur d'une tranche)`. Avec `blocks`
/// blocs par tranche, la grille en compte au moins [`NARROW_BLOCKS`] pour
/// occuper la carte, sans tranche de moins de `min_depth` éléments. Chaque
/// tranche est un multiple de `multiple`, sauf la dernière, qui peut être
/// plus courte. Les sommes partielles des tranches sont ensuite réunies.
fn split_depth(len: usize, blocks: usize, min_depth: usize, multiple: usize) -> (usize, usize) {
    let wanted = NARROW_BLOCKS
        .div_ceil(blocks)
        .clamp(1, len.div_ceil(min_depth).max(1));
    let depth = len.div_ceil(wanted).next_multiple_of(multiple);
    (len.div_ceil(depth).max(1), depth)
}

mod attention;
mod elementwise;
mod matmul;
mod primitives;
mod rows;
