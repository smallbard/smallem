//! Les primitives de `smallem::backend` sur la carte.

use super::*;

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
        let mut out = self.alloc(len);
        launch!(
            self,
            elementwise,
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
                rows,
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
                rows,
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
                rows,
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
            elementwise,
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
        check_expand(&a.shape, shape);
        let len = shape.iter().product();
        let mut out = self.alloc(len);
        launch!(
            self,
            elementwise,
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
        check_mask(&a.shape, &mask.shape);
        self.binary(BinaryOp::MaskedFill, value, a, mask)
    }

    fn gather_rows(&self, a: &CudaTensor, indices: &[u32]) -> CudaTensor {
        // Les indices sont vérifiés ici plutôt que dans le kernel : sur
        // l'hôte, l'erreur peut nommer l'indice fautif.
        let row_len = gather_rows_dims(&a.shape, indices);
        let len = indices.len() * row_len;
        let mut out = self.alloc(len);
        let device_indices = self.upload_indices(indices);

        launch!(
            self,
            elementwise,
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
        let row_len = scatter_add_rows_dims(&a.shape, indices, num_rows);
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
                elementwise,
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
        let shape = concat_rows_shape(&a.shape, &b.shape);
        let len = a.storage.buffer.len() + b.storage.buffer.len();
        let mut out = self.alloc(len);
        launch!(
            self,
            elementwise,
            concat,
            prepare_concat,
            launch_config(len),
            &a.storage.buffer,
            &b.storage.buffer,
            &mut out
        );
        self.wrap(out, shape)
    }

    fn write_rows(&self, table: &mut CudaTensor, indices: &CudaTensor, rows: &CudaTensor) {
        assert_eq!(
            indices.shape.len(),
            1,
            "write_rows : indices de forme {:?}",
            indices.shape
        );
        let row_len = write_rows_dims(&table.shape, indices.shape[0], &rows.shape);
        let table_rows = table.shape[0] as u32;
        let len = rows.storage.buffer.len();
        let table_ptr = unique_buffer(table).cu_deviceptr() as *mut f32;
        // SAFETY: la table, `table_rows × row_len` éléments, n'est partagée
        // avec aucun autre tenseur (`unique_buffer`). Le kernel ignore les
        // indices hors de la table.
        unsafe {
            launch!(
                self,
                elementwise,
                write_rows,
                prepare_write_rows,
                launch_config(len),
                table_ptr,
                table_rows,
                &indices.storage.buffer,
                &rows.storage.buffer,
                row_len as u32
            );
        }
    }

    fn sgd_step(&self, param: &mut CudaTensor, grad: &CudaTensor, learning_rate: f64) {
        check_step(&param.shape, &grad.shape, &[]);
        let len = grad.storage.buffer.len();
        launch!(
            self,
            elementwise,
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
        check_step(
            &param.shape,
            &grad.shape,
            &[&first_moment.shape, &second_moment.shape],
        );
        let params = step.scalars::<f32>();
        let len = grad.storage.buffer.len();
        launch!(
            self,
            elementwise,
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
