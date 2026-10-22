//! Le tenseur du backend.

use std::sync::Arc;

use super::Buffer;

/// Un tenseur du processeur : des données partagées, et une forme.
///
/// Les données sont derrière un compteur de références. Cloner un tenseur ou
/// en changer la forme ne recopie rien ; une mise à jour sur place ne recopie
/// que si les données sont aussi référencées ailleurs.
#[derive(Clone, Debug)]
pub struct FastTensor {
    pub(super) data: Arc<Buffer>,
    pub(super) shape: Vec<usize>,
}

impl FastTensor {
    pub(super) fn new(data: impl Into<Buffer>, shape: &[usize]) -> Self {
        let data = data.into();
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "{} éléments fournis pour la forme {shape:?}",
            data.len(),
        );
        Self {
            data: Arc::new(data),
            shape: shape.to_vec(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub(super) fn len(&self) -> usize {
        self.data.len()
    }

    /// Les données, modifiables : recopiées seulement si elles sont
    /// partagées.
    pub(super) fn data_mut(&mut self) -> &mut [f32] {
        Arc::make_mut(&mut self.data).as_mut_slice()
    }
}
