//! La réserve de tampons.
//!
//! Chaque opération rend un nouveau tenseur. Un gros tampon neuf vient du
//! système, page par page : à la première écriture, chaque page de 4 Ko
//! coûte une interruption et sa mise à zéro. Un pas d'entraînement du modèle
//! `small` en demandait ainsi 34 000, pour un tiers de son temps (étape C1 de
//! `PLAN-CPU.md`).
//!
//! Comme sur le GPU, un tampon abandonné retourne dans une réserve, rangée
//! par taille exacte, et la prochaine demande de la même taille le reprend.
//! Les formes d'un pas d'entraînement se répètent d'un pas à l'autre : après
//! le premier, presque tout vient de la réserve.

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::Mutex;

/// En dessous, l'allocateur ordinaire est déjà rapide : ses petits blocs
/// restent dans le tas du processus.
const SMALLEST: usize = 1 << 14;

/// La réserve ne garde pas plus que cela, en octets.
const LIMIT: usize = 8 << 30;

struct Reserve {
    free: HashMap<usize, Vec<Vec<f32>>>,
    bytes: usize,
}

static RESERVE: Mutex<Option<Reserve>> = Mutex::new(None);

/// Un tampon de `f32`, qui retourne dans la réserve quand on l'abandonne.
#[derive(Debug, Default)]
pub struct Buffer(Vec<f32>);

impl Buffer {
    /// Un tampon de `len` éléments, au contenu quelconque : repris de la
    /// réserve, il garde ce qu'il contenait ; neuf, il vaut zéro. Le
    /// système donne déjà des pages à zéro (`calloc`), la mise à zéro ne
    /// coûte donc rien de plus que la première écriture.
    pub fn any(len: usize) -> Self {
        if len >= SMALLEST {
            let mut reserve = RESERVE.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(reserve) = reserve.as_mut()
                && let Some(buffer) = reserve.free.get_mut(&len).and_then(Vec::pop)
            {
                reserve.bytes -= 4 * len;
                return Self(buffer);
            }
        }
        Self(vec![0.0; len])
    }

    pub fn into_vec(mut self) -> Vec<f32> {
        std::mem::take(&mut self.0)
    }

    /// Octets que garde la réserve.
    pub fn reserved() -> usize {
        RESERVE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map_or(0, |reserve| reserve.bytes)
    }
}

impl From<Vec<f32>> for Buffer {
    fn from(data: Vec<f32>) -> Self {
        Self(data)
    }
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        let mut copy = Self::any(self.0.len());
        copy.0.copy_from_slice(&self.0);
        copy
    }
}

impl Deref for Buffer {
    type Target = Vec<f32>;

    fn deref(&self) -> &Vec<f32> {
        &self.0
    }
}

impl DerefMut for Buffer {
    fn deref_mut(&mut self) -> &mut Vec<f32> {
        &mut self.0
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let len = self.0.len();
        if len < SMALLEST || len != self.0.capacity() {
            return;
        }
        let mut reserve = RESERVE.lock().unwrap_or_else(|e| e.into_inner());
        let reserve = reserve.get_or_insert_with(|| Reserve {
            free: HashMap::new(),
            bytes: 0,
        });
        if reserve.bytes + 4 * len <= LIMIT {
            reserve.bytes += 4 * len;
            reserve
                .free
                .entry(len)
                .or_default()
                .push(std::mem::take(&mut self.0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_abandoned_buffer_is_taken_again() {
        // Une taille que les autres tests n'utilisent pas.
        let len = SMALLEST + 12_345;
        let first = Buffer::any(len);
        let address = first.as_ptr();
        drop(first);
        let second = Buffer::any(len);
        assert_eq!(second.as_ptr(), address);
        assert_eq!(second.len(), len);
    }
}
