//! Arithmétique des formes, commune au CPU et au GPU : pas, broadcasting et
//! conversion d'indices. Rien ici ne touche aux données.

/// Pas de chaque dimension dans le stockage à plat : de combien avance
/// l'indice à plat quand la coordonnée augmente de 1.
/// Pour la forme `[2, 3, 4]`, les pas sont `[12, 4, 1]`.
pub fn strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for dim in (1..shape.len()).rev() {
        strides[dim - 1] = strides[dim] * shape[dim];
    }
    strides
}

/// Forme commune de deux tenseurs selon les règles de NumPy : on aligne les
/// formes par la droite ; deux dimensions sont compatibles si elles sont
/// égales ou si l'une vaut 1, auquel cas elle s'étire.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
    let ndim = a.len().max(b.len());
    let mut shape = vec![0; ndim];
    for from_right in 0..ndim {
        // Une dimension absente à gauche compte pour 1.
        let dim_a = a.len().checked_sub(from_right + 1).map_or(1, |i| a[i]);
        let dim_b = b.len().checked_sub(from_right + 1).map_or(1, |i| b[i]);
        shape[ndim - 1 - from_right] = match (dim_a, dim_b) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => panic!("formes incompatibles pour le broadcasting : {a:?} et {b:?}"),
        };
    }
    shape
}

/// Pas d'un tenseur de forme `shape` vu à travers la forme étirée `target`.
/// Une dimension étirée (ou absente) reçoit un pas nul : quelle que soit la
/// coordonnée demandée, on relit le même élément.
pub fn broadcast_strides(shape: &[usize], target: &[usize]) -> Vec<usize> {
    let own = strides(shape);
    let offset = target.len() - shape.len();
    let mut result = vec![0; target.len()];
    for dim in 0..shape.len() {
        if shape[dim] != 1 {
            result[offset + dim] = own[dim];
        }
    }
    result
}

/// Convertit un indice à plat dans la forme `shape` en coordonnées, puis ces
/// coordonnées en indice à plat dans un stockage décrit par `strides`.
pub fn source_index(mut flat: usize, shape: &[usize], strides: &[usize]) -> usize {
    let mut index = 0;
    for dim in (0..shape.len()).rev() {
        index += (flat % shape[dim]) * strides[dim];
        flat /= shape[dim];
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_are_row_major() {
        assert_eq!(strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(strides(&[5]), vec![1]);
        assert_eq!(strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn broadcast_shapes_follow_numpy_rules() {
        assert_eq!(broadcast_shapes(&[2, 3], &[2, 3]), vec![2, 3]);
        assert_eq!(broadcast_shapes(&[2, 3], &[3]), vec![2, 3]);
        assert_eq!(broadcast_shapes(&[2, 1], &[1, 3]), vec![2, 3]);
        assert_eq!(broadcast_shapes(&[4, 1, 3], &[5, 1]), vec![4, 5, 3]);
        assert_eq!(broadcast_shapes(&[2, 3], &[]), vec![2, 3]);
    }

    #[test]
    #[should_panic(expected = "formes incompatibles pour le broadcasting : [2, 3] et [2]")]
    fn broadcast_rejects_mismatch() {
        broadcast_shapes(&[2, 3], &[2]);
    }
}
