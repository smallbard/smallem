//! Arithmétique des formes, commune aux backends : pas, broadcasting,
//! conversion d'indices, et les contrôles de forme des primitives et des
//! opérations, qui rendent les dimensions dont le calcul a besoin. Rien ici
//! ne touche aux données.

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

// --- Contrôles de forme ---
//
// Les mêmes messages pour tous les backends : une erreur de forme est une
// erreur de programmation, et doit se lire pareil sur CPU et sur GPU.

/// La longueur d'une ligne de `shape` le long de la première dimension :
/// le produit des autres.
fn row_len(shape: &[usize]) -> usize {
    shape[1..].iter().product()
}

/// Vérifie qu'un indice désigne une des `rows` lignes d'un tenseur.
pub fn check_row_index(index: usize, rows: usize) {
    assert!(
        index < rows,
        "indice {index} hors de la première dimension ({rows})"
    );
}

/// Vérifie `gather_rows` et rend la longueur d'une ligne.
pub fn gather_rows_dims(shape: &[usize], indices: &[u32]) -> usize {
    assert!(
        !shape.is_empty(),
        "gather_rows demande au moins une dimension"
    );
    for &index in indices {
        check_row_index(index as usize, shape[0]);
    }
    row_len(shape)
}

/// Vérifie `scatter_add_rows` et rend la longueur d'une ligne.
pub fn scatter_add_rows_dims(shape: &[usize], indices: &[u32], num_rows: usize) -> usize {
    assert!(
        !shape.is_empty() && shape[0] == indices.len(),
        "scatter_add_rows : {} indices pour la forme {shape:?}",
        indices.len(),
    );
    for &index in indices {
        check_row_index(index as usize, num_rows);
    }
    row_len(shape)
}

/// Vérifie `concat_rows` et rend la forme du résultat.
pub fn concat_rows_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    assert!(
        !a.is_empty() && !b.is_empty() && a[1..] == b[1..],
        "concat_rows : formes incompatibles {a:?} et {b:?}",
    );
    let mut shape = a.to_vec();
    shape[0] += b[0];
    shape
}

/// Vérifie `write_rows` (`indices` éléments d'indices) et rend la longueur
/// d'une ligne. Les indices eux-mêmes se vérifient avec
/// [`check_row_index`], là où ils sont lus.
pub fn write_rows_dims(table: &[usize], indices: usize, rows: &[usize]) -> usize {
    assert!(
        !table.is_empty() && !rows.is_empty(),
        "write_rows demande au moins une dimension",
    );
    assert_eq!(
        table[1..],
        rows[1..],
        "write_rows : lignes {rows:?} pour une table {table:?}",
    );
    assert_eq!(
        indices, rows[0],
        "write_rows : {indices} indices pour {} lignes",
        rows[0],
    );
    row_len(table)
}

/// Vérifie que le masque de `masked_fill` s'étire à la forme du tenseur.
pub fn check_mask(shape: &[usize], mask: &[usize]) {
    assert_eq!(
        broadcast_shapes(shape, mask),
        shape,
        "le masque {mask:?} ne s'étire pas à la forme {shape:?}",
    );
}

/// Vérifie que `shape` s'étire à `target`, pour `expand`.
pub fn check_expand(shape: &[usize], target: &[usize]) {
    assert_eq!(
        broadcast_shapes(shape, target),
        target,
        "la forme {shape:?} ne s'étire pas à {target:?}",
    );
}

/// Vérifie les formes de l'entropie croisée, des logits `[N, V]` et `N`
/// cibles du vocabulaire, et rend `(N, V)`.
pub fn cross_entropy_dims(logits: &[usize], targets: &[u32]) -> (usize, usize) {
    assert!(
        logits.len() == 2 && logits[0] == targets.len(),
        "cross_entropy : logits {logits:?} pour {} cibles, [N, V] attendu",
        targets.len(),
    );
    let vocab = logits[1];
    for &target in targets {
        assert!(
            (target as usize) < vocab,
            "cible {target} hors du vocabulaire ({vocab})"
        );
    }
    (logits[0], vocab)
}

/// Vérifie qu'un pas d'optimiseur reçoit un gradient, et des moments, de la
/// forme du paramètre.
pub fn check_step(param: &[usize], grad: &[usize], moments: &[&[usize]]) {
    assert_eq!(param, grad, "gradient de forme inattendue");
    for moment in moments {
        assert_eq!(param, *moment, "moment de forme inattendue");
    }
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

    #[test]
    fn row_checks_give_the_row_length() {
        assert_eq!(gather_rows_dims(&[5, 2, 3], &[0, 4]), 6);
        assert_eq!(scatter_add_rows_dims(&[2, 3], &[1, 1], 4), 3);
        assert_eq!(concat_rows_shape(&[2, 3], &[4, 3]), vec![6, 3]);
        assert_eq!(write_rows_dims(&[8, 3], 2, &[2, 3]), 3);
        assert_eq!(cross_entropy_dims(&[2, 5], &[0, 4]), (2, 5));
    }

    #[test]
    #[should_panic(expected = "indice 5 hors de la première dimension (5)")]
    fn gather_rows_dims_refuses_a_missing_row() {
        gather_rows_dims(&[5, 2], &[5]);
    }

    #[test]
    #[should_panic(expected = "concat_rows : formes incompatibles")]
    fn concat_rows_shape_refuses_an_empty_shape() {
        concat_rows_shape(&[2, 3], &[]);
    }
}
