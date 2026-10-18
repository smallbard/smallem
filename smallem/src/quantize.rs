//! La quantification des poids : des entiers de 8 ou 4 bits, et une échelle
//! par groupe de poids.
//!
//! Une matrice de poids `W` `[K, N]` est découpée, colonne par colonne, en
//! groupes de `g` poids consécutifs le long de `K`. Chaque groupe reçoit
//! une échelle, la plus grande valeur absolue de ses poids divisée par le
//! plus grand code, et chaque poids devient le code entier le plus proche :
//!
//! ```text
//! échelle = max |w| / 127          (int8 ; 7 en int4)
//! code    = arrondi(w / échelle)   entre -127 et 127
//! w'      = code × échelle         ce que le modèle utilise ensuite
//! ```
//!
//! En int8, un groupe est une colonne entière : une échelle par canal de
//! sortie. En int4, quinze valeurs seulement pour un groupe : il faut des
//! groupes plus petits, de 64 poids, pour que l'échelle suive les poids.
//!
//! **Le rangement.** Les tenseurs du projet sont des `f32`. Un `f32`
//! représente exactement tout entier jusqu'à 2²⁴ : on y range donc 24 bits
//! de codes, trois codes de 8 bits ou six de 4 bits. C'est 1,33 octet par
//! poids en int8 et 0,67 en int4, au lieu de 1 et 0,5 avec un vrai type
//! entier, mais les mêmes tenseurs, les mêmes transferts et la même
//! définition de référence sur tous les backends. Les codes d'un mot sont
//! des poids consécutifs le long de `K`, dans une même colonne ; les mots
//! d'un groupe se suivent, et tous les mots d'un même rang forment une
//! ligne `[N]`, pour que des threads voisins lisent des colonnes voisines.

/// La forme des codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// Codes de 8 bits, une échelle par colonne.
    Int8,
    /// Codes de 4 bits, une échelle par groupe de `group` poids d'une
    /// colonne.
    Int4 { group: usize },
}

impl Format {
    /// Le format int4 habituel : des groupes de 64 poids.
    pub const INT4: Self = Self::Int4 { group: 64 };

    pub fn bits(self) -> u32 {
        match self {
            Self::Int8 => 8,
            Self::Int4 { .. } => 4,
        }
    }

    /// Le plus grand code, en valeur absolue : les codes sont symétriques.
    pub fn max_code(self) -> i32 {
        (1 << (self.bits() - 1)) - 1
    }

    /// Codes par mot de 24 bits.
    pub fn codes_per_word(self) -> usize {
        24 / self.bits() as usize
    }

    /// Taille des groupes pour une matrice de `rows` lignes.
    pub fn group(self, rows: usize) -> usize {
        match self {
            Self::Int8 => rows,
            Self::Int4 { group } => group.min(rows),
        }
    }

    pub fn name(self) -> String {
        match self {
            Self::Int8 => "int8".into(),
            Self::Int4 { group } => format!("int4/{group}"),
        }
    }
}

impl std::str::FromStr for Format {
    type Err = String;

    /// `int8`, `int4` (des groupes de 64) ou `int4/G`, comme les écrit
    /// [`Format::name`].
    fn from_str(name: &str) -> Result<Self, String> {
        match name {
            "int8" => Ok(Self::Int8),
            "int4" => Ok(Self::INT4),
            other => other
                .strip_prefix("int4/")
                .and_then(|group| group.parse().ok())
                .filter(|&group| group > 0)
                .map(|group| Self::Int4 { group })
                .ok_or_else(|| format!("format inconnu : {other} (int8, int4, int4/G)")),
        }
    }
}

/// Une matrice `[rows, cols]` quantifiée.
#[derive(Clone, Debug, PartialEq)]
pub struct Quantized {
    pub format: Format,
    pub rows: usize,
    pub cols: usize,
    /// `[groupes × mots par groupe, cols]` : les codes, 24 bits par `f32`.
    pub words: Vec<f32>,
    /// `[groupes, cols]`.
    pub scales: Vec<f32>,
}

/// Décalage ajouté aux codes pour les ranger sans signe.
fn offset(format: Format) -> i32 {
    1 << (format.bits() - 1)
}

impl Quantized {
    /// Taille des groupes, nombre de groupes, mots par groupe.
    pub fn layout(format: Format, rows: usize) -> (usize, usize, usize) {
        let group = format.group(rows);
        (
            group,
            rows.div_ceil(group),
            group.div_ceil(format.codes_per_word()),
        )
    }

    /// Quantifie `weight`, `[rows, cols]` rangée ligne par ligne, en
    /// arrondissant chaque poids au code le plus proche.
    pub fn round_to_nearest(weight: &[f64], rows: usize, cols: usize, format: Format) -> Self {
        assert_eq!(weight.len(), rows * cols, "poids de forme inattendue");
        let (group, groups, _) = Self::layout(format, rows);
        let max_code = f64::from(format.max_code());
        let mut scales = vec![0.0; groups * cols];
        for g in 0..groups {
            for col in 0..cols {
                let largest = (g * group..((g + 1) * group).min(rows))
                    .map(|k| weight[k * cols + col].abs())
                    .fold(0.0, f64::max);
                scales[g * cols + col] = if largest > 0.0 {
                    largest / max_code
                } else {
                    1.0
                };
            }
        }
        let mut codes = vec![0i32; rows * cols];
        for k in 0..rows {
            for col in 0..cols {
                let scale = scales[(k / group) * cols + col];
                let code = (weight[k * cols + col] / scale).round();
                codes[k * cols + col] = code.clamp(-max_code, max_code) as i32;
            }
        }
        Self::from_codes(&codes, scales, rows, cols, format)
    }

    /// Quantifie `weight` par GPTQ (Frantar et al., 2022) : au lieu
    /// d'arrondir chaque poids indépendamment, on choisit les codes pour
    /// que la sortie de la couche, `x · W`, change le moins possible sur
    /// des entrées typiques.
    ///
    /// `hessian` `[rows, rows]` est `Σ xᵀ · x` sur ces entrées. Les lignes de
    /// `W`, une par composante de `x`, sont quantifiées l'une après l'autre.
    /// L'erreur d'arrondi d'une ligne est aussitôt compensée sur les lignes
    /// pas encore quantifiées, dans la mesure où leurs composantes de `x`
    /// sont corrélées à la sienne. Le facteur de compensation se lit dans
    /// le facteur de Cholesky `U` de l'inverse du hessien, `H⁻¹ = Uᵀ · U` :
    ///
    /// ```text
    /// erreur      = (w_k − q_k) / U[k, k]
    /// w_j        -= erreur × U[k, j]      pour toutes les lignes j > k
    /// ```
    ///
    /// Comme dans l'article, un amortissement de 1 % de la moyenne de la
    /// diagonale est ajouté au hessien, et les échelles d'un groupe sont
    /// fixées quand on atteint sa première ligne, d'après les poids déjà
    /// corrigés.
    pub fn gptq(weight: &[f64], rows: usize, cols: usize, format: Format, hessian: &[f64]) -> Self {
        assert_eq!(weight.len(), rows * cols, "poids de forme inattendue");
        assert_eq!(hessian.len(), rows * rows, "hessien de forme inattendue");
        let mut w = weight.to_vec();
        let mut h = hessian.to_vec();
        // Une composante toujours nulle ne contraint rien : ses poids ne
        // servent pas, on les met à zéro.
        for k in 0..rows {
            if h[k * rows + k] == 0.0 {
                h[k * rows + k] = 1.0;
                w[k * cols..(k + 1) * cols].fill(0.0);
            }
        }
        let mean = (0..rows).map(|k| h[k * rows + k]).sum::<f64>() / rows as f64;
        for k in 0..rows {
            h[k * rows + k] += 0.01 * mean;
        }
        let inverse = invert_symmetric(&h, rows);
        // `H⁻¹ = L · Lᵀ`, donc `U = Lᵀ`, rangé ici par lignes de `L`.
        let lower = cholesky(&inverse, rows);
        let u = |k: usize, j: usize| lower[j * rows + k];

        let (group, groups, _) = Self::layout(format, rows);
        let max_code = f64::from(format.max_code());
        let mut scales = vec![1.0; groups * cols];
        let mut codes = vec![0i32; rows * cols];
        let mut error = vec![0.0; cols];
        for k in 0..rows {
            let g = k / group;
            if k % group == 0 {
                for col in 0..cols {
                    let largest = (k..(k + group).min(rows))
                        .map(|j| w[j * cols + col].abs())
                        .fold(0.0, f64::max);
                    if largest > 0.0 {
                        scales[g * cols + col] = largest / max_code;
                    }
                }
            }
            let diagonal = u(k, k);
            for col in 0..cols {
                let scale = scales[g * cols + col];
                let value = w[k * cols + col];
                let code = (value / scale).round().clamp(-max_code, max_code);
                codes[k * cols + col] = code as i32;
                error[col] = (value - code * scale) / diagonal;
            }
            for j in k + 1..rows {
                let factor = u(k, j);
                if factor != 0.0 {
                    for (weight, error) in w[j * cols..(j + 1) * cols].iter_mut().zip(&error) {
                        *weight -= error * factor;
                    }
                }
            }
        }
        Self::from_codes(&codes, scales, rows, cols, format)
    }

    /// Range des codes déjà choisis, `[rows, cols]`, avec leurs échelles
    /// `[groupes, cols]`.
    pub fn from_codes(
        codes: &[i32],
        scales: Vec<f64>,
        rows: usize,
        cols: usize,
        format: Format,
    ) -> Self {
        let (group, groups, per_group) = Self::layout(format, rows);
        assert_eq!(scales.len(), groups * cols, "échelles de forme inattendue");
        let per_word = format.codes_per_word();
        let mut words = vec![0u32; groups * per_group * cols];
        for k in 0..rows {
            let (g, j) = (k / group, k % group);
            let (word, slot) = (j / per_word, j % per_word);
            for col in 0..cols {
                let code = codes[k * cols + col];
                assert!(
                    code.abs() <= format.max_code(),
                    "code {code} hors du format {}",
                    format.name(),
                );
                let stored = (code + offset(format)) as u32;
                words[(g * per_group + word) * cols + col] |=
                    stored << (format.bits() * slot as u32);
            }
        }
        Self {
            format,
            rows,
            cols,
            words: words.into_iter().map(|word| word as f32).collect(),
            scales: scales.into_iter().map(|scale| scale as f32).collect(),
        }
    }

    /// Le code du poids `(k, col)`.
    pub fn code(&self, k: usize, col: usize) -> i32 {
        let (group, _, per_group) = Self::layout(self.format, self.rows);
        let per_word = self.format.codes_per_word();
        let (g, j) = (k / group, k % group);
        let word = self.words[(g * per_group + j / per_word) * self.cols + col] as u32;
        let mask = (1 << self.format.bits()) - 1;
        ((word >> (self.format.bits() * (j % per_word) as u32)) & mask) as i32 - offset(self.format)
    }

    /// Les poids que le modèle utilise : `code × échelle`, `[rows, cols]`.
    pub fn dequantize(&self) -> Vec<f64> {
        let group = self.format.group(self.rows);
        let mut weight = Vec::with_capacity(self.rows * self.cols);
        for k in 0..self.rows {
            for col in 0..self.cols {
                let scale = f64::from(self.scales[(k / group) * self.cols + col]);
                weight.push(f64::from(self.code(k, col)) * scale);
            }
        }
        weight
    }

    /// Octets qu'occupent les codes et les échelles.
    pub fn bytes(&self) -> usize {
        4 * (self.words.len() + self.scales.len())
    }
}

/// Le facteur de Cholesky `L` d'une matrice symétrique définie positive
/// `[n, n]` : `a = L · Lᵀ`, `L` triangulaire inférieure, rangée par lignes.
pub fn cholesky(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let dot: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
            if i == j {
                let pivot = a[i * n + i] - dot;
                assert!(
                    pivot > 0.0,
                    "matrice non définie positive (pivot {pivot} en {i})"
                );
                l[i * n + i] = pivot.sqrt();
            } else {
                l[i * n + j] = (a[i * n + j] - dot) / l[j * n + j];
            }
        }
    }
    l
}

/// L'inverse d'une matrice symétrique définie positive `[n, n]`, par son
/// facteur de Cholesky : `a⁻¹ = L⁻ᵀ · L⁻¹`.
pub fn invert_symmetric(a: &[f64], n: usize) -> Vec<f64> {
    let l = cholesky(a, n);
    // `L⁻¹`, triangulaire inférieure, colonne par colonne.
    let mut inverse_l = vec![0.0; n * n];
    for col in 0..n {
        inverse_l[col * n + col] = 1.0 / l[col * n + col];
        for i in col + 1..n {
            let dot: f64 = (col..i)
                .map(|k| l[i * n + k] * inverse_l[k * n + col])
                .sum();
            inverse_l[i * n + col] = -dot / l[i * n + i];
        }
    }
    // `L⁻ᵀ · L⁻¹` : l'élément `(i, j)` est la somme sur `k ≥ max(i, j)`.
    let mut inverse = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let dot: f64 = (i..n)
                .map(|k| inverse_l[k * n + i] * inverse_l[k * n + j])
                .sum();
            inverse[i * n + j] = dot;
            inverse[j * n + i] = dot;
        }
    }
    inverse
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::standard_normal;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn weight(rows: usize, cols: usize) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(1);
        (0..rows * cols)
            .map(|_| 0.02 * standard_normal(&mut rng))
            .collect()
    }

    #[test]
    fn every_code_fits_in_an_exact_f32() {
        for format in [Format::Int8, Format::INT4] {
            let q = Quantized::round_to_nearest(&weight(130, 7), 130, 7, format);
            for word in &q.words {
                assert!(*word >= 0.0 && *word < (1 << 24) as f32 && word.fract() == 0.0);
            }
        }
    }

    #[test]
    fn codes_are_read_back_as_written() {
        // Tous les codes possibles, dans tous les rangs d'un mot.
        for format in [Format::Int8, Format::Int4 { group: 10 }] {
            let max = format.max_code();
            let (rows, cols) = (37, 3);
            let codes: Vec<i32> = (0..rows * cols)
                .map(|i| (i as i32 * 7) % (2 * max + 1) - max)
                .collect();
            let (_, groups, _) = Quantized::layout(format, rows);
            let q = Quantized::from_codes(&codes, vec![1.0; groups * cols], rows, cols, format);
            for k in 0..rows {
                for col in 0..cols {
                    assert_eq!(
                        q.code(k, col),
                        codes[k * cols + col],
                        "{format:?} ({k}, {col})"
                    );
                }
            }
        }
    }

    #[test]
    fn rounding_error_is_at_most_half_a_step() {
        let (rows, cols) = (200, 5);
        let w = weight(rows, cols);
        for format in [Format::Int8, Format::INT4] {
            let q = Quantized::round_to_nearest(&w, rows, cols, format);
            let group = format.group(rows);
            for (index, (a, b)) in w.iter().zip(q.dequantize()).enumerate() {
                let scale = f64::from(q.scales[(index / cols / group) * cols + index % cols]);
                assert!((a - b).abs() <= 0.5 * scale * (1.0 + 1e-6), "{format:?}");
            }
            // Le plus grand poids tombe sur un code, aux arrondis près de
            // l'échelle, rangée en `f32`.
            let max = w.iter().map(|v| v.abs()).fold(0.0, f64::max);
            assert!(
                q.dequantize()
                    .iter()
                    .any(|v| (v.abs() - max).abs() < 1e-6 * max)
            );
        }
    }

    #[test]
    fn int4_takes_half_the_space_of_int8() {
        let (rows, cols) = (512, 512);
        let int8 = Quantized::round_to_nearest(&weight(rows, cols), rows, cols, Format::Int8);
        let int4 = Quantized::round_to_nearest(&weight(rows, cols), rows, cols, Format::INT4);
        // 171 mots de trois codes par colonne ; 8 groupes de 11 mots de six.
        assert_eq!(int8.bytes(), 4 * (171 * 512 + 512));
        assert_eq!(int4.bytes(), 4 * (8 * 11 * 512 + 8 * 512));
        assert!(int4.bytes() * 10 < 4 * rows * cols * 2);
    }

    /// `Σ ‖x · (W − Q)‖²` sur les lignes de `x` `[n, rows]`.
    fn output_error(x: &[f64], rows: usize, w: &[f64], q: &Quantized) -> f64 {
        let cols = q.cols;
        let dequantized = q.dequantize();
        x.chunks(rows)
            .map(|x| {
                (0..cols)
                    .map(|c| {
                        let d: f64 = (0..rows)
                            .map(|k| x[k] * (w[k * cols + c] - dequantized[k * cols + c]))
                            .sum();
                        d * d
                    })
                    .sum::<f64>()
            })
            .sum()
    }

    #[test]
    fn cholesky_and_inverse_are_exact() {
        let a = [4.0, 2.0, 0.4, 2.0, 3.0, 0.5, 0.4, 0.5, 2.0];
        let l = cholesky(&a, 3);
        for i in 0..3 {
            for j in 0..3 {
                let product: f64 = (0..3).map(|k| l[i * 3 + k] * l[j * 3 + k]).sum();
                assert!((product - a[i * 3 + j]).abs() < 1e-12);
            }
        }
        let inverse = invert_symmetric(&a, 3);
        for i in 0..3 {
            for j in 0..3 {
                let product: f64 = (0..3).map(|k| a[i * 3 + k] * inverse[k * 3 + j]).sum();
                assert!((product - f64::from(u8::from(i == j))).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn gptq_with_independent_inputs_is_rounding_to_nearest() {
        // Un hessien diagonal : aucune composante ne peut en compenser une
        // autre, et chaque poids est simplement arrondi.
        let (rows, cols) = (40, 6);
        let w = weight(rows, cols);
        let hessian: Vec<f64> = (0..rows * rows)
            .map(|i| {
                if i % (rows + 1) == 0 {
                    1.0 + (i % 7) as f64
                } else {
                    0.0
                }
            })
            .collect();
        for format in [Format::Int8, Format::Int4 { group: 8 }] {
            assert_eq!(
                Quantized::gptq(&w, rows, cols, format, &hessian),
                Quantized::round_to_nearest(&w, rows, cols, format),
            );
        }
    }

    #[test]
    fn gptq_changes_the_output_less_than_rounding() {
        // Des entrées dont les composantes sont corrélées, comme dans un
        // réseau : `x = z · M` avec `z` gaussien.
        let (rows, cols, samples) = (48, 16, 400);
        let mut rng = StdRng::seed_from_u64(9);
        let mixing: Vec<f64> = (0..rows * rows)
            .map(|_| standard_normal(&mut rng))
            .collect();
        let x: Vec<f64> = (0..samples)
            .flat_map(|_| {
                let z: Vec<f64> = (0..rows).map(|_| standard_normal(&mut rng)).collect();
                (0..rows)
                    .map(|k| (0..rows).map(|i| z[i] * mixing[i * rows + k]).sum::<f64>())
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut hessian = vec![0.0; rows * rows];
        for row in x.chunks(rows) {
            for i in 0..rows {
                for j in 0..rows {
                    hessian[i * rows + j] += row[i] * row[j];
                }
            }
        }
        let w = weight(rows, cols);
        let format = Format::Int4 { group: 16 };
        let nearest = output_error(
            &x,
            rows,
            &w,
            &Quantized::round_to_nearest(&w, rows, cols, format),
        );
        let gptq = output_error(
            &x,
            rows,
            &w,
            &Quantized::gptq(&w, rows, cols, format, &hessian),
        );
        assert!(gptq < 0.7 * nearest, "GPTQ {gptq} contre {nearest}");
    }
}
