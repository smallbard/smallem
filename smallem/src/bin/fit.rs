//! Ajuste les lois d'échelle sur les résultats des balayages, et prédit.
//!
//! Usage : `cargo run --release --bin fit -- [options] RÉSULTATS…`
//!
//! ```text
//! RÉSULTATS          fichiers `results.tsv` écrits par `sweep`
//! --non-embedding    compter les paramètres hors table d'embedding
//! --predict C        prédit la taille, les tokens et la perte optimaux pour un
//!                    calcul C (plusieurs fois possible)
//! --check NOM        ne pas ajuster sur l'entraînement NOM, et comparer sa perte
//!                    à la prédiction (plusieurs fois possible)
//! --bootstrap N      tirages pour les intervalles (défaut : 200)
//! ```
//!
//! Les budgets des courbes à calcul constant se lisent dans les noms
//! d'entraînement de la forme `c3e15-e256` ; les autres entraînements ne
//! servent qu'à la formule de la perte.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;

use rand::SeedableRng;
use rand::rngs::StdRng;
use smallem::cli::{self, Args};
use smallem::scaling::{
    LossLaw, Run, bootstrap, fit_loss_law, fit_parabola, fit_power_law, quantile,
};

/// Tokens par pas : 16 séquences de 512.
const TOKENS_PER_STEP: f64 = 8192.0;
/// Vocabulaire, pour retirer la table d'embedding du compte.
const VOCAB: f64 = 8192.0;

struct Row {
    name: String,
    parameters: f64,
    flops: f64,
    tokens: f64,
    loss: f64,
}

/// Les entraînements d'un fichier `results.tsv`. Une ligne illisible est
/// une erreur ; un entraînement raté (perte `NaN`) est écarté et signalé.
fn read(path: &str) -> Result<Vec<Row>, Box<dyn Error>> {
    let mut rows = Vec::new();
    let mut failed = Vec::new();
    // Après l'en-tête, la première ligne est la ligne 2.
    for (line_number, line) in (2..).zip(fs::read_to_string(path)?.lines().skip(1)) {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        let parsed = match fields[..] {
            [name, parameters, flops, steps, loss, ..] => (|| {
                Some((
                    name,
                    parameters.parse::<f64>().ok()?,
                    flops.parse::<f64>().ok()?,
                    steps.parse::<f64>().ok()?,
                    loss.parse::<f64>().ok()?,
                ))
            })(),
            _ => None,
        };
        let Some((name, parameters, flops, steps, loss)) = parsed else {
            return Err(format!("{path}, ligne {line_number} illisible : `{line}`").into());
        };
        if !loss.is_finite() {
            failed.push(name.to_string());
            continue;
        }
        rows.push(Row {
            name: name.to_string(),
            parameters,
            flops,
            tokens: steps * TOKENS_PER_STEP,
            loss,
        });
    }
    if !failed.is_empty() {
        eprintln!(
            "{path} : {} entraînements ratés écartés ({})",
            failed.len(),
            failed.join(", ")
        );
    }
    Ok(rows)
}

/// La largeur d'un entraînement, d'après son nombre de paramètres : la
/// plus proche de la famille.
fn width(parameters: f64) -> f64 {
    let family = [
        128.0, 192.0, 256.0, 320.0, 384.0, 448.0, 512.0, 576.0, 640.0, 768.0,
    ];
    family
        .iter()
        .cloned()
        .min_by(|a, b| {
            let error = |w: f64| (total_parameters(w) - parameters).abs();
            error(*a).total_cmp(&error(*b))
        })
        .expect("une largeur")
}

/// Les paramètres d'un modèle de la famille de largeur `c`.
fn total_parameters(c: f64) -> f64 {
    let layers = c / 64.0;
    let hidden = ((8.0 * c / 3.0).ceil() / 8.0).ceil() * 8.0;
    let block = c + 4.0 * (c * c + c) + c + 2.0 * (c * hidden + hidden) + hidden * c + c;
    layers * block + c + VOCAB * c
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut files = Vec::new();
    let (mut non_embedding, mut rounds) = (false, 200usize);
    let (mut predictions, mut checks) = (Vec::new(), Vec::new());
    let mut args = Args::from_env();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--non-embedding" => non_embedding = true,
            "--predict" => predictions.push(args.value::<f64>(&arg)?),
            "--check" => checks.push(args.value::<String>(&arg)?),
            "--bootstrap" => rounds = args.value(&arg)?,
            file => files.push(cli::positional(file)?.to_string()),
        }
    }
    if files.is_empty() {
        return Err("aucun fichier de résultats : fit [options] RÉSULTATS…".into());
    }
    let mut rows = Vec::new();
    for file in &files {
        rows.extend(read(file)?);
    }
    let count = |row: &Row| {
        if non_embedding {
            row.parameters - VOCAB * width(row.parameters)
        } else {
            row.parameters
        }
    };
    println!(
        "{} entraînements, paramètres comptés {}",
        rows.len(),
        if non_embedding {
            "hors embedding"
        } else {
            "en entier"
        }
    );

    // Les courbes à calcul constant.
    let mut budgets: BTreeMap<String, Vec<&Row>> = BTreeMap::new();
    for row in &rows {
        if checks.contains(&row.name) {
            continue;
        }
        if let Some(budget) = row.name.strip_prefix('c').and_then(|n| n.split('-').next()) {
            budgets.entry(budget.to_string()).or_default().push(row);
        }
    }
    let mut optima = Vec::new();
    for (budget, runs) in &budgets {
        if runs.len() < 3 {
            continue;
        }
        let points: Vec<(f64, f64)> = runs.iter().map(|r| (count(r).ln(), r.loss)).collect();
        let (a, b, c) = fit_parabola(&points);
        let best_log = -b / (2.0 * a);
        let n_opt = best_log.exp();
        let loss = a * best_log * best_log + b * best_log + c;
        let compute: f64 = runs.iter().map(|r| r.flops).sum::<f64>() / runs.len() as f64;
        // Les tokens de l'optimum : le calcul divisé par les opérations
        // par token de cette taille, interpolées entre les entraînements.
        let per_token: Vec<(f64, f64)> = runs
            .iter()
            .map(|r| (count(r).ln(), (r.flops / r.tokens).ln()))
            .collect();
        let (slope, intercept) = smallem::scaling::fit_line(&per_token);
        let d_opt = compute / (intercept + slope * best_log).exp();
        println!(
            "budget {budget} ({compute:.3e}) : N_opt = {:.2} M, D_opt = {:.0} M tokens, \
             {:.1} tokens par paramètre, perte {loss:.4}{}",
            n_opt / 1e6,
            d_opt / 1e6,
            d_opt / n_opt,
            if a <= 0.0 {
                "  (pas de minimum : parabole retournée)"
            } else {
                ""
            },
        );
        for run in runs {
            println!(
                "    {:<16} N = {:>7.2} M  D = {:>6.0} M  perte {:.4}",
                run.name,
                count(run) / 1e6,
                run.tokens / 1e6,
                run.loss
            );
        }
        if a > 0.0 {
            optima.push((compute, n_opt, d_opt, loss));
        }
    }
    if optima.len() >= 2 {
        let (a, ka) = fit_power_law(&optima.iter().map(|o| (o.0, o.1)).collect::<Vec<_>>());
        let (b, kb) = fit_power_law(&optima.iter().map(|o| (o.0, o.2)).collect::<Vec<_>>());
        let (l, kl) = fit_power_law(&optima.iter().map(|o| (o.0, o.3)).collect::<Vec<_>>());
        println!(
            "N_opt = {ka:.3e} × C^{a:.3}   D_opt = {kb:.3e} × C^{b:.3}   L_opt = {kl:.3} × C^{l:.4}"
        );
        for &compute in &predictions {
            let n = ka * compute.powf(a);
            let d = kb * compute.powf(b);
            println!(
                "  prédiction des courbes pour {compute:.3e} : N = {:.2} M, D = {:.0} M, perte {:.4}",
                n / 1e6,
                d / 1e6,
                kl * compute.powf(l)
            );
        }
    }

    // La formule de la perte, sur tous les entraînements hors contrôle.
    let fitted: Vec<Run> = rows
        .iter()
        .filter(|r| !checks.contains(&r.name))
        .map(|r| Run {
            parameters: count(r),
            tokens: r.tokens,
            loss: r.loss,
        })
        .collect();
    let law: LossLaw = fit_loss_law(&fitted);
    println!(
        "L(N, D) = {:.4} + {:.4e} / N^{:.4} + {:.4e} / D^{:.4}",
        law.e, law.a, law.alpha, law.b, law.beta
    );
    let residuals: Vec<f64> = fitted
        .iter()
        .map(|r| (law.predict(r.parameters, r.tokens) - r.loss).abs())
        .collect();
    println!(
        "écart moyen aux points ajustés : {:.4} nat par octet, au plus {:.4}",
        residuals.iter().sum::<f64>() / residuals.len() as f64,
        residuals.iter().cloned().fold(0.0, f64::max)
    );
    let mut rng = StdRng::seed_from_u64(0);
    let laws = bootstrap(&fitted, rounds, &mut rng);
    let interval = |values: Vec<f64>| {
        let mut values = values;
        (quantile(&mut values, 0.05), quantile(&mut values, 0.95))
    };
    for (name, values) in [
        ("E", laws.iter().map(|l| l.e).collect::<Vec<_>>()),
        ("α", laws.iter().map(|l| l.alpha).collect()),
        ("β", laws.iter().map(|l| l.beta).collect()),
    ] {
        let (low, high) = interval(values);
        println!("  {name} : intervalle à 90 % [{low:.4}, {high:.4}]");
    }
    // Le rapport du calcul aux paramètres et aux tokens, pour l'optimum de
    // la formule : `C = k × N × D`.
    let k = {
        let ratios: Vec<f64> = rows
            .iter()
            .map(|r| r.flops / (count(r) * r.tokens))
            .collect();
        ratios.iter().sum::<f64>() / ratios.len() as f64
    };
    for &compute in &predictions {
        let (n, d) = law.optimum(compute, k);
        let losses: Vec<f64> = laws
            .iter()
            .map(|l| {
                let (n, d) = l.optimum(compute, k);
                l.predict(n, d)
            })
            .collect();
        let (low, high) = interval(losses);
        println!(
            "  prédiction de la formule pour {compute:.3e} : N = {:.2} M, D = {:.0} M, perte {:.4} \
             [{low:.4}, {high:.4}]",
            n / 1e6,
            d / 1e6,
            law.predict(n, d)
        );
    }
    for name in &checks {
        let Some(row) = rows.iter().find(|r| &r.name == name) else {
            println!("  contrôle {name} : introuvable");
            continue;
        };
        let predicted = law.predict(count(row), row.tokens);
        let losses: Vec<f64> = laws
            .iter()
            .map(|l| l.predict(count(row), row.tokens))
            .collect();
        let (low, high) = interval(losses);
        println!(
            "  contrôle {name} : mesuré {:.4}, prédit {predicted:.4} [{low:.4}, {high:.4}], écart {:+.4}",
            row.loss,
            row.loss - predicted
        );
    }
    Ok(())
}
