#![feature(assert_matches)]

use std::{
    cmp::Reverse, fs::File, io::Write, num::ParseIntError, path::PathBuf,
    process::Command, time::Duration,
};

use clap::Parser;
use rand::prelude::*;
use swift_swallow_tree_search::{
    DefaultRng,
    game::{Color, Game, Move},
    mcts::Mcts,
};

use crate::colors::{PALETTE_SPECTRAL, interpolate_gradient};

/// Plots a flamegraph of the MCTS search tree after searching from the root.
#[derive(Debug, Parser)]
pub struct Args {
    /// Path to the output HTML file.
    #[clap(default_value = "/tmp/flamegraph.html")]
    pub destination: PathBuf,
    /// Seed for the game and search.
    #[clap(long, default_value_t = 0)]
    pub seed: u64,
    /// How much time to spend per move, in milliseconds.
    #[clap(long, value_parser = parse_millis, required_unless_present_any = ["iters", "nodes"])]
    pub time: Option<Duration>,
    /// How many iterations to search per move.
    #[clap(long, required_unless_present_any = ["time", "nodes"])]
    pub iters: Option<u32>,
    /// How many nodes to search per move.
    #[clap(long, required_unless_present_any = ["time", "iters"])]
    pub nodes: Option<usize>,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let game = Game::new(args.seed, false)?;

    let mut rng = DefaultRng::seed_from_u64(args.seed);
    let mut mcts = Mcts::new(Move::None { team: Color::Black });

    let _next = mcts.search(0, &game, &mut rng, args.time, args.iters, args.nodes);

    let pruning_threshold = {
        let mut visits: Vec<u32> = mcts.tree.iter().map(|node| node.visits).collect();
        visits.sort_unstable();

        let top_k = visits.len().saturating_sub(100_000);
        visits[top_k]
    };

    let mut file = File::create(&args.destination)?;
    tree_to_html(&mcts, pruning_threshold, &args, &mut file)?;

    let _ = Command::new("open").arg(&args.destination).spawn()?;

    Ok(())
}

fn parse_millis(arg: &str) -> Result<Duration, ParseIntError> {
    let millis = arg.parse()?;
    Ok(Duration::from_millis(millis))
}

fn sigmoid(x: f64) -> f64 {
    1. / (1. + f64::exp(-x))
}

// region: Plotting.

fn tree_to_html(
    mcts: &Mcts,
    pruning_threshold: u32,
    args: &Args,
    w: &mut impl Write,
) -> anyhow::Result<()> {
    writeln!(w, r"<!doctype html>")?;
    writeln!(w, r#"<html lang="en">"#)?;

    writeln!(w, r"  <head>")?;
    writeln!(w, r#"    <meta charset="utf-8" />"#)?;
    writeln!(
        w,
        r#"    <meta name="viewport" content="width=device-width,initial-scale=1" />"#
    )?;
    writeln!(w, r"    <title>Flamegraph</title>")?;
    writeln!(
        w,
        r#"    <style type="text/css">{css}</style>"#,
        css = include_str!("flamegraph.css")
    )?;
    writeln!(
        w,
        r#"    <script type="module">{js}</script>"#,
        js = include_str!("flamegraph.js")
    )?;
    writeln!(w, r"  </head>")?;

    writeln!(w, r"  <body>")?;

    writeln!(w, r"  <details closed>")?;
    writeln!(w, r"    <summary>Arguments</summary>")?;
    writeln!(w, r"    <pre><samp>{args:#?}</samp></pre>")?;
    writeln!(w, r"  </details>")?;

    node_to_html(0, mcts, pruning_threshold, true, w)?;

    writeln!(w, r"  </body>")?;

    writeln!(w, r"</html>")?;

    Ok(())
}

fn node_to_html(
    node: usize,
    mcts: &Mcts,
    pruning_threshold: u32,
    is_pv: bool,
    w: &mut impl Write,
) -> anyhow::Result<()> {
    let node = &mcts.tree[node];
    if !is_pv && node.visits <= pruning_threshold {
        return Ok(());
    }

    let mut children: Vec<usize> = (node.head..node.last).collect();
    children.sort_by_key(|&idx| Reverse(mcts.tree[idx].visits));

    let mov = mcts.moves.lookup(node.mov);
    let label = label(mov);

    let q_value = node.value;
    let predicted_win_rate = sigmoid(q_value);

    let parent_visit_share = if node.is_root() {
        100.
    } else {
        let parent = &mcts.tree[node.parent];
        100. * f64::from(node.visits) / f64::from(parent.visits)
    };

    let title = format!(
        r"{label}&#10;&#10;Visits: {visits}&#10;Share of parent's visits: {parent_visit_share:.3}%&#10;Mean value (Q): {q_value:.3}&#10;Predicted win rate: {win_rate:.2}%",
        visits = node.visits,
        win_rate = 100. * predicted_win_rate,
    );

    let color = interpolate_gradient(predicted_win_rate, &PALETTE_SPECTRAL);

    writeln!(w, r#"<div class="node" style="--width: {parent_visit_share:.3}%">"#)?;
    writeln!(
        w,
        r#"  <button class="bar" style="--color: oklab({l} {a} {b})" title="{title}">"#,
        l = color.l,
        a = color.a,
        b = color.b,
    )?;
    writeln!(w, r#"    <span class="label">{label}</span>"#)?;
    writeln!(w, r"  </button>")?;

    writeln!(w, r#"  <div class="children">"#)?;

    for (idx, child) in children.into_iter().enumerate() {
        node_to_html(child, mcts, pruning_threshold, is_pv && idx == 0, w)?;
    }

    writeln!(w, r"  </div>")?;
    writeln!(w, r"</div>")?;

    Ok(())
}

// endregion
// region: Move labeling.

const CHARACTER: [&str; 8] = [
    "Warden ⚪",
    "Warden ⚫",
    "Berserker ⚪",
    "Berserker ⚫",
    "Scholar ⚪",
    "Scholar ⚫",
    "Sidewinder ⚪",
    "Sidewinder ⚫",
];
const ACTION: [&str; 56] = [
    "Strike (Warden) ⚪",
    "Strike (Warden) ⚫",
    "Defend (Warden) ⚪",
    "Defend (Warden) ⚫",
    "Dash (Warden) ⚪",
    "Dash (Warden) ⚫",
    "Provoke ⚪",
    "Provoke ⚫",
    "Barricade ⚪",
    "Barricade ⚫",
    "Impervious ⚪",
    "Impervious ⚫",
    "Juggernaut ⚪",
    "Juggernaut ⚫",
    "Strike (Berserker) ⚪",
    "Strike (Berserker) ⚫",
    "Defend (Berserker) ⚪",
    "Defend (Berserker) ⚫",
    "Dash (Berserker) ⚪",
    "Dash (Berserker) ⚫",
    "Onslaught ⚪",
    "Onslaught ⚫",
    "Upheaval ⚪",
    "Upheaval ⚫",
    "Grit ⚪",
    "Grit ⚫",
    "Overpower ⚪",
    "Overpower ⚫",
    "Strike (Scholar) ⚪",
    "Strike (Scholar) ⚫",
    "Defend (Scholar) ⚪",
    "Defend (Scholar) ⚫",
    "Dash (Scholar) ⚪",
    "Dash (Scholar) ⚫",
    "Adloquium ⚪",
    "Adloquium ⚫",
    "Entrench ⚪",
    "Entrench ⚫",
    "Ruin ⚪",
    "Ruin ⚫",
    "Malaise ⚪",
    "Malaise ⚫",
    "Strike (Sidewinder) ⚪",
    "Strike (Sidewinder) ⚫",
    "Defend (Sidewinder) ⚪",
    "Defend (Sidewinder) ⚫",
    "Dash (Sidewinder) ⚪",
    "Dash (Sidewinder) ⚫",
    "Adrenaline ⚪",
    "Adrenaline ⚫",
    "Envenom ⚪",
    "Envenom ⚫",
    "Catalyst ⚪",
    "Catalyst ⚫",
    "Bane ⚪",
    "Bane ⚫",
];

fn label(mov: &Move) -> String {
    match mov {
        Move::None { team } => match team {
            Color::White => "⚪".to_owned(),
            Color::Black => "⚫".to_owned(),
        },
        Move::Pass { character } => {
            format!("⌛ {label} → Pass", label = CHARACTER[character.0])
        }
        Move::Move { character, destination } => {
            format!(
                "🧭 {label} → ⟨{x}, {y}⟩",
                label = CHARACTER[character.0],
                x = destination.q,
                y = destination.r,
            )
        }
        Move::Act { action, target_hint: Some(target), .. } => {
            format!(
                "🎯 {label} → {target}",
                label = ACTION[action.0],
                target = CHARACTER[target.0],
            )
        }
        Move::Act { action, destination, .. } => {
            format!(
                "🎯 {label} → ⟨{x}, {y}⟩",
                label = ACTION[action.0],
                x = destination.q,
                y = destination.r,
            )
        }
    }
}

// endregion

mod colors {
    use std::{assert_matches, cmp::min};

    #[derive(Copy, Clone, Debug, PartialEq)]
    pub struct Oklab {
        pub l: f64,
        pub a: f64,
        pub b: f64,
    }

    /// From <https://colorbrewer2.org/#type=diverging&scheme=Spectral&n=11>.
    pub const PALETTE_SPECTRAL: [Oklab; 11] = [
        Oklab { l: 0.484, a: 0.042, b: -0.122 }, // #5e4fa2
        Oklab { l: 0.599, a: -0.057, b: -0.099 }, // #3288bd
        Oklab { l: 0.749, a: -0.097, b: 0.016 }, // #66c2a5
        Oklab { l: 0.848, a: -0.073, b: 0.058 }, // #abdda4
        Oklab { l: 0.938, a: -0.052, b: 0.106 }, // #e6f598
        Oklab { l: 0.985, a: -0.025, b: 0.077 }, // #ffffbf
        Oklab { l: 0.913, a: -0.001, b: 0.110 }, // #fee08b
        Oklab { l: 0.812, a: 0.059, b: 0.117 },  // #fdae61
        Oklab { l: 0.692, a: 0.139, b: 0.108 },  // #f46d43
        Oklab { l: 0.589, a: 0.177, b: 0.061 },  // #d53e4f
        Oklab { l: 0.448, a: 0.177, b: 0.024 },  // #9e0142
    ];

    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    pub fn interpolate_gradient(t: f64, colors: &[Oklab]) -> Oklab {
        let n = colors.len();

        assert_matches!(n, 2..);
        assert_matches!(t, 0.0..=1.0);

        let segments = (n - 1) as f64;
        let position = t * segments;

        let idx = position as usize;
        assert!(idx < n);

        let p = position - (idx as f64);

        let start = colors[idx];
        let end = colors[min(idx + 1, n - 1)];

        Oklab {
            l: start.l + (end.l - start.l) * p,
            a: start.a + (end.a - start.a) * p,
            b: start.b + (end.b - start.b) * p,
        }
    }
}
