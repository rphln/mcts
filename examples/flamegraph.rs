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

/// Plots a flamegraph of the MCTS search tree after searching from the root.
#[derive(Parser, Debug)]
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

    let mut file = File::create(&args.destination)?;
    tree_to_html(&mcts, &args, &mut file)?;

    let _ = Command::new("open").arg(&args.destination).spawn()?;

    Ok(())
}

fn parse_millis(arg: &str) -> Result<Duration, ParseIntError> {
    let millis = arg.parse()?;
    Ok(Duration::from_millis(millis))
}

// region: Plotting.

fn tree_to_html(mcts: &Mcts, args: &Args, w: &mut impl Write) -> anyhow::Result<()> {
    let pruning_threshold = {
        let mut visits: Vec<u32> = mcts.tree.iter().map(|node| node.visits).collect();
        visits.sort_unstable();

        let top_k = visits.len().saturating_sub(500_000);
        visits[top_k]
    };

    let max = mcts
        .tree
        .iter()
        .filter(|node| node.visits > pruning_threshold)
        .fold(0., |max, node| f64::max(max, f64::abs(node.value)));
    let min = -max;

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

    node_to_html(0, mcts, pruning_threshold, min, max, w)?;

    writeln!(w, r"  </body>")?;

    writeln!(w, r"</html>")?;

    Ok(())
}

fn node_to_html(
    node: usize,
    mcts: &Mcts,
    pruning_threshold: u32,
    min: f64,
    max: f64,
    w: &mut impl Write,
) -> anyhow::Result<()> {
    let node = &mcts.tree[node];
    if node.visits <= pruning_threshold {
        return Ok(());
    }

    let mut children: Vec<usize> = (node.head..node.last).collect();
    children.sort_by_key(|&idx| Reverse(mcts.tree[idx].visits));

    let mov = mcts.moves.lookup(node.mov);

    let label = label(mov);
    let value = (node.value - min) / (max - min);

    let width = if node.is_root() {
        100.
    } else {
        let parent = &mcts.tree[node.parent];
        100. * f64::from(node.visits) / f64::from(parent.visits)
    };

    let title = format!(
        r"{label}&#10;Visits: {visits}&#10;Value: {value:.3}",
        visits = node.visits,
        value = node.value
    );

    writeln!(w, r#"<div class="node" style="--width: {width:.3}%">"#)?;
    writeln!(w, r#"  <div class="bar" style="--value: {value:.3}" title="{title}">"#)?;
    writeln!(w, r#"    <span class="label">{label}</span>"#)?;
    writeln!(w, r"  </div>")?;

    for child in children {
        node_to_html(child, mcts, pruning_threshold, min, max, w)?;
    }

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
