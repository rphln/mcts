use std::{cmp::Reverse, fs::File, io::Write, num::ParseIntError, time::Duration};

use clap::Parser;
use rand::prelude::*;
use swift_swallow::examples::setup;
use swift_swallow_tree_search::{
    DefaultRng,
    game::{Color, Game, Move},
    mcts::Mcts,
};

/// Plots a flamegraph of the MCTS search tree after searching from the root.
#[derive(Parser)]
pub struct Search {
    /// Seed for the game and search.
    #[clap(long, default_value_t = 0)]
    pub seed: u64,
    /// How much time to spend per move, in milliseconds.
    #[clap(long, value_parser = parse_millis, required_unless_present_any = ["max_iters", "max_nodes"])]
    pub max_time: Option<Duration>,
    /// How many iterations to search per move.
    #[clap(long, required_unless_present_any = ["max_time", "max_nodes"])]
    pub max_iters: Option<u32>,
    /// How many nodes to search per move.
    #[clap(long, required_unless_present_any = ["max_time", "max_iters"])]
    pub max_nodes: Option<usize>,
    /// Whether to randomize the initial `ready_at` of characters.
    #[clap(long)]
    pub randomize_ready_at: bool,
}

pub fn main() -> anyhow::Result<()> {
    let args = Search::parse();

    let game_seed = args.seed;
    let game =
        Game::new(setup(game_seed, args.randomize_ready_at).unwrap(), Color::White);

    let mut rng = DefaultRng::seed_from_u64(args.seed);
    let mut mcts = Mcts::new(Move::None { team: Color::Black });

    let root = 0;
    let _next = mcts.search(
        root,
        &game,
        &mut rng,
        args.max_time,
        args.max_iters,
        args.max_nodes,
    );

    let mut file = File::create("public_html/index.html")?;
    tree_to_html(&mcts, &mut file)?;

    Ok(())
}

fn parse_millis(arg: &str) -> Result<Duration, ParseIntError> {
    let millis = arg.parse()?;
    Ok(Duration::from_millis(millis))
}

// region: Plotting.

fn tree_to_html(mcts: &Mcts, w: &mut impl Write) -> anyhow::Result<()> {
    let pruning_threshold = {
        let mut visits: Vec<u32> = mcts.tree.iter().map(|node| node.visits).collect();
        visits.sort_unstable();

        let top_n = visits.len().saturating_sub(100_000);
        visits[top_n]
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

    let label = label(&node.mov);
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
const ACTION: [&str; 48] = [
    "Strike (Warden) ⚪",
    "Strike (Warden) ⚫",
    "Defend (Warden) ⚪",
    "Defend (Warden) ⚫",
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
        Move::Movement { character, destination } => {
            format!(
                "🧭 {label} → ⟨{x}, {y}⟩",
                label = CHARACTER[character.0],
                x = destination.q,
                y = destination.r,
            )
        }
        Move::Action { action, target_hint: Some(target), .. } => {
            format!(
                "🎯 {label} → {target}",
                label = ACTION[action.0],
                target = CHARACTER[target.0],
            )
        }
        Move::Action { action, destination, .. } => {
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
