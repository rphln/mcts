use std::{
    cmp::Reverse,
    fs::File,
    io::Write,
    num::ParseIntError,
    path::PathBuf,
    process::Command,
    time::{Duration, Instant},
};

use clap::Parser;
use rand::prelude::*;
use serde::Serialize;
use swift_swallow_tree_search::{
    DefaultRng,
    game::{Color, Game, Move},
    mcts::Mcts,
};

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
    let start_time = Instant::now();

    let game = Game::new(args.seed, false)?;

    let mut rng = DefaultRng::seed_from_u64(args.seed);
    let mut mcts = Mcts::new(Move::None { team: Color::Black });

    let _next = mcts.search(0, &game, &mut rng, args.time, args.iters, args.nodes);

    let elapsed = start_time.elapsed();
    eprintln!("Done in {elapsed:?}");

    let pruning_threshold = {
        let mut visits: Vec<u32> = mcts.tree.iter().map(|node| node.visits).collect();
        visits.sort_unstable();

        let top_k = visits.len().saturating_sub(1_000_000);
        visits[top_k]
    };

    let mut file = File::create(&args.destination)?;
    tree_to_html(&mcts, pruning_threshold, &mut file)?;

    let _ = Command::new("open").arg(&args.destination).spawn()?;

    Ok(())
}

fn parse_millis(arg: &str) -> Result<Duration, ParseIntError> {
    Ok(Duration::from_millis(arg.parse()?))
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}

#[derive(Serialize)]
struct NodeJson {
    label: String,
    title: String,
    visits: u32,
    win_rate: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    children: Vec<NodeJson>,
}

fn build_node_json(
    node_idx: usize,
    mcts: &Mcts,
    pruning_threshold: u32,
    is_pv: bool,
) -> Option<NodeJson> {
    let node = &mcts.tree[node_idx];
    if !is_pv && node.visits <= pruning_threshold {
        return None;
    }

    let mut child_indices: Vec<usize> = (node.head..node.last).collect();
    child_indices.sort_by_key(|&i| Reverse(mcts.tree[i].visits));

    let win_rate = sigmoid(node.value);
    let label = move_label(mcts.moves.resolve(node.mov).unwrap());

    let parent_visit_share = if node.is_root() {
        100.0
    } else {
        100.0 * f64::from(node.visits) / f64::from(mcts.tree[node.parent].visits)
    };

    let title = format!(
        "{label}\n\nVisits: {visits}\nShare of parent\u{2019}s visits: {share:.3}%\nMean value (Q): {q:.3}\nPredicted win rate: {wr:.2}%",
        visits = node.visits,
        share = parent_visit_share,
        q = node.value,
        wr = 100.0 * win_rate,
    );

    let children = child_indices
        .into_iter()
        .enumerate()
        .filter_map(|(i, child_idx)| {
            build_node_json(child_idx, mcts, pruning_threshold, is_pv && i == 0)
        })
        .collect();

    Some(NodeJson { label, title, visits: node.visits, win_rate, children })
}

fn tree_to_html(
    mcts: &Mcts,
    pruning_threshold: u32,
    w: &mut impl Write,
) -> anyhow::Result<()> {
    writeln!(w, "<!doctype html>")?;
    writeln!(w, r#"<html lang="en">"#)?;
    writeln!(w, "  <head>")?;
    writeln!(w, r#"    <meta charset="utf-8" />"#)?;
    writeln!(
        w,
        r#"    <meta name="viewport" content="width=device-width,initial-scale=1" />"#
    )?;
    writeln!(w, "    <title>Flamegraph</title>")?;
    writeln!(w, r"    <style>{}</style>", include_str!("flamegraph.css"))?;
    writeln!(
        w,
        r#"    <script type="module">{}</script>"#,
        include_str!("flamegraph.js")
    )?;
    writeln!(w, "  </head>")?;
    writeln!(w, "  <body>")?;
    writeln!(w, r#"    <div id="flamegraph-container"></div>"#)?;
    write!(w, r#"    <script id="flamegraph-data" type="application/json">"#)?;
    serde_json::to_writer(
        &mut *w,
        &build_node_json(0, mcts, pruning_threshold, true).unwrap(),
    )?;
    writeln!(w, "</script>")?;
    writeln!(w, "  </body>")?;
    writeln!(w, "</html>")?;
    Ok(())
}

const CHARACTERS: [&str; 8] = [
    "Warden ⚪",
    "Warden ⚫",
    "Berserker ⚪",
    "Berserker ⚫",
    "Scholar ⚪",
    "Scholar ⚫",
    "Sidewinder ⚪",
    "Sidewinder ⚫",
];

const ACTIONS: [&str; 56] = [
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

fn move_label(mov: &Move) -> String {
    match mov {
        Move::None { team } => match team {
            Color::White => "⚪".to_owned(),
            Color::Black => "⚫".to_owned(),
        },
        Move::Pass { character } => {
            format!("⌛ {label} → Pass", label = CHARACTERS[character.0])
        }
        Move::Move { character, destination } => {
            format!(
                "🧭 {label} → ⟨{x}, {y}⟩",
                label = CHARACTERS[character.0],
                x = destination.q,
                y = destination.r,
            )
        }
        Move::Act { action, target_hint: Some(target), .. } => {
            format!(
                "🎯 {label} → {target}",
                label = ACTIONS[action.0],
                target = CHARACTERS[target.0],
            )
        }
        Move::Act { action, destination, .. } => {
            format!(
                "🎯 {label} → ⟨{x}, {y}⟩",
                label = ACTIONS[action.0],
                x = destination.q,
                y = destination.r,
            )
        }
    }
}
