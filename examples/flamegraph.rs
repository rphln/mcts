use std::{
    fmt::Display,
    fs::File,
    io::{self, BufWriter, Write},
    num::ParseIntError,
    path::PathBuf,
    time::{Duration, Instant},
};

use clap::Parser;
use rand::prelude::*;
use swift_swallow::{dsl::Context, rules::rules, setup};
use swift_swallow_tree_search::{
    DefaultRng,
    game::{Color, Game, Move},
    mcts::{Mcts, Node},
};

/// Plots a flamegraph of the MCTS search tree after searching from the root.
#[derive(Debug, Parser)]
pub struct Args {
    /// Path to the output file.
    #[clap(default_value = "/tmp/flamegraph.bin")]
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

    let game = {
        let rules = rules();
        let world = setup(args.seed, false, &rules);

        Game::new(world, rules, None)
    };

    let mut rng = DefaultRng::seed_from_u64(args.seed);
    let mut mcts = Mcts::new(Move::None { team: Color::Black });

    let _next = mcts.search(0, &game, &mut rng, args.time, args.iters, args.nodes);
    eprintln!(
        "Expanded {n} nodes in {elapsed:.2?}",
        n = mcts.len(),
        elapsed = start_time.elapsed()
    );

    let names: Vec<String> = {
        let character_names: Vec<String> = game
            .world
            .characters
            .iter()
            .map(|character| match character.team {
                Color::White => format!("{} ⚪", character.name),
                Color::Black => format!("{} ⚫", character.name),
            })
            .collect();
        let action_names: Vec<String> = game
            .world
            .actions
            .iter()
            .map(|action| match game.world.characters[action.character].team {
                Color::White => format!("{} ⚪", action.name),
                Color::Black => format!("{} ⚫", action.name),
            })
            .collect();

        mcts.moves
            .keys()
            .map(|mov| move_label(mov, &character_names, &action_names))
            .collect()
    };

    let writer = BufWriter::new(File::create(args.destination)?);

    let start_time = Instant::now();
    let num_bytes = serialize(&mcts.nodes, &names, writer)?;

    eprintln!(
        "Written {num_bytes} bytes in {elapsed:.2?}",
        elapsed = start_time.elapsed()
    );

    Ok(())
}

/// ┌─ nodes ────────────────────────────────── `at_nodes` ─┐
/// │  Node × count                                         │
/// │    u16  mov                                           │
/// │    u32  visits                                        │
/// │    f64  value                                         │
/// │    u64  parent                                        │
/// │    u64  head                                          │
/// │    u64  last                                          │
/// ├─ name data ─────────────────────── `at_name_data` ────┤
/// │  u8[]                                                 │
/// ├─ name index ────────────────────── `at_name_index` ───┤
/// │  Entry × count                                        │
/// │    u64  ptr                                           │
/// │    u64  len                                           │
/// ├─ footer ──────────────────────────────────────────────┤
/// │  u64  ptr    (`at_nodes`)                             │
/// │  u64  count                                           │
/// │  u64  ptr    (`at_name_index`)                        │
/// │  u64  count                                           │
/// └───────────────────────────────────────────────────────┘
fn serialize(
    tree: &[Node],
    names: &[String],
    mut out: impl Write,
) -> io::Result<usize> {
    /// Writes a fixed-size byte array; the const `N` is a compile-time
    /// assertion that the field's wire size matches the format spec.
    fn field<const N: usize>(mut out: impl Write, bytes: [u8; N]) -> io::Result<usize> {
        out.write_all(&bytes)?;
        Ok(N)
    }

    let mut pos = 0;

    let at_nodes = pos;
    for node in tree {
        pos += field::<8>(&mut out, node.mov.to_le_bytes())?;
        pos += field::<4>(&mut out, node.visits.to_le_bytes())?;
        pos += field::<8>(&mut out, node.value.to_le_bytes())?;
        pos += field::<8>(&mut out, node.parent.to_le_bytes())?;
        pos += field::<8>(&mut out, node.head.to_le_bytes())?;
        pos += field::<8>(&mut out, node.last.to_le_bytes())?;
    }

    let at_name_data = pos;
    for name in names {
        out.write_all(name.as_bytes())?;
        pos += name.len();
    }

    let at_name_index = pos;
    let mut name_ptr = at_name_data;
    for name in names {
        pos += field::<8>(&mut out, name_ptr.to_le_bytes())?;
        pos += field::<8>(&mut out, name.len().to_le_bytes())?;
        name_ptr += name.len();
    }

    pos += field::<8>(&mut out, at_nodes.to_le_bytes())?;
    pos += field::<8>(&mut out, tree.len().to_le_bytes())?;

    pos += field::<8>(&mut out, at_name_index.to_le_bytes())?;
    pos += field::<8>(&mut out, names.len().to_le_bytes())?;

    Ok(pos)
}

fn parse_millis(arg: &str) -> Result<Duration, ParseIntError> {
    Ok(Duration::from_millis(arg.parse()?))
}

fn move_label(mov: &Move, names: &[impl Display], actions: &[impl Display]) -> String {
    match mov {
        Move::None { team } => match team {
            Color::White => "⚪".into(),
            Color::Black => "⚫".into(),
        },

        Move::Pass { character, .. } => {
            format!("⌛ {} → Pass", names[character.0])
        }

        Move::Move { character, destination } => {
            format!("🧭 {} → {}", names[character.0], destination)
        }

        Move::Act { action, context: Context { characters, positions }, .. } => {
            let label = format!("🎯 {}", actions[action.0]);
            let targets = characters
                .iter()
                .zip(positions)
                .skip(1) // Skip the caster.
                .filter_map(|(&character, &position)| {
                    let character = character.map(|chr| names[chr.0].to_string());
                    let position = position.map(|pos| pos.to_string());

                    character.or(position)
                })
                .reduce(|acc, target| format!("{acc} · {target}"));

            match targets {
                Some(targets) => format!("{label} → {targets}"),
                None => label,
            }
        }
    }
}
