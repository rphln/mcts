#![feature(file_buffered)]

use std::{
    cmp::max,
    fs::{File, create_dir_all},
    panic::catch_unwind,
    path::PathBuf,
    time::Instant,
};

use anyhow::{Result, bail};
use clap::Parser;
use rand::prelude::*;
use swift_swallow::{TeamKey, character::CharacterKey};
use swift_swallow_tree_search::{
    DefaultRng,
    game::{Color, Game, Move},
    mcts::{Mcts, Node},
};

#[derive(Clone, Debug, Parser)]
pub struct Args {
    #[arg(default_value = "dist/")]
    pub destination: PathBuf,
    #[arg(long, default_value_t = 65_536)]
    pub iters: u32,
    #[arg(long, default_value_t = rand::rng().next_u64())]
    pub seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    create_dir_all(&args.destination)?;

    let mut rng = DefaultRng::seed_from_u64(args.seed);

    loop {
        let now = Instant::now();
        let seed = rng.next_u64();

        let Ok(()) = catch_unwind(|| run_game(seed, &args)) else {
            bail!("seed={seed:016x} panicked");
        };

        let elapsed = now.elapsed();
        println!("seed={seed:016x} elapsed={elapsed:?}");
    }
}

fn run_game(seed: u64, args: &Args) {
    let mut game = Game::new(seed, true).unwrap();

    let mut search_rng = DefaultRng::seed_from_u64(seed);
    let mut mcts = Mcts::new(Move::None { team: Color::Black });

    let mut records = Vec::new();

    for turn in 0..1024 {
        if game.is_over() {
            break;
        }

        let _ = mcts
            .search(0, &game, &mut search_rng, None, Some(args.iters), None)
            .expect("`node` should have children");

        let Node { head, last, .. } = mcts.nodes[0];
        let root = (head..last)
            .max_by_key(|&node| mcts.nodes[node].visits)
            .expect("root should have children");

        let (&mov, _history) = mcts.moves.get_index(mcts.nodes[root].mov).unwrap();
        game.play(mov);

        mcts.reroot(root);

        let active_team = game.world.active_team();
        let active_character = game.world.active_character;

        for character in &game.world.characters {
            let record = Record {
                seed,

                turn,
                tick: game.world.tick,

                character: character.key,
                team: character.team,

                active_character,
                active_team,

                ready_at: character.ready_at,
                cooldown: character.cooldown,

                can_act: character.can_act,
                can_move: character.can_move,

                maximum_health: character.health,
                current_health: character.current_health(),

                is_alive: !character.is_defeated(),

                block: character.block,
                poison: character.counters.poison,

                strength: character.counters.strength,
                dexterity: character.counters.dexterity,

                weakness: max(character.timers.weak - game.world.tick, 0),
                vulnerable: max(character.timers.vulnerable - game.world.tick, 0),
            };

            records.push(record);
        }
    }

    if !game.is_over() {
        eprintln!("Seed {seed:016x} exceeded the turn limit; aborting.");
        return;
    }

    let destination = args.destination.join(format!("{seed:016x}.json"));
    let writer = File::create_buffered(&destination).unwrap();

    serde_json::to_writer(writer, &records).unwrap();
}

#[derive(Debug, serde::Serialize)]
pub struct Record {
    pub seed: u64,

    pub turn: u32,
    pub tick: i32,

    pub character: CharacterKey,
    pub team: TeamKey,

    pub active_character: CharacterKey,
    pub active_team: TeamKey,

    pub ready_at: i32,
    pub cooldown: i32,

    pub can_act: bool,
    pub can_move: bool,

    pub maximum_health: i32,
    pub current_health: i32,

    pub is_alive: bool,

    pub poison: i32,
    pub block: i32,

    pub strength: i32,
    pub dexterity: i32,

    pub weakness: i32,
    pub vulnerable: i32,
}
