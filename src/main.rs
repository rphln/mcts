#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![feature(int_roundings)]

pub mod game;
pub mod mcts;

use std::{fs::File, io::Write, time::Instant};

use rand::SeedableRng;
use swift_swallow::{
    Rng,
    action::{
        Action, ActionKey,
        Effect::{
            Block, Damage, DeploymentTactics, EmergencyTactics, FangAndClaw, Grit,
            LockAndLoad, Sidewinder,
        },
        enemy_in_range, friend_in_range, self_only,
    },
    character::{Character, CharacterKey, Markers},
    event::{Command, Turn},
    position::Position,
    rules::rules,
    subscribe::Subscribe,
    team::TeamKey::{self, Black, White},
    world::{
        Node::{self, Closed as X, Open as O},
        World,
    },
};

use crate::{
    game::Game,
    mcts::{Mcts, MctsNode},
};

fn main() -> anyhow::Result<()> {
    let game = Game::new(setup()?, White);

    let now = Instant::now();

    let mut mcts = Mcts::new(Command::None { team: Black });
    let mut rng = Rng::seed_from_u64(1);

    const N: usize = 8 * 1024 * 1024;

    for it in 1..=N {
        let mut game = game.clone();
        let node = mcts.select_and_expand(&mut game, &mut rng);

        let reward = game.evaluate_f64();
        mcts.backward(node, reward);

        if usize::is_power_of_two(it) {
            dbg!(it, now.elapsed(), mcts.len());
        }
    }

    let mut file = File::create("public_html/tree.tsv")?;
    visit_and_write(&mcts, N / 1024, &mut file)?;

    Ok(())
}

fn visit_and_write(
    mcts: &Mcts,
    min_visits: usize,
    writer: &mut impl Write,
) -> std::io::Result<()> {
    writeln!(writer, "id	parent	label	visits	utility")?;

    for (id, node) in mcts.tree.iter().enumerate() {
        let visits = node.visits;
        let utility = node.value;

        if visits < min_visits {
            continue;
        }

        let label = label(node);

        let parent_id =
            if node.is_root() { String::new() } else { node.parent.to_string() };
        writeln!(writer, "{id}	{parent_id}	{label}	{visits}	{utility:.3}")?;
    }

    Ok(())
}

fn setup() -> anyhow::Result<World> {
    const MAP: [[Node; 13]; 9] = [
        [O, O, O, O, O, O, X, O, O, O, O, O, O],
        [O, O, O, X, X, O, X, O, X, X, X, X, O],
        [O, X, O, O, O, O, O, O, O, O, O, O, O],
        [O, X, X, O, X, X, O, O, O, O, O, O, O],
        [O, O, O, O, O, O, O, O, O, O, O, O, O],
        [O, O, O, O, O, O, O, X, X, O, X, X, O],
        [O, O, O, O, O, O, O, O, O, O, O, X, O],
        [O, X, X, X, X, O, X, O, X, X, O, O, O],
        [O, O, O, O, O, O, X, O, O, O, O, O, O],
    ];

    let mut world = World::new(0, MAP);

    let mut character = |team, position, health, cooldown, movement| {
        let key = CharacterKey(world.characters.len());
        let character = Character {
            key,
            team,
            health,
            block: 0,
            damage: 0,
            movement,
            position,
            cooldown,
            ready_at: 0,
            can_act: false,
            can_move: false,
            markers: Markers::default(),
        };

        world.characters.push(character);
        world.map.insert(position, Node::Occupied { team, character: key });

        key
    };

    let mut action = |character, cooldown, conditions, effect| {
        let key = ActionKey(world.actions.len());
        let action =
            Action { key, character, cooldown, ready_at: 0, conditions, effect };

        world.actions.push(action);

        key
    };

    // Vanguard
    let wv = character(White, Position::new(0, 5), 80, 12, 7);
    let bv = character(Black, Position::new(12, 5), 80, 12, 7);

    // Onslaught
    action(wv, 0, enemy_in_range(1, 2), Damage { potency: 9 });
    action(bv, 0, enemy_in_range(1, 2), Damage { potency: 9 });

    // Unmend
    action(wv, 12, enemy_in_range(2, 7), Damage { potency: 8 });
    action(bv, 12, enemy_in_range(2, 7), Damage { potency: 8 });

    // Grit
    action(wv, 24, self_only(), Grit { potency: 13, area: enemy_in_range(0, 7) });
    action(bv, 24, self_only(), Grit { potency: 13, area: enemy_in_range(0, 7) });

    // Fang and Claw
    action(wv, 36, enemy_in_range(1, 2), FangAndClaw);
    action(bv, 36, enemy_in_range(1, 2), FangAndClaw);

    // Scholar
    let ws = character(White, Position::new(0, 8), 75, 14, 6);
    let bs = character(Black, Position::new(12, 0), 75, 14, 6);

    // Ruin
    action(ws, 0, enemy_in_range(2, 7), Damage { potency: 9 });
    action(bs, 0, enemy_in_range(2, 7), Damage { potency: 9 });

    // Adloquium
    action(ws, 12, friend_in_range(0, 7), Block { potency: 11 });
    action(bs, 12, friend_in_range(0, 7), Block { potency: 11 });

    // Deployment Tactics
    action(
        ws,
        36,
        friend_in_range(0, 7),
        DeploymentTactics { area: friend_in_range(0, 4) },
    );
    action(
        bs,
        36,
        friend_in_range(0, 7),
        DeploymentTactics { area: friend_in_range(0, 4) },
    );

    // Emergency Tactics
    action(ws, 36, friend_in_range(0, 7), EmergencyTactics);
    action(bs, 36, friend_in_range(0, 7), EmergencyTactics);

    // Marksman
    let wm = character(White, Position::new(0, 0), 70, 16, 7);
    let bm = character(Black, Position::new(12, 8), 70, 16, 7);

    // Bloodletter
    action(wm, 0, enemy_in_range(2, 7), Damage { potency: 9 });
    action(bm, 0, enemy_in_range(2, 7), Damage { potency: 9 });

    // Sidewinder
    action(wm, 36, enemy_in_range(2, 7), Sidewinder { duration: 36 });
    action(bm, 36, enemy_in_range(2, 7), Sidewinder { duration: 36 });

    // Lock and Load
    action(wm, 12, self_only(), LockAndLoad);
    action(bm, 12, self_only(), LockAndLoad);

    // Iron Jaws
    action(wm, 24, enemy_in_range(2, 7), Damage { potency: 18 });
    action(bm, 24, enemy_in_range(2, 7), Damage { potency: 18 });

    let event_bus = rules();

    let event = Turn { prevent_default: false };
    let _ = event_bus.on_turn_start(event, &event_bus, &mut world)?;

    Ok(world)
}

const CHARACTER: [&str; 6] = [
    "⚪ Vanguard",
    "⚫ Vanguard",
    "⚪ Scholar",
    "⚫ Scholar",
    "⚪ Marksman",
    "⚫ Marksman",
];
const ACTION: [&str; 24] = [
    "⚪ Onslaught",
    "⚫ Onslaught",
    "⚪ Unmend",
    "⚫ Unmend",
    "⚪ Grit",
    "⚫ Grit",
    "⚪ Fang and Claw",
    "⚫ Fang and Claw",
    "⚪ Ruin",
    "⚫ Ruin",
    "⚪ Adloquium",
    "⚫ Adloquium",
    "⚪ Deployment Tactics",
    "⚫ Deployment Tactics",
    "⚪ Emergency Tactics",
    "⚫ Emergency Tactics",
    "⚪ Bloodletter",
    "⚫ Bloodletter",
    "⚪ Sidewinder",
    "⚫ Sidewinder",
    "⚪ Lock and Load",
    "⚫ Lock and Load",
    "⚪ Iron Jaws",
    "⚫ Iron Jaws",
];

fn label(node: &MctsNode) -> String {
    match node.mov {
        Command::None { team } => match team {
            TeamKey::White => "⚪".to_owned(),
            TeamKey::Black => "⚫".to_owned(),
        },
        Command::Pass { team, can_act, can_move } => {
            let color = match team {
                TeamKey::White => "⚪",
                TeamKey::Black => "⚫",
            };

            let marker = if can_act && can_move {
                "‼️"
            } else if can_act || can_move {
                "❗"
            } else {
                ""
            };

            format!("{color} Pass {marker}")
        }
        Command::Movement { character, destination } => {
            format!(
                "{label} → ⟨{x}, {y}⟩",
                label = CHARACTER[character.0],
                x = destination.x,
                y = destination.y,
            )
        }
        Command::Action { action, destination } => {
            format!(
                "{label} → ⟨{x}, {y}⟩",
                label = ACTION[action.0],
                x = destination.x,
                y = destination.y,
            )
        }
    }
}
