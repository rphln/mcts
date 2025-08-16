#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![feature(int_roundings)]

pub mod game;
pub mod mcts;
pub mod pvs;

use std::{
    collections::{HashMap, VecDeque},
    fs::File,
    io::Write,
    time::Instant,
};

use rand::SeedableRng;
use swift_swallow::{
    Rng,
    action::{
        Action, ActionKey,
        Effect::{
            Block, Damage, DeploymentTactics, EmergencyTactics, FangAndClaw, Grit,
            LockAndLoad, Sidewinder,
        },
        Mask::{Enemy, Friend},
        Shape,
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

    if true {
        let mut mcts = Mcts::new(Command::None);
        let mut rng = Rng::seed_from_u64(1);

        for it in 0..=1_048_576 {
            let mut game = game.clone();
            let node = mcts.select_and_expand(&mut game, &mut rng);

            let reward = game.evaluate_f64();
            mcts.backward(node, reward);

            if usize::is_power_of_two(it) {
                dbg!(it, now.elapsed(), mcts.len());
            }
        }

        mcts.gc(1_000);

        let root = 0;
        let ply = 0;

        let mut file = File::create("public_html/tree.tsv")?;
        visit_and_write(root, root, ply, &mcts, &mut file);
    } else {
        let mut pv = VecDeque::default();
        let mut history = HashMap::default();

        for depth in 0.. {
            let score = pvs::search(depth, &mut pv, &mut history, &game);
            dbg!(depth, now.elapsed(), score, &pv);
        }
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
        world.map.insert(position, Node::Occupied { team });

        key
    };

    let mut action = |character, cooldown, range, effect| {
        let key = ActionKey(world.actions.len());
        let action = Action { key, character, cooldown, ready_at: 0, effect, range };

        world.actions.push(action);

        key
    };

    // Vanguard
    let wv = character(White, Position::new(0, 5), 80, 12, 7);
    let bv = character(Black, Position::new(12, 5), 80, 12, 7);

    // Onslaught
    action(wv, 0, Shape::circle(1, 1, Enemy), Damage { potency: 9 });
    action(bv, 0, Shape::circle(1, 1, Enemy), Damage { potency: 9 });

    // Unmend
    action(wv, 12, Shape::circle(2, 6, Enemy), Damage { potency: 8 });
    action(bv, 12, Shape::circle(2, 6, Enemy), Damage { potency: 8 });

    // Grit
    action(
        wv,
        24,
        Shape::Implicit,
        Grit { potency: 13, area: Shape::circle(0, 6, Enemy) },
    );
    action(
        bv,
        24,
        Shape::Implicit,
        Grit { potency: 13, area: Shape::circle(0, 6, Enemy) },
    );

    // Fang and Claw
    action(wv, 36, Shape::circle(1, 1, Enemy), FangAndClaw);
    action(bv, 36, Shape::circle(1, 1, Enemy), FangAndClaw);

    // Scholar
    let ws = character(White, Position::new(0, 8), 75, 14, 6);
    let bs = character(Black, Position::new(12, 0), 75, 14, 6);

    // Ruin
    action(ws, 0, Shape::circle(2, 6, Enemy), Damage { potency: 9 });
    action(bs, 0, Shape::circle(2, 6, Enemy), Damage { potency: 9 });

    // Adloquium
    action(ws, 12, Shape::circle(0, 6, Friend), Block { potency: 11 });
    action(bs, 12, Shape::circle(0, 6, Friend), Block { potency: 11 });

    // Deployment Tactics
    action(
        ws,
        36,
        Shape::circle(0, 6, Friend),
        DeploymentTactics { area: Shape::circle(0, 3, Friend) },
    );
    action(
        bs,
        36,
        Shape::circle(0, 6, Friend),
        DeploymentTactics { area: Shape::circle(0, 3, Friend) },
    );

    // Emergency Tactics
    action(ws, 36, Shape::circle(0, 6, Friend), EmergencyTactics);
    action(bs, 36, Shape::circle(0, 6, Friend), EmergencyTactics);

    // Marksman
    let wm = character(White, Position::new(0, 0), 70, 16, 7);
    let bm = character(Black, Position::new(12, 8), 70, 16, 7);

    // Bloodletter
    action(wm, 0, Shape::circle(2, 6, Enemy), Damage { potency: 9 });
    action(bm, 0, Shape::circle(2, 6, Enemy), Damage { potency: 9 });

    // Sidewinder
    action(wm, 36, Shape::circle(2, 6, Enemy), Sidewinder { duration: 36 });
    action(bm, 36, Shape::circle(2, 6, Enemy), Sidewinder { duration: 36 });

    // Lock and Load
    action(wm, 12, Shape::Implicit, LockAndLoad);
    action(bm, 12, Shape::Implicit, LockAndLoad);

    // Iron Jaws
    action(wm, 24, Shape::circle(2, 6, Enemy), Damage { potency: 18 });
    action(bm, 24, Shape::circle(2, 6, Enemy), Damage { potency: 18 });

    let event_bus = rules();

    let event = Turn { prevent_default: false };
    let _ = event_bus.on_turn_start(event, &event_bus, &mut world)?;

    Ok(world)
}

fn visit_and_write(
    node_ref: usize,
    mut parent_ref: usize,
    ply: usize,
    mcts: &Mcts,
    writer: &mut impl Write,
) {
    let is_root = node_ref == parent_ref;

    if is_root {
        let _ = writeln!(writer, "id	parent	label	value	utility");
    }

    let node = &mcts.tree[node_ref];

    if !is_root && matches!(node.mov, Command::None) {
        // Skip empty nodes.
    } else {
        let label = label(node);

        let value = node.visits;
        let utility = node.utility;

        // Plotly needs the parent of the root to be blank.
        let maybe_parent = if is_root { String::new() } else { parent_ref.to_string() };
        let _ =
            writeln!(writer, "{node_ref}	{maybe_parent}	{label}	{value}	{utility}",);

        parent_ref = node_ref;
    }

    for child_ref in node.head..node.last {
        visit_and_write(child_ref, parent_ref, ply + 1, mcts, writer);
    }
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
        Command::None => "🌳".to_owned(),
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
                "{name} → ⟨{x}, {y}⟩",
                name = CHARACTER[character.0],
                x = destination.x,
                y = destination.y,
            )
        }
        Command::Action { action, destination } => {
            format!(
                "{name} → ⟨{x}, {y}⟩",
                name = ACTION[action.0],
                x = destination.x,
                y = destination.y,
            )
        }
    }
}
