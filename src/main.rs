#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![feature(int_roundings)]

pub mod game;
pub mod mcts;
pub mod pvs;

use std::{cmp::Reverse, fs::File, io::Write, time::Instant};

use rand::SeedableRng;
use swift_swallow::{
    Rng,
    action::{
        Action, ActionKey,
        Effect::{
            Block, Damage, DeploymentTactics, EmergencyTactics, FangAndClaw, Grit,
            LockAndLoad, Sidewinder,
        },
        Mask, Shape,
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

use crate::{game::Game, mcts::Mcts};

fn main() -> anyhow::Result<()> {
    let game = Game::new(setup()?, White);

    if true {
        let mut mcts = Mcts::new(Command::None);
        let mut rng = Rng::seed_from_u64(1);

        for _ in 0..1_048_576 {
            let mut game = game.clone();
            let node = mcts.select_and_expand(&mut game, &mut rng);

            let reward = game.evaluate_f64();
            mcts.backward(node, reward);
        }

        let mut visits: Vec<usize> = mcts.tree.iter().map(|node| node.visits).collect();
        visits.sort_by_key(|&n| Reverse(n));

        mcts.gc(visits[10_000]);

        let root = 0;
        let ply = 0;

        let mut file = File::create("public_html/tree.tsv")?;
        visit_and_write(root, root, ply, &mcts, &mut file);
    } else {
        let now = Instant::now();

        let mut pv = Default::default();
        let mut history = Default::default();

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
    let wv = character(White, Position { y: 5, x: 0 }, 768, 12, 700);
    let bv = character(Black, Position { y: 5, x: 12 }, 768, 12, 700);

    // Onslaught
    action(wv, 0, Shape::circle(100, 150, Mask::Enemy), Damage { potency: 120 });
    action(bv, 0, Shape::circle(100, 150, Mask::Enemy), Damage { potency: 120 });

    // Unmend
    action(wv, 12, Shape::circle(200, 650, Mask::Enemy), Damage { potency: 90 });
    action(bv, 12, Shape::circle(200, 650, Mask::Enemy), Damage { potency: 90 });

    // Grit
    action(
        wv,
        24,
        Shape::Implicit,
        Grit { potency: 160, area: Shape::circle(0, 650, Mask::Enemy) },
    );
    action(
        bv,
        24,
        Shape::Implicit,
        Grit { potency: 160, area: Shape::circle(0, 650, Mask::Enemy) },
    );

    // Fang and Claw
    action(wv, 36, Shape::circle(100, 150, Mask::Enemy), FangAndClaw);
    action(bv, 36, Shape::circle(100, 150, Mask::Enemy), FangAndClaw);

    // Scholar
    let ws = character(White, Position { y: 8, x: 0 }, 576, 14, 600);
    let bs = character(Black, Position { y: 0, x: 12 }, 576, 14, 600);

    // Ruin
    action(ws, 0, Shape::circle(200, 650, Mask::Enemy), Damage { potency: 120 });
    action(bs, 0, Shape::circle(200, 650, Mask::Enemy), Damage { potency: 120 });

    // Adloquium
    action(ws, 12, Shape::circle(0, 600, Mask::Friend), Block { potency: 120 });
    action(bs, 12, Shape::circle(0, 600, Mask::Friend), Block { potency: 120 });

    // Deployment Tactics
    action(
        ws,
        36,
        Shape::circle(0, 600, Mask::Friend),
        DeploymentTactics { area: Shape::circle(0, 300, Mask::Friend) },
    );
    action(
        bs,
        36,
        Shape::circle(0, 600, Mask::Friend),
        DeploymentTactics { area: Shape::circle(0, 300, Mask::Friend) },
    );

    // Emergency Tactics
    action(ws, 36, Shape::circle(0, 600, Mask::Friend), EmergencyTactics);
    action(bs, 36, Shape::circle(0, 600, Mask::Friend), EmergencyTactics);

    // Marksman
    let wm = character(White, Position { y: 0, x: 0 }, 384, 16, 700);
    let bm = character(Black, Position { y: 8, x: 12 }, 384, 16, 700);

    // Bloodletter
    action(wm, 0, Shape::circle(200, 650, Mask::Enemy), Damage { potency: 120 });
    action(bm, 0, Shape::circle(200, 650, Mask::Enemy), Damage { potency: 120 });

    // Sidewinder
    action(wm, 36, Shape::Implicit, Sidewinder { duration: 36 });
    action(bm, 36, Shape::Implicit, Sidewinder { duration: 36 });

    // Lock and Load
    action(wm, 12, Shape::Implicit, LockAndLoad);
    action(bm, 12, Shape::Implicit, LockAndLoad);

    // Iron Jaws
    action(wm, 24, Shape::circle(200, 650, Mask::Enemy), Damage { potency: 240 });
    action(bm, 24, Shape::circle(200, 650, Mask::Enemy), Damage { potency: 240 });

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
        let label = label(&node.mov);
        let value = node.visits;

        let utility = if ply % 2 == 0 { -node.utility } else { node.utility };

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

fn label(mov: &Command) -> String {
    match mov {
        Command::None => "🌳".to_owned(),
        &Command::Pass { team, can_act, can_move } => {
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
