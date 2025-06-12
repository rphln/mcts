#![deny(clippy::all)]
#![warn(clippy::pedantic)]

pub mod game;
pub mod mcts;

use std::{cmp::Reverse, fs::File, io::Write};

use rand::SeedableRng;
use swift_swallow::{
    Rng,
    action::{
        Action, ActionKey,
        Effect::{
            Block, Damage, DeploymentTactics, EmergencyTactics, FangAndClaw,
            LockAndLoad, QuickStep, Reprisal,
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
            time_to_act: 0,
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
        let action =
            Action { key, character, cooldown, on_cooldown_until: 0, effect, range };

        world.actions.push(action);

        key
    };

    // Vanguard
    let wv = character(White, Position { y: 5, x: 0 }, 768, 60, 700);
    let bv = character(Black, Position { y: 5, x: 12 }, 768, 60, 700);

    // Onslaught
    action(wv, 60, Shape::circle(100, 150, Mask::Enemy), Damage { potency: 120 });
    action(bv, 60, Shape::circle(100, 150, Mask::Enemy), Damage { potency: 120 });

    // Unmend
    action(wv, 120, Shape::circle(200, 650, Mask::Enemy), Damage { potency: 120 });
    action(bv, 120, Shape::circle(200, 650, Mask::Enemy), Damage { potency: 120 });

    // Reprisal
    action(
        wv,
        240,
        Shape::Implicit,
        Reprisal { duration: 240, area: Shape::circle(0, 600, Mask::Enemy) },
    );
    action(
        bv,
        240,
        Shape::Implicit,
        Reprisal { duration: 240, area: Shape::circle(0, 600, Mask::Enemy) },
    );

    // Fang and Claw
    action(wv, 240, Shape::circle(100, 150, Mask::Enemy), FangAndClaw);
    action(bv, 240, Shape::circle(100, 150, Mask::Enemy), FangAndClaw);

    // Scholar
    let ws = character(White, Position { y: 8, x: 0 }, 576, 72, 600);
    let bs = character(Black, Position { y: 0, x: 12 }, 576, 72, 600);

    // Ruin
    action(ws, 60, Shape::circle(200, 650, Mask::Enemy), Damage { potency: 120 });
    action(bs, 60, Shape::circle(200, 650, Mask::Enemy), Damage { potency: 120 });

    // Adloquium
    action(ws, 120, Shape::circle(0, 600, Mask::Friend), Block { potency: 180 });
    action(bs, 120, Shape::circle(0, 600, Mask::Friend), Block { potency: 180 });

    // Deployment Tactics
    action(
        ws,
        120,
        Shape::circle(0, 600, Mask::Friend),
        DeploymentTactics { area: Shape::circle(0, 300, Mask::Friend) },
    );
    action(
        bs,
        120,
        Shape::circle(0, 600, Mask::Friend),
        DeploymentTactics { area: Shape::circle(0, 300, Mask::Friend) },
    );

    // Emergency Tactics
    action(ws, 120, Shape::circle(0, 600, Mask::Friend), EmergencyTactics);
    action(bs, 120, Shape::circle(0, 600, Mask::Friend), EmergencyTactics);

    // Marksman
    let wm = character(White, Position { y: 0, x: 0 }, 384, 96, 700);
    let bm = character(Black, Position { y: 8, x: 12 }, 384, 96, 700);

    // Sidewinder
    action(wm, 60, Shape::circle(200, 650, Mask::Enemy), Damage { potency: 120 });
    action(bm, 60, Shape::circle(200, 650, Mask::Enemy), Damage { potency: 120 });

    // Quick Step
    action(wm, 240, Shape::Implicit, QuickStep);
    action(bm, 240, Shape::Implicit, QuickStep);

    // Lock and Load
    action(wm, 120, Shape::Implicit, LockAndLoad);
    action(bm, 120, Shape::Implicit, LockAndLoad);

    // Iron Jaws
    action(wm, 240, Shape::circle(200, 650, Mask::Enemy), Damage { potency: 240 });
    action(bm, 240, Shape::circle(200, 650, Mask::Enemy), Damage { potency: 240 });

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
    "⚪ Reprisal",
    "⚫ Reprisal",
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
    "⚪ Sidewinder",
    "⚫ Sidewinder",
    "⚪ Quick Step",
    "⚫ Quick Step",
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
