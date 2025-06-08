#![deny(clippy::all)]
#![warn(clippy::pedantic)]

pub mod game;
pub mod mcts;

use std::time::Instant;

use mcts::MctsNode;
use rand::SeedableRng;
use swift_swallow::{
    Rng,
    action::{
        Action, ActionKey,
        Effect::{
            Block, Damage, DeploymentTactics, EmergencyTactics, ExchangePlates,
            LockAndLoad, Suppress, TacticalRelocation,
        },
    },
    character::{Character, CharacterKey, Markers},
    coordinate::xy,
    event::{Command, Turn},
    get_action_name_internet,
    rules::rules,
    subscribe::Subscribe,
    team::TeamKey::{Black, White},
    world::{
        Node::{self, Closed as X, Open as O},
        World,
    },
};

use crate::{game::Game, mcts::Mcts};

fn main() -> anyhow::Result<()> {
    let game = {
        let world = setup()?;
        Game::new(world)
    };

    let now = Instant::now();

    let mut mcts = Mcts::new(Command::Pass);
    let mut rng = Rng::seed_from_u64(0);

    for step in 1usize.. {
        let mut game = game.clone();
        let node = mcts.select_and_expand(&mut game, &mut rng);

        let reward = game.evaluate_f64();
        mcts.backward(node, reward);

        // 16 GiB.
        if mcts.len() >= 268_435_456 {
            mcts.gc(16_384);
        }

        if step.is_power_of_two() {
            let pv: Vec<&MctsNode> = mcts
                .principal_variation()
                .filter(|node| !matches!(node.mov, Command::None))
                .collect();
            dbg!(step, now.elapsed(), pv);
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

    let mut character = |_name: &str, team, position, health, cooldown| {
        let key = CharacterKey(world.characters.len());
        let character = Character {
            key,
            team,
            health,
            block: 0,
            damage: 0,
            movement: 600,
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

    let mut action =
        |character, name: &str, cooldown, (min_range, max_range), effect| {
            let key = ActionKey(world.actions.len());
            let action = Action {
                key,
                character,
                cooldown,
                on_cooldown_until: 0,
                effect,
                min_range,
                max_range,
            };

            let mut interner = get_action_name_internet();
            interner.insert(key, name.to_owned());

            world.actions.push(action);
        };

    // 0
    let wv = character("Vanguard", White, xy(0, 5), 768, 60);
    let bv = character("Vanguard", Black, xy(12, 5), 768, 60);

    // CommandKey(0)
    action(wv, "Vanguard Strike (W)", 60, (200, 400), Damage { potency: 120 });
    action(bv, "Vanguard Strike (B)", 60, (200, 400), Damage { potency: 120 });

    // CommandKey(2)
    action(wv, "Concentrated Gunfire (W)", 240, (200, 400), Damage { potency: 180 });
    action(bv, "Concentrated Gunfire (B)", 240, (200, 400), Damage { potency: 180 });

    // CommandKey(4)
    action(wv, "Suppressive Fire (W)", 240, (200, 400), Suppress { duration: 240 });
    action(bv, "Suppressive Fire (B)", 240, (200, 400), Suppress { duration: 240 });

    // CommandKey(6)
    action(wv, "Exchange Plates (W)", 120, (0, 0), ExchangePlates { potency: 180 });
    action(bv, "Exchange Plates (B)", 120, (0, 0), ExchangePlates { potency: 180 });

    // 2
    let ws = character("Specialist", White, xy(0, 8), 576, 72);
    let bs = character("Specialist", Black, xy(12, 0), 576, 72);

    // CommandKey(8)
    action(ws, "Specialist Strike (W)", 60, (200, 600), Damage { potency: 120 });
    action(bs, "Specialist Strike (B)", 60, (200, 600), Damage { potency: 120 });

    // CommandKey(10)
    action(ws, "Adloquium (W)", 120, (0, 600), Block { potency: 180 });
    action(bs, "Adloquium (B)", 120, (0, 600), Block { potency: 180 });

    // CommandKey(12)
    action(ws, "Deployment Tactics (W)", 120, (0, 600), DeploymentTactics);
    action(bs, "Deployment Tactics (B)", 120, (0, 600), DeploymentTactics);

    // CommandKey(14)
    action(ws, "Emergency Tactics (W)", 120, (0, 600), EmergencyTactics);
    action(bs, "Emergency Tactics (B)", 120, (0, 600), EmergencyTactics);

    // 4
    let wm = character("Marksman", White, xy(0, 0), 384, 96);
    let bm = character("Marksman", Black, xy(12, 8), 384, 96);

    // CommandKey(16)
    action(wm, "Marksman Strike (W)", 60, (200, 700), Damage { potency: 120 });
    action(bm, "Marksman Strike (B)", 60, (200, 700), Damage { potency: 120 });

    // CommandKey(18)
    action(wm, "Tactical Relocation (W)", 240, (0, 0), TacticalRelocation);
    action(bm, "Tactical Relocation (B)", 240, (0, 0), TacticalRelocation);

    // CommandKey(20)
    action(wm, "Lock and Load (W)", 120, (0, 0), LockAndLoad);
    action(bm, "Lock and Load (B)", 120, (0, 0), LockAndLoad);

    // CommandKey(24)
    action(wm, "Target Removal (W)", 240, (200, 700), Damage { potency: 240 });
    action(bm, "Target Removal (B)", 240, (200, 700), Damage { potency: 240 });

    let event_bus = rules();

    let event = Turn { prevent_default: false };
    let _ = event_bus.on_turn_start(event, &event_bus, &mut world)?;

    Ok(world)
}
