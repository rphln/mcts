use std::cmp::{max, min};

use swift_swallow::{
    command::Command,
    content::game,
    rules::{decide, query_actions, start},
    team::TeamKey,
    world::World,
};

pub type Move = Command;
pub type Color = TeamKey;

#[derive(Clone)]
pub struct Game {
    pub world: World,
    /// Fake turn counter to make the game alternate between sides on every
    /// move without the underlying [`world::World`] being aware of it.
    pub color: Color,
}

impl Game {
    #[must_use]
    pub fn new(mut world: World, color: Color) -> Self {
        let rules = game();
        start(&rules, &mut world).unwrap();

        Self { world, color }
    }

    #[must_use]
    pub fn is_over(&self) -> bool {
        self.world.is_game_over()
    }

    #[must_use]
    pub fn moves(&self) -> Vec<Move> {
        if self.world.active_team() != self.color {
            return vec![Move::None { team: self.color }];
        }

        let rules = game();
        query_actions(&rules, &self.world).unwrap()
    }

    pub fn play(&mut self, mov: Move) {
        let rules = game();
        let _decide = decide(mov, &rules, &mut self.world).unwrap();

        self.color = !self.color;
    }

    #[must_use]
    pub fn evaluate(&self) -> f64 {
        /// How much health a point of energy is worth, on average.
        const K: f64 = 8.;

        let mut max_score = 0.;
        let mut min_score = 0.;

        for character in &self.world.characters {
            if character.is_defeated() {
                continue;
            }

            let turn_health = f64::from({
                let health = character.current_health();

                let block = character.block;
                let poison = max(character.markers.poison_until - self.world.round, 0);

                let turn_health = min(health + block, health - poison);
                max(turn_health, 0)
            });

            let turn_energy = f64::from(min(
                character.maximum_action_points,
                character.current_action_points + 4,
            ));

            let points = (turn_health + K * turn_energy).sqrt();

            if character.team == self.color {
                max_score += points;
            } else {
                min_score += points;
            }
        }

        max_score - min_score
    }
}
