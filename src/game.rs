use std::cmp;

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
        let mut max = 0.;
        let mut min = 0.;

        for character in &self.world.characters {
            let health = f64::from(character.current_effective_health());
            let energy = f64::from(cmp::min(
                character.maximum_action_points,
                character.current_action_points + 4,
            ));

            let points = (64. * health + 8. * energy).sqrt();

            if character.team == self.color {
                max += points;
            } else {
                min += points;
            }
        }

        max - min
    }
}
