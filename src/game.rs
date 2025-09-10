pub use swift_swallow::{command::Command, team::TeamKey as Color};
use swift_swallow::{
    content::game,
    rules::{decide, query_actions, start},
    world::World,
};

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
    pub fn moves(&self) -> Vec<Command> {
        if self.world.active_team() != self.color {
            return vec![Command::None { team: self.color }];
        }

        let rules = game();
        query_actions(&rules, &self.world).unwrap()
    }

    pub fn play(&mut self, mov: Command) {
        let rules = game();
        let _decide = decide(mov, &rules, &mut self.world).unwrap();

        self.color = !self.color;
    }

    #[must_use]
    pub fn evaluate_i32(&self) -> i32 {
        let mut max = 0;
        let mut min = 0;

        for character in &self.world.characters {
            let health = character.current_effective_health();
            let points = (64 * health).isqrt();

            if character.team == self.color {
                max += points;
            } else {
                min += points;
            }
        }

        max - min
    }

    #[must_use]
    pub fn evaluate_f64(&self) -> f64 {
        let mut max = 0.;
        let mut min = 0.;

        for character in &self.world.characters {
            let health = f64::from(character.current_effective_health());
            let points = health.ln_1p();

            if character.team == self.color {
                max += points;
            } else {
                min += points;
            }
        }

        max - min
    }
}
