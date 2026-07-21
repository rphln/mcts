/// Aggregated statistics for a move.
#[derive(Clone, Default, Debug)]
pub struct History {
    /// Number of visits.
    pub visits: u32,
    /// Estimated reward.
    pub value: f64,
}

impl History {
    /// Incorporates a batch of samples with the given mean into the history.
    pub(crate) fn insert_batch(&mut self, mean: f64, num_samples: u32) {
        if num_samples == 0 {
            return;
        }

        let n = f64::from(self.visits);
        let k = f64::from(num_samples);

        self.value += (k / (n + k)) * (mean - self.value);
        self.visits += num_samples;
    }

    /// Removes a batch of samples with the given mean from the history.
    ///
    /// # Panics
    ///
    /// Panics if `num_samples` exceeds `visits`.
    pub(crate) fn remove_batch(&mut self, mean: f64, num_samples: u32) {
        if num_samples == 0 {
            return;
        }

        assert!(num_samples <= self.visits, "cannot remove more samples than present");

        if num_samples == self.visits {
            self.visits = 0;
            self.value = 0.0;

            return;
        }

        let n = f64::from(self.visits);
        let k = f64::from(num_samples);

        self.value -= (k / (n - k)) * (mean - self.value);
        self.visits -= num_samples;
    }
}
