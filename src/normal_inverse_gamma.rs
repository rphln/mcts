use std::ops::Neg;

/// Prior hyperparameters for a Normal-Inverse-Gamma conjugate model.
///
/// Together these four values specify the prior distribution over the unknown
/// mean μ and variance σ² of a Gaussian likelihood:
///
/// - μ | σ² ~ Normal(μ₀, σ²/κ₀)
/// - σ²      ~ InverseGamma(α₀, β₀)
///
/// `kappa` (κ₀) acts as an effective prior sample count for the mean;
/// larger values reduce posterior uncertainty more aggressively as
/// data accumulate. `alpha` (α₀) and `beta` (β₀) parameterise the
/// prior over the variance.
#[derive(Copy, Clone, Debug)]
pub struct NormalInverseGammaPriors {
    /// Prior mean μ₀.
    pub mean: f64,
    /// Prior rate parameter β₀; larger values correspond to a prior belief in
    /// higher variance.
    pub beta: f64,
    /// Prior shape parameter α₀; larger values concentrate the prior over the
    /// variance more tightly.
    pub alpha: f64,
    /// Prior mean confidence κ₀.
    pub kappa: f64,
}

/// Online Normal-Inverse-Gamma posterior entry.
///
/// Stores the two posterior parameters that require accumulation, the posterior
/// mean μₙ and the rate parameter βₙ, together with the sample count n. The
/// remaining parameters κₙ = κ₀ + n and αₙ = α₀ + n/2 are reconstructed on
/// demand from the priors.
///
/// All mutating methods require a reference to the [`NormalInverseGammaPriors`]
/// with which this entry was initialised. Mixing priors across calls produces
/// undefined statistical results.
#[derive(Copy, Clone, Debug)]
pub struct NormalInverseGamma {
    /// Posterior mean μₙ.
    pub mean: f64,
    /// Posterior rate parameter βₙ.
    pub beta: f64,
    /// Number of incorporated samples n.
    pub visits: u32,
}

impl NormalInverseGamma {
    /// Creates an empty posterior initialised to the given prior.
    #[must_use]
    pub const fn new(priors: &NormalInverseGammaPriors) -> NormalInverseGamma {
        Self { mean: priors.mean, beta: priors.beta, visits: 0 }
    }

    /// Incorporates a single sample into the posterior.
    pub fn update(&mut self, sample: f64, priors: &NormalInverseGammaPriors) {
        let kappa = priors.kappa + f64::from(self.visits);
        self.beta += 0.5 * (kappa / (kappa + 1.0)) * (sample - self.mean).powi(2);
        self.mean += (sample - self.mean) / (kappa + 1.0);
        self.visits += 1;
    }

    /// Merges another posterior into this one.
    ///
    /// The result is statistically equivalent to a single posterior that
    /// had incorporated all samples from both entries, starting from the
    /// same prior.
    ///
    /// Both entries must share the same prior hyperparameters.
    pub fn merge(&mut self, other: &Self, priors: &NormalInverseGammaPriors) {
        if other.visits == 0 {
            return;
        }

        if self.visits == 0 {
            *self = *other;
            return;
        }

        let n1 = f64::from(self.visits);
        let n2 = f64::from(other.visits);
        let n = n1 + n2;

        let k1 = priors.kappa + n1;
        let k2 = priors.kappa + n2;
        let k = priors.kappa + n;

        // Recover sample means from posterior means via
        // κₙμₙ = κ₀μ₀ + n·x̄  =>  x̄ = (κₙμₙ − κ₀μ₀) / n.
        let xbar1 = (k1 * self.mean - priors.kappa * priors.mean) / n1;
        let xbar2 = (k2 * other.mean - priors.kappa * priors.mean) / n2;
        let xbar = (n1 * xbar1 + n2 * xbar2) / n;

        // Recover within-group sums of squares from posterior betas via
        // βₙ = β₀ + S/2 + κ₀n/(2κₙ)·(x̄−μ₀)²  =>  S = 2(βₙ−β₀) − κ₀n/κₙ·(x̄−μ₀)².
        let s1 = 2.0 * (self.beta - priors.beta)
            - priors.kappa * n1 / k1 * (xbar1 - priors.mean).powi(2);
        let s2 = 2.0 * (other.beta - priors.beta)
            - priors.kappa * n2 / k2 * (xbar2 - priors.mean).powi(2);

        // Combined sum of squares via the parallel Welford identity:
        // S₁₂ = S₁ + S₂ + n₁n₂/n · (x̄₁ − x̄₂)².
        let s = s1 + s2 + n1 * n2 / n * (xbar1 - xbar2).powi(2);

        self.mean = (priors.kappa * priors.mean + n * xbar) / k;
        self.beta = priors.beta
            + 0.5 * s
            + priors.kappa * n / (2.0 * k) * (xbar - priors.mean).powi(2);
        self.visits += other.visits;
    }

    /// Removes another posterior from this one.
    ///
    /// The result is statistically equivalent to the posterior that would
    /// have been obtained by incorporating only the samples *not* present
    /// in `other`, starting from the same prior.
    ///
    /// Both entries must share the same prior hyperparameters.
    ///
    /// # Panics
    ///
    /// Panics if `other.visits` exceeds `self.visits`.
    pub fn remove(&mut self, other: &Self, priors: &NormalInverseGammaPriors) {
        if other.visits == 0 {
            return;
        }

        assert!(other.visits <= self.visits, "cannot remove more samples than present");

        if other.visits == self.visits {
            // Removing all samples restores the prior exactly.
            self.mean = priors.mean;
            self.beta = priors.beta;
            self.visits = 0;
            return;
        }

        let n_total = f64::from(self.visits);
        let n_remove = f64::from(other.visits);
        let n_remain = n_total - n_remove;

        let k_total = priors.kappa + n_total;
        let k_remove = priors.kappa + n_remove;
        let k_remain = priors.kappa + n_remain;

        // Recover sample means from posterior means.
        let xbar_total = (k_total * self.mean - priors.kappa * priors.mean) / n_total;
        let xbar_remove =
            (k_remove * other.mean - priors.kappa * priors.mean) / n_remove;

        // Invert n_total·x̄_total = n_remain·x̄_remain + n_remove·x̄_remove.
        let xbar_remain = (n_total * xbar_total - n_remove * xbar_remove) / n_remain;

        // Recover sums of squares from posterior betas.
        let s_total = 2.0 * (self.beta - priors.beta)
            - priors.kappa * n_total / k_total * (xbar_total - priors.mean).powi(2);
        let s_remove = 2.0 * (other.beta - priors.beta)
            - priors.kappa * n_remove / k_remove * (xbar_remove - priors.mean).powi(2);

        // Invert the parallel Welford identity.
        let s_remain = s_total
            - s_remove
            - n_remain * n_remove / n_total * (xbar_remain - xbar_remove).powi(2);

        self.mean = (priors.kappa * priors.mean + n_remain * xbar_remain) / k_remain;
        self.beta = priors.beta
            + 0.5 * s_remain
            + priors.kappa * n_remain / (2.0 * k_remain)
                * (xbar_remain - priors.mean).powi(2);
        self.visits -= other.visits;
    }

    /// Returns the posterior standard deviation of the mean.
    #[inline]
    #[must_use]
    pub fn stderr(&self, priors: &NormalInverseGammaPriors) -> f64 {
        let kappa = priors.kappa + f64::from(self.visits);
        let alpha = priors.alpha + 0.5 * f64::from(self.visits);

        f64::sqrt(self.beta / (alpha * kappa))
    }

    /// Returns `mean + z * stderr`: an upper confidence bound when `z > 0`,
    /// or a lower confidence bound when `z < 0`.
    #[inline]
    #[must_use]
    pub fn ucb(&self, z: f64, priors: &NormalInverseGammaPriors) -> f64 {
        self.mean + z * self.stderr(priors)
    }
}

impl Neg for NormalInverseGamma {
    type Output = NormalInverseGamma;

    /// Negates the mean while preserving the variance, reflecting a switch in
    /// the moving player's perspective. Used when transferring statistics to a
    /// forced-move child node.
    fn neg(self) -> Self::Output {
        Self { mean: -self.mean, beta: self.beta, visits: self.visits }
    }
}
