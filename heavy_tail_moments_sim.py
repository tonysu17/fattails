"""
Demonstration: Sample Moments Growing with Sample Size
For heavy-tailed distributions with infinite theoretical moments

This script illustrates how sample moments can grow with sample size
when the corresponding theoretical moment is infinite or undefined.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def simulate_growing_moments(dist_name, distribution, sample_sizes, moment_order=1, n_simulations=100):
    """
    Simulate how sample moments evolve with increasing sample size.

    Parameters:
    - dist_name: name of the distribution
    - distribution: function that generates random samples
    - sample_sizes: array of sample sizes to test
    - moment_order: which moment to compute (1=mean, 2=variance, etc.)
    - n_simulations: number of independent simulations

    Returns:
    - Array of shape (n_simulations, len(sample_sizes)) with moment estimates
    """
    results = np.zeros((n_simulations, len(sample_sizes)))

    for sim in range(n_simulations):
        # Generate one long sample for each simulation
        samples = distribution(max(sample_sizes))

        for i, q in enumerate(sample_sizes):
            sample_subset = samples[:q]

            if moment_order == 1:
                # Sample mean
                results[sim, i] = np.mean(sample_subset)
            elif moment_order == 2:
                # Sample variance
                results[sim, i] = np.var(sample_subset, ddof=1)
            elif moment_order == 3:
                # Sample skewness
                results[sim, i] = stats.skew(sample_subset)
            elif moment_order == 4:
                # Sample kurtosis
                results[sim, i] = stats.kurtosis(sample_subset)
            else:
                # Raw k-th moment
                results[sim, i] = np.mean(sample_subset ** moment_order)

    return results


def plot_moment_evolution(sample_sizes, moment_data, title, ylabel, theoretical_value=None):
    """Plot how moments evolve with sample size."""
    plt.figure(figsize=(10, 6))

    # Plot individual simulation paths (first 5)
    for i in range(min(5, moment_data.shape[0])):
        plt.plot(sample_sizes, moment_data[i, :], alpha=0.3, linewidth=0.8)

    # Plot median across simulations
    median_path = np.median(moment_data, axis=0)
    plt.plot(sample_sizes, median_path, 'r-', linewidth=2, label='Median across simulations')

    if theoretical_value is not None and np.isfinite(theoretical_value):
        plt.axhline(y=theoretical_value, color='g', linestyle='--', linewidth=2, 
                   label=f'Theoretical value = {theoretical_value}')

    plt.xlabel('Sample Size (q)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return plt


def main():
    np.random.seed(42)

    # Sample sizes to test (log-spaced from 100 to 100,000)
    sample_sizes = np.logspace(2, 5, 50).astype(int)
    n_sims = 50

    print("=" * 80)
    print("SAMPLE MOMENTS IN HEAVY-TAILED DISTRIBUTIONS")
    print("=" * 80)

    # ========================================================================
    # Example 1: Cauchy Distribution (no mean or variance)
    # ========================================================================
    print("\n1. CAUCHY DISTRIBUTION")
    print("-" * 80)
    print("   Theoretical: Mean = UNDEFINED, Variance = UNDEFINED")
    print("   Expected: Sample mean drifts and jumps, does not stabilize\n")

    cauchy_mean = simulate_growing_moments(
        "Cauchy", 
        lambda n: stats.cauchy.rvs(size=n),
        sample_sizes,
        moment_order=1,
        n_simulations=n_sims
    )

    print(f"   Sample size q=100:     Mean ≈ {np.median(cauchy_mean[:, 0]):.2f}")
    print(f"   Sample size q=10,000:  Mean ≈ {np.median(cauchy_mean[:, 25]):.2f}")
    print(f"   Sample size q=100,000: Mean ≈ {np.median(cauchy_mean[:, -1]):.2f}")
    print(f"   → Sample mean does NOT converge\n")

    plot_moment_evolution(sample_sizes, cauchy_mean, 
                         'Cauchy Distribution: Sample Mean vs Sample Size',
                         'Sample Mean')
    plt.savefig('cauchy_mean_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # Example 2: Pareto with α = 1.5 (mean exists, variance infinite)
    # ========================================================================
    print("\n2. PARETO DISTRIBUTION (α = 1.5)")
    print("-" * 80)
    print("   Theoretical: Mean = 3.0, Variance = INFINITE")
    print("   Expected: Mean converges slowly, variance grows\n")

    alpha_pareto = 1.5
    x_m = 1.5  # scale parameter

    pareto_mean = simulate_growing_moments(
        "Pareto",
        lambda n: (np.random.pareto(alpha_pareto, n) + 1) * x_m,
        sample_sizes,
        moment_order=1,
        n_simulations=n_sims
    )

    pareto_var = simulate_growing_moments(
        "Pareto",
        lambda n: (np.random.pareto(alpha_pareto, n) + 1) * x_m,
        sample_sizes,
        moment_order=2,
        n_simulations=n_sims
    )

    print(f"   MEAN:")
    print(f"   Sample size q=100:     Mean ≈ {np.median(pareto_mean[:, 0]):.2f}")
    print(f"   Sample size q=10,000:  Mean ≈ {np.median(pareto_mean[:, 25]):.2f}")
    print(f"   Sample size q=100,000: Mean ≈ {np.median(pareto_mean[:, -1]):.2f}")
    print(f"   (Theoretical mean = 3.0)\n")

    print(f"   VARIANCE:")
    print(f"   Sample size q=100:     Var ≈ {np.median(pareto_var[:, 0]):.1f}")
    print(f"   Sample size q=10,000:  Var ≈ {np.median(pareto_var[:, 25]):.1f}")
    print(f"   Sample size q=100,000: Var ≈ {np.median(pareto_var[:, -1]):.1f}")
    print(f"   → Variance GROWS with sample size (theoretical variance = ∞)\n")

    plot_moment_evolution(sample_sizes, pareto_var,
                         'Pareto (α=1.5): Sample Variance vs Sample Size',
                         'Sample Variance')
    plt.savefig('pareto_variance_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # Example 3: Pareto with α = 0.8 (infinite mean)
    # ========================================================================
    print("\n3. PARETO DISTRIBUTION (α = 0.8)")
    print("-" * 80)
    print("   Theoretical: Mean = INFINITE")
    print("   Expected: Sample mean grows without bound\n")

    alpha_inf = 0.8
    pareto_inf_mean = simulate_growing_moments(
        "Pareto",
        lambda n: np.random.pareto(alpha_inf, n) + 1,
        sample_sizes,
        moment_order=1,
        n_simulations=n_sims
    )

    print(f"   Sample size q=100:     Mean ≈ {np.median(pareto_inf_mean[:, 0]):.1f}")
    print(f"   Sample size q=10,000:  Mean ≈ {np.median(pareto_inf_mean[:, 25]):.1f}")
    print(f"   Sample size q=100,000: Mean ≈ {np.median(pareto_inf_mean[:, -1]):.1f}")
    print(f"   → Sample mean EXPLODES as q increases\n")

    plot_moment_evolution(sample_sizes, pareto_inf_mean,
                         'Pareto (α=0.8): Sample Mean vs Sample Size (Infinite Theoretical Mean)',
                         'Sample Mean')
    plt.savefig('pareto_infinite_mean_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # Example 4: Student's t (df=3) - infinite 4th moment
    # ========================================================================
    print("\n4. STUDENT'S T DISTRIBUTION (df = 3)")
    print("-" * 80)
    print("   Theoretical: Mean = 0, Variance = 3, 4th moment = INFINITE")
    print("   Expected: Kurtosis estimate grows with sample size\n")

    student_kurt = simulate_growing_moments(
        "Student-t",
        lambda n: stats.t.rvs(df=3, size=n),
        sample_sizes,
        moment_order=4,
        n_simulations=n_sims
    )

    print(f"   Sample size q=100:     Kurtosis ≈ {np.median(student_kurt[:, 0]):.1f}")
    print(f"   Sample size q=10,000:  Kurtosis ≈ {np.median(student_kurt[:, 25]):.1f}")
    print(f"   Sample size q=100,000: Kurtosis ≈ {np.median(student_kurt[:, -1]):.1f}")
    print(f"   → Kurtosis GROWS (theoretical 4th moment = ∞)\n")

    plot_moment_evolution(sample_sizes, student_kurt,
                         "Student's t (df=3): Sample Kurtosis vs Sample Size",
                         'Sample Kurtosis')
    plt.savefig('student_t_kurtosis_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
When a distribution has an infinite k-th theoretical moment:

1. The sample k-th moment does NOT converge to a finite limit
2. Instead, it tends to GROW as sample size increases
3. Rare extreme values dominate the calculation
4. Longer samples include more extreme observations
5. This instability makes moment-based inference unreliable

Applications:
• Financial returns: VaR and tail risk estimates may be unstable
• Wealth/income inequality: High-moment measures drift with data
• Network degree distributions: Variance estimates can explode
• Natural phenomena with power-law behavior

This behavior is described in statistical references as a fundamental
property of distributions with heavy tails and infinite moments.
""")
    print("=" * 80)
    print("\nPlots saved:")
    print("  - cauchy_mean_evolution.png")
    print("  - pareto_variance_evolution.png")
    print("  - pareto_infinite_mean_evolution.png")
    print("  - student_t_kurtosis_evolution.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
