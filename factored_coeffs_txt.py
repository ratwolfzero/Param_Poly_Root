import sympy as sp
from typing import List


def expand_to_coefficients(factorized_expr, variable='x') -> List[int]:
    """
    Expands a factorized polynomial and returns the list of coefficients
    from highest degree to constant term.
    Example input: (x - 5)**10 * (x**2 - 2*x + 2)**25
    """
    x = sp.symbols(variable)
    expanded = sp.expand(factorized_expr)
    poly = sp.Poly(expanded, x)
    coeffs = [int(c) for c in poly.all_coeffs()]
    return coeffs


# ============================
# Example Usage
# ============================
if __name__ == "__main__":
    x = sp.symbols('x')

    factored = (x**2 - 2*x + 2)**25 * (x - 5)**10

    coeffs = expand_to_coefficients(factored)
    print(f"Degree: {len(coeffs) - 1}")
    print("Coefficients:")
    print(coeffs)

    with open("coeffs.txt", "w") as f:
        f.write(" ".join(map(str, coeffs)))
    print("\nCoefficients saved to coeffs.txt")


# ===========================================================================
# (x - 1)^20
# ---------------------------------------------------------------------------
# Degree 20, single real root at x = 1 with multiplicity 20.
# Root splitting scales as |Da|^(1/20): at float64 the entire cluster
# drifts off the real axis. Coefficients are binomial C(20,k), so moderate
# scaling; the sole difficulty is pure multiplicity. Clean baseline for the
# precision-vs-multiplicity trade-off; needs mp.dps ~ 400 for all imaginary
# parts to snap to zero.
# ===========================================================================
# factored = (x - 1)**20


# ===========================================================================
# (x - 3)^50
# ---------------------------------------------------------------------------
# Degree 50, single real root at x = 3 with multiplicity 50.
# Double difficulty: multiplicity-50 splitting (|Da|^(1/50) ~ 10^-2 at
# float64) AND extreme coefficient scaling from 3^50 ~ 7e23 in the constant
# term -- both independently warrant WARNINGs. The hardest single-root case;
# realistically needs mp.dps > 1000 before all 50 imaginary parts snap to
# zero.
# ===========================================================================
# factored = (x - 3)**50


# ===========================================================================
# (x - 1)^20 * (x + 2)^15 * (x^2 + 1)^10
# ---------------------------------------------------------------------------
# Degree 55, three structurally different root types in one polynomial.
# x = 1  (real, mult 20): cluster splits toward both sides of the real axis.
# x = -2 (real, mult 15): cluster shifts along the real axis.
# x = +/-i (complex pair, mult 10 each): cluster noise appears in both real
# and imaginary parts simultaneously. Coefficient span from combining roots
# at 1, -2, and the unit circle causes moderate scaling. Good all-in-one
# integration test that exercises all three ill-conditioning mechanisms at
# once.
# ===========================================================================
# factored = (x - 1)**20 * (x + 2)**15 * (x**2 + 1)**10


# ===========================================================================
# (x^2 - 1)^30
# ---------------------------------------------------------------------------
# Equivalent to (x - 1)^30 * (x + 1)^30 -- degree 60, two real clusters at
# +/-1, each multiplicity 30, perfectly symmetric about the origin. Purely
# even polynomial: all odd-degree coefficients vanish, giving the companion
# matrix a checkerboard sparsity. The two clusters do not interact but within
# each cluster 30 eigenvalues collapse to one point (|Da|^(1/30)).
# Alternating-sign binomial coefficients produce near-cancellation during
# evaluation, inflating residuals independently of actual root error.
# ===========================================================================
# factored = (x**2 - 1)**30


# ===========================================================================
# (x^2 - 2x + 2)^25 * (x - 5)^10
# ---------------------------------------------------------------------------
# Degree 60, two root groups with different characters.
# x = 1+/-i (complex, |root| = sqrt(2), mult 25 each): the non-unit modulus
# compounds splitting -- absolute errors scale with both |Da|^(1/25) and
# sqrt(2)^25 ~ 180, amplifying the cluster noticeably.
# x = 5 (real, mult 10): moderately ill-conditioned, overshadowed by the
# complex cluster. Coefficient magnitudes dominated by 5^10 ~ 1e7; moderate
# scaling NOTICE. At equal precision the complex cluster (mult 25, off-axis)
# will be visibly larger than the real cluster (mult 10).
# ===========================================================================
# factored = (x**2 - 2*x + 2)**25 * (x - 5)**10


# ===========================================================================
# (x - 1)^40 * (x - 2)^30 * (x - 3)^20
# ---------------------------------------------------------------------------
# Degree 90, the largest polynomial here; three real clusters with steeply
# descending multiplicities. Splitting radii: |Da|^(1/40) at x=1,
# |Da|^(1/30) at x=2, |Da|^(1/20) at x=3 -- so paradoxically the highest-
# multiplicity cluster at x=1 is the tightest and most sensitive. Constant
# term 1^40 * 2^30 * 3^20 ~ 3.5e23 triggers extreme-scaling WARNING.
# At mp.dps = 100 only the x=3 cluster (mult 20) may snap fully to real;
# x=1 and x=2 need mp.dps > 500. Hardest case overall: high degree, extreme
# scaling, and three simultaneously active ill-conditioned clusters.
# ===========================================================================
# factored = (x - 1)**40 * (x - 2)**30 * (x - 3)**20