import numpy as np
from math import factorial
import matplotlib.pyplot as plt

def analyze_root_numeric(coeffs, a, m):
    """Pure numerical triplet computation (Version B from the paper)."""
    p = np.poly1d(coeffs)
    dp = p
    for _ in range(m):
        dp = np.polyder(dp)
    alpha = dp(a) / factorial(m)
    delta = abs(alpha) ** (-1.0 / m)
    return a, m, delta, alpha

def plot_local_numeric(p, a, m, delta, title=""):
    """Plot original and normalized local behavior."""
    x_vals = np.linspace(a - 1.5 * delta, a + 1.5 * delta, 500)
    y_vals = p(x_vals)

    t_vals = (x_vals - a) / delta
    y_norm = p(a + delta * t_vals)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original coordinates
    axes[0].plot(x_vals, y_vals)
    axes[0].axvline(a, linestyle='--', color='gray')
    axes[0].axhline(0, color='gray')
    axes[0].set_title(f"{title}\nOriginal (δ = {delta:.6f})")
    axes[0].grid(True)

    # Normalized coordinates
    axes[1].plot(t_vals, y_norm, label="Normalized P")
    axes[1].plot(t_vals, t_vals ** m, '--', label=f"t^{m}")
    axes[1].axhline(0, color='gray')
    axes[1].set_title("Canonical normalization: P(a + δt) ∼ t^m")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

# Build the exact polynomial from its factored form
p7   = np.poly1d([1, 0]) ** 7          # x^7
p3_1 = np.poly1d([1, -1]) ** 3         # (x-1)^3
p3_m1= np.poly1d([1,  1]) ** 3         # (x+1)^3
p5_3 = np.poly1d([1, -3]) ** 5         # (x-3)^5
p5_m3= np.poly1d([1,  3]) ** 5         # (x+3)^5

poly = p7 * p3_1 * p3_m1 * p5_3 * p5_m3
coeffs = poly.coefficients   # highest degree first

print("Polynomial degree:", len(coeffs)-1)
print("Leading coefficient:", coeffs[0])

# Compute triplets
roots_to_check = [(0.0, 7), (3.0, 5)]

print("\nTriplet results:")
print(f"{'a':<8} {'m':<4} {'alpha':<20} {'delta':<15}")
print("-" * 55)
for a, m in roots_to_check:
    a_val, m_val, delta, alpha = analyze_root_numeric(coeffs, a, m)
    print(f"{a_val:<8} {m_val:<4} {alpha:<20.6e} {delta:<15.6f}")
    plot_local_numeric(poly, a_val, m_val, delta,
                       title=f"Root at a = {a_val}, multiplicity m = {m_val}")
