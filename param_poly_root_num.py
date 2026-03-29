import numpy as np
from math import factorial
import matplotlib.pyplot as plt

def analyze_root_numeric(coeffs, a, m):
    """
    coeffs: polynomial coefficients (highest degree first)
    """
    p = np.poly1d(coeffs)
    
    dp = p
    for _ in range(m):
        dp = np.polyder(dp)
    
    alpha = dp(a) / factorial(m)
    delta = abs(alpha)**(-1.0 / m)
    
    return a, m, delta, alpha


def plot_local_numeric(p, a, m, delta):
    x_vals = np.linspace(a - 1.5*delta, a + 1.5*delta, 500)
    y_vals = p(x_vals)
    
    t_vals = (x_vals - a) / delta
    y_norm = p(a + delta * t_vals)
    
    plt.figure(figsize=(6,4))
    plt.plot(t_vals, y_norm, label="Normalized P")
    plt.plot(t_vals, t_vals**m, linestyle='--', label="t^m")
    plt.legend()
    plt.title(f"Normalized root at a={a}")
    plt.axhline(0)
    plt.show()


# Define polynomial via its roots, matching x^7*(x-1)^3*(x+1)^3*(x-3)^5*(x+3)^5
roots_list = (
    [0]*7 + [1]*3 + [-1]*3 + [3]*5 + [-3]*5
)
coeffs = np.poly(roots_list)  # returns highest-degree-first coefficients
p = np.poly1d(coeffs)

roots_to_analyze = [(0, 7), (3, 5)]

print(f"{'a':<5} {'m':<5} {'alpha':<15} {'delta':<15}")
print("-" * 45)

for a, m in roots_to_analyze:
    a_val, m_val, delta, alpha = analyze_root_numeric(coeffs, a, m)
    print(f"{a_val:<5} {m_val:<5} {alpha:<15.6e} {delta:<15.6f}")
    plot_local_numeric(p, a_val, m_val, delta)


