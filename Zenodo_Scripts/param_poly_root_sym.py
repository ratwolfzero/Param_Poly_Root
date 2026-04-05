import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def analyze_root_symbolic(P_expr, x, a, m):
    """
    Compute (a, m, delta, alpha) using exact derivatives.
    """
    Pm = sp.diff(P_expr, x, m)
    alpha = Pm.subs(x, a) / sp.factorial(m)
    alpha = float(alpha)
    
    delta = abs(alpha)**(-1.0 / m)
    
    return a, m, delta, alpha


def plot_local_behavior(P_func, a, m, delta, title):
    x_vals = np.linspace(a - 1.5*delta, a + 1.5*delta, 500)
    y_vals = P_func(x_vals)
    
    t_vals = (x_vals - a) / delta
    y_norm = P_func(a + delta * t_vals)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original
    axes[0].plot(x_vals, y_vals)
    axes[0].axvline(a, linestyle='--')
    axes[0].axhline(0)
    axes[0].set_title(f"{title}\nOriginal (delta={delta:.4f})")
    
    # Normalized
    axes[1].plot(t_vals, y_norm, label="Normalized P")
    axes[1].plot(t_vals, t_vals**m, linestyle='--', label="t^m")
    axes[1].axhline(0)
    axes[1].set_title("Normalized coordinates")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


# Polynomial definition
x = sp.symbols('x')
P_expr = (x**7)*(x-1)**3*(x+1)**3*(x-3)**5*(x+3)**5
P_func = sp.lambdify(x, P_expr, 'numpy')

roots = [(0,7), (3,5)]

print(f"{'a':<5} {'m':<5} {'alpha':<15} {'delta':<15}")
print("-"*45)

for a, m in roots:
    a_val, m_val, delta, alpha = analyze_root_symbolic(P_expr, x, a, m)
    print(f"{a_val:<5} {m_val:<5} {alpha:<15.6e} {delta:<15.6f}")
    
    plot_local_behavior(P_func, a_val, m_val, delta,
                        title=f"Root at a={a_val}, m={m_val}")
