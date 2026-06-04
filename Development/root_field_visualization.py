#!/usr/bin/env python3
# ================================================================
# δ-Normalized Root Influence Field + Newton Flow
# Pure Python Version (SymPy + mpmath)
# ================================================================

import matplotlib
matplotlib.use('TkAgg', force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from matplotlib.patches import Circle

import sympy as sp
from mpmath import mp, mpc, nstr
import re
import textwrap
import os
import sys
from math import factorial

# ================================================================
# SETTINGS
# ================================================================
mp.dps = 600  # Arbitrary precision: 600 decimal places

print_digits = 20
USE_GLOBAL_SCALING = True
MAX_PLOT_RADIUS = 1e8
GRID_RESOLUTION = 400

# ================================================================
# PRECISION HELPERS
# ================================================================
def to_mpc(val):
    """Convert value to mpmath complex."""
    if isinstance(val, mpc):
        return val
    try:
        return mpc(complex(val))
    except:
        return mpc(val)

def poly_eval(coeffs_mpc, r):
    """Evaluate polynomial via Horner's method."""
    p = mpc(0)
    for c in coeffs_mpc:
        p = p * r + c
    return p

# ================================================================
# INPUT: COEFFICIENT LOADING
# ================================================================
def get_polynomial_coeffs():
    coeff_file = 'coeffs.txt'
    
    # === Try reading from file first ===
    if os.path.exists(coeff_file):
        print(f"Found '{coeff_file}'. Attempting to read coefficients...")
        try:
            with open(coeff_file, 'r') as f:
                content = f.read()
            
            lines = content.splitlines()
            clean_content = " ".join(
                line.strip() 
                for line in lines 
                if line.strip() and not line.strip().startswith('#')
            )
            tokens = clean_content.split()
            
            if not tokens:
                print(f"'{coeff_file}' is empty. Falling back to manual input.")
            else:
                pattern = re.compile(r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$')
                coeffs_mpc = []
                coeff_strings = []
                
                for t in tokens:
                    t_clean = t.lstrip("0") or "0"
                    if t_clean.startswith("."):
                        t_clean = "0" + t_clean
                    if not pattern.fullmatch(t_clean):
                        print(f"Invalid number in file: '{t}'")
                        break
                    try:
                        coeffs_mpc.append(to_mpc(t_clean))
                        coeff_strings.append(t)
                    except Exception:
                        print(f"Could not convert '{t}' from file")
                        break
                else:
                    if all(c == 0 for c in coeffs_mpc):
                        print("Polynomial cannot be identically zero.")
                    else:
                        while coeffs_mpc and coeffs_mpc[0] == 0:
                            coeffs_mpc.pop(0)
                            coeff_strings.pop(0)
                        print(f"✅ Successfully loaded {len(coeffs_mpc)} coefficients from '{coeff_file}'")
                        sys.stdout.flush()
                        return coeffs_mpc, coeff_strings
                        
        except Exception as e:
            print(f"Error reading '{coeff_file}': {e}")
            print("Falling back to manual input.")
    
    # === Manual Input (fallback) ===
    pattern = re.compile(r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$')
    while True:
        s = input("Enter polynomial coefficients (highest degree first): ").strip()
        if not s:
            print("Error: No input provided.")
            continue
            
        tokens = s.split()
        coeffs_mpc = []
        coeff_strings = []
        
        for t in tokens:
            t_clean = t.lstrip("0") or "0"
            if t_clean.startswith("."):
                t_clean = "0" + t_clean
            if not pattern.fullmatch(t_clean):
                print(f"Invalid number: '{t}'")
                break
            try:
                coeffs_mpc.append(to_mpc(t_clean))
                coeff_strings.append(t)
            except Exception:
                print(f"Could not convert '{t}'")
                break
        else:
            if all(c == 0 for c in coeffs_mpc):
                print("Polynomial cannot be identically zero.")
                continue
            while coeffs_mpc and coeffs_mpc[0] == 0:
                coeffs_mpc.pop(0)
                coeff_strings.pop(0)
            return coeffs_mpc, coeff_strings


coeffs_mpc, coeff_strings = get_polynomial_coeffs()

# ================================================================
# POLYNOMIAL CONSTRUCTION (SymPy)
# ================================================================
x = sp.Symbol('x')

# Build SymPy polynomial from coefficients
# Use sympify() to support integers, decimals, and scientific notation
coeffs_sympy = [sp.sympify(c) for c in coeff_strings]
poly_sympy = sum(coeffs_sympy[i] * x**(len(coeffs_sympy)-1-i) 
                 for i in range(len(coeffs_sympy)))

degree = len(coeffs_mpc) - 1

print(f"\nPolynomial degree: {degree}")

# ================================================================
# POLYNOMIAL STRING REPRESENTATION
# ================================================================
def simple_polynomial_string(coeff_strings, var="x"):
    if not coeff_strings:
        return "0"
    
    def fmt_coeff(c_str):
        """Format coefficient: remove trailing zeros, handle signs."""
        try:
            val = float(c_str)
            # Format with up to 8 significant figures, strip trailing zeros
            if abs(val) < 1e-6 or abs(val) > 1e6:
                # Use scientific notation for very small/large
                formatted = f"{val:.6e}".rstrip('0').rstrip('.')
            else:
                # Standard notation
                formatted = f"{val:.8f}".rstrip('0').rstrip('.')
            return formatted
        except:
            return c_str
    
    terms = []
    n = len(coeff_strings)
    
    for i, coeff in enumerate(coeff_strings):
        power = n - i - 1
        coeff_float = float(coeff)
        
        # Skip zero terms (except constant)
        if coeff_float == 0 and power > 0:
            continue
        
        sign = "-" if coeff_float < 0 else "+"
        coeff_abs = fmt_coeff(str(abs(coeff_float)))
        
        if power == 0:
            term = coeff_abs
        elif power == 1:
            term = var if coeff_abs == "1" else f"{coeff_abs}{var}"
        else:
            term = (f"{var}^{power}" if coeff_abs == "1" 
                   else f"{coeff_abs}{var}^{power}")
        
        terms.append((sign, term))
    
    if not terms:
        return "0"
    
    first_sign, first_term = terms[0]
    result = first_term if first_sign == "+" else f"-{first_term}"
    
    for sign, term in terms[1:]:
        result += f" {sign} {term}"
    
    return result

poly_str = simple_polynomial_string(coeff_strings)
print(f"\n📐 f(x) = {poly_str} = 0")

# ================================================================
# ROOT FINDING WITH MULTIPLICITY (SymPy)
# ================================================================
print("\nComputing exact roots with multiplicities...")
roots_dict = sp.roots(poly_sympy, x)

# Convert SymPy roots to (root_mpc, multiplicity) tuples
roots = []
for root_expr, mult in roots_dict.items():
    root_mpc = to_mpc(sp.N(root_expr, mp.dps))
    roots.append((root_mpc, mult))

print(f"Found {len(roots)} distinct roots\n")

# ================================================================
# DERIVATIVE TOWER (coefficient lists)
# ================================================================
def build_derivative_tower_coeffs(coeffs_mpc):
    """Build tower of derivatives as coefficient lists."""
    tower = [list(coeffs_mpc)]
    degree = len(coeffs_mpc) - 1
    
    for _ in range(degree):
        prev = tower[-1]
        m = len(prev) - 1
        
        if m <= 0:
            tower.append([mpc(0)])
            continue
        
        next_coeffs = [prev[i] * mpc(m - i) for i in range(m)]
        tower.append(next_coeffs)
    
    return tower

derivative_tower = build_derivative_tower_coeffs(coeffs_mpc)

# ================================================================
# TRIPLET CACHE: (root, multiplicity) → (root, m, δ, α)
# ================================================================
triplet_cache = {}

for root_mpc, mult in roots:
    alpha = poly_eval(derivative_tower[mult], root_mpc) / mpc(factorial(mult))
    abs_alpha = abs(alpha)
    
    delta = (None if abs_alpha == 0 
            else abs_alpha ** (-mpc(1) / mpc(mult)))
    
    triplet_cache[(root_mpc, mult)] = (root_mpc, mult, delta, alpha)

# ================================================================
# ROOT TABLE
# ================================================================
if roots:
    print("Local Asymptotic Triplets T = (a, m, δ)")
    print("-" * 120)
    
    rows = []
    for i, (root_mpc, mult) in enumerate(roots, 1):
        _, _, delta, _ = triplet_cache[(root_mpc, mult)]
        
        # Format root as real + imaginary (without complex type notation)
        root_real = float(root_mpc.real)
        root_imag = float(root_mpc.imag)
        
        if abs(root_imag) < 1e-12:
            root_str = f"{root_real:.16g}"
        else:
            root_str = f"{root_real:.16g} + {abs(root_imag):.16g}i" if root_imag >= 0 else f"{root_real:.16g} - {abs(root_imag):.16g}i"
        
        # Format delta
        if delta is None:
            delta_str = "∞"
        else:
            delta_float = float(abs(delta))
            delta_str = f"{delta_float:.16g}"
        
        # Residual
        residual_val = abs(poly_eval(coeffs_mpc, root_mpc))
        residual_float = float(residual_val)
        if residual_float < 1e-10:
            residual_str = f"{residual_float:.2e}"
        else:
            residual_str = f"{residual_float:.6g}"
        
        rows.append((i, root_str, mult, delta_str, residual_str))
    
    # Print header with proper column widths
    col_widths = [3, 30, 3, 20, 14]
    header = f"{'#':<{col_widths[0]}} {'Root a':<{col_widths[1]}} {'m':<{col_widths[2]}} {'δ':<{col_widths[3]}} {'Residual':<{col_widths[4]}}"
    print(header)
    print("-" * 120)
    
    # Print rows
    for i, root_str, mult, delta_str, residual_str in rows:
        print(f"{i:<{col_widths[0]}} {root_str:<{col_widths[1]}} {mult:<{col_widths[2]}} {delta_str:<{col_widths[3]}} {residual_str:<{col_widths[4]}}")
    
    print()

# ================================================================
# FIELD RADIUS COMPUTATION
# ================================================================
def compute_field_radius(roots_with_mult, use_global_scaling=True):
    if not roots_with_mult:
        return 1.0
    
    centroids = []
    deltas = []
    
    for r, m in roots_with_mult:
        _, _, delta, _ = triplet_cache[(r, m)]
        centroids.append(float(abs(r)))
        deltas.append(float(abs(delta)) if delta is not None else float('inf'))
    
    if use_global_scaling:
        extents = [c + d for c, d in zip(centroids, deltas) 
                  if d != float('inf')]
        
        if extents:
            R = max(extents) * 1.05
        else:
            R = float('inf')
        
        if R == float('inf') or R > MAX_PLOT_RADIUS:
            fallback = max(centroids) * 1.5
            return min(fallback, MAX_PLOT_RADIUS)
        
        return float(R)
    
    return float(max(centroids)) * 1.5

# ================================================================
# ROOT FIELD COMPUTATION
# ================================================================
def compute_root_field(roots_with_mult, N=GRID_RESOLUTION, use_global_scaling=True):
    centroids = np.array([complex(r) for r, _ in roots_with_mult], dtype=complex)
    mults = np.array([float(m) for _, m in roots_with_mult], dtype=float)
    deltas = np.array([
        np.inf if triplet_cache[(r, m)][2] is None
        else float(abs(triplet_cache[(r, m)][2]))
        for r, m in roots_with_mult
    ], dtype=float)
    
    R = compute_field_radius(roots_with_mult, use_global_scaling)
    R = float(abs(R))  # Convert mpmath types to Python float
    print(f"Plot radius R = {R:.6e}\n")
    
    xs = np.linspace(-R, R, N)
    ys = np.linspace(-R, R, N)
    
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y
    
    # Distance field
    min_dist = np.full(X.shape, np.inf, dtype=float)
    
    for a, d in zip(centroids, deltas):
        if np.isfinite(d) and d > 0:
            min_dist = np.minimum(min_dist, np.abs(Z - a) / d)
    
    dist_field = np.log10(np.clip(min_dist, 1e-30, None))
    
    # Newton flow field
    log_deriv = np.zeros(Z.shape, dtype=complex)
    EPS = 1e-30
    
    for a, m_val in zip(centroids, mults):
        dz = Z - a
        safe = np.where(np.abs(dz) < EPS, EPS + 0j, dz)
        log_deriv += m_val / safe
    
    with np.errstate(divide='ignore', invalid='ignore'):
        V = -1.0 / log_deriv
        mag = np.abs(V)
        mag_safe = np.where(mag > 0, mag, 1.0)
        flow_u = np.real(V) / mag_safe
        flow_v = np.imag(V) / mag_safe
    
    return xs, ys, dist_field, flow_u, flow_v, R

# ================================================================
# PLOTTING
# ================================================================
def truncate_polynomial(poly_str, max_len=80, keep_head=3, keep_tail=2):
    if len(poly_str) <= max_len:
        return poly_str
    
    terms = re.findall(r'[+-]?\s*[^+-]+', poly_str)
    terms = [t.strip() for t in terms if t.strip()]
    
    if len(terms) <= keep_head + keep_tail:
        return poly_str[:max_len - 3] + "..."
    
    head = terms[:keep_head]
    tail = terms[-keep_tail:]
    candidate = " ".join(head) + " + ⋯ " + " ".join(tail)
    candidate = candidate.replace("+ -", "- ")
    
    while len(candidate) > max_len and keep_tail > 1:
        keep_tail -= 1
        tail = terms[-keep_tail:]
        candidate = " + ".join(head) + " + ⋯ + " + " + ".join(tail)
        candidate = candidate.replace("+ -", "- ")
    
    while len(candidate) > max_len and keep_head > 1:
        keep_head -= 1
        head = terms[:keep_head]
        candidate = " + ".join(head) + " + ⋯ + " + " + ".join(tail)
        candidate = candidate.replace("+ -", "- ")
    
    if len(candidate) > max_len:
        candidate = candidate[:max_len - 1] + "…"
    
    return candidate

# ================================================================
# FIGURE 1: Root Field + Newton Flow
# ================================================================
if roots:
    print("Rendering Figure 1: δ-Normalized Root Influence Field + Newton Flow")
    
    xs, ys, dist, fu, fv, R = compute_root_field(roots, N=GRID_RESOLUTION, 
                                                  use_global_scaling=USE_GLOBAL_SCALING)
    
    fig, ax = plt.subplots(figsize=(12, 10.5))
    
    im = ax.imshow(dist, extent=[xs[0], xs[-1], ys[0], ys[-1]], 
                   origin='lower', cmap='viridis', aspect='equal')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                label=r'$\log_{10}(\min_i |z-a_i|/\delta_i)$')
    
    ax.streamplot(xs, ys, fu, fv, density=1.3, color='black', 
                 linewidth=0.55, arrowsize=0.9)
    
    # Plot roots and δ-circles
    for r, m in roots:
        r_complex = complex(r)
        _, _, delta, _ = triplet_cache[(r, m)]
        delta_f = float(abs(delta)) if delta is not None else np.inf
        
        if np.isfinite(delta_f) and 0 < delta_f < R * 2:
            circle = Circle((r_complex.real, r_complex.imag), delta_f,
                          fill=False, color='red', linestyle='--', 
                          linewidth=1.5, alpha=0.75)
            ax.add_patch(circle)
        
        ax.scatter(r_complex.real, r_complex.imag, color='red',
                  s=40 * m**0.65, edgecolors='black', linewidth=0.8, zorder=5)
    
    # Title
    full_equation = poly_str.strip()
    if not (full_equation.endswith("= 0") or full_equation.endswith("=0")):
        full_equation += " = 0"
    
    window_title = truncate_polynomial(full_equation, max_len=75, 
                                      keep_head=3, keep_tail=1)
    title_equation = truncate_polynomial(full_equation, max_len=180, 
                                        keep_head=5, keep_tail=2)
    wrapped_eq = textwrap.fill(title_equation, width=80)
    
    try:
        fig.canvas.manager.set_window_title(window_title)
    except:
        pass
    
    ax.set_title("δ-Normalized Root Influence Field + Newton Flow\n" + wrapped_eq,
                fontsize=11, pad=18)
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.grid(True, alpha=0.25)
    
    plt.tight_layout()
    
    # ================================================================
    # FIGURE 2: Complex Plane + Real Line
    # ================================================================
    print("Rendering Figure 2: Roots in Complex Plane + Real Line")
    
    roots_np = np.array([complex(r) for r, _ in roots], dtype=complex)
    
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Complex plane
    ax1.axhline(0, color='gray', lw=1)
    ax1.axvline(0, color='gray', lw=1)
    
    max_delta = 0.0
    for r, m in roots:
        r_complex = complex(r)
        ax1.scatter(r_complex.real, r_complex.imag, color='red', s=12*m, zorder=5)
        
        _, _, delta, _ = triplet_cache[(r, m)]
        delta_float = float(abs(delta)) if delta is not None else 0.0
        
        if delta_float > 0:
            circle = Circle((r_complex.real, r_complex.imag), delta_float,
                          fill=False, linestyle='--', linewidth=1.5,
                          edgecolor='blue', alpha=0.75)
            ax1.add_patch(circle)
            max_delta = max(max_delta, delta_float)
    
    ax1.set_title("Roots in Complex Plane\n(red size ∝ multiplicity | blue dashed = δ)")
    ax1.set_xlabel("Re")
    ax1.set_ylabel("Im")
    ax1.grid(True)
    
    if len(roots_np) > 0:
        max_root = float(max(np.abs(roots_np)))
        plot_radius = max((max_root + max_delta) * 1.28, 2.5)
        ax1.set_xlim(-plot_radius, plot_radius)
        ax1.set_ylim(-plot_radius, plot_radius)
    
    ax1.set_aspect('equal')
    
    # Real line plot
    real_roots = [(r, m) for r, m in roots if abs(r.imag) < 1e-12]
    
    xmin = (float(min(roots_np.real)) - 1.0 if roots_np.size > 0 else -5.0)
    xmax = (float(max(roots_np.real)) + 1.0 if roots_np.size > 0 else 5.0)
    
    x_list = np.linspace(xmin, xmax, 2000).tolist()
    for r, _ in real_roots:
        root_float = float(r.real)
        if not any(abs(root_float - x) < 1e-12 for x in x_list):
            x_list.append(root_float)
    
    x_list.sort()
    x_vals = np.array(x_list)
    y_vals = np.array([float(abs(poly_eval(coeffs_mpc, to_mpc(xx)))) for xx in x_vals])
    
    if np.max(np.abs(y_vals)) > 1e300:
        y_vals = np.clip(y_vals, -1e300, 1e300)
    
    ax2.plot(x_vals, y_vals, lw=1, label="f(x)")
    ax2.axhline(0, color='gray', lw=1)
    
    for r, m in real_roots:
        ax2.scatter(float(r.real), 0.0, color='red', s=10*m, zorder=5)
    
    ax2.set_title("Polynomial on Real Line")
    ax2.set_xlabel("x")
    ax2.set_ylabel("f(x)")
    ax2.grid(True)
    ax2.legend()
    
    init_ylim = ax2.get_ylim()
    
    # Interactive controls
    linthresh = 1.0
    
    def set_linear(event):
        ax2.set_yscale('linear')
        ax2.set_ylim(init_ylim)
        fig2.canvas.draw_idle()
    
    def set_symlog(event):
        ax2.set_yscale('symlog', linthresh=linthresh, linscale=1.0)
        fig2.canvas.draw_idle()
    
    axlinear = plt.axes([0.8, 0.02, 0.1, 0.04])
    axsymlog = plt.axes([0.65, 0.02, 0.1, 0.04])
    
    b_linear = Button(axlinear, 'Linear')
    b_symlog = Button(axsymlog, 'Symlog')
    
    b_linear.on_clicked(set_linear)
    b_symlog.on_clicked(set_symlog)
    
    plt.tight_layout()
    
    # ================================================================
    # DISPLAY
    # ================================================================
    print("\n✓ Visualization complete. Displaying...")
    plt.show()

else:
    print("No roots to display.")
