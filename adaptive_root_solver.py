import sys
import os
import math
import numpy as np
import mpmath as mp

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
mp.dps = 600
# ---------------------------------------------------------

# Note: this solver is a delta-guided adaptive heuristic.
# It combines float64 root scouting, clustering, and arbitrary
# precision refinement to recover triplets (a, m, δ). It is
# not a theoretically guaranteed exact polynomial solver for
# all ill-conditioned inputs.

def get_derivative_coeffs(coeffs, m=1):
    """Compute coefficients of the m-th derivative of a polynomial."""
    der_coeffs = list(coeffs)
    for _ in range(m):
        if len(der_coeffs) <= 1:
            return [mp.mpf('0')]
        deg = len(der_coeffs) - 1
        der_coeffs = [der_coeffs[i] * (deg - i) for i in range(deg)]
    return der_coeffs


def evaluate_polynomial(coeffs, z):
    return mp.polyval(coeffs, mp.mpc(z))


def evaluate_polynomial_float(coeffs, z):
    z = complex(z)
    result = 0+0j
    for c in coeffs:
        result = result * z + complex(c)
    return result


def get_derivative_coeffs_float(coeffs, m=1):
    der_coeffs = list(coeffs)
    for _ in range(m):
        if len(der_coeffs) <= 1:
            return [0.0]
        deg = len(der_coeffs) - 1
        der_coeffs = [der_coeffs[i] * (deg - i) for i in range(deg)]
    return der_coeffs


def calculate_delta(coeffs, a, m, prefer='auto'):
    """Calculate the characteristic deflection distance delta for a triplet."""
    if prefer in ('float', 'auto'):
        try:
            coeffs_float = [float(c) for c in coeffs]
            p_m = evaluate_polynomial_float(get_derivative_coeffs_float(coeffs_float, m), complex(a))
            if p_m == 0:
                return float('inf')
            alpha = p_m / math.factorial(m)
            return abs(alpha) ** (-1.0 / m)
        except Exception:
            pass

    with mp.workdps(max(80, 20 * m + 60)):
        coeffs_mp = [mp.mpc(c) for c in coeffs]
        p_m = evaluate_polynomial(get_derivative_coeffs(coeffs_mp, m), a)
        if p_m == 0:
            return mp.inf
        alpha = p_m / mp.factorial(m)
        return mp.power(1 / mp.fabs(alpha), mp.mpf(1) / m)


def refine_root_float(coeffs, a_est, m, max_iter=25, tol=1e-12):
    coeffs_float = [float(c) for c in coeffs]
    z = complex(a_est)

    if m == 1:
        f_coeffs = coeffs_float
        f_prime_coeffs = get_derivative_coeffs_float(coeffs_float, 1)
    else:
        f_coeffs = get_derivative_coeffs_float(coeffs_float, m - 1)
        f_prime_coeffs = get_derivative_coeffs_float(coeffs_float, m)

    for _ in range(max_iter):
        f_val = evaluate_polynomial_float(f_coeffs, z)
        f_prime_val = evaluate_polynomial_float(f_prime_coeffs, z)
        if abs(f_prime_val) < 1e-16:
            break
        step = f_val / f_prime_val
        z -= step
        if abs(step) < tol:
            break

    return z


def refine_root_mpmath(coeffs, a_est, m, max_iter=120, tol=mp.mpf('1e-60')):
    """Refine a root estimate using arbitrary-precision Newton iteration."""
    with mp.workdps(max(90, 25 * m + 60)):
        z = mp.mpc(a_est)
        coeffs_mp = [mp.mpc(c) for c in coeffs]

        if m == 1:
            f_coeffs = coeffs_mp
            f_prime_coeffs = get_derivative_coeffs(coeffs_mp, 1)
        else:
            f_coeffs = get_derivative_coeffs(coeffs_mp, m - 1)
            f_prime_coeffs = get_derivative_coeffs(coeffs_mp, m)

        for _ in range(max_iter):
            f_val = evaluate_polynomial(f_coeffs, z)
            f_prime_val = evaluate_polynomial(f_prime_coeffs, z)

            if mp.fabs(f_prime_val) < mp.mpf('1e-60'):
                break

            step = f_val / f_prime_val
            z -= step

            if mp.fabs(step) < tol:
                break

        return z


def adaptive_cluster_threshold(roots_np):
    """Estimate a clustering radius from the scatter of approximate float roots."""
    if len(roots_np) < 2:
        return 1e-12

    min_distances = []
    for i, root in enumerate(roots_np):
        distances = [np.abs(root - other) for j, other in enumerate(roots_np) if i != j]
        min_distances.append(min(distances))

    median_gap = np.median(min_distances)
    return max(1e-12, median_gap * 5)


def cluster_roots(roots_np):
    """Group approximate float roots into clusters without a hardcoded distance."""
    if len(roots_np) == 0:
        return []

    eps = adaptive_cluster_threshold(roots_np)
    clusters = []

    for root in roots_np:
        assigned = False
        for cluster in clusters:
            if any(np.abs(root - member) <= eps for member in cluster):
                cluster.append(root)
                assigned = True
                break
        if not assigned:
            clusters.append([root])

    clusters.sort(key=lambda c: (np.real(np.mean(c)), np.imag(np.mean(c))))
    return clusters


def choose_precision(delta_est, m, cluster_size):
    """Choose whether to use float or arbitrary precision based on δ and multiplicity."""
    if delta_est is None:
        return 'mpmath'

    delta_val = float(delta_est)
    if m == 1 and cluster_size == 1:
        if 1e-12 < delta_val < 1e2:
            return 'float'
    return 'mpmath'


def refine_root_adaptive(coeffs, a_est, m, cluster_size):
    delta_est = calculate_delta(coeffs, a_est, m, prefer='float')
    precision = choose_precision(delta_est, m, cluster_size)

    if precision == 'float':
        z = refine_root_float(coeffs, a_est, m)
        try:
            coeffs_float = [float(c) for c in coeffs]
            derivative_coeffs = get_derivative_coeffs_float(coeffs_float, m)
            residual = abs(evaluate_polynomial_float(derivative_coeffs, z))
            if residual < 1e-8:
                return z, precision
        except Exception:
            pass

    return refine_root_mpmath(coeffs, a_est, m), 'mpmath'


def plot_triplets(triplets, save_path=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. Install it with pip install matplotlib.")
        return

    xs = [t['a'].real for t in triplets]
    ys = [t['a'].imag for t in triplets]
    deltas = np.array([t['delta'] for t in triplets], dtype=float)
    delta_clamped = np.clip(deltas, 1e-6, 1e6)
    sizes = 80 + 300 * np.log10(delta_clamped)
    sizes = np.clip(sizes, 80, 1200)

    colors = [t['m'] for t in triplets]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(xs, ys, s=sizes, c=colors, cmap='plasma', alpha=0.7, edgecolors='k')

    for t in triplets:
        ax.annotate(f"m={t['m']}\nδ={t['delta']:.2g}",
                    (t['a'].real, t['a'].imag),
                    textcoords='offset points', xytext=(5, 5), fontsize=8)

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Real part')
    ax.set_ylabel('Imaginary part')
    ax.set_title('Polynomial Root Triplets')
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Multiplicity')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved triplet plot to {save_path}")
    plt.show()


def evaluate_polynomial_float_array(coeffs, z):
    coeffs_float = [float(c) for c in coeffs]
    return np.polyval(coeffs_float, z)


def plot_delta_field(triplets, coeffs, save_path=None, resolution=400, flow=True):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. Install it with pip install matplotlib.")
        return

    xs = np.array([t['a'].real for t in triplets])
    ys = np.array([t['a'].imag for t in triplets])
    deltas = np.array([t['delta'] for t in triplets], dtype=float)

    if len(xs) == 0:
        print("No triplets available for field plotting.")
        return

    x_span = max(np.ptp(xs), 1.0)
    y_span = max(np.ptp(ys), 1.0)
    margin = 1.0 + 0.5 * max(x_span, y_span)
    x_min, x_max = xs.min() - margin, xs.max() + margin
    y_min, y_max = ys.min() - margin, ys.max() + margin

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    ratios = np.stack([np.abs(Z - complex(t['a'])) / max(1e-18, float(t['delta'])) for t in triplets], axis=-1)
    field = np.min(ratios, axis=-1)
    field_log = np.log10(np.clip(field, 1e-12, 1e12))

    coeffs_float = [float(c) for c in coeffs]
    deriv_float = get_derivative_coeffs_float(coeffs_float, 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(field_log, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='viridis', aspect='auto')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'$\log_{10}(\min_i |z-a_i|/\delta_i)$')

    if flow:
        P = np.polyval(coeffs_float, Z)
        Pp = np.polyval(deriv_float, Z)
        V = np.zeros_like(P, dtype=complex)
        mask = np.abs(Pp) > 1e-14
        V[mask] = -P[mask] / Pp[mask]
        U = np.real(V)
        Vv = np.imag(V)
        mag = np.hypot(U, Vv)
        mag[mag == 0] = 1.0
        U /= mag
        Vv /= mag

        stride = max(1, resolution // 25)
        ax.quiver(X[::stride, ::stride], Y[::stride, ::stride], U[::stride, ::stride], Vv[::stride, ::stride],
                  color='white', alpha=0.7, pivot='mid', headwidth=3, headlength=4, headaxislength=3, linewidth=0.5)

    for t in triplets:
        ax.scatter(t['a'].real, t['a'].imag, c='red', s=80, edgecolors='black', zorder=5)
        circle = plt.Circle((t['a'].real, t['a'].imag), t['delta'], color='white', fill=False, linewidth=1.5, alpha=0.8)
        ax.add_patch(circle)
        ax.text(t['a'].real, t['a'].imag, f"m={t['m']}\nδ={t['delta']:.2g}", color='white', fontsize=8,
                ha='left', va='bottom', bbox=dict(facecolor='black', alpha=0.4, pad=1, edgecolor='none'))

    ax.set_xlabel('Real part')
    ax.set_ylabel('Imaginary part')
    ax.set_title('δ-Normalized Distance Field and Newton Flow')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=220, bbox_inches='tight')
        print(f"Saved field plot to {save_path}")
    plt.show()


def plot_combined(triplets, coeffs, save_path=None, resolution=320):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. Install it with pip install matplotlib.")
        return

    fig, axs = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=False)
    axs = axs.ravel()

    xs = [t['a'].real for t in triplets]
    ys = [t['a'].imag for t in triplets]
    deltas = np.array([t['delta'] for t in triplets], dtype=float)
    delta_clamped = np.clip(deltas, 1e-6, 1e6)
    sizes = 80 + 300 * np.log10(delta_clamped)
    sizes = np.clip(sizes, 80, 1200)
    colors = [t['m'] for t in triplets]

    sc = axs[0].scatter(xs, ys, s=sizes, c=colors, cmap='plasma', alpha=0.75, edgecolors='k')
    for t in triplets:
        axs[0].annotate(f"m={t['m']}\nδ={t['delta']:.2g}",
                        (t['a'].real, t['a'].imag),
                        textcoords='offset points', xytext=(5, 5), fontsize=8)
    axs[0].axhline(0, color='gray', linewidth=0.5)
    axs[0].axvline(0, color='gray', linewidth=0.5)
    axs[0].set_xlabel('Real part')
    axs[0].set_ylabel('Imaginary part')
    axs[0].set_title('Polynomial Root Triplets')
    fig.colorbar(sc, ax=axs[0], label='Multiplicity')

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    x_span = max(np.ptp(xs_arr), 1.0)
    y_span = max(np.ptp(ys_arr), 1.0)
    margin = 1.0 + 0.5 * max(x_span, y_span)
    x_min, x_max = xs_arr.min() - margin, xs_arr.max() + margin
    y_min, y_max = ys_arr.min() - margin, ys_arr.max() + margin

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    ratios = np.stack([np.abs(Z - complex(t['a'])) / max(1e-18, float(t['delta'])) for t in triplets], axis=-1)
    field = np.min(ratios, axis=-1)
    field_log = np.log10(np.clip(field, 1e-12, 1e12))

    coeffs_float = [float(c) for c in coeffs]
    deriv_float = get_derivative_coeffs_float(coeffs_float, 1)

    im = axs[1].imshow(field_log, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='viridis', aspect='auto')
    fig.colorbar(im, ax=axs[1], label=r'$\log_{10}(\min_i |z-a_i|/\delta_i)$')

    P = np.polyval(coeffs_float, Z)
    Pp = np.polyval(deriv_float, Z)
    V = np.zeros_like(P, dtype=complex)
    mask = np.abs(Pp) > 1e-14
    V[mask] = -P[mask] / Pp[mask]
    U = np.real(V)
    Vv = np.imag(V)
    mag = np.hypot(U, Vv)
    mag[mag == 0] = 1.0
    U /= mag
    Vv /= mag

    stride = max(1, resolution // 25)
    axs[1].quiver(X[::stride, ::stride], Y[::stride, ::stride], U[::stride, ::stride], Vv[::stride, ::stride],
                  color='white', alpha=0.7, pivot='mid', headwidth=3, headlength=4, headaxislength=3, linewidth=0.5)

    for t in triplets:
        axs[1].scatter(t['a'].real, t['a'].imag, c='red', s=70, edgecolors='black', zorder=5)
        circle = plt.Circle((t['a'].real, t['a'].imag), t['delta'], color='white', fill=False, linewidth=1.5, alpha=0.8)
        axs[1].add_patch(circle)
        axs[1].text(t['a'].real, t['a'].imag, f"m={t['m']}\nδ={t['delta']:.2g}", color='white', fontsize=8,
                    ha='left', va='bottom', bbox=dict(facecolor='black', alpha=0.4, pad=1, edgecolor='none'))

    axs[1].set_xlabel('Real part')
    axs[1].set_ylabel('Imaginary part')
    axs[1].set_title('δ-Normalized Distance Field + Newton Flow')
    axs[1].set_xlim(x_min, x_max)
    axs[1].set_ylim(y_min, y_max)

    plt.subplots_adjust(wspace=0.25, left=0.06, right=0.96, top=0.94, bottom=0.08)
    if save_path:
        fig.savefig(save_path, dpi=220, bbox_inches='tight')
        print(f"Saved combined plot to {save_path}")
    plt.show()


def verify_multiplicity(coeffs, a, max_m):
    """Verify a guessed multiplicity by checking the highest relevant derivative."""
    with mp.workdps(max(90, 25 * max_m + 60)):
        coeffs_mp = [mp.mpc(c) for c in coeffs]
        scale = max(mp.mpf('1'), mp.fabs(mp.mpc(a)))
        tol = mp.power(10, -mp.mp.dps / 2) * scale

        for k in range(max_m, 0, -1):
            val = mp.fabs(evaluate_polynomial(get_derivative_coeffs(coeffs_mp, k), a))
            if val > tol:
                return k

    return 1


def adaptive_root_finder(coeffs_mp):
    print("\n--- Adaptive Triplet Analysis ---")
    print("Step 1: Global float64 scout for approximate root locations...")

    coeffs_float = [float(c) for c in coeffs_mp]
    roots_raw = np.roots(coeffs_float)

    print("Step 2: Adaptive clustering of the float roots...")
    clusters = cluster_roots(roots_raw)
    print(f"Found {len(clusters)} cluster(s) from {len(roots_raw)} float64 estimates.")

    final_triplets = []
    degree = len(coeffs_mp) - 1

    for cluster in clusters:
        a_est = np.mean(cluster)
        cluster_size = len(cluster)
        print(f"\nCluster centroid {a_est} with {cluster_size} member(s).")

        a_refined, precision = refine_root_adaptive(coeffs_mp, a_est, max(1, cluster_size), cluster_size)
        multiplicity = verify_multiplicity(coeffs_mp, a_refined, cluster_size)

        if multiplicity != cluster_size:
            print(f"  Adjusted multiplicity from {cluster_size} to {multiplicity}.")
            a_refined, precision = refine_root_adaptive(coeffs_mp, a_refined, multiplicity, cluster_size)

        delta_final = calculate_delta(coeffs_mp, a_refined, multiplicity, prefer=precision)
        final_triplets.append({
            'a': complex(a_refined),
            'm': multiplicity,
            'delta': float(delta_final),
            'precision': precision
        })

    return final_triplets


def parse_input():
    coeffs = []
    if os.path.exists("coeffs.txt"):
        print("Reading coefficients from coeffs.txt...")
        with open("coeffs.txt", "r") as f:
            data = f.read().replace(',', ' ').split()
            coeffs = [mp.mpf(x) for x in data]
    else:
        print("Enter polynomial coefficients (highest degree to constant).")
        user_input = input("> ").replace(',', ' ').split()
        if not user_input:
            sys.exit(1)
        coeffs = [mp.mpf(x) for x in user_input]
    return coeffs


def print_usage():
    usage_text = '''
Usage:
  python3 untitled.py [options]

Options:
  --help, -h              Show this help message and exit
  --plot                  Display a scatter plot of root triplets
  --field                 Display the δ-normalized distance field with Newton flow
  --combined              Display both triplets and field side-by-side
  --save-plot=FILE        Save the triplet scatter plot to FILE
  --save-field=FILE       Save the δ-field plot to FILE
  --save-combined=FILE    Save the combined panel to FILE

If coeffs.txt exists in the current directory, the script reads polynomial coefficients
from that file. Otherwise it prompts for coefficients on stdin.
'''
    print(usage_text)


def format_root(z):
    real_part = z.real if np.abs(z.real) > 1e-20 else 0.0
    imag_part = z.imag if np.abs(z.imag) > 1e-20 else 0.0
    r_str = f"{real_part:.12g}"
    if imag_part != 0.0:
        sign = "+" if imag_part > 0 else "-"
        r_str += f" {sign} {abs(imag_part):.12g}j"
    return r_str


def main():
    try:
        if '--help' in sys.argv or '-h' in sys.argv:
            print_usage()
            return

        coeffs = parse_input()
        degree = len(coeffs) - 1
        print(f"\nLoaded polynomial of degree {degree}")
        print("Warning: this is a delta-guided adaptive heuristic solver, not a fully general exact solver.")
        print("Run with --help for usage and plotting options.")

        triplets = adaptive_root_finder(coeffs)
        triplets.sort(key=lambda t: (t['a'].real, t['a'].imag))

        print("=" * 85)
        print(f"{'Root (a)':<35} | {'m':<3} | {'Delta (δ)':<12} | {'Precision Used'}")
        print("=" * 85)
        for t in triplets:
            print(f"{format_root(t['a']):<35} | {t['m']:<3} | {t['delta']:<12.6f} | {t['precision']}")
        print("=" * 85)

        save_plot = None
        save_field = None
        save_combined = None
        if '--plot' in sys.argv or '--visualize' in sys.argv or any(arg.startswith('--save-plot=') for arg in sys.argv):
            for arg in sys.argv:
                if arg.startswith('--save-plot='):
                    save_plot = arg.split('=', 1)[1]
                    break
            plot_triplets(triplets, save_path=save_plot)

        if '--field' in sys.argv or '--delta-field' in sys.argv or any(arg.startswith('--save-field=') for arg in sys.argv):
            for arg in sys.argv:
                if arg.startswith('--save-field='):
                    save_field = arg.split('=', 1)[1]
                    break
            plot_delta_field(triplets, coeffs, save_path=save_field)

        if '--combined' in sys.argv or '--panel' in sys.argv or any(arg.startswith('--save-combined=') for arg in sys.argv):
            for arg in sys.argv:
                if arg.startswith('--save-combined='):
                    save_combined = arg.split('=', 1)[1]
                    break
            plot_combined(triplets, coeffs, save_path=save_combined)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
