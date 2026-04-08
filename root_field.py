import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, mpc, mpf, matrix, eig
import textwrap

# ========================= SETTINGS ========================= #
# set arbitrary precision
mp.dps = 600

# ------------------------------------------------------------------
# CLUSTERING & FLOAT64 SAFETY (linked automatically)
# ------------------------------------------------------------------
# CLUSTER_TOL is the SINGLE most important numerical parameter.
#
# It defines when roots are considered part of the same cluster.
#
# --- NUMERICAL LIMIT (VERY IMPORTANT) ---
# The smallest *meaningful* separation between roots is limited by:
#
#     min separation ≈ 10^(-mp.dps / m)
#
# where:
#   mp.dps = working precision
#   m      = root multiplicity (unknown beforehand)
#
# Example:
#   dps = 600, m = 20  →  ~1e-30
#
# This matches observed behavior: clustering becomes unstable below ~1e-29.
#
# --- PRACTICAL CONSEQUENCES ---
# • If CLUSTER_TOL is TOO SMALL:
#     - true multiple roots split into artificial sub-clusters
#     - δ computation becomes unstable (can overflow → inf)
#     - may lead to NaN / crashes in field computation
#
# • If CLUSTER_TOL is TOO LARGE:
#     - distinct roots may merge incorrectly
#
# --- TUNING GUIDELINE ---
# The formula
#
#     noise scale ≈ 10^(-mp.dps / m)
#
# gives an estimate of the *minimum meaningful separation*
# (numerical noise floor).
#
# IMPORTANT:
#   CLUSTER_TOL must be ABOVE this scale.
#
# If CLUSTER_TOL is set BELOW the noise floor:
#   - multiple roots split into artificial clusters
#   - δ becomes unstable (can overflow)
#   - may lead to NaN / crashes
#
# In practice, choose CLUSTER_TOL somewhat ABOVE the noise floor.
#
# Examples (dps = 600):
#   m = 20 → noise ~1e-30 → usable CLUSTER_TOL ≈ 1e-29 … 1e-20
#   m = 30 → noise ~1e-20 → usable CLUSTER_TOL ≈ 1e-19 … 1e-10
#   m = 50 → noise ~1e-12 → usable CLUSTER_TOL ≈ 1e-11 … 1e-6
#
# So with increasing multiplicity:
#   → the noise floor increases
#   → the MINIMUM safe CLUSTER_TOL increases
#
# (even though numerically the tolerance value itself becomes larger)

# ------------------------------------------------------------------
# FLOAT64 SAFETY LINK
# ------------------------------------------------------------------
# FLOAT64_SAFE_REL_THRESHOLD is derived from CLUSTER_TOL.
#
# It controls when 'auto' mode switches from float64 → mpmath:
#
#   float64 is used if:
#       separation > CLUSTER_TOL × 1e7
#
#   mpmath is used if:
#       CLUSTER_TOL < separation ≤ CLUSTER_TOL × 1e7
#
# This ensures:
#   - float64 is used when clusters are clearly separated
#   - mpmath is used when clusters are distinct but too close for double precision
#
CLUSTER_TOL = mpf('1e-29')
FLOAT64_SAFE_REL_THRESHOLD = CLUSTER_TOL * mpf('1e7')

# ------------------------------------------------------------------
# COMPUTATION MODE
# ------------------------------------------------------------------
# 'fast'  : always uses float64 (fastest, may lose precision on very close clusters)
# 'auto'  : uses float64 by default, falls back to mpmath ONLY when needed
#           (recommended: balances speed and numerical stability)
MODE = 'auto'

# ------------------------------------------------------------------
# GRID RESOLUTION
# ------------------------------------------------------------------
# Number of points per axis for field computation.
#
# Note:
# - float64 path uses full resolution
# - mpmath path is automatically capped at 200×200 for performance
#
GRID_RESOLUTION = 800

# ------------------------------------------------------------------
# SCALING MODE
# ------------------------------------------------------------------
# True  = global scaling (includes largest δ, shows full field structure)
# False = root-focused scaling (zooms around roots, ignores large δ influence)
#
USE_GLOBAL_SCALING = True

# ========================= INPUT ========================= #


def parse_coefficients_strict(text):
    coeffs = []
    tokens = text.strip().split()
    if not tokens:
        raise ValueError("Empty input.")
    for token in tokens:
        original = token
        token = token.replace('I', 'i').replace('i', 'j')
        token = token.replace(' ', '')
        if token in ('j', '+j'):
            coeffs.append(mpc(0, 1))
            continue
        elif token == '-j':
            coeffs.append(mpc(0, -1))
            continue
        try:
            z = complex(token)
            coeffs.append(mpc(mpf(z.real), mpf(z.imag)))
        except Exception:
            raise ValueError(f"Invalid coefficient: '{original}'")
    while len(coeffs) > 1 and coeffs[0] == 0:
        coeffs.pop(0)
    return coeffs


def get_coefficients_from_user():
    print("\nEnter polynomial coefficients (highest degree first).")
    print("Separate coefficients by spaces or commas.")
    print("IMPORTANT: Do not put spaces inside complex numbers (e.g., use 1+2i, not 1 + 2i).")
    print("Supports: i, I, j, complex numbers\n")
    while True:
        text = input("> ").replace(',', ' ')
        try:
            coeffs = parse_coefficients_strict(text)
            if len(coeffs) < 2:
                raise ValueError("Polynomial degree must be at least 1.")
            if all(c == 0 for c in coeffs):
                raise ValueError("Polynomial cannot be all zeros.")
            return coeffs
        except ValueError as e:
            print(f"\n❌ Input error: {e}")
            print("Please try again.\n")

# ========================= POLYNOMIAL FORMAT ========================= #


def is_real_polynomial(coeffs):
    return all(mp.im(c) == 0 for c in coeffs)


def format_real(r, precision=6):
    if mp.almosteq(r, mp.nint(r)):
        return str(int(mp.nint(r)))
    else:
        return mp.nstr(r, precision)


def format_complex(z, precision=6):
    re = mp.re(z)
    im = mp.im(z)
    re_str = format_real(re, precision)
    im_str = format_real(im, precision)
    if im == 0:
        return re_str
    if re == 0:
        if im_str == "1":
            return "i"
        elif im_str == "-1":
            return "-i"
        return f"{im_str}i"
    sign = "+" if im > 0 else ""
    if im_str == "1":
        im_part = "+i"
    elif im_str == "-1":
        im_part = "-i"
    else:
        im_part = f"{sign}{im_str}i"
    return f"({re_str}{im_part})"


def polynomial_to_string(coeffs, var='z', precision=6):
    terms = []
    n = len(coeffs) - 1
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        power = n - i
        coeff_str = format_complex(c, precision)
        if power > 0:
            if coeff_str == "1":
                coeff_str = ""
            elif coeff_str == "-1":
                coeff_str = "-"
            if power == 1:
                term = f"{coeff_str}{var}"
            else:
                term = f"{coeff_str}{var}^{power}"
        else:
            term = f"{coeff_str}"
        terms.append(term)
    if not terms:
        return "0"
    poly = terms[0]
    for t in terms[1:]:
        if t.startswith('-'):
            poly += " - " + t[1:]
        else:
            poly += " + " + t
    return poly

# ========================= POLYNOMIAL ========================= #


def poly_eval(coeffs, x):
    p = mpc(0)
    for c in coeffs:
        p = p * x + c
    return p


def poly_derivative(coeffs):
    n = len(coeffs) - 1
    return [coeffs[i] * (n - i) for i in range(len(coeffs)-1)]

# ========================= COMPANION ROOT SOLVER ========================= #


def build_companion(coeffs):
    a0 = coeffs[0]
    a = [c / a0 for c in coeffs[1:]]
    n = len(a)
    C = matrix(n)
    for i in range(1, n):
        C[i, i-1] = 1
    for i in range(n):
        C[i, n-1] = -a[n-1-i]
    return C


def compute_roots(coeffs):
    C = build_companion(coeffs)
    vals, _ = eig(C)
    roots = [mpc(v) for v in vals]
    return roots

# ========================= CLUSTERING ========================= #


def cluster_roots(roots, tol=CLUSTER_TOL):
    clusters = [[r] for r in roots]
    while True:
        merged = False
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                c1 = sum(clusters[i]) / len(clusters[i])
                c2 = sum(clusters[j]) / len(clusters[j])
                scale = max(abs(c1), abs(c2), mpf(1))
                if abs(c1 - c2) < tol * scale:
                    clusters[i].extend(clusters[j])
                    clusters.pop(j)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    return clusters

# ========================= DELTA ========================= #


def compute_cluster_delta(cluster, clusters, lc):
    a = sum(cluster) / len(cluster)
    m = len(cluster)
    log_sum = mpf(0)
    for other in clusters:
        if other is cluster:
            continue
        b = sum(other) / len(other)
        k = len(other)
        dist = abs(a - b)
        if dist > 0:
            log_sum += k * mp.log(dist)
    delta = mp.e ** (-log_sum / m) if m > 0 else mpf(0)
    if m > 0 and lc != 0:
        delta *= abs(lc) ** (-mpf(1) / m)
    return a, m, delta

# ========================= FIELD ========================= #


def _min_centroid_separation(root_data):
    """Minimum pairwise distance between cluster centroids."""
    centroids = [a for a, m, delta in root_data]
    if len(centroids) < 2:
        return mp.inf
    min_sep = mp.inf
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            d = abs(centroids[i] - centroids[j])
            if d < min_sep:
                min_sep = d
    return min_sep


def compute_field_fast(coeffs, root_data, N=400):
    """
    Vectorized float64 field computation.
    Fast and stable when all cluster centroids are separated by more than the safety threshold.
    """
    use_global_scaling = USE_GLOBAL_SCALING
    if use_global_scaling:
        R = max([abs(a) + delta for a, _, delta in root_data] + [mpf(1)]) * 1.2
    else:
        max_abs_root = max([abs(a) for a, _, _ in root_data] + [mpf(1)])
        R = max_abs_root * 1.5
    roots_f = np.array([complex(a)
                       for a, m, delta in root_data], dtype=complex)
    m_f = np.array([float(m) for a, m, delta in root_data], dtype=float)
    deltas_f = np.array([float(delta)
                        for a, m, delta in root_data], dtype=float)
    xs = np.linspace(-float(R), float(R), N)
    ys = np.linspace(-float(R), float(R), N)
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y
    diffs = Z[..., np.newaxis] - roots_f
    dist_matrix = np.abs(diffs) / deltas_f
    min_dist = np.min(dist_matrix, axis=-1)
    dist = np.log10(min_dist + 1e-30)
    EPS = 1e-30
    abs_diffs = np.abs(diffs)
    safe_diffs = np.where(abs_diffs < EPS, EPS, diffs)
    log_deriv = np.sum(m_f / safe_diffs, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        w = -1.0 / log_deriv
    mag = np.abs(w)
    inv_mag = np.where(mag > 0, mag, 1.0)
    flow_u = np.real(w) / inv_mag
    flow_v = np.imag(w) / inv_mag
    bad = (mag == 0) | ~np.isfinite(mag)
    flow_u[bad] = 0.0
    flow_v[bad] = 0.0
    return xs, ys, dist, flow_u, flow_v


def compute_field_mpmath(coeffs, root_data, N=200):
    """
    Mpmath pixel-loop field computation (used internally by auto mode when needed).
    """
    use_global_scaling = USE_GLOBAL_SCALING
    if use_global_scaling:
        R = max([abs(a) + delta for a, _, delta in root_data] + [mpf(1)]) * 1.2
    else:
        max_abs_root = max([abs(a) for a, _, _ in root_data] + [mpf(1)])
        R = max_abs_root * 1.5
    xs = np.linspace(-float(R), float(R), N)
    ys = np.linspace(-float(R), float(R), N)
    dist = np.zeros((N, N))
    flow_u = np.zeros((N, N))
    flow_v = np.zeros((N, N))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            z = mpc(x, y)
            dmin = mp.inf
            for a, m, delta in root_data:
                if delta > 0:
                    val = abs(z - a) / delta
                    if val < dmin:
                        dmin = val
            dist[j, i] = float(mp.log10(dmin + mpf('1e-30')))
            log_deriv = mpc(0)
            for a, m, delta in root_data:
                dz = z - a
                if abs(dz) > mpf('1e-30'):
                    log_deriv += m / dz
            if abs(log_deriv) > mpf('1e-30'):
                w = -mpc(1) / log_deriv
                mag = abs(w)
                if mag > 0:
                    w = w / mag
                else:
                    w = mpc(0)
            else:
                w = mpc(0)
            flow_u[j, i] = float(mp.re(w))
            flow_v[j, i] = float(mp.im(w))
        if i % (N // 10) == 0:
            print(f" Computing... {i * 100 // N}% complete")
    print(" Computation complete")
    return xs, ys, dist, flow_u, flow_v


def compute_field(coeffs, root_data, N=800, mode='auto'):
    """
    Dispatcher: routes to fast (float64) or precise (mpmath) implementation.
    In 'auto' mode mpmath is used ONLY when:
        CLUSTER_TOL < relative separation ≤ CLUSTER_TOL × 1e7
    """
    use_global_scaling = USE_GLOBAL_SCALING
    if use_global_scaling:
        R = max([abs(a) + delta for a, _, delta in root_data] + [mpf(1)]) * 1.2
        mode_desc = "GLOBAL SCALING (includes largest δ)"
    else:
        max_abs_root = max([abs(a) for a, _, _ in root_data] + [mpf(1)])
        R = max_abs_root * 1.5
        mode_desc = "ROOT-FOCUSED SCALING"
    print(f" → Using {mode_desc} with R = {float(R):.1f}")

    if mode == 'fast':
        print(" → Forced fast (float64 vectorized)")
        return compute_field_fast(coeffs, root_data, N)

    # AUTO MODE
    if len(root_data) == 1:
        print(" → Single cluster — no centroid separation to measure")
        print(" → float64 path (fast vectorized)")
        return compute_field_fast(coeffs, root_data, N)

    min_sep = _min_centroid_separation(root_data)
    print(f" → Minimum centroid separation: {mp.nstr(min_sep, 6)}")

    root_scale = max([abs(a) for a, _, _ in root_data] + [mpf(1)])
    effective_threshold = FLOAT64_SAFE_REL_THRESHOLD * root_scale

    print(f" → Effective float64 safe threshold: {mp.nstr(effective_threshold, 6)} "
          f"(root_scale ≈ {mp.nstr(root_scale, 4)})")

    if min_sep > effective_threshold:
        print(" → float64 path (fast vectorized)")
        return compute_field_fast(coeffs, root_data, N)
    else:
        print(
            f" → mpmath path (separation {mp.nstr(min_sep, 3)} "
            f"≤ effective threshold {mp.nstr(effective_threshold, 3)})")
        print("    (clusters are distinct but too close for reliable float64)")
        N_mpmath = min(N, 200)
        print(f" Using grid resolution: {N_mpmath}x{N_mpmath}")
        return compute_field_mpmath(coeffs, root_data, N_mpmath)

# ========================= PLOT ========================= #


def plot_field(xs, ys, dist, flow_u, flow_v, root_data, poly_str, var):
    X, Y = np.meshgrid(xs, ys)
    plt.figure(figsize=(10, 9))
    im = plt.imshow(dist, extent=[xs[0], xs[-1], ys[0], ys[-1]],
                    origin='lower', cmap='viridis', zorder=1)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(
        r'Log Normalized Distance: $\log_{10}(|z - a| / \delta)$', fontsize=10)
    plt.streamplot(X, Y, flow_u, flow_v, density=1.2,
                   color='black', linewidth=0.5, zorder=2)
    for a, m, delta in root_data:
        ar = float(mp.re(a))
        ai = float(mp.im(a))
        dr = float(delta)
        circle = plt.Circle((ar, ai), dr, fill=False, color='red',
                            linestyle='--', linewidth=1.5, zorder=3)
        plt.gca().add_patch(circle)
        plt.scatter(ar, ai, color='red', s=40, zorder=4)
    full_equation = f"{poly_str} = 0"
    window_title = (
        full_equation[:75] + '...') if len(full_equation) > 75 else full_equation
    plt.gcf().canvas.manager.set_window_title(window_title)
    wrapped_eq = textwrap.fill(full_equation, width=80)
    combined_title = (
        "Global Newton Flow over δ-Normalized Root Influence Fields\n"
        rf"$\mathbf{{P({var}):}}$ {wrapped_eq}"
    )
    plt.title(combined_title, fontsize=11, pad=15)
    remark = ("Physically honest auto-scaling\n"
              "δ-boundary (field = 0) colour shifts\n"
              "with global field range of this polynomial")
    plt.text(0.02, 0.98, remark, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=3))
    plt.gca().set_aspect('equal')
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.tight_layout()
    plt.show()

# ========================= MAIN ========================= #


def main():
    coeffs = get_coefficients_from_user()
    lc = coeffs[0]
    degree = len(coeffs) - 1
    var = 'x' if is_real_polynomial(coeffs) else 'z'
    poly_str = polynomial_to_string(coeffs, var=var)
    print("\nPolynomial Equation:")
    print(f" Degree: {degree}")
    print(f" P({var}) = {poly_str} = 0")
    print("\nComputing clustered roots...")
    roots = compute_roots(coeffs)
    clusters = cluster_roots(roots)
    root_data = [compute_cluster_delta(c, clusters, lc) for c in clusters]
    print("\nClustered roots:")
    for a, m, delta in root_data:
        print(f"a={mp.nstr(a, 6)}, m={m}, δ={mp.nstr(delta, 6)}")

    field_mode = MODE
    grid_resolution = GRID_RESOLUTION
    print(f"\nUsing computation mode: {field_mode}")
    print(f"Grid resolution: {grid_resolution}x{grid_resolution}")
    print(
        f"Scaling mode: {'global' if USE_GLOBAL_SCALING else 'root-focused'}")
    print("\nComputing field layout...")
    xs, ys, dist, fu, fv = compute_field(
        coeffs, root_data, mode=field_mode, N=grid_resolution)
    plot_field(xs, ys, dist, fu, fv, root_data, poly_str, var)


if __name__ == "__main__":
    main()
