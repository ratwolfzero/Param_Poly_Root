import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, mpc, mpf, matrix, eig
import textwrap

# ========================= SETTINGS ========================= #
mp.dps = 600

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
        # Replace commas with spaces to safely support comma-separated lists
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

    # Wrap in parentheses if it has both real and imaginary parts
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

        if power == 0:
            term = f"{coeff_str}"
        elif power == 1:
            term = f"{coeff_str}{var}"
        else:
            term = f"{coeff_str}{var}^{power}"

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

def cluster_roots(roots, tol=mp.mpf('1e-20')):
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

    # === NON-MONIC POLYNOMIALS ===
    if m > 0 and lc != 0:
        delta *= abs(lc) ** (-mpf(1) / m)  # |lc|^{-1/m}

    return a, m, delta


# ========================= FIELD ========================= #

def compute_field(coeffs, root_data, N=200):

    # =============================================
    # SWITCH BETWEEN THE TWO SCALING MODES HERE
    # =============================================
    use_global_scaling = True      # ←←← CHANGE THIS TO True / False

    if use_global_scaling:
        # MODE 1: Global scaling
        # Includes every δ → plot can become huge when there are very large δ values
        R = max([abs(a) + delta for a, _, delta in root_data] + [mpf(1)]) * 1.2
        mode_desc = "GLOBAL SCALING (includes largest δ)"
    else:
        # MODE 2: Root-focused scaling
        # Only looks at the actual root positions
        max_abs_root = max([abs(a) for a, _, _ in root_data] + [mpf(1)])
        R = max_abs_root * 1.5
        mode_desc = "ROOT-FOCUSED SCALING"

    print(f"   → Using {mode_desc} with R = {float(R):.1f}")

    # ====================== REST OF THE FUNCTION ======================
    xs = np.linspace(-float(R), float(R), N)
    ys = np.linspace(-float(R), float(R), N)

    dist = np.zeros((N, N))
    flow_u = np.zeros((N, N))
    flow_v = np.zeros((N, N))

    dcoeffs = poly_derivative(coeffs)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            z = mpc(x, y)

            # δ-distance field
            dmin = mp.inf
            for a, m, delta in root_data:
                if delta > 0:
                    val = abs(z - a) / delta
                    if val < dmin:
                        dmin = val
            dist[j, i] = float(mp.log10(dmin + 1e-30))

            # Newton flow (direction only)
            p = poly_eval(coeffs, z)
            dp = poly_eval(dcoeffs, z)
            if abs(dp) > mp.mpf('1e-30'):
                w = -p / dp
                mag = abs(w)
                if mag > 0:
                    w = w / mag
                else:
                    w = mpc(0)
            else:
                w = mpc(0)

            flow_u[j, i] = float(mp.re(w))
            flow_v[j, i] = float(mp.im(w))

    return xs, ys, dist, flow_u, flow_v


# ========================= PLOT ========================= #

def plot_field(xs, ys, dist, flow_u, flow_v, root_data, poly_str, var):
    """
    Physically most honest scaling:
    - Dynamic variable P(x) or P(z) based on coefficient types
    - Wrapped polynomial equation included below the main title
    """
    X, Y = np.meshgrid(xs, ys)
    plt.figure(figsize=(10, 9))

    # Background field
    im = plt.imshow(dist, extent=[xs[0], xs[-1], ys[0], ys[-1]],
                    origin='lower', cmap='viridis', zorder=1)

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(
        r'Log Normalized Distance: $\log_{10}(|z - a| / \delta)$', fontsize=10)

    # Newton Flow
    plt.streamplot(X, Y, flow_u, flow_v, density=1.2,
                   color='black', linewidth=0.5, zorder=2)

    # Roots and Delta-Boundaries
    for a, m, delta in root_data:
        ar = float(mp.re(a))
        ai = float(mp.im(a))
        dr = float(delta)

        circle = plt.Circle((ar, ai), dr, fill=False, color='red',
                            linestyle='--', linewidth=1.5, zorder=3)
        plt.gca().add_patch(circle)
        plt.scatter(ar, ai, color='red', s=40, zorder=4)

    # === DYNAMIC TITLES ===
    full_equation = f"{poly_str} = 0"
    
    # 1. Window Title
    window_title = (full_equation[:75] + '...') if len(full_equation) > 75 else full_equation
    plt.gcf().canvas.manager.set_window_title(window_title)

    # 2. Combined Plot Title (Using the correct variable x or z)
    wrapped_eq = textwrap.fill(full_equation, width=80)
    combined_title = (
        "Global Newton Flow over δ-Normalized Root Influence Fields\n"
        f"$\mathbf{{P({var}):}}$ {wrapped_eq}"
    )
    
    plt.title(combined_title, fontsize=11, pad=15)

    # Physical Remark
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
    print(f"   Degree: {degree}")
    print(f"   P({var}) = {poly_str} = 0")

    print("\nComputing clustered roots...")
    roots = compute_roots(coeffs)
    clusters = cluster_roots(roots)

    # pass lc to the delta function
    root_data = [compute_cluster_delta(c, clusters, lc) for c in clusters]

    print("\nClustered roots:")
    for a, m, delta in root_data:
        print(f"a={mp.nstr(a, 6)}, m={m}, δ={mp.nstr(delta, 6)}")

    print("\nComputing field layout...")
    xs, ys, dist, fu, fv = compute_field(coeffs, root_data)
    plot_field(xs, ys, dist, fu, fv, root_data, poly_str, var)


if __name__ == "__main__":
    main()
