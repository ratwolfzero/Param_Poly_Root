"""
root_field.py
=============
Global Newton flow over δ-normalized root influence fields.

For each root of a univariate polynomial P(x) this script computes the
local parameterization triplet (a, m, δ) defined in the accompanying paper:

    a — root location (cluster centroid)
    m — algebraic multiplicity (cluster size)
    δ — characteristic deflection distance = |α|^(-1/m),
        where α = P^(m)(a) / m! is the leading asymptotic coefficient.

The script then:
  1. Evaluates a residual check |P(a)| and the scale-invariant quantity
     |P(a)|·δᵐ to assess numerical reliability of each root.
  2. Optionally refines unreliable roots via Halley / Newton iteration.
  3. Computes and plots:
       • the δ-normalized distance field  log₁₀(min_i |z−aᵢ|/δᵢ)
       • the Newton flow  V(z) = −P(z)/P'(z)  as a streamline overlay.

Dependencies: numpy, matplotlib, mpmath
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, mpc, mpf, matrix, eig
import textwrap
import re

# ========================= SETTINGS ========================= #

# Arbitrary-precision decimal places.
# This is the single most important accuracy parameter.
# The companion-matrix eigenvalue solver works entirely at this precision,
# so increasing mp.dps directly reduces rounding error in the computed roots.
#
# Rule of thumb: for a polynomial of degree n with roots of multiplicity m,
# the minimum safe value is roughly  mp.dps ≈ n * m * 2.
# The default 600 comfortably covers all provided test cases.
mp.dps = 600

# ------------------------------------------------------------------
# CLUSTERING TOLERANCE
# ------------------------------------------------------------------
# CLUSTER_TOL controls when two computed roots are merged into a single
# cluster (treated as one root with higher multiplicity).
#
# The smallest *meaningful* separation between roots is bounded below by
# the numerical noise floor of the eigenvalue computation:
#
#     noise floor ≈ 10^(-mp.dps / m)
#
# where m is the (unknown) multiplicity.  CLUSTER_TOL must stay above
# this floor or true multiple roots split into artificial sub-clusters,
# causing δ to overflow to inf and the field to become meaningless.
#
# Conversely, CLUSTER_TOL must stay below the smallest genuine root
# separation to avoid merging distinct roots.
#
# Practical guide (mp.dps = 600):
#   m = 20  →  noise ≈ 1e-30  →  safe range: 1e-29 … 1e-20
#   m = 30  →  noise ≈ 1e-20  →  safe range: 1e-19 … 1e-10
#   m = 50  →  noise ≈ 1e-12  →  safe range: 1e-11 … 1e-6
#
# The default 1e-10 provides a good balance for multiplicities up to ~40
# at mp.dps = 600. It works well for high-degree polynomials (e.g., degree 90)
# and is validated against Wilkinson-type polynomials. For very high
# multiplicities (>50) or lower precision, increase CLUSTER_TOL proportionally.
CLUSTER_TOL = mpf('1e-10')

# ------------------------------------------------------------------
# FLOAT64 SAFETY THRESHOLD
# ------------------------------------------------------------------
# Controls the auto-dispatch between the fast float64 field computation
# and the slower but more accurate mpmath pixel loop.
#
# float64 is used when:   min_centroid_separation > CLUSTER_TOL × 1e7
# mpmath  is used when:   CLUSTER_TOL < separation ≤ CLUSTER_TOL × 1e7
#
# The 1e7 factor provides a safety margin: float64 carries ~15 significant
# digits, so any separation smaller than CLUSTER_TOL × 1e7 is within the
# danger zone where float64 arithmetic can confuse adjacent clusters.
FLOAT64_SAFE_REL_THRESHOLD = CLUSTER_TOL * mpf('1e7')

# ------------------------------------------------------------------
# COMPUTATION MODE
# ------------------------------------------------------------------
# 'fast'  — always use float64 vectorized path (fastest, may lose
#            precision on very closely spaced clusters)
# 'auto'  — use float64 by default, fall back to mpmath only when
#            the minimum centroid separation is too small for float64
#            (recommended: balances speed and numerical stability)
MODE = 'auto'

# ------------------------------------------------------------------
# GRID RESOLUTION
# ------------------------------------------------------------------
# Number of grid points per axis for the field computation.
# The float64 path uses the full resolution.
# The mpmath pixel-loop path is automatically capped at 200×200
# because it is O(N²) in pure Python.
GRID_RESOLUTION = 800

# ------------------------------------------------------------------
# SCALING MODE
# ------------------------------------------------------------------
# True  — global scaling: the plot window extends to include the largest
#          δ-disk, giving a "physically honest" view of the full field.
# False — root-focused: the window is scaled to the root locations only,
#          ignoring the spatial reach of large-δ roots.
USE_GLOBAL_SCALING = True

# ------------------------------------------------------------------
# MAXIMUM SAFE PLOT RADIUS
# ------------------------------------------------------------------
# Used to avoid infinite or overflowed axis limits when a δ value is
# astronomically large.  If global scaling would produce a non-finite
# or impractically huge plot radius, the code falls back to root-focused
# scaling instead.
MAX_PLOT_RADIUS = mpf('1e8')

# ------------------------------------------------------------------
# RESIDUAL WARNING THRESHOLDS
# ------------------------------------------------------------------
# The scale-invariant residual  rel = |P(a)| · δᵐ  measures how far
# a computed centroid deviates from a true root, normalised by δ so
# that roots of vastly different scales are judged on equal footing.
#
# At an exact root rel = 0.  In practice:
#
#   rel < RESIDUAL_WARN_SOFT  →  tier 'ok'   — root is trusted
#   rel < RESIDUAL_WARN_HARD  →  tier 'warn' — marginal, likely usable
#   rel ≥ RESIDUAL_WARN_HARD  →  tier 'bad'  — root is unreliable;
#                                  δ and the field visualization may
#                                  be wrong for this root
#
# These thresholds are conservative.  With mp.dps = 600 all well-posed
# polynomials produce rel ≪ 1e-10 (tier 'ok').  The 'warn' and 'bad'
# tiers appear only when the companion matrix is severely ill-conditioned
# (e.g. Wilkinson-type polynomials at low precision).
RESIDUAL_WARN_SOFT = mpf('1e-10')
RESIDUAL_WARN_HARD = mpf('1e-3')

# ========================= INPUT ========================= #


def parse_coefficients_strict(text):
    """
    Parse a whitespace-separated string of coefficients with full precision.
    Accepts real, imaginary, and complex numbers in any common format.
    Examples:
        "1 2i 3+4I -5-6j"   -> three coefficients: 1, 2i, 3+4i
        "I 0 I"             -> i, 0, i
        "2+I 1 0"           -> 2+i, 1, 0
        "1.2345678901234567890" -> preserved exactly
        "+j -J"             -> i, -i
    """
    tokens = text.strip().split()
    if not tokens:
        raise ValueError("Empty input.")

    coeffs = []
    for token in tokens:
        # Remove any spaces (just in case)
        token = token.replace(' ', '')

        # Normalise all imaginary unit variants to 'j'
        # Replace i/I/j/J with j, but careful with numbers like '2i' -> '2j'
        token = re.sub(r'([0-9)])[iIjJ]', r'\1j', token)   # 2i -> 2j
        token = re.sub(r'^[iIjJ](?=[-+]|$)', 'j', token)   # i at start -> j
        # any remaining i/I -> j
        token = re.sub(r'[iIjJ]', 'j', token)

        # Special case: pure imaginary without a digit (e.g., 'j', '+j', '-j')
        # Convert to '0+1j', '0+1j', '0-1j' for uniform parsing
        if token in ('j', '+j'):
            token = '0+1j'
        elif token == '-j':
            token = '0-1j'

        # Try to parse as complex number using regex that allows missing real part
        # Pattern: (optional real part) (optional imaginary part)
        # Real part: optional sign, digits, decimal, exponent
        # Imag part: sign, then digits/decimal/exponent, then 'j'
        # Both parts are optional, but at least one must be present.
        pattern = re.compile(
            r'^'
            r'(?:(?P<real>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|)'  # real part
            # imag part
            r'(?:(?P<imag_sign>[+-])?(?P<imag_num>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)j|)'
            r'$'
        )
        match = pattern.match(token)
        if match:
            real_str = match.group('real')
            imag_sign = match.group('imag_sign')
            imag_num = match.group('imag_num')

            # Determine real part
            if real_str is None:
                real = mpf(0)
            else:
                real = mpf(real_str)

            # Determine imaginary part
            if imag_num is None:
                imag = mpf(0)
            else:
                # If sign is missing, default to '+'
                sign = 1 if imag_sign is None or imag_sign == '+' else -1
                imag = sign * mpf(imag_num)

            coeffs.append(mpc(real, imag))
            continue

        # Not a complex number: try as plain real
        try:
            coeffs.append(mpc(mpf(token)))
            continue
        except:
            raise ValueError(f"Invalid coefficient: '{token}'")

    # Strip leading zeros
    while len(coeffs) > 1 and coeffs[0] == 0:
        coeffs.pop(0)

    return coeffs


def load_coefficients_from_file(path):
    """
    Read coefficients from a text file and parse them.
    """
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return parse_coefficients_strict(text)


def get_coefficients_from_user():
    """
    Interactively prompt the user for polynomial coefficients and
    return a validated list of mpc values (highest degree first).

    Loops until the input passes parse_coefficients_strict and the
    basic sanity checks (degree ≥ 1, not all-zero).
    """
    print("\nEnter polynomial coefficients (highest degree first).")
    print("Separate coefficients by spaces or commas.")
    print("IMPORTANT: Do not put spaces inside complex numbers "
          "(e.g., use 1+2i, not 1 + 2i).")
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
    """Return True if every coefficient has zero imaginary part."""
    return all(mp.im(c) == 0 for c in coeffs)


def format_real(r, precision=6):
    """
    Format an mpf as a compact string.
    Integer values are printed without a decimal point.
    """
    if mp.almosteq(r, mp.nint(r)):
        return str(int(mp.nint(r)))
    else:
        return mp.nstr(r, precision)


def format_complex(z, precision=6):
    """
    Format an mpc as a human-readable string, suppressing zero parts
    and simplifying ±1 imaginary coefficients to ±i.
    """
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


def format_root_location(z, precision=6, width=28):
    """
    Format a cluster centroid in a compact fixed-width form.
    """
    re = mp.re(z)
    im = mp.im(z)
    if mp.almosteq(im, 0):
        s = format_real(re, precision)
    elif mp.almosteq(re, 0):
        if mp.almosteq(abs(im), 1):
            s = "i" if im > 0 else "-i"
        else:
            s = f"{format_real(im, precision)}i"
    else:
        re_str = format_real(re, precision)
        im_str = format_real(abs(im), precision)
        sign = "+" if im > 0 else "-"
        if mp.almosteq(abs(im), 1):
            im_part = "i"
        else:
            im_part = f"{im_str}i"
        s = f"{re_str}{sign}{im_part}"
    if len(s) > width:
        s = s[:width - 3] + "..."
    return s


def format_value(value, precision=4):
    """
    Format a floating-point value with a consistent significant-digit count.
    """
    if value == mp.inf:
        return "inf"
    return mp.nstr(value, precision, strip_zeros=True)


def polynomial_to_string(coeffs, var='z', precision=6):
    """
    Convert a coefficient list to a human-readable polynomial string.

    Zero terms are omitted.  The variable name is 'x' for real
    polynomials and 'z' for complex ones (caller's responsibility).
    """
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
    """
    Evaluate the polynomial at x using Horner's method.

    Horner's scheme minimises the number of multiplications and
    accumulates rounding error far more slowly than naive evaluation,
    which matters at the high precision used here.

    Parameters
    ----------
    coeffs : list of mpc
        Coefficients in descending degree order.
    x : mpc
        Evaluation point.

    Returns
    -------
    mpc
        P(x).
    """
    p = mpc(0)
    for c in coeffs:
        p = p * x + c
    return p


def poly_derivative(coeffs):
    """
    Return the coefficient list of P'(x) from the coefficient list of P(x).

    Uses the standard power-rule formula:  if P = Σ cᵢ xⁿ⁻ⁱ then
    P' = Σ cᵢ·(n−i) xⁿ⁻ⁱ⁻¹  for i < n.

    Parameters
    ----------
    coeffs : list of mpc
        Coefficients of P in descending degree order.

    Returns
    -------
    list of mpc
        Coefficients of P' in descending degree order.
        Returns an empty list if P is constant (degree 0).
    """
    n = len(coeffs) - 1
    return [coeffs[i] * (n - i) for i in range(len(coeffs) - 1)]

# ========================= COMPANION ROOT SOLVER ========================= #


def build_companion(coeffs):
    """
    Build the companion matrix of the monic polynomial obtained by
    dividing coeffs by its leading coefficient.

    The companion matrix is the n×n matrix whose characteristic polynomial
    equals the monic form of P, so its eigenvalues are the roots of P.
    It is constructed in the standard colleague / Frobenius form:
        C[i, i-1] = 1  for i = 1 … n-1   (sub-diagonal)
        C[i, n-1] = -aᵢ                   (last column)

    Parameters
    ----------
    coeffs : list of mpc
        Coefficients in descending degree order.  coeffs[0] is the
        leading coefficient and must be non-zero.

    Returns
    -------
    mpmath.matrix
        n×n companion matrix at the current mp.dps precision.
    """
    a0 = coeffs[0]
    a = [c / a0 for c in coeffs[1:]]
    n = len(a)
    C = matrix(n)
    for i in range(1, n):
        C[i, i - 1] = 1
    for i in range(n):
        C[i, n - 1] = -a[n - 1 - i]
    return C


def compute_roots(coeffs):
    """
    Compute all roots of P by finding the eigenvalues of its companion matrix.

    Uses mpmath's arbitrary-precision eigenvalue solver (QR algorithm),
    so accuracy scales with mp.dps.  For ill-conditioned polynomials
    (Wilkinson-type, high multiplicity) more digits may be needed —
    the residual check in compute_residuals will flag any failures.

    Parameters
    ----------
    coeffs : list of mpc
        Coefficients in descending degree order.

    Returns
    -------
    list of mpc
        Unordered list of n roots (with repetition for multiple roots,
        before clustering).
    """
    C = build_companion(coeffs)
    vals, _ = eig(C)
    roots = [mpc(v) for v in vals]
    return roots

# ========================= CLUSTERING ========================= #


def cluster_roots(roots, tol=CLUSTER_TOL):
    """
    Merge roots that are numerically indistinguishable into clusters.

    Two cluster centroids c₁ and c₂ are merged when:

        |c₁ − c₂| < tol × max(|c₁|, |c₂|, 1)

    The relative scaling by max(|c₁|, |c₂|, 1) makes the tolerance
    scale-invariant: roots near the origin and roots far from it are
    treated consistently.

    Merging is repeated until no further merges occur (single-pass
    greedy algorithm).  The cluster size becomes the algebraic
    multiplicity estimate for the root.

    Parameters
    ----------
    roots : list of mpc
        Raw roots from the eigenvalue solver.
    tol : mpf
        Clustering tolerance.  See CLUSTER_TOL for tuning guidance.

    Returns
    -------
    list of list of mpc
        Each inner list is one cluster; its length is the multiplicity.
    """
    clusters = [[r] for r in roots]
    while True:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
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
    # Sort clusters by their centroids for consistent output
    clusters.sort(key=lambda c: (float(mp.re(sum(c) / len(c))),
                                 float(mp.im(sum(c) / len(c)))))
    return clusters

# ========================= DELTA ========================= #


def compute_cluster_delta(cluster, clusters, lc):
    """
    Compute the local parameterization triplet (a, m, δ) for one cluster.

    The characteristic deflection distance δ is defined as:

        δ = |α|^(-1/m)

    where α = P^(m)(a) / m! is the leading asymptotic coefficient.
    For a polynomial written in factored form as

        P(x) = lc · (x−a)^m · ∏ₖ (x−bₖ)^{mₖ}

    this simplifies to:

        α = lc · ∏ₖ (a − bₖ)^{mₖ}

    and therefore:

        δ = |lc|^(-1/m) · (∏ₖ |a − bₖ|^{mₖ})^(-1/m)

    The logarithmic sum avoids overflow when distances or multiplicities
    are large.

    Parameters
    ----------
    cluster : list of mpc
        The cluster whose triplet is being computed.  Its centroid is
        used as the root location a.
    clusters : list of list of mpc
        All clusters (including the current one, which is skipped in
        the product).
    lc : mpc
        Leading coefficient of P.

    Returns
    -------
    a     : mpc   cluster centroid (root location)
    m     : int   cluster size (algebraic multiplicity)
    delta : mpf   characteristic deflection distance δ
    """
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

# ========================= RESIDUAL CHECK ========================= #


def compute_residuals(coeffs, root_data):
    """
    Assess the numerical reliability of each computed root via residuals.

    For each cluster centroid a with multiplicity m and scale δ, compute:

        abs_residual = |P(a)|
        rel_residual = |P(a)| · δᵐ    (scale-invariant, dimensionless)

    The relative residual is zero at an exact root.  It is dimensionless
    because δᵐ ≈ 1/|α|, so multiplying by δᵐ normalises |P(a)| by the
    magnitude of the leading asymptotic term.  This makes roots of very
    different sizes comparable on the same scale.

    Tier assignment (governed by RESIDUAL_WARN_SOFT / RESIDUAL_WARN_HARD):
        'ok'   — root is trusted; δ and the field are reliable
        'warn' — marginal; root may be slightly off (ill-conditioned)
        'bad'  — root is unreliable; δ and the field may be wrong

    Parameters
    ----------
    coeffs    : list of mpc
        Polynomial coefficients.
    root_data : list of (mpc, int, mpf)
        Output of compute_cluster_delta.

    Returns
    -------
    list of dict, one per cluster:
        'abs'  : mpf   |P(a)|
        'rel'  : mpf   |P(a)| · δᵐ
        'tier' : str   'ok' | 'warn' | 'bad'
        'note' : str   human-readable annotation (empty string when ok)
    """
    results = []
    for a, m, delta in root_data:
        pa = poly_eval(coeffs, a)
        abs_res = abs(pa)

        if delta > 0:
            rel_res = abs_res * delta ** m
        else:
            # delta == 0 only for a degenerate cluster; treat as unresolvable
            rel_res = mp.inf

        if rel_res < RESIDUAL_WARN_SOFT:
            tier = 'ok'
            note = ''
        elif rel_res < RESIDUAL_WARN_HARD:
            tier = 'warn'
            note = ('marginal — root may be inaccurate '
                    '(ill-conditioned polynomial or tight cluster)')
        else:
            tier = 'bad'
            note = ('UNRELIABLE — |P(a)| is large relative to δᵐ; '
                    'δ and field visualization may be wrong for this root')

        results.append({
            'abs':  abs_res,
            'rel':  rel_res,
            'tier': tier,
            'note': note,
        })
    return results


def format_root_report(root_data, residuals, dps=None, cluster_tol=None):
    """
    Print the clustered-root table with full residual diagnostics.

    Columns
    -------
    #          — 1-based index
    a          — cluster centroid (root location)
    m          — algebraic multiplicity
    δ          — characteristic deflection distance
    |P(a)|     — absolute residual at the centroid
    |P(a)|·δᵐ  — scale-invariant relative residual
    status     — 'ok' / '~' (warn) / '!' (bad)

    A ↳ note is printed below any non-ok row.
    A summary block is printed after the table with a diagnostic suggestion
    if many clusters are detected (suggesting higher multiplicities).

    Parameters
    ----------
    root_data   : list of (mpc, int, mpf)
    residuals   : list of dict
    dps         : int or None    current mp.dps for suggestions
    cluster_tol : mpf or None    current CLUSTER_TOL for suggestions
    """
    TIER_LABEL = {'ok': 'ok', 'warn': '~', 'bad': '!'}

    col_a = 28
    col_m = 3
    col_d = 14
    col_abs = 14
    col_rel = 18
    col_st = 6

    header = (f"  {'#':>2}  {'a':>{col_a}}  {'m':>{col_m}}  "
              f"{'δ':>{col_d}}  {'|P(a)|':>{col_abs}}  "
              f"{'|P(a)|·δᵐ':>{col_rel}}  {'status':>{col_st}}")
    print("\nClustered roots with residual diagnostics:")
    print(header)
    print("  " + "-" * (len(header) - 2))
    print("  Legend: ok = trusted, ~ = marginal, ! = unreliable")
    print("  " + "-" * (len(header) - 2))

    any_warn = False
    any_bad = False
    ok_count = 0
    warn_count = 0
    bad_count = 0

    for i, ((a, m, delta), res) in enumerate(zip(root_data, residuals)):
        marker = TIER_LABEL[res['tier']]
        abs_str = format_value(res['abs'], 4)
        rel_str = format_value(res['rel'], 4)
        a_str = format_root_location(a, precision=6, width=col_a)
        d_str = format_value(delta, 6)

        print(f"  {i+1:>2}  {a_str:>{col_a}}  {m:>{col_m}}  "
              f"{d_str:>{col_d}}  {abs_str:>{col_abs}}  "
              f"{rel_str:>{col_rel}}  {marker:>{col_st}}")

        if res['note']:
            print(f"        ↳ {res['note']}")

        if res['tier'] == 'warn':
            any_warn = True
            warn_count += 1
        elif res['tier'] == 'bad':
            any_bad = True
            bad_count += 1
        else:
            ok_count += 1

    print()
    print(f"  Summary: {ok_count} ok, {warn_count} warn, {bad_count} bad")
    total_roots = sum(m for _, m, _ in root_data)
    print(f"  Total roots (sum of multiplicities): {total_roots}")
    print(f"  Total clusters displayed         : {len(root_data)}")

    # Diagnostic suggestion for parameter tuning
    num_clusters = len(root_data)
    num_m1_clusters = sum(1 for _, m, _ in root_data if m == 1)
    if dps is not None and cluster_tol is not None and num_m1_clusters > total_roots / 2:
        noise_floor = mpf(10) ** (-(dps / num_m1_clusters))
        # 10× above noise floor for safety
        suggested_tol = noise_floor * mpf(10)
        print(f"\n  💡 DIAGNOSTIC HINT:")
        print(f"     Detected {num_clusters} clusters with mostly m=1.")
        print(f"     If higher multiplicities are expected, try:")
        print(
            f"       python3 root_field.py --cluster-tol {mp.nstr(suggested_tol, 3)} ...")
        print(f"     Or increase precision with:  --dps {dps + 200}")
    print()

    if any_bad:
        print("  ⚠  WARNING: one or more roots are UNRELIABLE.")
        print("     The companion-matrix solver has lost accuracy for those roots.")
        print("     Likely cause: ill-conditioned polynomial (e.g. Wilkinson-type),")
        print("     very high degree, or extreme coefficient dynamic range.")
        print("     δ values and field visualization are untrustworthy for")
        print("     marked roots.")
        print("     Suggestions: increase mp.dps, or call refine_bad_roots().")
        print()
    elif any_warn:
        print("  ~  CAUTION: some roots have marginal residuals.")
        print("     Results are likely usable but inspect δ values carefully.")
        print()
    else:
        print("  ✓  All roots verified — residuals within tolerance.")
        print()


# ========================= NEWTON REFINEMENT ========================= #


def _newton_refine(coeffs, z0, max_iter=40, tol=None):
    """
    Polish a single root estimate z0 using Halley's method.

    Halley's iteration:

        zₙ₊₁ = zₙ − P(zₙ) / [ P'(zₙ) − P(zₙ)·P''(zₙ) / (2·P'(zₙ)) ]

    This is a third-order method: it converges cubically near a simple
    root, which means it can recover many digits in just a few steps
    even when the starting point is moderately inaccurate.  When the
    Halley correction P''·P / (2·P'²) is negligible the step reduces
    to the ordinary Newton step.

    The iteration terminates when |step| < tol or max_iter is reached.
    The default tol is set to 10^(-(mp.dps - 10)), leaving a 10-digit
    safety margin below full working precision.

    Parameters
    ----------
    coeffs   : list of mpc   polynomial coefficients
    z0       : mpc           starting estimate (companion-matrix root)
    max_iter : int           iteration cap (default 40)
    tol      : mpf or None   convergence threshold (default auto)

    Returns
    -------
    (z, abs_res) : (mpc, mpf)
        Refined root and its absolute residual |P(z)|.
    """
    if tol is None:
        tol = mpf(10) ** (-(mp.dps - 10))

    deriv1 = poly_derivative(coeffs)
    deriv2 = poly_derivative(deriv1)

    z = mpc(z0)
    for _ in range(max_iter):
        pz = poly_eval(coeffs,  z)
        p1z = poly_eval(deriv1,  z)

        if abs(p1z) == 0:
            break   # at a critical point; Newton/Halley cannot step

        p2z = poly_eval(deriv2, z)
        denom = p1z - (pz * p2z) / (2 * p1z)
        if abs(denom) == 0:
            break

        step = pz / denom
        z = z - step
        if abs(step) < tol:
            break

    return z, abs(poly_eval(coeffs, z))


def refine_bad_roots(coeffs, root_data, residuals, tiers=('bad', 'warn')):
    """
    Re-polish non-ok cluster centroids via Halley / Newton iteration.

    For each centroid whose tier is in `tiers`, _newton_refine is called.
    If the refined point achieves a smaller |P(a)| it replaces the
    original centroid; otherwise the original is kept.

    After all refinements, all δ values are recomputed from scratch
    using the updated centroids (because δ for each root depends on
    the distances to *all* other roots, so refining one root changes
    the δ of its neighbours too).

    A fresh residual check is run on the updated root_data before
    returning, so the caller can call format_root_report immediately
    to display the improvement.

    Parameters
    ----------
    coeffs    : list of mpc
        Polynomial coefficients.
    root_data : list of (mpc, int, mpf)
        Current triplets from compute_cluster_delta.
    residuals : list of dict
        Current residual dicts from compute_residuals.
    tiers : tuple of str
        Which tiers to attempt refinement on.
        Default ('bad', 'warn') — refine anything that is not clean.
        Pass ('bad',) to only touch outright failures.

    Returns
    -------
    (root_data, residuals) : updated versions of both inputs.
    """
    lc = coeffs[0]
    centroids = [a for a, m, delta in root_data]
    mults = [m for a, m, delta in root_data]

    # Work on a mutable copy so we can update individual centroids.
    refined_centroids = list(centroids)

    for i, ((a, m, delta), res) in enumerate(zip(root_data, residuals)):
        if res['tier'] in tiers:
            z_new, abs_new = _newton_refine(coeffs, a)
            abs_old = res['abs']

            if abs_new < abs_old:
                refined_centroids[i] = z_new
                print(f"  Newton refinement  a ≈ {mp.nstr(a, 6)}: "
                      f"|P(a)| {mp.nstr(abs_old, 3)} → {mp.nstr(abs_new, 3)}")
            else:
                print(f"  Newton refinement  a ≈ {mp.nstr(a, 6)}: "
                      f"no improvement ({mp.nstr(abs_old, 3)} → {mp.nstr(abs_new, 3)})")

    # Rebuild synthetic single-point clusters from the (possibly updated)
    # centroids, then recompute all δ values — δ is a global quantity
    # that depends on every other root, so one refinement affects all δ.
    rebuilt_clusters = [[refined_centroids[i]] * mults[i]
                        for i in range(len(mults))]

    new_root_data = []
    for i, cluster in enumerate(rebuilt_clusters):
        new_a, new_m, new_delta = compute_cluster_delta(
            cluster, rebuilt_clusters, lc
        )
        new_root_data.append((new_a, new_m, new_delta))

    # Sort refined root_data by centroid for consistent output
    new_root_data.sort(key=lambda t: (float(mp.re(t[0])), float(mp.im(t[0]))))

    new_residuals = compute_residuals(coeffs, new_root_data)
    return new_root_data, new_residuals

# ========================= FIELD ========================= #


def _min_centroid_separation(root_data):
    """
    Return the minimum pairwise distance between all cluster centroids.

    Used by compute_field to decide whether float64 arithmetic is
    safe for the distance-field computation.  Returns mp.inf when
    there is only one cluster.
    """
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


def _field_radius(root_data):
    """
    Return the half-width of the plotting window for the distance field.

    If global scaling is enabled, include the largest δ-disk radius.
    If that radius is non-finite or impractically large, fall back to
    root-focused scaling to keep plotting stable.
    """
    if USE_GLOBAL_SCALING:
        R = max([abs(a) + delta for a, _, delta in root_data] +
                [mpf(1)]) * mpf('1.05')
        if not mp.isfinite(R) or float(R) == float('inf') or R > MAX_PLOT_RADIUS:
            fallback = max([abs(a)
                           for a, _, _ in root_data] + [mpf(1)]) * mpf('1.5')
            if not mp.isfinite(fallback) or float(fallback) == float('inf') or fallback > MAX_PLOT_RADIUS:
                return MAX_PLOT_RADIUS
            return fallback
        return R
    return max([abs(a) for a, _, _ in root_data] + [mpf(1)]) * mpf('1.5')


def compute_field_fast(coeffs, root_data, N=400):
    """
    Compute the δ-normalized distance field and Newton flow using
    vectorized float64 arithmetic (fast path).

    Distance field
    --------------
    For each grid point z:

        field(z) = log₁₀( min_i  |z − aᵢ| / δᵢ )

    The contour field = 0 marks the boundary of each δ-disk.

    Newton flow
    -----------
    For a polynomial in factored form the Newton step −P/P' equals

        V(z) = −1 / Σᵢ  mᵢ / (z − aᵢ)

    Each flow vector is unit-normalised so the streamplot density is
    uniform regardless of magnitude.

    Parameters
    ----------
    coeffs    : list of mpc   (unused here; retained for API symmetry)
    root_data : list of (mpc, int, mpf)
    N         : int   grid points per axis

    Returns
    -------
    xs, ys   : 1-D numpy arrays of grid coordinates
    dist     : 2-D array  log₁₀ normalized distance field
    flow_u   : 2-D array  x-component of unit Newton flow
    flow_v   : 2-D array  y-component of unit Newton flow
    """
    R = _field_radius(root_data)

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

    # Newton flow: V = -1 / Σ mᵢ/(z−aᵢ)
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
    Compute the δ-normalized distance field and Newton flow using
    mpmath arbitrary-precision arithmetic (precision path).

    Identical in definition to compute_field_fast but evaluates each
    pixel in a Python loop at mp.dps precision.  Used automatically
    by compute_field when cluster centroids are too close for reliable
    float64 arithmetic.

    Because the loop is O(N²) in pure Python, N is capped at 200 by
    the dispatcher regardless of GRID_RESOLUTION.

    Parameters and return value are identical to compute_field_fast.
    """
    R = _field_radius(root_data)

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
                w = w / mag if mag > 0 else mpc(0)
            else:
                w = mpc(0)

            flow_u[j, i] = float(mp.re(w))
            flow_v[j, i] = float(mp.im(w))

        if i % (N // 10) == 0:
            print(f" Computing... {i * 100 // N}% complete")

    print(" Computation complete")
    # Clip dist to prevent overflow in plotting
    dist = np.clip(dist, -100, 100)

    # Handle non-finite flow values
    flow_u = np.where(np.isfinite(flow_u), flow_u, 0.0)
    flow_v = np.where(np.isfinite(flow_v), flow_v, 0.0)

    return xs, ys, dist, flow_u, flow_v


def compute_field(coeffs, root_data, N=800, mode='auto'):
    """
    Dispatcher: route to the float64 or mpmath field computation.

    In 'auto' mode the decision is based on the minimum pairwise
    centroid separation relative to FLOAT64_SAFE_REL_THRESHOLD:

        separation > threshold  →  compute_field_fast   (float64)
        separation ≤ threshold  →  compute_field_mpmath (mpmath,
                                    grid capped at 200×200)

    In 'fast' mode compute_field_fast is always used regardless of
    separation, which may produce inaccurate results for tightly
    clustered roots.

    Parameters
    ----------
    coeffs    : list of mpc
    root_data : list of (mpc, int, mpf)
    N         : int   desired grid resolution (may be capped for mpmath)
    mode      : str   'auto' | 'fast'

    Returns
    -------
    Same as compute_field_fast / compute_field_mpmath.
    """
    if USE_GLOBAL_SCALING:
        raw_R = max([abs(a) + delta for a, _, delta in root_data] +
                    [mpf(1)]) * mpf('1.05')
        R = _field_radius(root_data)
        mode_desc = "GLOBAL SCALING"
        if R != raw_R:
            mode_desc = "ROOT-FOCUSED SCALING (fallback)"
    else:
        R = _field_radius(root_data)
        mode_desc = "ROOT-FOCUSED SCALING"

    print(f" → Using {mode_desc} with R = {float(R):.1f}")

    if mode == 'fast':
        print(" → Forced fast (float64 vectorized)")
        return compute_field_fast(coeffs, root_data, N)

    if len(root_data) == 1:
        print(" → Single cluster — float64 path (fast vectorized)")
        return compute_field_fast(coeffs, root_data, N)

    min_sep = _min_centroid_separation(root_data)
    root_scale = max([abs(a) for a, _, _ in root_data] + [mpf(1)])
    eff_thresh = FLOAT64_SAFE_REL_THRESHOLD * root_scale

    print(f" → Minimum centroid separation : {mp.nstr(min_sep, 6)}")
    print(f" → Effective float64 threshold : {mp.nstr(eff_thresh, 6)} "
          f"(root_scale ≈ {mp.nstr(root_scale, 4)})")

    if min_sep > eff_thresh:
        print(" → float64 path (fast vectorized)")
        return compute_field_fast(coeffs, root_data, N)
    else:
        print(f" → mpmath path (separation {mp.nstr(min_sep, 3)} "
              f"<= effective threshold {mp.nstr(eff_thresh, 3)})")
        print("    (clusters are distinct but too close for reliable float64)")
        N_mpmath = min(N, 200)
        print(f" Using grid resolution: {N_mpmath}×{N_mpmath}")
        return compute_field_mpmath(coeffs, root_data, N_mpmath)

# ========================= PLOT ========================= #


def plot_field(xs, ys, dist, flow_u, flow_v, root_data, residuals,
               poly_str, var):
    """
    Render the δ-normalized distance field with Newton flow overlay.

    Background colour map
    ---------------------
    viridis-scaled log₁₀(|z−a|/δ).  The zero contour (colour boundary
    where the field changes sign) marks the edge of each δ-disk.

    Newton streamlines
    ------------------
    Black streamlines show the gradient flow of −Re(P/P'), i.e. the
    trajectories of Newton's method in the complex plane.  Each stream-
    line terminates at the root that attracts it, tracing the basin of
    attraction for that root.

    Root markers and δ-circles
    --------------------------
    Each root is drawn as a scatter point and a dashed circle of radius δ.
    Colour and marker shape encode the residual tier from compute_residuals:

        red    ●   — ok        (|P(a)|·δᵐ < 1e-10, trusted)
        orange ■   — warn      (marginal, likely usable)
        magenta ✕  — bad       (unreliable; δ may be wrong)

    Parameters
    ----------
    xs, ys    : 1-D numpy arrays   grid coordinate axes
    dist      : 2-D numpy array    log₁₀ normalized distance field
    flow_u    : 2-D numpy array    x-component of unit Newton flow
    flow_v    : 2-D numpy array    y-component of unit Newton flow
    root_data : list of (mpc, int, mpf)   triplets (a, m, δ)
    residuals : list of dict   from compute_residuals;
                               must be in the same order as root_data
    poly_str  : str   human-readable polynomial expression (for title)
    var       : str   variable name 'x' or 'z'
    """
    X, Y = np.meshgrid(xs, ys)
    plt.figure(figsize=(10, 9))

    im = plt.imshow(dist,
                    extent=[xs[0], xs[-1], ys[0], ys[-1]],
                    origin='lower', cmap='viridis', zorder=1)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(
        r'Log Normalized Distance: $\log_{10}(|z - a| / \delta)$',
        fontsize=10)

    plt.streamplot(X, Y, flow_u, flow_v,
                   density=1.2, color='black', linewidth=0.5, zorder=2)

    # Colour and marker shape encode residual reliability tier.
    TIER_COLOR = {'ok': 'red',    'warn': 'orange', 'bad': 'magenta'}
    TIER_MARKER = {'ok': 'o',      'warn': 's',       'bad': 'X'}

    for (a, m, delta), res in zip(root_data, residuals):
        ar = float(mp.re(a))
        ai = float(mp.im(a))
        dr = float(delta)
        col = TIER_COLOR[res['tier']]
        mrk = TIER_MARKER[res['tier']]

        circle = plt.Circle((ar, ai), dr,
                            fill=False, color=col,
                            linestyle='--', linewidth=1.5, zorder=3)
        plt.gca().add_patch(circle)
        plt.scatter(ar, ai, color=col, marker=mrk, s=50, zorder=4)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=8, label='ok  (|P(a)|·δᵐ < 1e-10)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='orange',
               markersize=8, label='~   marginal (< 1e-3)'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='magenta',
               markersize=8, label='!   unreliable (≥ 1e-3)'),
    ]
    plt.legend(handles=legend_elements, loc='lower right',
               fontsize=8, framealpha=0.85)

    full_equation = f"{poly_str} = 0"
    window_title = (full_equation[:75] + '...') \
        if len(full_equation) > 75 else full_equation
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
    """
    Main pipeline:

        1. Read polynomial coefficients from a file, stdin, or terminal.
        2. Compute all roots via the companion-matrix eigenvalue solver.
        3. Cluster near-coincident roots to estimate multiplicities.
        4. Compute the local triplet (a, m, δ) for each cluster.
        5. Run the residual check and print the diagnostic table.
        6. If any root is non-ok, attempt Newton / Halley refinement
           and re-print the updated table.
        7. Compute the δ-normalized distance field and Newton flow.
        8. Render the field plot with reliability-coded root markers.
    """
    parser = argparse.ArgumentParser(
        description="Compute root fields for a univariate polynomial."
    )
    parser.add_argument('--coeffs', help='Space-separated coefficient string.')
    parser.add_argument(
        '--coeffs-file', help='Path to a coefficient text file.')
    parser.add_argument('--dps', type=int, default=600,
                        help='Arbitrary-precision decimal places (default: 600). '
                             'Increase for higher multiplicities or ill-conditioned polynomials.')
    parser.add_argument('--cluster-tol', type=str, default='1e-22',
                        help='Clustering tolerance for merging roots (default: 1e-22). '
                             'Increase if many m=1 clusters should be higher multiplicities.')
    args = parser.parse_args()

    if args.coeffs and args.coeffs_file:
        parser.error('Use either --coeffs or --coeffs-file, not both.')

    # Apply CLI overrides for precision and clustering
    if args.dps != 600:
        mp.dps = args.dps

    if args.cluster_tol != '1e-10':
        global CLUSTER_TOL, FLOAT64_SAFE_REL_THRESHOLD
        CLUSTER_TOL = mpf(args.cluster_tol)
        FLOAT64_SAFE_REL_THRESHOLD = CLUSTER_TOL * mpf('1e7')

    if args.coeffs:
        coeffs = parse_coefficients_strict(args.coeffs)
    elif args.coeffs_file:
        coeffs = load_coefficients_from_file(args.coeffs_file)
    elif not sys.stdin.isatty():
        coeffs = parse_coefficients_strict(sys.stdin.read())
    else:
        coeffs = get_coefficients_from_user()

    lc = coeffs[0]
    degree = len(coeffs) - 1
    var = 'x' if is_real_polynomial(coeffs) else 'z'

    poly_str = polynomial_to_string(coeffs, var=var)
    print("\nPolynomial Equation:")
    print(f" Degree: {degree}")
    print(f" P({var}) = {poly_str} = 0")

    # Steps 2–4: compute roots, cluster by proximity, compute triplets.
    print("\nComputing clustered roots...")
    roots = compute_roots(coeffs)
    clusters = cluster_roots(roots)
    root_data = [compute_cluster_delta(c, clusters, lc) for c in clusters]

    # Step 5: residual check.
    # Evaluates |P(a)| and |P(a)|·δᵐ for every centroid and assigns
    # a reliability tier ('ok', 'warn', 'bad').
    residuals = compute_residuals(coeffs, root_data)
    format_root_report(root_data, residuals, dps=mp.dps,
                       cluster_tol=CLUSTER_TOL)

    # Step 6: optional Newton / Halley refinement.
    # Triggered automatically whenever at least one root is non-ok.
    # refine_bad_roots polishes the centroid, rebuilds all δ values
    # (because δ is a global quantity that depends on all other roots),
    # and re-runs the residual check so improvement is visible.
    any_nonok = any(r['tier'] != 'ok' for r in residuals)
    if any_nonok:
        print("  Attempting Newton refinement for non-ok roots...")
        root_data, residuals = refine_bad_roots(
            coeffs, root_data, residuals, tiers=('bad', 'warn'))
        print()
        print("  Post-refinement diagnostics:")
        format_root_report(root_data, residuals, dps=mp.dps,
                           cluster_tol=CLUSTER_TOL)

    # Steps 7–8: field computation and plot.
    print(f"Using computation mode : {MODE}")
    print(f"Grid resolution        : {GRID_RESOLUTION}×{GRID_RESOLUTION}")
    print(f"Scaling mode           : "
          f"{'global' if USE_GLOBAL_SCALING else 'root-focused'}")
    print("\nComputing field layout...")

    xs, ys, dist, fu, fv = compute_field(
        coeffs, root_data, mode=MODE, N=GRID_RESOLUTION)

    plot_field(xs, ys, dist, fu, fv, root_data, residuals, poly_str, var)


if __name__ == "__main__":
    main()
