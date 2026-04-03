import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, mpc, mpf, matrix, eig

# ========================= SETTINGS ========================= #
mp.dps = 400

# ========================= INPUT ========================= #
def parse_coefficients(text):
    coeffs = []
    for token in text.strip().split():
        token = token.replace('I', 'i').replace('i', 'j')  # normalize
        token = token.replace(' ', '')  # remove spaces

        try:
            # convert string to Python complex first
            z = complex(token)
            coeffs.append(mpc(mpf(z.real), mpf(z.imag)))
        except ValueError:
            # handle pure "j" or "-j" like before
            if token in ('j', '+j'):
                coeffs.append(mpc(0, 1))
            elif token == '-j':
                coeffs.append(mpc(0, -1))
            else:
                # maybe a real number
                coeffs.append(mpc(mpf(token), 0))

    # remove leading zeros
    while len(coeffs) > 1 and coeffs[0] == 0:
        coeffs.pop(0)
    return coeffs

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
def compute_cluster_delta(cluster, clusters):
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
    return a, m, delta

# ========================= FIELD ========================= #
def compute_field(coeffs, root_data, N=200):
    R = max([abs(a) + delta for a, _, delta in root_data] + [1]) * 1.2

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

            flow_u[j, i] = float(mp.re(w))
            flow_v[j, i] = float(mp.im(w))

    return xs, ys, dist, flow_u, flow_v

# ========================= PLOT ========================= #
def plot_field(xs, ys, dist, flow_u, flow_v, root_data):
    X, Y = np.meshgrid(xs, ys)

    plt.figure(figsize=(10,9))

    plt.imshow(dist, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin='lower')

    plt.streamplot(X, Y, flow_u, flow_v, density=1.2)

    for a, m, delta in root_data:
        ar = float(mp.re(a))
        ai = float(mp.im(a))
        dr = float(delta)

        plt.plot(ar, ai, 'ro')
        circle = plt.Circle((ar, ai), dr, fill=False)
        plt.gca().add_patch(circle)

        plt.text(ar, ai, f"m={m}\nδ={mp.nstr(delta,3)}",
                 fontsize=8, ha='center', va='bottom')

    plt.gca().set_aspect('equal')
    plt.title("δ-Root Field")
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.show()

# ========================= MAIN ========================= #
def main():
    text = input("Coefficients: ")
    coeffs = parse_coefficients(text)
    print("\nComputing clustered roots...")
    roots = compute_roots(coeffs)
    clusters = cluster_roots(roots)
    root_data = [compute_cluster_delta(c, clusters) for c in clusters]

    print("\nClustered roots:")
    for a, m, delta in root_data:
        print(f"a={mp.nstr(a,6)}, m={m}, δ={mp.nstr(delta,6)}")

    print("\nComputing field layout...")
    xs, ys, dist, fu, fv = compute_field(coeffs, root_data)
    plot_field(xs, ys, dist, fu, fv, root_data)

if __name__ == "__main__":
    main()
