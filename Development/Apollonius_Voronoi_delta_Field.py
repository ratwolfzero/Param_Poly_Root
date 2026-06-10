import numpy as np
import matplotlib.pyplot as plt

def analyze_polynomial_from_coefficients(coeffs, cluster_tol=1e-4):
    """
    Calculates the local parameterization triplets (a, m, delta) directly from polynomial coefficients.
    coeffs: List/array of coefficients in descending order (e.g., [1, 0, -1] for x^2 - 1)
    cluster_tol: Tolerance radius to merge numerically split roots into true multiplicities.
    """
    # 1. Find numerical roots
    raw_roots = np.roots(coeffs)
    
    # 2. Cluster roots to determine the algebraic multiplicity 'm'
    grouped_roots = []
    used = np.zeros(len(raw_roots), dtype=bool)
    
    for i in range(len(raw_roots)):
        if used[i]:
            continue
        # Find all roots lying within the tolerance radius
        distances = np.abs(raw_roots - raw_roots[i])
        cluster_indices = np.where((distances < cluster_tol) & (~used))
        
        # Define the root location 'a' as the centroid of the cluster
        cluster_mean = np.mean(raw_roots[cluster_indices])
        multiplicity = len(cluster_indices[0])
        
        grouped_roots.append((cluster_mean, multiplicity))
        used[cluster_indices] = True
        
    # 3. Calculate Delta (delta_i) based on global interactions (Section 9)
    # The leading coefficient of the polynomial (c_n) must be included
    leading_coeff = np.abs(coeffs[0])
    triplets = []
    
    for i, (a_i, m_i) in enumerate(grouped_roots):
        product_distances = 1.0
        for j, (a_j, m_j) in enumerate(grouped_roots):
            if i != j:
                product_distances *= (np.abs(a_i - a_j))**m_j
        
        # Include the leading coefficient into the total product
        total_product = leading_coeff * product_distances
        
        # Delta is the inverse root of order m_i
        delta_i = (total_product)**(-1.0 / m_i)
        triplets.append({'a': a_i, 'm': m_i, 'delta': delta_i})
        
    return triplets

# ==========================================
# APPLICATION EXAMPLE / INPUT
# ==========================================
# Example: P(x) = (x - 1)^2 * (x + 2) = x^3 - 3x + 2
# Coefficients in descending order: [1, 0, -3, 2]
polynomial_coefficients = [1, 0, -3, 2- 1, 1, 1,-9, 7]

# Execute analysis
triplets = analyze_polynomial_from_coefficients(polynomial_coefficients)

# Display results in the console
print("Detected Local Parameter Triplets (a, m, delta):")
for idx, t in enumerate(triplets):
    print(f"Root {idx+1}: a = {t['a']:.4f}, m = {t['m']}, delta = {t['delta']:.4f}")

# ==========================================
# VISUALIZATION (SECTION 9)
# ==========================================
# Dynamically adjust the plot boundaries based on the root positions
all_roots = [t['a'] for t in triplets]
real_parts = [r.real for r in all_roots]
imag_parts = [r.imag for r in all_roots]

x_min, x_max = min(real_parts) - 1.5, max(real_parts) + 1.5
y_min, y_max = min(imag_parts) - 1.5, max(imag_parts) + 1.5

x = np.linspace(x_min, x_max, 400)
y = np.linspace(y_min, y_max, 400)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Calculate the delta-normalized distance field
field = np.zeros_like(Z, dtype=float) + np.inf
for t in triplets:
    rho_i = np.abs(Z - t['a']) / t['delta']
    field = np.minimum(field, rho_i)

log_field = np.log10(field)

# Initialize plotting
plt.figure(figsize=(8, 7))
contour = plt.contourf(X, Y, log_field, levels=30, cmap='twilight_r')
plt.colorbar(contour, label=r'$\log_{10}(\min \rho_i(z))$')

# Draw contour lines (The log10(rho)=0 level marks the delta-circle boundaries)
plt.contour(X, Y, log_field, levels=[0.0], colors='white', linewidths=2, linestyles='--')

# Plot roots and their exact delta-disks
for t in triplets:
    plt.scatter(t['a'].real, t['a'].imag, color='red', s=t['m']*50, zorder=5)
    circle = plt.Circle((t['a'].real, t['a'].imag), t['delta'], color='white', fill=False, alpha=0.5, linewidth=1.5)
    plt.gca().add_patch(circle)

# Using raw string r"..." to cleanly bypass escape sequence syntax warnings
plt.title(r"Apollonius-Voronoi $\delta$-Field from Coefficients (Section 9)")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
