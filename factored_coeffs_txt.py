import sympy as sp
from typing import List

def expand_to_coefficients(factorized_expr, variable='x') -> List[int]:
    """
    Expands a factorized polynomial and returns the list of coefficients
    from highest degree to constant term.
    
    Example input: (x - 5)**10 * (x**2 - 2*x + 2)**25
    """
    x = sp.symbols(variable)
    
    # Expand the expression
    expanded = sp.expand(factorized_expr)
    
    # Get coefficients as Python integers (from highest degree to constant)
    poly = sp.Poly(expanded, x)
    coeffs = [int(c) for c in poly.all_coeffs()]
    
    return coeffs


# ============================
# Example Usage
# ============================

if __name__ == "__main__":
    x = sp.symbols('x')
    
    
    #factored = (x**2 - 2*x + 2)**25 * (x - 5)**10
    factored = (x - 1)**20 * (x + 2)**15 * (x**2 + 1)**10
    #factored = (x - 3)**60

    coeffs = expand_to_coefficients(factored)
    
    print(f"Degree: {len(coeffs) - 1}")
    print("Coefficients:")
    print(coeffs)
    
    # Optional: Save to file
    with open("coeffs.txt", "w") as f:
        f.write(" ".join(map(str, coeffs)))
    print("\nCoefficients saved to coeffs.txt")


"""# (x - 1)^20
factored = (x - 1)**20


# (x - 3)^50
factored = (x - 3)**50


# (x - 1)^20 (x + 2)^15 (x^2 + 1)^10
factored = (x - 1)**20 * (x + 2)**15 * (x**2 + 1)**10


# (x^2 - 1)^30
factored = (x**2 - 1)**30


# (x^2 - 2x + 2)^25 (x - 5)^10
factored = (x**2 - 2*x + 2)**25 * (x - 5)**10


# (x - 1)^40 (x - 2)^30 (x - 3)^20
factored = (x - 1)**40 * (x - 2)**30 * (x - 3)**20
"""
