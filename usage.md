# Usage Guide

This guide explains how to use `generate_coeffs_txt.py` and `root_field.py` with safe coefficient input.

## 1. Generate a coefficients file

To create a file named `coeffs.txt` with default example coefficients:

```bash
cd /Users/ralf/Projects/Python/Param_Poly_Root
python3 generate_coeffs_txt.py
```

This writes the coefficient list into `coeffs.txt` in the current folder.

### Optional: use a custom output path

```bash
python3 generate_coeffs_txt.py --output my_coeffs.txt
```

### Optional: use custom coefficients directly

```bash
python3 generate_coeffs_txt.py --coeffs "1 -100 4875 ..."
```


## 2. Use `root_field.py` with a coefficients file

This is the safest way to pass long coefficient lists without shell truncation problems:

```bash
python3 root_field.py --coeffs-file coeffs.txt
```

If you use a different file name:

```bash
python3 root_field.py --coeffs-file my_coeffs.txt
```


## 3. Use `root_field.py` with standard input

If you prefer to pipe coefficients directly into the script, use standard input:

```bash
echo "1 -100 4875 -154550 ..." | python3 root_field.py
```

or:

```bash
cat coeffs.txt | python3 root_field.py
```

This reads coefficients from stdin and avoids placing a long list directly in the command line.


## 4. Advanced options for precision and clustering

### Precision control (`--dps`)

Increase arbitrary-precision decimal places for higher multiplicities or ill-conditioned polynomials:

```bash
python3 root_field.py --coeffs-file coeffs.txt --dps 800
```

### Clustering tolerance (`--cluster-tol`)

Adjust how roots are merged into clusters. Increase if many `m=1` clusters should be higher multiplicities:

```bash
python3 root_field.py --coeffs-file coeffs.txt --cluster-tol 1e-22
```

**Example:** The default example polynomial (degree 60) shows 51 clusters with default tolerance, but correctly merges to 3 clusters with `--cluster-tol 1e-22`.

**Important:** `dps` and `cluster-tol` are interdependent. The noise floor is ≈ `10^(-dps / m_max)`. `cluster-tol` must stay above this floor, or true multiple roots will split artificially.

### Combined usage

```bash
python3 root_field.py --coeffs-file coeffs.txt --dps 800 --cluster-tol 1e-22
```

The script provides diagnostic hints if many `m=1` clusters are detected, suggesting parameter adjustments.


## 5. Notes

- `coeffs.txt` should contain a single line with coefficients separated by spaces or commas.
- Coefficients must be provided in descending degree order, starting with the coefficient of `x^n`.
- The script supports real and complex coefficients.
- Use `--coeffs-file` when large integers or long coefficient lists are involved.

## 6. Example workflow

```bash
cd /Users/ralf/Projects/Python/Param_Poly_Root
python3 generate_coeffs_txt.py
python3 root_field.py --coeffs-file coeffs.txt
```

This workflow keeps the coefficient input safe and reproducible.
