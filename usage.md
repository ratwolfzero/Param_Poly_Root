# Usage Guide

This guide explains how to use `faxtored_coeffs_txt.py` and `root_field.py` with safe coefficient input.

## 1. Generate a coefficients file from a factorized polynomial

The repository includes `faxtored_coeffs_txt.py`, which expands a factorized polynomial into a coefficient list and writes it to `coeffs.txt`.

### Use the script

1. Open `faxtored_coeffs_txt.py`.
2. Set the factorized polynomial expression in the `factored = ...` line.
3. Run the script:

```bash
cd /Users/ralf/Projects/Python/Param_Poly_Root
python3 faxtored_coeffs_txt.py
```

This writes the expanded coefficients into `coeffs.txt` in the current folder.

### Example factorization

In `faxtored_coeffs_txt.py`, you can use expressions such as:

```python
x = sp.symbols('x')
factored = (x - 5)**10 * (x**2 - 2*x + 2)**25
# factored = (x - 1)**40 * (x - 2)**30 * (x - 3)**20
```

Then run the script to generate the coefficients file.

## 2. Use `root_field.py` with a coefficients file

Once `coeffs.txt` exists, run:

```bash
python3 root_field.py --coeffs-file coeffs.txt
```

If you have a different filename:

```bash
python3 root_field.py --coeffs-file my_coeffs.txt
```

## 3. Use `root_field.py` with standard input

If you prefer to pipe coefficients directly into the script:

```bash
echo "1 -100 4875 -154550 ..." | python3 root_field.py
```

or:

```bash
cat coeffs.txt | python3 root_field.py
```

This avoids placing a long list directly in the shell command.

## 4. Advanced options for precision and clustering

### Precision control (`--dps`)

Increase arbitrary-precision decimal places for higher multiplicities or ill-conditioned polynomials:

```bash
python3 root_field.py --coeffs-file coeffs.txt --dps 800
```

### Clustering tolerance (`--cluster-tol`)

Adjust how roots are merged into clusters. Increase this when many `m=1` clusters should be grouped into higher multiplicities:

```bash
python3 root_field.py --coeffs-file coeffs.txt --cluster-tol 1e-22
```

**Important:** `dps` and `cluster-tol` are interdependent. The noise floor is roughly `10^(-dps / m_max)`, so `cluster-tol` should remain above that floor to avoid splitting true multiple roots.

### Combined usage

```bash
python3 root_field.py --coeffs-file coeffs.txt --dps 800 --cluster-tol 1e-22
```

The script prints diagnostic hints if many `m=1` clusters are found, suggesting better tuning.

## 5. Notes

- `coeffs.txt` should contain a single line with coefficients separated by spaces or commas.
- Coefficients must be provided in descending degree order, starting with the coefficient of `x^n`.
- `root_field.py` supports real and complex coefficients.
- Use the file-based input path when coefficients are large or too long for a shell command.

## 6. Example workflow

```bash
cd /Users/ralf/Projects/Python/Param_Poly_Root
python3 faxtored_coeffs_txt.py
python3 root_field.py --coeffs-file coeffs.txt
```

This workflow uses a factorized polynomial definition to generate safe coefficient input and then analyses the roots with `root_field.py`.
