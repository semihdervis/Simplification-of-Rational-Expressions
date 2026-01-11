# Simplification of Rational Expressions

A **from-scratch** Python implementation for simplifying rational expressions `p(x)/g(x)` where both `p(x)` and `g(x)` are polynomials in a single variable. This project implements multiple algorithms without relying on computer algebra systems—only basic Python operations.

## Features

### Polynomial Arithmetic
- Full polynomial class implementation with addition, subtraction, multiplication, and division
- Coefficient normalization and degree computation
- Horner's method for efficient polynomial evaluation
- Content and primitive part extraction

### GCD Algorithms
Four different methods for computing polynomial GCD:

| Method | Description | Best For |
|--------|-------------|----------|
| **Heuristic GCD** | Interpolation-based using evaluation at strategic points | General use, fast |
| **Euclidean GCD** | Classical polynomial Euclidean algorithm | Numerical stability |
| **Factorization** | Factor both polynomials and cancel common factors | Exact integer coefficients |
| **Sylvester Matrix** | Linear algebra approach using null space computation | Theoretical completeness |

### Polynomial Factorization
- Square-free decomposition
- Zassenhaus algorithm for irreducible factorization over integers
- Trial division for small-degree polynomials
- Support for repeated factors with multiplicity tracking

### Expression Parsing
Flexible parser supporting:
- Standard notation: `x^3 + 2x^2 - 5x + 1`
- Power expressions: `(x + 1)^5`
- Products: `(x - 1)(x + 2)^3`
- Implicit multiplication: `2x`, `3x^2`

## Installation

No external dependencies required for core functionality:

```bash
git clone https://github.com/yourusername/Simplification-of-Rational-Expressions.git
cd Simplification-of-Rational-Expressions
python simplification.py
```

**Optional:** Install SymPy for result verification:
```bash
pip install sympy
```

## Usage

### Basic Simplification

```python
from simplification import simplify_rational, format_result

# Simplify (x^2 - 1) / (x - 1)
num, den, method = simplify_rational("x^2 - 1", "x - 1")
print(format_result(num, den))  # Output: x + 1
```

### Using Specific Methods

```python
from simplification import (
    simplify_heuristic,
    simplify_euclidean,
    simplify_factorization_method,
    simplify_sylvester
)

numerator = "(x + 1)^3"
denominator = "(x + 1)^2"

# Heuristic (Interpolation-based)
num, den, method = simplify_heuristic(numerator, denominator)

# Euclidean Algorithm
num, den, method = simplify_euclidean(numerator, denominator)

# Factorization-based
num, den, method = simplify_factorization_method(numerator, denominator)

# Sylvester Matrix
result = simplify_sylvester(numerator, denominator)
```

### Working with Polynomials Directly

```python
from simplification import Polynomial

# Create polynomials: x^2 - 1 and x - 1
p = Polynomial([-1, 0, 1])  # coefficients [a_0, a_1, a_2] = -1 + 0x + 1x^2
q = Polynomial([-1, 1])     # -1 + x

# Polynomial operations
sum_poly = p + q
diff_poly = p - q
prod_poly = p * q
quotient, remainder = p.divmod(q)

# GCD computation
gcd_poly, _ = p.gcd(q)

# Factorization
content, factors = p.factor()
```

## Algorithm Details

### Heuristic GCD (Interpolation-Based)
Evaluates polynomials at carefully chosen points, computes integer GCD of the values, and reconstructs the polynomial GCD through interpolation. Includes:
- Smart evaluation point selection based on coefficient bounds
- Fast coprime detection for early termination
- Binary GCD for faster integer operations

### Euclidean Algorithm
Classical polynomial division algorithm adapted for floating-point coefficients with numerical stability improvements:
- Coefficient tolerance handling
- Subresultant optimization for better numerical behavior

### Factorization Method
1. Parse expressions and expand to polynomial form
2. Factor both numerator and denominator
3. Cancel common factors while tracking multiplicities
4. Reconstruct simplified expression

### Sylvester Matrix
Uses linear algebra to find common factors:
1. Construct Sylvester matrix from polynomial coefficients
2. Compute null space using Gaussian elimination with pivoting
3. Extract common factor from null space vectors
4. Handle numerical precision issues with iterative refinement

## Examples

The script includes test cases demonstrating various simplification scenarios:

```
Input: (x^2 - 1) / (x - 1)
Result: x + 1

Input: (x^3 - 8) / (x - 2)
Result: x^2 + 2x + 4

Input: (x + 1)^5 / (x + 1)^3
Result: (x + 1)^2

Input: (x^2 - 4)(x + 3) / (x - 2)(x^2 + 5x + 6)
Result: 1
```

## Project Structure

```
Simplification-of-Rational-Expressions/
├── simplification.py    # Main implementation
└── README.md           # This file
```

### Key Components in `simplification.py`

| Component | Description |
|-----------|-------------|
| `Polynomial` | Core polynomial class with all arithmetic operations |
| `parse_polynomial()` | Expression string parser |
| `simplify_rational()` | Main simplification entry point |
| `simplify_heuristic()` | Interpolation-based GCD simplification |
| `simplify_euclidean()` | Euclidean algorithm simplification |
| `simplify_factorization_method()` | Factorization-based simplification |
| `simplify_sylvester()` | Sylvester matrix simplification |
| `build_sylvester_matrix()` | Constructs Sylvester matrix |
| `gaussian_elimination_with_pivoting()` | Linear algebra utilities |

## Limitations

- Designed for single-variable polynomials only
- Floating-point arithmetic may introduce small numerical errors
- Very high-degree polynomials (>100) may have precision issues
- Complex coefficients are not supported

## Testing

Run the built-in test suite:

```bash
python simplification.py
```

If SymPy is installed, results are automatically verified against SymPy's `simplify()` function.

## Contributing

Contributions are welcome! Areas for potential improvement:
- Support for multivariate polynomials
- Arbitrary precision arithmetic
- Additional factorization algorithms
- Performance optimizations for high-degree polynomials

## License

This project is open source. Feel free to use, modify, and distribute.

## References

- Knuth, D. E. *The Art of Computer Programming, Vol. 2: Seminumerical Algorithms*
- Geddes, K. O., Czapor, S. R., & Labahn, G. *Algorithms for Computer Algebra*
- von zur Gathen, J., & Gerhard, J. *Modern Computer Algebra*
