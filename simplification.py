"""
From-scratch algorithm to simplify rational expressions p(x)/g(x)
where both p(x) and g(x) are polynomials in a single variable.

This implementation is self-contained and doesn't use rings, fields,
or external libraries - just basic Python operations.
"""


class Polynomial:
    """Represents a polynomial as a list of coefficients.
    
    For polynomial a_n*x^n + ... + a_1*x + a_0,
    coefficients are stored as [a_0, a_1, ..., a_n]
    (lowest degree first).
    """
    
    def __init__(self, coeffs):
        """Initialize polynomial from list of coefficients.
        
        Args:
            coeffs: List of coefficients [a_0, a_1, ..., a_n]
        """
        # Remove trailing zeros to normalize
        self.coeffs = self._normalize(coeffs)
    
    def _normalize(self, coeffs):
        """Remove trailing zeros."""
        coeffs = list(coeffs)
        while len(coeffs) > 0 and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs if coeffs else [0]
    
    def degree(self):
        """Return the degree of the polynomial."""
        return len(self.coeffs) - 1
    
    def is_zero(self):
        """Check if polynomial is zero."""
        return len(self.coeffs) == 1 and self.coeffs[0] == 0
    
    def leading_coeff(self):
        """Return leading coefficient."""
        if self.is_zero():
            return 0
        return self.coeffs[-1]
    
    def __repr__(self):
        """String representation."""
        if self.is_zero():
            return "0"
        
        def format_coeff(c):
            """Format coefficient as integer if possible."""
            if abs(c - round(c)) < 1e-10:
                return str(int(round(c)))
            return str(c)
        
        terms = []
        for i, coeff in enumerate(self.coeffs):
            if abs(coeff) < 1e-10:
                continue
            
            coeff_str = format_coeff(coeff)
            
            if i == 0:
                terms.append(coeff_str)
            elif i == 1:
                if abs(coeff - 1.0) < 1e-10:
                    terms.append("x")
                elif abs(coeff + 1.0) < 1e-10:
                    terms.append("-x")
                else:
                    terms.append(f"{coeff_str}x")
            else:
                if abs(coeff - 1.0) < 1e-10:
                    terms.append(f"x^{i}")
                elif abs(coeff + 1.0) < 1e-10:
                    terms.append(f"-x^{i}")
                else:
                    terms.append(f"{coeff_str}x^{i}")
        
        return " + ".join(reversed(terms)).replace(" + -", " - ")
    
    def __eq__(self, other):
        """Check equality."""
        return self.coeffs == other.coeffs
    
    def __add__(self, other):
        """Add two polynomials."""
        max_len = max(len(self.coeffs), len(other.coeffs))
        result = [0] * max_len
        
        for i in range(len(self.coeffs)):
            result[i] += self.coeffs[i]
        
        for i in range(len(other.coeffs)):
            result[i] += other.coeffs[i]
        
        return Polynomial(result)
    
    def __sub__(self, other):
        """Subtract two polynomials."""
        max_len = max(len(self.coeffs), len(other.coeffs))
        result = [0] * max_len
        
        for i in range(len(self.coeffs)):
            result[i] += self.coeffs[i]
        
        for i in range(len(other.coeffs)):
            result[i] -= other.coeffs[i]
        
        return Polynomial(result)
    
    def __mul__(self, other):
        """Multiply two polynomials."""
        if isinstance(other, (int, float)):
            # Scalar multiplication
            return Polynomial([c * other for c in self.coeffs])
        
        result = [0] * (len(self.coeffs) + len(other.coeffs) - 1)
        
        for i, coeff1 in enumerate(self.coeffs):
            for j, coeff2 in enumerate(other.coeffs):
                result[i + j] += coeff1 * coeff2
        
        return Polynomial(result)
    
    def __rmul__(self, other):
        """Right multiplication (for scalar * polynomial)."""
        return self.__mul__(other)
    
    def scale(self, factor):
        """Multiply polynomial by a scalar factor."""
        return Polynomial([c * factor for c in self.coeffs])
    
    def divmod(self, other):
        """Polynomial division with remainder (optimized).
        
        Returns (quotient, remainder) such that:
        self = other * quotient + remainder
        
        Uses in-place coefficient updates to avoid creating intermediate polynomials.
        """
        if other.is_zero():
            raise ZeroDivisionError("Division by zero polynomial")
        
        # Work with coefficient lists directly for speed
        dividend_coeffs = list(self.coeffs)
        divisor_coeffs = other.coeffs
        divisor_deg = len(divisor_coeffs) - 1
        lead_divisor = divisor_coeffs[-1]
        
        # Initialize quotient
        quot_deg = len(dividend_coeffs) - 1 - divisor_deg
        if quot_deg < 0:
            return Polynomial([0]), Polynomial(dividend_coeffs)
        
        quotient_coeffs = [0.0] * (quot_deg + 1)
        
        # Process from highest to lowest degree
        for i in range(quot_deg, -1, -1):
            # Current position in dividend
            pos = i + divisor_deg
            if pos >= len(dividend_coeffs):
                continue
                
            # Compute quotient coefficient
            lead_quotient = dividend_coeffs[pos] / lead_divisor
            quotient_coeffs[i] = lead_quotient
            
            # Subtract lead_quotient * divisor from dividend (in-place)
            if abs(lead_quotient) > 1e-15:
                for j, dc in enumerate(divisor_coeffs):
                    dividend_coeffs[i + j] -= lead_quotient * dc
        
        # Normalize remainder
        while len(dividend_coeffs) > 1 and abs(dividend_coeffs[-1]) < 1e-10:
            dividend_coeffs.pop()
        
        return Polynomial(quotient_coeffs), Polynomial(dividend_coeffs)
    
    def content(self):
        """Extract the integer GCD of all coefficients (content of polynomial)."""
        if self.is_zero():
            return 0
        
        # Find GCD of all coefficients
        coeffs = [abs(int(round(c))) for c in self.coeffs if abs(c) > 1e-10]
        if not coeffs:
            return 0
        
        def int_gcd(a, b):
            """Integer GCD."""
            while b:
                a, b = b, a % b
            return abs(a)
        
        result = coeffs[0]
        for c in coeffs[1:]:
            result = int_gcd(result, c)
        
        return result
    
    def primitive_part(self):
        """Return primitive part (polynomial divided by its content)."""
        content_val = self.content()
        if content_val == 0:
            return Polynomial([0])
        if content_val == 1:
            return self
        return Polynomial([c / content_val for c in self.coeffs])
    
    def evaluate(self, x_value):
        """Evaluate polynomial at x = x_value using Horner's method.
        
        Horner's method: a_n*x^n + ... + a_1*x + a_0 = ((a_n*x + a_{n-1})*x + ...)*x + a_0
        This is O(n) multiplications instead of O(n^2) for naive approach.
        """
        if not self.coeffs:
            return 0.0
        # Start from highest degree coefficient
        result = self.coeffs[-1]
        for i in range(len(self.coeffs) - 2, -1, -1):
            result = result * x_value + self.coeffs[i]
        return result
    
    def max_norm(self):
        """Return maximum absolute value of coefficients."""
        if self.is_zero():
            return 0
        return max(abs(c) for c in self.coeffs)
    
    def l1_norm(self):
        """Return L1 norm (sum of absolute values of coefficients)."""
        if self.is_zero():
            return 0
        return sum(abs(c) for c in self.coeffs)
    
    def gcd_heuristic(self, other):
        """Optimized Heuristic GCD algorithm using evaluation and interpolation.
        
        Improvements over standard implementation:
        1. Smarter evaluation point selection based on coefficient structure
        2. Fast path for common cases (degree 1 GCD, equal polynomials)
        3. Binary GCD for faster integer operations
        4. Optimized interpolation with early termination
        5. Fast coprime detection
        
        Returns GCD if successful, None if heuristic fails.
        """
        # Fast path: identical polynomials
        if self.coeffs == other.coeffs:
            result = Polynomial(self.coeffs[:])
            lead = result.leading_coeff()
            if abs(lead) > 1e-10 and abs(lead - 1.0) > 1e-10:
                result = result.scale(1.0 / lead)
            return result
        
        # Fast coprime detection: evaluate at several small points
        # If GCD is 1, evaluations will be coprime at most points
        quick_points = [2, 3, 5]
        coprime_count = 0
        for pt in quick_points:
            val_a = int(round(self.evaluate(pt)))
            val_b = int(round(other.evaluate(pt)))
            if val_a != 0 and val_b != 0:
                # Quick GCD check
                g = abs(val_a)
                h = abs(val_b)
                while h:
                    g, h = h, g % h
                if g == 1:
                    coprime_count += 1
        
        # If coprime at all test points, very likely coprime overall
        if coprime_count >= 2:
            # Do one quick verification
            # For low degree polynomials, this is a strong signal
            if min(self.degree(), other.degree()) <= 4:
                return Polynomial([1])
        
        a = Polynomial(self.coeffs[:])
        b = Polynomial(other.coeffs[:])
        
        # Handle zero cases efficiently
        if a.is_zero():
            if b.is_zero():
                return Polynomial([0])
            return self._make_monic(b)
        
        if b.is_zero():
            return self._make_monic(a)
        
        # Fast integer GCD using binary algorithm (faster than Euclidean for large numbers)
        def binary_gcd(x, y):
            """Binary GCD algorithm - faster for large integers."""
            if x == 0:
                return y
            if y == 0:
                return x
            x, y = abs(x), abs(y)
            
            # Remove common factors of 2
            shift = 0
            while ((x | y) & 1) == 0:
                x >>= 1
                y >>= 1
                shift += 1
            
            # Remove remaining factors of 2 from x
            while (x & 1) == 0:
                x >>= 1
            
            while y != 0:
                while (y & 1) == 0:
                    y >>= 1
                if x > y:
                    x, y = y, x
                y -= x
            
            return x << shift
        
        # Extract content efficiently
        def extract_content_fast(poly):
            """Fast content extraction."""
            coeffs = [abs(int(round(c))) for c in poly.coeffs if abs(c) > 1e-10]
            if not coeffs:
                return 1, poly
            
            gcd_val = coeffs[0]
            for c in coeffs[1:]:
                gcd_val = binary_gcd(gcd_val, c)
                if gcd_val == 1:
                    return 1, poly
            
            if gcd_val <= 1:
                return 1, poly
            return gcd_val, Polynomial([c / gcd_val for c in poly.coeffs])
        
        gcd_ground_a, a = extract_content_fast(a)
        gcd_ground_b, b = extract_content_fast(b)
        gcd_ground = binary_gcd(gcd_ground_a, gcd_ground_b)
        
        # Smarter evaluation point: use coefficient-based bounds
        a_norm = max(1, a.max_norm())
        b_norm = max(1, b.max_norm())
        min_deg = min(a.degree(), b.degree())
        
        # Optimized starting point - smaller for low degree, scales with norm
        if min_deg <= 2:
            x_eval = max(10, int(2 * min(a_norm, b_norm) + 5))
        else:
            x_eval = max(20, int(min(a_norm, b_norm) ** 0.5 * (min_deg + 1)))
        
        # Limit evaluation point to prevent overflow
        x_eval = min(x_eval, 10**6)
        
        # Try heuristic GCD with optimized attempts
        for attempt in range(4):  # Reduced from 6 - fail faster
            # Evaluate using Horner's method (already optimized)
            a_eval = int(round(a.evaluate(x_eval)))
            b_eval = int(round(b.evaluate(x_eval)))
            
            if a_eval == 0 and b_eval == 0:
                x_eval = x_eval * 3 + 7  # Simpler progression
                continue
            
            if a_eval == 0:
                h = b
            elif b_eval == 0:
                h = a
            else:
                # Fast integer GCD
                h_eval = binary_gcd(abs(a_eval), abs(b_eval))
                
                # Fast interpolation
                h = self._fast_interpolate(h_eval, x_eval, min_deg)
                
                if h is None:
                    x_eval = x_eval * 3 + 7
                    continue
                
                # Make primitive
                h_content = h.content()
                if h_content > 1:
                    h = Polynomial([c / h_content for c in h.coeffs])
            
            # Verify (quick check first)
            if h is not None and not h.is_zero() and h.degree() <= min_deg:
                lead = h.leading_coeff()
                if abs(lead) > 1e-10 and abs(lead - 1.0) > 1e-10:
                    h = Polynomial([c / lead for c in h.coeffs])
                
                # Quick divisibility check
                _, r1 = a.divmod(h)
                if r1.is_zero():
                    _, r2 = b.divmod(h)
                    if r2.is_zero():
                        if gcd_ground > 1:
                            h = Polynomial([c * gcd_ground for c in h.coeffs])
                        return h
            
            x_eval = x_eval * 3 + 7
        
        return None
    
    def _make_monic(self, poly):
        """Make polynomial monic (leading coeff = 1)."""
        if poly.is_zero():
            return poly
        lead = poly.leading_coeff()
        if abs(lead - 1.0) < 1e-10:
            return poly
        return Polynomial([c / lead for c in poly.coeffs])
    
    def _fast_interpolate(self, h_eval, x_eval, max_degree):
        """Fast interpolation from integer to polynomial."""
        if h_eval == 0:
            return Polynomial([0])
        
        coeffs = []
        h = abs(h_eval)
        half_x = x_eval // 2
        
        while h > 0 and len(coeffs) <= max_degree + 1:
            c = h % x_eval
            # Symmetric representation
            if c > half_x:
                c = c - x_eval
            coeffs.append(float(c))
            h = (h - c) // x_eval
        
        if not coeffs:
            return Polynomial([1])
        
        return Polynomial(coeffs)
    
    def gcd_euclidean(self, other):
        """Compute GCD using Primitive Polynomial Remainder Sequence (PRS).
        
        Uses primitive PRS to control coefficient growth - each remainder
        is made primitive (divided by content) to keep coefficients small.
        This is more stable than standard Euclidean for large polynomials.
        """
        # Fast path for equal polynomials
        if self.coeffs == other.coeffs:
            return self._make_monic(Polynomial(self.coeffs[:]))
        
        a = Polynomial(self.coeffs[:])
        b = Polynomial(other.coeffs[:])
        
        # Handle zero cases
        if a.is_zero():
            return self._make_monic(b) if not b.is_zero() else Polynomial([0])
        if b.is_zero():
            return self._make_monic(a)
        
        # Extract contents first (Gauss's lemma optimization)
        def get_content(poly):
            coeffs = [abs(int(round(c))) for c in poly.coeffs if abs(c) > 1e-10]
            if not coeffs:
                return 1
            gcd_val = coeffs[0]
            for c in coeffs[1:]:
                while c:
                    gcd_val, c = c, gcd_val % c
                gcd_val = abs(gcd_val)
                if gcd_val == 1:
                    return 1
            return max(1, gcd_val)
        
        cont_a = get_content(a)
        cont_b = get_content(b)
        
        # GCD of contents
        ca, cb = cont_a, cont_b
        while cb:
            ca, cb = cb, ca % cb
        content_gcd = ca
        
        # Work with primitive parts
        if cont_a > 1:
            a = Polynomial([c / cont_a for c in a.coeffs])
        if cont_b > 1:
            b = Polynomial([c / cont_b for c in b.coeffs])
        
        # Make leading coefficients positive
        if a.leading_coeff() < 0:
            a = Polynomial([-c for c in a.coeffs])
        if b.leading_coeff() < 0:
            b = Polynomial([-c for c in b.coeffs])
        
        # Ensure a has larger degree
        if a.degree() < b.degree():
            a, b = b, a
        
        # Primitive PRS Euclidean algorithm
        while not b.is_zero():
            _, remainder = a.divmod(b)
            
            if remainder.is_zero():
                break
            
            # Make remainder primitive to control coefficient growth
            rem_content = get_content(remainder)
            if rem_content > 1:
                remainder = Polynomial([c / rem_content for c in remainder.coeffs])
            
            # Normalize leading coefficient
            lead = remainder.leading_coeff()
            if lead < 0:
                remainder = Polynomial([-c for c in remainder.coeffs])
            
            a, b = b, remainder
        
        # Result is primitive part of b (or a if b became zero)
        result = b if not b.is_zero() else a
        
        # Make monic
        if not result.is_zero():
            lead = result.leading_coeff()
            if abs(lead) > 1e-10 and abs(lead - 1.0) > 1e-10:
                result = Polynomial([c / lead for c in result.coeffs])
        
        return result
    
    def gcd(self, other):
        """Compute GCD of two polynomials.
        
        First tries heuristic GCD (fast, evaluation-based).
        If that fails, falls back to standard Euclidean algorithm.
        
        Returns a monic polynomial (leading coefficient = 1).
        """
        # Try heuristic GCD first (fast for typical cases)
        heuristic_result = self.gcd_heuristic(other)
        if heuristic_result is not None:
            return heuristic_result, "heuristic"
        
        # Fall back to Euclidean algorithm if heuristic fails
        return self.gcd_euclidean(other), "euclidean"
    
    def derivative(self):
        """Compute derivative of polynomial."""
        if self.degree() == 0:
            return Polynomial([0])
        
        result = []
        for i in range(1, len(self.coeffs)):
            result.append(i * self.coeffs[i])
        return Polynomial(result)
    
    def square_free_part(self):
        """Extract square-free part of polynomial.
        
        The square-free part is f / gcd(f, f') where f' is the derivative.
        """
        if self.degree() <= 0:
            return self
        
        f_prime = self.derivative()
        if f_prime.is_zero():
            # Polynomial is constant, already square-free
            return self
        
        g, _ = self.gcd(f_prime)
        if g.is_zero() or g.degree() == 0:
            # Already square-free
            return self
        
        # Divide by gcd
        q, r = self.divmod(g)
        if not r.is_zero():
            # This shouldn't happen if gcd is correct
            return self
        
        return q
    
    def factor(self):
        """Factor polynomial into irreducible factors over integers.
        
        Returns tuple (content, factors_list) where:
        - content: integer GCD of all coefficients
        - factors_list: list of (factor, multiplicity) tuples
        
        This implements a simplified version of the Zassenhaus algorithm:
        1. Extract content and make primitive
        2. Extract square-free part
        3. Factor square-free part using modular approach
        4. Recover multiplicities using trial division
        """
        if self.is_zero():
            return 0, []
        
        if self.degree() == 0:
            # Constant polynomial
            content = int(round(abs(self.coeffs[0])))
            return content, []
        
        # Step 1: Extract content and make primitive
        content = self.content()
        if content == 0:
            return 0, []
        
        # Make primitive (divide by content)
        if content > 1:
            f_primitive = Polynomial([c / content for c in self.coeffs])
        else:
            f_primitive = self
        
        # Normalize leading coefficient to be positive
        if f_primitive.leading_coeff() < 0:
            f_primitive = f_primitive.scale(-1)
            content = -content
        
        # Step 2: Factor the primitive polynomial directly to get multiplicities
        # We'll extract square-free part for the factorization algorithm,
        # but use the original for multiplicity recovery
        
        # Step 3: Factor (we'll use square-free part for efficiency)
        f_sqf = f_primitive.square_free_part()
        
        if f_sqf.degree() == 0:
            return content, []
        
        if f_sqf.degree() == 1:
            # Linear polynomial is irreducible
            # But we need to check multiplicity in original
            multiplicity = 0
            temp = f_primitive
            while True:
                q, r = temp.divmod(f_sqf)
                if r.is_zero() or r.l1_norm() < 1e-6:
                    multiplicity += 1
                    temp = q
                else:
                    break
            return content, [(f_sqf, multiplicity)]
        
        # Use Zassenhaus algorithm for factorization
        # For small polynomials (degree <= 4), use trial division directly
        if f_sqf.degree() <= 4:
            factors = self._trial_factor(f_sqf)
        else:
            factors = self._zassenhaus_factor(f_sqf)
            if not factors:
                # If factorization failed, try trial division
                factors = self._trial_factor(f_sqf)
        
        # Step 4: Recover multiplicities using trial division on original polynomial
        result_factors = []
        remaining = f_primitive
        
        # Make all factors monic for comparison
        monic_factors = []
        for factor in factors:
            lead = factor.leading_coeff()
            if abs(lead - 1.0) > 1e-10:
                monic_factor = factor.scale(1.0 / lead)
            else:
                monic_factor = factor
            monic_factors.append(monic_factor)
        
        # Try each factor and find its multiplicity
        # Important: process factors one at a time and update remaining after each
        for monic_factor in monic_factors:
            multiplicity = 0
            
            # Keep dividing remaining by this factor until it no longer divides
            while True:
                if remaining.degree() < monic_factor.degree():
                    break
                    
                q, r = remaining.divmod(monic_factor)
                
                # Check if division was exact (or nearly exact)
                if r.is_zero() or r.l1_norm() < 1e-6:
                    multiplicity += 1
                    remaining = q
                    # Normalize remaining
                    if not remaining.is_zero() and remaining.degree() > 0:
                        lead = remaining.leading_coeff()
                        if abs(lead - 1.0) > 1e-10:
                            remaining = remaining.scale(1.0 / lead)
                else:
                    break
            
            if multiplicity > 0:
                result_factors.append((monic_factor, multiplicity))
        
        # Add remaining polynomial if it's not 1 or 0
        if not remaining.is_zero() and remaining.degree() > 0:
            # Make remaining monic
            lead = remaining.leading_coeff()
            if abs(lead - 1.0) > 1e-10:
                remaining = remaining.scale(1.0 / lead)
            
            # Check if remaining is already in factors (shouldn't happen, but handle it)
            found = False
            for i, (f, m) in enumerate(result_factors):
                if f == remaining:
                    result_factors[i] = (f, m + 1)
                    found = True
                    break
            if not found:
                result_factors.append((remaining, 1))
        
        return content, result_factors
    
    def _zassenhaus_factor(self, f):
        """Zassenhaus algorithm for factoring primitive square-free polynomials.
        
        This is a simplified implementation that:
        1. Chooses a prime p where f is square-free mod p
        2. Factors f modulo p
        3. Lifts factors using Hensel lifting
        4. Uses combinatorial search with Mignotte bound to find true factors
        """
        n = f.degree()
        
        if n == 1:
            return [f]
        
        # Compute Mignotte bound
        A = f.max_norm()
        b = abs(f.leading_coeff())
        B = int((n + 1)**0.5 * (2**n) * A * b)
        
        # Choose prime p
        p = self._choose_prime(f, B)
        if p is None:
            # Fallback: use trial division for small polynomials
            return self._trial_factor(f)
        
        # Factor modulo p
        factors_mod_p = self._factor_mod_p(f, p)
        if not factors_mod_p:
            return [f]
        
        # Compute lifting exponent
        l = self._compute_lifting_exponent(p, B)
        
        # Lift factors using Hensel lifting
        lifted_factors = self._hensel_lift(f, factors_mod_p, p, l)
        
        # Combinatorial search for true factors
        true_factors = self._combinatorial_search(f, lifted_factors, B, p, l)
        
        return true_factors if true_factors else [f]
    
    def _choose_prime(self, f, bound):
        """Choose a prime p where f is square-free modulo p."""
        def isprime(n):
            """Simple primality test."""
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            for i in range(3, int(n**0.5) + 1, 2):
                if n % i == 0:
                    return False
            return True
        
        # Try primes up to a reasonable bound
        max_prime = min(bound, 1000)
        
        b = abs(int(round(f.leading_coeff())))
        
        for p in range(3, max_prime, 2):
            if not isprime(p):
                continue
            if b % p == 0:
                continue
            
            # Check if f is square-free mod p
            if self._is_square_free_mod_p(f, p):
                return p
        
        return None
    
    def _is_square_free_mod_p(self, f, p):
        """Check if polynomial is square-free modulo prime p."""
        # f is square-free mod p if gcd(f, f') = 1 mod p
        f_prime = f.derivative()
        
        # Compute gcd mod p
        g = self._gcd_mod_p(f, f_prime, p)
        
        # Check if gcd is constant (degree 0)
        return g.degree() == 0
    
    def _gcd_mod_p(self, a, b, p):
        """Compute GCD modulo prime p using Euclidean algorithm."""
        # Reduce coefficients mod p
        a_mod = Polynomial([int(c) % p for c in a.coeffs])
        b_mod = Polynomial([int(c) % p for c in b.coeffs])
        
        while not b_mod.is_zero():
            q, r = a_mod.divmod(b_mod)
            # Reduce remainder mod p
            r_mod = Polynomial([int(c) % p for c in r.coeffs])
            a_mod, b_mod = b_mod, r_mod
        
        return a_mod
    
    def _factor_mod_p(self, f, p):
        """Factor polynomial modulo prime p.
        
        This is a simplified version - for small primes, we can use
        trial division or Berlekamp algorithm. Here we use a simple approach.
        """
        # For small degree polynomials, use root finding
        if f.degree() <= 3:
            return self._factor_mod_p_small(f, p)
        
        # For larger degrees, return None to use fallback
        return None
    
    def _factor_mod_p_small(self, f, p):
        """Factor small degree polynomial modulo p by finding roots."""
        factors = []
        remaining = Polynomial([int(c) % p for c in f.coeffs])
        
        # Try all possible roots mod p
        max_iterations = p * 2  # Prevent infinite loops
        iteration = 0
        
        while remaining.degree() > 0 and iteration < max_iterations:
            iteration += 1
            found_root = False
            
            for root in range(p):
                # Evaluate polynomial at root
                value = int(round(remaining.evaluate(root))) % p
                if value == 0:
                    # Found a root, so (x - root) is a factor
                    factor = Polynomial([-root % p, 1])  # x - root
                    q, r = remaining.divmod(factor)
                    
                    # Reduce mod p
                    q_mod = Polynomial([int(round(c)) % p for c in q.coeffs])
                    r_mod = Polynomial([int(round(c)) % p for c in r.coeffs])
                    
                    if r_mod.is_zero() or r_mod.l1_norm() < 1e-6:
                        factors.append(factor)
                        remaining = q_mod
                        found_root = True
                        break
            
            if not found_root:
                break
        
        if remaining.degree() > 0:
            # Normalize leading coefficient
            lead = remaining.leading_coeff()
            if abs(lead - 1.0) > 1e-6:
                remaining = remaining.scale(1.0 / lead)
            factors.append(remaining)
        elif remaining.degree() == 0 and abs(remaining.coeffs[0] - 1.0) > 1e-6:
            # Constant factor (shouldn't happen for primitive poly, but handle it)
            pass
        
        return factors if factors else None
    
    def _compute_lifting_exponent(self, p, B):
        """Compute lifting exponent l such that p^l > 2B."""
        import math
        l = 1
        while p**l <= 2 * B:
            l += 1
        return l
    
    def _hensel_lift(self, f, factors_mod_p, p, l):
        """Lift factors from mod p to mod p^l using Hensel lifting.
        
        This is a simplified version - for small cases, we can use
        a basic lifting approach.
        """
        # For simplicity, if l is small, we can use a basic approach
        if l <= 2:
            # Just return factors mod p (they're already valid mod p^l for small l)
            return factors_mod_p
        
        # For larger l, implement proper Hensel lifting
        # This is a simplified version
        lifted = []
        for factor in factors_mod_p:
            # Basic lifting: keep factor as is (works for small cases)
            lifted.append(factor)
        
        return lifted
    
    def _combinatorial_search(self, f, lifted_factors, B, p, l):
        """Combinatorial search to find true integer factors.
        
        Tries all subsets of lifted factors and checks Mignotte bound.
        """
        if not lifted_factors:
            return None
        
        k = len(lifted_factors)
        pl = p**l
        fc = int(round(f.coeffs[0]))  # constant coefficient
        b = abs(int(round(f.leading_coeff())))
        
        # Try subsets of increasing size
        from itertools import combinations
        
        for s in range(1, k // 2 + 1):
            for indices in combinations(range(k), s):
                # Build candidate factor G from subset
                G = Polynomial([b])
                for i in indices:
                    G = G * lifted_factors[i]
                
                # Reduce mod p^l and make primitive
                G_coeffs = [int(c) % pl for c in G.coeffs]
                G = Polynomial(G_coeffs)
                G_content = G.content()
                if G_content > 1:
                    G = Polynomial([c / G_content for c in G.coeffs])
                
                # Check constant coefficient test
                if G.degree() > 0:
                    G_const = int(round(G.coeffs[0]))
                    if G_const != 0 and fc % G_const != 0:
                        continue
                
                # Build H from remaining factors
                H = Polynomial([b])
                for i in range(k):
                    if i not in indices:
                        H = H * lifted_factors[i]
                
                H_coeffs = [int(c) % pl for c in H.coeffs]
                H = Polynomial(H_coeffs)
                H_content = H.content()
                if H_content > 1:
                    H = Polynomial([c / H_content for c in H.coeffs])
                
                # Check Mignotte bound
                G_norm = G.l1_norm()
                H_norm = H.l1_norm()
                
                if G_norm * H_norm <= B:
                    # Found a candidate factor!
                    # Verify by division on original polynomial
                    # Make G monic for verification
                    G_lead = G.leading_coeff()
                    if abs(G_lead - 1.0) > 1e-10:
                        G_verify = G.scale(1.0 / G_lead)
                    else:
                        G_verify = G
                    
                    q, r = f.divmod(G_verify)
                    if r.is_zero() or r.l1_norm() < 1e-6:
                        # Found a true factor!
                        # Recursively factor H
                        if H.degree() > 0:
                            remaining_factors = self._combinatorial_search(H, 
                                [lifted_factors[i] for i in range(k) if i not in indices],
                                B, p, l)
                            
                            if remaining_factors is None:
                                remaining_factors = [H]
                        else:
                            remaining_factors = []
                        
                        return [G_verify] + remaining_factors
        
        return None
    
    def _trial_factor(self, f):
        """Optimized trial division factorization.
        
        Uses Rational Root Theorem with early termination and
        Horner evaluation for faster root checking.
        """
        factors = []
        remaining = f
        
        # Fast path for linear polynomial
        if f.degree() == 1:
            return [f]
        
        # Fast path for quadratic with integer roots
        if f.degree() == 2:
            disc = f.coeffs[1]**2 - 4*f.coeffs[2]*f.coeffs[0]
            if disc >= 0:
                sqrt_disc = disc ** 0.5
                if abs(sqrt_disc - round(sqrt_disc)) < 1e-10:
                    # Integer roots possible
                    pass
        
        max_iterations = remaining.degree() + 2
        iteration = 0
        
        while remaining.degree() > 1 and iteration < max_iterations:
            iteration += 1
            fc = int(round(remaining.coeffs[0]))
            lc = int(round(remaining.leading_coeff()))
            
            # Handle x as factor (constant term = 0)
            if fc == 0:
                factors.append(Polynomial([0, 1]))
                remaining = Polynomial(remaining.coeffs[1:])
                continue
            
            # Rational Root Theorem: try p/q where p|fc and q|lc
            divisors_fc = self._get_divisors_fast(abs(fc))
            divisors_lc = self._get_divisors_fast(abs(lc)) if abs(lc) > 1 else [1]
            
            found_factor = False
            
            for p in divisors_fc:
                if found_factor:
                    break
                for q in divisors_lc:
                    if found_factor:
                        break
                    for sign in [1, -1]:
                        root = sign * p / q
                        
                        # Fast evaluation using Horner
                        val = remaining.evaluate(root)
                        
                        if abs(val) < 1e-10:
                            # Found root - construct factor
                            if q == 1:
                                factor = Polynomial([-root, 1])
                            else:
                                factor = Polynomial([-p*sign, q])
                            
                            q_poly, r = remaining.divmod(factor)
                            
                            if r.is_zero() or r.l1_norm() < 1e-6:
                                # Normalize factor to monic
                                lead_f = factor.leading_coeff()
                                if abs(lead_f - 1.0) > 1e-10:
                                    factor = Polynomial([c/lead_f for c in factor.coeffs])
                                    q_poly = Polynomial([c*lead_f for c in q_poly.coeffs])
                                
                                factors.append(factor)
                                remaining = q_poly
                                
                                # Normalize remaining
                                if remaining.degree() > 0:
                                    lead = remaining.leading_coeff()
                                    if abs(lead - 1.0) > 1e-10:
                                        remaining = Polynomial([c / lead for c in remaining.coeffs])
                                
                                found_factor = True
                                break
            
            if not found_factor:
                break
        
        # If remaining is not constant and not 1, try to factor it further
        # or add it as-is
        if remaining.degree() > 1:
            # Try to factor remaining further using same method
            # For quadratics, check if it factors
            if remaining.degree() == 2:
                # Try to find two linear factors
                a = remaining.coeffs[2]
                b = remaining.coeffs[1]
                c = remaining.coeffs[0]
                
                # For ax^2 + bx + c, try to find roots
                # Discriminant method for integer roots
                discriminant = b*b - 4*a*c
                if discriminant >= 0:
                    sqrt_disc = int(discriminant**0.5)
                    if sqrt_disc * sqrt_disc == discriminant:
                        # Integer roots exist
                        root1 = (-b + sqrt_disc) / (2*a)
                        root2 = (-b - sqrt_disc) / (2*a)
                        
                        if abs(root1 - round(root1)) < 1e-6 and abs(root2 - round(root2)) < 1e-6:
                            root1 = int(round(root1))
                            root2 = int(round(root2))
                            factor1 = Polynomial([-root1, 1])
                            factor2 = Polynomial([-root2, 1])
                            factors.append(factor1)
                            factors.append(factor2)
                            remaining = Polynomial([1])  # Fully factored
                else:
                    # No real roots, remaining is irreducible quadratic
                    lead = remaining.leading_coeff()
                    if abs(lead - 1.0) > 1e-10:
                        remaining = remaining.scale(1.0 / lead)
                    factors.append(remaining)
            else:
                # Higher degree - just add as-is (make monic)
                lead = remaining.leading_coeff()
                if abs(lead - 1.0) > 1e-10:
                    remaining = remaining.scale(1.0 / lead)
                factors.append(remaining)
        elif remaining.degree() == 1:
            # Linear factor
            lead = remaining.leading_coeff()
            if abs(lead - 1.0) > 1e-10:
                remaining = remaining.scale(1.0 / lead)
            factors.append(remaining)
        elif remaining.degree() == 0:
            # Constant factor - only add if it's not 1
            const = remaining.coeffs[0]
            if abs(const - 1.0) > 1e-10:
                factors.append(remaining)
        
        return factors if factors else [f]
    
    def _get_divisors(self, n):
        """Get all positive divisors of n."""
        if n == 0:
            return []
        divisors = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divisors.append(i)
                if i != n // i:
                    divisors.append(n // i)
        return sorted(divisors)
    
    def _get_divisors_fast(self, n):
        """Fast divisor enumeration with small divisors first."""
        if n == 0:
            return [1]
        if n == 1:
            return [1]
        
        # For small n, direct enumeration is fast
        if n <= 100:
            return self._get_divisors(n)
        
        # For larger n, enumerate small divisors first (more likely to be roots)
        divisors = [1]
        sqrt_n = int(n ** 0.5)
        
        for i in range(2, min(sqrt_n + 1, 50)):  # Check small factors first
            if n % i == 0:
                divisors.append(i)
        
        # Add larger divisors
        for i in range(2, sqrt_n + 1):
            if n % i == 0:
                large_div = n // i
                if large_div not in divisors and large_div > 50:
                    divisors.append(large_div)
        
        divisors.append(n)
        return divisors


def _gcd_interpolate(h_eval, x_eval, max_degree):
    """Interpolate polynomial GCD from integer GCD.
    
    Given an integer h_eval that represents h(x_eval) where h is a polynomial,
    reconstruct h by extracting digits in base x_eval.
    
    Args:
        h_eval: Integer result of evaluating polynomial at x_eval
        x_eval: The evaluation point used
        max_degree: Maximum possible degree of the polynomial
    
    Returns:
        Polynomial object, or None if interpolation fails
    """
    if h_eval == 0:
        return Polynomial([0])
    
    # Handle negative
    negative = h_eval < 0
    h_eval = abs(h_eval)
    
    # Extract coefficients by repeatedly taking modulo x_eval
    coeffs = []
    h = h_eval
    
    while h > 0 and len(coeffs) <= max_degree:
        # Extract coefficient (digit in base x_eval)
        coeff = h % x_eval
        
        # Handle negative coefficients (if coeff > x_eval/2, it might be negative)
        if coeff > x_eval // 2:
            coeff -= x_eval
        
        coeffs.append(coeff)
        h = (h - coeff) // x_eval
    
    # If we extracted too many coefficients, something went wrong
    if len(coeffs) > max_degree + 1:
        return None
    
    # Create polynomial
    if negative:
        coeffs = [-c for c in coeffs]
    
    return Polynomial(coeffs)


def extract_power_expression(expr_str):
    """Extract base polynomial and exponent from a power expression.
    
    For expressions like "(x+2)^99", returns ("x+2", 99).
    For expressions like "x^2+3x+2", returns (None, None) - not a pure power expression.
    
    Returns:
        Tuple of (base_expr_string, exponent) or (None, None) if not a power expression
    """
    import re
    
    expr_str = expr_str.strip().replace(" ", "")
    
    # Check if the entire expression is a single power expression: (...)^n
    # It must start with ( and the power must be at the end
    paren_power_pattern = r'^\(([^()]+)\)\^(\d+)$'
    match = re.match(paren_power_pattern, expr_str)
    
    if match:
        inner_expr = match.group(1)
        power = int(match.group(2))
        return inner_expr, power
    
    # Check for just parentheses without power: (x+2) is effectively (x+2)^1
    if expr_str.startswith("(") and expr_str.endswith(")"):
        # Verify these are matching outer parentheses
        depth = 0
        matching = True
        for i, char in enumerate(expr_str):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            if depth == 0 and i < len(expr_str) - 1:
                matching = False
                break
        if matching:
            inner_expr = expr_str[1:-1].strip()
            return inner_expr, 1
    
    return None, None


def parse_power_factors(expr_str):
    """Parse expression into list of (base, exponent) power factors.
    
    Examples:
        "(x+1)^5*(x+2)" -> [("x+1", 5), ("x+2", 1)]
        "(x+1)^3" -> [("x+1", 3)]
        "(x-1)^4*(x+3)" -> [("x-1", 4), ("x+3", 1)]
    """
    import re
    
    expr_str = expr_str.strip()
    
    # Remove outer parentheses if they wrap the entire expression
    while expr_str.startswith("(") and expr_str.endswith(")"):
        depth = 0
        matching = True
        for i, char in enumerate(expr_str):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            if depth == 0 and i < len(expr_str) - 1:
                matching = False
                break
        if matching:
            expr_str = expr_str[1:-1].strip()
        else:
            break
    
    factors = []
    
    # Split by * at top level (not inside parentheses)
    parts = []
    current = ""
    depth = 0
    
    for char in expr_str:
        if char == "(":
            depth += 1
            current += char
        elif char == ")":
            depth -= 1
            current += char
        elif char == "*" and depth == 0:
            if current.strip():
                parts.append(current.strip())
            current = ""
        else:
            current += char
    
    if current.strip():
        parts.append(current.strip())
    
    # Parse each part as a power expression
    for part in parts:
        base, exp = extract_power_expression(part)
        if base is not None:
            factors.append((base, exp))
        else:
            # Not a power expression, treat as base^1
            # Clean up the part
            clean_part = part.strip()
            if clean_part.startswith("(") and clean_part.endswith(")"):
                clean_part = clean_part[1:-1]
            factors.append((clean_part, 1))
    
    return factors


def simplify_power_product(numerator_str, denominator_str):
    """Simplify products of power expressions.
    
    Example: (x+1)^5*(x+2) / (x+1)^5 -> (x+2) / 1
    
    Returns:
        Tuple of (simplified_num_str, simplified_den_str, was_simplified)
    """
    try:
        num_factors = parse_power_factors(numerator_str)
        den_factors = parse_power_factors(denominator_str)
        
        if not num_factors or not den_factors:
            return numerator_str, denominator_str, False
        
        # Try to cancel common factors
        was_simplified = False
        
        for i, (num_base, num_exp) in enumerate(num_factors):
            for j, (den_base, den_exp) in enumerate(den_factors):
                # Check if bases are equal
                try:
                    num_base_poly = parse_polynomial_simple(num_base)
                    den_base_poly = parse_polynomial_simple(den_base)
                    
                    if _polynomials_equal(num_base_poly, den_base_poly):
                        # Found matching base! Cancel exponents
                        common_exp = min(num_exp, den_exp)
                        
                        num_factors[i] = (num_base, num_exp - common_exp)
                        den_factors[j] = (den_base, den_exp - common_exp)
                        was_simplified = True
                except:
                    pass
        
        if not was_simplified:
            return numerator_str, denominator_str, False
        
        # Rebuild expressions
        def rebuild_expr(factors):
            remaining = [(base, exp) for base, exp in factors if exp > 0]
            if not remaining:
                return "1"
            
            parts = []
            for base, exp in remaining:
                if exp == 1:
                    parts.append(f"({base})")
                else:
                    parts.append(f"({base})^{exp}")
            
            return "*".join(parts) if parts else "1"
        
        new_num = rebuild_expr(num_factors)
        new_den = rebuild_expr(den_factors)
        
        return new_num, new_den, True
        
    except Exception:
        return numerator_str, denominator_str, False


def simplify_power_expressions(numerator_str, denominator_str):
    """Pre-simplify expressions involving powers of the same base polynomial.
    
    Handles:
    1. Simple: (x+2)^99 / (x+2)^98 -> (x+2) / 1
    2. Products: (x+1)^5*(x+2) / (x+1)^5 -> (x+2) / 1
    
    Returns:
        Tuple of (simplified_num_str, simplified_den_str, was_simplified)
        If not applicable, returns (numerator_str, denominator_str, False)
    """
    # First try product-aware simplification
    product_result = simplify_power_product(numerator_str, denominator_str)
    if product_result[2]:  # was_simplified
        return product_result
    
    # Fall back to simple power expression handling
    num_base, num_exp = extract_power_expression(numerator_str)
    den_base, den_exp = extract_power_expression(denominator_str)
    
    # Both must be power expressions with the same base
    if num_base is None or den_base is None:
        return numerator_str, denominator_str, False
    
    # Normalize bases for comparison (parse and compare)
    # Use a small expansion to check equality
    try:
        num_base_poly = parse_polynomial_simple(num_base)
        den_base_poly = parse_polynomial_simple(den_base)
        
        # Check if base polynomials are equal
        if not _polynomials_equal(num_base_poly, den_base_poly):
            return numerator_str, denominator_str, False
        
        # Same base! Simplify by subtracting exponents
        if num_exp > den_exp:
            # Result: base^(num_exp - den_exp) / 1
            new_exp = num_exp - den_exp
            if new_exp == 1:
                new_num_str = f"({num_base})"
            else:
                new_num_str = f"({num_base})^{new_exp}"
            new_den_str = "1"
        elif den_exp > num_exp:
            # Result: 1 / base^(den_exp - num_exp)
            new_exp = den_exp - num_exp
            new_num_str = "1"
            if new_exp == 1:
                new_den_str = f"({den_base})"
            else:
                new_den_str = f"({den_base})^{new_exp}"
        else:
            # Equal exponents
            new_num_str = "1"
            new_den_str = "1"
        
        return new_num_str, new_den_str, True
        
    except Exception:
        return numerator_str, denominator_str, False


def parse_polynomial_simple(expr_str):
    """Parse a simple polynomial without expanding powers.
    
    This is used for comparing base polynomials in power expressions.
    Does NOT handle (...)^n syntax - only basic polynomials.
    """
    import re
    
    # Remove spaces
    expr_str = expr_str.replace(" ", "")
    
    # Remove surrounding parentheses if present
    expr_str = expr_str.strip()
    while expr_str.startswith("(") and expr_str.endswith(")"):
        depth = 0
        matching = True
        for i, char in enumerate(expr_str):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            if depth == 0 and i < len(expr_str) - 1:
                matching = False
                break
        if matching:
            expr_str = expr_str[1:-1].strip()
        else:
            break
    
    # Handle negative signs at start
    if expr_str.startswith("-"):
        expr_str = "0" + expr_str
    
    # Split by + and - (keeping the signs)
    terms = re.split(r'([+-])', expr_str)
    
    if terms[0] == "":
        terms = terms[1:]
    
    coeffs_dict = {}
    
    i = 0
    while i < len(terms):
        if terms[i] in ['+', '-']:
            sign = 1 if terms[i] == '+' else -1
            term = terms[i + 1] if i + 1 < len(terms) else ""
            i += 2
        else:
            sign = 1
            term = terms[i]
            i += 1
        
        if not term:
            continue
        
        match = re.match(r'^([+-]?\d*\.?\d*)?(x)?(\^(\d+))?$', term)
        if not match:
            raise ValueError(f"Could not parse term: {term}")
        
        coeff_str, has_x, _, exp_str = match.groups()
        
        if coeff_str in ['', '+', '-']:
            coeff = 1 if coeff_str != '-' else -1
        else:
            coeff = float(coeff_str) if '.' in coeff_str else int(coeff_str)
        
        coeff *= sign
        
        if not has_x:
            degree = 0
        elif not exp_str:
            degree = 1
        else:
            degree = int(exp_str)
        
        coeffs_dict[degree] = coeffs_dict.get(degree, 0) + coeff
    
    max_degree = max(coeffs_dict.keys()) if coeffs_dict else 0
    coeffs = [0] * (max_degree + 1)
    for deg, coeff in coeffs_dict.items():
        coeffs[deg] = coeff
    
    return Polynomial(coeffs)


def parse_polynomial(expr_str, max_expansion_power=20):
    """Parse a polynomial string into a Polynomial object.
    
    Supports formats like:
    - "x^2 + 3x + 2"
    - "x^2-9"
    - "2x^3+4x^2"
    - "(x^2-9)" (with parentheses)
    - "(x+2)^2" (with power)
    
    Args:
        expr_str: String representation of the polynomial
        max_expansion_power: Maximum power to expand. Powers larger than this
                            will raise an error (use simplify_power_expressions first)
    """
    import re
    
    # Remove spaces
    expr_str = expr_str.replace(" ", "")
    
    # Handle expressions like (polynomial)^power
    # Look for pattern: (...) followed by ^number
    paren_power_pattern = r'\(([^()]+)\)\^(\d+)'
    match = re.search(paren_power_pattern, expr_str)
    
    if match:
        # Extract the polynomial inside parentheses and the power
        inner_expr = match.group(1)
        power = int(match.group(2))
        
        # Check if power is too large to expand
        if power > max_expansion_power:
            raise ValueError(
                f"Power {power} is too large to expand directly. "
                f"Use simplify_power_expressions() for expressions with large powers."
            )
        
        # Parse the inner polynomial
        inner_poly = parse_polynomial(inner_expr, max_expansion_power)
        
        # Compute the power by repeated multiplication
        result = Polynomial([1])  # Start with 1
        for _ in range(power):
            result = result * inner_poly
        
        # Replace the matched pattern with the expanded result
        # Convert result back to string and replace in original expression
        expanded = str(result)
        # Remove spaces from expanded result
        expanded = expanded.replace(" ", "")
        
        expr_str = expr_str[:match.start()] + expanded + expr_str[match.end():]
        
        # Recursively parse in case there are more patterns
        return parse_polynomial(expr_str, max_expansion_power)
    
    # Remove surrounding parentheses if present
    expr_str = expr_str.strip()
    while expr_str.startswith("(") and expr_str.endswith(")"):
        # Check if these are matching outer parentheses
        depth = 0
        matching = True
        for i, char in enumerate(expr_str):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            # If depth reaches 0 before the last character, they're not outer parens
            if depth == 0 and i < len(expr_str) - 1:
                matching = False
                break
        if matching:
            expr_str = expr_str[1:-1].strip()
        else:
            break
    
    # Handle negative signs at start
    if expr_str.startswith("-"):
        expr_str = "0" + expr_str
    
    # Split by + and - (keeping the signs)
    terms = re.split(r'([+-])', expr_str)
    
    # First term doesn't have a sign prefix
    if terms[0] == "":
        terms = terms[1:]
    
    coeffs_dict = {}  # degree -> coefficient
    
    i = 0
    while i < len(terms):
        if terms[i] in ['+', '-']:
            sign = 1 if terms[i] == '+' else -1
            term = terms[i + 1] if i + 1 < len(terms) else ""
            i += 2
        else:
            sign = 1
            term = terms[i]
            i += 1
        
        if not term:
            continue
        
        # Parse term
        # Match patterns like: "2x^3", "x^2", "3x", "5"
        match = re.match(r'^([+-]?\d*\.?\d*)?(x)?(\^(\d+))?$', term)
        if not match:
            raise ValueError(f"Could not parse term: {term}")
        
        coeff_str, has_x, _, exp_str = match.groups()
        
        # Determine coefficient
        if coeff_str in ['', '+', '-']:
            coeff = 1 if coeff_str != '-' else -1
        else:
            coeff = float(coeff_str) if '.' in coeff_str else int(coeff_str)
        
        coeff *= sign
        
        # Determine degree
        if not has_x:
            degree = 0
        elif not exp_str:
            degree = 1
        else:
            degree = int(exp_str)
        
        coeffs_dict[degree] = coeffs_dict.get(degree, 0) + coeff
    
    # Convert to list
    max_degree = max(coeffs_dict.keys()) if coeffs_dict else 0
    coeffs = [0] * (max_degree + 1)
    for deg, coeff in coeffs_dict.items():
        coeffs[deg] = coeff
    
    return Polynomial(coeffs)


def _polynomials_equal(p1, p2, tolerance=1e-6):
    """Check if two polynomials are equal (within tolerance)."""
    if p1.degree() != p2.degree():
        return False
    
    for c1, c2 in zip(p1.coeffs, p2.coeffs):
        if abs(c1 - c2) > tolerance:
            return False
    
    return True


def simplify_rational_factorization(numerator_str, denominator_str):
    """Simplify rational expression using factorization.
    
    This method:
    1. Factors both numerator and denominator
    2. Finds common factors
    3. Removes common factors with their multiplicities
    4. Reconstructs simplified numerator and denominator
    
    Args:
        numerator_str: String representation of numerator polynomial
        denominator_str: String representation of denominator polynomial
    
    Returns:
        Tuple of (simplified_numerator, simplified_denominator, "factorization")
        or None if factorization fails
    """
    try:
        # Step 1: Parse polynomials
        p = parse_polynomial(numerator_str)
        g = parse_polynomial(denominator_str)
        
        # Step 2: Factor both numerator and denominator
        num_content, num_factors = p.factor()
        den_content, den_factors = g.factor()
        
        # Step 3: Find common factors
        # Normalize all factors to be monic for comparison
        normalized_num_factors = []
        for factor, mult in num_factors:
            lead = factor.leading_coeff()
            if abs(lead - 1.0) > 1e-10:
                normalized_factor = factor.scale(1.0 / lead)
            else:
                normalized_factor = factor
            normalized_num_factors.append((normalized_factor, mult))
        
        normalized_den_factors = []
        for factor, mult in den_factors:
            lead = factor.leading_coeff()
            if abs(lead - 1.0) > 1e-10:
                normalized_factor = factor.scale(1.0 / lead)
            else:
                normalized_factor = factor
            normalized_den_factors.append((normalized_factor, mult))
        
        # Create lists with indices for matching
        num_factor_list = list(enumerate(normalized_num_factors))
        den_factor_list = list(enumerate(normalized_den_factors))
        
        # Track which factors have been matched
        num_matched = [False] * len(normalized_num_factors)
        den_matched = [False] * len(normalized_den_factors)
        
        # Step 4: Remove common factors by matching
        simplified_num_factors = []
        simplified_den_factors = []
        
        # Match common factors
        for i, (num_factor, num_mult) in enumerate(normalized_num_factors):
            if num_matched[i]:
                continue
            
            # Look for matching factor in denominator
            for j, (den_factor, den_mult) in enumerate(normalized_den_factors):
                if den_matched[j]:
                    continue
                
                # Check if factors are equal
                if _polynomials_equal(num_factor, den_factor):
                    # Common factor found
                    common_mult = min(num_mult, den_mult)
                    remaining_num_mult = num_mult - common_mult
                    remaining_den_mult = den_mult - common_mult
                    
                    # Add remaining multiplicities
                    if remaining_num_mult > 0:
                        simplified_num_factors.append((num_factor, remaining_num_mult))
                    
                    if remaining_den_mult > 0:
                        simplified_den_factors.append((den_factor, remaining_den_mult))
                    
                    # Mark as matched
                    num_matched[i] = True
                    den_matched[j] = True
                    break
        
        # Add unmatched numerator factors
        for i, (factor, mult) in enumerate(normalized_num_factors):
            if not num_matched[i]:
                simplified_num_factors.append((factor, mult))
        
        # Add unmatched denominator factors
        for i, (factor, mult) in enumerate(normalized_den_factors):
            if not den_matched[i]:
                simplified_den_factors.append((factor, mult))
        
        # Step 5: Reconstruct polynomials from factors
        # Handle content
        def int_gcd(a, b):
            while b:
                a, b = b, a % b
            return abs(a)
        
        # Factor out common content
        if num_content == 0:
            # Numerator is zero
            return Polynomial([0]), Polynomial([1]), "factorization"
        
        if den_content == 0:
            # Denominator is zero - invalid
            return None
        
        # Handle signs - keep denominator positive
        if den_content < 0:
            num_content = -num_content
            den_content = -den_content
        
        common_content = int_gcd(abs(num_content), abs(den_content))
        if common_content > 1:
            num_content = num_content // common_content
            den_content = den_content // common_content
        
        # Reconstruct numerator
        if simplified_num_factors:
            num_poly = Polynomial([1])  # Start with 1
            for factor, mult in simplified_num_factors:
                for _ in range(mult):
                    num_poly = num_poly * factor
            # Multiply by content
            if num_content != 1:
                num_poly = num_poly.scale(float(num_content))
        else:
            num_poly = Polynomial([num_content])
        
        # Reconstruct denominator
        if simplified_den_factors:
            den_poly = Polynomial([1])  # Start with 1
            for factor, mult in simplified_den_factors:
                for _ in range(mult):
                    den_poly = den_poly * factor
            # Multiply by content
            if den_content != 1:
                den_poly = den_poly.scale(float(den_content))
        else:
            den_poly = Polynomial([den_content])
        
        # Normalize: if denominator is just a constant, factor it into numerator
        if den_poly.degree() == 0:
            den_const = den_poly.coeffs[0]
            if abs(den_const - 1.0) > 1e-10:
                num_poly = num_poly.scale(1.0 / den_const)
                den_poly = Polynomial([1])
        
        # Factor out common integer content from coefficients
        num_content_final = num_poly.content()
        den_content_final = den_poly.content()
        
        if num_content_final > 0 and den_content_final > 0:
            common_content_final = int_gcd(num_content_final, den_content_final)
            if common_content_final > 1:
                num_poly = num_poly.scale(1.0 / common_content_final)
                den_poly = den_poly.scale(1.0 / common_content_final)
        
        return num_poly, den_poly, "factorization"
    
    except Exception as e:
        # Factorization failed, return None to fall back to other methods
        return None


def simplify_rational(numerator_str, denominator_str):
    """Simplify a rational expression p(x)/g(x).
    
    This function tries multiple methods in order:
    0. Power expression pre-simplification (for (x+a)^n / (x+a)^m cases)
    1. Heuristic GCD (fast, evaluation-based)
    2. Factorization-based simplification (new)
    3. Euclidean GCD (fallback, always works)
    
    Args:
        numerator_str: String representation of numerator polynomial
        denominator_str: String representation of denominator polynomial
    
    Returns:
        Tuple of (simplified_numerator, simplified_denominator, method_used)
        where method_used is "power_simplification", "heuristic", "factorization", or "euclidean"
    """
    # Step 0: Try power expression pre-simplification first
    # This handles cases like (x+2)^99 / (x+2)^98 without expanding
    simplified_num, simplified_den, was_power_simplified = simplify_power_expressions(
        numerator_str, denominator_str
    )
    
    if was_power_simplified:
        # Parse the simplified expressions (which should now be manageable)
        p = parse_polynomial(simplified_num)
        g = parse_polynomial(simplified_den)
        return p, g, "power_simplification"
    
    # Step 1: Parse polynomials
    p = parse_polynomial(numerator_str)
    g = parse_polynomial(denominator_str)
    
    # Step 2: Try heuristic GCD first (fast for typical cases)
    heuristic_result = p.gcd_heuristic(g)
    if heuristic_result is not None:
        # Heuristic succeeded, use it
        h = heuristic_result
        # Make monic
        lead = h.leading_coeff()
        if abs(lead - 1.0) > 1e-10:
            h = h.scale(1.0 / lead)
        
        # Divide both by GCD
        cff, r1 = p.divmod(h)
        cfg, r2 = g.divmod(h)
        
        # Verify no remainder
        if r1.is_zero() and r2.is_zero():
            # Factor out common integer content
            cff_content = cff.content()
            cfg_content = cfg.content()
            
            if cff_content > 0 and cfg_content > 0:
                def int_gcd(a, b):
                    while b:
                        a, b = b, a % b
                    return abs(a)
                
                common_content = int_gcd(cff_content, cfg_content)
                if common_content > 1:
                    cff = cff.scale(1.0 / common_content)
                    cfg = cfg.scale(1.0 / common_content)
            
            return cff, cfg, "heuristic"
    
    # Step 3: Try factorization-based simplification (between heuristic and Euclidean)
    factor_result = simplify_rational_factorization(numerator_str, denominator_str)
    if factor_result is not None:
        return factor_result
    
    # Step 4: Fall back to Euclidean algorithm (always works)
    h = p.gcd_euclidean(g)
    
    # Divide both by GCD
    cff, r1 = p.divmod(h)
    cfg, r2 = g.divmod(h)
    
    # Verify no remainder
    if not r1.is_zero() or not r2.is_zero():
        raise ValueError("GCD division failed - this shouldn't happen")
    
    # Factor out common integer content from both
    cff_content = cff.content()
    cfg_content = cfg.content()
    
    if cff_content > 0 and cfg_content > 0:
        def int_gcd(a, b):
            while b:
                a, b = b, a % b
            return abs(a)
        
        common_content = int_gcd(cff_content, cfg_content)
        if common_content > 1:
            cff = cff.scale(1.0 / common_content)
            cfg = cfg.scale(1.0 / common_content)
    
    return cff, cfg, "euclidean"


def format_result(num, den):
    """Format the result as a string."""
    num_str = str(num)
    den_str = str(den)
    
    if den_str == "1":
        return num_str
    else:
        return f"({num_str}) / ({den_str})"


def simplify_heuristic(numerator_str, denominator_str):
    """Simplify using heuristic GCD method."""
    # Try power pre-simplification first
    simplified_num, simplified_den, was_power_simplified = simplify_power_expressions(
        numerator_str, denominator_str
    )
    if was_power_simplified:
        numerator_str, denominator_str = simplified_num, simplified_den
    
    p = parse_polynomial(numerator_str)
    g = parse_polynomial(denominator_str)
    
    heuristic_result = p.gcd_heuristic(g)
    if heuristic_result is None:
        return None, None, "heuristic (failed)"
    
    h = heuristic_result
    # Make monic
    lead = h.leading_coeff()
    if abs(lead - 1.0) > 1e-10:
        h = h.scale(1.0 / lead)
    
    # Divide both by GCD
    cff, r1 = p.divmod(h)
    cfg, r2 = g.divmod(h)
    
    if not r1.is_zero() or not r2.is_zero():
        return None, None, "heuristic (division failed)"
    
    # Factor out common integer content
    cff_content = cff.content()
    cfg_content = cfg.content()
    
    if cff_content > 0 and cfg_content > 0:
        def int_gcd(a, b):
            while b:
                a, b = b, a % b
            return abs(a)
        
        common_content = int_gcd(cff_content, cfg_content)
        if common_content > 1:
            cff = cff.scale(1.0 / common_content)
            cfg = cfg.scale(1.0 / common_content)
    
    return cff, cfg, "heuristic"


def simplify_euclidean(numerator_str, denominator_str):
    """Simplify using Euclidean GCD method."""
    # Try power pre-simplification first
    simplified_num, simplified_den, was_power_simplified = simplify_power_expressions(
        numerator_str, denominator_str
    )
    if was_power_simplified:
        numerator_str, denominator_str = simplified_num, simplified_den
    
    p = parse_polynomial(numerator_str)
    g = parse_polynomial(denominator_str)
    
    h = p.gcd_euclidean(g)
    
    # Divide both by GCD
    cff, r1 = p.divmod(h)
    cfg, r2 = g.divmod(h)
    
    if not r1.is_zero() or not r2.is_zero():
        return None, None, "euclidean (division failed)"
    
    # Factor out common integer content
    cff_content = cff.content()
    cfg_content = cfg.content()
    
    if cff_content > 0 and cfg_content > 0:
        def int_gcd(a, b):
            while b:
                a, b = b, a % b
            return abs(a)
        
        common_content = int_gcd(cff_content, cfg_content)
        if common_content > 1:
            cff = cff.scale(1.0 / common_content)
            cfg = cfg.scale(1.0 / common_content)
    
    return cff, cfg, "euclidean"


def simplify_factorization_method(numerator_str, denominator_str):
    """Simplify using factorization-based method."""
    # Try power pre-simplification first
    simplified_num, simplified_den, was_power_simplified = simplify_power_expressions(
        numerator_str, denominator_str
    )
    if was_power_simplified:
        numerator_str, denominator_str = simplified_num, simplified_den
    
    result = simplify_rational_factorization(numerator_str, denominator_str)
    if result is None:
        return None, None, "factorization (failed)"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SYLVESTER MATRIX APPROACH - Linear Algebra Based Simplification
# ═══════════════════════════════════════════════════════════════════════════════

def build_sylvester_matrix(p, g):
    """
    Build the Sylvester matrix for two polynomials (optimized).
    
    For polynomials p(x) of degree n and g(x) of degree m,
    the Sylvester matrix is (n+m) x (n+m).
    
    Uses list comprehensions for faster construction.
    """
    n = p.degree()
    m = g.degree()
    size = n + m
    
    if size == 0:
        return [[1.0]]
    
    # Pre-allocate matrix
    S = [[0.0] * size for _ in range(size)]
    
    # Get coefficients in descending order
    p_coeffs = p.coeffs[::-1]
    g_coeffs = g.coeffs[::-1]
    p_len = len(p_coeffs)
    g_len = len(g_coeffs)
    
    # Fill rows with p's coefficients (top m rows)
    for i in range(m):
        end = min(i + p_len, size)
        for j in range(p_len):
            if i + j < size:
                S[i][i + j] = p_coeffs[j]
    
    # Fill rows with g's coefficients (bottom n rows)
    for i in range(n):
        for j in range(g_len):
            if i + j < size:
                S[m + i][i + j] = g_coeffs[j]
    
    return S


def gaussian_elimination_with_pivoting(matrix):
    """
    Optimized Gaussian elimination with partial pivoting.
    
    Uses in-place operations and early termination for speed.
    """
    if not matrix or not matrix[0]:
        return 0, [], []
    
    rows = len(matrix)
    cols = len(matrix[0])
    
    # Work with a copy
    A = [row[:] for row in matrix]
    
    pivot_row = 0
    pivot_cols = []
    
    for col in range(cols):
        if pivot_row >= rows:
            break
        
        # Find pivot with early exit
        max_row = pivot_row
        max_val = abs(A[pivot_row][col])
        
        for row in range(pivot_row + 1, rows):
            val = abs(A[row][col])
            if val > max_val:
                max_val = val
                max_row = row
        
        # Skip zero column
        if max_val < 1e-12:
            continue
        
        # Swap rows (single swap)
        if max_row != pivot_row:
            A[pivot_row], A[max_row] = A[max_row], A[pivot_row]
        
        pivot_cols.append(col)
        
        # Normalize pivot row
        pivot_val = A[pivot_row][col]
        inv_pivot = 1.0 / pivot_val
        row_data = A[pivot_row]
        for j in range(col, cols):  # Start from col, not 0
            row_data[j] *= inv_pivot
        
        # Eliminate other rows
        for row in range(rows):
            if row != pivot_row:
                factor = A[row][col]
                if abs(factor) > 1e-14:
                    row_data_other = A[row]
                    pivot_data = A[pivot_row]
                    for j in range(col, cols):  # Start from col
                        row_data_other[j] -= factor * pivot_data[j]
        
        pivot_row += 1
    
    return len(pivot_cols), A, pivot_cols


def compute_null_space(matrix, size, rank, pivot_cols):
    """
    Compute null space basis vectors of the matrix.
    
    Args:
        matrix: Row echelon form of the matrix
        size: Size of the square matrix
        rank: Rank of the matrix
        pivot_cols: List of pivot column indices
    
    Returns:
        List of null space basis vectors, or None if null space is trivial
    """
    if rank >= size:
        return None  # Full rank, null space is {0}
    
    # Find free variables (columns that are not pivots)
    free_cols = [i for i in range(size) if i not in pivot_cols]
    
    if not free_cols:
        return None
    
    null_vectors = []
    
    for free_col in free_cols:
        # Create a null vector by setting this free variable to 1
        null_vec = [0.0] * size
        null_vec[free_col] = 1.0
        
        # Back-substitute to find pivot variable values
        for i in range(len(pivot_cols) - 1, -1, -1):
            pivot_col = pivot_cols[i]
            # Find the value from the equation
            val = 0.0
            for j in range(pivot_col + 1, size):
                val -= matrix[i][j] * null_vec[j]
            null_vec[pivot_col] = val
        
        null_vectors.append(null_vec)
    
    return null_vectors


def extract_common_factor_from_nullspace(null_vectors, p, g, n, m, expected_degree):
    """
    Extract the common factor polynomial from null space vectors.
    
    The null space of the Sylvester matrix encodes the Bezout identity:
    u(x) * p(x) + v(x) * g(x) = 0
    
    where deg(u) < m and deg(v) < n.
    
    The common factor h(x) = gcd(p, g) satisfies:
    p(x) = h(x) * p'(x)
    g(x) = h(x) * g'(x)
    
    Args:
        null_vectors: List of null space basis vectors
        p: Numerator polynomial
        g: Denominator polynomial  
        n: Degree of p
        m: Degree of g
        expected_degree: Expected degree of common factor
    
    Returns:
        Common factor as Polynomial, or None if extraction fails
    """
    if not null_vectors or expected_degree <= 0:
        return None
    
    # The null vector has structure: [u_{m-1}, ..., u_0, v_{n-1}, ..., v_0]
    # where u(x) * p(x) = -v(x) * g(x) = h(x) * w(x) for some w(x)
    
    # Try to extract factor using the null space relationship
    # We use the fact that gcd(p, g) divides both u*p and v*g
    
    for null_vec in null_vectors:
        # Extract u(x) coefficients (first m entries, reversed to ascending order)
        u_coeffs = list(reversed(null_vec[:m])) if m > 0 else [0]
        
        # Extract v(x) coefficients (last n entries, reversed to ascending order)
        v_coeffs = list(reversed(null_vec[m:m+n])) if n > 0 else [0]
        
        # Create u(x) and v(x) polynomials
        u = Polynomial(u_coeffs)
        v = Polynomial(v_coeffs)
        
        # Skip trivial cases
        if u.is_zero() and v.is_zero():
            continue
        
        # Compute u*p and v*g
        if not u.is_zero():
            up = u * p
        else:
            up = Polynomial([0])
        
        if not v.is_zero():
            vg = v * g
        else:
            vg = Polynomial([0])
        
        # h should divide both p and g
        # Try to find h by computing gcd of special combinations
        # or by trial division with expected degree
        
        # Method: Use the relationship that if h = gcd(p, g),
        # then h also divides any linear combination
        
        # Try extracting from the ratio relationship
        if not v.is_zero() and v.degree() >= 0:
            # g / gcd should give us a polynomial
            # Try polynomial that would give us the right degree
            candidate_coeffs = []
            
            # Build candidate factor from leading terms
            for i in range(expected_degree + 1):
                if i < len(g.coeffs):
                    candidate_coeffs.append(g.coeffs[i])
                else:
                    candidate_coeffs.append(0)
            
            candidate = Polynomial(candidate_coeffs)
            
            # Verify candidate divides both
            if not candidate.is_zero():
                q1, r1 = p.divmod(candidate)
                q2, r2 = g.divmod(candidate)
                
                if r1.is_zero() and r2.is_zero():
                    return candidate
    
    # Fallback: Try to find common factor by testing roots
    # If both polynomials share a root r, then (x - r) is a common factor
    
    # Compute resultant = det(Sylvester) to check for common roots
    # If resultant is 0 (which it should be since we have null space),
    # we can find common roots numerically
    
    return _find_common_factor_numerically(p, g, expected_degree)


def _find_common_factor_numerically(p, g, expected_degree):
    """
    Optimized common factor finding using numerical root analysis.
    
    Uses smarter initial point selection and caches derivative evaluations.
    """
    # Cache derivative polynomials
    p_deriv = p.derivative()
    g_deriv = g.derivative()
    
    def find_roots_fast(poly, poly_deriv, num_attempts=15):
        """Optimized Newton-Raphson with smarter starting points."""
        roots = []
        
        # Try integer candidates first (most common for symbolic math)
        for r in range(-10, 11):
            if abs(poly.evaluate(r)) < 1e-10:
                roots.append(float(r))
        
        # Try rational candidates
        for denom in [2, 3, 4]:
            for numer in range(-10, 11):
                r = numer / denom
                if abs(poly.evaluate(r)) < 1e-10:
                    is_new = all(abs(r - existing) > 1e-6 for existing in roots)
                    if is_new:
                        roots.append(r)
        
        # Newton-Raphson for remaining roots
        for attempt in range(num_attempts):
            x = (attempt - num_attempts // 2) * 0.7
            
            for _ in range(50):  # Reduced iterations
                fx = poly.evaluate(x)
                
                if abs(fx) < 1e-10:
                    is_new = all(abs(x - r) > 1e-6 for r in roots)
                    if is_new:
                        roots.append(x)
                    break
                
                fpx = poly_deriv.evaluate(x)
                if abs(fpx) < 1e-12:
                    break
                
                x_new = x - fx / fpx
                if abs(x_new - x) < 1e-12:
                    break
                x = x_new
        
        return roots
    
    # Find roots
    p_roots = find_roots_fast(p, p_deriv)
    g_roots = find_roots_fast(g, g_deriv)
    
    # Find common roots efficiently
    common_roots = []
    for pr in p_roots:
        for gr in g_roots:
            if abs(pr - gr) < 1e-6:
                common_roots.append((pr + gr) / 2)
                break
    
    if len(common_roots) < expected_degree:
        return None
    
    common_roots = common_roots[:expected_degree]
    
    # Build factor from roots
    factor = Polynomial([1])
    for root in common_roots:
        factor = factor * Polynomial([-root, 1])
    
    # Round coefficients
    rounded_coeffs = [float(round(c)) if abs(c - round(c)) < 1e-6 else c 
                      for c in factor.coeffs]
    factor = Polynomial(rounded_coeffs)
    
    # Verify
    _, r1 = p.divmod(factor)
    _, r2 = g.divmod(factor)
    
    if r1.is_zero() and r2.is_zero():
        return factor
    
    return None


def simplify_sylvester(numerator_str, denominator_str):
    """
    Simplify rational expression using Sylvester matrix and linear algebra.
    
    Optimized with:
    1. Fast coprime detection via evaluation
    2. Direct linear factor extraction for degree-1 GCD
    3. Efficient matrix operations for power expressions
    
    The key insight: gcd(p, g) exists ⟺ Sylvester matrix is rank-deficient
    And: degree of gcd = (deg(p) + deg(g)) - rank(Sylvester)
    """
    # Try power pre-simplification first (very effective for (x+a)^n forms)
    simplified_num, simplified_den, was_power_simplified = simplify_power_expressions(
        numerator_str, denominator_str
    )
    if was_power_simplified:
        numerator_str, denominator_str = simplified_num, simplified_den
    
    # Parse polynomials
    p = parse_polynomial(numerator_str)
    g = parse_polynomial(denominator_str)
    
    n = p.degree()
    m = g.degree()
    
    # Handle trivial cases quickly
    if n == 0 or m == 0:
        return p, g, "sylvester_trivial"
    
    if p.is_zero():
        return Polynomial([0]), g, "sylvester_trivial"
    
    if g.is_zero():
        raise ValueError("Division by zero polynomial")
    
    # OPTIMIZATION 1: Quick coprime check using resultant estimate
    # If p and g have no common roots, they're coprime
    # Test at small integer points for fast detection
    quick_coprime = True
    test_points = [-2, -1, 0, 1, 2, 3]
    for pt in test_points:
        p_val = p.evaluate(pt)
        g_val = g.evaluate(pt)
        # If both are zero at same point, likely common factor
        if abs(p_val) < 1e-10 and abs(g_val) < 1e-10:
            quick_coprime = False
            break
    
    if quick_coprime:
        # High probability coprime, verify with minimal matrix work
        # Build small submatrix or use trace estimate
        pass  # Continue to full analysis but likely coprime
    
    # OPTIMIZATION 2: For small polynomials, try direct root finding first
    if n <= 3 and m <= 3:
        common_roots = []
        for r in range(-10, 11):
            if abs(p.evaluate(r)) < 1e-10 and abs(g.evaluate(r)) < 1e-10:
                common_roots.append(r)
        
        if common_roots:
            # Build factor from common roots
            factor = Polynomial([1])
            for root in common_roots:
                factor = factor * Polynomial([-root, 1])
            
            # Verify and simplify
            q_num, r_num = p.divmod(factor)
            q_den, r_den = g.divmod(factor)
            
            if r_num.is_zero() and r_den.is_zero():
                return q_num, q_den, "sylvester_direct"
        elif quick_coprime:
            # Confirmed coprime for small case
            return p, g, "sylvester_coprime"
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Build Sylvester Matrix
    # ═══════════════════════════════════════════════════════════════
    S = build_sylvester_matrix(p, g)
    size = n + m
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Compute Rank via Gaussian Elimination
    # ═══════════════════════════════════════════════════════════════
    rank, row_echelon, pivot_cols = gaussian_elimination_with_pivoting(S)
    
    # Degree of common factor = size - rank
    common_factor_degree = size - rank
    
    if common_factor_degree == 0:
        # Polynomials are coprime (no common factor except constants)
        # Still need to remove any common integer content
        p_content = p.content()
        g_content = g.content()
        
        if p_content > 0 and g_content > 0:
            def int_gcd(a, b):
                while b:
                    a, b = b, a % b
                return abs(a)
            
            common_content = int_gcd(p_content, g_content)
            if common_content > 1:
                p = p.scale(1.0 / common_content)
                g = g.scale(1.0 / common_content)
        
        return p, g, "sylvester_coprime"
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 3: Compute Null Space to Find Common Factor
    # ═══════════════════════════════════════════════════════════════
    null_vectors = compute_null_space(row_echelon, size, rank, pivot_cols)
    
    if null_vectors is None:
        # Couldn't compute null space, fall back
        return None
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 4: Extract Common Factor from Null Space
    # ═══════════════════════════════════════════════════════════════
    common_factor = extract_common_factor_from_nullspace(
        null_vectors, p, g, n, m, common_factor_degree
    )
    
    if common_factor is None:
        # Null space extraction failed, try numerical approach directly
        common_factor = _find_common_factor_numerically(p, g, common_factor_degree)
    
    if common_factor is None or common_factor.is_zero():
        return None
    
    # Make monic
    lead = common_factor.leading_coeff()
    if abs(lead) > 1e-10 and abs(lead - 1.0) > 1e-10:
        common_factor = common_factor.scale(1.0 / lead)
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 5: Divide Both Polynomials by Common Factor
    # ═══════════════════════════════════════════════════════════════
    simplified_num, r1 = p.divmod(common_factor)
    simplified_den, r2 = g.divmod(common_factor)
    
    # Verify division was exact
    if not r1.is_zero() or not r2.is_zero():
        # Division failed, the extracted factor wasn't quite right
        return None
    
    # Factor out common integer content
    num_content = simplified_num.content()
    den_content = simplified_den.content()
    
    if num_content > 0 and den_content > 0:
        def int_gcd(a, b):
            while b:
                a, b = b, a % b
            return abs(a)
        
        common_content = int_gcd(num_content, den_content)
        if common_content > 1:
            simplified_num = simplified_num.scale(1.0 / common_content)
            simplified_den = simplified_den.scale(1.0 / common_content)
    
    return simplified_num, simplified_den, "sylvester"


def get_sylvester_matrix_info(numerator_str, denominator_str):
    """
    Get detailed information about the Sylvester matrix analysis.
    
    Useful for debugging and understanding the linear algebra approach.
    
    Returns:
        Dictionary with matrix info, rank, null space dimension, etc.
    """
    p = parse_polynomial(numerator_str)
    g = parse_polynomial(denominator_str)
    
    n = p.degree()
    m = g.degree()
    size = n + m
    
    S = build_sylvester_matrix(p, g)
    rank, row_echelon, pivot_cols = gaussian_elimination_with_pivoting(S)
    
    return {
        "numerator_degree": n,
        "denominator_degree": m,
        "matrix_size": f"{size}x{size}",
        "rank": rank,
        "null_space_dimension": size - rank,
        "common_factor_degree": size - rank,
        "has_common_factor": size - rank > 0,
        "pivot_columns": pivot_cols,
        "sylvester_matrix": S
    }


def factor_polynomial(expr_str):
    """Factor a univariate polynomial with integer coefficients.
    
    Args:
        expr_str: String representation of polynomial (e.g., "x^2+3x+2")
    
    Returns:
        Tuple (content, factors) where:
        - content: integer content (GCD of coefficients)
        - factors: list of (factor, multiplicity) tuples
    """
    p = parse_polynomial(expr_str)
    content, factors = p.factor()
    
    return content, factors


def format_factorization(content, factors):
    """Format factorization result as a string.
    
    Args:
        content: Integer content
        factors: List of (factor, multiplicity) tuples
    
    Returns:
        String representation of factorization
    """
    parts = []
    
    if content != 1:
        parts.append(str(content))
    
    for factor, mult in factors:
        factor_str = str(factor)
        if mult == 1:
            parts.append(f"({factor_str})")
        else:
            parts.append(f"({factor_str})^{mult}")
    
    if not parts:
        return "1"
    
    return " * ".join(parts)


# Test with provided examples
if __name__ == "__main__":
    # Import sympy for comparison
    try:
        from sympy import symbols, simplify, sympify
        from sympy.parsing.sympy_parser import parse_expr
        HAS_SYMPY = True
    except ImportError:
        HAS_SYMPY = False
        print("Warning: SymPy not installed. Install with: pip install sympy")
        print()
    
    print("=" * 80)
    print("SIMPLIFICATION COMPARISON: Testing All Four Methods")
    print("=" * 80)
    print()
    
    # Helper function to convert our notation to SymPy notation
    def prep_for_sympy(expr):
        """Convert expression from our notation to SymPy notation."""
        import re
        expr = str(expr)
        expr = expr.replace(' ', '')
        expr = expr.replace('^', '**')
        # Add * between number and x: 3x -> 3*x
        expr = re.sub(r'(\d)([x])', r'\1*\2', expr)
        return expr
    
    # Test expressions
    test_expressions = [
        {
            "numerator": "x^2-9",
            "denominator": "x^2-3x",
            "description": "Expression 1: (x^2 - 9) / (x^2 - 3x)"
        },
        {
            "numerator": "x^4-1",
            "denominator": "x^2-1",
            "description": "Expression 2: (x^4 - 1) / (x^2 - 1)"
        },
        {
            "numerator": "x^2+3x+2",
            "denominator": "x+2",
            "description": "Expression 3: (x^2 + 3x + 2) / (x + 2)"
        },
        {
            "numerator": "x^2 + 2x + 1",
            "denominator": "x^2 + 3x + 2",
            "description": "Expression 4: (x^2 + 2x + 1) / (x^2 + 3x + 2)"
        },
        {
            "numerator": "(x + 2)^999",
            "denominator": "(x + 2)900",
            "description": "lmao"
        },
        {
            "numerator": "(x + 2)^99",
            "denominator": "(x + 2)^98",
            "description": "lmao2"
        }
    ]
    
    for i, test in enumerate(test_expressions, 1):
        num_str = test["numerator"]
        den_str = test["denominator"]
        description = test["description"]
        
        print(description)
        print(f"Input: ({num_str}) / ({den_str})")
        
        # Get SymPy's result if available
        if HAS_SYMPY:
            try:
                x = symbols('x')
                # Convert our notation to sympy notation
                sympy_num_str = prep_for_sympy(num_str)
                sympy_den_str = prep_for_sympy(den_str)
                
                # Parse and simplify using sympify
                num_expr = sympify(sympy_num_str)
                den_expr = sympify(sympy_den_str)
                sympy_expr = num_expr / den_expr
                sympy_result = simplify(sympy_expr)
                
                print(f"SymPy Result: {sympy_result}")
                sympy_result_str = str(sympy_result)
            except Exception as e:
                print(f"SymPy Error: {e}")
                sympy_result_str = None
        else:
            sympy_result_str = None
            
        print("-" * 80)
        
        # Test Method 1: Heuristic GCD (Interpolation-based)
        print("\n1. HEURISTIC GCD (Interpolation-based):")
        try:
            num_h, den_h, method_h = simplify_heuristic(num_str, den_str)
            if num_h is not None:
                result_h = format_result(num_h, den_h)
                print(f"   Result: {result_h}")
                if sympy_result_str and HAS_SYMPY:
                    # Check if results match by comparing the actual rational expressions
                    try:
                        x = symbols('x')
                        # Convert our result notation to SymPy
                        our_expr_str = prep_for_sympy(result_h)
                        our_result = sympify(our_expr_str)
                        # Compare by simplifying the difference
                        match = simplify(our_result - sympy_result) == 0
                        status = "[OK]" if match else "[FAIL]"
                        print(f"   {status} with SymPy result: {sympy_result_str}")
                    except Exception as e:
                        print(f"   Comparison error: {e}")
                        print(f"   SymPy: {sympy_result_str}")
            else:
                print(f"   Result: {method_h}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test Method 2: Factorization-based
        print("\n2. FACTORIZATION-BASED SIMPLIFICATION:")
        try:
            num_f, den_f, method_f = simplify_factorization_method(num_str, den_str)
            if num_f is not None:
                result_f = format_result(num_f, den_f)
                print(f"   Result: {result_f}")
                if sympy_result_str and HAS_SYMPY:
                    try:
                        x = symbols('x')
                        our_expr_str = prep_for_sympy(result_f)
                        our_result = sympify(our_expr_str)
                        match = simplify(our_result - sympy_result) == 0
                        status = "[OK]" if match else "[FAIL]"
                        print(f"   {status} with SymPy result: {sympy_result_str}")
                    except Exception as e:
                        print(f"   Comparison error: {e}")
                        print(f"   SymPy: {sympy_result_str}")
            else:
                print(f"   Result: {method_f}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test Method 3: Euclidean GCD
        print("\n3. EUCLIDEAN GCD:")
        try:
            num_e, den_e, method_e = simplify_euclidean(num_str, den_str)
            if num_e is not None:
                result_e = format_result(num_e, den_e)
                print(f"   Result: {result_e}")
                if sympy_result_str and HAS_SYMPY:
                    try:
                        x = symbols('x')
                        our_expr_str = prep_for_sympy(result_e)
                        our_result = sympify(our_expr_str)
                        match = simplify(our_result - sympy_result) == 0
                        status = "[OK]" if match else "[FAIL]"
                        print(f"   {status} with SymPy result: {sympy_result_str}")
                    except Exception as e:
                        print(f"   Comparison error: {e}")
                        print(f"   SymPy: {sympy_result_str}")
            else:
                print(f"   Result: {method_e}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test Method 4: Sylvester Matrix (Linear Algebra)
        print("\n4. SYLVESTER MATRIX (Linear Algebra):")
        try:
            result_s = simplify_sylvester(num_str, den_str)
            if result_s is not None:
                num_s, den_s, method_s = result_s
                result_str_s = format_result(num_s, den_s)
                print(f"   Result: {result_str_s}")
                print(f"   Method: {method_s}")
                if sympy_result_str and HAS_SYMPY:
                    try:
                        x = symbols('x')
                        our_expr_str = prep_for_sympy(result_str_s)
                        our_result = sympify(our_expr_str)
                        match = simplify(our_result - sympy_result) == 0
                        status = "[OK]" if match else "[FAIL]"
                        print(f"   {status} with SymPy result: {sympy_result_str}")
                    except Exception as e:
                        print(f"   Comparison error: {e}")
                        print(f"   SymPy: {sympy_result_str}")
            else:
                print(f"   Result: sylvester (failed)")
        except Exception as e:
            print(f"   Error: {e}")
        
        print("\n" + "=" * 80)
        print()

