import streamlit as st
import sympy as sp
import numpy as np
from fractions import Fraction
import sys


# ------------------ Session State ------------------
sys.set_int_max_str_digits(1000000)

st.set_page_config(
    page_title="Regular Stochastic Matrix Calculator",
    page_icon="📐",
    layout="centered",
    initial_sidebar_state="collapsed"
)

if "fixed_point" not in st.session_state:
    st.session_state.fixed_point = None

if "powers" not in st.session_state:
    st.session_state.powers = {}  # key: n, value: A^n

if "matrix_signature" not in st.session_state:
    st.session_state.matrix_signature = None
# --------------------------------------------------

# ------------------ INPUT BOX STYLING (ADD HERE) ------------------
st.markdown(
    """
    <style>
    input[type="text"], input[type="number"] {
        border: 2px solid #333 !important;
        border-radius: 6px;
        padding: 8px;
        font-size: 16px;
        background-color: #ffffff;
        text-align: center;
    }

    /* Make matrix rows flexible and shrinkable */
    .matrix-row input[type="text"] {
        flex: 1 1 auto;
        min-width: 40px;
        max-width: 80px;
        box-sizing: border-box;
        margin: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ---------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Make Streamlit buttons more visible */
    div.stButton > button {
        border: 3px solid #333333;  /* thick blue border */
        border-radius: 8px;         /* rounded corners */
        padding: 10px 20px;         /* extra space inside */
        font-size: 20px;            /* larger text */
        background-color: #ffffff;  /* button background */
        color: #333333;             /* text color */
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Appearance ------------------
BG_COLOR = "#f5f6f7"
ACCENT_COLOR = "#2c3e50"
BTN_COLOR = "#3498db"
TEXT_COLOR = "#333333"

# Fonts (clarity)
TITLE_FONT  = ("Segoe UI", 24, "bold")
LABEL_FONT  = ("Segoe UI", 18)
ENTRY_FONT  = ("Consolas", 18)
OUTPUT_FONT = ("Consolas", 18)

# ------------------ Utility Functions ------------------
def show_matrix(M, name=""):
    M = sp.Matrix(M)
    latex = sp.latex(M).replace(r"\\", r"\\[6pt]")
    if name:
        st.latex(rf"{name} = {latex}")
    else:
        st.latex(latex)

def show_matrix_with_decimal(M, name="", decimals=5):
    M_sym = sp.Matrix(M)

    # Exact (fraction) LaTeX
    exact_latex = sp.latex(M_sym)

    # Decimal approximation
    M_dec = M_sym.evalf(decimals)
    dec_latex = sp.latex(M_dec)

    if name:
        st.latex(
            rf"{name} = {exact_latex} \;\approx\; {dec_latex}"
        )
    else:
        st.latex(
            rf"{exact_latex} \;\approx\; {dec_latex}"
        )

#def show_matrix_with_decimal_stacked(M, name="", decimals=5):
    #M_sym = sp.Matrix(M)

    # Exact (fraction)
    #exact_latex = sp.latex(M_sym).replace(r"\\", r"\\[6pt]")

    # Decimal approximation
    #dec_latex = sp.latex(M_sym.evalf(decimals))

    #st.latex(rf"{name} = {exact_latex}")
    #st.latex(rf"{name} \approx {dec_latex}")

def show_matrix_with_decimal_stacked(M, name="", decimals=5):
    M_sym = sp.Matrix(M)

    exact_latex = sp.latex(M_sym).replace(r"\\", r"\\[6pt]")
    dec_latex   = sp.latex(M_sym.evalf(decimals))

    st.latex(
        rf"""
        \begin{{aligned}}
        {name} &= {exact_latex} \\
        {name} &\approx {dec_latex}
        \end{{aligned}}
        """
    )

def validate_stochastic_matrix(A):
    size = len(A)

    for i in range(size):
        row_sum = Fraction(0)

        for j in range(size):
            val = A[i][j]

            # 1. Non-negative entries
            if val < 0:
                raise ValueError(
                    f"Matrix is not stochastic: entry A[{i+1},{j+1}] is negative."
                )

            row_sum += val

        # 2. Row sum equals 1
        if row_sum != 1:
            raise ValueError(
                f"Matrix is not stochastic: row {i+1} sums to {row_sum}, not 1."
            )

def is_proportional(v, v1):
    return sp.Matrix.hstack(v, v1).rank() == 1

def is_regular(A, max_power=10):
    size = len(A)
    M = sp.Matrix(A)

    for k in range(1, max_power + 1):
        Mk = M ** k
        if all(Mk[i, j] > 0 for i in range(size) for j in range(size)):
            return True
    return False

def matrix_fingerprint(A):
    return tuple(tuple(row) for row in A)

MAX_EXACT_POWER = 5000  # after this, switch to numeric mode

# ------------------ 2x2 Functions ------------------
def stationary_distribution_formula(a12, a21):
    X = Fraction(a21, a12 + a21)
    Y = Fraction(a12, a12 + a21)
    return np.array([X,Y])

def matrix_power_formula(A, n):
    a12 = A[0][1]
    a21 = A[1][0]
    detA = A[0][0]*A[1][1] - A[0][1]*A[1][0]
    X = Fraction(a21, a12 + a21)
    Y = Fraction(a12, a12 + a21)
    An_11 = X + Y*(detA**sp.Integer(n))
    An_12 = Y*(1 - (detA**sp.Integer(n)))
    An_21 = X*(1 - (detA**sp.Integer(n)))
    An_22 = Y + X*(detA**sp.Integer(n))
    return [[An_11, An_12],[An_21, An_22]]


# ------------------ 3x3 Functions ------------------
def stationary_distribution_3x3(A):
    a11,a12 = A[0][0],A[0][1]
    a21,a22 = A[1][0],A[1][1]
    a31,a32 = A[2][0],A[2][1]
    denominator = (1 + a32 - a22)*(1 + a31 - a11) - (a12 - a32)*(a21 - a31)
    α = (a31*(1 - a22) + a32*a21)/denominator
    β = (a32*(1 - a11) + a31*a12)/denominator
    δ = (1 - a11 - a22*(1 - a11) - a21*a12)/denominator
    return np.array([α, β, δ])

def b1_3x3(a11,a12,a21,a22,a31,a32):
    return sp.sqrt(
        a11*(a11 - 2*a22 - 2*a31 + 2*a32)
        + a22*(a22 + 2*a31 - 2*a32)
        + a31*(a31 + 2*a32 - 4*a12)
        + a32*(a32 - 4*a21)
        + 4*a12*a21
    )

def lambdas_3x3(a11,a12,a21,a22,a31,a32,b1):
    λ1 = (a11 + a22 - a31 - a32 - b1)/2
    λ2 = (a11 + a22 - a31 - a32 + b1)/2
    return sp.simplify(λ1), sp.simplify(λ2)

def p_values_3x3(a11,a12,a21,a22,a31,a32,b1):
    D = a12*a31**2 - a21*a32**2 - a31*a32*(a11-a22)

    if sp.simplify(D) == 0:
        raise ValueError(
            "The closed-form for Aⁿ of 3×3 regular stochastic matrix is not valid for this matrix "
            "(singular eigenstructure)."
        )

    p1 = (-a31*(a12*(1-a31)-a32*(1-a11+a12))
          -a32*(a12*a21+a22*(1-a11)-a32*(1-a11))
          -(a32*(1-a11)+a12*a31)*(a31-a22-a11+a32+b1)/2)/D

    p2 = (a31*(a12*a21+a11*(1-a22)-a31*(1-a22))
          +a32*(a21*(1-a32)-a31*(1+a21-a22))
          +(a31*(1-a22)+a21*a32)*(a31-a22-a11+a32+b1)/2)/D

    p3 = (-a31*(a12*(1-a31)-a32*(1-a11+a12))
          -a32*(a12*a21+a22*(1-a11)-a32*(1-a11))
          +(a32*(1-a11)+a12*a31)*(a11+a22-a31-a32+b1)/2)/D

    p4 = (a31*(a12*a21+a11*(1-a22)-a31*(1-a22))
          +a32*(a21*(1-a32)-a31*(1+a21-a22))
          -(a31*(1-a22)+a21*a32)*(a11+a22-a31-a32+b1)/2)/D

    return map(sp.simplify,(p1,p2,p3,p4))

def c_values_3x3(a11,a12,a21,a22,a31,a32,b1):
    D = 2*b1*(1 - a11*(1-a22) - a21*(a12-a32)
              - a22*(1+a31) + a31*(1+a12) + a32*(1-a11))

    if sp.simplify(D) == 0:
        raise ValueError(
            "The closed-form for Aⁿ of 3×3 regular stochastic matrix is not valid for this matrix "
            "(Matrix is not diagonalizable)."
        )

    c1 = (-a31*(-a11+a22*(1+a11-a22)+b1*(1-a22)
                +a31*(1+2*a12-a22)-2*a12*a21)
          -a32*(a21*(b1+a11+a22-a32-2)
                -a31*(2*a11-a21-a22-1)))/D

    c2 = (a31*(-a11+a22*(1+a11-a22)+b1*(a22-1)
               +a31*(1+2*a12-a22)-2*a12*a21)
          +a32*(a21*(-b1+a11+a22-a32-2)
                -a31*(2*a11-a21-a22-1)))/D

    c3 = (-a31*(a12*(b1+a22+a11-a31-2)
                -a32*(2*a22-a12-a11-1))
          -a32*(-a22+a11*(1+a22-a11)
                +b1*(1-a11)+a32*(1+2*a21-a11)
                -2*a12*a21))/D

    c4 = (a31*(a12*(-b1+a22+a11-a31-2)
               -a32*(2*a22-a12-a11-1))
          +a32*(-a22+a11*(1+a22-a11)
                +b1*(a11-1)+a32*(1+2*a21-a11)
                -2*a12*a21))/D

    c5 = (a31*(a11*(a12+a22-a32-1)
               +a12*(a22+a32-2*a21-2))
          +a31*(a22*(1-a22)+a31*(1+a12-a22)
                +b1*(1+a12-a22))
          +a11*a32*(1-a11-a32)
          +a32*(a21*(a11-2*a12+a31)
                +a22*(a11+a21-a31-1)+a32*(1+a21)
                +b1*(1-a11+a21)-2*(a21-a31)))/D

    c6 = (-a31*(a11*(a12+a22-a32-1)
                +a12*(a22+a32-2*a21-2))
          -a31*(a22*(1-a22)+a31*(1+a12-a22)
                +b1*(a22-a12-1))
          -a11*a32*(1-a11-a32)
          -a32*(a21*(a11-2*a12+a31)
                +a22*(a11+a21-a31-1)+a32*(1+a21)
                +b1*(a11-a21-1)-2*(a21-a31)))/D

    return map(sp.simplify,(c1,c2,c3,c4,c5,c6))

def matrix_power_formula_3x3_explicit(n,α,β,δ,λ1,λ2,p1,p2,p3,p4,c1,c2,c3,c4,c5,c6):
    λ1 = λ1**sp.Integer(n)
    λ2 = λ2**sp.Integer(n)

    return sp.simplify(sp.Matrix([
        [α+c1*p1*λ1+c2*p3*λ2, β+c3*p1*λ1+c4*p3*λ2, δ+c5*p1*λ1+c6*p3*λ2],
        [α+c1*p2*λ1+c2*p4*λ2, β+c3*p2*λ1+c4*p4*λ2, δ+c5*p2*λ1+c6*p4*λ2],
        [α+c1*λ1+c2*λ2,      β+c3*λ1+c4*λ2,      δ+c5*λ1+c6*λ2]
    ]))

# --- Stationary distribution for 4x4 (using given formula) ---
def stationary_distribution_4x4(A):
    a11, a12, a13, a14 = A[0]
    a21, a22, a23, a24 = A[1]
    a31, a32, a33, a34 = A[2]
    a41, a42, a43, a44 = A[3]
    # α
    numerator_alpha = (
        a41*(1 - a22 - a33*(1 - a22) - a23*a32)
        + a42*(a21*(1 - a33) + a23*a31)
        + a43*(a31*(1 - a22) + a21*a32)
    )
    denominator_alpha = (
        ((1 - a11 + a41)*(1 - a22 + a42) - (a12 - a42)*(a21 - a41))*(1 - a33 + a43)
        - ((1 - a11 + a41)*(a32 - a42) + (a12 - a42)*(a31 - a41))*(a23 - a43)
        - ((a21 - a41)*(a32 - a42) + (1 - a22 + a42)*(a31 - a41))*(a13 - a43)
    )
    alpha = numerator_alpha / denominator_alpha
    # β
    numerator_beta = (
        a41*(a12*(1 - a33) + a13*a32)
        + a42*(1 - a11 - a33*(1 - a11) - a13*a31)
        + a43*(a32*(1 - a11) + a12*a31)
    )
    beta = numerator_beta / denominator_alpha
    # δ
    numerator_delta = (
        a41*(a13*(1 - a22) + a12*a23)
        + a42*(a23*(1 - a11) + a13*a21)
        + a43*(1 - a11 - a22*(1 - a11) - a12*a21)
    )
    delta = numerator_delta / denominator_alpha
    # γ
    numerator_gamma = (
        1 - a22 - a11*(1 - a22 - a33) - a12*(a21*(1 - a33) + a23*a31)
        - a13*(a31*(1 - a22) + a21*a32) - a23*a32*(1 - a11)
        - a33*(1 - a22*(1 - a11))
    )
    gamma = numerator_gamma / denominator_alpha
    return np.array([alpha, beta, delta, gamma])

# --- Diagonalization of 4x4 (new) ---
def diagonalize_matrix_4x4(A):
    """
    Returns: P, D, P_inv, metadata
    P: 4x4 sympy Matrix with first column = [1,1,1,1]^T
    D: diag(1, lambda1, lambda2, lambda3)
    P_inv: inverse of P
    metadata: dict containing p_values, c_values, alpha/beta/delta/gamma, lambda_values
    """
    # Ensure sympy Matrix of rationals
    A_sym = sp.Matrix([[sp.Rational(a) for a in row] for row in A])

    # Compute eigenvects
    eigen_data = A_sym.eigenvects()

    eig_pairs = []
    for val, mult, vects in eigen_data:
        for v in vects:
            eig_pairs.append((sp.simplify(val), sp.Matrix(v)))

    # First eigenvector: ones (for row-stochastic A)
    v1 = sp.Matrix([1, 1, 1, 1])

    # Build remaining list excluding eigenvectors proportional to v1
    remaining = []
    for lam, v in eig_pairs:
        # check if v is proportional to v1

        if not is_proportional(sp.Matrix(v), v1):
            remaining.append((sp.simplify(lam), sp.Matrix(v)))

    # If not enough, add from eig_pairs (avoid duplicates)
    if len(remaining) < 3:
        for lam, v in eig_pairs:
            if len(remaining) >= 3:
                break
            if not any(sp.Matrix(v).equals(r[1]) for r in remaining):
                # avoid proportional to v1

                if not is_proportional(sp.Matrix(v), v1):
                    remaining.append((sp.simplify(lam), sp.Matrix(v)))

    # Numeric fallback to find independent eigenvectors if still short
    if len(remaining) < 3:
        numeric_eigs = A_sym.evalf().eigenvects()
        for val, mult, vects in numeric_eigs:
            for v in vects:
                if len(remaining) >= 3:
                    break
                v_rat = sp.Matrix([sp.nsimplify(x) for x in v])
                # avoid proportional to v1

                if not is_proportional(v_rat, v1):
                    remaining.append((sp.nsimplify(val), v_rat))
            if len(remaining) >= 3:
                break

    if len(remaining) < 3:
        raise ValueError("Could not find 3 independent eigenvectors to build P (matrix may not be diagonalizable).")

    # sort by numeric magnitude descending to mimic 3x3 behavior
    def numeric_key(pair):
        lam = pair[0]
        try:
            return float(sp.Abs(sp.N(lam)))
        except Exception:
            return float(abs(complex(sp.N(lam))))

    remaining_sorted = sorted(remaining, key=numeric_key, reverse=True)[:3]

    lam1 = sp.simplify(remaining_sorted[0][0])
    lam2 = sp.simplify(remaining_sorted[1][0])
    lam3 = sp.simplify(remaining_sorted[2][0])

    v2 = sp.Matrix(remaining_sorted[0][1])
    v3 = sp.Matrix(remaining_sorted[1][1])
    v4 = sp.Matrix(remaining_sorted[2][1])

    # Final diagonalizability check: eigenvectors must be independent
    P_test = sp.Matrix.hstack(v1, v2, v3, v4)
    if P_test.rank() < 4:
        raise ValueError(
            "Matrix is not diagonalizable: eigenvectors do not form a basis."
        )

    P = sp.Matrix.hstack(v1, v2, v3, v4)
    D = sp.diag(sp.Integer(1), lam1, lam2, lam3)

    try:
        if P.det() == 0:
            raise ValueError("Eigenvector matrix P is singular; cannot invert.")
        P_inv = sp.simplify(P.inv())
    except Exception:
        # fallback: numeric inversion then nsimplify
        P_inv = sp.simplify(P.evalf().inv())
        P_inv = sp.Matrix([[sp.nsimplify(x) for x in row] for row in P_inv.tolist()])

    # Extract p-values and c-values in the template order
    p_values = [
        sp.simplify(P[0,1]), sp.simplify(P[1,1]), sp.simplify(P[2,1]), sp.simplify(P[3,1]),
        sp.simplify(P[0,2]), sp.simplify(P[1,2]), sp.simplify(P[2,2]), sp.simplify(P[3,2]),
        sp.simplify(P[0,3]), sp.simplify(P[1,3]), sp.simplify(P[2,3]), sp.simplify(P[3,3]),
    ]

    alpha = sp.simplify(P_inv[0,0]); beta = sp.simplify(P_inv[0,1])
    delta = sp.simplify(P_inv[0,2]); gamma = sp.simplify(P_inv[0,3])

    c_values = [
        sp.simplify(P_inv[1,0]), sp.simplify(P_inv[2,0]), sp.simplify(P_inv[3,0]),
        sp.simplify(P_inv[1,1]), sp.simplify(P_inv[2,1]), sp.simplify(P_inv[3,1]),
        sp.simplify(P_inv[1,2]), sp.simplify(P_inv[2,2]), sp.simplify(P_inv[3,2]),
        sp.simplify(P_inv[1,3]), sp.simplify(P_inv[2,3]), sp.simplify(P_inv[3,3]),
    ]

    metadata = {
        'p_values': p_values,
        'c_values': c_values,
        'alpha': alpha,
        'beta': beta,
        'delta': delta,
        'gamma': gamma,
        'lambda_values': (lam1, lam2, lam3),
        'P': P,
        'D': D,
        'P_inv': P_inv
    }

    return P, D, P_inv, metadata


# --- EXPLICIT 4x4 formula (merged) ---
def matrix_power_formula_4x4_explicit(n, meta):
    """
    Build A^n with the explicit template (matching your image/template).
    meta is the dict returned by diagonalize_matrix_4x4.
    """
    # unpack meta
    p_vals = meta['p_values']   # p1..p12 in order: P[0,1],P[1,1],P[2,1],P[3,1], P[0,2],...,P[3,2], P[0,3],...,P[3,3]
    c_vals = meta['c_values']   # c1..c12 in order: P_inv[1,0],P_inv[2,0],P_inv[3,0], P_inv[1,1],...
    alpha = sp.simplify(meta['alpha'])
    beta  = sp.simplify(meta['beta'])
    delta = sp.simplify(meta['delta'])
    gamma = sp.simplify(meta['gamma'])
    lam1, lam2, lam3 = meta['lambda_values']

    # build p-blocks: each block is length-4 vector (rows 0..3)
    p_block1 = [sp.simplify(x) for x in p_vals[0:4]]   # p1..p4  (column 2 of P)
    p_block2 = [sp.simplify(x) for x in p_vals[4:8]]   # p5..p8  (column 3 of P)
    p_block3 = [sp.simplify(x) for x in p_vals[8:12]]  # p9..p12 (column 4 of P)

    # c groups for each output column (each group has 3 elements c_k for lam1,lam2,lam3)
    c_group_col0 = [sp.simplify(x) for x in c_vals[0:3]]   # c1,c2,c3
    c_group_col1 = [sp.simplify(x) for x in c_vals[3:6]]   # c4,c5,c6
    c_group_col2 = [sp.simplify(x) for x in c_vals[6:9]]   # c7,c8,c9
    c_group_col3 = [sp.simplify(x) for x in c_vals[9:12]]  # c10,c11,c12

    # prepare powers
    lam1n = lam1**sp.Integer(n)
    lam2n = lam2**sp.Integer(n)
    lam3n = lam3**sp.Integer(n)

    # build matrix row by row
    rows = []
    for r in range(4):
        p1r = p_block1[r]
        p2r = p_block2[r]
        p3r = p_block3[r]

        # column 0
        col0 = alpha + c_group_col0[0]*p1r*lam1n + c_group_col0[1]*p2r*lam2n + c_group_col0[2]*p3r*lam3n
        # column 1
        col1 = beta  + c_group_col1[0]*p1r*lam1n + c_group_col1[1]*p2r*lam2n + c_group_col1[2]*p3r*lam3n
        # column 2
        col2 = delta + c_group_col2[0]*p1r*lam1n + c_group_col2[1]*p2r*lam2n + c_group_col2[2]*p3r*lam3n
        # column 3
        col3 = gamma + c_group_col3[0]*p1r*lam1n + c_group_col3[1]*p2r*lam2n + c_group_col3[2]*p3r*lam3n

        rows.append([sp.simplify(col0), sp.simplify(col1), sp.simplify(col2), sp.simplify(col3)])

    A_n = sp.Matrix(rows)
    return sp.simplify(A_n)


# --- Keep a simple diagonalization fallback (still useful) ---
def matrix_power_formula_4x4(n, P, D, P_inv):
    Dn = sp.diag(sp.Integer(1), lam1**sp.Integer(n), lam2**sp.Integer(n), lam3**sp.Integer(n))
    A_n = sp.simplify(P * Dn * P_inv)
    return sp.simplify(A_n)


# --- Store values once ---
stored_constants = {
    "c_values": None,
    "p_values": None,
    "lambda_values": None
}

# --- Big bold title ---
st.markdown(
    """
    <h1 style="
        font-size: 36px;   /* adjust size */
        font-weight: bold;
        margin-bottom: 10px;  /* space below title */
    ">
        Regular Stochastic Matrix Calculator
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="
        font-size: 18px;
        margin-bottom: 0px;
        text-align: justify;
    ">
        <strong>Instructions:</strong> Use fractions values only (e.g., 1/2) when entering entries of a regular stochastic matrix. Ensure that each row sums to 1, as required for a row-stochastic matrix.
    </p>
    """,
    unsafe_allow_html=True
)


# --- Matrix size selector in line with label ---
col_label, col_radio = st.columns([1, 3])

with col_label:
    st.markdown(
        """
        <div style="
            font-size: 22px; 
            line-height:97px; 
            height:97px;
        ">
            Matrix size:
        </div>
        """,
        unsafe_allow_html=True
    )

with col_radio:
    size_selector = st.radio(
        "",
        [2, 3, 4],
        index=1,
        horizontal=True
    )

def build_grid(size):
    matrix = []
    for i in range(size):
        # Use a div flex container for the row
        st.markdown(f'<div class="matrix-row" style="display:flex; justify-content:center; flex-wrap:wrap;">', unsafe_allow_html=True)

        # Give each column equal fraction of available space
        cols = st.columns(size, gap="small")
        row = []
        for j in range(size):
            val = cols[j].text_input(
                "",
                value="",
                key=f"{i}-{j}",
                label_visibility="collapsed"
            )
            row.append(val)
        matrix.append(row)
        st.markdown('</div>', unsafe_allow_html=True)
    return matrix

matrix_entries = build_grid(size_selector)

def read_matrix():
    size = size_selector
    A = []
    errors = []
    suggestions = []

    for i in range(size):
        row = []
        empty_indices = []
        row_sum = Fraction(0)

        # Read row entries
        for j in range(size):
            text = matrix_entries[i][j].strip()

            if text == "":
                row.append(None)
                empty_indices.append(j)
            else:
                val = Fraction(text) if '/' in text else Fraction(float(text))
                if val < 0:
                    errors.append(f"Negative entry at row {i+1}, column {j+1}")
                row.append(val)
                row_sum += val

        # Case 1: exactly one empty → auto-fill
        if len(empty_indices) == 1:
            missing = Fraction(1) - row_sum
            if missing < 0:
                errors.append(f"Row {i+1} exceeds sum 1")
            else:
                row[empty_indices[0]] = missing
                row_sum += missing

        # Case 2: multiple empties → not ready
        elif len(empty_indices) > 1:
            errors.append(f"Row {i+1} has multiple empty entries")

        # Case 3: full row but wrong sum → suggest fix
        elif row_sum != 1:
            diff = abs(row_sum - 1)

            if row_sum > 1:
                action = f"subtracting {diff}"
            else:
                action = f"adding {diff}"

            suggestions.append(
                f"Row {i+1} sums to {row_sum}. "
                f"Change any entry by {action} to make the row sum to 1."
            )
            errors.append("Row sum incorrect")

        A.append(row)

    # ---- SHOW SUGGESTIONS FIRST ----
    for msg in suggestions:
        st.info("💡 " + msg)

    # ---- STOP IF ERRORS EXIST ----
    if errors:
        raise ValueError(errors[0])

    # ---- detect matrix change ----
    fp = matrix_fingerprint(A)
    if st.session_state.matrix_signature != fp:
        st.session_state.matrix_signature = fp
        st.session_state.fixed_point = None
        st.session_state.powers = {}

    return A

try:
    A_current = read_matrix()
    st.markdown("### Original Matrix A")
    show_matrix(A_current, "A")
except Exception as e:
    st.warning(f"Matrix not ready: {e}")

def compute_fixed_point(b):
    try:
        A = read_matrix()
        size = len(A)

        # --- Regularity check ---
        if not is_regular(A, max_power=50):
            st.warning(
                "⚠ WARNING: Matrix may not be regular.\n"
                "Convergence to a unique fixed probability vector is not guaranteed."
            )
            return  # skip computing fixed point

        if size == 2:
            pi = stationary_distribution_formula(A[0][1], A[1][0])
        elif size == 3:
            pi = stationary_distribution_3x3(A)
        else:
            pi = stationary_distribution_4x4(A)

        pi = sp.Matrix([sp.Rational(p) for p in pi]).T
        st.session_state.fixed_point = pi

    except Exception as e:
        st.error(str(e))

if st.button("Compute Fixed Point"):
    compute_fixed_point(None)

if st.session_state.fixed_point is not None:
    show_matrix_with_decimal(
        st.session_state.fixed_point,
        r"\text{Fixed point } t"
    )

st.markdown(
    """
    <p style="
        font-size:22px;
        font-weight:600;
        margin-bottom:4px;
    ">
        Enter n to compute Aⁿ
    </p>
    """,
    unsafe_allow_html=True
)

n_input = st.number_input(
    "",
    min_value=1,
    value=1,
    step=1
)

def compute_power(b):
    try:
        A = read_matrix()
        n = int(n_input)
        use_numeric = n > MAX_EXACT_POWER
        size = len(A)

        # --- Regularity check ---
        if not is_regular(A, max_power=50):
            st.warning(
                "⚠ WARNING: Matrix may not be regular.\n"
                "Convergence to a unique fixed probability vector is not guaranteed."
            )
            return  # skip computing fixed point

        if size == 2:
            A_n = matrix_power_formula(A, n)
            if use_numeric:
                A_n = sp.Matrix(A_n).evalf(8)
            st.session_state.powers[n] = A_n

        elif size == 3:
            try:
                α, β, δ = stationary_distribution_3x3(A)
                a11,a12,_ = A[0]
                a21,a22,_ = A[1]
                a31,a32,_ = A[2]

                b1 = b1_3x3(a11,a12,a21,a22,a31,a32)
                λ1, λ2 = lambdas_3x3(a11,a12,a21,a22,a31,a32,b1)
                p1,p2,p3,p4 = p_values_3x3(a11,a12,a21,a22,a31,a32,b1)
                c1,c2,c3,c4,c5,c6 = c_values_3x3(a11,a12,a21,a22,a31,a32,b1)

                A_n = matrix_power_formula_3x3_explicit(
                    n, α, β, δ, λ1, λ2,
                    p1,p2,p3,p4, c1,c2,c3,c4,c5,c6
                )
                if use_numeric:
                    A_n = A_n.evalf(8)

            except ValueError as e:
                st.warning(f"Use standard matrix multiplication: {e}")
                A_n = sp.Matrix(A) ** n

            st.session_state.powers[n] = A_n

        else:
            try:
                P, D, P_inv, meta = diagonalize_matrix_4x4(A)
                A_n = matrix_power_formula_4x4_explicit(n, meta)
                if use_numeric:
                    A_n = A_n.evalf(8)

            except (ValueError, sp.NonInvertibleMatrixError) as e:
                st.warning(f"Use standard matrix multiplication: {e}")
                A_n = sp.Matrix(A) ** n

            st.session_state.powers[n] = A_n

    except Exception as e:
        st.error(str(e))

if st.button("Compute Aⁿ"):
    compute_power(None)

# ------------------ DISPLAY STORED RESULTS ------------------

for n, A_n in sorted(st.session_state.powers.items()):
    if n > MAX_EXACT_POWER:
        st.latex(rf"A^{{{n}}} \approx {sp.latex(A_n)}")
    else:
        show_matrix_with_decimal_stacked(A_n, rf"A^{{{n}}}")
