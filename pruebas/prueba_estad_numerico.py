"""Pruebas del módulo estad_numerico (probabilidad, estadística, numérico, electrotecnia, IO).

Los valores esperados se calculan de forma INDEPENDIENTE del motor: con scipy/numpy si están instalados en la PC
(solo aquí, el motor no los usa), y si no con fórmulas cerradas / mpmath.
Uso:  cd app-ingecalc && PYTHONIOENCODING=utf-8 python pruebas/prueba_estad_numerico.py
"""
import cmath
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'motor'))
import sympy as sp  # noqa: E402

from ingecalc.api import resolver  # noqa: E402

try:
    import numpy as np
    from scipy import stats, interpolate, integrate, optimize
    HAY_SCIPY = True
except ImportError:  # pragma: no cover
    HAY_SCIPY = False

if not HAY_SCIPY:
    print('Este script necesita scipy y numpy en la PC (solo para calcular los valores esperados).')
    sys.exit(1)


def num(v):
    return complex(sp.N(sp.sympify(v)))


def lista(v):
    e = sp.sympify(v)
    if isinstance(e, sp.MatrixBase):
        return [complex(sp.N(q)) for q in e]
    return [complex(sp.N(q)) for q in e]


def cerca(a, b, tol):
    return abs(a - b) <= tol * max(1.0, abs(b))


# ----------------------------------------------------------------------------
# Valores esperados independientes
# ----------------------------------------------------------------------------
d6 = [2, 4, 4, 5, 7, 9]
xs_c = [1, 2, 3, 4, 5]
ys_c = [2.1, 3.9, 6.2, 7.8, 10.1]


def rk4_sis(f, t0, y0, h, tf):
    y_ = np.array(y0, float)
    t_ = t0
    for _ in range(int(round((tf - t0) / h))):
        k1 = f(t_, y_)
        k2 = f(t_ + h / 2, y_ + h / 2 * k1)
        k3 = f(t_ + h / 2, y_ + h / 2 * k2)
        k4 = f(t_ + h, y_ + h * k3)
        y_ = y_ + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t_ += h
    return list(y_)


def ab4(f, x0, y0, h, xf):
    xs, ys = [x0], [y0]
    for i in range(int(round((xf - x0) / h))):
        xn, yn = xs[-1], ys[-1]
        if i < 3:
            k1 = f(xn, yn); k2 = f(xn + h / 2, yn + h / 2 * k1); k3 = f(xn + h / 2, yn + h / 2 * k2); k4 = f(xn + h, yn + h * k3)
            y1 = yn + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            fs = [f(xs[-1 - j], ys[-1 - j]) for j in range(4)]
            y1 = yn + h / 24 * (55 * fs[0] - 59 * fs[1] + 37 * fs[2] - 9 * fs[3])
        xs.append(x0 + (i + 1) * h); ys.append(y1)
    return ys[-1]


def jacobi_np(A, b, its, gs=False):
    A = np.array(A, float); b = np.array(b, float); x_ = np.zeros(len(b))
    for _ in range(its):
        nuevo = x_.copy()
        for i in range(len(b)):
            fuente = nuevo if gs else x_
            nuevo[i] = (b[i] - sum(A[i, j] * fuente[j] for j in range(len(b)) if j != i)) / A[i, i]
        x_ = nuevo
    return list(x_)


def regula_np(f, a, b):
    return optimize.brentq(f, a, b, xtol=1e-14)


def hungaro(C):
    from scipy.optimize import linear_sum_assignment
    C = np.array(C)
    f, c = linear_sum_assignment(C)
    return C[f, c].sum()


def simpson38_np(f, a, b, n):
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (2 if i % 3 == 0 else 3) * f(a + i * h)
    return 3 * h / 8 * s


def romberg_np(f, a, b, n):
    R = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        m = 2 ** i
        xs = np.linspace(a, b, m + 1)
        R[i][0] = np.trapz(f(xs), xs) if hasattr(np, 'trapz') else np.trapezoid(f(xs), xs)
        for j in range(1, i + 1):
            R[i][j] = R[i][j - 1] + (R[i][j - 1] - R[i - 1][j - 1]) / (4 ** j - 1)
    return R[n][n]


def gl_np(f, a, b, n):
    t_, w_ = np.polynomial.legendre.leggauss(n)
    return (b - a) / 2 * sum(w_ * f((b - a) / 2 * t_ + (a + b) / 2))


def mmc(lam, mu, c):
    a = lam / mu
    rho = a / c
    p0 = 1 / (sum(a ** k / math.factorial(k) for k in range(c)) + a ** c / (math.factorial(c) * (1 - rho)))
    lq = p0 * a ** c * rho / (math.factorial(c) * (1 - rho) ** 2)
    return lq + a


def tabla_contingencia(M):
    chi2, p, dof, _ = stats.chi2_contingency(np.array(M), correction=False)
    return chi2


def muller_np():
    return [r_.real for r_ in np.roots([1, 0, -2, -5]) if abs(r_.imag) < 1e-9][0]


S_pc = 220 * 10 * cmath.exp(1j * math.radians(30))
Qc_fp = 10000 * (math.tan(math.acos(0.7)) - math.tan(math.acos(0.95)))
C_fp = Qc_fp / (2 * math.pi * 50 * 220 ** 2) * 1e6
pl = optimize.linprog([-3, -5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18], bounds=[(0, None)] * 2)
pl3 = optimize.linprog([2, 3, 1], A_ub=[[-1, -1, -1], [-1, 1, 0], [0, 1, 2]], b_ub=[-10, -2, 12], bounds=[(0, None)] * 3)
pl_min = optimize.linprog([1, 1], A_ub=[[-1, -2], [-3, -1]], b_ub=[-4, -6], bounds=[(0, None)] * 2)
cs = interpolate.CubicSpline([0, 1, 2, 3], [1, 3, 2, 5], bc_type='natural')
cs2 = interpolate.CubicSpline([1, 2, 4, 5, 7], [2, 1, 3, 0, 4], bc_type='natural')
xq = np.array([0, 1, 2, 3, 4.0]); yq = np.array([1, 1.8, 3.3, 4.5, 6.3])
cq = np.polyfit(xq, yq, 2)
xe = np.array([0, 1, 2, 3.0]); ye = np.array([1.5, 2.5, 3.5, 5.0])
be, ae = np.polyfit(xe, np.log(ye), 1)
r_pear, p_pear = stats.pearsonr(xs_c, ys_c)
obs = [18, 22, 30, 30]
esp_p = [0.25, 0.25, 0.25, 0.25]
chi_b = stats.chisquare(obs, [25] * 4)
f_prop = lambda X, Y: X * Y ** 2  # noqa
cuantiles_np = np.percentile([3, 7, 8, 5, 12, 14, 21, 13, 18], [25, 50, 75])
prop_err = abs(3 ** 2) * 0.1 + abs(2 * 2 * 3) * 0.2

# (entrada, esperado, tolerancia relativa[, clave de valores])
CASOS = [
    # --- Cuantiles y distribuciones continuas
    ('cuantil normal 0.975', stats.norm.ppf(0.975), 1e-9),
    ('cuantil normal 0.05', stats.norm.ppf(0.05), 1e-9),
    ('cuantil normal 0.9, 100, 15', stats.norm.ppf(0.9, 100, 15), 1e-9),
    ('cuantil t 0.975, 15', stats.t.ppf(0.975, 15), 1e-9),
    ('cuantil t 0.995, 3', stats.t.ppf(0.995, 3), 1e-9),
    ('cuantil t 0.9995, 1', stats.t.ppf(0.9995, 1), 1e-8),
    ('cuantil chi2 0.975, 15', stats.chi2.ppf(0.975, 15), 1e-9),
    ('cuantil chi2 0.025, 15', stats.chi2.ppf(0.025, 15), 1e-9),
    ('cuantil chi2 0.95, 1', stats.chi2.ppf(0.95, 1), 1e-9),
    ('cuantil f 0.95, 3, 10', stats.f.ppf(0.95, 3, 10), 1e-9),
    ('cuantil f 0.99, 5, 20', stats.f.ppf(0.99, 5, 20), 1e-9),
    ('normal inversa 100, 15, 0.9', stats.norm.ppf(0.9, 100, 15), 1e-9),
    ('normal inversa 50, 10, 0.05, derecha', stats.norm.isf(0.05, 50, 10), 1e-9),
    ('t student 15, -oo, 2.13145', stats.t.cdf(2.13145, 15), 1e-8),
    ('t student 10, -1, 2', stats.t.cdf(2, 10) - stats.t.cdf(-1, 10), 1e-8),
    ('chi2 15, 0, 27.488', stats.chi2.cdf(27.488, 15), 1e-8),
    ('chi2 4, 2, 6', stats.chi2.cdf(6, 4) - stats.chi2.cdf(2, 4), 1e-8),
    ('f snedecor 3, 10, 3.7083', stats.f.cdf(3.7083, 3, 10), 1e-8),
    ('exponencial 0.5, 0, 2', stats.expon.cdf(2, scale=2), 1e-9),
    ('exponencial 2, 1', stats.expon.cdf(1, scale=0.5), 1e-9),
    ('exponencial 0.25, 2, oo', stats.expon.sf(2, scale=4), 1e-9),
    ('uniforme 0, 10, 2, 5', 0.3, 1e-12),
    ('uniforme 2, 8, 7', stats.uniform.cdf(7, 2, 6), 1e-12),
    # --- Discretas
    ('geometrica 0.2, 3', stats.geom.pmf(3, 0.2), 1e-12),
    ('geometrica 0.5, 1', stats.geom.pmf(1, 0.5), 1e-12),
    ('hipergeometrica 20, 5, 4, 2', stats.hypergeom.pmf(2, 20, 5, 4), 1e-12),
    ('hipergeometrica 50, 10, 5, 0', stats.hypergeom.pmf(0, 50, 10, 5), 1e-12),
    # --- Bayes / probabilidad total
    ('bayes P(A)=0.01, P(B|A)=0.95, P(B|no A)=0.05', 0.01 * 0.95 / (0.01 * 0.95 + 0.99 * 0.05), 1e-12),
    ("bayes P(E)=0.3, P(+|E)=0.9, P(+|E')=0.2", 0.3 * 0.9 / (0.3 * 0.9 + 0.7 * 0.2), 1e-12),
    ('bayes prevalencia=0.01, sensibilidad=0.95, especificidad=0.95', 0.01 * 0.95 / (0.01 * 0.95 + 0.99 * 0.05), 1e-12),
    ('bayes 0.01, 0.95, 0.05', 0.0095 / 0.059, 1e-12),
    ('bayes P(A1)=0.5, P(A2)=0.3, P(A3)=0.2, P(B|A1)=0.02, P(B|A2)=0.03, P(B|A3)=0.04, P(A3|B)', 0.2 * 0.04 / 0.027, 1e-12),
    ('bayes [0.5,0.3,0.2], [0.02,0.03,0.04]', [0.01 / 0.027, 0.009 / 0.027, 0.008 / 0.027], 1e-12),
    ('probabilidad total [0.5,0.3,0.2], [0.02,0.03,0.04]', 0.027, 1e-12),
    ('probabilidad total P(A)=0.4, P(B|A)=0.1, P(B|no A)=0.3', 0.4 * 0.1 + 0.6 * 0.3, 1e-12),
    # --- Descriptiva
    ('percentil 90, 2, 4, 4, 5, 7, 9', np.percentile(d6, 90), 1e-12),
    ('percentil 25, 3, 7, 8, 5, 12, 14, 21, 13, 18', np.percentile([3, 7, 8, 5, 12, 14, 21, 13, 18], 25), 1e-12),
    ('percentil 50, 1, 2, 3, 4', np.percentile([1, 2, 3, 4], 50), 1e-12),
    ('asimetria 2, 4, 4, 5, 7, 9', stats.skew(d6, bias=False), 1e-9),
    ('asimetria 1, 2, 3, 10, 20, 50', stats.skew([1, 2, 3, 10, 20, 50], bias=False), 1e-9),
    ('curtosis 2, 4, 4, 5, 7, 9', stats.kurtosis(d6, bias=False), 1e-9),
    ('curtosis 1, 2, 3, 10, 20, 50, 3, 4', stats.kurtosis([1, 2, 3, 10, 20, 50, 3, 4], bias=False), 1e-9),
    ('cuartiles 3, 7, 8, 5, 12, 14, 21, 13, 18', list(cuantiles_np), 1e-12),
    ('tabla de frecuencias 2, 4, 4, 5, 7, 9', [1, 2, 1, 1, 1], 0),
    ('tabla de frecuencias 1,2,2,3,3,3,4,4,4,4; clases 2', [3, 7], 0),
    # --- Inferencia
    ('intervalo media x̄=50, σ=5, n=36, 95%', list(stats.norm.interval(0.95, 50, 5 / 6)), 1e-9),
    ('intervalo media x̄=50, s=5, n=16, 95%', list(stats.t.interval(0.95, 15, 50, 5 / 4)), 1e-9),
    ('intervalo media media=20, s=4, n=10, nc=99%', list(stats.t.interval(0.99, 9, 20, 4 / math.sqrt(10))), 1e-9),
    ('intervalo varianza s2=25, n=16, 95%', [15 * 25 / stats.chi2.ppf(0.975, 15), 15 * 25 / stats.chi2.ppf(0.025, 15)], 1e-9),
    ('intervalo proporcion p=0.4, n=100, 95%', [0.4 - stats.norm.ppf(0.975) * math.sqrt(0.24 / 100), 0.4 + stats.norm.ppf(0.975) * math.sqrt(0.24 / 100)], 1e-9),
    ('intervalo proporcion x=45, n=150, 90%', [0.3 - stats.norm.ppf(0.95) * math.sqrt(0.21 / 150), 0.3 + stats.norm.ppf(0.95) * math.sqrt(0.21 / 150)], 1e-9),
    ('tamano de muestra σ=5, E=1, 95%', math.ceil((stats.norm.ppf(0.975) * 5) ** 2), 0),
    ('prueba z x̄=52, μ0=50, σ=5, n=36, α=0.05', 2.4, 1e-12),
    ('prueba z x̄=52, μ0=50, σ=5, n=36, α=0.05', 2 * stats.norm.sf(2.4), 1e-8, 'p'),
    ('prueba t x̄=52, μ0=50, s=5, n=16, α=0.05, cola derecha', stats.t.sf(1.6, 15), 1e-8, 'p'),
    ('prueba t x̄=48, μ0=50, s=4, n=25, cola izquierda', stats.t.cdf(-2.5, 24), 1e-8, 'p'),
    ('prueba proporcion p=0.45, p0=0.4, n=100', 2 * stats.norm.sf(0.05 / math.sqrt(0.24 / 100)), 1e-8, 'p'),
    ('chi2 bondad observados [10,20,30], esperados [20,20,20]', 10, 1e-12),
    ('chi2 bondad observados [10,20,30], esperados [20,20,20]', stats.chi2.sf(10, 2), 1e-8, 'p'),
    ('chi2 bondad [18,22,30,30], probabilidades [0.25,0.25,0.25,0.25]', chi_b.statistic, 1e-12),
    ('chi2 bondad [18,22,30,30]', chi_b.pvalue, 1e-8, 'p'),
    ('chi2 independencia [[20,30],[30,20]]', tabla_contingencia([[20, 30], [30, 20]]), 1e-12),
    ('correlacion (1,2.1),(2,3.9),(3,6.2),(4,7.8),(5,10.1)', r_pear, 1e-10),
    ('correlacion (1,2.1),(2,3.9),(3,6.2),(4,7.8),(5,10.1)', p_pear, 1e-6, 'p'),
    # --- Cálculo numérico
    ('regula falsi x^2-2, 1, 2', regula_np(lambda q: q * q - 2, 1, 2), 1e-9),
    ('regula falsi cos(x)-x, 0, 1', regula_np(lambda q: math.cos(q) - q, 0, 1), 1e-9),
    ('muller x^3-2x-5, 0, 1, 2', muller_np(), 1e-9),
    ('muller x^3-2x-5, 2', muller_np(), 1e-9),
    ('jacobi [[4,1,1],[1,5,2],[1,2,6]], [6,8,9], 1', jacobi_np([[4, 1, 1], [1, 5, 2], [1, 2, 6]], [6, 8, 9], 1), 1e-9),
    ('jacobi [[4,1,1],[1,5,2],[1,2,6]], [6,8,9], 2', jacobi_np([[4, 1, 1], [1, 5, 2], [1, 2, 6]], [6, 8, 9], 2), 1e-9),
    ('jacobi [[10,-1,2],[-1,11,-1],[2,-1,10]], [6,25,-11]', list(np.linalg.solve([[10, -1, 2], [-1, 11, -1], [2, -1, 10]], [6, 25, -11])), 1e-8),
    ('gauss seidel [[4,1,1],[1,5,2],[1,2,6]], [6,8,9], 1', jacobi_np([[4, 1, 1], [1, 5, 2], [1, 2, 6]], [6, 8, 9], 1, True), 1e-9),
    ('gauss seidel [[4,1,1],[1,5,2],[1,2,6]], [6,8,9], 2', jacobi_np([[4, 1, 1], [1, 5, 2], [1, 2, 6]], [6, 8, 9], 2, True), 1e-9),
    ('gauss seidel [[4,1,1],[1,5,2],[1,2,6]], [6,8,9]', [1, 1, 1], 1e-8),
    ('lu [[4,3],[6,3]]', [[[1, 0], [1.5, 1]], [[4, 3], [0, -1.5]]], 1e-12),
    ('lu [[2,1,1],[4,3,3],[8,7,9]], [1,2,3]', list(np.linalg.solve([[2, 1, 1], [4, 3, 3], [8, 7, 9]], [1, 2, 3])), 1e-12),
    ('spline (0,1),(1,3),(2,2),(3,5); 1.5', float(cs(1.5)), 1e-12),
    ('spline (1,2),(2,1),(4,3),(5,0),(7,4); 3', float(cs2(3)), 1e-12),
    ("hermite p(0)=1, p'(0)=0, p(1)=2, p'(1)=1", 'expr:-x**3+2*x**2+1', 0),
    ("hermite (0,0,1),(1,1,0),(2,0,-1)", 'eval:1.5:' + repr(float(interpolate.KroghInterpolator([0, 0, 1, 1, 2, 2], [0, 1, 1, 0, 0, -1])(1.5))), 1e-10),
    ('minimos cuadrados grado 2 (0,1),(1,2),(2,5),(3,10)', 'expr:x**2+1', 0),
    ('minimos cuadrados grado 2 (0,1),(1,1.8),(2,3.3),(3,4.5),(4,6.3)', 'eval:2.5:' + repr(float(np.polyval(cq, 2.5))), 1e-10),
    ('ajuste exponencial (0,1.5),(1,2.5),(2,3.5),(3,5)', 'eval:1.7:' + repr(float(math.exp(ae) * math.exp(be * 1.7))), 1e-8),
    ('simpson 3/8 x^3, 0, 3, 3', 20.25, 1e-12),
    ('simpson 3/8 sen(x), 0, pi, 6', simpson38_np(math.sin, 0, math.pi, 6), 1e-12),
    ('romberg e^(x^2), 0, 1, 3', romberg_np(lambda q: np.exp(q ** 2), 0, 1, 3), 1e-12),
    ('romberg 1/x, 1, 2, 4', romberg_np(lambda q: 1 / q, 1, 2, 4), 1e-12),
    ('gauss legendre e^x, 0, 1, 3', gl_np(np.exp, 0, 1, 3), 1e-12),
    ('gauss legendre 1/(1+x^2), -1, 1, 5', gl_np(lambda q: 1 / (1 + q ** 2), -1, 1, 5), 1e-12),
    ('derivada numerica sen(x), 1, 0.1', (math.sin(1.1) - math.sin(0.9)) / 0.2, 1e-12),
    ('derivada numerica sen(x), 1, 0.1', (math.sin(1.1) - math.sin(1)) / 0.1, 1e-10, 'progresiva'),
    ('derivada numerica sen(x), 1, 0.1', (math.sin(1) - math.sin(0.9)) / 0.1, 1e-10, 'regresiva'),
    ("adams bashforth y'=x+y, 0, 1, 0.1, 1", ab4(lambda a, b: a + b, 0, 1, 0.1, 1), 1e-12),
    ("rk4 sistema x'=y, y'=-x, 0, (1,0), 0.1, 1", rk4_sis(lambda t_, Y: np.array([Y[1], -Y[0]]), 0, [1, 0], 0.1, 1), 1e-12),
    ("rk4 sistema x'=x-x*y, y'=-y+x*y, 0, (2,1), 0.05, 1",
     rk4_sis(lambda t_, Y: np.array([Y[0] - Y[0] * Y[1], -Y[1] + Y[0] * Y[1]]), 0, [2, 1], 0.05, 1), 1e-10),
    ('error 3.14, pi', abs(math.pi - 3.14), 1e-12),
    ('error 3.14, pi', abs(math.pi - 3.14) / math.pi, 1e-10, 'relativo'),
    ('propagacion de errores x*y^2, x=2±0.1, y=3±0.2', prop_err, 1e-12),
    # --- Electrotecnia
    ('potencia compleja V=220@0, I=10@-30', S_pc, 1e-10),
    ('potencia compleja V=220@0, Z=10+5j', 220 * (220 / (10 + 5j)).conjugate(), 1e-12),
    ('correccion fp P=10000, fp1=0.7, fp2=0.95, V=220, f=50', C_fp, 1e-10),
    ('correccion fp P=10000, fp1=0.7, fp2=0.95, V=220, f=50', Qc_fp, 1e-10, 'Qc'),
    ('trifasico V=380, I=10, fp=0.8', math.sqrt(3) * 380 * 10 * 0.8, 1e-12),
    ('trifasico V=380, I=10, fp=0.8', math.sqrt(3) * 380 * 10 * 0.6, 1e-12, 'Q'),
    # --- Investigación operativa
    ('programacion lineal max 3x+5y; x<=4; 2y<=12; 3x+2y<=18', -pl.fun, 1e-9, 'Z'),
    ('programacion lineal min 2x+3y+z; x+y+z>=10; x-y>=2; y+2z<=12', pl3.fun, 1e-9, 'Z'),
    ('programacion lineal min x+y; x+2y>=4; 3x+y>=6', pl_min.fun, 1e-9, 'Z'),
    ('asignacion [[9,2,7],[6,4,3],[5,8,1]]', hungaro([[9, 2, 7], [6, 4, 3], [5, 8, 1]]), 0),
    ('asignacion [[10,19,8,15],[10,18,7,17],[13,16,9,14],[12,19,8,18]]', hungaro([[10, 19, 8, 15], [10, 18, 7, 17], [13, 16, 9, 14], [12, 19, 8, 18]]), 0),
    ('asignacion [[3,8,2,10,3],[8,7,2,9,7],[6,4,2,7,5],[8,4,2,3,5],[9,10,6,9,10]]',
     hungaro([[3, 8, 2, 10, 3], [8, 7, 2, 9, 7], [6, 4, 2, 7, 5], [8, 4, 2, 3, 5], [9, 10, 6, 9, 10]]), 0),
    ('programacion lineal max x+y; x+y>=1', 'txt:no acotado', 0),
    ('programacion lineal max x+y; x<=1; x>=2', 'txt:infactible', 0),
    ('cola mm1 λ=2, μ=3', 2, 1e-12),
    ('cola mm1 λ=2, μ=3', 4 / 3, 1e-10, 'Lq'),
    ('cola mm1 λ=2, μ=3', 2 / 3, 1e-10, 'Wq'),
    ('cola mmc λ=2, μ=3, c=2', mmc(2, 3, 2), 1e-12),
    ('cola mmc λ=5, μ=2, c=3', mmc(5, 2, 3), 1e-12),
]

# Respuestas del alumno para el modo «¿Está bien mi respuesta?» (entrada, respuesta, correcto esperado)
COMPROBACIONES = [
    ('cuantil normal 0.975', '1.96', True), ('geometrica 0.2, 3', '0.128', True), ('geometrica 0.2, 3', '0.2', False),
    ('intervalo media x̄=50, σ=5, n=36, 95%', '[48.367, 51.633]', True), ('intervalo media x̄=50, σ=5, n=36, 95%', '[45, 55]', False),
    ('prueba z x̄=52, μ0=50, σ=5, n=36, α=0.05', '2.4', True), ('lu [[4,3],[6,3]]', '[[4,3],[0,-3/2]]', True),
    ("hermite p(0)=1, p'(0)=0, p(1)=2, p'(1)=1", '-x^3+2x^2+1', True), ('spline (0,1),(1,3),(2,2),(3,5); 1.5', '2.425', True),
    ('programacion lineal max 3x+5y; x<=4; 2y<=12; 3x+2y<=18', 'x=2, y=6', True),
    ('programacion lineal max 3x+5y; x<=4; 2y<=12; 3x+2y<=18', 'x=4, y=3', False),
    ('asignacion [[9,2,7],[6,4,3],[5,8,1]]', '9', True), ('cola mm1 λ=2, μ=3', '2', True),
    ('potencia compleja V=220@0, I=10@-30', '1905.26+1100j', True), ('correccion fp P=10000, fp1=0.7, fp2=0.95, V=220, f=50', '454.8', True),
    ("rk4 sistema x'=y, y'=-x, 0, (1,0), 0.1, 1", '(0.540303, -0.841470)', True), ('error 3.14, pi', '0.0015927', True),
    ('jacobi [[4,1,1],[1,5,2],[1,2,6]], [6,8,9], 2', '(0.725, 0.7, 0.71667)', True),
]

# Entradas inválidas: deben dar ok=False con un mensaje (no un error interno)
ERRORES = [
    'cuantil normal 0', 'cuantil normal 120%', 'cuantil t 0.9', 'geometrica 0.2', 'hipergeometrica 10, 20, 4, 2', 'uniforme 5, 2, 1, 3',
    'bayes P(A)=0.5, P(A2)=0.7, P(B|A)=0.1, P(B|A2)=0.2', 'intervalo media x̄=50, n=36', 'prueba z x̄=52, σ=5, n=36',
    'regula falsi x^2+1, 0, 1', 'jacobi [[0,1],[1,0]], [1,1]', 'jacobi [[1,5],[7,1]], [1,1]', 'simpson 3/8 x^2, 0, 1, 4',
    'minimos cuadrados grado 3 (0,1),(1,2)', 'cola mm1 λ=3, μ=2', 'correccion fp P=1000, fp1=0.9, fp2=0.8',
    'spline (0,1)', 'asignacion 3', 'chi2 bondad',
]


def comparar(d, esperado, tol, clave):
    if clave:
        v = d['valores'].get(clave)
        return cerca(num(v), complex(esperado), tol), v
    v = d.get('valor')
    if isinstance(esperado, str) and esperado.startswith('txt:'):
        return v == esperado[4:], v
    if isinstance(esperado, str) and esperado.startswith('expr:'):
        return sp.simplify(sp.sympify(v) - sp.sympify(esperado[5:])) == 0, v
    if isinstance(esperado, str) and esperado.startswith('eval:'):
        _, xv, ev = esperado.split(':')
        val = complex(sp.N(sp.sympify(v).subs(sp.Symbol('x'), float(xv))))
        return cerca(val, float(ev), tol), v
    if isinstance(esperado, (list, tuple)):
        if esperado and isinstance(esperado[0], (list, tuple)):  # lista de matrices
            mats = sp.sympify(v)
            ok = all(cerca(complex(sp.N(a)), float(b), max(tol, 1e-12)) for M, E in zip(mats, esperado)
                     for fa, fe in zip(M, E) for a, b in zip(fa, fe))
            return ok, v
        vals = lista(v)
        return len(vals) == len(esperado) and all(cerca(a, complex(b), max(tol, 1e-12)) for a, b in zip(vals, esperado)), v
    return cerca(num(v), complex(esperado), max(tol, 1e-12)), v


def main():
    bien = mal = 0
    t0 = time.time()
    print('== Resultados ==')
    for caso in CASOS:
        entrada, esperado, tol = caso[:3]
        clave = caso[3] if len(caso) > 3 else None
        d = resolver(entrada)
        if not d.get('ok'):
            mal += 1
            print('MAL', entrada, '-> ERROR', d.get('error'), str(d.get('detalle', ''))[-300:])
            continue
        try:
            ok, v = comparar(d, esperado, tol, clave)
        except Exception as e:  # noqa
            ok, v = False, f'(no comparable: {e})'
        if ok:
            bien += 1
            print(f"OK  {d['tiempo']:5.2f}s", entrada, ('[' + clave + ']') if clave else '', '=', v)
        else:
            mal += 1
            print('MAL', entrada, ('[' + clave + ']') if clave else '', '=', v, ' esperado', esperado)
    print('\n== Comprobación de respuestas ==')
    for entrada, resp, correcto in COMPROBACIONES:
        d = resolver(entrada, resp)
        c = (d.get('comprobacion') or {}).get('correcto')
        if c is correcto:
            bien += 1
            print('OK ', entrada, '| respuesta', resp, '->', c)
        else:
            mal += 1
            print('MAL', entrada, '| respuesta', resp, '->', c, d.get('comprobacion'))
    print('\n== Entradas inválidas (deben dar un mensaje claro) ==')
    for entrada in ERRORES:
        d = resolver(entrada)
        interno = (not d.get('ok')) and ('interno' in (d.get('error') or '') or 'revisá la escritura (' in (d.get('error') or ''))
        if not d.get('ok') and not interno:
            bien += 1
            print('OK ', entrada, '->', d.get('error'))
        else:
            mal += 1
            print('MAL', entrada, '->', d.get('ok'), d.get('error'), str(d.get('detalle', ''))[-300:])
    total = bien + mal
    print(f'\nRESUMEN: {bien}/{total} correctos, {mal} mal ({round(time.time() - t0, 1)} s)')
    return mal


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
