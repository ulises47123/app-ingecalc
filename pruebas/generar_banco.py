# -*- coding: utf-8 -*-
"""
Genera pruebas/banco_500.json: 500 ejercicios de matemática de ingeniería con la
sintaxis de SINTAXIS.md, y VERIFICA cada respuesta esperada de forma independiente
con SymPy/mpmath (no usa el motor IngeCalc).

Uso:  python generar_banco.py
Si algún esperado no se verifica, el script termina con código 1 y lista los fallos.
"""
import cmath
import json
import os
import re as rx
import sys
import unicodedata

import mpmath as mp
from sympy import *  # noqa: F401,F403
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                        implicit_multiplication_application,
                                        convert_xor)
from sympy.polys.polyfuncs import interpolate

mp.mp.dps = 50

x, y, z, t, s, n, k = symbols('x y z t s n k')
C1, C2, C3 = symbols('C1 C2 C3')
SYMS = {str(v): v for v in (x, y, z, t, s, n, k, C1, C2, C3)}
yf = Function('y')
D1, D2, D3, Y0 = symbols('D1 D2 D3 Y0')
TRANS = standard_transformations + (implicit_multiplication_application, convert_xor)


def _log(a, b=None):
    """log(x) de la entrada = base 10; log(x, b) = base b."""
    return log(a, 10) if b is None else log(a, b)


LOCAL = dict(SYMS)
LOCAL.update({
    'sen': sin, 'sin': sin, 'cos': cos, 'tg': tan, 'tan': tan, 'cot': cot,
    'sec': sec, 'csc': csc, 'arcsen': asin, 'asin': asin, 'arccos': acos,
    'acos': acos, 'arctg': atan, 'atan': atan, 'senh': sinh, 'sinh': sinh,
    'cosh': cosh, 'tgh': tanh, 'tanh': tanh, 'ln': log, 'log': _log,
    'exp': exp, 'sqrt': sqrt, 'raiz': sqrt, 'cbrt': cbrt, 'root': root,
    'abs': Abs, 'e': E, 'pi': pi, 'oo': oo, 'inf': oo, 'infinito': oo,
    'factorial': factorial, 'binomial': binomial, 'u': Heaviside,
    'heaviside': Heaviside, 'delta': DiracDelta, 'i': I, 'j': I,
    'Matrix': Matrix, 'D1': D1, 'D2': D2, 'D3': D3, 'Y0': Y0,
})


# ----------------------------------------------------------------------------
# Parser de la sintaxis amigable (solo para verificar; independiente del motor)
# ----------------------------------------------------------------------------
def norm(txt):
    txt = unicodedata.normalize('NFD', txt.lower())
    txt = ''.join(c for c in txt if unicodedata.category(c) != 'Mn')
    return unicodedata.normalize('NFC', txt)


def P(txt):
    txt = txt.strip().replace('π', 'pi').replace('√(', 'sqrt(').replace('∞', 'oo')
    return parse_expr(txt, local_dict=dict(LOCAL), transformations=TRANS)


def wrapmat(txt):
    out, i = [], 0
    while i < len(txt):
        if txt[i] == '[':
            depth, j = 0, i
            while True:
                if txt[j] == '[':
                    depth += 1
                elif txt[j] == ']':
                    depth -= 1
                if depth == 0:
                    break
                j += 1
            out.append('Matrix(' + txt[i:j + 1] + ')')
            i = j + 1
        else:
            out.append(txt[i])
            i += 1
    return ''.join(out)


def split_top(txt, sep=','):
    parts, depth, cur = [], 0, ''
    for ch in txt:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == sep and depth == 0:
            parts.append(cur.strip())
            cur = ''
        else:
            cur += ch
    parts.append(cur.strip())
    return parts


def mainvar(e):
    fs = getattr(e, 'free_symbols', set())
    if x in fs:
        return x
    if len(fs) == 1:
        return next(iter(fs))
    for v in (t, s):
        if v in fs:
            return v
    return x


def eqn(txt):
    if '=' in txt:
        l, r = txt.split('=', 1)
        return P(l) - P(r)
    return P(txt)


def rat(txt):
    return nsimplify(P(txt), rational=True)


def mpf_of(txt):
    return mp.mpf(str(N(P(txt), 50)))


def fnum(f, vs):
    return lambdify(vs, f, modules='mpmath')


# ----------------------------------------------------------------------------
# Cálculo de referencia por comando (devuelve dict R)
# ----------------------------------------------------------------------------
def h_bare(a):
    e = P(wrapmat(a)) if '[' in a else P(a)
    if isinstance(e, MatrixBase):
        return {'valor': e}
    return {'valor': simplify(e)}


def h_evaluar(a):
    e = P(a)
    return {'valor': e, 'valores': {'decimal': N(e)}}


def h_expandir(a):
    return {'valor': expand(P(a)), 'form': 'expand'}


def h_factorizar(a):
    return {'valor': factor(P(a)), 'form': 'factor'}


def h_fracciones(a):
    f = P(a)
    return {'valor': apart(f, mainvar(f)), 'form': 'apart'}


def h_resolver(a):
    parts = split_top(a, ';')
    if len(parts) > 1:
        eqs = [eqn(p) for p in parts]
        vs = sorted(set().union(*[e.free_symbols for e in eqs]), key=str)
        sol = solve(eqs, vs, dict=True)
        if len(sol) != 1:
            raise ValueError('sistema sin solución única: %s' % sol)
        return {'valor': sol[0]}
    if '<' in a or '>' in a:
        rel = P(a)
        return {'valor': solveset(rel, mainvar(rel), S.Reals)}
    e = eqn(a)
    v = mainvar(e)
    sols = solve(e, v)
    return {'valor': sols, 'check_sol': (e, v)}


def h_mcd(a):
    return {'valor': reduce_int(igcd, a)}


def h_mcm(a):
    return {'valor': reduce_int(ilcm, a)}


def reduce_int(fn, a):
    vals = [int(P(p)) for p in split_top(a)]
    r = vals[0]
    for v in vals[1:]:
        r = fn(r, v)
    return Integer(r)


def h_sumatoria(a):
    f, kv, lo, hi = [P(p) for p in split_top(a)]
    return {'valor': simplify(summation(f, (kv, lo, hi)))}


def h_derivar(a):
    args = split_top(a)
    f = P(args[0])
    v = P(args[1]) if len(args) > 1 else mainvar(f)
    o = int(P(args[2])) if len(args) > 2 else 1
    return {'valor': diff(f, v, o)}


def h_implicita(a):
    l, r = a.split('=', 1)
    F = P(l) - P(r)
    return {'valor': simplify(-diff(F, x) / diff(F, y))}


def h_integrar(a):
    m = rx.match(r'^(.*)\s+de\s+(.+)\s+a\s+(.+)$', a)
    if m:
        f = P(m.group(1))
        v = mainvar(f)
        return {'valor': integrate(f, (v, P(m.group(2)), P(m.group(3))))}
    args = split_top(a)
    f = P(args[0])
    v = P(args[1]) if len(args) >= 2 else mainvar(f)
    if len(args) >= 4:
        return {'valor': integrate(f, (v, P(args[2]), P(args[3])))}

    def check(esp):
        return eq_expr(diff(esp, v), f)
    return {'check': check}


def h_limite(a):
    args = split_top(a)
    f = P(args[0])
    vtxt, pt = args[-1].split('->')
    v = P(vtxt)
    pt = pt.strip()
    d = '+-'
    if len(pt) > 1 and pt.endswith('+'):
        d, pt = '+', pt[:-1]
    elif len(pt) > 1 and pt.endswith('-'):
        d, pt = '-', pt[:-1]
    p = P(pt)
    if p in (oo, -oo):
        return {'valor': limit(f, v, p)}
    return {'valor': limit(f, v, p, d)}


def h_taylor(a):
    f, v, a0, nn = split_top(a)
    f, v, a0, nn = P(f), P(v), P(a0), int(P(nn))
    return {'valor': series(f, v, a0, nn + 1).removeO()}


def h_recta(a):
    ftxt, atxt = split_top(a)
    f = P(ftxt)
    v = mainvar(f)
    a0 = P(atxt)
    return {'valor': expand(f.subs(v, a0) + diff(f, v).subs(v, a0) * (v - a0))}


def h_extremos(a):
    f = P(a)
    vs = [v for v in (x, y, z) if v in f.free_symbols]
    if len(vs) == 1:
        pts = [r for r in solve(diff(f, vs[0]), vs[0]) if r.is_real]
    else:
        sols = solve([diff(f, v) for v in vs], vs, dict=True)
        pts = [tuple(sl[v] for v in vs) for sl in sols
               if all(v in sl and sl[v].is_real for v in vs)]
    return {'valor': pts}


def h_area(a):
    ft, gt, at, bt = split_top(a)
    f, g, A, B = P(ft), P(gt), P(at), P(bt)
    h = f - g
    v = mainvar(h)
    roots = [r for r in solve(h, v) if r.is_real and A < r < B]
    pts = [A] + sorted(set(roots), key=lambda q: float(q)) + [B]
    tot = sum(Abs(integrate(h, (v, p, q))) for p, q in zip(pts[:-1], pts[1:]))
    return {'valor': simplify(tot)}


def h_gradiente(a):
    f = P(a)
    vs = [v for v in (x, y, z) if v in f.free_symbols]
    return {'valor': [diff(f, v) for v in vs]}


def h_divergencia(a):
    F = P(a)
    return {'valor': sum(diff(c, v) for c, v in zip(F, (x, y, z)))}


def h_rotacional(a):
    Pp, Q, R = P(a)
    return {'valor': [diff(R, y) - diff(Q, z), diff(Pp, z) - diff(R, x),
                      diff(Q, x) - diff(Pp, y)]}


def h_laplaciano(a):
    f = P(a)
    return {'valor': simplify(sum(diff(f, v, 2) for v in (x, y, z)))}


def h_jacobiano(a):
    F = P(a)
    fs = set().union(*[sympify(c).free_symbols for c in F])
    vs = [v for v in (x, y, z) if v in fs]
    return {'valor': Matrix([[diff(c, v) for v in vs] for c in F])}


def h_hessiano(a):
    f = P(a)
    vs = [v for v in (x, y, z) if v in f.free_symbols]
    return {'valor': hessian(f, vs)}


def h_intmult(a, nv):
    args = split_top(a)
    f = P(args[0])
    lims = [(P(args[1 + 3 * i]), P(args[2 + 3 * i]), P(args[3 + 3 * i])) for i in range(nv)]
    # Integración numérica anidada (la primera variable es la más interna).
    # Es independiente de integrate() de SymPy y evita casos muy lentos.
    vs = [lim[0] for lim in lims]
    F = fnum(f, vs)
    los = [fnum(lim[1], vs) for lim in lims]
    his = [fnum(lim[2], vs) for lim in lims]

    def rec(level, vals):
        def g(u):
            v2 = dict(vals)
            v2[level] = u
            if level == 0:
                return F(*[v2[i] for i in range(nv)])
            return rec(level - 1, v2)
        argv = [vals.get(i, 0) for i in range(nv)]
        return mp.quad(g, [los[level](*argv), his[level](*argv)])

    with mp.workdps(25):
        val = rec(nv - 1, {})
    return {'valor': Float(mp.nstr(val, 25), 25)}


def h_lagrange(a):
    fp, gp = split_top(a, ';')
    f = P(fp)
    g = eqn(gp)
    vs = [v for v in (x, y, z) if v in (f.free_symbols | g.free_symbols)]
    lam = Symbol('lam')
    eqs = [diff(f, v) - lam * diff(g, v) for v in vs] + [g]
    pts = []
    for sl in solve(eqs, vs + [lam], dict=True):
        if all(v in sl and sl[v].is_real for v in vs):
            p = tuple(sl[v] for v in vs)
            if p not in pts:
                pts.append(p)
    vals = [f.subs(dict(zip(vs, p))) for p in pts]
    return {'valor': pts, 'valores': {'maximo': max(vals, key=float),
                                      'minimo': min(vals, key=float)}}


def vecs(a):
    return [Matrix(P(p)) for p in split_top(a)]


def h_punto(a):
    u, v = vecs(a)
    return {'valor': u.dot(v)}


def h_cruz(a):
    u, v = vecs(a)
    return {'valor': list(u.cross(v))}


def h_norma(a):
    (u,) = vecs(a)
    return {'valor': simplify(u.norm())}


def h_angulo(a):
    u, v = vecs(a)
    ang = acos(simplify(u.dot(v) / (u.norm() * v.norm())))
    return {'valor': simplify(ang), 'valores': {'grados': ang * 180 / pi}}


def h_proyeccion(a):
    u, v = vecs(a)
    return {'valor': list(simplify(u.dot(v) / v.dot(v)) * v)}


def mat(a):
    return Matrix(P(a))


def h_autovalores(a):
    out = []
    for val, mult in mat(a).eigenvals().items():
        out += [val] * mult
    return {'valor': out}


def h_gauss(a):
    return {'valor': mat(a).rref()[0]}


# --------------------------- EDO --------------------------------------------
def ode_txt(txt):
    txt = txt.replace("y'''", " D3 ").replace("y''", " D2 ").replace("y'", " D1 ")
    return rx.sub(r"(?<![A-Za-z_])y(?![A-Za-z_0-9])", " Y0 ", txt)


def h_edo(a):
    parts = split_top(a, ';')
    eqt = parts[0]
    v = t if 't' in rx.findall(r'[a-z]+', eqt) else x
    l, r = eqt.split('=', 1) if '=' in eqt else (eqt, '0')
    rep = {Y0: yf(v), D1: yf(v).diff(v), D2: yf(v).diff(v, 2), D3: yf(v).diff(v, 3)}
    L = P(ode_txt(l)).xreplace(rep)
    R = P(ode_txt(r)).xreplace(rep)
    order = max([dd.derivative_count for dd in (L - R).atoms(Derivative)] + [0])
    ics = []
    for p in parts[1:]:
        m = rx.match(r"^y('*)\((.+)\)\s*=\s*(.+)$", p.strip())
        ics.append((len(m.group(1)), P(m.group(2)), P(m.group(3))))

    def residual_ok(sol):
        res = (L - R).subs(yf(v), sol).doit()
        return eq_expr(res, 0)

    def check(sol):
        if not residual_ok(sol):
            return False
        return all(eq_expr(diff(sol, v, o).subs(v, p0), val) for o, p0, val in ics)

    def check_general(sol):
        consts = [c for c in (C1, C2, C3) if c in sol.free_symbols]
        if len(consts) != order or ics:
            return False
        if not residual_ok(sol):
            return False
        W = Matrix([[diff(diff(sol, v, i), c) for c in consts] for i in range(order)])
        sub = {v: Rational(37, 100), C1: Rational(1, 2), C2: Rational(2, 3), C3: Rational(3, 4)}
        return abs(complex(N(W.det().subs(sub)))) > 1e-12

    return {'check': check, 'check_general': check_general,
            'edo': str(Eq(L, R)), 'orden': order}


def h_laplace(a):
    return {'valor': simplify(laplace_transform(P(a), t, s, noconds=True))}


def h_laplace_inv(a):
    F = P(a)

    def check(esp):
        return eq_expr(laplace_transform(esp, t, s, noconds=True), F)
    return {'check': check}


def h_fourier(a):
    ft, vt, at, bt, nt = split_top(a)
    f, v, A, B, nn = P(ft), P(vt), P(at), P(bt), int(P(nt))
    L = (B - A) / 2
    xr = Symbol('xr', real=True)
    fr = f.subs(v, xr)
    total = integrate(fr, (xr, A, B)) / L / 2
    for kk in range(1, nn + 1):
        ak = integrate(fr * cos(kk * pi * xr / L), (xr, A, B)) / L
        bk = integrate(fr * sin(kk * pi * xr / L), (xr, A, B)) / L
        total += ak * cos(kk * pi * v / L) + bk * sin(kk * pi * v / L)
    return {'valor': simplify(total)}


# --------------------------- Métodos numéricos -------------------------------
def root_result(f, v, r):
    g = fnum(f, v)
    return {'valor': Float(mp.nstr(r, 40), 40),
            'sanity': (abs(g(r)) < mp.mpf(10) ** -20, 'f(raiz) no es ~0')}


def h_newton(a):
    args = split_top(a)
    f = P(args[0])
    v = mainvar(f)
    g, dg = fnum(f, v), fnum(diff(f, v), v)
    xk = mpf_of(args[1])
    for _ in range(500):
        d = g(xk) / dg(xk)
        xk -= d
        if abs(d) < mp.mpf(10) ** -40:
            break
    return root_result(f, v, xk)


def h_biseccion(a):
    args = split_top(a)
    f = P(args[0])
    v = mainvar(f)
    g = fnum(f, v)
    lo, hi = mpf_of(args[1]), mpf_of(args[2])
    assert g(lo) * g(hi) < 0, 'sin cambio de signo'
    for _ in range(200):
        mid = (lo + hi) / 2
        if g(lo) * g(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return root_result(f, v, (lo + hi) / 2)


def h_secante(a):
    args = split_top(a)
    f = P(args[0])
    v = mainvar(f)
    g = fnum(f, v)
    x0, x1 = mpf_of(args[1]), mpf_of(args[2])
    for _ in range(500):
        f0, f1 = g(x0), g(x1)
        if f1 == f0:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, x1 = x1, x2
        if abs(x1 - x0) < mp.mpf(10) ** -40:
            break
    return root_result(f, v, x1)


def h_punto_fijo(a):
    args = split_top(a)
    gx = P(args[0])
    v = mainvar(gx)
    g = fnum(gx, v)
    xk = mpf_of(args[1])
    for _ in range(20000):
        xn = g(xk)
        if abs(xn - xk) < mp.mpf(10) ** -40:
            xk = xn
            break
        xk = xn
    return root_result(gx - v, v, xk)


def h_cuadratura(a, metodo):
    ft, at, bt, nt = split_top(a)
    f = P(ft)
    v = mainvar(f)
    g = fnum(f, v)
    A, B, nn = mpf_of(at), mpf_of(bt), int(P(nt))
    h = (B - A) / nn
    ys = [g(A + i * h) for i in range(nn + 1)]
    if metodo == 'trapecio':
        val = h / 2 * (ys[0] + 2 * sum(ys[1:-1]) + ys[-1])
    else:
        assert nn % 2 == 0
        val = h / 3 * (ys[0] + 4 * sum(ys[1:-1:2]) + 2 * sum(ys[2:-1:2]) + ys[-1])
    exacto = mp.quad(g, [A, B])
    return {'valor': Float(mp.nstr(val, 40), 40),
            'sanity': (abs(val - exacto) <= mp.mpf('0.05') * abs(exacto) + mp.mpf('1e-3'),
                       'resultado del método lejos de la integral')}


def h_ode_num(a, metodo):
    args = split_top(a)
    rhs = P(args[0].split('=', 1)[1])
    F = fnum(rhs, (x, y))
    x0, y0, h, xf = [mpf_of(p) for p in args[1:5]]
    steps = int(mp.nint((xf - x0) / h))
    X, Y = x0, y0
    for _ in range(steps):
        if metodo == 'euler':
            Y = Y + h * F(X, Y)
        else:
            k1 = F(X, Y)
            k2 = F(X + h / 2, Y + h * k1 / 2)
            k3 = F(X + h / 2, Y + h * k2 / 2)
            k4 = F(X + h, Y + h * k3)
            Y = Y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        X = X + h
    verdadero = mp.odefun(lambda X_, Y_: F(X_, Y_), x0, y0)(xf)
    tol = mp.mpf('0.2') if metodo == 'euler' else mp.mpf('1e-3')
    return {'valor': Float(mp.nstr(Y, 40), 40),
            'sanity': (abs(Y - verdadero) <= tol * abs(verdadero) + mp.mpf('1e-6'),
                       'resultado del método lejos de la solución verdadera')}


def h_interpolar(a):
    pts = [tuple(P(p)) for p in split_top(a)]
    return {'valor': expand(interpolate(pts, x))}


def h_regresion(a):
    pts = [tuple(nsimplify(c, rational=True) for c in P(p)) for p in split_top(a)]
    m = len(pts)
    xm = sum(p[0] for p in pts) / m
    ym = sum(p[1] for p in pts) / m
    sxy = sum((p[0] - xm) * (p[1] - ym) for p in pts)
    sxx = sum((p[0] - xm) ** 2 for p in pts)
    sst = sum((p[1] - ym) ** 2 for p in pts)
    aa = sxy / sxx
    bb = ym - aa * xm
    return {'valor': aa * x + bb, 'valores': {'r2': aa ** 2 * sxx / sst}}


# --------------------------- Complejos, circuitos, estadística ---------------
def h_polar(a):
    zc = P(a)
    return {'valor': [simplify(Abs(zc)), simplify(arg(zc))]}


def h_fasor(a):
    def rep(m):
        A, th = Rational(m.group(1)), Rational(m.group(2))
        return '((%s)*(cos(pi*(%s)/180)+i*sin(pi*(%s)/180)))' % (A, th, th)
    txt = rx.sub(r'(\d+(?:\.\d+)?)\s*@\s*(-?\d+(?:\.\d+)?)', rep, a)
    val = expand(P(txt))
    return {'valor': val, 'valores': {'modulo': Abs(val), 'angulo': arg(val) * 180 / pi}}


def h_paralelo(a):
    return {'valor': 1 / sum(1 / rat(p) for p in split_top(a))}


def h_serie(a):
    return {'valor': sum(rat(p) for p in split_top(a))}


def h_estadistica(a):
    d = sorted(rat(p) for p in split_top(a))
    m = len(d)
    media = sum(d) / m
    med = d[m // 2] if m % 2 else (d[m // 2 - 1] + d[m // 2]) / 2
    ss = sum((v - media) ** 2 for v in d)
    vals = {'media': media, 'mediana': med, 'varianza': ss / (m - 1),
            'desvio': sqrt(ss / (m - 1)), 'varianza_pob': ss / m,
            'desvio_pob': sqrt(ss / m), 'n': m}
    return {'valor': media, 'valores': vals}


def h_binomial(a):
    nn, p, kk = split_top(a)
    nn, p, kk = int(P(nn)), rat(p), int(P(kk))
    return {'valor': binomial(nn, kk) * p ** kk * (1 - p) ** (nn - kk)}


def h_poisson(a):
    lam, kk = split_top(a)
    lam, kk = rat(lam), int(P(kk))
    return {'valor': exp(-lam) * lam ** kk / factorial(kk)}


def h_normal(a):
    mu, sg, A, B = [rat(p) for p in split_top(a)]
    phi = lambda q: (1 + erf((q - mu) / (sg * sqrt(2)))) / 2  # noqa: E731
    return {'valor': phi(B) - phi(A)}


def h_graficar(a):
    if a.startswith('polar '):
        tipo, ex = 'polar', [P(a[6:])]
        need = {t}
    elif a.startswith('parametrica '):
        tipo, ex = 'parametrica', [P(p) for p in split_top(a[12:])]
        need = {t}
    elif a.startswith('3d '):
        tipo, ex = '3d', [P(a[3:])]
        need = {x, y}
    elif a.startswith('datos '):
        tipo, ex = 'datos', [P(p) for p in split_top(a[6:])]
        need = set()
    elif '=' in a:
        tipo, ex = 'implicita', [eqn(a)]
        need = {x, y}
    else:
        tipo, ex = '2d', [P(p) for p in split_top(a)]
        need = {x}
    fs = set().union(*[sympify(e).free_symbols for e in ex])
    ok = need <= fs if tipo != '2d' else fs <= {x}
    return {'grafico': tipo, 'sanity': (ok, 'variables inesperadas en el gráfico')}


HANDLERS = {
    'fracciones parciales': h_fracciones, 'derivar implicita': h_implicita,
    'recta tangente': h_recta, 'laplace inversa': h_laplace_inv,
    'integral doble': lambda a: h_intmult(a, 2),
    'integral triple': lambda a: h_intmult(a, 3),
    'punto fijo': h_punto_fijo, 'graficar': h_graficar,
    'simplificar': h_bare, 'evaluar': h_evaluar, 'expandir': h_expandir,
    'factorizar': h_factorizar, 'resolver': h_resolver, 'mcd': h_mcd,
    'mcm': h_mcm, 'sumatoria': h_sumatoria, 'derivar': h_derivar,
    'integrar': h_integrar, 'limite': h_limite, 'taylor': h_taylor,
    'extremos': h_extremos, 'area': h_area, 'gradiente': h_gradiente,
    'divergencia': h_divergencia, 'rotacional': h_rotacional,
    'laplaciano': h_laplaciano, 'jacobiano': h_jacobiano,
    'hessiano': h_hessiano, 'lagrange': h_lagrange, 'punto': h_punto,
    'cruz': h_cruz, 'norma': h_norma, 'angulo': h_angulo,
    'proyeccion': h_proyeccion,
    'det': lambda a: {'valor': mat(a).det()},
    'inversa': lambda a: {'valor': mat(a).inv().applyfunc(simplify)},
    'transpuesta': lambda a: {'valor': mat(a).T},
    'traza': lambda a: {'valor': mat(a).trace()},
    'rango': lambda a: {'valor': Integer(mat(a).rank())},
    'autovalores': h_autovalores, 'gauss': h_gauss, 'edo': h_edo,
    'laplace': h_laplace, 'fourier': h_fourier, 'newton': h_newton,
    'biseccion': h_biseccion, 'secante': h_secante,
    'trapecio': lambda a: h_cuadratura(a, 'trapecio'),
    'simpson': lambda a: h_cuadratura(a, 'simpson'),
    'euler': lambda a: h_ode_num(a, 'euler'),
    'rk4': lambda a: h_ode_num(a, 'rk4'),
    'interpolar': h_interpolar, 'regresion': h_regresion, 'polar': h_polar,
    'fasor': h_fasor, 'paralelo': h_paralelo, 'serie': h_serie,
    'estadistica': h_estadistica, 'binomial': h_binomial,
    'poisson': h_poisson, 'normal': h_normal,
}
COMMANDS = sorted(HANDLERS, key=len, reverse=True)


def compute(entrada):
    txt = norm(entrada).strip()
    for cmd in COMMANDS:
        if txt.startswith(cmd + ' '):
            return HANDLERS[cmd](txt[len(cmd) + 1:].strip())
    return h_bare(txt)


# ----------------------------------------------------------------------------
# Comparadores
# ----------------------------------------------------------------------------
PTS = [0.31, 0.57, 0.83, 0.44, 0.69, 0.23, 0.77]


def num_eq(a, b, tol=1e-8):
    fs = sorted(a.free_symbols | b.free_symbols, key=str)
    ok = 0
    for trial in range(6):
        sub = {v: PTS[(trial + 2 * i) % len(PTS)] + 0.05 * i for i, v in enumerate(fs)}
        try:
            va = complex(N(a.subs(sub), 30))
            vb = complex(N(b.subs(sub), 30))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if not (cmath.isfinite(va) and cmath.isfinite(vb)):
            continue
        if abs(va - vb) > tol * max(1.0, abs(vb)):
            return False
        ok += 1
    return ok >= 3


def eq_expr(a, b):
    if isinstance(a, MatrixBase) or isinstance(b, MatrixBase):
        A, B = Matrix(a), Matrix(b)
        return A.shape == B.shape and all(eq_expr(p, q) for p, q in zip(A, B))
    if isinstance(a, (list, tuple, Tuple)) or isinstance(b, (list, tuple, Tuple)):
        a, b = list(a), list(b)
        return len(a) == len(b) and all(eq_expr(p, q) for p, q in zip(a, b))
    a, b = sympify(a), sympify(b)
    if a == b:
        return True
    try:
        if simplify(a - b) == 0:
            return True
    except Exception:  # noqa: BLE001
        pass
    return num_eq(a, b)


def num_close(a, b, tol):
    va, vb = complex(N(sympify(a), 30)), complex(N(sympify(b), 30))
    if vb == 0:
        return abs(va) <= tol
    return abs(va - vb) <= tol * abs(vb)


def elem_eq(a, b):
    if isinstance(a, (list, tuple, Tuple)) or isinstance(b, (list, tuple, Tuple)):
        a, b = list(a), list(b)
        return len(a) == len(b) and all(elem_eq(p, q) for p, q in zip(a, b))
    return num_close(a, b, 1e-9)


def multiset_eq(c, e):
    c, e = list(c), list(e)
    if len(c) != len(e):
        return False
    used = set()
    for el in e:
        idx = next((i for i, cv in enumerate(c) if i not in used and elem_eq(cv, el)), None)
        if idx is None:
            return False
        used.add(idx)
    return True


def dict_eq(c, e):
    c = {str(kk): v for kk, v in c.items()}
    e = {str(kk): v for kk, v in e.items()}
    return c.keys() == e.keys() and all(num_close(c[kk], e[kk], 1e-9) for kk in e)


def sets_eq(a, b):
    if a == b:
        return True
    try:
        if Complement(a, b) == S.EmptySet and Complement(b, a) == S.EmptySet:
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def S_(txt):
    return sympify(txt, locals=dict(SYMS))


def fmt_num(v):
    c = complex(N(v, 30))
    re_s = '%.15g' % c.real
    if abs(c.imag) <= 1e-14 * max(1.0, abs(c)):
        return re_s
    return '%s %s %s*I' % (re_s, '-' if c.imag < 0 else '+', '%.15g' % abs(c.imag))


def verificar(caso):
    """Devuelve (ok, mensaje, extras). Completa caso['esperado'] si era None."""
    R = compute(caso['entrada'])
    modo = caso['modo']
    if 'sanity' in R and not R['sanity'][0]:
        return False, 'sanity: ' + R['sanity'][1]
    if modo == 'grafico':
        return R['grafico'] == caso['esperado'], 'tipo calculado: %s' % R['grafico']
    if caso['esperado'] is None:
        assert modo in ('numero', 'clave')
        caso['esperado'] = fmt_num(R['valores'][caso['clave']] if modo == 'clave' else R['valor'])
    esp = S_(caso['esperado'])
    if 'edo' in R:
        caso['edo'] = R['edo']
    if modo == 'expr':
        ok = R['check'](esp) if 'check' in R else eq_expr(R['valor'], esp)
    elif modo == 'antiderivada':
        ok = R['check'](esp)
    elif modo == 'edo_general':
        ok = R['check_general'](esp)
    elif modo == 'numero':
        ok = num_close(R['valor'], esp, caso['tol'])
    elif modo == 'conjunto':
        ok = multiset_eq(R['valor'], esp)
        if ok and 'check_sol' in R:  # cada raíz esperada satisface la ecuación
            e, v = R['check_sol']
            ok = all(abs(complex(N(e.subs(v, r), 30))) < 1e-12 for r in esp)
    elif modo == 'dict':
        ok = dict_eq(R['valor'], esp)
    elif modo in ('matriz', 'vector'):
        ok = eq_expr(R['valor'], esp)
    elif modo == 'intervalo':
        ok = sets_eq(R['valor'], esp)
    elif modo == 'clave':
        v = R['valores'][caso['clave']]
        ok = num_close(v, esp, caso['tol']) if 'tol' in caso else eq_expr(v, esp)
    else:
        return False, 'modo desconocido'
    if ok and R.get('form') == 'expand':
        ok = esp == expand(esp)
    if ok and R.get('form') == 'factor':
        ok = not esp.is_Add
    if ok and R.get('form') == 'apart':
        ok = esp.is_Add
    calc = R.get('valor', '(verificación por sustitución)')
    return ok, 'calculado: %s' % (calc,)


# ----------------------------------------------------------------------------
# Casos
# ----------------------------------------------------------------------------
ALG, ECU, LIM, DER = 'algebra', 'ecuaciones', 'limites', 'derivadas'
IND, DEF, TAY, EXT = ('integrales indefinidas', 'integrales definidas',
                      'taylor y sumatorias', 'extremos')
VV, VEC, LIN, EDO = 'varias variables', 'vectores', 'algebra lineal', 'edo'
LAP, FOU, NUM, CPX = 'laplace', 'fourier', 'metodos numericos', 'complejos y circuitos'
EST, GRA = 'estadistica', 'graficos'
TEMAS = [ALG, ECU, LIM, DER, IND, DEF, TAY, EXT, VV, VEC, LIN, EDO, LAP, FOU,
         NUM, CPX, EST, GRA]

CASES = []


def C(d, tema, entrada, modo, esperado, **kw):
    CASES.append(dict(d=d, tema=tema, entrada=entrada, modo=modo,
                      esperado=esperado, **kw))


# ---------------- Álgebra (35) ----------------
C(1, ALG, 'simplificar (x^2-9)/(x+3)', 'expr', 'x - 3')
C(1, ALG, 'expandir (x+2)^2', 'expr', 'x**2 + 4*x + 4')
C(1, ALG, 'mcd 84, 120', 'expr', '12')
C(1, ALG, 'mcm 4, 6, 10', 'expr', '60')
C(1, ALG, '(x^2+5x+6)/(x+2)', 'expr', 'x + 3')
C(2, ALG, 'expandir (2x-3)(x+4)', 'expr', '2*x**2 + 5*x - 12')
C(2, ALG, 'factorizar x^2-5x+6', 'expr', '(x - 2)*(x - 3)')
C(2, ALG, 'factorizar 2x^2+7x+3', 'expr', '(2*x + 1)*(x + 3)')
C(2, ALG, 'evaluar sen(pi/6)+cos(pi/3)', 'expr', '1')
C(2, ALG, 'evaluar log(1000)+ln(e^2)', 'expr', '5')
C(2, ALG, 'simplificar (x^2-4)/(x^2+4x+4)', 'expr', '(x - 2)/(x + 2)')
C(3, ALG, 'factorizar x^3-8', 'expr', '(x - 2)*(x**2 + 2*x + 4)')
C(3, ALG, 'factorizar x^4-16', 'expr', '(x - 2)*(x + 2)*(x**2 + 4)')
C(3, ALG, 'fracciones parciales 1/(x^2-1)', 'expr', '1/(2*(x - 1)) - 1/(2*(x + 1))')
C(3, ALG, 'fracciones parciales (5x-1)/(x^2-x-2)', 'expr', '3/(x - 2) + 2/(x + 1)')
C(3, ALG, 'simplificar sen(x)^2+cos(x)^2+tg(x)*cos(x)', 'expr', 'sin(x) + 1')
C(3, ALG, 'mcd 391, 299', 'expr', '23')
C(4, ALG, 'expandir (x+y)^3', 'expr', 'x**3 + 3*x**2*y + 3*x*y**2 + y**3')
C(4, ALG, 'factorizar x^3-2x^2-5x+6', 'expr', '(x - 1)*(x - 3)*(x + 2)')
C(4, ALG, 'fracciones parciales (x+3)/(x^2+x)', 'expr', '3/x - 2/(x + 1)')
C(4, ALG, 'simplificar (1/x-1/y)/(1/x+1/y)', 'expr', '(y - x)/(x + y)')
C(4, ALG, 'simplificar (e^(2x)-1)/(e^x+1)', 'expr', 'exp(x) - 1')
C(4, ALG, 'evaluar ln(sqrt(e))+log(1/100)', 'expr', '-3/2')
C(5, ALG, 'factorizar x^4-5x^2+4', 'expr', '(x - 2)*(x - 1)*(x + 1)*(x + 2)')
C(5, ALG, 'fracciones parciales (2x^2+3)/(x^3+x)', 'expr', '3/x - x/(x**2 + 1)')
C(5, ALG, 'fracciones parciales (x^2+1)/((x-1)^2*(x+2))', 'expr',
  '4/(9*(x - 1)) + 2/(3*(x - 1)**2) + 5/(9*(x + 2))')
C(5, ALG, 'simplificar sen(2x)/(2cos(x))', 'expr', 'sin(x)')
C(5, ALG, 'factorizar 6x^2-x*y-2y^2', 'expr', '(2*x + y)*(3*x - 2*y)')
C(5, ALG, 'expandir (x^2+x+1)(x^2-x+1)', 'expr', 'x**4 + x**2 + 1')
C(6, ALG, 'fracciones parciales (s+3)/(s^3+3s^2+2s)', 'expr', '3/(2*s) - 2/(s + 1) + 1/(2*(s + 2))')
C(6, ALG, 'simplificar (x^3-27)/(x^2-9)-(x^2+3x+9)/(x+3)', 'expr', '0')
C(6, ALG, 'factorizar x^5-x', 'expr', 'x*(x - 1)*(x + 1)*(x**2 + 1)')
C(6, ALG, 'simplificar (1-cos(2x))/sen(2x)', 'expr', 'tan(x)')
C(6, ALG, 'fracciones parciales (x^2+2x+3)/((x+1)(x^2+1))', 'expr', '1/(x + 1) + 2/(x**2 + 1)')
C(7, ALG, 'fracciones parciales (x^3+2)/(x^2-1)', 'expr', 'x + 3/(2*(x - 1)) - 1/(2*(x + 1))')

# ---------------- Ecuaciones, sistemas, inecuaciones (38) ----------------
C(1, ECU, 'resolver 2x+3=11', 'conjunto', '[4]')
C(1, ECU, 'resolver x^2-7x+12=0', 'conjunto', '[3, 4]')
C(1, ECU, 'resolver 3(x-2)=x+4', 'conjunto', '[5]')
C(2, ECU, 'resolver x+y=10; x-y=4', 'dict', '{x: 7, y: 3}')
C(2, ECU, 'resolver 2x+3y=12; x-y=1', 'dict', '{x: 3, y: 2}')
C(2, ECU, 'resolver x^2+2x-15', 'conjunto', '[-5, 3]')
C(2, ECU, 'resolver 2x^2-3x-2=0', 'conjunto', '[2, -1/2]')
C(2, ECU, 'resolver 3x-5<7', 'intervalo', 'Interval.open(-oo, 4)')
C(3, ECU, 'resolver x^3-6x^2+11x-6=0', 'conjunto', '[1, 2, 3]')
C(3, ECU, 'resolver x^2<9', 'intervalo', 'Interval.open(-3, 3)')
C(3, ECU, 'resolver x^2-x-6>=0', 'intervalo', 'Union(Interval(-oo, -2), Interval(3, oo))')
C(3, ECU, 'resolver e^x=5', 'conjunto', '[log(5)]')
C(3, ECU, 'resolver log(x)=2', 'conjunto', '[100]')
C(3, ECU, 'resolver x+y+z=6; x-y+z=2; 2x+y-z=1', 'dict', '{x: 1, y: 2, z: 3}')
C(4, ECU, 'resolver sqrt(x+3)=x-3', 'conjunto', '[6]')
C(4, ECU, 'resolver x^4-5x^2+4=0', 'conjunto', '[-2, -1, 1, 2]')
C(4, ECU, 'resolver abs(2x-1)<5', 'intervalo', 'Interval.open(-2, 3)')
C(4, ECU, 'resolver (x-1)/(x+2)>=0', 'intervalo', 'Union(Interval.open(-oo, -2), Interval(1, oo))')
C(4, ECU, 'resolver 2^(x+1)=32', 'conjunto', '[4]')
C(4, ECU, 'resolver x^2+2x+5=0', 'conjunto', '[-1 - 2*I, -1 + 2*I]')
C(4, ECU, 'resolver 3x-2y+z=2; 2x+y-z=1; x+3y+2z=13', 'dict', '{x: 1, y: 2, z: 3}')
C(4, ECU, 'resolver x^2-2x-8<=0', 'intervalo', 'Interval(-2, 4)')
C(4, ECU, 'resolver cbrt(x+7)=2', 'conjunto', '[1]')
C(5, ECU, 'resolver ln(x)+ln(x-1)=ln(6)', 'conjunto', '[3]')
C(5, ECU, 'resolver x^3-3x+2=0', 'conjunto', '[-2, 1]')
C(5, ECU, 'resolver e^(2x)-3e^x+2=0', 'conjunto', '[0, log(2)]')
C(5, ECU, 'resolver x^3-x>0', 'intervalo', 'Union(Interval.open(-1, 0), Interval.open(1, oo))')
C(5, ECU, 'resolver abs(x-3)>=2', 'intervalo', 'Union(Interval(-oo, 1), Interval(5, oo))')
C(5, ECU, 'resolver 0.5x+1.5y=4; 2x-y=2', 'dict', '{x: 2, y: 2}')
C(5, ECU, 'resolver log(x, 2)+log(x-2, 2)=3', 'conjunto', '[4]')
C(6, ECU, 'resolver (x^2-1)/(x-3)<0', 'intervalo', 'Union(Interval.open(-oo, -1), Interval.open(1, 3))')
C(6, ECU, 'resolver x+y+z+t=10; x-y=-1; y-z=-1; z-t=-1', 'dict', '{x: 1, y: 2, z: 3, t: 4}')
C(6, ECU, 'resolver x^4+x^2-12=0', 'conjunto', '[-sqrt(3), sqrt(3), -2*I, 2*I]')
C(6, ECU, 'resolver 9^x-4*3^x+3=0', 'conjunto', '[0, 1]')
C(6, ECU, 'resolver x/(x-1)+2/(x+1)=1', 'conjunto', '[1/3]')
C(6, ECU, 'resolver x^3-2x^2-x+2>0', 'intervalo', 'Union(Interval.open(-1, 1), Interval.open(2, oo))')
C(7, ECU, 'resolver 3x+2y-z=1; 2x-2y+4z=-2; -x+y/2-z=0', 'dict', '{x: 1, y: -2, z: -2}')
C(7, ECU, 'resolver x^5-x^4-x+1=0', 'conjunto', '[-1, 1, -I, I]')

# ---------------- Límites (35) ----------------
C(1, LIM, 'limite (x^2-4)/(x-2), x->2', 'expr', '4')
C(1, LIM, 'limite (2x+1)/(x-3), x->oo', 'expr', '2')
C(2, LIM, 'limite sen(3x)/x, x->0', 'expr', '3')
C(2, LIM, 'limite 1/x^2, x->0', 'expr', 'oo')
C(2, LIM, 'limite (3x^2-x)/(5x^2+2), x->oo', 'expr', '3/5')
C(2, LIM, 'limite (e^x-1)/x, x->0', 'expr', '1')
C(2, LIM, 'limite (x^3-27)/(x-3), x->3', 'expr', '27')
C(3, LIM, 'limite (sqrt(x)-2)/(x-4), x->4', 'expr', '1/4')
C(3, LIM, 'limite (1-cos(x))/x^2, x->0', 'expr', '1/2')
C(3, LIM, 'limite (1+1/x)^(2x), x->oo', 'expr', 'exp(2)')
C(3, LIM, 'limite 1/(x-2), x->2-', 'expr', '-oo')
C(3, LIM, 'limite ln(x)/(x-1), x->1', 'expr', '1')
C(3, LIM, 'limite (sqrt(x+9)-3)/x, x->0', 'expr', '1/6')
C(3, LIM, 'limite arctg(x), x->-oo', 'expr', '-pi/2')
C(4, LIM, 'limite x*ln(x), x->0+', 'expr', '0')
C(4, LIM, 'limite sqrt(x^2+1)-x, x->oo', 'expr', '0')
C(4, LIM, 'limite x^5*e^(-x), x->oo', 'expr', '0')
C(4, LIM, 'limite (1+3/x)^x, x->oo', 'expr', 'exp(3)')
C(4, LIM, 'limite (2^x-1)/x, x->0', 'expr', 'log(2)')
C(4, LIM, 'limite abs(x)/x, x->0-', 'expr', '-1')
C(4, LIM, 'limite (1-cos(2x))/(x*sen(x)), x->0', 'expr', '2')
C(5, LIM, 'limite (x-sen(x))/x^3, x->0', 'expr', '1/6')
C(5, LIM, 'limite x^x, x->0+', 'expr', '1')
C(5, LIM, 'limite sqrt(x^2+3x)-x, x->oo', 'expr', '3/2')
C(5, LIM, 'limite (e^x-1-x)/x^2, x->0', 'expr', '1/2')
C(5, LIM, 'limite x*sen(1/x), x->oo', 'expr', '1')
C(6, LIM, 'limite (1+sen(x))^(1/x), x->0', 'expr', 'E')
C(6, LIM, 'limite (tg(x)-sen(x))/x^3, x->0', 'expr', '1/2')
C(6, LIM, 'limite (x/(x+1))^x, x->oo', 'expr', 'exp(-1)')
C(6, LIM, 'limite (senh(x)-x)/x^3, x->0', 'expr', '1/6')
C(6, LIM, 'limite cot(x)-1/x, x->0', 'expr', '0')
C(7, LIM, 'limite cos(x)^(1/x^2), x->0', 'expr', 'exp(-1/2)')
C(7, LIM, 'limite 1/x-1/(e^x-1), x->0', 'expr', '1/2')
C(7, LIM, 'limite (arcsen(x)-x)/x^3, x->0', 'expr', '1/6')
C(8, LIM, 'limite ((1+x)^(1/x)-e)/x, x->0', 'expr', '-E/2')

# ---------------- Derivadas (43) ----------------
C(1, DER, 'derivar x^3', 'expr', '3*x**2')
C(1, DER, 'derivar 5x^2-3x+7', 'expr', '10*x - 3')
C(1, DER, 'derivar sen(x)', 'expr', 'cos(x)')
C(1, DER, 'derivar e^x+ln(x)', 'expr', 'exp(x) + 1/x')
C(2, DER, 'derivar x*e^x', 'expr', '(x + 1)*exp(x)')
C(2, DER, 'derivar cos(x)^3', 'expr', '-3*sin(x)*cos(x)**2')
C(2, DER, 'derivar tg(x)', 'expr', '1/cos(x)**2')
C(2, DER, 'derivar ln(x^2+1)', 'expr', '2*x/(x**2 + 1)')
C(2, DER, 'derivar x^4, x, 2', 'expr', '12*x**2')
C(3, DER, 'derivar x^2*sen(x)', 'expr', '2*x*sin(x) + x**2*cos(x)')
C(3, DER, 'derivar (x+1)/(x-1)', 'expr', '-2/(x - 1)**2')
C(3, DER, 'derivar log(x)', 'expr', '1/(x*log(10))')
C(3, DER, 'derivar arctg(2x)', 'expr', '2/(4*x**2 + 1)')
C(3, DER, 'derivar e^(-x^2)', 'expr', '-2*x*exp(-x**2)')
C(3, DER, 'derivar x^3*y^2+sen(y), y', 'expr', '2*x**3*y + cos(y)')
C(3, DER, 'derivar implicita x*y=6', 'expr', '-y/x')
C(3, DER, 'recta tangente x^3, 2', 'expr', '12*x - 16')
C(3, DER, 'recta tangente sqrt(x), 4', 'expr', 'x/4 + 1')
C(3, DER, 'derivar (x^2+1)^10', 'expr', '20*x*(x**2 + 1)**9')
C(3, DER, 'derivar 2^x', 'expr', '2**x*log(2)')
C(4, DER, 'derivar ln(sen(x))', 'expr', 'cos(x)/sin(x)')
C(4, DER, 'derivar arcsen(x^2)', 'expr', '2*x/sqrt(1 - x**4)')
C(4, DER, 'derivar e^x*cos(x), x, 2', 'expr', '-2*exp(x)*sin(x)')
C(4, DER, 'derivar senh(2x)*cosh(x)', 'expr', '2*cosh(2*x)*cosh(x) + sinh(2*x)*sinh(x)')
C(4, DER, 'derivar implicita y^2=x^3+x', 'expr', '(3*x**2 + 1)/(2*y)')
C(4, DER, 'recta tangente sen(x), pi', 'expr', 'pi - x')
C(4, DER, 'recta tangente e^x*sen(x), 0', 'expr', 'x')
C(4, DER, 'derivar x^2*e^(y*z), z', 'expr', 'x**2*y*exp(y*z)')
C(5, DER, 'derivar x^x', 'expr', 'x**x*(log(x) + 1)')
C(5, DER, 'derivar implicita x^3+y^3=6x*y', 'expr', '(2*y - x**2)/(y**2 - 2*x)')
C(5, DER, 'derivar sqrt(1+sen(x)^2)', 'expr', 'sin(x)*cos(x)/sqrt(1 + sin(x)**2)')
C(5, DER, 'derivar ln((1+x)/(1-x))', 'expr', '2/(1 - x**2)')
C(5, DER, 'derivar x^2*e^(-x), x, 3', 'expr', '-(x**2 - 6*x + 6)*exp(-x)')
C(5, DER, 'derivar implicita e^y+x*y=e', 'expr', '-y/(x + exp(y))')
C(5, DER, 'derivar ln(sec(x)+tg(x))', 'expr', '1/cos(x)')
C(5, DER, 'derivar x^3*ln(x), x, 2', 'expr', '6*x*log(x) + 5*x')
C(5, DER, 'derivar tgh(x^2)', 'expr', '2*x*(1 - tanh(x**2)**2)')
C(6, DER, 'derivar arctg((1+x)/(1-x))', 'expr', '1/(x**2 + 1)')
C(6, DER, 'derivar x*sen(x), x, 4', 'expr', 'x*sin(x) - 4*cos(x)')
C(6, DER, 'derivar implicita sen(x+y)=y', 'expr', 'cos(x + y)/(1 - cos(x + y))')
C(7, DER, 'derivar sen(x)^x', 'expr', 'sin(x)**x*(log(sin(x)) + x*cos(x)/sin(x))')
C(7, DER, 'derivar implicita x^2*y^2+x*y=2', 'expr', '-y/x')
C(8, DER, 'derivar arccos(x)/sqrt(1-x^2)', 'expr', '-1/(1 - x**2) + x*acos(x)/(1 - x**2)**(3/2)')

# ---------------- Integrales indefinidas (38) ----------------
C(1, IND, 'integrar 3x^2', 'antiderivada', 'x**3')
C(1, IND, 'integrar x^4+2x', 'antiderivada', 'x**5/5 + x**2')
C(1, IND, 'integrar cos(x)', 'antiderivada', 'sin(x)')
C(1, IND, 'integrar 1/x', 'antiderivada', 'log(x)')
C(2, IND, 'integrar sen(2x)', 'antiderivada', '-cos(2*x)/2')
C(2, IND, 'integrar sqrt(x)', 'antiderivada', '2*x**(3/2)/3')
C(2, IND, 'integrar 1/(1+x^2)', 'antiderivada', 'atan(x)')
C(2, IND, 'integrar 4t^3-2t, t', 'antiderivada', 't**4 - t**2')
C(3, IND, 'integrar x*cos(x)', 'antiderivada', 'x*sin(x) + cos(x)')
C(3, IND, 'integrar x*e^(2x)', 'antiderivada', '(2*x - 1)*exp(2*x)/4')
C(3, IND, 'integrar ln(x)', 'antiderivada', 'x*log(x) - x')
C(3, IND, 'integrar x/(x^2+1)', 'antiderivada', 'log(x**2 + 1)/2')
C(3, IND, 'integrar tg(x)', 'antiderivada', '-log(cos(x))')
C(3, IND, 'integrar 1/sqrt(1-x^2)', 'antiderivada', 'asin(x)')
C(4, IND, 'integrar 1/(x^2-1)', 'antiderivada', 'log(x - 1)/2 - log(x + 1)/2')
C(4, IND, 'integrar x^2*ln(x)', 'antiderivada', 'x**3*log(x)/3 - x**3/9')
C(4, IND, 'integrar x^2*e^x', 'antiderivada', '(x**2 - 2*x + 2)*exp(x)')
C(4, IND, 'integrar sen(x)^2', 'antiderivada', 'x/2 - sin(2*x)/4')
C(4, IND, 'integrar 1/(x*ln(x))', 'antiderivada', 'log(log(x))')
C(5, IND, 'integrar e^x*sen(x)', 'antiderivada', 'exp(x)*(sin(x) - cos(x))/2')
C(5, IND, 'integrar 1/(x^2+4x+5)', 'antiderivada', 'atan(x + 2)')
C(5, IND, 'integrar (3x+5)/(x^2+3x+2)', 'antiderivada', '2*log(x + 1) + log(x + 2)')
C(5, IND, 'integrar arctg(x)', 'antiderivada', 'x*atan(x) - log(x**2 + 1)/2')
C(6, IND, 'integrar sen(x)^3*cos(x)^2', 'antiderivada', '-cos(x)**3/3 + cos(x)**5/5')
C(6, IND, 'integrar sqrt(4-x^2)', 'antiderivada', 'x*sqrt(4 - x**2)/2 + 2*asin(x/2)')
C(6, IND, 'integrar x^3*e^(x^2)', 'antiderivada', '(x**2 - 1)*exp(x**2)/2')
C(6, IND, 'integrar sec(x)', 'antiderivada', 'log(1/cos(x) + tan(x))')
C(6, IND, 'integrar ln(x)^2', 'antiderivada', 'x*log(x)**2 - 2*x*log(x) + 2*x')
C(7, IND, 'integrar e^(2x)*cos(3x)', 'antiderivada', 'exp(2*x)*(2*cos(3*x) + 3*sin(3*x))/13')
C(7, IND, 'integrar x/((x-1)*(x+2)^2)', 'antiderivada',
  'log(x - 1)/9 - log(x + 2)/9 - 2/(3*(x + 2))')
C(7, IND, 'integrar x*arctg(x)', 'antiderivada', '(x**2 + 1)*atan(x)/2 - x/2')
C(7, IND, 'integrar (2x+3)/(x^2+2x+5)', 'antiderivada', 'log(x**2 + 2*x + 5) + atan((x + 1)/2)/2')
C(7, IND, 'integrar e^(sqrt(x))', 'antiderivada', '2*(sqrt(x) - 1)*exp(sqrt(x))')
C(7, IND, 'integrar 1/(x^2*sqrt(x^2+1))', 'antiderivada', '-sqrt(x**2 + 1)/x')
C(8, IND, 'integrar cos(ln(x))', 'antiderivada', 'x*(cos(log(x)) + sin(log(x)))/2')
C(8, IND, 'integrar 1/(x^2+1)^2', 'antiderivada', 'x/(2*(x**2 + 1)) + atan(x)/2')
C(8, IND, 'integrar x^2/sqrt(1-x^2)', 'antiderivada', '(asin(x) - x*sqrt(1 - x**2))/2')
C(9, IND, 'integrar 1/(x^3+1)', 'antiderivada',
  'log(x + 1)/3 - log(x**2 - x + 1)/6 + sqrt(3)*atan((2*x - 1)/sqrt(3))/3')

# ---------------- Integrales definidas y área (30) ----------------
C(1, DEF, 'integrar x^2, x, 0, 2', 'expr', '8/3')
C(1, DEF, 'integrar 2x+1, x, 1, 3', 'expr', '10')
C(2, DEF, 'integrar cos(x) de 0 a pi/2', 'expr', '1')
C(2, DEF, 'integrar e^x, x, 0, 1', 'expr', 'E - 1')
C(2, DEF, 'integrar 1/x de 1 a e', 'expr', '1')
C(2, DEF, 'integrar sen(x), x, 0, pi', 'expr', '2')
C(3, DEF, 'area 4-x^2, 0, -2, 2', 'expr', '32/3')
C(3, DEF, 'area x^2, x^3, 0, 1', 'expr', '1/12')
C(3, DEF, 'integrar x*e^x, x, 0, 1', 'expr', '1')
C(3, DEF, 'integrar 1/(1+x^2), x, 0, 1', 'expr', 'pi/4')
C(3, DEF, 'integrar sqrt(x)*(1+x), x, 0, 1', 'expr', '16/15')
C(4, DEF, 'integrar e^(-x), x, 0, oo', 'expr', '1')
C(4, DEF, 'integrar x*sen(x), x, 0, pi', 'expr', 'pi')
C(4, DEF, 'area x^2, 2x, 0, 2', 'expr', '4/3')
C(5, DEF, 'area sen(x), cos(x), 0, pi/2', 'expr', '2*sqrt(2) - 2')
C(5, DEF, 'integrar sqrt(1-x^2), x, -1, 1', 'expr', 'pi/2')
C(5, DEF, 'integrar 1/sqrt(x), x, 0, 4', 'expr', '4')
C(5, DEF, 'area x^3, x, -1, 1', 'expr', '1/2')
C(5, DEF, 'area sen(x), 0, 0, 2pi', 'expr', '4')
C(6, DEF, 'integrar x^2*e^(-x), x, 0, oo', 'expr', '2')
C(6, DEF, 'integrar 1/(x^2+4), x, -oo, oo', 'expr', 'pi/2')
C(6, DEF, 'area e^x, x+1, 0, 1', 'expr', 'E - 5/2')
C(6, DEF, 'integrar sen(x)/x, x, 0, 1', 'numero', '0.946083070367183', tol=1e-6)
C(6, DEF, 'integrar e^(-x^2), x, 0, 1', 'numero', '0.746824132812427', tol=1e-6)
C(6, DEF, 'area x^2-4, 0, -3, 3', 'expr', '46/3')
C(6, DEF, 'integrar cos(x)^4, x, 0, pi/2', 'expr', '3*pi/16')
C(7, DEF, 'integrar e^(-x^2), x, -oo, oo', 'expr', 'sqrt(pi)')
C(7, DEF, 'integrar 1/(x*ln(x)^2), x, e, oo', 'expr', '1')
C(7, DEF, 'integrar x^3*e^(-x^2), x, 0, oo', 'expr', '1/2')
C(7, DEF, 'integrar ln(x)/x^2, x, 1, oo', 'expr', '1')

# ---------------- Taylor y sumatorias (20) ----------------
C(1, TAY, 'sumatoria k, k, 1, 100', 'expr', '5050')
C(2, TAY, 'sumatoria k^2, k, 1, 10', 'expr', '385')
C(2, TAY, 'taylor cos(x), x, 0, 4', 'expr', 'x**4/24 - x**2/2 + 1')
C(3, TAY, 'taylor sen(x), x, 0, 5', 'expr', 'x - x**3/6 + x**5/120')
C(3, TAY, 'taylor ln(1+x), x, 0, 4', 'expr', 'x - x**2/2 + x**3/3 - x**4/4')
C(3, TAY, 'sumatoria 1/2^k, k, 0, oo', 'expr', '2')
C(3, TAY, 'sumatoria k, k, 1, n', 'expr', 'n*(n + 1)/2')
C(4, TAY, 'taylor sqrt(x), x, 4, 2', 'expr', '2 + (x - 4)/4 - (x - 4)**2/64')
C(4, TAY, 'taylor ln(x), x, 1, 3', 'expr', '(x - 1) - (x - 1)**2/2 + (x - 1)**3/3')
C(4, TAY, 'sumatoria (2/3)^k, k, 1, oo', 'expr', '2')
C(5, TAY, 'taylor arctg(x), x, 0, 5', 'expr', 'x - x**3/3 + x**5/5')
C(5, TAY, 'taylor cos(x), x, pi, 2', 'expr', '-1 + (x - pi)**2/2')
C(5, TAY, 'sumatoria 1/(k*(k+1)), k, 1, oo', 'expr', '1')
C(5, TAY, 'sumatoria k^3, k, 1, n', 'expr', 'n**2*(n + 1)**2/4')
C(5, TAY, 'taylor 1/x, x, 2, 3', 'expr', '1/2 - (x - 2)/4 + (x - 2)**2/8 - (x - 2)**3/16')
C(6, TAY, 'taylor e^x*sen(x), x, 0, 4', 'expr', 'x + x**2 + x**3/3')
C(6, TAY, 'taylor tg(x), x, 0, 5', 'expr', 'x + x**3/3 + 2*x**5/15')
C(6, TAY, 'sumatoria 1/k^2, k, 1, oo', 'expr', 'pi**2/6')
C(7, TAY, 'sumatoria (-1)^(k+1)/k, k, 1, oo', 'expr', 'log(2)')
C(7, TAY, 'sumatoria (-1)^k/factorial(k), k, 0, oo', 'expr', 'exp(-1)')

# ---------------- Extremos y Lagrange (20) ----------------
C(2, EXT, 'extremos x^2-4x+3', 'conjunto', '[2]')
C(3, EXT, 'extremos x^3-12x', 'conjunto', '[-2, 2]')
C(3, EXT, 'extremos x^3-6x^2+9x+1', 'conjunto', '[1, 3]')
C(4, EXT, 'extremos x^4-8x^2', 'conjunto', '[-2, 0, 2]')
C(4, EXT, 'extremos x*e^(-x)', 'conjunto', '[1]')
C(4, EXT, 'extremos x^2+y^2-4x+6y', 'conjunto', '[(2, -3)]')
C(4, EXT, 'extremos x^2-y^2', 'conjunto', '[(0, 0)]')
C(5, EXT, 'extremos ln(x)/x', 'conjunto', '[E]')
C(5, EXT, 'extremos x*y-x^2-y^2+3y', 'conjunto', '[(1, 2)]')
C(5, EXT, 'lagrange x*y; 2x+y=20', 'conjunto', '[(5, 10)]')
C(6, EXT, 'extremos x^3+y^3-3x*y', 'conjunto', '[(0, 0), (1, 1)]')
C(6, EXT, 'lagrange x+y; x^2+y^2=8', 'conjunto', '[(2, 2), (-2, -2)]')
C(6, EXT, 'lagrange 3x+4y; x^2+y^2=25', 'clave', '25', clave='maximo')
C(7, EXT, 'extremos x^4+y^4-4x*y', 'conjunto', '[(0, 0), (1, 1), (-1, -1)]')
C(7, EXT, 'extremos x^2*y-x^2-2y^2', 'conjunto', '[(0, 0), (2, 1), (-2, 1)]')
C(7, EXT, 'lagrange x^2+2y^2; x^2+y^2=1', 'conjunto', '[(0, 1), (0, -1), (1, 0), (-1, 0)]')
C(7, EXT, 'lagrange x^2+y^2; x*y=1', 'clave', '2', clave='minimo')
C(7, EXT, 'lagrange x^2+y^2+z^2; x+2y+3z=14', 'conjunto', '[(1, 2, 3)]')
C(8, EXT, 'lagrange x*y*z; x^2+y^2+z^2=3', 'clave', '1', clave='maximo')
C(8, EXT, 'extremos 2x^3+6x*y^2-3y^3-150x', 'conjunto', '[(5, 0), (-5, 0), (3, 4), (-3, -4)]')

# ---------------- Varias variables y vectorial (35) ----------------
C(2, VV, 'derivar x^2*y+3y^2, x', 'expr', '2*x*y')
C(2, VV, 'gradiente x^2+y^2', 'vector', '[2*x, 2*y]')
C(3, VV, 'gradiente x*y*z', 'vector', '[y*z, x*z, x*y]')
C(3, VV, 'gradiente e^x*sen(y)', 'vector', '[exp(x)*sin(y), exp(x)*cos(y)]')
C(3, VV, 'divergencia [x^2, y^2, z^2]', 'expr', '2*x + 2*y + 2*z')
C(3, VV, 'rotacional [-y, x, 0]', 'vector', '[0, 0, 2]')
C(3, VV, 'integral doble x+y, x, 0, 1, y, 0, 1', 'expr', '1')
C(3, VV, 'integral doble x^2*y, x, 0, 2, y, 0, 3', 'expr', '12')
C(3, VV, 'derivar x*e^(x*y), y', 'expr', 'x**2*exp(x*y)')
C(3, VV, 'derivar ln(x^2+y^2), x', 'expr', '2*x/(x**2 + y**2)')
C(4, VV, 'laplaciano x^3-3x*y^2', 'expr', '0')
C(4, VV, 'jacobiano [x*y, x^2-y^2]', 'matriz', '[[y, x], [2*x, -2*y]]')
C(4, VV, 'hessiano x^2*y+y^3', 'matriz', '[[2*y, 2*x], [2*x, 6*y]]')
C(4, VV, 'integral triple x*y*z, x, 0, 1, y, 0, 1, z, 0, 1', 'expr', '1/8')
C(4, VV, 'integral doble e^(x+y), x, 0, 1, y, 0, 1', 'expr', '(E - 1)**2')
C(4, VV, 'divergencia [x^2*y, -x*y^2, z]', 'expr', '1')
C(4, VV, 'rotacional [y*z, x*z, x*y]', 'vector', '[0, 0, 0]')
C(4, VV, 'gradiente ln(x^2+y^2+z^2)', 'vector',
  '[2*x/(x**2 + y**2 + z**2), 2*y/(x**2 + y**2 + z**2), 2*z/(x**2 + y**2 + z**2)]')
C(4, VV, 'derivar x^3*y^2*z, x, 2', 'expr', '6*x*y**2*z')
C(5, VV, 'integral doble x*y, y, 0, x, x, 0, 1', 'expr', '1/8')
C(5, VV, 'integral doble 1, y, x^2, x, x, 0, 1', 'expr', '1/6')
C(5, VV, 'rotacional [x^2*y, y*z, x*z^2]', 'vector', '[-y, -z**2, -x**2]')
C(5, VV, 'laplaciano x^2*y*z^3', 'expr', '2*y*z**3 + 6*x**2*y*z')
C(5, VV, 'jacobiano [x*cos(y), x*sen(y)]', 'matriz', '[[cos(y), -x*sin(y)], [sin(y), x*cos(y)]]')
C(5, VV, 'jacobiano [x+y+z, x*y, y*z]', 'matriz', '[[1, 1, 1], [y, x, 0], [0, z, y]]')
C(5, VV, 'hessiano x^2*y^2+x*y', 'matriz', '[[2*y**2, 4*x*y + 1], [4*x*y + 1, 2*x**2]]')
C(5, VV, 'integral triple x+y+z, x, 0, 1, y, 0, 2, z, 0, 3', 'expr', '18')
C(6, VV, 'integral triple 1, z, 0, 1-x-y, y, 0, 1-x, x, 0, 1', 'expr', '1/6')
C(6, VV, 'hessiano e^(x*y)', 'matriz',
  '[[y**2*exp(x*y), (x*y + 1)*exp(x*y)], [(x*y + 1)*exp(x*y), x**2*exp(x*y)]]')
C(6, VV, 'integral doble x*e^(x*y), y, 0, 1, x, 0, 1', 'expr', 'E - 2')
C(6, VV, 'integral triple z, z, 0, x+y, y, 0, 1, x, 0, 1', 'expr', '7/12')
C(7, VV, 'laplaciano 1/sqrt(x^2+y^2+z^2)', 'expr', '0')
C(7, VV, 'integral doble x*y^2, x, y^2, 4, y, 0, 2', 'expr', '256/21')
C(8, VV, 'integral doble x^2+y^2, y, 0, sqrt(1-x^2), x, 0, 1', 'expr', 'pi/8')
C(8, VV, 'jacobiano [x*cos(y)*sen(z), x*sen(y)*sen(z), x*cos(z)]', 'matriz',
  '[[cos(y)*sin(z), -x*sin(y)*sin(z), x*cos(y)*cos(z)], '
  '[sin(y)*sin(z), x*cos(y)*sin(z), x*sin(y)*cos(z)], [cos(z), 0, -x*sin(z)]]')

# ---------------- Vectores (15) ----------------
C(1, VEC, 'punto [1,2,3], [4,5,6]', 'expr', '32')
C(1, VEC, 'norma [1,2,2]', 'expr', '3')
C(2, VEC, 'cruz [1,2,3], [4,5,6]', 'vector', '[-3, 6, -3]')
C(2, VEC, 'norma [1,-2,3,-4]', 'expr', 'sqrt(30)')
C(2, VEC, 'punto [2,-1,3], [1,4,-2]', 'expr', '-8')
C(3, VEC, 'angulo [1,1], [1,0]', 'expr', 'pi/4')
C(3, VEC, 'proyeccion [2,3], [1,1]', 'vector', '[5/2, 5/2]')
C(3, VEC, 'angulo [1,2,2], [2,-2,1]', 'expr', 'pi/2')
C(3, VEC, 'cruz [2,-1,1], [1,3,-2]', 'vector', '[-1, 5, 7]')
C(3, VEC, 'norma [1/2, sqrt(3)/2]', 'expr', '1')
C(4, VEC, 'angulo [1,0,1], [0,1,1]', 'expr', 'pi/3')
C(4, VEC, 'proyeccion [3,4,0], [1,0,1]', 'vector', '[3/2, 0, 3/2]')
C(5, VEC, 'angulo [3,0], [-1,sqrt(3)]', 'expr', '2*pi/3')
C(5, VEC, 'proyeccion [1,1,1], [2,-1,2]', 'vector', '[2/3, -1/3, 2/3]')
C(6, VEC, 'angulo [1,2,3], [3,2,1]', 'expr', 'acos(5/7)')

# ---------------- Álgebra lineal (42) ----------------
C(1, LIN, 'det [[2,1],[5,3]]', 'expr', '1')
C(1, LIN, 'transpuesta [[1,2,3],[4,5,6]]', 'matriz', '[[1, 4], [2, 5], [3, 6]]')
C(1, LIN, 'traza [[1,2],[3,4]]', 'expr', '5')
C(1, LIN, '[[1,2],[3,4]]+[[5,6],[7,8]]', 'matriz', '[[6, 8], [10, 12]]')
C(1, LIN, '3*[[1,-1],[0,2]]', 'matriz', '[[3, -3], [0, 6]]')
C(2, LIN, 'rango [[1,2],[2,4]]', 'expr', '1')
C(2, LIN, '[[1,2],[3,4]]*[[2,0],[1,3]]', 'matriz', '[[4, 6], [10, 12]]')
C(2, LIN, 'det [[1,2,3],[4,5,6],[7,8,9]]', 'expr', '0')
C(2, LIN, '[[2,1],[1,1]]^2', 'matriz', '[[5, 3], [3, 2]]')
C(2, LIN, 'inversa [[2,0,0],[0,4,0],[0,0,5]]', 'matriz', '[[1/2, 0, 0], [0, 1/4, 0], [0, 0, 1/5]]')
C(3, LIN, 'det [[1,2,3],[0,1,4],[5,6,0]]', 'expr', '1')
C(3, LIN, 'inversa [[2,1],[5,3]]', 'matriz', '[[3, -1], [-5, 2]]')
C(3, LIN, 'autovalores [[4,1],[2,3]]', 'conjunto', '[2, 5]')
C(3, LIN, '[[1,1],[0,1]]^3', 'matriz', '[[1, 3], [0, 1]]')
C(3, LIN, 'rango [[1,2,3],[4,5,6],[7,8,9]]', 'expr', '2')
C(3, LIN, 'autovalores [[2,0,0],[1,3,0],[4,5,6]]', 'conjunto', '[2, 3, 6]')
C(3, LIN, '[[1,0,2],[0,1,1]]*[[1,2],[0,1],[3,0]]', 'matriz', '[[7, 2], [3, 1]]')
C(3, LIN, '[[1,2],[3,4]]*[[4,-2],[-3,1]]', 'matriz', '[[-2, 0], [0, -2]]')
C(4, LIN, 'gauss [[1,2,5],[3,-1,1]]', 'matriz', '[[1, 0, 1], [0, 1, 2]]')
C(4, LIN, 'autovalores [[1,1],[0,1]]', 'conjunto', '[1, 1]')
C(4, LIN, 'autovalores [[0,-1],[1,0]]', 'conjunto', '[I, -I]')
C(4, LIN, 'autovalores [[4,-2],[1,1]]', 'conjunto', '[2, 3]')
C(4, LIN, 'rango [[1,0,1],[0,1,1],[1,1,2]]', 'expr', '2')
C(4, LIN, 'det [[x,1],[1,x]]', 'expr', 'x**2 - 1')
C(4, LIN, '[[1,2],[3,4]]^2-5*[[1,2],[3,4]]', 'matriz', '[[2, 0], [0, 2]]')
C(4, LIN, 'inversa [[1,1,1],[0,1,1],[0,0,1]]', 'matriz', '[[1, -1, 0], [0, 1, -1], [0, 0, 1]]')
C(4, LIN, 'inversa [[4,7],[2,6]]', 'matriz', '[[3/5, -7/10], [-1/5, 2/5]]')
C(5, LIN, 'inversa [[1,0,2],[2,-1,3],[4,1,8]]', 'matriz', '[[-11, 2, 2], [-4, 0, 1], [6, -1, -1]]')
C(5, LIN, 'autovalores [[1,2,0],[2,1,0],[0,0,3]]', 'conjunto', '[-1, 3, 3]')
C(5, LIN, 'gauss [[1,1,1,6],[2,-1,1,3],[1,2,-1,2]]', 'matriz',
  '[[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]')
C(5, LIN, 'autovalores [[3,1],[-1,1]]', 'conjunto', '[2, 2]')
C(5, LIN, 'autovalores [[1,-1],[1,1]]', 'conjunto', '[1 - I, 1 + I]')
C(5, LIN, 'gauss [[2,1,-1,8],[-3,-1,2,-11],[-2,1,2,-3]]', 'matriz',
  '[[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, -1]]')
C(6, LIN, 'autovalores [[2,1,1],[1,2,1],[1,1,2]]', 'conjunto', '[1, 1, 4]')
C(6, LIN, 'gauss [[1,2,1,4],[2,4,3,9],[1,2,2,5]]', 'matriz',
  '[[1, 2, 0, 3], [0, 0, 1, 1], [0, 0, 0, 0]]')
C(6, LIN, 'det [[1,2,0,1],[0,1,3,0],[2,0,1,1],[1,1,0,2]]', 'expr', '16')
C(6, LIN, 'rango [[1,2,1,0],[2,1,0,1],[3,3,1,1],[1,-1,-1,1]]', 'expr', '2')
C(6, LIN, 'inversa [[cos(x), -sen(x)],[sen(x), cos(x)]]', 'matriz',
  '[[cos(x), sin(x)], [-sin(x), cos(x)]]')
C(6, LIN, 'det [[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]]', 'expr', '5')
C(6, LIN, 'gauss [[1,1,2,9],[2,4,-3,1],[3,6,-5,0]]', 'matriz',
  '[[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]')
C(7, LIN, 'det [[1,1,1],[x,y,z],[x^2,y^2,z^2]]', 'expr', '(y - x)*(z - x)*(z - y)')
C(7, LIN, 'autovalores [[0,1,0],[0,0,1],[6,-11,6]]', 'conjunto', '[1, 2, 3]')

# ---------------- EDO (35) ----------------
C(2, EDO, "edo y'=2x", 'edo_general', 'x**2 + C1')
C(2, EDO, "edo y'=3y", 'edo_general', 'C1*exp(3*x)')
C(2, EDO, "edo y'=x^2; y(0)=1", 'expr', 'x**3/3 + 1')
C(3, EDO, "edo y'=x*y", 'edo_general', 'C1*exp(x**2/2)')
C(3, EDO, "edo y''+y=0", 'edo_general', 'C1*cos(x) + C2*sin(x)')
C(3, EDO, "edo y'=2y; y(0)=3", 'expr', '3*exp(2*x)')
C(4, EDO, "edo y''-5y'+6y=0", 'edo_general', 'C1*exp(2*x) + C2*exp(3*x)')
C(4, EDO, "edo y''+4y'+4y=0", 'edo_general', '(C1 + C2*x)*exp(-2*x)')
C(4, EDO, "edo y'+2y=e^(-x)", 'edo_general', 'exp(-x) + C1*exp(-2*x)')
C(4, EDO, "edo y'=-2t*y; y(0)=1", 'expr', 'exp(-t**2)')
C(4, EDO, "edo (x+1)y'=y", 'edo_general', 'C1*(x + 1)')
C(5, EDO, "edo y''+2y'+5y=0", 'edo_general', 'exp(-x)*(C1*cos(2*x) + C2*sin(2*x))')
C(5, EDO, "edo y'+y/x=x^2", 'edo_general', 'x**3/4 + C1/x')
C(5, EDO, "edo y''-y=x", 'edo_general', 'C1*exp(x) + C2*exp(-x) - x')
C(5, EDO, "edo y'=y^2", 'edo_general', '-1/(x + C1)')
C(5, EDO, "edo y''+4y=0; y(0)=1; y'(0)=2", 'expr', 'cos(2*x) + sin(2*x)')
C(5, EDO, "edo y'+y=x; y(0)=0", 'expr', 'x - 1 + exp(-x)')
C(6, EDO, "edo y''-3y'+2y=e^(3x)", 'edo_general', 'C1*exp(x) + C2*exp(2*x) + exp(3*x)/2')
C(6, EDO, "edo y'''-y'=0", 'edo_general', 'C1 + C2*exp(x) + C3*exp(-x)')
C(6, EDO, "edo x*y'-2y=x^3", 'edo_general', 'x**3 + C1*x**2')
C(6, EDO, "edo y''+9y=cos(t)", 'edo_general', 'C1*cos(3*t) + C2*sin(3*t) + cos(t)/8')
C(6, EDO, "edo y'=e^(x-y)", 'edo_general', 'log(exp(x) + C1)')
C(6, EDO, "edo y''+2y'+y=0; y(0)=1; y'(0)=-3", 'expr', '(1 - 2*x)*exp(-x)')
C(6, EDO, "edo y''+y=0; y(0)=0; y(pi/2)=1", 'expr', 'sin(x)')
C(6, EDO, "edo y'-y=e^(2t); y(0)=1", 'expr', 'exp(2*t)')
C(7, EDO, "edo y''+y=sen(x)", 'edo_general', 'C1*cos(x) + C2*sin(x) - x*cos(x)/2')
C(7, EDO, "edo y''-2y'+y=e^x", 'edo_general', '(C1 + C2*x)*exp(x) + x**2*exp(x)/2')
C(7, EDO, "edo y'+y*tg(x)=cos(x)", 'edo_general', '(x + C1)*cos(x)')
C(7, EDO, "edo y''-4y'+13y=0; y(0)=1; y'(0)=2", 'expr', 'exp(2*x)*cos(3*x)')
C(7, EDO, "edo y'=y*(1-y); y(0)=1/2", 'expr', '1/(1 + exp(-x))')
C(7, EDO, "edo y''+3y'+2y=4; y(0)=0; y'(0)=0", 'expr', '2 - 4*exp(-x) + 2*exp(-2*x)')
C(8, EDO, "edo y''+4y=sen(t); y(0)=0; y'(0)=0", 'expr', 'sin(t)/3 - sin(2*t)/6')
C(8, EDO, "edo x^2*y''-2x*y'+2y=0", 'edo_general', 'C1*x + C2*x**2')
C(8, EDO, "edo y''-y'-2y=4x; y(0)=0; y'(0)=1", 'expr', '2*exp(2*x)/3 - 5*exp(-x)/3 - 2*x + 1')
C(9, EDO, "edo y'''-6y''+11y'-6y=0; y(0)=0; y'(0)=0; y''(0)=2", 'expr',
  'exp(x) - 2*exp(2*x) + exp(3*x)')

# ---------------- Laplace (25) ----------------
C(1, LAP, 'laplace 1', 'expr', '1/s')
C(2, LAP, 'laplace t', 'expr', '1/s**2')
C(2, LAP, 'laplace e^(3t)', 'expr', '1/(s - 3)')
C(2, LAP, 'laplace sen(2t)', 'expr', '2/(s**2 + 4)')
C(2, LAP, 'laplace inversa 1/(s-5)', 'expr', 'exp(5*t)')
C(3, LAP, 'laplace t*e^(2t)', 'expr', '1/(s - 2)**2')
C(3, LAP, 'laplace senh(2t)', 'expr', '2/(s**2 - 4)')
C(3, LAP, 'laplace 3-2t+4t^2', 'expr', '3/s - 2/s**2 + 8/s**3')
C(3, LAP, 'laplace inversa 6/s^4', 'expr', 't**3')
C(3, LAP, 'laplace inversa s/(s^2+16)', 'expr', 'cos(4*t)')
C(4, LAP, 'laplace e^(-t)*sen(2t)', 'expr', '2/((s + 1)**2 + 4)')
C(4, LAP, 'laplace inversa 1/(s*(s+1))', 'expr', '1 - exp(-t)')
C(4, LAP, 'laplace inversa 1/(s+2)^2', 'expr', 't*exp(-2*t)')
C(5, LAP, 'laplace t*sen(t)', 'expr', '2*s/(s**2 + 1)**2')
C(5, LAP, 'laplace u(t-2)', 'expr', 'exp(-2*s)/s')
C(5, LAP, 'laplace delta(t-3)', 'expr', 'exp(-3*s)')
C(5, LAP, 'laplace inversa (s+1)/(s^2+2s+5)', 'expr', 'exp(-t)*cos(2*t)')
C(5, LAP, 'laplace inversa 3/(s^2+2s+10)', 'expr', 'exp(-t)*sin(3*t)')
C(5, LAP, 'laplace inversa (2s+3)/(s^2+3s+2)', 'expr', 'exp(-t) + exp(-2*t)')
C(5, LAP, 'laplace inversa 2/(s-1)^3', 'expr', 't**2*exp(t)')
C(6, LAP, 'laplace (t-1)*u(t-1)', 'expr', 'exp(-s)/s**2')
C(6, LAP, 'laplace inversa 1/(s^2*(s^2+1))', 'expr', 't - sin(t)')
C(7, LAP, 'laplace inversa e^(-2s)/s^2', 'expr', '(t - 2)*Heaviside(t - 2)')
C(7, LAP, 'laplace inversa 5/((s+1)*(s^2+4))', 'expr', 'exp(-t) - cos(2*t) + sin(2*t)/2')
C(7, LAP, 'laplace t^2*cos(t)', 'expr', '2*s*(s**2 - 3)/(s**2 + 1)**3')

# ---------------- Fourier (6) ----------------
C(6, FOU, 'fourier x, x, -pi, pi, 2', 'expr', '2*sin(x) - sin(2*x)')
C(7, FOU, 'fourier x^2, x, -pi, pi, 3', 'expr', 'pi**2/3 - 4*cos(x) + cos(2*x) - 4*cos(3*x)/9')
C(7, FOU, 'fourier x, x, -1, 1, 3', 'expr',
  '2*sin(pi*x)/pi - sin(2*pi*x)/pi + 2*sin(3*pi*x)/(3*pi)')
C(8, FOU, 'fourier abs(x), x, -pi, pi, 3', 'expr', 'pi/2 - 4*cos(x)/pi - 4*cos(3*x)/(9*pi)')
C(8, FOU, 'fourier 1-x^2, x, -1, 1, 2', 'expr', '2/3 + 4*cos(pi*x)/pi**2 - cos(2*pi*x)/pi**2')
C(8, FOU, 'fourier u(x), x, -pi, pi, 3', 'expr', '1/2 + 2*sin(x)/pi + 2*sin(3*x)/(3*pi)')

# ---------------- Métodos numéricos (34) ----------------
# Raíces: esperado = raíz exacta (o None -> la calcula el método y se verifica f(raíz)=0).
C(3, NUM, 'newton x^2-2, 1', 'numero', 'sqrt(2)', tol=1e-6)
C(4, NUM, 'newton x^3-x-1, 1.5', 'numero', None, tol=1e-6)
C(4, NUM, 'newton cos(x)-x, 1', 'numero', None, tol=1e-6)
C(5, NUM, 'newton e^x-3x, 0', 'numero', None, tol=1e-6)
C(6, NUM, 'newton x*e^x-1, 0.5', 'numero', 'LambertW(1)', tol=1e-6)
C(3, NUM, 'biseccion x^2-3, 1, 2', 'numero', 'sqrt(3)', tol=1e-6)
C(4, NUM, 'biseccion x^3+x-1, 0, 1', 'numero', None, tol=1e-6)
C(5, NUM, 'biseccion e^x+x-2, 0, 1', 'numero', None, tol=1e-6)
C(6, NUM, 'biseccion tg(x)-x, 4, 4.6', 'numero', None, tol=1e-6)
C(4, NUM, 'secante x^2-x-1, 1, 2', 'numero', '(1 + sqrt(5))/2', tol=1e-6)
C(6, NUM, 'secante sen(x)-x/2, 1.5, 2', 'numero', None, tol=1e-6)
C(6, NUM, 'secante x^4-x-10, 1.5, 2', 'numero', None, tol=1e-6)
C(5, NUM, 'punto fijo e^(-x), 0.5', 'numero', 'LambertW(1)', tol=1e-6)
C(5, NUM, 'punto fijo (x+3/x)/2, 1', 'numero', 'sqrt(3)', tol=1e-6)
C(5, NUM, 'punto fijo 1+1/x, 1', 'numero', '(1 + sqrt(5))/2', tol=1e-6)
# Cuadratura y EDO: esperado = resultado EXACTO del método (calculado aquí).
C(4, NUM, 'trapecio x^2, 0, 2, 4', 'numero', '11/4', tol=1e-9)
C(6, NUM, 'trapecio e^(-x^2), 0, 1, 4', 'numero', None, tol=1e-9)
C(6, NUM, 'trapecio sen(x), 0, pi, 6', 'numero', None, tol=1e-9)
C(4, NUM, 'simpson x^3, 0, 2, 2', 'numero', '4', tol=1e-9)
C(6, NUM, 'simpson 1/(1+x^2), 0, 1, 4', 'numero', None, tol=1e-9)
C(7, NUM, 'simpson sqrt(1+x^3), 0, 1, 4', 'numero', None, tol=1e-9)
C(5, NUM, "euler y'=y, 0, 1, 0.25, 1", 'numero', '625/256', tol=1e-9)
C(6, NUM, "euler y'=x+y, 0, 1, 0.2, 1", 'numero', None, tol=1e-9)
C(6, NUM, "euler y'=-2x*y, 0, 1, 0.1, 0.5", 'numero', None, tol=1e-9)
C(7, NUM, "rk4 y'=y, 0, 1, 0.5, 1", 'numero', None, tol=1e-9)
C(8, NUM, "rk4 y'=y-x^2+1, 0, 0.5, 0.2, 2", 'numero', None, tol=1e-9)
C(8, NUM, "rk4 y'=-y+sen(x), 0, 0, 0.2, 1", 'numero', None, tol=1e-9)
C(4, NUM, 'interpolar (1,2), (2,3), (3,6)', 'expr', 'x**2 - 2*x + 3')
C(5, NUM, 'interpolar (0,0), (1,1), (2,8), (3,27)', 'expr', 'x**3')
C(5, NUM, 'interpolar (-1,4), (0,1), (2,1)', 'expr', 'x**2 - 2*x + 1')
C(3, NUM, 'regresion (1,2), (2,4), (3,6)', 'expr', '2*x')
C(4, NUM, 'regresion (0,1), (1,3), (2,5), (3,7)', 'expr', '2*x + 1')
C(5, NUM, 'regresion (1,1), (2,2), (3,2), (4,3)', 'expr', '3*x/5 + 1/2')
C(7, NUM, 'regresion (1,3), (2,5), (3,6), (4,9), (5,10)', 'clave', '81/83', clave='r2', tol=1e-6)

# ---------------- Complejos y circuitos (18) ----------------
C(1, CPX, 'paralelo 6, 3', 'expr', '2')
C(1, CPX, 'serie 100, 220, 330', 'expr', '650')
C(2, CPX, 'paralelo 4, 6, 12', 'expr', '2')
C(2, CPX, '(2+3i)(1-i)', 'expr', '5 + I')
C(2, CPX, 'abs(3-4i)', 'expr', '5')
C(3, CPX, '(1+i)^8', 'expr', '16')
C(3, CPX, '(5+5i)/(1+2i)', 'expr', '3 - I')
C(3, CPX, 'polar 3i', 'vector', '[3, pi/2]')
C(3, CPX, 'fasor 10@0 + 10@90', 'numero', '10 + 10*I', tol=1e-6)
C(4, CPX, 'polar -1+sqrt(3)i', 'vector', '[2, 2*pi/3]')
C(4, CPX, 'polar -2-2i', 'vector', '[2*sqrt(2), -3*pi/4]')
C(4, CPX, 'fasor 20@60 - 10@0', 'numero', '10*sqrt(3)*I', tol=1e-6)
C(4, CPX, 'e^(i*pi/2)', 'expr', 'I')
C(4, CPX, '(2-i)^3', 'expr', '2 - 11*I')
C(5, CPX, 'fasor 5@53.13 + 3@-90', 'numero', None, tol=1e-6)
C(6, CPX, 'fasor 100@0 + 100@120 + 100@-120', 'expr', '0')
C(6, CPX, '(10+5i)*(4-3i)/((10+5i)+(4-3i))', 'expr', '15/4 - 5*I/4')
C(6, CPX, 'fasor 220@0 - 220@-120', 'numero', '330 + 110*sqrt(3)*I', tol=1e-6)

# ---------------- Estadística y probabilidad (15) ----------------
C(1, EST, 'estadistica 3, 5, 7, 9', 'numero', '6', tol=1e-9)
C(3, EST, 'estadistica 1, 2, 3, 4, 5', 'clave', '5/2', clave='varianza', tol=1e-9)
C(4, EST, 'estadistica 2, 4, 4, 4, 5, 5, 7, 9', 'clave', '2', clave='desvio_pob', tol=1e-9)
C(4, EST, 'estadistica 10, 12, 23, 23, 16, 23, 21, 16', 'clave', '37/2', clave='mediana', tol=1e-9)
C(5, EST, 'estadistica 4, 8, 6, 5, 3, 7', 'clave', 'sqrt(7/2)', clave='desvio', tol=1e-9)
C(5, EST, 'estadistica 1.5, 2.5, 3.5, 4.5', 'clave', '5/4', clave='varianza_pob', tol=1e-9)
C(3, EST, 'binomial 5, 0.5, 2', 'numero', '5/16', tol=1e-9)
C(5, EST, 'binomial 10, 0.3, 4', 'numero', 'binomial(10, 4)*(3/10)**4*(7/10)**6', tol=1e-9)
C(4, EST, 'binomial 20, 0.1, 0', 'numero', '(9/10)**20', tol=1e-9)
C(4, EST, 'poisson 3, 2', 'numero', '9*exp(-3)/2', tol=1e-9)
C(5, EST, 'poisson 2.5, 4', 'numero', 'exp(-5/2)*(5/2)**4/24', tol=1e-9)
C(4, EST, 'normal 0, 1, -1, 1', 'numero', 'erf(sqrt(2)/2)', tol=1e-6)
C(5, EST, 'normal 50, 10, 60, oo', 'numero', '0.158655253931457', tol=1e-6)
C(6, EST, 'normal 100, 15, 85, 130', 'numero', '0.818594614120', tol=1e-6)
C(6, EST, 'normal 0, 1, -oo, 1.645', 'numero', '0.950015094', tol=1e-6)

# ---------------- Gráficos (16) ----------------
C(1, GRA, 'graficar x^2-4', 'grafico', '2d')
C(2, GRA, 'graficar e^x, ln(x)', 'grafico', '2d')
C(2, GRA, 'graficar tg(x)', 'grafico', '2d')
C(2, GRA, 'graficar abs(x-1)', 'grafico', '2d')
C(2, GRA, 'graficar datos (1,2), (2,3), (3,5), (4,4)', 'grafico', 'datos')
C(3, GRA, 'graficar x^3-3x, 3x^2-3', 'grafico', '2d')
C(3, GRA, 'graficar x^2/4+y^2/9=1', 'grafico', 'implicita')
C(3, GRA, 'graficar x^2-y^2=1', 'grafico', 'implicita')
C(3, GRA, 'graficar polar 1-sen(t)', 'grafico', 'polar')
C(3, GRA, 'graficar parametrica 3cos(t), 2sen(t)', 'grafico', 'parametrica')
C(4, GRA, 'graficar y^2=x^3-x', 'grafico', 'implicita')
C(4, GRA, 'graficar polar 2sen(3t)', 'grafico', 'polar')
C(4, GRA, 'graficar polar t', 'grafico', 'polar')
C(4, GRA, 'graficar parametrica t-sen(t), 1-cos(t)', 'grafico', 'parametrica')
C(5, GRA, 'graficar 3d sen(x)*cos(y)', 'grafico', '3d')
C(5, GRA, 'graficar 3d e^(-x^2-y^2)', 'grafico', '3d')


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------
MODOS = {'expr', 'antiderivada', 'numero', 'conjunto', 'dict', 'matriz', 'vector',
         'intervalo', 'edo_general', 'clave', 'grafico'}


def main():
    solo = sys.argv[1:]  # opcional: filtrar por texto de entrada (depuración)
    fallos = []
    casos = [c for c in CASES if not solo or any(q in c['entrada'] for q in solo)]
    for c in casos:
        try:
            ok, msg = verificar(c)
        except Exception as ex:  # noqa: BLE001
            ok, msg = False, 'excepción: %r' % (ex,)
        if not ok:
            fallos.append((c, msg))
            print('FALLA  [%s] %s\n   esperado: %s\n   %s' % (c['tema'], c['entrada'], c['esperado'], msg))
    if solo:
        print('%d casos filtrados, %d fallos' % (len(casos), len(fallos)))
        return 1 if fallos else 0

    # Chequeos estructurales del banco
    errores = []
    if len(CASES) != 500:
        errores.append('hay %d casos (se esperaban 500)' % len(CASES))
    entradas = [c['entrada'] for c in CASES]
    dup = {e for e in entradas if entradas.count(e) > 1}
    if dup:
        errores.append('entradas duplicadas: %s' % sorted(dup))
    for c in CASES:
        if c['modo'] not in MODOS:
            errores.append('modo inválido en %s' % c['entrada'])
        if c['modo'] == 'numero' and 'tol' not in c:
            errores.append('falta tol en %s' % c['entrada'])
        if c['modo'] == 'clave' and 'clave' not in c:
            errores.append('falta clave en %s' % c['entrada'])
        if c['modo'] == 'edo_general' and 'edo' not in c:
            errores.append('falta edo en %s' % c['entrada'])
        if c['modo'] != 'grafico' and c['esperado'] is not None:
            S_(c['esperado'])  # debe ser sympificable
    for e in errores:
        print('ERROR:', e)
    if fallos or errores:
        print('\n%d fallos de verificación, %d errores estructurales' % (len(fallos), len(errores)))
        return 1

    # Orden por dificultad, mezclando temas dentro de cada dificultad
    vistos = {}
    for c in CASES:
        key = (c['tema'], c['d'])
        c['_rank'] = vistos.get(key, 0)
        vistos[key] = c['_rank'] + 1
    orden = sorted(CASES, key=lambda c: (c['d'], c['_rank'], TEMAS.index(c['tema'])))
    salida = []
    for i, c in enumerate(orden):
        obj = {'id': i + 1, 'nivel': i // 50 + 1, 'tema': c['tema'],
               'entrada': c['entrada'], 'modo': c['modo'], 'esperado': c['esperado']}
        for extra in ('tol', 'clave', 'edo'):
            if extra in c:
                obj[extra] = c[extra]
        salida.append(obj)
    destino = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'banco_500.json')
    with open(destino, 'w', encoding='utf-8') as fh:
        json.dump(salida, fh, ensure_ascii=False, indent=1)

    from collections import Counter
    print('OK: %d casos verificados -> %s' % (len(salida), destino))
    print('Por tema:', dict(Counter(o['tema'] for o in salida)))
    print('Por nivel:', dict(sorted(Counter(o['nivel'] for o in salida).items())))
    print('Por modo:', dict(Counter(o['modo'] for o in salida)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
