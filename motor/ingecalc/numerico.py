"""Cálculo Numérico: raíces, integración, EDO, interpolación y regresión, con tabla de iteraciones."""
import math

import sympy as sp

from . import graficos as G
from .nucleo import ErrorEntrada, Resultado, dividir, leer, tex, variable_principal, x, y, fmt_num, normalizar

MAX_ITER = 100


def _f(expr, *vs):
    return sp.lambdify(vs, expr, modules=['math'])


def _num(texto):
    v = leer(texto)
    try:
        return float(v)
    except Exception as exc:  # noqa
        raise ErrorEntrada(f'«{texto}» debería ser un número.') from exc


def _r(titulo):
    return Resultado('Cálculo numérico', titulo)


def _redondear(v, dec=10):
    return round(v, dec)


def _cierre(r, f, v, raiz, iteraciones, tol, a=None, b=None):
    r.valor = sp.Float(_redondear(raiz), 15)
    r.valores['iteraciones'] = iteraciones
    r.resultado_latex = tex(v) + r'\approx ' + fmt_num(raiz, 12)
    r.paso('Resultado', f'Se detiene cuando el error es menor que la tolerancia {tol:g} ({iteraciones} iteraciones).',
           tex(v) + r'\approx ' + fmt_num(raiz, 12), 0)
    r.comparar = ('numero', raiz, {'tol': 1e-6})
    try:
        ancho = max(abs(raiz) * 0.8, 2.0)
        r.grafico = G.grafico_2d([f], v, [G.punto(raiz, 0, f'raíz ≈ {fmt_num(raiz, 6)}')],
                                 xmin=(a if a is not None else raiz - ancho), xmax=(b if b is not None else raiz + ancho))
    except Exception:  # noqa
        pass


def newton(arg):
    partes = dividir(arg)
    if len(partes) < 2:
        raise ErrorEntrada('Uso: newton f, x0[, tolerancia]')
    f = leer(partes[0])
    v = variable_principal(f)
    x0 = _num(partes[1])
    tol = _num(partes[2]) if len(partes) > 2 else 1e-10
    df = sp.diff(f, v)
    r = _r('Método de Newton-Raphson')
    r.entrada_latex = 'f(' + tex(v) + ') = ' + tex(f) + r',\quad ' + tex(v) + '_0 = ' + fmt_num(x0)
    r.paso('Fórmula', '', tex(v) + r'_{n+1} = ' + tex(v) + r'_n - \frac{f(' + tex(v) + '_n)}{f\'(' + tex(v) + '_n)},\quad f\'(' + tex(v) + ') = ' + tex(df), 0)
    F, DF = _f(f, v), _f(df, v)
    filas = []
    xn = x0
    for i in range(MAX_ITER):
        fx, dfx = F(xn), DF(xn)
        if dfx == 0:
            raise ErrorEntrada(f"f'(x) = 0 en la iteración {i}: Newton no puede continuar; probá otro x0.")
        x1 = xn - fx / dfx
        err = abs(x1 - xn)
        filas.append([str(i), fmt_num(xn, 12), fmt_num(fx, 6), fmt_num(dfx, 6), fmt_num(x1, 12), fmt_num(err, 3)])
        xn = x1
        if err < tol:
            break
    else:
        r.avisos.append('No convergió en el máximo de iteraciones.')
    r.tabla = {'columnas': ['n', 'xₙ', 'f(xₙ)', "f'(xₙ)", 'xₙ₊₁', '|Δ|'], 'filas': filas}
    _cierre(r, f, v, xn, len(filas), tol)
    return r


def biseccion(arg):
    partes = dividir(arg)
    if len(partes) < 3:
        raise ErrorEntrada('Uso: biseccion f, a, b[, tolerancia]')
    f = leer(partes[0])
    v = variable_principal(f)
    a, b = _num(partes[1]), _num(partes[2])
    tol = _num(partes[3]) if len(partes) > 3 else 1e-10
    F = _f(f, v)
    r = _r('Método de bisección')
    r.entrada_latex = 'f(' + tex(v) + ') = ' + tex(f) + r',\quad [' + fmt_num(a) + ', ' + fmt_num(b) + ']'
    fa, fb = F(a), F(b)
    if fa * fb > 0:
        raise ErrorEntrada(f'f(a) = {fmt_num(fa, 6)} y f(b) = {fmt_num(fb, 6)} tienen el mismo signo: Bolzano no garantiza raíz.')
    r.paso('Teorema de Bolzano', 'f es continua y cambia de signo en [a, b]: hay al menos una raíz.',
           f'f({fmt_num(a)}) = {fmt_num(fa, 6)},\\ f({fmt_num(b)}) = {fmt_num(fb, 6)}', 0)
    nec = math.ceil(math.log2((b - a) / tol)) if tol > 0 else 0
    r.paso('Iteraciones necesarias', r'\(n \ge \log_2\frac{b-a}{\varepsilon}\)', f'n \\ge {nec}', 0)
    filas = []
    for i in range(200):
        c = (a + b) / 2
        fc = F(c)
        filas.append([str(i + 1), fmt_num(a, 12), fmt_num(b, 12), fmt_num(c, 12), fmt_num(fc, 6), fmt_num((b - a) / 2, 3)])
        if fc == 0 or (b - a) / 2 < tol:
            break
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    r.tabla = {'columnas': ['n', 'a', 'b', 'c', 'f(c)', 'error'], 'filas': filas[:60]}
    if len(filas) > 60:
        r.avisos.append(f'Se muestran 60 de {len(filas)} iteraciones.')
    _cierre(r, f, v, c, len(filas), tol)
    return r


def secante(arg):
    partes = dividir(arg)
    if len(partes) < 3:
        raise ErrorEntrada('Uso: secante f, x0, x1[, tolerancia]')
    f = leer(partes[0])
    v = variable_principal(f)
    x0, x1 = _num(partes[1]), _num(partes[2])
    tol = _num(partes[3]) if len(partes) > 3 else 1e-10
    F = _f(f, v)
    r = _r('Método de la secante')
    r.entrada_latex = 'f(' + tex(v) + ') = ' + tex(f) + r',\quad x_0 = ' + fmt_num(x0) + r',\ x_1 = ' + fmt_num(x1)
    r.paso('Fórmula', 'Newton con la derivada aproximada por la pendiente de la secante.',
           r'x_{n+1} = x_n - f(x_n)\frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}', 0)
    filas = []
    for i in range(MAX_ITER):
        f0, f1 = F(x0), F(x1)
        if f1 == f0:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        err = abs(x2 - x1)
        filas.append([str(i + 1), fmt_num(x0, 12), fmt_num(x1, 12), fmt_num(x2, 12), fmt_num(err, 3)])
        x0, x1 = x1, x2
        if err < tol:
            break
    r.tabla = {'columnas': ['n', 'xₙ₋₁', 'xₙ', 'xₙ₊₁', '|Δ|'], 'filas': filas}
    _cierre(r, f, v, x1, len(filas), tol)
    return r


def punto_fijo(arg):
    partes = dividir(arg)
    if len(partes) < 2:
        raise ErrorEntrada('Uso: punto fijo g, x0[, tolerancia]   (itera x = g(x))')
    g = leer(partes[0])
    v = variable_principal(g)
    x0 = _num(partes[1])
    tol = _num(partes[2]) if len(partes) > 2 else 1e-10
    Gf = _f(g, v)
    dg = sp.diff(g, v)
    r = _r('Método de punto fijo')
    r.entrada_latex = tex(v) + ' = g(' + tex(v) + ') = ' + tex(g) + r',\quad x_0 = ' + fmt_num(x0)
    try:
        cota = abs(float(dg.subs(v, x0)))
        r.paso('Condición de convergencia', r'Converge si \(|g\'(x)| < 1\) cerca del punto fijo.',
               r"|g'(x_0)| = " + fmt_num(cota, 6) + (' < 1' if cota < 1 else r' \ge 1\ (\text{puede no converger})'), 0)
    except Exception:  # noqa
        pass
    filas = []
    xn = x0
    for i in range(500):
        x1 = Gf(xn)
        err = abs(x1 - xn)
        if i < 80:
            filas.append([str(i), fmt_num(xn, 12), fmt_num(x1, 12), fmt_num(err, 3)])
        xn = x1
        if err < tol:
            break
        if not math.isfinite(xn) or abs(xn) > 1e100:
            raise ErrorEntrada('La iteración diverge: |g\'(x)| ≥ 1. Probá despejar x de otra forma.')
    r.tabla = {'columnas': ['n', 'xₙ', 'g(xₙ)', '|Δ|'], 'filas': filas}
    _cierre(r, g - v, v, xn, i + 1, tol)
    try:
        r.grafico = G.grafico_2d([g, v], v, [G.punto(xn, xn, f'punto fijo ≈ {fmt_num(xn, 6)}')], etiquetas=['g(x)', 'y = x'],
                                 xmin=xn - 3, xmax=xn + 3)
    except Exception:  # noqa
        pass
    return r


def trapecio(arg):
    partes = dividir(arg)
    if len(partes) < 4:
        raise ErrorEntrada('Uso: trapecio f, a, b, n')
    f = leer(partes[0])
    v = variable_principal(f)
    a, b, nn = _num(partes[1]), _num(partes[2]), int(_num(partes[3]))
    F = _f(f, v)
    h = (b - a) / nn
    r = _r('Regla del trapecio compuesta')
    r.entrada_latex = r'\int_{' + fmt_num(a) + '}^{' + fmt_num(b) + '} ' + tex(f) + r'\,d' + tex(v) + r',\quad n = ' + str(nn)
    r.paso('Fórmula', '', r'T = \frac{h}{2}\left[f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n)\right],\quad h = \frac{b-a}{n} = ' + fmt_num(h, 10), 0)
    filas, total = [], 0.0
    for i in range(nn + 1):
        xi = a + i * h
        fi = F(xi)
        c = 1 if i in (0, nn) else 2
        total += c * fi
        if i <= 60:
            filas.append([str(i), fmt_num(xi, 10), fmt_num(fi, 10), str(c), fmt_num(c * fi, 10)])
    res = h / 2 * total
    r.tabla = {'columnas': ['i', 'xᵢ', 'f(xᵢ)', 'coef.', 'coef·f'], 'filas': filas}
    r.paso('Suma', '', r'T = \frac{' + fmt_num(h, 10) + r'}{2}\cdot ' + fmt_num(total, 12) + ' = ' + fmt_num(res, 12), 0)
    _cierre_integral(r, f, v, a, b, res)
    return r


def simpson(arg):
    partes = dividir(arg)
    if len(partes) < 4:
        raise ErrorEntrada('Uso: simpson f, a, b, n   (n par)')
    f = leer(partes[0])
    v = variable_principal(f)
    a, b, nn = _num(partes[1]), _num(partes[2]), int(_num(partes[3]))
    if nn % 2:
        raise ErrorEntrada('Simpson 1/3 necesita una cantidad PAR de subintervalos.')
    F = _f(f, v)
    h = (b - a) / nn
    r = _r('Regla de Simpson 1/3 compuesta')
    r.entrada_latex = r'\int_{' + fmt_num(a) + '}^{' + fmt_num(b) + '} ' + tex(f) + r'\,d' + tex(v) + r',\quad n = ' + str(nn)
    r.paso('Fórmula', '', r'S = \frac{h}{3}\left[f_0 + 4\sum f_{impares} + 2\sum f_{pares} + f_n\right],\quad h = ' + fmt_num(h, 10), 0)
    filas, total = [], 0.0
    for i in range(nn + 1):
        xi = a + i * h
        fi = F(xi)
        c = 1 if i in (0, nn) else (4 if i % 2 else 2)
        total += c * fi
        if i <= 60:
            filas.append([str(i), fmt_num(xi, 10), fmt_num(fi, 10), str(c), fmt_num(c * fi, 10)])
    res = h / 3 * total
    r.tabla = {'columnas': ['i', 'xᵢ', 'f(xᵢ)', 'coef.', 'coef·f'], 'filas': filas}
    r.paso('Suma', '', r'S = \frac{' + fmt_num(h, 10) + r'}{3}\cdot ' + fmt_num(total, 12) + ' = ' + fmt_num(res, 12), 0)
    _cierre_integral(r, f, v, a, b, res)
    return r


def _cierre_integral(r, f, v, a, b, res):
    r.valor = sp.Float(res, 15)
    r.resultado_latex = r'\approx ' + fmt_num(res, 12)
    try:
        exacto = sp.integrate(f, (v, a, b))
        if exacto.is_number and not exacto.has(sp.Integral):
            ex = float(exacto)
            r.extra('Comparación con el valor exacto', fmt_num(ex, 12), f'Error absoluto: {fmt_num(abs(ex - res), 4)}')
    except Exception:  # noqa
        pass
    r.comparar = ('numero', res, {'tol': 1e-8})
    try:
        g = G.grafico_2d([f], v, xmin=a - 0.2 * (b - a), xmax=b + 0.2 * (b - a))
        g['sombra'] = G.sombra_entre(f, sp.Integer(0), a, b, v)
        r.grafico = g
    except Exception:  # noqa
        pass


def _leer_edo1(texto):
    s_ = normalizar(texto)
    if '=' in s_:
        izq, der = s_.split('=', 1)
        if "'" not in izq and 'dy' not in izq:
            raise ErrorEntrada("Escribí la EDO despejada: y' = f(x, y)")
        s_ = der
    return leer(s_)


def _edo_num(arg, metodo):
    partes = dividir(arg)
    if len(partes) < 5:
        raise ErrorEntrada(f"Uso: {metodo} y'=f(x,y), x0, y0, h, xf")
    f = _leer_edo1(partes[0])
    tv = sp.Symbol('t')
    v = tv if f.has(tv) and not f.has(x) else x
    x0, y0, h, xf = (_num(p) for p in partes[1:5])
    F = _f(f, v, y)
    pasos_n = int(round((xf - x0) / h))
    nombres = {'euler': 'Método de Euler', 'rk4': 'Runge-Kutta de orden 4', 'heun': 'Método de Heun (Euler mejorado)'}
    r = _r(nombres[metodo])
    r.entrada_latex = r"y' = " + tex(f) + r',\quad y(' + fmt_num(x0) + ') = ' + fmt_num(y0) + r',\quad h = ' + fmt_num(h) + r',\ ' + tex(v) + r'_f = ' + fmt_num(xf)
    if metodo == 'euler':
        r.paso('Fórmula', '', r'y_{n+1} = y_n + h\,f(x_n, y_n)', 0)
        cols = ['n', 'xₙ', 'yₙ', 'f(xₙ,yₙ)', 'yₙ₊₁']
    elif metodo == 'heun':
        r.paso('Fórmula', '', r'y^*_{n+1} = y_n + h f(x_n,y_n),\ y_{n+1} = y_n + \frac{h}{2}\left[f(x_n,y_n) + f(x_{n+1}, y^*_{n+1})\right]', 0)
        cols = ['n', 'xₙ', 'yₙ', 'predictor', 'yₙ₊₁']
    else:
        r.paso('Fórmulas', '', r'k_1 = f(x_n,y_n),\ k_2 = f(x_n+\tfrac h2, y_n+\tfrac h2 k_1),\ k_3 = f(x_n+\tfrac h2, y_n+\tfrac h2 k_2),\ '
               r'k_4 = f(x_n+h, y_n+h k_3),\ y_{n+1} = y_n + \tfrac h6(k_1+2k_2+2k_3+k_4)', 0)
        cols = ['n', 'xₙ', 'yₙ', 'k₁', 'k₂', 'k₃', 'k₄', 'yₙ₊₁']
    filas = []
    xn, yn = x0, y0
    pts = [(xn, yn)]
    for i in range(pasos_n):
        if metodo == 'euler':
            k1 = F(xn, yn)
            y1 = yn + h * k1
            fila = [str(i), fmt_num(xn, 8), fmt_num(yn, 10), fmt_num(k1, 10), fmt_num(y1, 10)]
        elif metodo == 'heun':
            k1 = F(xn, yn)
            yp = yn + h * k1
            y1 = yn + h / 2 * (k1 + F(xn + h, yp))
            fila = [str(i), fmt_num(xn, 8), fmt_num(yn, 10), fmt_num(yp, 10), fmt_num(y1, 10)]
        else:
            k1 = F(xn, yn)
            k2 = F(xn + h / 2, yn + h / 2 * k1)
            k3 = F(xn + h / 2, yn + h / 2 * k2)
            k4 = F(xn + h, yn + h * k3)
            y1 = yn + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            fila = [str(i), fmt_num(xn, 8), fmt_num(yn, 10), fmt_num(k1, 8), fmt_num(k2, 8), fmt_num(k3, 8), fmt_num(k4, 8), fmt_num(y1, 10)]
        if i < 100:
            filas.append(fila)
        xn, yn = x0 + (i + 1) * h, y1
        pts.append((xn, yn))
    r.tabla = {'columnas': cols, 'filas': filas}
    r.valor = sp.Float(yn, 15)
    r.resultado_latex = 'y(' + fmt_num(xf) + r') \approx ' + fmt_num(yn, 12)
    r.paso('Resultado', f'{pasos_n} pasos de tamaño h = {fmt_num(h)}.', r.resultado_latex, 0)
    # solución exacta para comparar
    try:
        Y = sp.Function('y')
        sol = sp.dsolve(sp.Eq(Y(v).diff(v), f.subs(y, Y(v))), Y(v), ics={Y(x0 if True else 0): y0})
        exacta = sol.rhs
        ex = float(exacta.subs(v, xf))
        r.extra('Solución exacta', 'y = ' + tex(exacta), f'y({fmt_num(xf)}) = {fmt_num(ex, 12)}; error = {fmt_num(abs(ex - yn), 4)}')
        r.grafico = G.grafico_datos(pts, exacta, 'exacta')
    except Exception:  # noqa
        r.grafico = G.grafico_datos(pts)
    r.comparar = ('numero', yn, {'tol': 1e-8})
    return r


def euler(arg):
    return _edo_num(arg, 'euler')


def rk4(arg):
    return _edo_num(arg, 'rk4')


def heun(arg):
    return _edo_num(arg, 'heun')


def _puntos(arg):
    pts = []
    for p in dividir(arg):
        e = leer(p)
        if not isinstance(e, (sp.Tuple, tuple, list)) or len(e) != 2:
            raise ErrorEntrada('Escribí los puntos como (x, y), separados por comas.')
        pts.append((sp.nsimplify(e[0]), sp.nsimplify(e[1])))
    return pts


def interpolar(arg):
    pts = _puntos(arg)
    r = _r('Interpolación de Lagrange')
    r.entrada_latex = ', '.join(f'({tex(a)}, {tex(b)})' for a, b in pts)
    L = 0
    for i, (xi, yi) in enumerate(pts):
        li = 1
        for j, (xj, _) in enumerate(pts):
            if i != j:
                li *= (x - xj) / (xi - xj)
        r.paso(f'L_{i}(x)', '', f'L_{{{i}}}(x) = ' + tex(sp.factor(li)), 1)
        L += yi * li
    P = sp.expand(L)
    r.paso('Polinomio', r'\(P(x) = \sum y_i L_i(x)\)', 'P(x) = ' + tex(P), 0)
    # diferencias divididas
    n_ = len(pts)
    tabla = [[pts[i][1]] for i in range(n_)]
    for j in range(1, n_):
        for i in range(n_ - j):
            tabla[i].append((tabla[i + 1][j - 1] - tabla[i][j - 1]) / (pts[i + j][0] - pts[i][0]))
    r.tabla = {'columnas': ['xᵢ', 'yᵢ'] + [f'Δ^{j}' for j in range(1, n_)],
               'filas': [[fmt_num(pts[i][0]), fmt_num(tabla[i][0])] + [fmt_num(v) for v in tabla[i][1:]] for i in range(n_)]}
    r.extra('Forma de Newton (diferencias divididas)', 'P(x) = ' + ' + '.join(
        tex(tabla[0][j]) + ''.join(f'(x - {tex(pts[i][0])})' for i in range(j)) for j in range(n_) if tabla[0][j] != 0))
    r.valor = P
    r.resultado_latex = 'P(x) = ' + tex(P)
    r.comparar = ('expr', P, {})
    r.grafico = G.grafico_datos(pts, P, 'P(x)')
    return r


def regresion(arg):
    pts = _puntos(arg)
    n_ = len(pts)
    if n_ < 2:
        raise ErrorEntrada('Se necesitan al menos 2 puntos.')
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    sx, sy = sum(xs), sum(ys)
    sxx = sum(a * a for a in xs)
    sxy = sum(a * b for a, b in pts)
    syy = sum(b * b for b in ys)
    r = _r('Regresión lineal (mínimos cuadrados)')
    r.entrada_latex = ', '.join(f'({tex(a)}, {tex(b)})' for a, b in pts)
    a = (n_ * sxy - sx * sy) / (n_ * sxx - sx ** 2)
    b = (sy - a * sx) / n_
    r.tabla = {'columnas': ['xᵢ', 'yᵢ', 'xᵢ²', 'xᵢyᵢ'], 'filas': [[fmt_num(p), fmt_num(q), fmt_num(p * p), fmt_num(p * q)] for p, q in pts]
               + [['Σ ' + fmt_num(sx), fmt_num(sy), fmt_num(sxx), fmt_num(sxy)]]}
    r.paso('Ecuaciones normales', '', r'a = \frac{n\sum xy - \sum x\sum y}{n\sum x^2 - (\sum x)^2} = ' + tex(sp.nsimplify(a))
           + r',\quad b = \frac{\sum y - a\sum x}{n} = ' + tex(sp.nsimplify(b)), 0)
    recta = sp.nsimplify(a) * x + sp.nsimplify(b)
    rnum = n_ * sxy - sx * sy
    rden = sp.sqrt((n_ * sxx - sx ** 2) * (n_ * syy - sy ** 2))
    rr = rnum / rden if rden != 0 else 1
    r2 = sp.N(rr ** 2, 12)
    r.paso('Coeficiente de determinación', 'Cuanto más cerca de 1, mejor ajusta la recta.', f'r = {fmt_num(rr, 8)},\\ R^2 = {fmt_num(r2, 8)}', 0)
    r.valor = recta
    r.valores['r2'] = fmt_num(r2, 12)
    r.valores['r'] = fmt_num(sp.N(rr), 12)
    r.resultado_latex = 'y = ' + tex(recta) + r',\quad R^2 = ' + fmt_num(r2, 6)
    r.comparar = ('expr_num', recta, {'tol': 1e-6})
    r.grafico = G.grafico_datos(pts, recta, 'ajuste')
    return r
