"""Pruebas de motor/ingecalc/calculo_extra.py (estudio de funciones, aplicaciones de la integral, series,
cálculo vectorial, límites dobles, sistemas de EDO).

Los valores esperados se calculan aparte con SymPy directo (no con el módulo probado).
Uso (desde la carpeta del proyecto):  PYTHONIOENCODING=utf-8 python pruebas/prueba_calculo_extra.py"""
import os
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..', 'motor'))

import sympy as sp  # noqa: E402
from sympy.calculus.util import continuous_domain  # noqa: E402

from ingecalc.api import resolver  # noqa: E402

x, y, z, t, n = sp.symbols('x y z t n')
xr = sp.Symbol('x', real=True)
C1, C2 = sp.symbols('C1 C2')
oo, pi = sp.oo, sp.pi


# ----------------------------------------------------------------------------
# Comparadores
# ----------------------------------------------------------------------------
def leer_valor(d):
    v = d.get('valor')
    if isinstance(v, str):
        try:
            return sp.sympify(v, locals={'E': sp.E, 'x': x, 'y': y, 'z': z, 't': t, 'n': n})
        except Exception:  # noqa
            return v
    return v


def iguales(a, b, tol=1e-9):
    a, b = sp.sympify(a), sp.sympify(b)
    if a == b:
        return True
    try:
        if sp.simplify(a - b) == 0:
            return True
    except Exception:  # noqa
        pass
    libres = sorted((a - b).free_symbols, key=str)
    for val in ([], [sp.Rational(7, 10)], [sp.Rational(13, 10)], [sp.Rational(23, 10)]) if libres else ([],):
        sub = {s_: val[0] for s_ in libres} if val else {}
        try:
            d = complex(sp.N((a - b).subs(sub), 20))
            if abs(d) > tol * (1 + abs(complex(sp.N(b.subs(sub), 20)))):
                return False
        except Exception:  # noqa
            return False
    return True


def mismo_conjunto(lista, esperado):
    lista, esperado = list(lista), list(esperado)
    if len(lista) != len(esperado):
        return False
    pendientes = list(esperado)
    for a in lista:
        i = next((j for j, b in enumerate(pendientes) if iguales(a, b)), None)
        if i is None:
            return False
        pendientes.pop(i)
    return True


def mismo_intervalo(a, b):
    a, b = sp.sympify(a), sp.sympify(b)
    return a == b or sp.simplify(sp.SymmetricDifference(a, b)) == sp.EmptySet


# ----------------------------------------------------------------------------
# Esperados calculados con SymPy directo
# ----------------------------------------------------------------------------
def esp_asintotas_racional():
    f = (x ** 2 + 1) / (x - 1)
    assert sp.limit(f, x, 1, '+') == oo
    m = sp.limit(f / x, x, oo)
    b = sp.limit(f - m * x, x, oo)
    return [sp.Eq(x, 1), sp.Eq(y, m * x + b)]


def esp_inflexion(f):
    d2 = sp.diff(f, x, 2)
    return [c for c in sp.solve(d2, x) if c.is_real]


def esp_normal(f, a):
    fa = f.subs(x, a)
    m = sp.diff(f, x).subs(x, a)
    return sp.expand(fa - (x - a) / m)


def esp_lagrange(f, a, b):
    m = (f.subs(x, b) - f.subs(x, a)) / (b - a)
    return [c for c in sp.solve(sp.diff(f, x) - m, x) if c.is_real and a < c < b]


def esp_area_polar(rr, a, b):
    th = sp.Symbol('t')
    return sp.integrate(rr.subs(x, th) ** 2 / 2, (th, a, b))


def esp_arco(f, a, b):
    return sp.integrate(sp.sqrt(1 + sp.diff(f, x) ** 2), (x, a, b))


def esp_suma_conv(a):
    nn = sp.Symbol('n', integer=True, positive=True)
    return 'converge' if sp.Sum(a.subs(n, nn), (nn, 1, oo)).is_convergent() else 'diverge'


def esp_direccional(f, P, u):
    g = [sp.diff(f, v).subs({x: P[0], y: P[1]}) for v in (x, y)]
    nu = sp.sqrt(u[0] ** 2 + u[1] ** 2)
    return (g[0] * u[0] + g[1] * u[1]) / nu


def esp_linea(F, R, a, b):
    sub = {x: R[0], y: R[1]}
    return sp.integrate(sum(Fi.subs(sub) * sp.diff(Ri, t) for Fi, Ri in zip(F, R)), (t, a, b))


def esp_curvatura_3d(R):
    tr = sp.Symbol('t', real=True)
    R = [sp.sympify(c).subs(t, tr) for c in R]
    d1 = sp.Matrix([sp.diff(c, tr) for c in R])
    d2 = sp.Matrix([sp.diff(c, tr, 2) for c in R])
    return sp.simplify(d1.cross(d2).norm() / d1.norm() ** 3)


def verifica_potencial(F):
    def chk(phi):
        return all(sp.simplify(sp.diff(phi, v) - Fi) == 0 for v, Fi in zip((x, y, z), F))
    return chk


def verifica_sistema(A, ics=None):
    def chk(sol):
        X, Y = sol[x], sol[y]
        ok = sp.simplify(sp.diff(X, t) - (A[0][0] * X + A[0][1] * Y)) == 0 and \
            sp.simplify(sp.diff(Y, t) - (A[1][0] * X + A[1][1] * Y)) == 0
        if ics:
            ok = ok and sp.simplify(X.subs(t, 0) - ics[0]) == 0 and sp.simplify(Y.subs(t, 0) - ics[1]) == 0
        else:
            ok = ok and {C1, C2} <= (X + Y).free_symbols
        return ok
    return chk


def verifica_ortogonales(familia_pend):
    """familia_pend(x, y) = pendiente de la familia original; la solución debe tener pendiente -1/esa."""
    def chk(sol):
        yy = sol[0] if isinstance(sol, list) else sol
        pend = sp.diff(yy, x)
        return sp.simplify(pend * familia_pend(x, yy) + 1) == 0
    return chk


# ----------------------------------------------------------------------------
# Casos: (entrada, tipo, esperado)
#   tipo: 'expr' | 'conj' | 'texto' | 'interv' | 'fn' (función que recibe el valor) | 'error' (debe fallar con mensaje)
#         'comp' (esperado = (respuesta, correcto))
# ----------------------------------------------------------------------------
f_rac = (x ** 2 + 1) / (x - 1)
CASOS = [
    # --- Análisis I
    ('asintotas (x^2+1)/(x-1)', 'conj', esp_asintotas_racional()),
    ('asintotas 1/(x^2-4)', 'conj', [sp.Eq(x, -2), sp.Eq(x, 2), sp.Eq(y, sp.limit(1 / (x ** 2 - 4), x, oo))]),
    ('asintotas e^x', 'conj', [sp.Eq(y, sp.limit(sp.exp(x), x, -oo))]),
    ('asintotas sqrt(x^2+1)', 'conj', [sp.Eq(y, x), sp.Eq(y, -x)]),
    ('asintotas ln(x)', 'conj', [sp.Eq(x, 0)]),
    ('inflexion x^3-3x^2', 'conj', esp_inflexion(x ** 3 - 3 * x ** 2)),
    ('concavidad x^4-6x^2', 'conj', esp_inflexion(x ** 4 - 6 * x ** 2)),
    ('puntos de inflexion x*e^(-x)', 'conj', esp_inflexion(x * sp.exp(-x))),
    ('dominio sqrt(4-x^2)/ln(x)', 'interv', continuous_domain(sp.sqrt(4 - xr ** 2) / sp.log(xr), xr, sp.S.Reals)),
    ('dominio ln(x^2-1)', 'interv', continuous_domain(sp.log(xr ** 2 - 1), xr, sp.S.Reals)),
    ('dominio (x+1)/(x^2-9)', 'interv', continuous_domain((xr + 1) / (xr ** 2 - 9), xr, sp.S.Reals)),
    ('estudio completo x^3-3x', 'conj', [c for c in sp.solve(sp.diff(x ** 3 - 3 * x, x), x)]),
    ('estudio completo x*e^(-x)', 'conj', sp.solve(sp.diff(x * sp.exp(-x), x), x)),
    ('estudio completo (x^2+1)/(x-1)', 'conj', sp.solve(sp.diff(f_rac, x), x)),
    ('estudio completo x^(2/3)', 'conj', [0]),
    ('recta normal x^2, 1', 'expr', esp_normal(x ** 2, 1)),
    ('recta normal sen(x), 0', 'expr', esp_normal(sp.sin(x), 0)),
    ('diferencial x^3, 2, 0.1', 'expr', sp.diff(x ** 3, x).subs(x, 2) * sp.Rational(1, 10)),
    ('aproximar sqrt(4.1)', 'expr', sp.sqrt(4) + sp.diff(sp.sqrt(x), x).subs(x, 4) * sp.Rational(1, 10)),
    ('aproximar cbrt(26.9)', 'expr', 3 + sp.diff(x ** sp.Rational(1, 3), x).subs(x, 27) * sp.Rational(-1, 10)),
    ('aproximar e^0.1', 'expr', 1 + sp.Rational(1, 10)),
    ('linealizar sqrt(x), 4', 'expr', 2 + (x - 4) / 4),
    ('valor medio x^3, 0, 2', 'conj', esp_lagrange(x ** 3, 0, 2)),
    ('valor medio x^2+2x, 0, 4', 'conj', esp_lagrange(x ** 2 + 2 * x, 0, 4)),
    ('rolle x^2-4x, 0, 4', 'conj', [c for c in sp.solve(sp.diff(x ** 2 - 4 * x, x), x)]),
    ('funcion inversa (2x+3)/(x-1)', 'expr', sp.solve(sp.Eq(y, (2 * x + 3) / (x - 1)), x)[0].subs(y, x)),
    ('funcion inversa e^(2x)-1', 'expr', sp.log(x + 1) / 2),
    ('funcion inversa sen(x)', 'expr', sp.asin(x)),
    ('funcion inversa x^2', 'expr', sp.sqrt(x)),
    # --- Análisis II: aplicaciones de la integral
    ('area polar 1+cos(t), 0, 2pi', 'expr', esp_area_polar(1 + sp.cos(x), 0, 2 * pi)),
    ('area polar 2*sen(t), 0, pi', 'expr', esp_area_polar(2 * sp.sin(x), 0, pi)),
    ('longitud de arco x^(3/2), 0, 4', 'expr', esp_arco(x ** sp.Rational(3, 2), 0, 4)),
    ('longitud de arco cosh(x), 0, 1', 'expr', esp_arco(sp.cosh(x), 0, 1)),
    ('longitud de arco (cos(t), sen(t)), 0, pi', 'expr', pi),
    ('longitud de arco (t^2, t^3), 0, 1', 'expr', sp.integrate(sp.sqrt((2 * t) ** 2 + (3 * t ** 2) ** 2), (t, 0, 1))),
    ('volumen de revolucion x^2, 0, 1', 'expr', pi * sp.integrate(x ** 4, (x, 0, 1))),
    ('volumen de revolucion x^2, 0, 1, eje y', 'expr', 2 * pi * sp.integrate(x * x ** 2, (x, 0, 1))),
    ('volumen de revolucion sqrt(x), x^2, 0, 1', 'expr', pi * sp.integrate(x - x ** 4, (x, 0, 1))),
    ('volumen de revolucion x, 0, 2, y=-1', 'expr', pi * sp.integrate((x + 1) ** 2, (x, 0, 2))),
    ('superficie de revolucion x, 0, 1', 'expr', 2 * pi * sp.integrate(x * sp.sqrt(2), (x, 0, 1))),
    ('superficie de revolucion sqrt(x), 0, 1', 'expr',
     2 * pi * sp.integrate(sp.sqrt(x) * sp.sqrt(1 + 1 / (4 * x)), (x, 0, 1))),
    ('derivada parametrica cos(t), sen(t)', 'expr', sp.diff(sp.sin(t), t) / sp.diff(sp.cos(t), t)),
    ('derivada parametrica t^2, t^3, 1', 'expr', (sp.diff(t ** 3, t) / sp.diff(t ** 2, t)).subs(t, 1)),
    # --- Series
    ('criterio serie n!/n^n', 'texto', esp_suma_conv(sp.factorial(n) / n ** n)),
    ('criterio serie 1/n', 'texto', esp_suma_conv(1 / n)),
    ('criterio serie 1/n^2', 'texto', esp_suma_conv(1 / n ** 2)),
    ('criterio serie (-1)^n/n', 'texto', 'converge'),
    ('criterio serie n/(n+1)', 'texto', 'diverge'),
    ('criterio serie 2^n/n!', 'texto', esp_suma_conv(2 ** n / sp.factorial(n))),
    ('criterio serie 1/(n*ln(n)), 2', 'texto', 'diverge'),
    ('criterio serie (n/(2n+1))^n', 'texto', 'converge'),
    ('criterio serie sen(n)/n^2', 'texto', 'converge'),
    ('criterio serie 1/sqrt(n)', 'texto', esp_suma_conv(1 / sp.sqrt(n))),
    ('radio de convergencia x^n/n', 'expr', sp.limit((1 / n) / (1 / (n + 1)), n, oo)),
    ('radio de convergencia (x-2)^n/(n*3^n)', 'expr', sp.limit((1 / (n * 3 ** n)) / (1 / ((n + 1) * 3 ** (n + 1))), n, oo)),
    ('radio de convergencia x^n/n!', 'expr', oo),
    ('radio de convergencia n!*x^n', 'expr', 0),
    # --- Análisis III
    ('derivada direccional x^2*y, (1,2), (3,4)', 'expr', esp_direccional(x ** 2 * y, (1, 2), (3, 4))),
    ('derivada direccional x*y*z, (1,1,1), (1,2,2)', 'expr', sp.Rational(1 + 2 + 2, 3)),
    ('diferencial total x^2*y, (1,2), 0.01, 0.02', 'expr', 2 * 1 * 2 * sp.Rational(1, 100) + 1 * sp.Rational(2, 100)),
    ('limite doble x*y/(x^2+y^2), (x,y)->(0,0)', 'texto',
     'no existe' if sp.limit((x * y / (x ** 2 + y ** 2)).subs(y, x), x, 0) != 0 else '0'),
    ('limite doble x^2*y/(x^2+y^2), (x,y)->(0,0)', 'expr', 0),
    ('limite doble (x^2-y^2)/(x^2+y^2), (x,y)->(0,0)', 'texto', 'no existe'),
    ('limite doble x^2*y/(x^4+y^2), (x,y)->(0,0)', 'texto', 'no existe'),
    ('limite doble sen(x^2+y^2)/(x^2+y^2), (x,y)->(0,0)', 'expr', sp.limit(sp.sin(x) / x, x, 0)),
    ('limite doble x+y, (x,y)->(1,2)', 'expr', 3),
    ('limite doble x^2*y^2/(x^2*y^2+(x-y)^2), x->0, y->0', 'texto', 'no existe'),
    ('integral de linea (-y, x), (cos(t), sen(t)), 0, 2pi', 'expr', esp_linea((-y, x), (sp.cos(t), sp.sin(t)), 0, 2 * pi)),
    ('integral de linea (x^2, y^2), (cos(t), sen(t)), 0, pi', 'expr', esp_linea((x ** 2, y ** 2), (sp.cos(t), sp.sin(t)), 0, pi)),
    ('integral de linea x*y, (t, t^2), 0, 1', 'expr', sp.integrate(t ** 3 * sp.sqrt(1 + 4 * t ** 2), (t, 0, 1))),
    ('potencial (2x*y, x^2)', 'fn', verifica_potencial((2 * x * y, x ** 2))),
    ('potencial (y*z, x*z, x*y)', 'fn', verifica_potencial((y * z, x * z, x * y))),
    ('potencial (y, -x)', 'texto', 'no existe'),
    ('flujo (x, y, z), z=1-x-y, x, 0, 1, y, 0, 1-x', 'expr', sp.Rational(1, 2)),
    ('curvatura (cos(t), sen(t), t)', 'expr', esp_curvatura_3d([sp.cos(t), sp.sin(t), t])),
    ('curvatura (t, t^2), 1', 'expr', esp_curvatura_3d([t, t ** 2, 0]).subs(sp.Symbol('t', real=True), 1)),
    ('curvatura x^2, 0', 'expr', 2),
    ('funcion vectorial (cos(t), sen(t), t)', 'fn',
     lambda v: all(iguales(a, b) for a, b in zip(v, [-sp.sin(t) / sp.sqrt(2), sp.cos(t) / sp.sqrt(2), 1 / sp.sqrt(2)]))),
    ("sistema edo x'=x+y, y'=4x+y", 'fn', verifica_sistema([[1, 1], [4, 1]])),
    ("sistema edo x'=x+y, y'=4x+y; x(0)=1; y(0)=0", 'fn', verifica_sistema([[1, 1], [4, 1]], (1, 0))),
    ("sistema edo x'=-y, y'=x; x(0)=1; y(0)=0", 'fn', verifica_sistema([[0, -1], [1, 0]], (1, 0))),
    ("sistema edo x'=2x+y, y'=2y; x(0)=1; y(0)=1", 'fn', verifica_sistema([[2, 1], [0, 2]], (1, 1))),
    ('trayectorias ortogonales x^2+y^2=c', 'fn', verifica_ortogonales(lambda a, b: -a / b)),
    ('trayectorias ortogonales y=c*x^2', 'fn', verifica_ortogonales(lambda a, b: 2 * b / a)),
    # --- Comprobación de respuestas del alumno
    ('asintotas (x^2+1)/(x-1)', 'comp', ('x=1, y=x+1', True)),
    ('recta normal x^2, 1', 'comp', ('y = -x/2 + 3/2', True)),
    ('valor medio x^3, 0, 2', 'comp', ('2/sqrt(3)', True)),
    ('aproximar sqrt(4.1)', 'comp', ('2.025', True)),
    ('limite doble x*y/(x^2+y^2), (x,y)->(0,0)', 'comp', ('no existe', True)),
    ('trayectorias ortogonales x^2+y^2=c', 'comp', ('y = k*x', True)),
    ('area polar 1+cos(t), 0, 2pi', 'comp', ('3pi/2', True)),
    ('derivada direccional x^2*y, (1,2), (3,4)', 'comp', ('3', False)),
    # --- Errores de entrada (mensaje claro, no error interno)
    ('valor medio x^3, 0', 'error', None),
    ('limite doble x*y', 'error', None),
    ('sistema edo hola', 'error', None),
    ('radio de convergencia 1/n^2', 'error', None),
]


def correr():
    buenos = malos = 0
    lentos = []
    t_total = time.time()
    for entrada, tipo, esperado in CASOS:
        t0 = time.time()
        try:
            if tipo == 'comp':
                d = resolver(entrada, esperado[0])
                c = (d.get('comprobacion') or {}).get('correcto')
                ok = d.get('ok') and c is esperado[1]
                detalle = d.get('comprobacion')
            elif tipo == 'error':
                d = resolver(entrada)
                ok = not d.get('ok') and d.get('error') and 'interno' not in d.get('error', '') and 'revisá la escritura' \
                    not in d.get('error', '')
                detalle = d.get('error')
            else:
                d = resolver(entrada)
                if not d.get('ok'):
                    ok, detalle = False, (d.get('error'), (d.get('detalle') or '')[-300:])
                else:
                    v = leer_valor(d)
                    detalle = d.get('valor')
                    if tipo == 'expr':
                        ok = iguales(v, esperado)
                    elif tipo == 'conj':
                        ok = mismo_conjunto(v, esperado)
                    elif tipo == 'texto':
                        ok = str(d.get('valor')).strip() == str(esperado)
                    elif tipo == 'interv':
                        ok = mismo_intervalo(v, esperado)
                    elif tipo == 'fn':
                        ok = bool(esperado(v))
                    else:
                        ok = False
        except Exception as e:  # noqa
            ok, detalle, d = False, repr(e), {}
        seg = time.time() - t0
        if seg > 10:
            lentos.append((entrada, round(seg, 1)))
        if ok:
            buenos += 1
        else:
            malos += 1
        print(('OK  ' if ok else 'MAL ') + f'{seg:5.2f}s  {entrada}  =>  {detalle}' + ('' if ok else f'   (esperado: {esperado})'))
    print(f'\nResumen: {buenos} OK, {malos} MAL de {len(CASOS)} casos; tiempo total {time.time() - t_total:.1f} s')
    if lentos:
        print('Casos de más de 10 s:', lentos)
    return malos


if __name__ == '__main__':
    sys.exit(1 if correr() else 0)
