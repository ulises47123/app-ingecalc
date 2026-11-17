"""Pruebas del módulo geo_lineal (geometría analítica, álgebra lineal y discreta).
Los esperados se calculan de forma independiente con SymPy (geometry, ntheory, logic, Matrix).
Uso: python pruebas/prueba_geo_lineal.py"""
import os
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..', 'motor'))

import sympy as sp  # noqa: E402
from sympy import Matrix, Point3D, Plane, Line3D, Point2D, Line2D  # noqa: E402
from sympy.logic.boolalg import And, Or, Not, Implies, Equivalent, Xor  # noqa: E402
from sympy.ntheory.modular import crt  # noqa: E402

from ingecalc.api import _detectar, resolver  # noqa: E402

x, y, z, t, k = sp.symbols('x y z t k')
LOC = {'x': x, 'y': y, 'z': z, 't': t, 'k': k, 'I': sp.I, 'pi': sp.pi, 'sqrt': sp.sqrt, 'Eq': sp.Eq}


def S(v):
    if isinstance(v, (int, float)):
        return sp.nsimplify(v)
    return sp.sympify(v, locals=LOC)


def igual(a, b):
    try:
        return sp.simplify(S(a) - S(b)) == 0
    except Exception:  # noqa
        return False


def igual_lista(a, b):
    a, b = list(S(a)), list(b)
    return len(a) == len(b) and all(igual(p, q) for p, q in zip(a, b))


def paralelos(u, v):
    return Matrix.hstack(Matrix(u), Matrix(v)).rank(simplify=True) <= 1


# ---------------------------------------------------------------------------
# Verificadores: reciben el dict de resolver() y devuelven True/False
# ---------------------------------------------------------------------------
def v_valor(esperado):
    return lambda d: igual(d['valor'], esperado)


def v_lista(esperado):
    return lambda d: igual_lista(d['valor'], esperado)


def v_texto(esperado):
    return lambda d: str(d['valor']).startswith(esperado)


def v_clave(clave, esperado):
    return lambda d: igual(d['valores'][clave], esperado)


def v_clave_lista(clave, esperado):
    def f(d):
        val = S(d['valores'][clave])
        if esperado and isinstance(esperado[0], (list, tuple)):
            return len(val) == len(esperado) and all(any(igual_lista(a, b) for a in val) for b in esperado)
        return igual_lista(val, esperado)
    return f


def v_todos(*fs):
    return lambda d: all(f(d) for f in fs)


def v_recta3d(P, D):
    """La paramétrica devuelta pasa por P y tiene dirección paralela a D."""
    def f(d):
        vec = Matrix(S(d['valor']))
        par = list(vec.free_symbols)[0]
        P0, D0 = vec.subs(par, 0), vec.diff(par)
        return paralelos(D0, D) and paralelos(Matrix(P) - P0, D)
    return f


def v_plano(normal, punto):
    def f(d):
        eq = S(d['valor'])
        e = eq.lhs - eq.rhs
        n = Matrix([e.coeff(v) for v in (x, y, z)])
        return paralelos(n, normal) and igual(e.subs(dict(zip((x, y, z), punto))), 0)
    return f


def v_base(A, dim, de_columnas=False):
    """Vectores devueltos: pertenecen al subespacio, son independientes y son dim."""
    def f(d):
        vecs = [Matrix(v) for v in S(d['valor'])]
        if len(vecs) != dim:
            return False
        if not vecs:
            return True
        if de_columnas:
            dentro = all(A.row_join(v).rank() == A.rank() for v in vecs)
        else:
            dentro = all(A * v == sp.zeros(A.rows, 1) for v in vecs)
        return dentro and Matrix.hstack(*vecs).rank() == dim
    return f


def v_diofantica(a, b, c):
    def f(d):
        val = d['valor']
        dic = {}
        for parte in val.strip('{}').split(','):
            kk, vv = parte.split(':')
            dic[S(kk)] = S(vv)
        ex, ey = dic[x], dic[y]
        g = sp.gcd(a, b)
        par = list(ex.free_symbols)[0]
        return sp.expand(a * ex + b * ey - c) == 0 and abs(sp.diff(ex, par)) == abs(b // g)
    return f


def clasificar(f):
    if sp.satisfiable(Not(f)) is False:
        return 'tautología'
    if sp.satisfiable(f) is False:
        return 'contradicción'
    return 'contingencia'


def v_booleana(esperado):
    def f(d):
        e = sp.sympify(d['valor'])
        return sp.satisfiable(Xor(e, esperado)) is False
    return f


# ---------------------------------------------------------------------------
# Esperados independientes
# ---------------------------------------------------------------------------
p, q, r_, a, b, c = sp.symbols('p q r a b c')

# Geometría con sympy.geometry
pl_123 = Plane(Point3D(1, 0, 0), Point3D(0, 1, 0), Point3D(0, 0, 1))
dist_pl = pl_123.distance(Point3D(1, 1, 1))
dist_pr = Line2D(Point2D(0, sp.Rational(5, 4)), slope=sp.Rational(-3, 4)).distance(Point2D(1, 2))
n1, n2 = Matrix([1, 1, 1]), Matrix([1, -1, 0])
ang_planos = sp.acos(abs(n1.dot(n2)) / (n1.norm() * n2.norm()))
inter_planos = Plane(Point3D(0, 0, 1), normal_vector=(1, 1, 1)).intersection(Plane(Point3D(0, 0, 0), normal_vector=(1, -1, 0)))[0]
L1 = Line3D(Point3D(0, 0, 0), direction_ratio=(1, 0, 0))
L2 = Line3D(Point3D(0, 1, 1), direction_ratio=(0, 1, 0))
w, cr = Matrix([0, 1, 1]), Matrix([1, 0, 0]).cross(Matrix([0, 1, 0]))
dist_alabeadas = abs(w.dot(cr)) / cr.norm()
inter_rp = Plane(Point3D(6, 0, 0), normal_vector=(1, 1, 1)).intersection(Line3D(Point3D(1, 0, 0), direction_ratio=(1, 1, 1)))[0]
dist_p_r3 = Line3D(Point3D(0, 0, 0), direction_ratio=(1, 1, 1)).distance(Point3D(1, 2, 3))

# Cónicas: completando cuadrados "a mano" con sympy
def elementos_elipse(A, C, D, E, F):
    h, k_ = -D / (2 * A), -E / (2 * C)
    R = -(F - A * h ** 2 - C * k_ ** 2)
    a2, b2 = sorted([R / A, R / C], reverse=True)
    return h, k_, sp.sqrt(a2), sp.sqrt(b2), sp.sqrt(a2 - b2)


h_e, k_e, a_e, b_e, c_e = elementos_elipse(sp.Integer(1), sp.Integer(4), sp.Integer(-2), sp.Integer(0), sp.Integer(-3))

# Álgebra lineal
A_nuc = Matrix([[1, 2], [2, 4]])
A_t = Matrix([[1, 1, 0], [0, 1, -1]])
sol_cramer = sp.solve([sp.Eq(x + y + z, 6), sp.Eq(2 * y + 5 * z, -4), sp.Eq(2 * x + 5 * y - z, 27)], [x, y, z])
q_div, r_div = sp.div(x ** 3 - 2 * x + 1, x - 1, x)
q_div2, r_div2 = sp.div(x ** 4 + 3 * x ** 2 - 2, x ** 2 + 1, x)
q_div3, r_div3 = sp.div(2 * x ** 4 - 3 * x + 5, x + 2, x)
coord = Matrix([[1, 1], [1, -1]]).solve(Matrix([3, 1]))
B1, B2 = Matrix([[1, 0], [0, 1]]), Matrix([[1, 1], [1, -1]])
cambio = B2.inv() * B1
ev = sorted(Matrix([[2, 1], [1, 2]]).eigenvals().keys())

CASOS = [
    # --- Geometría analítica ---
    ('recta por (1,2), (3,4)', v_valor(sp.Rational(4 - 2, 3 - 1) * (x - 1) + 2)),
    ('recta pendiente punto 2, (1,3)', v_valor(2 * (x - 1) + 3)),
    ('recta por (1,2,3), (3,5,4)', v_recta3d([1, 2, 3], [2, 3, 1])),
    ('recta paralela 2x+y=3, (1,1)', v_valor(-2 * (x - 1) + 1)),
    ('recta perpendicular 2x+y=3, (1,1)', v_valor(sp.Rational(1, 2) * (x - 1) + 1)),
    ('distancia punto recta (1,2), 3x+4y-5=0', v_valor(dist_pr)),
    ('distancia entre puntos (1,2), (4,6)', v_valor(Point2D(1, 2).distance(Point2D(4, 6)))),
    ('plano por (1,0,0), (0,1,0), (0,0,1)', v_plano(Matrix(pl_123.normal_vector), (1, 0, 0))),
    ('plano por (1,2,3), normal [1,1,1]', v_plano(Matrix([1, 1, 1]), (1, 2, 3))),
    ('distancia punto plano (1,1,1), x+y+z=1', v_valor(dist_pl)),
    ('angulo entre planos x+y+z=1, x-y=0', v_valor(ang_planos)),
    ('interseccion de planos x+y+z=1, x-y=0', v_recta3d(list(inter_planos.p1), list(inter_planos.direction_ratio))),
    ('angulo entre rectas y=x, y=-x', v_valor(sp.pi / 2)),
    ('distancia entre rectas (x,y,z)=(0,0,0)+t(1,0,0), (x,y,z)=(0,1,1)+t(0,1,0)', v_valor(dist_alabeadas)),
    ('distancia (1,2,3), (x,y,z)=(0,0,0)+t(1,1,1)', v_valor(dist_p_r3)),
    ('interseccion recta plano (x,y,z)=(1,0,0)+t(1,1,1), x+y+z=6', v_lista(list(inter_rp))),
    ('posicion relativa x+y+z=1, x-y=0', v_texto('secantes')),
    ('posicion relativa (x,y,z)=(0,0,0)+t(1,0,0), (x,y,z)=(0,1,1)+t(0,1,0)',
     v_texto('alabeadas' if not L1.intersection(L2) and not L1.is_parallel(L2) else 'secantes')),
    ('conica x^2+4y^2-2x-3=0', v_todos(v_texto('elipse'), v_clave_lista('centro', [h_e, k_e]), v_clave('a', a_e),
                                        v_clave('b', b_e), v_clave('c', c_e), v_clave('excentricidad', c_e / a_e),
                                        v_clave_lista('focos', [[h_e - c_e, k_e], [h_e + c_e, k_e]]),
                                        v_clave_lista('vertices', [[h_e - a_e, k_e], [h_e + a_e, k_e]]))),
    ('conica 9x^2-16y^2-18x-64y-199=0', v_todos(v_texto('hipérbola'), v_clave('a', 4), v_clave('b', 3),
                                                 v_clave('c', sp.sqrt(4 ** 2 + 3 ** 2)), v_clave_lista('centro', [1, -2]))),
    ('conica y^2-8x=0', v_todos(v_texto('parábola'), v_clave_lista('foco', [sp.Rational(8, 4), 0]),
                                 v_clave_lista('vertice', [0, 0]))),
    ('conica x^2+y^2-4x+6y-3=0', v_todos(v_texto('circunferencia'), v_clave('radio', sp.sqrt(3 + 2 ** 2 + 3 ** 2)))),
    ('conica xy=1', v_todos(v_texto('hipérbola'), v_clave('excentricidad', sp.sqrt(2)))),
    ('cuadrica x^2+y^2-z^2=1', v_texto('hiperboloide de una hoja')),
    ('cuadrica z=x^2-y^2', v_texto('paraboloide hiperbólico')),
    ('cuadrica x^2+2y^2+3z^2=6', v_texto('elipsoide')),
    # --- Álgebra lineal ---
    ('producto mixto [1,2,3],[4,5,6],[7,8,10]', v_valor(Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]]).det())),
    ('nucleo [[1,2],[2,4]]', v_base(A_nuc, len(A_nuc.nullspace()))),
    ('imagen [[1,2],[2,4]]', v_base(A_nuc, len(A_nuc.columnspace()), True)),
    ('nucleo T(x,y,z)=(x+y, y-z)', v_base(A_t, len(A_t.nullspace()))),
    ('independencia [1,2,3], [4,5,6], [7,8,9]',
     v_texto('dependientes' if Matrix([[1, 4, 7], [2, 5, 8], [3, 6, 9]]).rank() < 3 else 'independientes')),
    ('independencia [1,0,0], [0,1,0], [0,0,1]', v_texto('independientes')),
    ('base y dimension x+y-z=0', v_base(Matrix([[1, 1, -1]]), 3 - Matrix([[1, 1, -1]]).rank())),
    ('diagonalizar [[2,1],[1,2]]', lambda d: Matrix(S(d['valor'])) == sp.diag(*ev)),
    ('cramer x+y+z=6; 2y+5z=-4; 2x+5y-z=27', lambda d: igual_lista(
        [S(v.split(':')[1]) for v in d['valor'].strip('{}').split(',')], [sol_cramer[x], sol_cramer[y], sol_cramer[z]])),
    ('division de polinomios x^3-2x+1, x-1', v_lista([q_div, r_div])),
    ('division de polinomios x^4+3x^2-2, x^2+1', v_lista([q_div2, r_div2])),
    ('ruffini x^3-2x+1, 1', v_lista([q_div, r_div])),
    ('ruffini 2x^4-3x+5, -2', v_lista([q_div3, r_div3])),
    ('coordenadas [3,1] en base [1,1], [1,-1]', v_lista(list(coord))),
    ('cambio de base [1,0],[0,1]; [1,1],[1,-1]', lambda d: Matrix(S(d['valor'])) == cambio),
    # --- Discreta ---
    ('factores primos 360', v_todos(v_valor(360), lambda d: S(d['valores']['factores']) == {S(kk): S(vv) for kk, vv in sp.factorint(360).items()})),
    ('modulo 2^100, 7', v_valor(pow(2, 100, 7))),
    ('mod 3^200 mod 13', v_valor(pow(3, 200, 13))),
    ('modulo 17, 5', v_valor(17 % 5)),
    ('congruencia 3x≡4 (mod 7)', v_valor(next(v for v in range(7) if (3 * v) % 7 == 4))),
    ('congruencia x≡2 (mod 3); x≡3 (mod 5); x≡2 (mod 7)', v_valor(crt([3, 5, 7], [2, 3, 2])[0])),
    ('euclides 252, 198', v_todos(v_valor(sp.gcd(252, 198)),
                                   lambda d: 252 * d['valores']['s'] + 198 * d['valores']['t'] == sp.gcd(252, 198))),
    ('euclides 17, 5', v_valor(sp.gcd(17, 5))),
    ('diofantica 6x+9y=21', v_diofantica(6, 9, 21)),
    ('diofantica 3x+5y=47', v_diofantica(3, 5, 47)),
    ('tabla de verdad p -> (q y r)', v_texto(clasificar(Implies(p, And(q, r_))))),
    ('tabla de verdad (p -> q) <-> (~q -> ~p)', v_texto(clasificar(Equivalent(Implies(p, q), Implies(Not(q), Not(p)))))),
    ('tabla de verdad p y no p', v_texto(clasificar(And(p, Not(p))))),
    ("simplificar booleana ab+ab'", v_booleana(sp.simplify_logic(Or(And(a, b), And(a, Not(b)))))),
    ("simplificar booleana a'b'c + a'bc + abc + ab'c", v_booleana(sp.SOPform([a, b, c], [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]]))),
    ('es primo 91', v_texto('primo' if sp.isprime(91) else 'compuesto')),
    ('divisores de 36', v_lista(sp.divisors(36))),
    ('combinaciones 10, 3', v_valor(sp.binomial(10, 3))),
    ('variaciones 10, 3', v_valor(sp.ff(10, 3))),
    ('permutaciones 5', v_valor(sp.factorial(5))),
    ('combinaciones con repeticion 5, 3', v_valor(sp.binomial(5 + 3 - 1, 3))),
    ('variaciones con repeticion 2, 8', v_valor(2 ** 8)),
    ('permutaciones con repeticion 10, 3, 2, 5', v_valor(sp.factorial(10) / (sp.factorial(3) * sp.factorial(2) * sp.factorial(5)))),
    # --- No tapar comandos base ---
    ('modulo [3,4]', v_valor(5)),
    ('angulo [1,1], [1,0]', v_valor(sp.pi / 4)),
    ('recta tangente x^2, 1', v_valor(2 * x - 1)),
]

# Comparación de respuestas del alumno (función propia en r.comparar)
COMPROBACIONES = [
    ('recta por (1,2), (3,4)', 'x - y + 1 = 0', True),
    ('recta por (1,2), (3,4)', 'y = x + 2', False),
    ('recta por (1,2,3), (3,5,4)', '(x-1)/2=(y-2)/3=(z-3)/1', True),
    ('plano por (1,0,0), (0,1,0), (0,0,1)', '2x+2y+2z-2=0', True),
    ('nucleo [[1,2],[2,4]]', 'gen{(2,-1)}', True),
    ('diagonalizar [[2,1],[1,2]]', '[[3,0],[0,1]]', True),
    ('diofantica 6x+9y=21', 'x=-1+3k, y=3-2k', True),
    ('congruencia 3x≡4 (mod 7)', '13', True),
    ('factores primos 360', '8*45', False),
    ('tabla de verdad p -> (q y r)', '~p v (q & r)', True),
    ("simplificar booleana ab+ab'", 'b', False),
    ('conica x^2+4y^2-2x-3=0', '(1+sqrt(3),0), (1-sqrt(3),0)', True),
]


def main():
    ok = mal = 0
    t0 = time.time()
    for entrada, verif in CASOS:
        d = resolver(entrada, tiempo_max=60)
        try:
            bien = d.get('ok') and verif(d)
        except Exception as e:  # noqa
            bien = False
            d['error'] = (d.get('error') or '') + f' [verificador: {e!r}]'
        if bien:
            ok += 1
            print(f'OK   {entrada}  =>  {d["valor"]}')
        else:
            mal += 1
            print(f'MAL  {entrada}  =>  {d.get("valor")}  {d.get("error") or ""}')
    for entrada, resp, esperado in COMPROBACIONES:
        fn, arg = _detectar(entrada)
        res = fn(arg)
        modo, ref, ctx = res.comparar
        veredicto = ctx['comprobar'](resp) if modo == 'propio' else None
        bien = veredicto is not None and veredicto.get('correcto') is esperado
        ok += bien
        mal += not bien
        print(('OK   ' if bien else 'MAL  ') + f'comprobar «{entrada}» con «{resp}» -> {veredicto and veredicto.get("mensaje")}')
    total = ok + mal
    print(f'\nResumen: {ok}/{total} correctos, {mal} mal ({time.time() - t0:.1f} s)')
    return mal == 0


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
