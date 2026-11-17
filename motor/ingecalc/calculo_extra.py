"""Cálculo extra de IngeCalc.

- Análisis I: dominio, asíntotas, concavidad e inflexión, estudio completo de funciones, recta normal,
  diferencial y aproximación lineal, teoremas del valor medio (Lagrange y Rolle), función inversa.
- Análisis II: área en polares, longitud de arco, volumen y superficie de revolución, derivada de
  paramétricas, criterios de convergencia de series y radio de convergencia de series de potencias.
- Análisis III: derivada direccional, diferencial total, límites dobles (por trayectorias y polares),
  integral de línea, función potencial, flujo, curvatura y triedro de una función vectorial,
  sistemas lineales de EDO y trayectorias ortogonales.

Cada comando recibe el texto que sigue al comando y devuelve un Resultado (ver nucleo.py).
Solo usa sympy y mpmath (corre en el celular con Chaquopy)."""
import ctypes
import math
import re
import threading

import mpmath
import sympy as sp
from sympy.calculus.util import continuous_domain, function_range

from . import graficos as G
from .nucleo import (ErrorEntrada, Resultado, coma_decimal, dividir, fmt_num, leer, leer_ecuacion,
                     normalizar, simplificar_bonito, sin_tildes, tex, variables_de, x, y, z, t, n, k)

oo = sp.oo
PI = sp.pi
VENTANA_PERIODICA = (-2 * sp.pi, 2 * sp.pi)
SALTO = '\\\\'  # salto de línea en LaTeX


# ============================================================================
# Utilidades generales
# ============================================================================
class _Corte(Exception):
    """Se inyecta en un hilo de cálculo que superó su tiempo."""


def _con_tiempo(fn, seg=6.0, defecto=None):
    """Ejecuta fn() con un límite de tiempo; si tarda más o falla devuelve `defecto`."""
    caja = {}

    def objetivo():
        try:
            caja['r'] = fn()
        except BaseException as e:  # noqa
            caja['e'] = e

    h = threading.Thread(target=objetivo, daemon=True)
    h.start()
    h.join(seg)
    if h.is_alive():
        try:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(h.ident), ctypes.py_object(_Corte))
        except Exception:  # noqa
            pass
        return defecto
    if 'e' in caja:
        if isinstance(caja['e'], ErrorEntrada):
            raise caja['e']
        return defecto
    return caja['r']


def _plano(e):
    """Cambia los símbolos con supuestos (real, positivo...) por los símbolos comunes del motor."""
    if isinstance(e, dict):
        return {_plano(k_): _plano(v) for k_, v in e.items()}
    if isinstance(e, list):
        return [_plano(a) for a in e]
    if isinstance(e, tuple):
        return tuple(_plano(a) for a in e)
    if not isinstance(e, sp.Basic):
        return e
    rep = {}
    for s_ in e.free_symbols:
        if type(s_) is sp.Symbol:
            p = sp.Symbol(s_.name)
            if p != s_:
                rep[s_] = p
    return e.xreplace(rep) if rep else e


def _real(v):
    return sp.Symbol(v.name, real=True)


def _num(e):
    """Valor real (float) de una expresión numérica, o None."""
    try:
        v = complex(sp.N(e, 15))
    except Exception:  # noqa
        return None
    if not (math.isfinite(v.real) and math.isfinite(v.imag)):
        return None
    if abs(v.imag) > 1e-9 * (1 + abs(v.real)):
        return None
    return v.real


def _finito(L):
    """¿L es un valor finito (número real o expresión sin infinitos)?"""
    if L is None or isinstance(L, sp.AccumBounds):
        return False
    try:
        if L.has(oo, -oo, sp.zoo, sp.nan, sp.AccumBounds, sp.Limit):
            return False
    except Exception:  # noqa
        return False
    if L.is_number:
        return _num(L) is not None
    return True


def _infinito(L):
    return L is not None and not isinstance(L, sp.AccumBounds) and L in (oo, -oo, sp.zoo)


def _limite(e, v, p, lado='+', seg=6.0):
    def calc():
        if p in (oo, -oo):
            return sp.limit(e, v, p)
        return sp.limit(e, v, p, lado)
    return _con_tiempo(calc, seg)


def _tex_num(v, dig=6):
    """LaTeX exacto y, si no es racional, su aproximación decimal."""
    s_ = tex(v)
    try:
        if v.is_number and not v.is_Rational and v.is_finite:
            d = _num(v)
            if d is not None:
                s_ += r' \approx ' + fmt_num(d, dig)
    except Exception:  # noqa
        pass
    return s_


def _decimal_txt(v, dig=12):
    d = _num(v) if isinstance(v, sp.Basic) else None
    return fmt_num(d, dig) if d is not None else None


def _ti(p, q, ci=False, cd=False):
    """Intervalo en LaTeX."""
    a = r'-\infty' if p == -oo else tex(p)
    b = r'\infty' if q == oo else tex(q)
    return ('[' if ci and p != -oo else '(') + a + ',\\ ' + b + (']' if cd and q != oo else ')')


def _tex_conjunto(c):
    """LaTeX de un conjunto de reales, con la notación de la cursada (ℝ − {a})."""
    if c == sp.S.Reals:
        return r'\mathbb{R}'
    if c == sp.S.EmptySet:
        return r'\varnothing'
    try:
        resto = sp.Complement(sp.S.Reals, c)
        if isinstance(resto, sp.FiniteSet) and len(resto) <= 6:
            return r'\mathbb{R} - \left\{' + ',\\ '.join(tex(p) for p in sorted(resto, key=lambda q: _num(q) or 0)) + r'\right\}'
    except Exception:  # noqa
        pass
    if isinstance(c, sp.Interval):
        return _ti(c.start, c.end, not c.left_open, not c.right_open)
    if isinstance(c, sp.Union) and all(isinstance(a, sp.Interval) for a in c.args):
        return r' \cup '.join(_ti(a.start, a.end, not a.left_open, not a.right_open) for a in c.args)
    return tex(c)


def _simp_conjunto(c):
    """Simplifica los extremos de un intervalo o unión de intervalos."""
    def s_(q):
        if q in (oo, -oo):
            return q
        r_ = _con_tiempo(lambda: sp.nsimplify(sp.simplify(q)) if q.is_number else sp.simplify(q), 3)
        return r_ if r_ is not None else q
    try:
        if isinstance(c, sp.Interval):
            return sp.Interval(s_(c.start), s_(c.end), c.left_open, c.right_open)
        if isinstance(c, sp.Union) and all(isinstance(a, sp.Interval) for a in c.args):
            return sp.Union(*[_simp_conjunto(a) for a in c.args])
    except Exception:  # noqa
        pass
    return c


def _union(intervalos):
    if not intervalos:
        return sp.S.EmptySet
    return sp.Union(*intervalos)


def _leer_vector(texto, nombre='un vector'):
    t_ = texto.strip()
    if t_.startswith('<') and t_.endswith('>'):
        t_ = '(' + t_[1:-1] + ')'
    e = leer(t_)
    if isinstance(e, (list, tuple, sp.Tuple)):
        return [sp.sympify(c) for c in e]
    if isinstance(e, sp.MatrixBase):
        return list(e)
    raise ErrorEntrada(f'Se esperaba {nombre} entre paréntesis, por ejemplo (1, 2) o [1, 2, 3]; recibí «{texto}».')


def _es_vector_texto(texto):
    t_ = texto.strip()
    return (t_.startswith('(') or t_.startswith('[') or t_.startswith('<')) and len(dividir(t_[1:-1])) >= 2 and \
        (t_.endswith(')') or t_.endswith(']') or t_.endswith('>'))


def _vector_tex(v):
    return r'\left(' + ',\\ '.join(tex(c) for c in v) + r'\right)'


def _norma(v):
    return sp.sqrt(sum(c ** 2 for c in v))


def _simp(e, seg=6.0):
    """simplificar_bonito con límite de tiempo."""
    r_ = _con_tiempo(lambda: simplificar_bonito(e), seg)
    return r_ if r_ is not None else e


def _uso(texto):
    return ErrorEntrada('Uso: ' + texto)


def _graficar(r, exprs, v, puntos=(), etiquetas=None, punteadas=(), verticales=(), **kw):
    """Gráfico 2D con curvas punteadas (asíntotas, rectas auxiliares) y rectas verticales x = c."""
    try:
        g = G.grafico_2d([_plano(e) for e in exprs], _plano(v), [p for p in puntos if p], etiquetas=etiquetas, **kw)
        for i in punteadas:
            if i < len(g['curvas']):
                g['curvas'][i]['estilo'] = 'punteado'
        if verticales:
            g['verticales'] = [{'x': float(_num(c)), 'etiqueta': f'x = {fmt_num(_num(c), 4)}', 'color': '#6b7280'}
                               for c in verticales if _num(c) is not None]
        r.grafico = g
    except Exception:  # noqa
        pass


# ----------------------------------------------------------------------------
# Raíces, dominio, intervalos y signos (una variable real)
# ----------------------------------------------------------------------------
def _bonito_num(c):
    """0.5 -> 1/2, 1.0471975 -> pi/3 (si es exacto); si no, decimal con 10 cifras."""
    try:
        q = sp.nsimplify(c, [sp.pi], tolerance=1e-10, rational=False)
        if sp.count_ops(q) <= 8 and abs(float(sp.N(q)) - c) < 1e-9:
            return q
    except Exception:  # noqa
        pass
    return sp.Float(round(c, 10), 12)


def _raices_numericas(g, vr, dom=sp.S.Reals, a=-10.0, b=10.0, m=400):
    try:
        fl = sp.lambdify(vr, g, 'mpmath')
    except Exception:  # noqa
        return []

    def val(q):
        try:
            r_ = fl(mpmath.mpf(q))
            if isinstance(r_, mpmath.mpc):
                if abs(r_.imag) > 1e-10:
                    return None
                r_ = r_.real
            r_ = float(r_)
            return r_ if math.isfinite(r_) else None
        except Exception:  # noqa
            return None
    xs = [a + (b - a) * i / m for i in range(m + 1)]
    vs = [val(q) for q in xs]
    halladas = []
    for i in range(m):
        v1, v2 = vs[i], vs[i + 1]
        if v1 is None or v2 is None:
            continue
        if v1 == 0:
            cand = xs[i]
        elif v1 * v2 < 0:
            try:
                cand = float(mpmath.findroot(fl, (xs[i], xs[i + 1]), solver='illinois'))
            except Exception:  # noqa
                continue
        else:
            continue
        fc = val(cand)
        if fc is None or abs(fc) > 1e-7:
            continue
        if any(abs(cand - h) < 1e-7 for h in halladas):
            continue
        halladas.append(cand)
    out = []
    for c in halladas:
        q = _bonito_num(c)
        try:
            if dom.contains(q) is sp.false:
                continue
        except Exception:  # noqa
            pass
        out.append(q)
    return out


def _raices(g, vr, dom=sp.S.Reals, seg=8.0):
    """Raíces reales de g en dom. Devuelve (lista ordenada, nota) con nota '', 'periodica' o 'numerica'."""
    g = sp.sympify(g)
    if not g.has(vr):
        return [], ''
    nota = ''
    lista = None
    sol = None
    # Atajo: numerador polinómico (el caso más común en la cursada)
    try:
        num, den = sp.fraction(sp.together(g))
        num = sp.factor(num)
        factores = [b for b, _ in sp.factor_list(num)[1]] if num.has(vr) else []
        if factores and all(fa.is_polynomial(vr) for fa in factores):
            cand = []
            for fa in factores:
                if fa.has(vr):
                    cand += [c for c in sp.Poly(fa, vr).real_roots()]
            lista = []
            for c in cand:
                if isinstance(c, sp.CRootOf):
                    c = sp.Float(sp.N(c, 15), 12)
                dv =_num(_real_num(den.subs(vr, c)))
                if dv is None or abs(dv) < 1e-14:
                    continue
                try:
                    if dom.contains(c) is sp.false:
                        continue
                except Exception:  # noqa
                    pass
                lista.append(c)
    except Exception:  # noqa
        lista = None
    if lista is None:
        sol = _con_tiempo(lambda: sp.solveset(g, vr, dom), seg)
    if lista is not None:
        pass  # ya resuelto por el atajo polinómico
    elif isinstance(sol, sp.FiniteSet):
        lista = [c for c in sol if c.is_real is not False and _num(c) is not None]
    elif sol is sp.S.EmptySet:
        lista = []
    elif sol is not None and not isinstance(sol, sp.ConditionSet):
        rec = _con_tiempo(lambda: sol.intersect(sp.Interval(*VENTANA_PERIODICA)), 4)
        if isinstance(rec, sp.FiniteSet):
            lista = [c for c in rec if _num(c) is not None]
            nota = 'periodica'
        elif rec is sp.S.EmptySet:
            lista, nota = [], 'periodica'
    if lista is None:
        lista = _raices_numericas(g, vr, dom)
        nota = 'numerica'
    unicas = []
    for c in sorted(lista, key=_num):
        if not any(abs(_num(c) - _num(u)) < 1e-12 for u in unicas):
            unicas.append(c)
    return unicas, nota


_FUNCIONES_CONOCIDAS = (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc, sp.asin, sp.acos, sp.atan, sp.acot, sp.exp,
                        sp.log, sp.sinh, sp.cosh, sp.tanh, sp.asinh, sp.Abs, sp.sign)


def _dominio(f, vr, seg=8.0):
    """Dominio real de f: intersección de las condiciones (rápido); si hay funciones raras, continuous_domain."""
    if not f.has(vr):
        return sp.S.Reals
    simple = all(isinstance(a, _FUNCIONES_CONOCIDAS) for a in sp.preorder_traversal(f) if isinstance(a, sp.Function)) and         not any(isinstance(a, sp.Pow) and a.exp.has(vr) and a.base.has(vr) for a in sp.preorder_traversal(f))
    if simple:
        dom = sp.S.Reals
        ok = True
        for e, tipo, _ in _condiciones(f, vr):
            c = _con_tiempo(lambda: _resolver_condicion(e, tipo, vr), seg / 2)
            if not isinstance(c, sp.Set) or isinstance(c, sp.ConditionSet):
                ok = False
                break
            dom = _con_tiempo(lambda: sp.Intersection(dom, c), 4)
            if dom is None:
                ok = False
                break
        if ok:
            return dom
    d = _con_tiempo(lambda: continuous_domain(f, vr, sp.S.Reals), seg)
    return d if isinstance(d, sp.Set) else sp.S.Reals


def _partir(I, puntos):
    """Parte el intervalo I en los puntos interiores dados."""
    ps = []
    for p in puntos:
        try:
            if I.contains(p) is sp.true and p != I.start and p != I.end:
                ps.append(p)
        except Exception:  # noqa
            continue
    ps = sorted(set(ps), key=_num)
    bordes = [I.start] + ps + [I.end]
    out = []
    for i in range(len(bordes) - 1):
        izq_abierto = True if i > 0 else I.left_open
        der_abierto = True if i < len(bordes) - 2 else I.right_open
        J = sp.Interval(bordes[i], bordes[i + 1], izq_abierto, der_abierto)
        if isinstance(J, sp.Interval):
            out.append(J)
    return out


def _intervalos(dom):
    """Intervalos que forman el dominio. Si es periódico se recorta a [-2π, 2π]. Devuelve (lista, recortado)."""
    if dom == sp.S.Reals:
        return [sp.Interval(-oo, oo)], False
    if isinstance(dom, sp.Interval):
        return [dom], False
    if isinstance(dom, sp.Union) and all(isinstance(a, (sp.Interval, sp.FiniteSet)) for a in dom.args):
        return [a for a in dom.args if isinstance(a, sp.Interval)], False
    if isinstance(dom, sp.Complement):
        base, quitar = dom.args
        if isinstance(quitar, sp.FiniteSet) and isinstance(base, sp.Interval):
            return _partir(base, list(quitar)), False
        w = sp.Interval(*VENTANA_PERIODICA)
        base_w = _con_tiempo(lambda: base.intersect(w), 4)
        pts = _con_tiempo(lambda: quitar.intersect(w), 4)
        if isinstance(base_w, sp.Interval) and isinstance(pts, sp.FiniteSet):
            return _partir(base_w, list(pts)), True
    rec = _con_tiempo(lambda: dom.intersect(sp.Interval(*VENTANA_PERIODICA)), 4)
    if isinstance(rec, sp.Interval):
        return [rec], True
    if isinstance(rec, sp.Union) and all(isinstance(a, (sp.Interval, sp.FiniteSet)) for a in rec.args):
        return [a for a in rec.args if isinstance(a, sp.Interval)], True
    return [sp.Interval(-oo, oo)], True


def _muestra(p, q):
    if p == -oo and q == oo:
        return sp.Integer(0)
    if p == -oo:
        return q - 1
    if q == oo:
        return p + 1
    return (p + q) / 2


def _real_num(e):
    """(-1)**(2/3) -> 1, (-8)**(1/3) -> -2: raíces de índice impar de negativos como en la cursada."""
    try:
        return e.replace(lambda q: q.is_Pow and q.base.is_number and q.base.is_negative and q.exp.is_Rational
                         and not q.exp.is_Integer and q.exp.q % 2 == 1,
                         lambda q: sp.real_root(q.base, q.exp.q) ** q.exp.p)
    except Exception:  # noqa
        return e


def _valor_en(f, vr, c):
    return _simp(_real_num(f.subs(vr, c)))


def _signo(g, vr, m):
    v = _num(_real_num(g.subs(vr, m)))
    if v is None:
        return None
    return 1 if v > 1e-13 else (-1 if v < -1e-13 else 0)


def _tramos(g, vr, intervalos, puntos):
    """[(J, signo de g en J)] partiendo los intervalos del dominio en los puntos dados."""
    out = []
    for I in intervalos:
        for J in _partir(I, puntos):
            out.append((J, _signo(g, vr, _muestra(J.start, J.end))))
    return out


def _vecinos(tramos, c):
    izq = next((s_ for J, s_ in tramos if J.end == c), None)
    der = next((s_ for J, s_ in tramos if J.start == c), None)
    return izq, der


def _puntos_no_derivables(g, vr, dom_f, intervalos_f):
    """Puntos del dominio de f donde g (= f' o f'') no está definida."""
    if not g.has(vr):
        return []
    dg = _dominio(g, vr, 6)
    ints, _ = _intervalos(dg)
    cand = set()
    # sign(x), Abs, Heaviside: g salta donde se anula el argumento
    for a in sp.preorder_traversal(g):
        if isinstance(a, (sp.sign, sp.Heaviside)):
            cand |= set(_raices(a.args[0], vr)[0])
    for I in ints:
        for p in (I.start, I.end):
            if p not in (oo, -oo):
                cand.add(p)
    try:
        resto = sp.Complement(sp.S.Reals, dg)
        if isinstance(resto, sp.FiniteSet):
            cand |= set(resto)
    except Exception:  # noqa
        pass
    out = []
    for c in cand:
        try:
            if dom_f.contains(c) is sp.true and (dg.contains(c) is not sp.true or g.has(sp.sign, sp.Heaviside)):
                out.append(c)
        except Exception:  # noqa
            continue
    return sorted(out, key=_num)


def _texto_tramo(J):
    return '$' + _ti(J.start, J.end, not J.left_open, not J.right_open) + '$'


# ============================================================================
# ANÁLISIS I
# ============================================================================
def _sin_piecewise_raiz(e):
    """cbrt(u) se lee como Piecewise(|u|^(1/3) sign(u), ...): se vuelve a u^(1/3) (los negativos se tratan aparte)."""
    try:
        return e.replace(lambda q: isinstance(q, sp.Piecewise) and len(q.args) == 2 and q.args[1].cond == sp.true
                         and q.args[1].expr.is_Pow and q.args[1].expr.exp.is_Rational and q.args[1].expr.exp.q % 2 == 1,
                         lambda q: q.args[1].expr)
    except Exception:  # noqa
        return e


def _leer_funcion(texto, uso):
    if not texto or not texto.strip():
        raise _uso(uso)
    f = leer(texto)
    if isinstance(f, (sp.Tuple, list, tuple)):
        raise ErrorEntrada('Se esperaba una función de una variable. ' + 'Uso: ' + uso)
    f = _sin_piecewise_raiz(f)
    vs = variables_de(f)
    if len(vs) > 1:
        raise ErrorEntrada(f'La función tiene más de una variable ({", ".join(map(str, vs))}). Uso: {uso}')
    v = vs[0] if vs else x
    return f, v


def _condiciones(f, vr):
    """Restricciones de dominio: [(expresión, tipo, motivo)]."""
    conds = []

    def agregar(e, tipo, motivo):
        if e.has(vr) and not any(c[0] == e and c[1] == tipo for c in conds):
            conds.append((e, tipo, motivo))
    for a in sp.preorder_traversal(f):
        if isinstance(a, sp.Pow):
            b, e = a.args
            if e.is_Rational and not e.is_Integer and e.q % 2 == 0:
                agregar(b, '>' if e < 0 else '>=', 'radicando de una raíz de índice par')
            elif e.is_negative:
                agregar(b, '!=', 'denominador')
        elif isinstance(a, sp.log):
            agregar(a.args[0], '>', 'argumento de un logaritmo')
        elif isinstance(a, (sp.tan, sp.sec)):
            agregar(sp.cos(a.args[0]), '!=', 'tg y sec no existen donde cos = 0')
        elif isinstance(a, (sp.cot, sp.csc)):
            agregar(sp.sin(a.args[0]), '!=', 'cotg y cosec no existen donde sen = 0')
        elif isinstance(a, (sp.asin, sp.acos)):
            agregar(a.args[0], '[-1,1]', 'argumento de arcsen / arccos')
    return conds


def _resolver_condicion(e, tipo, vr):
    try:
        if tipo == '>':
            return sp.solveset(e > 0, vr, sp.S.Reals)
        if tipo == '>=':
            return sp.solveset(e >= 0, vr, sp.S.Reals)
        if tipo == '!=':
            ceros = sp.solveset(e, vr, sp.S.Reals)
            return sp.Complement(sp.S.Reals, ceros)
        if tipo == '[-1,1]':
            return sp.Intersection(sp.solveset(e >= -1, vr, sp.S.Reals), sp.solveset(e <= 1, vr, sp.S.Reals))
    except Exception:  # noqa
        return None
    return None


def _analizar_dominio(r, f, vr, nivel=0, titulo='Dominio'):
    conds = _condiciones(f, vr)
    if not conds:
        r.paso(titulo, 'No hay denominadores con la variable, ni raíces de índice par, ni logaritmos: '
               r'la función está definida para todo real.', r'\mathrm{Dom}\,f = \mathbb{R}', nivel)
    else:
        r.paso(titulo, 'Se piden las condiciones para que la expresión tenga sentido en los reales.', '', nivel)
        simbolo = {'>': '> 0', '>=': r'\geq 0', '!=': r'\neq 0', '[-1,1]': ''}
        for e, tipo, motivo in conds:
            sol = _con_tiempo(lambda: _resolver_condicion(e, tipo, vr), 6)
            cond_tex = (r'-1 \leq ' + tex(e) + r' \leq 1') if tipo == '[-1,1]' else tex(e) + ' ' + simbolo[tipo]
            if sol is not None:
                if tipo == '!=':
                    ceros = sp.Complement(sp.S.Reals, sol)
                    if isinstance(ceros, sp.FiniteSet) and len(ceros) > 0:
                        sol_tex = tex(vr) + r' \neq ' + ',\\ '.join(tex(c) for c in sorted(ceros, key=_num))
                    else:
                        sol_tex = tex(vr) + r' \in ' + _tex_conjunto(sol)
                else:
                    sol_tex = tex(vr) + r' \in ' + _tex_conjunto(sol)
                cond_tex += r' \Rightarrow ' + sol_tex
            r.paso(motivo[0].upper() + motivo[1:], '', cond_tex, nivel + 1)
    dom = _dominio(f, vr)
    if conds:
        r.paso('Dominio (intersección de las condiciones)', '', r'\mathrm{Dom}\,f = ' + _tex_conjunto(dom), nivel)
    return dom


def dominio(arg):
    uso = 'dominio f   (por ejemplo: dominio sqrt(4-x^2)/ln(x))'
    f, v = _leer_funcion(dividir(arg)[0] if dividir(arg) else '', uso)
    vr = _real(v)
    fr = f.subs(v, vr)
    r = Resultado('Cálculo', 'Dominio e imagen')
    r.entrada_latex = 'f(' + tex(v) + ') = ' + tex(f)
    dom = _analizar_dominio(r, fr, vr)
    img = _con_tiempo(lambda: function_range(fr, vr, dom), 4)
    if isinstance(img, sp.Set):
        img = _simp_conjunto(img)
        r.extra('Imagen (recorrido)', r'\mathrm{Im}\,f = ' + _tex_conjunto(img))
        r.valores['imagen'] = _plano(img)
    r.valor = _plano(dom)
    r.resultado_latex = r'\mathrm{Dom}\,f = ' + _tex_conjunto(dom)
    r.comparar = ('intervalo', _plano(dom), {'var': v})
    _graficar(r, [f], v)
    return r


# ----------------------------------------------------------------------------
# Asíntotas
# ----------------------------------------------------------------------------
def _analizar_asintotas(r, f, vr, dom, ints, recortado, nivel=0):
    """Devuelve dict con verticales [c], horizontales [(L, lados)], oblicuas [(m, b, lados)]."""
    res = {'verticales': [], 'horizontales': [], 'oblicuas': [], 'evitables': []}
    # --- verticales
    candidatos = {}
    for I in ints:
        if I.start != -oo:
            candidatos.setdefault(I.start, set()).add('+')
        if I.end != oo:
            candidatos.setdefault(I.end, set()).add('-')
    cands = []
    for c, lados in candidatos.items():
        try:
            if dom.contains(c) is sp.true:
                continue  # f es continua en c (extremo de la ventana o borde cerrado)
        except Exception:  # noqa
            pass
        cands.append((c, lados))
    cands.sort(key=lambda par: _num(par[0]))
    if cands:
        r.paso('Asíntotas verticales', 'Se buscan en los puntos donde f no está definida (bordes del dominio): '
               'hay asíntota vertical si algún límite lateral es infinito.', '', nivel)
    else:
        r.paso('Asíntotas verticales', 'El dominio no tiene puntos frontera finitos: no hay asíntotas verticales.', '', nivel)
    for c, lados in cands:
        lims = []
        infinito = False
        valores = []
        for lado in sorted(lados, reverse=True):
            L = _limite(f, vr, c, lado)
            valores.append(L)
            if L is None:
                lims.append(r'\lim_{' + tex(vr) + r'\to ' + tex(c) + '^' + lado + r'} f\ \text{(no se pudo calcular)}')
                continue
            lims.append(r'\lim_{' + tex(vr) + r'\to ' + tex(c) + '^' + lado + '} f = ' + tex(L))
            if _infinito(L):
                infinito = True
        if infinito:
            res['verticales'].append(c)
            conc = r'\Rightarrow\ \text{asíntota vertical } ' + tex(vr) + ' = ' + tex(c)
        elif len(valores) == 2 and all(_finito(L) for L in valores) and valores[0] == valores[1]:
            res['evitables'].append((c, valores[0]))
            conc = r'\Rightarrow\ \text{discontinuidad evitable (no hay asíntota)}'
        else:
            conc = r'\Rightarrow\ \text{no hay asíntota vertical}'
        r.paso(f'En x = {tex(c)}', '', r',\quad '.join(lims) + r'\ ' + conc, nivel + 1)
    # --- horizontales y oblicuas
    sup_inf = any(I.end == oo for I in ints) or recortado
    inf_inf = any(I.start == -oo for I in ints) or recortado
    lados = [(oo, '+'), (-oo, '-')]
    hallado = {}
    info = {}
    u_ = sp.Symbol('u', positive=True)

    def lim_inf(e, p):
        if p == oo:
            return _limite(e, vr, oo)
        return _limite(_real_num(e.subs(vr, -u_)), u_, oo)   # x = -u, u -> +oo (raíces impares reales)
    for p, lado in lados:
        if (lado == '+' and not sup_inf) or (lado == '-' and not inf_inf):
            continue
        L = lim_inf(f, p)
        if _finito(L):
            info[lado] = ('h', L)
            continue
        if L is None or not _infinito(L):
            info[lado] = ('x', L)
            continue
        m = lim_inf(f / vr, p)
        if _finito(m) and m != 0:
            b = lim_inf(f - m * vr, p)
            if _finito(b):
                info[lado] = ('o', m, b)
                continue
        info[lado] = ('n', L, m)
    juntos = '+' in info and '-' in info and info['+'] == info['-']
    for lado, dato in info.items():
        if juntos and lado == '-':
            hallado[lado] = dato if dato[0] in ('h', 'o') else None
            continue
        sim = (r'\pm\infty' if juntos else (r'+\infty' if lado == '+' else r'-\infty'))
        nombre = '±∞' if juntos else ('+∞' if lado == '+' else '−∞')
        lim = r'\lim_{' + tex(vr) + r'\to ' + sim + '} '
        if dato[0] == 'h':
            hallado[lado] = dato
            r.paso(f'Asíntota horizontal (x → {nombre})', 'El límite en el infinito es finito.',
                   lim + 'f = ' + tex(dato[1]) + r'\Rightarrow\ y = ' + tex(dato[1]), nivel)
        elif dato[0] == 'o':
            hallado[lado] = dato
            m, b = dato[1], dato[2]
            r.paso(f'Asíntota oblicua (x → {nombre})',
                   r'El límite es infinito: se busca \(y = mx + b\) con \(m = \lim f/x\) y \(b = \lim (f - mx)\).',
                   r'm = ' + lim + r'\frac{f}{' + tex(vr) + '} = ' + tex(m) + r',\quad b = ' + lim + r'\left(f '
                   + (('+ ' + tex(-m * vr)) if (m * vr).could_extract_minus_sign() else ('- ' + tex(m * vr)))
                   + r'\right) = ' + tex(b) + r'\Rightarrow\ y = ' + tex(m * vr + b), nivel)
        elif dato[0] == 'x':
            r.paso(f'Comportamiento para x → {nombre}',
                   'El límite no existe (la función oscila) o no se pudo calcular: no hay asíntota horizontal ni oblicua.',
                   (lim + 'f = ' + tex(dato[1])) if dato[1] is not None else '', nivel)
        else:
            L, m = dato[1], dato[2]
            r.paso(f'Sin asíntota horizontal ni oblicua (x → {nombre})',
                   r'\(\lim f = \infty\) y \(f/x\) no tiende a un número distinto de 0 (o \(f - mx\) no tiene límite finito).',
                   lim + 'f = ' + tex(L) + ((r',\quad ' + lim + r'\frac{f}{' + tex(vr) + '} = ' + tex(m)) if m is not None else ''),
                   nivel)
    hallado = {k_: v_ for k_, v_ in hallado.items() if v_}
    # agrupar lados
    for lado, dato in hallado.items():
        if dato[0] == 'h':
            prev = next((h for h in res['horizontales'] if h[0] == dato[1]), None)
            if prev:
                prev[1].append(lado)
            else:
                res['horizontales'].append((dato[1], [lado]))
        else:
            prev = next((o for o in res['oblicuas'] if o[0] == dato[1] and o[1] == dato[2]), None)
            if prev:
                prev[2].append(lado)
            else:
                res['oblicuas'].append((dato[1], dato[2], [lado]))
    return res


def _lista_asintotas(res, vr):
    """(lista de Eq para mostrar, lista para comparar)."""
    eqs, comp = [], []
    for c in res['verticales']:
        eqs.append(sp.Eq(_plano(vr), c))
        comp.append(c)
    for L, lados in res['horizontales']:
        eqs.append(sp.Eq(y, L))
        comp.append(L)
    for m, b, lados in res['oblicuas']:
        eqs.append(sp.Eq(y, _plano(m * vr + b)))
        comp.append(_plano(m * vr + b))
    return eqs, comp


def _latex_asintotas(res, vr):
    partes = []
    if res['verticales']:
        partes.append(r'\text{Verticales: }' + ',\\ '.join(tex(vr) + ' = ' + tex(c) for c in res['verticales']))
    if res['horizontales']:
        partes.append(r'\text{Horizontales: }' + ',\\ '.join(
            'y = ' + tex(L) + ('' if len(l_) == 2 else (r'\ (x\to+\infty)' if l_ == ['+'] else r'\ (x\to-\infty)'))
            for L, l_ in res['horizontales']))
    if res['oblicuas']:
        partes.append(r'\text{Oblicuas: }' + ',\\ '.join(
            'y = ' + tex(m * vr + b) + ('' if len(l_) == 2 else (r'\ (x\to+\infty)' if l_ == ['+'] else r'\ (x\to-\infty)'))
            for m, b, l_ in res['oblicuas']))
    return r'\\ '.join(partes) if partes else r'\text{No tiene asíntotas}'


def asintotas(arg):
    uso = 'asintotas f   (por ejemplo: asintotas (x^2+1)/(x-1))'
    f, v = _leer_funcion(dividir(arg)[0] if dividir(arg) else '', uso)
    vr = _real(v)
    fr = f.subs(v, vr)
    r = Resultado('Cálculo', 'Asíntotas')
    r.entrada_latex = 'f(' + tex(v) + ') = ' + tex(f)
    dom = _dominio(fr, vr)
    ints, recortado = _intervalos(dom)
    r.paso('Dominio', 'Las asíntotas verticales solo pueden estar en los bordes del dominio.',
           r'\mathrm{Dom}\,f = ' + _tex_conjunto(dom), 0)
    if recortado:
        r.avisos.append('La función es periódica o tiene infinitos puntos fuera del dominio: se analizan los de [-2π, 2π].')
    res = _analizar_asintotas(r, fr, vr, dom, ints, recortado)
    eqs, comp = _lista_asintotas(res, vr)
    r.valor = eqs
    r.valores['verticales'] = _plano(res['verticales'])
    r.valores['horizontales'] = [_plano(L) for L, _ in res['horizontales']]
    r.valores['oblicuas'] = [_plano(m * vr + b) for m, b, _ in res['oblicuas']]
    r.resultado_latex = _latex_asintotas(res, vr)
    r.comparar = ('conjunto', comp, {})
    lineas = [L for L, _ in res['horizontales']] + [m * vr + b for m, b, _ in res['oblicuas']]
    _graficar(r, [fr] + lineas, vr, etiquetas=['f'] + ['asíntota'] * len(lineas),
              punteadas=range(1, 1 + len(lineas)), verticales=res['verticales'])
    return r


# ----------------------------------------------------------------------------
# Monotonía y concavidad
# ----------------------------------------------------------------------------
def _analizar_monotonia(r, fr, vr, dom, ints, nivel=0):
    df = _simp(sp.diff(fr, vr))
    r.paso('Derivada primera', '', "f'(" + tex(vr) + ') = ' + tex(df), nivel)
    ceros, nota = _raices(df, vr, dom)
    nd = _puntos_no_derivables(df, vr, dom, ints)
    crit = sorted(set(ceros) | set(nd), key=_num)
    texto = "Se resuelve f'(x) = 0" + (' y se agregan los puntos del dominio donde f no es derivable.' if nd else '.')
    if nota == 'numerica':
        texto += ' (Las raíces se aproximaron numéricamente en [-10, 10].)'
    elif nota == 'periodica':
        texto += ' (Se muestran las de [-2π, 2π].)'
    r.paso('Puntos críticos', texto, "f'(" + tex(vr) + r') = 0 \Rightarrow ' +
           (tex(vr) + ' = ' + ',\\ '.join(tex(c) for c in ceros) if ceros else r'\text{sin soluciones}')
           + ((r';\quad \nexists f' + "'" + r'\ \text{en } ' + ',\\ '.join(tex(c) for c in nd)) if nd else ''), nivel)
    tramos = _tramos(df, vr, ints, crit)
    extremos = []
    for c in crit:
        izq, der = _vecinos(tramos, c)
        fc = _valor_en(fr, vr, c)
        if izq == 1 and der == -1:
            tipo = 'máximo'
        elif izq == -1 and der == 1:
            tipo = 'mínimo'
        else:
            tipo = None
        if tipo:
            extremos.append((c, fc, tipo))
            r.paso(f'x = {tex(c)}', '', r"f' \text{ pasa de } " + ('+' if izq == 1 else '-') + r'\text{ a } '
                   + ('+' if der == 1 else '-') + r'\Rightarrow\ \text{' + tipo + ' relativo en } (' + tex(c) + ',\\ '
                   + _tex_num(fc, 5) + ')', nivel + 1)
        else:
            r.paso(f'x = {tex(c)}', '', r"f' \text{ no cambia de signo} \Rightarrow \text{no hay extremo}", nivel + 1)
    return df, tramos, crit, extremos


def _analizar_concavidad(r, fr, vr, dom, ints, nivel=0):
    d2 = _simp(sp.diff(fr, vr, 2))
    r.paso('Derivada segunda', '', "f''(" + tex(vr) + ') = ' + tex(d2), nivel)
    ceros, nota = _raices(d2, vr, dom)
    nd = _puntos_no_derivables(d2, vr, dom, ints)
    cand = sorted(set(ceros) | set(nd), key=_num)
    texto = "Candidatos: donde f''(x) = 0" + (" o donde f'' no existe (dentro del dominio)." if nd else '.')
    if nota == 'numerica':
        texto += ' (Aproximados numéricamente en [-10, 10].)'
    elif nota == 'periodica':
        texto += ' (Se muestran los de [-2π, 2π].)'
    r.paso('Candidatos a punto de inflexión', texto, "f''(" + tex(vr) + r') = 0 \Rightarrow ' +
           (tex(vr) + ' = ' + ',\\ '.join(tex(c) for c in ceros) if ceros else r'\text{sin soluciones}')
           + ((r";\quad \nexists f''\ \text{en } " + ',\\ '.join(tex(c) for c in nd)) if nd else ''), nivel)
    tramos = _tramos(d2, vr, ints, cand)
    infl = []
    for c in cand:
        izq, der = _vecinos(tramos, c)
        fc = _valor_en(fr, vr, c)
        if izq is not None and der is not None and izq * der == -1:
            infl.append((c, fc))
            r.paso(f'x = {tex(c)}', '', r"f'' \text{ cambia de signo} \Rightarrow\ \text{punto de inflexión } (" + tex(c)
                   + ',\\ ' + _tex_num(fc, 5) + ')', nivel + 1)
        else:
            r.paso(f'x = {tex(c)}', '', r"f'' \text{ no cambia de signo} \Rightarrow \text{no es punto de inflexión}", nivel + 1)
    return d2, tramos, cand, infl


def _tabla_signos(tramos, nombre, mas, menos):
    filas = []
    for J, s_ in tramos:
        filas.append([_texto_tramo(J), '+' if s_ == 1 else ('−' if s_ == -1 else ('0' if s_ == 0 else '?')),
                      mas if s_ == 1 else (menos if s_ == -1 else '—')])
    return {'columnas': ['Intervalo', 'signo ' + nombre, 'f'], 'filas': filas}


def concavidad(arg):
    uso = 'concavidad f   o   inflexion f   (por ejemplo: inflexion x^3-3x^2)'
    f, v = _leer_funcion(dividir(arg)[0] if dividir(arg) else '', uso)
    vr = _real(v)
    fr = f.subs(v, vr)
    r = Resultado('Cálculo', 'Concavidad y puntos de inflexión')
    r.entrada_latex = 'f(' + tex(v) + ') = ' + tex(f)
    dom = _dominio(fr, vr)
    ints, recortado = _intervalos(dom)
    if dom != sp.S.Reals:
        r.paso('Dominio', '', r'\mathrm{Dom}\,f = ' + _tex_conjunto(dom), 0)
    if recortado:
        r.avisos.append('Se analiza el intervalo [-2π, 2π].')
    d2, tramos, cand, infl = _analizar_concavidad(r, fr, vr, dom, ints)
    r.tabla = _tabla_signos(tramos, "f''", 'cóncava hacia arriba ∪', 'cóncava hacia abajo ∩')
    arriba = [J for J, s_ in tramos if s_ == 1]
    abajo = [J for J, s_ in tramos if s_ == -1]
    r.paso('Conclusión', r"Donde \(f'' > 0\) la función es cóncava hacia arriba (convexa); donde \(f'' < 0\), "
           'cóncava hacia abajo.',
           r'\cup:\ ' + (r' \cup '.join(_ti(J.start, J.end) for J in arriba) or r'\varnothing') + r'\qquad \cap:\ '
           + (r' \cup '.join(_ti(J.start, J.end) for J in abajo) or r'\varnothing'), 0)
    xs = [c for c, _ in infl]
    r.valor = _plano(xs)
    r.valores['puntos'] = [sp.Tuple(*_plano((c, fc))) for c, fc in infl]
    r.valores['concava_arriba'] = _plano(_union([sp.Interval.open(J.start, J.end) for J in arriba]))
    r.valores['concava_abajo'] = _plano(_union([sp.Interval.open(J.start, J.end) for J in abajo]))
    r.valores['segunda_derivada'] = _plano(d2)
    r.resultado_latex = (r'\text{Puntos de inflexión: }' + ',\\ '.join('(' + tex(c) + ',\\ ' + tex(fc) + ')' for c, fc in infl)
                         if infl else r'\text{No tiene puntos de inflexión}')
    r.comparar = ('conjunto', _plano(xs), {})
    _graficar(r, [fr, d2], vr, [G.punto(c, fc, f'inflexión ({fmt_num(_num(c), 4)}, {fmt_num(_num(fc), 4)})')
                                for c, fc in infl], etiquetas=['f', "f''"], punteadas=[1])
    return r


# ----------------------------------------------------------------------------
# Estudio completo
# ----------------------------------------------------------------------------
def estudio_completo(arg):
    uso = 'estudio completo f   (por ejemplo: estudio completo x^3-3x)'
    f, v = _leer_funcion(dividir(arg)[0] if dividir(arg) else '', uso)
    vr = _real(v)
    fr = f.subs(v, vr)
    r = Resultado('Cálculo', 'Estudio completo de la función')
    r.entrada_latex = 'f(' + tex(v) + ') = ' + tex(f)
    # 1. dominio
    dom = _analizar_dominio(r, fr, vr, 0, '1. Dominio')
    ints, recortado = _intervalos(dom)
    if recortado:
        r.avisos.append('La función es periódica o tiene infinitas discontinuidades: el estudio se hace en [-2π, 2π].')
    # 2. paridad
    fm = _simp(fr.subs(vr, -vr))

    def simetria(signo):
        if _simp(fm - signo * fr) == 0:
            return True
        hay = False
        for c in (sp.Rational(37, 100), sp.Rational(13, 10), sp.Rational(21, 10), sp.Rational(7, 2)):
            if dom.contains(c) is not sp.true or dom.contains(-c) is not sp.true:
                continue
            a_, b_ = _num(_real_num(fr.subs(vr, -c))), _num(_real_num(fr.subs(vr, c)))
            if a_ is None or b_ is None or abs(a_ - signo * b_) > 1e-9 * (1 + abs(b_)):
                return False
            hay = True
        return hay
    if simetria(1):
        r.paso('2. Simetría', r'\(f(-x) = f(x)\): la función es par (simétrica respecto del eje y).', 'f(-' + tex(vr) + ') = ' + tex(fm), 0)
        paridad = 'par'
    elif simetria(-1):
        r.paso('2. Simetría', r'\(f(-x) = -f(x)\): la función es impar (simétrica respecto del origen).', 'f(-' + tex(vr) + ') = ' + tex(fm), 0)
        paridad = 'impar'
    else:
        r.paso('2. Simetría', 'No es par ni impar.', 'f(-' + tex(vr) + ') = ' + tex(fm), 0)
        paridad = 'ninguna'
    # 3. cortes con los ejes
    raices, nota = _raices(fr, vr, dom)
    corte_y = _valor_en(fr, vr, 0) if dom.contains(0) is sp.true else None
    r.paso('3. Intersecciones con los ejes',
           'Eje x: se resuelve f(x) = 0. Eje y: se calcula f(0) (si 0 está en el dominio).'
           + (' (Raíces aproximadas numéricamente.)' if nota == 'numerica' else ''),
           r'\text{Eje } x:\ ' + (',\\ '.join('(' + tex(c) + ',\\ 0)' for c in raices) if raices else r'\text{no corta}')
           + r'\qquad \text{Eje } y:\ ' + ('(0,\\ ' + tex(corte_y) + ')' if corte_y is not None else r'\text{no corta}'), 0)
    # 4. asíntotas
    r.paso('4. Asíntotas', '', '', 0)
    asin = _analizar_asintotas(r, fr, vr, dom, ints, recortado, nivel=1)
    # 5. monotonía y extremos
    r.paso('5. Crecimiento y extremos relativos', '', '', 0)
    df, tr1, crit, extremos = _analizar_monotonia(r, fr, vr, dom, ints, nivel=1)
    # 6. concavidad e inflexión
    r.paso('6. Concavidad y puntos de inflexión', '', '', 0)
    d2, tr2, cand, infl = _analizar_concavidad(r, fr, vr, dom, ints, nivel=1)
    # tabla combinada
    todos = sorted(set(crit) | set(cand), key=_num)
    filas = []
    crece, decrece, arriba, abajo = [], [], [], []
    for I in ints:
        for J in _partir(I, todos):
            m = _muestra(J.start, J.end)
            s1, s2 = _signo(df, vr, m), _signo(d2, vr, m)
            filas.append([_texto_tramo(J), '+' if s1 == 1 else ('−' if s1 == -1 else '0'),
                          '+' if s2 == 1 else ('−' if s2 == -1 else '0'),
                          ('crece' if s1 == 1 else ('decrece' if s1 == -1 else 'constante')) + ', ' +
                          ('cóncava ∪' if s2 == 1 else ('cóncava ∩' if s2 == -1 else 'recta'))])
    for J, s_ in tr1:
        (crece if s_ == 1 else decrece if s_ == -1 else []).append(J)
    for J, s_ in tr2:
        (arriba if s_ == 1 else abajo if s_ == -1 else []).append(J)
    r.tabla = {'columnas': ['Intervalo', "f'", "f''", 'Comportamiento'], 'filas': filas}
    r.paso('7. Resumen', '',
           r'\nearrow:\ ' + (r' \cup '.join(_ti(J.start, J.end) for J in crece) or r'\varnothing')
           + r'\qquad \searrow:\ ' + (r' \cup '.join(_ti(J.start, J.end) for J in decrece) or r'\varnothing'), 0)
    img = _con_tiempo(lambda: function_range(fr, vr, dom), 4) if not recortado else None
    if isinstance(img, sp.Set):
        img = _simp_conjunto(img)
        r.extra('Imagen', r'\mathrm{Im}\,f = ' + _tex_conjunto(img))
    # valores
    eqs, _ = _lista_asintotas(asin, vr)
    r.valores['dominio'] = _plano(dom)
    r.valores['paridad'] = paridad
    r.valores['raices'] = _plano(raices)
    if corte_y is not None:
        r.valores['corte_y'] = _plano(corte_y)
    r.valores['asintotas'] = _plano(eqs)
    r.valores['maximos'] = [sp.Tuple(*_plano((c, fc))) for c, fc, tp in extremos if tp == 'máximo']
    r.valores['minimos'] = [sp.Tuple(*_plano((c, fc))) for c, fc, tp in extremos if tp == 'mínimo']
    r.valores['inflexion'] = [sp.Tuple(*_plano((c, fc))) for c, fc in infl]
    r.valores['crece'] = _plano(_union([sp.Interval.open(J.start, J.end) for J in crece]))
    r.valores['decrece'] = _plano(_union([sp.Interval.open(J.start, J.end) for J in decrece]))
    xs = [c for c, _, _ in extremos]
    r.valor = _plano(xs)
    lineas_res = []
    if extremos:
        lineas_res.append(r'\text{Extremos: }' + ',\\ '.join(r'\text{' + tp[:3] + '. } (' + tex(c) + ',\\ ' + tex(fc) + ')'
                                                          for c, fc, tp in extremos))
    if infl:
        lineas_res.append(r'\text{Inflexión: }' + ',\\ '.join('(' + tex(c) + ',\\ ' + tex(fc) + ')' for c, fc in infl))
    lineas_res.append(_latex_asintotas(asin, vr))
    r.resultado_latex = r'\\ '.join(lineas_res)
    r.comparar = ('conjunto', _plano(xs), {})
    puntos = [G.punto(c, 0, 'raíz') for c in raices]
    if corte_y is not None:
        puntos.append(G.punto(0, corte_y, 'corte eje y'))
    puntos += [G.punto(c, fc, f'{tp} ({fmt_num(_num(c), 4)}, {fmt_num(_num(fc), 4)})') for c, fc, tp in extremos]
    puntos += [G.punto(c, fc, f'inflexión ({fmt_num(_num(c), 4)}, {fmt_num(_num(fc), 4)})', '#7c3aed') for c, fc in infl]
    lineas = [L for L, _ in asin['horizontales']] + [m * vr + b for m, b, _ in asin['oblicuas']]
    _graficar(r, [fr] + lineas, vr, puntos, etiquetas=['f'] + ['asíntota'] * len(lineas),
              punteadas=range(1, 1 + len(lineas)), verticales=asin['verticales'])
    return r


# ----------------------------------------------------------------------------
# Recta normal, diferencial, aproximación lineal
# ----------------------------------------------------------------------------
def recta_normal(arg):
    uso = 'recta normal f, a   (por ejemplo: recta normal x^2, 1)'
    partes = dividir(arg)
    if len(partes) < 2:
        raise _uso(uso)
    f, v = _leer_funcion(partes[0], uso)
    a = leer(partes[1])
    r = Resultado('Cálculo', 'Recta normal')
    r.entrada_latex = 'f(' + tex(v) + ') = ' + tex(f) + r',\quad ' + tex(v) + '_0 = ' + tex(a)
    fa = _simp(f.subs(v, a))
    if not _finito(fa):
        raise ErrorEntrada(f'La función no está definida en x = {a}.')
    df = _simp(sp.diff(f, v))
    m = _simp(df.subs(v, a))
    r.paso('Punto de la curva', '', f'f({tex(a)}) = ' + tex(fa), 0)
    r.paso('Pendiente de la tangente', '', "f'(" + tex(v) + ') = ' + tex(df) + r'\Rightarrow\ m_t = f\'(' + tex(a) + ') = ' + tex(m), 0)
    if m == 0:
        r.paso('Pendiente de la normal', 'La tangente es horizontal, así que la normal es la recta vertical por el punto.',
               tex(v) + ' = ' + tex(a), 0)
        r.valor = sp.Eq(v, a)
        r.resultado_latex = tex(v) + ' = ' + tex(a)
        r.comparar = ('conjunto', [a], {})
        r.extra('Recta tangente', 'y = ' + tex(fa))
        _graficar(r, [f, fa + 0 * v], v, [G.punto(a, fa)], ['f', 'tangente'], punteadas=[1], verticales=[a])
        return r
    if not _finito(m):
        r.paso('Pendiente de la normal', 'La tangente es vertical, así que la normal es horizontal.', 'y = ' + tex(fa), 0)
        normal = fa + 0 * v
    else:
        mn = _simp(-1 / m)
        r.paso('Pendiente de la normal', r'La normal es perpendicular a la tangente: \(m_n = -\dfrac{1}{m_t}\).',
               'm_n = -\\frac{1}{' + tex(m) + '} = ' + tex(mn), 0)
        normal = sp.expand(fa + mn * (v - a))
        r.paso('Ecuación punto-pendiente', r'\(y - f(x_0) = m_n\,(x - x_0)\)',
               'y - ' + tex(fa) + ' = ' + tex(mn) + r'\left(' + tex(v - a) + r'\right) \Rightarrow y = ' + tex(normal), 0)
        r.extra('Recta tangente', 'y = ' + tex(sp.expand(fa + m * (v - a))))
    r.valor = normal
    r.resultado_latex = 'y = ' + tex(normal)
    r.comparar = ('expr', normal, {})
    tang = sp.expand(fa + m * (v - a)) if _finito(m) else None
    curvas = [f, normal] + ([tang] if tang is not None else [])
    _graficar(r, curvas, v, [G.punto(a, fa, f'({fmt_num(_num(a), 4)}, {fmt_num(_num(fa), 4)})')],
              ['f', 'normal', 'tangente'], punteadas=[2], xmin=_num(a) - 3, xmax=_num(a) + 3)
    return r


DX = sp.Symbol('dx')
DY = sp.Symbol('dy')
DZ = sp.Symbol('dz')


def diferencial(arg):
    uso = 'diferencial f, a, dx   (por ejemplo: diferencial x^3, 2, 0.1)   o   diferencial f'
    partes = dividir(arg)
    if not partes:
        raise _uso(uso)
    f, v = _leer_funcion(partes[0], uso)
    df = _simp(sp.diff(f, v))
    r = Resultado('Cálculo', 'Diferencial')
    if len(partes) == 2:
        raise _uso(uso)
    if len(partes) < 3:
        r.entrada_latex = 'dy\\ \\text{de}\\ y = ' + tex(f)
        r.paso('Definición', r"El diferencial es \(dy = f'(x)\,dx\).", "f'(" + tex(v) + ') = ' + tex(df), 0)
        dy = df * DX
        r.valor = dy
        r.resultado_latex = 'dy = ' + tex(df) + r'\,dx'
        r.comparar = ('expr', dy, {})
        return r
    a, h = leer(coma_decimal(partes[1])), leer(coma_decimal(partes[2]))
    r.entrada_latex = 'y = ' + tex(f) + r',\quad ' + tex(v) + ' = ' + tex(a) + r',\quad dx = \Delta x = ' + tex(h)
    fpa = _simp(df.subs(v, a))
    dy = _simp(fpa * h)
    r.paso('Derivada', '', "f'(" + tex(v) + ') = ' + tex(df) + r'\Rightarrow\ f\'(' + tex(a) + ') = ' + tex(fpa), 0)
    r.paso('Diferencial', r"\(dy = f'(a)\,dx\)", 'dy = ' + tex(fpa) + r'\cdot ' + tex(h) + ' = ' + _tex_num(dy, 8), 0)
    Dy = _simp(f.subs(v, a + h) - f.subs(v, a))
    r.paso('Comparación con el incremento real', r'\(\Delta y = f(a + \Delta x) - f(a)\); el diferencial lo aproxima.',
           r'\Delta y = ' + _tex_num(Dy, 8) + r',\quad |\Delta y - dy| \approx ' + fmt_num(abs(_num(Dy - dy) or 0), 4), 0)
    r.valor = dy
    r.valores['decimal'] = _decimal_txt(dy)
    r.valores['incremento'] = Dy
    r.resultado_latex = 'dy = ' + _tex_num(dy, 8)
    r.comparar = ('numero_o_expr', dy, {})
    return r


def _elegir_punto(fx, xv, v0):
    """Elige el punto a 'cómodo' cerca de v0 donde f(a) y f'(a) son exactos y simples."""
    cands = set()
    base = int(math.floor(v0))
    radio = max(1.0, 0.25 * abs(v0))
    for i in range(base - int(radio) - 2, base + int(radio) + 3):
        cands.add(sp.Integer(i))
    if abs(v0) < 3:
        for i in range(-12, 13):
            cands.add(sp.Rational(i, 2))
    for i in range(-24, 25):
        cands.add(PI * sp.Rational(i, 12))
    mejores = []
    for a in cands:
        da = abs(float(sp.N(a)) - v0)
        if da < 1e-12 or da > radio + 1e-9:
            continue
        try:
            fa = sp.simplify(_real_num(fx.subs(xv, a)))
            dfa = sp.simplify(_real_num(sp.diff(fx, xv).subs(xv, a)))
        except Exception:  # noqa
            continue
        if not (_finito(fa) and _finito(dfa)):
            continue
        puntaje = 0 if fa.is_Rational else (1 if (fa / PI).is_Rational else 4)
        puntaje += 0 if dfa.is_Rational else 1
        puntaje += 0 if a.is_Integer else (1 if a.is_Rational else 1)
        mejores.append((puntaje, da, a))
    if not mejores:
        return None
    mejores.sort(key=lambda q: (q[0], q[1]))
    return mejores[0][2]


def aproximar(arg):
    uso = 'aproximar expresión[, a]   (por ejemplo: aproximar sqrt(4.1)   o   aproximar cbrt(26), 27)'
    texto = coma_decimal(normalizar(arg).strip())
    partes = dividir(texto)
    if not partes:
        raise _uso(uso)
    cuerpo = partes[0]
    a_dado = leer(partes[1]) if len(partes) > 1 else None
    # El número a aproximar: un literal decimal (o, si no hay, un entero que no sea exponente)
    decs = list(re.finditer(r'(?<![\w.])(\d+\.\d+|\.\d+)(?![\w.])', cuerpo))
    if len(decs) > 1:
        raise ErrorEntrada('Hay varios números decimales: no sé cuál aproximar. Usá linealizar f, a '
                           '(por ejemplo: linealizar sqrt(x), 4).')
    if decs:
        lit = decs[0]
    else:
        ents = [m for m in re.finditer(r'(?<![\w.])(\d+)(?![\w.])', cuerpo)
                if not re.search(r'(\^|\*\*)\s*\(?\s*-?\s*$', cuerpo[:m.start()]) and
                not re.search(r'(root|raiz)\s*\([^()]*,\s*$', cuerpo[:m.start()])]
        if not ents:
            raise ErrorEntrada('No encontré el número a aproximar. Ejemplo: aproximar sqrt(4.1)')
        lit = ents[0]
    if re.search(r'(?<![A-Za-z_])x(?![A-Za-z_])', cuerpo):
        raise ErrorEntrada('Escribí el número a aproximar (sin x). Para una función con variable usá: linealizar f, a')
    fx = _sin_piecewise_raiz(leer(cuerpo[:lit.start()] + '(x)' + cuerpo[lit.end():]))
    x0 = leer(lit.group(1))
    v0 = float(x0)
    a = a_dado if a_dado is not None else _elegir_punto(fx, x, v0)
    if a is None:
        raise ErrorEntrada('No encontré un punto cercano donde la función sea fácil de calcular. '
                           'Indicalo: aproximar expresión, a   (por ejemplo: aproximar sqrt(17), 16)')
    r = Resultado('Cálculo', 'Aproximación lineal (diferencial)')
    expr_num = tex(fx.subs(x, sp.Symbol(lit.group(1))))
    r.entrada_latex = r'\text{Aproximar } ' + expr_num
    fa = sp.simplify(_real_num(fx.subs(x, a)))
    df = _simp(sp.diff(fx, x))
    dfa = sp.simplify(_real_num(df.subs(x, a)))
    h = x0 - a
    r.paso('Función y punto conocido', 'Se elige un punto a cercano donde la función es fácil de evaluar.',
           'f(x) = ' + tex(fx) + r',\quad a = ' + tex(a) + r',\quad \Delta x = ' + tex(x0) + ' - ' + tex(a) + ' = ' + tex(h), 0)
    r.paso('Valores en a', '', 'f(' + tex(a) + ') = ' + tex(fa) + r',\quad f\'(x) = ' + tex(df) + r',\quad f\'(' + tex(a)
           + ') = ' + tex(dfa), 0)
    L = sp.expand(fa + dfa * (x - a))
    r.paso('Linealización', r"\(f(x) \approx L(x) = f(a) + f'(a)\,(x - a)\)", 'L(x) = ' + tex(L), 0)
    aprox = sp.nsimplify(sp.expand(fa + dfa * h)) if (fa + dfa * h).is_Rational else sp.expand(fa + dfa * h)
    real = _real_num(fx.subs(x, x0))
    r.paso('Aproximación', '',
           'f(' + tex(x0) + r') \approx ' + tex(fa) + ' + ' + tex(dfa) + r'\cdot\left(' + tex(h) + r'\right) = '
           + _tex_num(aprox, 8), 0)
    rd, ad = _num(real), _num(aprox)
    if rd is not None and ad is not None:
        r.paso('Error cometido', 'Comparando con el valor de la calculadora.',
               r'f(' + tex(x0) + r') \approx ' + fmt_num(rd, 10) + r',\quad \text{error} \approx ' + fmt_num(abs(rd - ad), 3), 0)
        r.valores['valor_real'] = fmt_num(rd, 12)
    r.valor = aprox
    r.valores['decimal'] = _decimal_txt(aprox)
    r.valores['linealizacion'] = L
    r.valores['a'] = a
    r.resultado_latex = expr_num + r' \approx ' + _tex_num(aprox, 8)
    r.comparar = ('numero_o_expr', aprox, {})
    _graficar(r, [fx, L], x, [G.punto(a, fa, 'a'), G.punto(x0, aprox, 'aprox.')], ['f', 'L(x)'], punteadas=[1],
              xmin=min(float(sp.N(a)), v0) - 2, xmax=max(float(sp.N(a)), v0) + 2)
    return r


def linealizar(arg):
    uso = 'linealizar f, a   (por ejemplo: linealizar sqrt(x), 4)'
    partes = dividir(coma_decimal(arg))
    if len(partes) < 2:
        raise _uso(uso)
    f, v = _leer_funcion(partes[0], uso)
    a = leer(partes[1])
    fa = _simp(f.subs(v, a))
    df = _simp(sp.diff(f, v))
    dfa = _simp(df.subs(v, a))
    if not (_finito(fa) and _finito(dfa)):
        raise ErrorEntrada(f'f no es derivable en {v} = {a}.')
    r = Resultado('Cálculo', 'Linealización')
    r.entrada_latex = 'L(' + tex(v) + r')\ \text{de}\ ' + tex(f) + r'\ \text{en}\ ' + tex(v) + ' = ' + tex(a)
    r.paso('Fórmula', '', r"L(x) = f(a) + f'(a)\,(x - a)", 0)
    r.paso('Valores en a', '', 'f(' + tex(a) + ') = ' + tex(fa) + r',\quad f\'(' + tex(a) + ') = ' + tex(dfa), 0)
    L = sp.expand(fa + dfa * (v - a))
    r.paso('Resultado', '', 'L(' + tex(v) + ') = ' + tex(fa) + ' + ' + tex(dfa) + r'\left(' + tex(v - a) + r'\right) = ' + tex(L), 0)
    r.valor = L
    r.resultado_latex = 'L(' + tex(v) + ') = ' + tex(L)
    r.comparar = ('expr', L, {})
    _graficar(r, [f, L], v, [G.punto(a, fa)], ['f', 'L'], punteadas=[1])
    return r


# ----------------------------------------------------------------------------
# Teoremas del valor medio y de Rolle
# ----------------------------------------------------------------------------
def _resolver_en(g, v, a, b):
    """Soluciones de g = 0 en (a, b)."""
    vr = _real(v)
    sol, nota = _raices(g.subs(v, vr), vr, sp.Interval.open(a, b))
    out = [c for c in sol if _num(c) is not None and float(sp.N(a)) < _num(c) < float(sp.N(b))]
    if not out and nota != 'numerica':
        out = [c for c in _raices_numericas(g.subs(v, vr), vr, sp.Interval.open(a, b), float(sp.N(a)), float(sp.N(b)))
               if float(sp.N(a)) < _num(c) < float(sp.N(b))]
    return out


def valor_medio(arg, rolle=False):
    uso = ('rolle f, a, b   (por ejemplo: rolle x^2-4x, 0, 4)' if rolle else
           'valor medio f, a, b   (por ejemplo: valor medio x^3, 0, 2)')
    partes = dividir(arg)
    if len(partes) < 3:
        raise _uso(uso)
    f, v = _leer_funcion(partes[0], uso)
    a, b = leer(partes[1]), leer(partes[2])
    if not (a.is_number and b.is_number) or not (_num(a) < _num(b)):
        raise ErrorEntrada('Los extremos deben ser números con a < b.')
    vr = _real(v)
    fr = f.subs(v, vr)
    r = Resultado('Cálculo', 'Teorema de Rolle' if rolle else 'Teorema del valor medio (Lagrange)')
    r.entrada_latex = 'f(' + tex(v) + ') = ' + tex(f) + r'\ \text{en}\ ' + _ti(a, b, True, True)
    dom = _dominio(fr, vr)
    continua = sp.Interval(a, b).is_subset(dom)
    df = _simp(sp.diff(fr, vr))
    dom_d = _dominio(df, vr)
    derivable = sp.Interval.open(a, b).is_subset(dom_d)
    r.paso('Hipótesis', 'f debe ser continua en [a, b] y derivable en (a, b).',
           r'\text{continua en } ' + _ti(a, b, True, True) + (r': \text{sí}' if continua else r': \text{no}')
           + r',\quad \text{derivable en } ' + _ti(a, b) + (r': \text{sí}' if derivable else r': \text{no}'), 0)
    if not continua or not derivable:
        r.avisos.append('No se cumplen las hipótesis del teorema: no se garantiza que exista c.')
    fa, fb = _simp(fr.subs(vr, a)), _simp(fr.subs(vr, b))
    if rolle:
        r.paso('Condición f(a) = f(b)', '', 'f(' + tex(a) + ') = ' + tex(fa) + r',\quad f(' + tex(b) + ') = ' + tex(fb), 0)
        if _simp(fa - fb) != 0:
            r.avisos.append('f(a) ≠ f(b): no se cumple la hipótesis de Rolle.')
        m = sp.Integer(0)
        r.paso('Tesis', r"Existe \(c \in (a, b)\) con \(f'(c) = 0\).", "f'(" + tex(vr) + ') = ' + tex(df) + ' = 0', 0)
    else:
        m = _simp((fb - fa) / (b - a))
        r.paso('Pendiente de la secante', r'\(\dfrac{f(b) - f(a)}{b - a}\)',
               r'\frac{' + tex(fb) + ' - ' + tex(fa) + '}{' + tex(b) + ' - ' + tex(a) + '} = ' + tex(m), 0)
        if _simp(fa - fb) == 0:
            r.paso('Observación', 'Como f(a) = f(b), es el caso particular del teorema de Rolle.', '', 0)
        r.paso('Tesis', r"Existe \(c \in (a, b)\) con \(f'(c)\) igual a esa pendiente.",
               "f'(c) = " + tex(df.subs(vr, sp.Symbol('c'))) + ' = ' + tex(m), 0)
    cs = _resolver_en(df - m, vr, a, b)
    r.paso('Valores de c', 'Se resuelve la ecuación y se conservan las soluciones del intervalo abierto.',
           'c = ' + (',\\ '.join(_tex_num(c, 6) for c in cs) if cs else r'\text{ninguno en el intervalo}'), 0)
    r.valor = _plano(cs)
    r.valores['pendiente'] = _plano(m)
    r.resultado_latex = ('c = ' + ',\\ '.join(_tex_num(c, 6) for c in cs)) if cs else r'\text{No existe } c \text{ en } ' + _ti(a, b)
    r.comparar = ('conjunto', _plano(cs), {})
    puntos = [G.punto(a, fa), G.punto(b, fb)] + [G.punto(c, fr.subs(vr, c), 'c') for c in cs]
    curvas = [fr, fa + m * (vr - a)] + [fr.subs(vr, c) + m * (vr - c) for c in cs[:2]]
    _graficar(r, curvas, vr, puntos, ['f', 'secante'] + ['tangente en c'] * len(cs[:2]), punteadas=[1],
              xmin=_num(a) - 0.5 * (_num(b) - _num(a)) - 0.5, xmax=_num(b) + 0.5 * (_num(b) - _num(a)) + 0.5)
    return r


def rolle(arg):
    return valor_medio(arg, rolle=True)


# ----------------------------------------------------------------------------
# Función inversa
# ----------------------------------------------------------------------------
def funcion_inversa(arg):
    uso = 'funcion inversa f   (por ejemplo: funcion inversa (2x+3)/(x-1))'
    f, v = _leer_funcion(dividir(arg)[0] if dividir(arg) else '', uso)
    if not f.has(v):
        raise ErrorEntrada('Una función constante no tiene inversa.')
    vr = _real(v)
    fr = f.subs(v, vr)
    Y = sp.Symbol('y')
    r = Resultado('Cálculo', 'Función inversa')
    r.entrada_latex = 'f(' + tex(v) + ') = ' + tex(f)
    dom = _dominio(fr, vr)
    ints, _ = _intervalos(dom)
    r.paso('Dominio de f', '', r'\mathrm{Dom}\,f = ' + _tex_conjunto(dom), 0)
    df = _simp(sp.diff(fr, vr))
    ceros, _ = _raices(df, vr, dom)
    tramos = _tramos(df, vr, ints, ceros)
    signos = {s_ for _, s_ in tramos if s_ in (1, -1)}
    restringido = None
    if len(signos) <= 1 and len(ints) == 1:
        r.paso('Inyectividad', "f' no cambia de signo en el dominio: f es estrictamente monótona, por lo tanto inyectiva.",
               "f'(" + tex(vr) + ') = ' + tex(df), 0)
    elif len(signos) <= 1:
        r.paso('Inyectividad', "f' no cambia de signo en cada intervalo del dominio. Se verifica al despejar que cada "
               'valor de y proviene de un único x.', "f'(" + tex(vr) + ') = ' + tex(df), 0)
    else:
        # rama "principal": el intervalo de monotonía que contiene a 0 o empieza en 0 (como arcsen, arccos, √x)
        J = next((J_ for J_, _ in tramos if J_.contains(0) is sp.true), None) or \
            next((J_ for J_, _ in tramos if J_.start == 0), None) or tramos[-1][0]
        restringido = sp.Interval(J.start, J.end, J.start == -oo or dom.contains(J.start) is not sp.true,
                                  J.end == oo or dom.contains(J.end) is not sp.true)
        r.paso('Inyectividad', 'f no es inyectiva en todo su dominio (crece y decrece). Se restringe a un intervalo '
               'donde es monótona para poder invertirla.',
               "f'(" + tex(vr) + ') = ' + tex(df) + r'\Rightarrow\ \text{se toma } ' + tex(vr) + r' \in '
               + _tex_conjunto(restringido), 0)
        r.avisos.append('La función no es inyectiva: la inversa corresponde a la restricción ' + str(restringido) + '.')
    dom_inv_f = restringido if restringido is not None else dom
    # despejar
    sols = _con_tiempo(lambda: sp.solve(sp.Eq(Y, f), v), 10)
    if not sols:
        raise ErrorEntrada('No pude despejar x de y = f(x) en forma exacta.')
    sols = [_sin_piecewise_raiz(s_) for s_ in sols if not s_.has(sp.I)] or sols
    img = _con_tiempo(lambda: function_range(fr, vr, dom_inv_f), 8)
    img = _simp_conjunto(img) if isinstance(img, sp.Set) else None
    # elegir la rama correcta: sus valores deben caer en el dominio (restringido) de f
    pruebas = []
    if img is not None:
        for I in (_intervalos(img)[0] if not isinstance(img, sp.FiniteSet) else []):
            m_ = _muestra(I.start, I.end)
            pruebas += [m_, (m_ + (I.end if I.end != oo else m_ + 3)) / 2]
    if not pruebas:
        pruebas = [sp.Rational(1, 2), sp.Integer(2), sp.Integer(5)]
    elegida = None
    for s_ in sols:
        ok = True
        for p in pruebas[:4]:
            val = _num(_real_num(s_.subs(Y, p)))
            if val is None or dom_inv_f.contains(sp.nsimplify(val) if abs(val) < 1e12 else val) is sp.false:
                ok = False
                break
            fv = _num(_real_num(_sin_piecewise_raiz(f).subs(v, sp.Float(val, 15))))
            if fv is None or abs(fv - float(sp.N(p))) > 1e-6 * (1 + abs(float(sp.N(p)))):
                ok = False
                break
        if ok:
            elegida = s_
            break
    if elegida is None:
        elegida = sols[0]
    r.paso('Despejar x', 'Se escribe y = f(x) y se despeja x en función de y.',
           'y = ' + tex(f) + r'\Rightarrow\ ' + tex(v) + ' = ' + tex(elegida)
           + ((r'\quad (\text{otras ramas: } ' + ',\\ '.join(tex(s_) for s_ in sols if s_ != elegida) + ')') if len(sols) > 1 else ''), 0)
    inv = _simp(elegida.subs(Y, v))
    r.paso('Intercambiar x e y', r'Se renombra la variable para escribir \(f^{-1}(x)\).', 'f^{-1}(' + tex(v) + ') = ' + tex(inv), 0)
    if img is not None:
        r.paso('Dominio e imagen de la inversa', r'\(\mathrm{Dom}\,f^{-1} = \mathrm{Im}\,f\) y \(\mathrm{Im}\,f^{-1} = \mathrm{Dom}\,f\).',
               r'\mathrm{Dom}\,f^{-1} = ' + _tex_conjunto(img) + r',\quad \mathrm{Im}\,f^{-1} = ' + _tex_conjunto(dom_inv_f), 0)
        r.valores['dominio_inversa'] = _plano(img)
    r.valores['imagen_inversa'] = _plano(dom_inv_f)
    comp = _con_tiempo(lambda: sp.simplify(f.subs(v, inv)), 6)
    if comp is not None:
        r.paso('Verificación', '', r'f\left(f^{-1}(x)\right) = ' + tex(comp), 0)
    r.valor = inv
    r.resultado_latex = 'f^{-1}(' + tex(v) + ') = ' + tex(inv)
    r.comparar = ('expr', inv, {})
    _graficar(r, [f, inv, v], v, etiquetas=['f', 'f⁻¹', 'y = x'], punteadas=[2], xmin=-8, xmax=8)
    if r.grafico:
        r.grafico['ventana'] = {'xmin': -8.0, 'xmax': 8.0, 'ymin': -6.0, 'ymax': 6.0}
    return r


# ============================================================================
# ANÁLISIS II: aplicaciones de la integral
# ============================================================================
def _integrar_def(h, v, a, b, seg=8.0):
    """(valor, exacto). Intenta la integral exacta con tiempo límite; si no, cuadratura numérica (mpmath)."""
    val = _con_tiempo(lambda: sp.integrate(h, (v, a, b)), seg)
    if val is not None and not val.has(sp.Integral) and _finito(val):
        val2 = _con_tiempo(lambda: simplificar_bonito(val), 4)
        return (val2 if val2 is not None else val), True
    if val is not None and _infinito(val):
        return val, True
    try:
        fl = sp.lambdify(v, h, 'mpmath')
        aa = mpmath.mpf(float(sp.N(a))) if a not in (oo, -oo) else (mpmath.inf if a == oo else -mpmath.inf)
        bb = mpmath.mpf(float(sp.N(b))) if b not in (oo, -oo) else (mpmath.inf if b == oo else -mpmath.inf)
        q = mpmath.quad(lambda s_: mpmath.re(fl(s_)), [aa, bb])
        return sp.Float(float(q), 15), False
    except Exception:  # noqa
        raise ErrorEntrada('No pude calcular la integral ni siquiera numéricamente.')


def _integral_abs(h, v, a, b):
    """∫|h| de a a b partiendo en los cambios de signo. Devuelve (valor, exacto, [(p, q, signo)])."""
    vr = _real(v)
    hr = h.subs(v, vr)
    cortes = [c for c in _raices(hr, vr, sp.Interval.open(a, b))[0] if _num(a) < _num(c) < _num(b)]
    bordes = [a] + cortes + [b]
    total, exacto, piezas = sp.Integer(0), True, []
    for i in range(len(bordes) - 1):
        p, q = bordes[i], bordes[i + 1]
        s_ = _signo(hr, vr, (p + q) / 2) or 1
        val, ex = _integrar_def(s_ * h, v, p, q)
        exacto = exacto and ex
        total += val
        piezas.append((p, q, s_))
    total = _simp(total) if exacto else sp.Float(float(sp.N(total)), 15)
    return total, exacto, piezas


def _variable_param(*exprs, defecto=t):
    libres = set()
    for e in exprs:
        libres |= getattr(e, 'free_symbols', set())
    for nombre in ('t', 'theta', 'u', 's', 'x'):
        for s_ in libres:
            if s_.name == nombre:
                return s_
    return sorted(libres, key=str)[0] if libres else defecto


def area_polar(arg):
    uso = 'area polar r, a, b   (por ejemplo: area polar 1+cos(t), 0, 2pi)   o   area polar r1, r2, a, b'
    partes = dividir(arg)
    if not partes:
        raise _uso(uso)
    if len(partes) == 1:
        partes += ['0', '2pi']
    if len(partes) not in (3, 4):
        raise _uso(uso)
    rs = [leer(p) for p in partes[:-2]]
    a, b = leer(partes[-2]), leer(partes[-1])
    th = _variable_param(*rs)
    r = Resultado('Análisis II', 'Área en coordenadas polares')
    if len(rs) == 1:
        r1, r2 = rs[0], sp.Integer(0)
        r.entrada_latex = r'A = \frac{1}{2}\int_{' + tex(a) + '}^{' + tex(b) + '} ' + r'\left(' + tex(r1) + r'\right)^2\,d' + tex(th)
        r.paso('Fórmula', r'El área barrida por \(r = f(\theta)\) entre \(\alpha\) y \(\beta\) es '
               r'\(A = \tfrac12\int_\alpha^\beta r^2\,d\theta\).', r.entrada_latex, 0)
    else:
        r1, r2 = rs
        r.entrada_latex = (r'A = \frac{1}{2}\int_{' + tex(a) + '}^{' + tex(b) + r'} \left[\left(' + tex(r1) + r'\right)^2 - \left('
                           + tex(r2) + r'\right)^2\right]d' + tex(th))
        r.paso('Fórmula', 'Área entre dos curvas polares (la exterior menos la interior).', r.entrada_latex, 0)
    integrando = _simp(sp.expand(r1 ** 2 - r2 ** 2) / 2)
    integrando_t = _con_tiempo(lambda: sp.expand(sp.expand_trig(integrando)), 4) or integrando
    r.paso('Integrando', 'Se desarrolla el cuadrado (en trigonométricas conviene usar '
           r'\(\cos^2\theta = \tfrac{1 + \cos 2\theta}{2}\)).', r'\frac{1}{2}\left(r^2\right) = ' + tex(integrando), 0)
    if len(rs) == 2:
        val, exacto, _ = _integral_abs(r1 ** 2 - r2 ** 2, th, a, b)
        val = _simp(val / 2) if exacto else val / 2
    else:
        val, exacto = _integrar_def(integrando_t, th, a, b)
    prim = _con_tiempo(lambda: sp.integrate(integrando_t, th), 6)
    if prim is not None and not prim.has(sp.Integral):
        r.paso('Primitiva', '', r'\int ' + tex(integrando) + r'\,d' + tex(th) + ' = ' + tex(_simp(prim)), 0)
    r.paso('Barrow', '', 'A = ' + _tex_num(val, 8), 0)
    if not exacto:
        r.avisos.append('La integral se calculó numéricamente.')
    r.valor = _plano(val)
    r.valores['decimal'] = _decimal_txt(val)
    r.resultado_latex = 'A = ' + _tex_num(val, 8)
    r.comparar = ('numero_o_expr', _plano(val), {})
    try:
        g = G.grafico_polar(_plano(r1), _plano(th), 'r = ' + tex(r1), float(sp.N(a)), float(sp.N(b)))
        if len(rs) == 2:
            g['curvas'].append({'fn': G.js(_plano(r2)), 'var': str(th), 'etiqueta': 'r = ' + tex(r2), 'color': G.COLORES[1],
                                'desde': float(sp.N(a)), 'hasta': float(sp.N(b))})
        r.grafico = g
    except Exception:  # noqa
        pass
    return r


def _raiz_simplificada(e, v):
    """sqrt(e) simplificada (1 + sinh² -> cosh, cuadrados perfectos -> |.|)."""
    vr = _real(v)
    er = e.subs(v, vr)
    cand = [er]
    for fn in (sp.simplify, sp.factor, lambda q: sp.trigsimp(sp.factor(q))):
        q = _con_tiempo(lambda: fn(er), 4)
        if q is not None:
            cand.append(q)
    mejores = []
    for c in cand:
        s_ = sp.sqrt(c)
        mejores.append(s_)
        q = _con_tiempo(lambda: sp.simplify(s_), 4)
        if q is not None:
            mejores.append(q)
    best = min(mejores, key=lambda q: (sp.count_ops(q), len(str(q))))
    return _plano(best)


def longitud_arco(arg):
    uso = ('longitud de arco f, a, b   |   longitud de arco (x(t), y(t)), a, b   |   longitud de arco x(t), y(t), a, b   |   '
           'longitud de arco polar r, a, b')
    texto = arg.strip()
    polar = False
    m = re.match(r'^\s*(en\s+)?polar(es)?\s+(.*)$', sin_tildes(texto), re.I)
    if m:
        polar = True
        texto = texto[m.start(3):]
    partes = dividir(texto)
    if len(partes) < 3:
        raise _uso(uso)
    r = Resultado('Análisis II', 'Longitud de arco')
    if polar:
        rr = leer(partes[0])
        th = _variable_param(rr)
        a, b = leer(partes[1]), leer(partes[2])
        drr = _simp(sp.diff(rr, th))
        r.entrada_latex = r'L = \int_{' + tex(a) + '}^{' + tex(b) + r'} \sqrt{r^2 + \left(\frac{dr}{d' + tex(th) + r'}\right)^2}\,d' + tex(th)
        r.paso('Fórmula (polares)', '', r.entrada_latex, 0)
        r.paso('Derivada', '', r"r' = " + tex(drr), 0)
        cuad = rr ** 2 + drr ** 2
        v, curva_xy = th, (rr * sp.cos(th), rr * sp.sin(th))
    elif _es_vector_texto(partes[0]) or len(partes) >= 4:
        if _es_vector_texto(partes[0]):
            comps = _leer_vector(partes[0], 'el vector r(t)')
            resto = partes[1:]
        else:
            comps = [leer(p) for p in partes[:-2]]
            resto = partes[-2:]
        if len(resto) != 2 or len(comps) not in (2, 3):
            raise _uso(uso)
        v = _variable_param(*comps)
        a, b = leer(resto[0]), leer(resto[1])
        ders = [_simp(sp.diff(c, v)) for c in comps]
        nombres = ['x', 'y', 'z'][:len(comps)]
        r.entrada_latex = r'L = \int_{' + tex(a) + '}^{' + tex(b) + r'} \sqrt{' + ' + '.join(
            f"{q}'({tex(v)})^2" for q in nombres) + r'}\,d' + tex(v)
        r.paso('Fórmula (curva paramétrica)', r'\(L = \int_a^b \|\vec r\,\'(t)\|\,dt\)', r.entrada_latex, 0)
        r.paso('Derivadas', '', r',\quad '.join(f"{q}' = " + tex(d) for q, d in zip(nombres, ders)), 0)
        cuad = sum(d ** 2 for d in ders)
        curva_xy = tuple(comps[:2])
    else:
        f, v = _leer_funcion(partes[0], uso)
        a, b = leer(partes[1]), leer(partes[2])
        df = _simp(sp.diff(f, v))
        r.entrada_latex = r'L = \int_{' + tex(a) + '}^{' + tex(b) + r'} \sqrt{1 + \left(f\'(' + tex(v) + r')\right)^2}\,d' + tex(v)
        r.paso('Fórmula', r'Para \(y = f(x)\): \(L = \int_a^b \sqrt{1 + f\'(x)^2}\,dx\).', r.entrada_latex, 0)
        r.paso('Derivada', '', "f'(" + tex(v) + ') = ' + tex(df), 0)
        cuad = 1 + df ** 2
        curva_xy = None
    integrando = _raiz_simplificada(cuad, v)
    r.paso('Integrando', 'Se simplifica la expresión dentro de la raíz.', r'\sqrt{' + tex(_simp(cuad)) + '} = ' + tex(integrando), 0)
    vr = _real(v)
    val, exacto = _integrar_def(integrando.subs(v, vr), vr, a, b)
    val = _plano(val)
    r.paso('Integrar', '' if exacto else 'No hay primitiva elemental: se integra numéricamente.',
           'L = ' + r'\int_{' + tex(a) + '}^{' + tex(b) + '} ' + tex(integrando) + r'\,d' + tex(v) + ' = ' + _tex_num(val, 8), 0)
    if not exacto:
        r.avisos.append('La integral se calculó numéricamente.')
    r.valor = val
    r.valores['decimal'] = _decimal_txt(val)
    r.resultado_latex = 'L = ' + _tex_num(val, 8)
    r.comparar = ('numero_o_expr', val, {})
    try:
        if curva_xy is None:
            g = G.grafico_2d([f], v, [G.punto(a, f.subs(v, a)), G.punto(b, f.subs(v, b))],
                             xmin=_num(a) - 0.3 * (_num(b) - _num(a)) - 0.5, xmax=_num(b) + 0.3 * (_num(b) - _num(a)) + 0.5)
        else:
            g = G.grafico_parametrica(_plano(curva_xy[0]), _plano(curva_xy[1]), _plano(v), 'curva', float(sp.N(a)), float(sp.N(b)))
        r.grafico = g
    except Exception:  # noqa
        pass
    return r


_RE_EJE = re.compile(r'^(?:alrededor\s+del?\s+|respecto\s+del?\s+|girando\s+(?:en\s+|alrededor\s+del?\s+)?)?(?:eje\s*)?'
                     r'(?:de\s+las\s+)?(x|y|abscisas|ordenadas)\s*(?:=\s*(.+))?$', re.I)


def _leer_eje(partes):
    """Si la última parte indica el eje, la quita. Devuelve (partes, ('h'|'v', c))."""
    if len(partes) >= 4:
        m = _RE_EJE.match(sin_tildes(partes[-1].strip()))
        if m:
            letra = {'abscisas': 'x', 'ordenadas': 'y'}.get(m.group(1).lower(), m.group(1).lower())
            c = leer(m.group(2)) if m.group(2) else sp.Integer(0)
            # eje x / recta y = c -> horizontal ; eje y / recta x = c -> vertical
            if m.group(2):
                tipo = 'h' if letra == 'y' else 'v'
            else:
                tipo = 'h' if letra == 'x' else 'v'
            return partes[:-1], (tipo, c)
    return partes, ('h', sp.Integer(0))


def _nombre_eje(eje):
    tipo, c = eje
    if c == 0:
        return 'del eje x' if tipo == 'h' else 'del eje y'
    return ('de la recta y = ' if tipo == 'h' else 'de la recta x = ') + str(c)


def _abs_tex(e, v, a, b):
    """|e| en LaTeX, sin las barras si e no cambia de signo y es positiva en [a, b]."""
    vr = _real(v)
    er = sp.sympify(e).subs(v, vr)
    try:
        cortes = [c for c in _raices(er, vr, sp.Interval.open(a, b))[0] if _num(a) < _num(c) < _num(b)]
        if not cortes and (_signo(er, vr, (a + b) / 2) or 0) >= 0:
            return tex(e)
    except Exception:  # noqa
        pass
    return r'\left|' + tex(e) + r'\right|'


def volumen_revolucion(arg):
    uso = ('volumen de revolucion f, a, b[, eje x | eje y | y=c | x=c]   o   volumen de revolucion f, g, a, b[, eje]   '
           '(por ejemplo: volumen de revolucion x^2, 0, 1, eje y)')
    partes, eje = _leer_eje(dividir(arg))
    if len(partes) not in (3, 4):
        raise _uso(uso)
    f = leer(partes[0])
    g = leer(partes[1]) if len(partes) == 4 else None
    a, b = leer(partes[-2]), leer(partes[-1])
    v = variables_de(f if g is None else f - g)
    v = v[0] if v else x
    if len(variables_de(f if g is None else f + g)) > 1:
        raise ErrorEntrada('Las funciones deben ser de una sola variable. Uso: ' + uso)
    tipo, c = eje
    r = Resultado('Análisis II', 'Volumen de un sólido de revolución')
    nombre = _nombre_eje(eje)
    if tipo == 'h':
        if g is None:
            g = c
        metodo = 'discos' if g == c else 'arandelas'
        h = sp.expand((f - c) ** 2 - (g - c) ** 2)
        if metodo == 'discos':
            r.entrada_latex = r'V = \pi\int_{' + tex(a) + '}^{' + tex(b) + r'} \left(' + tex(f - c) + r'\right)^2\,d' + tex(v)
            r.paso('Método de discos', f'Al girar alrededor {nombre}, cada sección es un disco de radio '
                   r'\(R = |f(x) - c|\): \(V = \pi\int_a^b R^2\,dx\).', r.entrada_latex, 0)
        else:
            r.entrada_latex = (r'V = \pi\int_{' + tex(a) + '}^{' + tex(b) + r'} \left[\left(' + tex(f - c) + r'\right)^2 - \left('
                               + tex(g - c) + r'\right)^2\right]d' + tex(v))
            r.paso('Método de arandelas', f'Al girar alrededor {nombre}, cada sección es una arandela: '
                   r'\(V = \pi\int_a^b (R^2 - r^2)\,dx\) con R el radio exterior y r el interior.', r.entrada_latex, 0)
        factor = PI
    else:
        if g is None:
            g = sp.Integer(0)
        metodo = 'capas'
        h = sp.expand((v - c) * (f - g))
        r.entrada_latex = (r'V = 2\pi\int_{' + tex(a) + '}^{' + tex(b) + r'} ' + _abs_tex(v - c, v, a, b) + r'\,\left('
                           + _abs_tex(f - g, v, a, b) + r'\right)\,d' + tex(v))
        r.paso('Método de capas (cilindros)', f'Al girar alrededor {nombre}, cada franja vertical genera una capa '
               r'cilíndrica de radio \(|x - c|\) y altura \(|f(x) - g(x)|\): \(V = 2\pi\int_a^b \text{radio}\cdot\text{altura}\,dx\).',
               r.entrada_latex, 0)
        factor = 2 * PI
    integ, exacto, piezas = _integral_abs(h, v, a, b)
    if len(piezas) > 1:
        r.paso('Cambio de signo', 'El integrando cambia de signo: se separa la integral en los puntos '
               + ', '.join(f'x = {tex(p)}' for p, _, _ in piezas[1:]) + '.', '', 1)
    integ_tex = tex(h) if len(piezas) == 1 else r'\left|' + tex(h) + r'\right|'
    r.paso('Integrar', '', r'\int_{' + tex(a) + '}^{' + tex(b) + '} ' + integ_tex + r'\,d' + tex(v) + ' = ' + _tex_num(integ, 8), 0)
    val = _simp(factor * integ) if exacto else sp.Float(float(sp.N(factor * integ)), 15)
    val = _plano(val)
    r.paso('Volumen', '','V = ' + tex(factor) + r'\cdot ' + tex(integ) + ' = ' + _tex_num(val, 8), 0)
    if not exacto:
        r.avisos.append('La integral se calculó numéricamente.')
    # método alternativo como dato
    r.extra('Método usado', '', f'{metodo.capitalize()}, girando alrededor {nombre}.')
    r.valor = val
    r.valores['decimal'] = _decimal_txt(val)
    r.valores['metodo'] = metodo
    r.resultado_latex = 'V = ' + _tex_num(val, 8)
    r.comparar = ('numero_o_expr', val, {})
    curvas = [f] + ([g] if g is not None and g != 0 else [])
    lineas = [c + 0 * v] if tipo == 'h' and c != 0 else []
    _graficar(r, curvas + lineas, v, [], etiquetas=['f', 'g'][:len(curvas)] + ['eje'] * len(lineas),
              punteadas=range(len(curvas), len(curvas) + len(lineas)), verticales=[c] if tipo == 'v' else [],
              xmin=min(_num(a), _num(c) if tipo == 'v' else _num(a)) - 0.5,
              xmax=max(_num(b), _num(c) if tipo == 'v' else _num(b)) + 0.5)
    if r.grafico:
        r.grafico['sombra'] = {'arriba': G.js(_plano(f)), 'abajo': G.js(_plano(g if g is not None else c)), 'var': str(v),
                               'desde': float(sp.N(a)), 'hasta': float(sp.N(b)), 'entre': True}
    return r


def superficie_revolucion(arg):
    uso = 'superficie de revolucion f, a, b[, eje x | eje y]   (por ejemplo: superficie de revolucion x, 0, 1)'
    partes, eje = _leer_eje(dividir(arg))
    if len(partes) != 3:
        raise _uso(uso)
    f, v = _leer_funcion(partes[0], uso)
    a, b = leer(partes[1]), leer(partes[2])
    tipo, c = eje
    df = _simp(sp.diff(f, v))
    r = Resultado('Análisis II', 'Área de una superficie de revolución')
    radio = (f - c) if tipo == 'h' else (v - c)
    r.entrada_latex = (r'S = 2\pi\int_{' + tex(a) + '}^{' + tex(b) + r'} \left|' + tex(radio) + r'\right|\sqrt{1 + \left('
                       + tex(df) + r'\right)^2}\,d' + tex(v))
    r.paso('Fórmula', f'Al girar la curva alrededor {_nombre_eje(eje)}: '
           r'\(S = 2\pi\int_a^b (\text{radio})\,\sqrt{1 + f\'(x)^2}\,dx\).', r.entrada_latex, 0)
    r.paso('Derivada', '', "f'(" + tex(v) + ') = ' + tex(df), 0)
    raiz = _raiz_simplificada(1 + df ** 2, v)
    vr = _real(v)
    vals = []
    integrando = _simp(radio * raiz)
    r.paso('Integrando', '', tex(radio) + r'\cdot ' + tex(raiz) + ' = ' + tex(integrando), 0)
    total, exacto, piezas = _integral_abs(integrando.subs(v, vr), vr, a, b)
    val = _simp(2 * PI * total) if exacto else sp.Float(float(sp.N(2 * PI * total)), 15)
    val = _plano(val)
    vals.append(val)
    r.paso('Integrar', '' if exacto else 'Se integra numéricamente.', 'S = 2\\pi\\cdot ' + _tex_num(_plano(total), 8)
           + ' = ' + _tex_num(val, 8), 0)
    if not exacto:
        r.avisos.append('La integral se calculó numéricamente.')
    r.valor = val
    r.valores['decimal'] = _decimal_txt(val)
    r.resultado_latex = 'S = ' + _tex_num(val, 8)
    r.comparar = ('numero_o_expr', val, {})
    _graficar(r, [f], v, [G.punto(a, f.subs(v, a)), G.punto(b, f.subs(v, b))],
              xmin=_num(a) - 0.5 * (_num(b) - _num(a)) - 0.5, xmax=_num(b) + 0.5 * (_num(b) - _num(a)) + 0.5)
    return r


def _cot(e):
    """1/tan(u) -> cot(u) (como se escribe en la cursada)."""
    try:
        return e.replace(lambda q: q.is_Pow and q.exp.is_negative and q.exp.is_Integer and isinstance(q.base, sp.tan),
                         lambda q: sp.cot(q.base.args[0]) ** (-q.exp))
    except Exception:  # noqa
        return e


def derivada_parametrica(arg):
    uso = 'derivada parametrica x(t), y(t)[, t0]   (por ejemplo: derivada parametrica cos(t), sen(t))'
    partes = dividir(arg)
    if partes and _es_vector_texto(partes[0]):
        partes = dividir(partes[0].strip()[1:-1]) + partes[1:]
    if len(partes) < 2:
        raise _uso(uso)
    X, Yp = leer(partes[0]), leer(partes[1])
    t0 = leer(partes[2]) if len(partes) > 2 else None
    v = _variable_param(X, Yp)
    r = Resultado('Análisis II', 'Derivada de una función paramétrica')
    r.entrada_latex = r'x = ' + tex(X) + r',\quad y = ' + tex(Yp)
    dx, dy = _simp(sp.diff(X, v)), _simp(sp.diff(Yp, v))
    r.paso('Derivadas respecto del parámetro', '', r'\frac{dx}{d' + tex(v) + '} = ' + tex(dx) + r',\quad \frac{dy}{d' + tex(v) + '} = ' + tex(dy), 0)
    if dx == 0:
        raise ErrorEntrada('dx/dt es idénticamente 0: la curva es una recta vertical.')
    d1 = _cot(_simp(dy / dx))
    r.paso('Primera derivada', r'\(\dfrac{dy}{dx} = \dfrac{dy/dt}{dx/dt}\)', r'\frac{dy}{dx} = \frac{' + tex(dy) + '}{' + tex(dx) + '} = ' + tex(d1), 0)
    d2 = _cot(_simp(sp.diff(d1, v) / dx))
    r.paso('Segunda derivada', r'\(\dfrac{d^2y}{dx^2} = \dfrac{\frac{d}{dt}\left(\frac{dy}{dx}\right)}{dx/dt}\)',
           r'\frac{d^2y}{dx^2} = ' + tex(d2), 0)
    r.valores['segunda'] = _plano(d2)
    if t0 is not None:
        val = _simp(d1.subs(v, t0))
        x0, y0 = _simp(X.subs(v, t0)), _simp(Yp.subs(v, t0))
        r.paso(f'Evaluar en {tex(v)} = {tex(t0)}', '', r'\left.\frac{dy}{dx}\right|_{' + tex(v) + '=' + tex(t0) + '} = '
               + tex(val) + r',\quad (x_0,\ y_0) = (' + tex(x0) + ',\\ ' + tex(y0) + ')', 0)
        if _finito(val):
            recta = sp.expand(y0 + val * (x - x0))
            r.extra('Recta tangente', 'y = ' + tex(recta))
            r.valores['tangente'] = recta
        r.valor = _plano(val)
        r.resultado_latex = r'\frac{dy}{dx}\Big|_{' + tex(v) + '=' + tex(t0) + '} = ' + tex(val)
        r.comparar = ('numero_o_expr', _plano(val), {})
    else:
        r.valor = _plano(d1)
        r.resultado_latex = r'\frac{dy}{dx} = ' + tex(d1)
        r.comparar = ('expr', _plano(d1), {})
    try:
        r.grafico = G.grafico_parametrica(_plano(X), _plano(Yp), _plano(v), 'curva')
        if t0 is not None:
            p = G.punto(X.subs(v, t0), Yp.subs(v, t0), f'{v} = {t0}')
            r.grafico['puntos'] = [p] if p else []
    except Exception:  # noqa
        pass
    return r


# ----------------------------------------------------------------------------
# Series numéricas: criterios de convergencia
# ----------------------------------------------------------------------------
NI = sp.Symbol('n', integer=True, positive=True)
NP = sp.Symbol('n', positive=True)
XP = sp.Symbol('x', positive=True)


def _lim_n(e, seg=6.0):
    return _limite(sp.sympify(e).subs(NI, NP), NP, oo, seg=seg)


def _tex_frac(e):
    """n!/n^n como fracción (sympy lo escribiría n^{-n} n!)."""
    e = _plano(e)
    try:
        num, den = sp.fraction(e)
        if den != 1:
            signo = ''
            if num.could_extract_minus_sign():
                signo, num = '-', -num
            return signo + r'\frac{' + tex(num) + '}{' + tex(den) + '}'
    except Exception:  # noqa
        pass
    return tex(e)


def _sumtex(a, n0):
    return r'\sum_{n=' + tex(n0) + r'}^{\infty} ' + _tex_frac(a)


def _primer_indice(a, n0):
    for j in range(int(n0), int(n0) + 6):
        if _num(a.subs(NI, j)) is not None:
            return sp.Integer(j)
    return sp.Integer(n0)


def _tiene_n_exponente(b):
    return any(isinstance(p, sp.Pow) and p.exp.has(NI) for p in sp.preorder_traversal(b))


def _simp_cociente(b):
    q = b.subs(NI, NI + 1) / b
    for fn in (lambda e: sp.simplify(sp.powsimp(sp.combsimp(e), force=True)), sp.simplify, lambda e: e):
        r_ = _con_tiempo(lambda: fn(q), 6)
        if r_ is not None:
            return r_
    return q


def _criterio_cociente(r, b, nivel):
    q = _simp_cociente(b)
    L = _lim_n(q)
    if L is None:
        return None, None
    r.paso("Criterio del cociente (D'Alembert)", r'\(L = \lim \dfrac{a_{n+1}}{a_n}\). Si L < 1 converge; si L > 1 diverge; si L = 1 no decide.',
           r'\frac{a_{n+1}}{a_n} = ' + tex(_plano(q)) + r'\ \Rightarrow\ L = ' + tex(L)
           + (r' < 1' if _finito(L) and L < 1 else (r' > 1' if (L == oo or (_finito(L) and L > 1)) else r' = 1')), nivel)
    if q.is_number or not q.has(NI):
        r.paso('Serie geométrica', r'El cociente es constante: es una serie geométrica de razón \(r = ' + tex(q) + r'\).', '', nivel + 1)
    if L == oo or (_finito(L) and L > 1):
        return 'diverge', L
    if _finito(L) and L < 1:
        return 'converge', L
    return None, L


def _criterio_raiz(r, b, nivel):
    raiz = _con_tiempo(lambda: sp.simplify(sp.powdenest(b ** (1 / NI), force=True)), 6) or b ** (1 / NI)
    L = _lim_n(raiz)
    if L is None:
        return None, None
    r.paso('Criterio de la raíz (Cauchy)', r'\(L = \lim \sqrt[n]{a_n}\). Si L < 1 converge; si L > 1 diverge; si L = 1 no decide.',
           r'\sqrt[n]{a_n} = ' + tex(_plano(raiz)) + r'\ \Rightarrow\ L = ' + tex(L), nivel)
    if L == oo or (_finito(L) and L > 1):
        return 'diverge', L
    if _finito(L) and L < 1:
        return 'converge', L
    return None, L


def _analizar_positiva(r, b, n0, nivel=0):
    """Serie de términos positivos Σ b. Devuelve (veredicto, criterio, dato)."""
    fact = b.has(sp.factorial, sp.gamma, sp.binomial, sp.RisingFactorial, sp.FallingFactorial)
    L0 = _lim_n(b, seg=2.5 if fact else 6)
    if L0 is not None and L0 != 0:
        r.paso('Condición necesaria', r'Si \(\lim a_n \neq 0\) la serie diverge.',
               r'\lim_{n\to\infty} ' + tex(_plano(b)) + ' = ' + tex(L0) + r' \neq 0 \Rightarrow \text{diverge}', nivel)
        return 'diverge', 'condición necesaria', L0
    if L0 == 0:
        r.paso('Condición necesaria', r'\(\lim a_n = 0\): se cumple, pero no alcanza para asegurar convergencia.',
               r'\lim_{n\to\infty} ' + tex(_plano(b)) + ' = 0', nivel)
    expo = _tiene_n_exponente(b)
    if fact or expo:
        orden = ['raiz', 'cociente'] if (expo and not fact and isinstance(b, sp.Pow) and b.base.has(NI) and b.exp.has(NI)) \
            else ['cociente', 'raiz']
        for crit in orden:
            ver, L = (_criterio_cociente if crit == 'cociente' else _criterio_raiz)(r, b, nivel)
            if ver:
                return ver, ('cociente' if crit == 'cociente' else 'raíz'), L
    if not fact:
        # comparación con p-serie
        p = _lim_n(-sp.log(b) / sp.log(NI))
        if _finito(p) and p.is_number:
            p = sp.nsimplify(p) if p.is_Float else p
            cte = _lim_n(sp.powsimp(b * NI ** p, force=True))
            es_p = _con_tiempo(lambda: sp.simplify(b * NI ** p) == 1, 4)
            if es_p:
                ver = 'converge' if p > 1 else 'diverge'
                r.paso('p-serie', r'\(\sum \frac{1}{n^p}\) converge si y solo si \(p > 1\).',
                       r'p = ' + tex(p) + (r' > 1' if p > 1 else r' \leq 1') + r'\Rightarrow\ \text{' + ver + '}', nivel)
                return ver, 'p-serie', p
            if cte is not None:
                if _finito(cte) and cte != 0:
                    ver = 'converge' if p > 1 else 'diverge'
                    r.paso('Comparación en el límite con una p-serie',
                           r'Se compara con \(\sum \frac{1}{n^{' + tex(p) + r'}}\): si el límite del cociente es finito y '
                           r'positivo, ambas series tienen el mismo carácter.',
                           r'\lim \frac{a_n}{1/n^{' + tex(p) + '}} = ' + tex(cte) + r',\quad p = ' + tex(p)
                           + (r' > 1' if p > 1 else r' \leq 1') + r'\Rightarrow\ \text{' + ver + '}', nivel)
                    return ver, 'comparación', p
                if cte == 0 and p > 1:
                    r.paso('Comparación con una p-serie', r'\(a_n\) es despreciable frente a \(1/n^{' + tex(p) + r'}\), '
                           r'cuya serie converge.', r'\lim \frac{a_n}{1/n^{' + tex(p) + '}} = 0', nivel)
                    return 'converge', 'comparación', p
                if cte == oo and p <= 1:
                    r.paso('Comparación con una p-serie', r'\(a_n\) es mayor que \(1/n^{' + tex(p) + r'}\), cuya serie diverge.',
                           r'\lim \frac{a_n}{1/n^{' + tex(p) + '}} = \infty', nivel)
                    return 'diverge', 'comparación', p
        # criterio integral
        fx = b.subs(NI, XP)
        I = _con_tiempo(lambda: sp.integrate(fx, (XP, n0, oo)), 8)
        if I is not None and not I.has(sp.Integral):
            ver = 'converge' if _finito(I) else ('diverge' if _infinito(I) else None)
            if ver:
                r.paso('Criterio de la integral', r'f(x) es positiva y decreciente: la serie y \(\int_{' + tex(n0)
                       + r'}^{\infty} f(x)\,dx\) tienen el mismo carácter.',
                       r'\int_{' + tex(n0) + r'}^{\infty} ' + tex(_plano(fx)) + r'\,dx = ' + tex(I) + r'\Rightarrow\ \text{' + ver + '}', nivel)
                return ver, 'integral', I
    if not (fact or expo):
        ver, L = _criterio_cociente(r, b, nivel)
        if ver:
            return ver, 'cociente', L
    res = _con_tiempo(lambda: sp.Sum(b, (NI, n0, oo)).is_convergent(), 8)
    if res is True or res is False:
        ver = 'converge' if res else 'diverge'
        r.paso('Conclusión por comparación', 'Combinando criterios de comparación (cálculo simbólico).',
               r'\text{La serie ' + ver + '}', nivel)
        return ver, 'comparación', None
    return None, None, None


def _decreciente(b, n0):
    """¿b_n es decreciente (a partir de cierto n)?"""
    vals = [_num(b.subs(NI, j)) for j in range(int(n0), int(n0) + 40)]
    if any(v is None for v in vals):
        return None
    desde = None
    for i in range(len(vals) - 1):
        if vals[i + 1] > vals[i] + 1e-15:
            desde = None
        elif desde is None:
            desde = int(n0) + i
    if desde is None:
        return False
    if b.has(sp.factorial, sp.gamma):
        return desde
    d = _con_tiempo(lambda: sp.diff(b.subs(NI, XP), XP), 4)
    if d is not None:
        for q in (100, 1000, 10 ** 5):
            s_ = _num(d.subs(XP, q))
            if s_ is not None and s_ > 0:
                return False
    return desde


def _analizar_serie(r, a, n0, nivel=0):
    """Devuelve (veredicto, tipo) con veredicto 'converge'/'diverge'/None y tipo 'absolutamente'/'condicionalmente'/''."""
    vals = [_num(a.subs(NI, j)) for j in range(int(n0), int(n0) + 14)]
    signos = [1 if v > 0 else -1 for v in vals if v is not None and v != 0]
    if not signos:
        return None, ''
    if all(s_ == 1 for s_ in signos):
        r.paso('Tipo de serie', 'Serie de términos positivos.', _sumtex(a, n0), nivel)
        ver, crit, dato = _analizar_positiva(r, a, n0, nivel)
        return ver, ''
    if all(s_ == -1 for s_ in signos):
        r.paso('Tipo de serie', 'Todos los términos son negativos: se estudia la serie de los opuestos (mismo carácter).',
               _sumtex(-a, n0), nivel)
        ver, crit, dato = _analizar_positiva(r, _simp(-a), n0, nivel)
        return ver, ''
    alterna = all(signos[i] == -signos[i + 1] for i in range(len(signos) - 1))
    b = _simp(sp.Abs(a))
    if b.has(sp.Abs):
        for cand in (a * (-1) ** NI, -a * (-1) ** NI):
            cand = _simp(cand)
            if all((_num(cand.subs(NI, j)) or -1) >= 0 for j in range(int(n0), int(n0) + 8)):
                b = cand
                break
    if alterna:
        r.paso('Tipo de serie', r'Serie alternada: \(a_n = (-1)^n b_n\) con \(b_n = ' + tex(_plano(b)) + r'\).', _sumtex(a, n0), nivel)
        L0 = _lim_n(b)
        if L0 is not None and L0 != 0:
            r.paso('Condición necesaria', r'\(b_n\) no tiende a 0, entonces \(a_n\) tampoco: la serie diverge.',
                   r'\lim b_n = ' + tex(L0), nivel)
            return 'diverge', ''
        r.paso('Convergencia absoluta', r'Se estudia \(\sum |a_n| = \sum b_n\).', _sumtex(b, n0), nivel)
        ver, crit, dato = _analizar_positiva(r, b, n0, nivel + 1)
        if ver == 'converge':
            r.paso('Conclusión', 'La serie de los módulos converge: la serie converge absolutamente.', '', nivel)
            return 'converge', 'absolutamente'
        dec = _decreciente(b, n0)
        if L0 == 0 and dec:
            r.paso('Criterio de Leibniz', r'\(b_n\) es decreciente' + (f' (desde n = {dec})' if dec and dec > n0 else '')
                   + r' y \(b_n \to 0\): la serie alternada converge.', r'b_{n+1} \leq b_n,\quad \lim b_n = 0', nivel)
            if ver == 'diverge':
                r.paso('Conclusión', 'Converge, pero la serie de los módulos diverge: converge condicionalmente.', '', nivel)
                return 'converge', 'condicionalmente'
            return 'converge', ''
        r.paso('Criterio de Leibniz', 'No se pudo verificar que los módulos decrezcan hacia 0.', '', nivel)
        return None, ''
    # signo variable: comparación con la serie de módulos acotados
    r.paso('Tipo de serie', 'Términos de signo variable: se estudia la convergencia absoluta.', _sumtex(a, n0), nivel)
    cota = sp.Abs(a).replace(lambda e: isinstance(e, sp.Abs) and e.args[0].func in (sp.sin, sp.cos), lambda e: sp.Integer(1))
    cota = cota.replace(lambda e: isinstance(e, (sp.sin, sp.cos)), lambda e: sp.Integer(1))
    cota = _simp(cota)
    if not cota.has(sp.Abs) and cota != b:
        r.paso('Acotación', r'\(|\operatorname{sen}|\leq 1\) y \(|\cos|\leq 1\):', r'|a_n| \leq ' + tex(_plano(cota)), nivel)
    ver, crit, dato = _analizar_positiva(r, cota if not cota.has(sp.Abs) else b, n0, nivel + 1)
    if ver == 'converge':
        r.paso('Conclusión', 'La serie de los módulos converge: la serie converge absolutamente.', '', nivel)
        return 'converge', 'absolutamente'
    res = _con_tiempo(lambda: sp.Sum(a, (NI, n0, oo)).is_convergent(), 8)
    if res is True or res is False:
        return ('converge' if res else 'diverge'), ''
    return None, ''


def _leer_serie(arg, uso):
    partes = dividir(arg)
    if not partes:
        raise _uso(uso)
    a = leer(partes[0])
    var = n if n in a.free_symbols else (k if k in a.free_symbols else None)
    n0 = sp.Integer(1)
    if len(partes) == 2:
        n0 = leer(partes[1])
    elif len(partes) >= 3:
        var = leer(partes[1])
        n0 = leer(partes[2])
    if var is None:
        raise ErrorEntrada('El término general debe depender de n (o de k). ' + 'Uso: ' + uso)
    return a.subs(var, NI), n0


def criterio_serie(arg):
    uso = 'criterio serie a_n[, n0]   (por ejemplo: criterio serie n!/n^n   o   criterio serie (-1)^n/n)'
    a, n0 = _leer_serie(arg, uso)
    if a.has(x):
        raise ErrorEntrada('El término tiene x: para series de potencias usá «radio de convergencia».')
    n0 = _primer_indice(a, n0)
    r = Resultado('Análisis II', 'Convergencia de una serie')
    r.entrada_latex = _sumtex(a, n0)
    ver, tipo = _analizar_serie(r, a, n0)
    if ver is None:
        r.avisos.append('Ningún criterio permitió decidir.')
        r.valor = 'no se pudo determinar'
        r.resultado_latex = r'\text{No se pudo determinar el carácter de la serie}'
        r.comparar = None
        return r
    texto = ver + (' ' + tipo if tipo else '')
    r.valor = ver
    r.valores['tipo'] = tipo or ''
    if ver == 'converge':
        s_ = _con_tiempo(lambda: sp.summation(_plano(a).subs(n, NI), (NI, n0, oo)), 3)
        if s_ is not None and _finito(s_) and not s_.has(sp.Sum):
            r.extra('Suma de la serie', _sumtex(a, n0) + ' = ' + _tex_num(s_, 8))
            r.valores['suma'] = _plano(s_)
    r.resultado_latex = _sumtex(a, n0) + r'\ \text{' + texto + '}'
    # 'texto' acepta diverge/no existe; para 'converge' hace falta el modo 'convergencia' (ver informe)
    r.comparar = ('texto', 'diverge', {}) if ver == 'diverge' else ('convergencia', texto, {})
    return r


def radio_convergencia(arg):
    uso = 'radio de convergencia a_n(x)   (por ejemplo: radio de convergencia x^n/n   o   radio de convergencia (x-2)^n/(n*3^n))'
    a, n0 = _leer_serie(arg, uso)
    if not a.has(x):
        raise ErrorEntrada('El término general debe tener x (serie de potencias). Para series numéricas usá «criterio serie».')
    n0 = _primer_indice(a.subs(x, sp.Rational(1, 3)), n0)
    xr = sp.Symbol('x', real=True)
    a = a.subs(x, xr)
    r = Resultado('Análisis II', 'Radio e intervalo de convergencia')
    r.entrada_latex = r'\sum_{n=' + tex(n0) + r'}^{\infty} ' + tex(_plano(a))
    q = _simp_cociente(a)
    Lx = _lim_n(sp.Abs(q))
    if Lx is None or (Lx.has(sp.Limit)):
        Lx = _lim_n(sp.Abs(a) ** (1 / NI))
        r.paso('Criterio de la raíz', r'\(L(x) = \lim \sqrt[n]{|a_n(x)|}\); la serie converge absolutamente si L(x) < 1.',
               'L(x) = ' + tex(_plano(Lx)) if Lx is not None else '', 0)
    else:
        r.paso('Criterio del cociente', r'\(L(x) = \lim \left|\dfrac{a_{n+1}(x)}{a_n(x)}\right|\); la serie converge '
               r'absolutamente si \(L(x) < 1\) y diverge si \(L(x) > 1\).',
               r'\left|\frac{a_{n+1}}{a_n}\right| = \left|' + tex(_plano(q)) + r'\right| \Rightarrow\ L(x) = ' + tex(_plano(Lx)), 0)
    if Lx is None:
        raise ErrorEntrada('No pude calcular el límite del criterio del cociente para esta serie.')
    centro_cands = _raices(a.subs(NI, 1), xr)[0] if a.subs(NI, 1).has(xr) else []
    if not Lx.has(xr):
        if Lx == 0:
            R, intervalo = oo, sp.S.Reals
            r.paso('Radio', 'L(x) = 0 < 1 para todo x: converge en toda la recta real.', r'R = \infty', 0)
        elif _infinito(Lx) or (_finito(Lx) and Lx > 1):
            R = sp.Integer(0)
            c = centro_cands[0] if centro_cands else sp.Integer(0)
            intervalo = sp.FiniteSet(c)
            r.paso('Radio', 'L(x) es infinito (o mayor que 1) para todo x distinto del centro: solo converge en el centro.',
                   'R = 0,\\ x = ' + tex(c), 0)
        else:
            raise ErrorEntrada('El criterio no depende de x: revisá el término general.')
    elif Lx.has(oo) or Lx.has(sp.sign):
        R = sp.Integer(0)
        c = centro_cands[0] if centro_cands else sp.Integer(0)
        intervalo = sp.FiniteSet(c)
        r.paso('Radio', 'El límite es infinito para todo x distinto del centro: la serie solo converge en x = ' + str(c) + '.',
               'R = 0', 0)
    else:
        sol = _con_tiempo(lambda: sp.solveset(Lx < 1, xr, sp.S.Reals), 8)
        if not isinstance(sol, sp.Interval) or sol.start in (-oo, oo) or sol.end in (-oo, oo):
            if sol == sp.S.Reals:
                R, intervalo = oo, sp.S.Reals
                r.paso('Radio', 'L(x) < 1 para todo x.', r'R = \infty', 0)
            else:
                raise ErrorEntrada('No pude despejar el intervalo de L(x) < 1 (resultado: ' + str(sol) + ').')
        else:
            izq, der = sol.start, sol.end
            R = _simp((der - izq) / 2)
            c = _simp((der + izq) / 2)
            r.paso('Desigualdad L(x) < 1', 'Se despeja x: la serie converge absolutamente en el intervalo abierto.',
                   tex(_plano(Lx)) + r' < 1 \Rightarrow ' + _ti(izq, der) + r'\ \Rightarrow\ R = ' + tex(R)
                   + r',\ \text{centro } ' + tex(c), 0)
            r.paso('Extremos del intervalo', 'En los extremos el criterio da L = 1 y no decide: se analiza cada serie numérica.', '', 0)
            incluye = []
            for e in (izq, der):
                ae = _simp(sp.powsimp(a.subs(xr, e), force=True))
                r.paso(f'x = {tex(e)}', '', r'\sum ' + tex(_plano(ae)), 1)
                ver, tipo = _analizar_serie(r, ae, n0, nivel=2)
                incluye.append(ver == 'converge')
                r.paso(f'En x = {tex(e)}', '', r'\text{' + (ver or 'no se pudo decidir') + (' ' + tipo if tipo else '') + '}', 1)
            intervalo = sp.Interval(izq, der, not incluye[0], not incluye[1])
    r.valor = _plano(R)
    r.valores['intervalo'] = _plano(intervalo)
    r.resultado_latex = 'R = ' + tex(R) + r',\quad I = ' + _tex_conjunto(intervalo)
    r.comparar = ('numero_o_expr', _plano(R), {})
    return r


# ============================================================================
# ANÁLISIS III: varias variables y cálculo vectorial
# ============================================================================
def _vars_campo(exprs, minimo=2):
    vs = variables_de(sp.Tuple(*exprs))
    base = [x, y, z]
    if set(vs) <= set(base):
        nn = max([base.index(v) + 1 for v in vs] + [minimo])
        return base[:nn]
    return vs


def derivada_direccional(arg):
    uso = 'derivada direccional f, punto, vector   (por ejemplo: derivada direccional x^2*y, (1,2), (3,4))'
    partes = dividir(arg)
    if len(partes) < 3:
        raise _uso(uso)
    f = leer(partes[0])
    P = _leer_vector(partes[1], 'el punto')
    u = _leer_vector(partes[2], 'el vector dirección')
    vs = _vars_campo([f], len(P))
    if len(P) != len(vs) or len(u) != len(vs):
        raise ErrorEntrada(f'El punto y el vector deben tener {len(vs)} componentes (variables: {", ".join(map(str, vs))}).')
    r = Resultado('Análisis II', 'Derivada direccional')
    r.entrada_latex = r'D_{\vec u} f\ \text{para}\ f = ' + tex(f) + r'\ \text{en}\ ' + _vector_tex(P) + r',\ \vec v = ' + _vector_tex(u)
    grad = [_simp(sp.diff(f, v)) for v in vs]
    r.paso('Gradiente', '', r'\nabla f = ' + _vector_tex(grad), 0)
    sub = dict(zip(vs, P))
    gp = [_simp(g.subs(sub)) for g in grad]
    r.paso('Gradiente en el punto', '', r'\nabla f' + _vector_tex(P) + ' = ' + _vector_tex(gp), 0)
    nu = _simp(_norma(u))
    if nu == 0:
        raise ErrorEntrada('El vector dirección no puede ser nulo.')
    uu = [_simp(c / nu) for c in u]
    if nu != 1:
        r.paso('Versor de la dirección', r'La derivada direccional usa un vector unitario: \(\vec u = \vec v / \|\vec v\|\).',
               r'\|\vec v\| = ' + tex(nu) + r'\Rightarrow\ \vec u = ' + _vector_tex(uu), 0)
    val = _simp(sum(a_ * b_ for a_, b_ in zip(gp, uu)))
    r.paso('Producto escalar', r'\(D_{\vec u} f = \nabla f \cdot \vec u\)',
           ' + '.join(f'{tex(a_)}\\cdot {tex(b_)}' for a_, b_ in zip(gp, uu)) + ' = ' + _tex_num(val, 8), 0)
    ng = _simp(_norma(gp))
    r.extra('Máxima derivada direccional', r'\|\nabla f\| = ' + _tex_num(ng, 6),
            'Se alcanza en la dirección del gradiente.')
    r.valor = val
    r.valores['decimal'] = _decimal_txt(val)
    r.valores['gradiente'] = gp
    r.valores['maxima'] = ng
    r.resultado_latex = r'D_{\vec u} f = ' + _tex_num(val, 8)
    r.comparar = ('numero_o_expr', val, {})
    return r


def diferencial_total(arg):
    uso = ('diferencial total f[, punto, dx, dy]   (por ejemplo: diferencial total x^2*y, (1,2), 0.01, 0.02)')
    partes = dividir(arg)
    if not partes:
        raise _uso(uso)
    f = leer(partes[0])
    vs = _vars_campo([f])
    difs = [sp.Symbol('d' + v.name) for v in vs]
    r = Resultado('Análisis II', 'Diferencial total')
    parciales = [_simp(sp.diff(f, v)) for v in vs]
    r.paso('Derivadas parciales', '', r',\quad '.join(r'f_{' + tex(v) + '} = ' + tex(p) for v, p in zip(vs, parciales)), 0)
    if len(partes) == 1:
        r.entrada_latex = 'df\\ \\text{de}\\ f = ' + tex(f)
        df = sum(p * d for p, d in zip(parciales, difs))
        r.paso('Diferencial total', r'\(df = f_x\,dx + f_y\,dy\,(+ f_z\,dz)\)', 'df = ' + tex(df), 0)
        r.valor = df
        r.resultado_latex = 'df = ' + ' + '.join(r'\left(' + tex(p) + r'\right)d' + tex(v) for p, v in zip(parciales, vs))
        r.comparar = ('expr', df, {})
        return r
    P = _leer_vector(partes[1], 'el punto')
    incs = [leer(p) for p in partes[2:]]
    if len(P) != len(vs) or len(incs) != len(vs):
        raise ErrorEntrada(f'Indicá el punto y {len(vs)} incrementos. Uso: ' + uso)
    sub = dict(zip(vs, P))
    r.entrada_latex = 'df\\ \\text{de}\\ ' + tex(f) + r'\ \text{en}\ ' + _vector_tex(P) + ',\\ ' + ',\\ '.join(
        'd' + tex(v) + ' = ' + tex(h) for v, h in zip(vs, incs))
    pp = [_simp(p.subs(sub)) for p in parciales]
    r.paso('Parciales en el punto', '', r',\quad '.join(r'f_{' + tex(v) + '} = ' + tex(p) for v, p in zip(vs, pp)), 0)
    df = _simp(sum(p * h for p, h in zip(pp, incs)))
    r.paso('Diferencial total', r'\(df = f_x\,\Delta x + f_y\,\Delta y\)',
           'df = ' + ' + '.join(f'{tex(p)}\\cdot {tex(h)}' for p, h in zip(pp, incs)) + ' = ' + _tex_num(df, 8), 0)
    Df = _simp(f.subs({v: c + h for v, c, h in zip(vs, P, incs)}) - f.subs(sub))
    r.paso('Incremento real', r'\(\Delta f = f(P + \Delta) - f(P)\); el diferencial lo aproxima.',
           r'\Delta f = ' + _tex_num(Df, 8), 0)
    r.valor = df
    r.valores['decimal'] = _decimal_txt(df)
    r.valores['incremento'] = Df
    r.resultado_latex = 'df = ' + _tex_num(df, 8)
    r.comparar = ('numero_o_expr', df, {})
    return r


# ----------------------------------------------------------------------------
# Límite doble
# ----------------------------------------------------------------------------
def _leer_limite_doble(arg):
    texto = normalizar(arg)
    partes = dividir(texto)
    con_flecha = [p for p in partes if '->' in p]
    if len(con_flecha) == 1:
        izq, der = con_flecha[0].split('->', 1)
        vs, pt = _leer_vector(izq, 'el par de variables (x, y)'), _leer_vector(der, 'el punto')
        cuerpo = [p for p in partes if p is not con_flecha[0]]
    elif len(con_flecha) == 2:
        vs, pt = [], []
        for p in con_flecha:
            a_, b_ = p.split('->', 1)
            vs.append(leer(a_))
            pt.append(leer(b_))
        cuerpo = [p for p in partes if '->' not in p]
    else:
        raise _uso('limite doble f, (x,y)->(a,b)   (por ejemplo: limite doble x*y/(x^2+y^2), (x,y)->(0,0))')
    if len(vs) != 2 or len(pt) != 2:
        raise ErrorEntrada('El límite doble es en dos variables: (x,y)->(a,b).')
    f = leer(','.join(cuerpo))
    return f, vs, pt


def _distintos(a, b):
    if _infinito(a) or _infinito(b):
        return a != b
    return _simp(a - b) != 0


def limite_doble(arg):
    f, (vx, vy), (a, b) = _leer_limite_doble(arg)
    r = Resultado('Análisis II', 'Límite doble')
    r.entrada_latex = r'\lim_{(' + tex(vx) + ',' + tex(vy) + r')\to(' + tex(a) + ',' + tex(b) + ')} ' + tex(f)
    # 1. sustitución directa
    directo = _con_tiempo(lambda: sp.simplify(f.subs({vx: a, vy: b})), 4)
    den = sp.denom(sp.together(f))
    den0 = _con_tiempo(lambda: sp.simplify(den.subs({vx: a, vy: b})), 4)
    if directo is not None and _finito(directo) and directo.is_number and den0 is not None and den0 != 0:
        r.paso('Sustitución directa', 'La función es continua en el punto (el denominador no se anula): se reemplaza.',
               r.entrada_latex + ' = ' + tex(directo), 0)
        r.valor = directo
        r.resultado_latex = r.entrada_latex + ' = ' + tex(directo)
        r.comparar = ('numero_o_expr', directo, {})
        _grafico_3d(r, f, vx, vy)
        return r
    r.paso('Indeterminación', 'Al reemplazar queda una forma indeterminada: se prueba acercarse por distintos caminos. '
           'Si dos caminos dan límites distintos, el límite no existe.', '', 0)
    s_ = sp.Symbol('s', real=True)
    m_ = sp.Symbol('m', real=True)
    k_ = sp.Symbol('k', real=True)
    caminos = [
        ('Eje paralelo a x', r'y = ' + tex(b), {vx: a + s_, vy: b}),
        ('Eje paralelo a y', r'x = ' + tex(a), {vx: a, vy: b + s_}),
        ('Rectas y = mx', r'y - ' + tex(b) + r' = m\,(x - ' + tex(a) + ')', {vx: a + s_, vy: b + m_ * s_}),
        ('Parábolas y = kx²', r'y - ' + tex(b) + r' = k\,(x - ' + tex(a) + ')^2', {vx: a + s_, vy: b + k_ * s_ ** 2}),
        ('Parábolas x = ky²', r'x - ' + tex(a) + r' = k\,(y - ' + tex(b) + ')^2', {vx: a + k_ * s_ ** 2, vy: b + s_}),
        ('Cúbicas y = kx³', r'y - ' + tex(b) + r' = k\,(x - ' + tex(a) + ')^3', {vx: a + s_, vy: b + k_ * s_ ** 3}),
    ]
    resultados = []
    no_existe = None
    for nombre, ec, sub in caminos:
        g = _con_tiempo(lambda: sp.simplify(f.subs(sub)), 5)
        if g is None:
            continue
        L1 = _limite(g, s_, 0, '+', seg=5)
        L2 = _limite(g, s_, 0, '-', seg=5)
        if L1 is None:
            continue
        L = L1
        if L2 is not None and _simp(L1 - L2) != 0 and not (_infinito(L1) and _infinito(L2)):
            r.paso(nombre, 'Los límites a izquierda y derecha sobre el camino difieren.',
                   ec + r':\ \lim_{s\to0^+} = ' + tex(L1) + r',\ \lim_{s\to0^-} = ' + tex(L2), 1)
            no_existe = 'Sobre el camino ' + nombre.lower() + ' los límites laterales difieren.'
            break
        param = [p for p in (m_, k_) if L.has(p)]
        r.paso(nombre, '', ec + r':\quad \lim = ' + tex(_plano(L)), 1)
        if param:
            p = param[0]
            v0 = _simp(L.subs(p, 0))
            v1 = _simp(L.subs(p, 1))
            if v0 == v1:
                v1 = _simp(L.subs(p, 2))
            no_existe = (f'El límite depende de {p}: por ejemplo con {p} = 0 da {v0} y con {p} = '
                         f'{1 if _simp(L.subs(p, 1)) != v0 else 2} da {v1}.')
            r.paso('El límite depende del camino', no_existe, tex(_plano(L)), 1)
            break
        resultados.append(L)
        if len(resultados) >= 2 and _distintos(resultados[-1], resultados[0]):
            no_existe = f'Por distintos caminos se obtienen {tex(resultados[0])} y {tex(resultados[-1])}.'
            break
    if no_existe:
        r.paso('Conclusión', 'Como el valor depende del camino, el límite doble no existe.', '', 0)
        r.valor = 'no existe'
        r.resultado_latex = r.entrada_latex + r'\ \text{no existe}'
        r.extra('Conclusión', '', 'El límite no existe: ' + no_existe)
        r.comparar = ('texto', 'no existe', {})
        _grafico_3d(r, f, vx, vy)
        return r
    if not resultados:
        raise ErrorEntrada('No pude calcular los límites por trayectorias.')
    L = resultados[0]
    # 2. coordenadas polares
    rr = sp.Symbol('r', positive=True)
    th = sp.Symbol('theta', real=True)
    E = _con_tiempo(lambda: sp.factor(sp.together(f.subs({vx: a + rr * sp.cos(th), vy: b + rr * sp.sin(th)}))), 6)
    if E is not None:
        E = _con_tiempo(lambda: sp.factor(sp.trigsimp(E)), 3) or E
    probado = False
    if E is not None:
        r.paso('Coordenadas polares', r'Se reemplaza \(x = a + r\cos\theta,\ y = b + r\sin\theta\) y se hace \(r \to 0^+\).',
               'f = ' + tex(E), 0)
        Lp = _limite(E, rr, 0, '+', seg=6)
        if Lp is not None and Lp.has(th):
            r.paso('El límite depende de θ', 'Distintas direcciones dan distintos valores: el límite no existe.',
                   r'\lim_{r\to0^+} = ' + tex(Lp), 1)
            r.valor = 'no existe'
            r.resultado_latex = r.entrada_latex + r'\ \text{no existe}'
            r.comparar = ('texto', 'no existe', {})
            _grafico_3d(r, f, vx, vy)
            return r
        if _finito(L):
            D = E if L == 0 else _con_tiempo(lambda: sp.factor(sp.together(E - L)), 6)
            if D is not None:
                if not D.has(th):
                    Ld = _limite(D, rr, 0, '+')
                    probado = Ld == 0
                    if probado:
                        r.paso('Acotación', r'La diferencia no depende de \(\theta\) y tiende a 0.',
                               '|f - ' + tex(L) + '| = ' + tex(sp.Abs(D)) + r'\to 0', 1)
                else:
                    Gt, Rr = D.as_independent(rr, as_Add=False)
                    if Rr.has(rr) and not Rr.has(th) and not Gt.has(rr) and _limite(Rr, rr, 0, '+') == 0:
                        denG = sp.denom(sp.together(Gt))
                        acotada = not denG.has(th)
                        if not acotada:
                            ceros = _con_tiempo(lambda: sp.solveset(denG, th, sp.Interval(0, 2 * PI)), 5)
                            acotada = ceros is sp.S.EmptySet
                        if acotada:
                            probado = True
                            r.paso('Acotación', r'La parte que depende de \(\theta\) está acotada (seno y coseno valen a lo sumo 1 '
                                   r'y el denominador no se anula) y el factor en \(r\) tiende a 0.',
                                   '|f - ' + tex(L) + r'| = |' + tex(Rr) + r'|\cdot\left|' + tex(Gt) + r'\right| \leq C\,|'
                                   + tex(Rr) + r'| \to 0', 1)
    if not probado and E is not None:
        # Direcciones críticas: donde se anula el denominador en polares (con r -> 0). Se prueban curvas tangentes a ellas.
        den_e = sp.denom(sp.together(E))
        den0 = _con_tiempo(lambda: sp.simplify(den_e.subs(rr, 0)), 4)
        ceros = _con_tiempo(lambda: sp.solveset(den0, th, sp.Interval.Ropen(0, 2 * PI)), 5) \
            if den0 is not None and den0.has(th) else None
        if isinstance(ceros, sp.FiniteSet):
            for t0 in list(ceros)[:2]:
                c0, s0 = sp.cos(t0), sp.sin(t0)
                sub = {vx: a + s_ * c0 - k_ * s_ ** 2 * s0, vy: b + s_ * s0 + k_ * s_ ** 2 * c0}
                g = _con_tiempo(lambda: sp.simplify(f.subs(sub)), 5)
                Lk = _limite(g, s_, 0, '+', seg=5) if g is not None else None
                if Lk is None:
                    continue
                distinto = Lk.has(k_) or (_finito(Lk) and _finito(L) and _simp(Lk - L) != 0) or (_infinito(Lk) != _infinito(L))
                r.paso('Camino crítico', r'En polares el denominador se anula en la dirección \(\theta = ' + tex(t0)
                       + r'\): se prueba una curva que llega al punto tangente a esa dirección.',
                       r'(x, y) = (a, b) + s\,(\cos\theta_0, \sin\theta_0) + k s^2(-\sin\theta_0, \cos\theta_0):\quad \lim = '
                       + tex(_plano(Lk)), 1)
                if distinto:
                    r.paso('Conclusión', 'Por ese camino el límite es distinto (o depende de k): el límite doble no existe.', '', 0)
                    r.valor = 'no existe'
                    r.resultado_latex = r.entrada_latex + r'\ \text{no existe}'
                    r.extra('Conclusión', '', 'El límite no existe: depende del camino de aproximación.')
                    r.comparar = ('texto', 'no existe', {})
                    _grafico_3d(r, f, vx, vy)
                    return r
    if not probado:
        r.avisos.append('Todas las trayectorias probadas dan el mismo valor, pero no se pudo demostrar la acotación en '
                        'polares: el resultado es probable, no está demostrado.')
    r.paso('Conclusión', 'El límite existe y vale ' + str(L) + '.' if probado else
           'Todas las trayectorias dan el mismo valor.', r.entrada_latex + ' = ' + tex(L), 0)
    r.valor = L
    r.resultado_latex = r.entrada_latex + ' = ' + tex(L)
    r.comparar = ('numero_o_expr', L, {})
    _grafico_3d(r, f, vx, vy)
    return r


def _grafico_3d(r, f, vx, vy):
    try:
        r.grafico = G.grafico_3d(_plano(f), (vx, vy), tex(f), rango=2.0)
    except Exception:  # noqa
        pass


# ----------------------------------------------------------------------------
# Integral de línea, potencial, flujo
# ----------------------------------------------------------------------------
def integral_linea(arg):
    uso = ('integral de linea F, r(t), a, b   (por ejemplo: integral de linea (-y, x), (cos(t), sen(t)), 0, 2pi); '
           'si F es escalar se calcula ∫ f ds')
    partes = dividir(arg)
    if len(partes) != 4:
        raise _uso(uso)
    F_txt, r_txt = partes[0], partes[1]
    rv = _leer_vector(r_txt, 'la curva r(t)')
    a, b = leer(partes[2]), leer(partes[3])
    v = _variable_param(*rv)
    dim = len(rv)
    coords = [x, y, z][:dim]
    vr = _real(v)
    rv_r = [c.subs(v, vr) for c in rv]
    dr = [_simp(sp.diff(c, vr)) for c in rv_r]
    sub = dict(zip(coords, rv_r))
    r = Resultado('Análisis II', 'Integral de línea')
    if _es_vector_texto(F_txt):
        F = _leer_vector(F_txt, 'el campo F')
        if len(F) != dim:
            raise ErrorEntrada(f'El campo y la curva deben tener la misma dimensión ({len(F)} y {dim}).')
        r.entrada_latex = r'\int_C \vec F\cdot d\vec r,\quad \vec F = ' + _vector_tex(F) + r',\ \vec r(' + tex(v) + ') = ' + _vector_tex(rv)
        Fr = [_simp(c.subs(sub)) for c in F]
        r.paso('Parametrizar el campo', r'Se reemplaza \(x, y\,(, z)\) por las componentes de \(\vec r(t)\).',
               r'\vec F(\vec r(' + tex(v) + ')) = ' + _vector_tex(Fr), 0)
        r.paso('Derivada de la curva', '', r"\vec r\,'(" + tex(v) + ') = ' + _vector_tex(dr), 0)
        integrando = _simp(sum(p * q for p, q in zip(Fr, dr)))
        r.paso('Producto escalar', r"\(\vec F(\vec r(t))\cdot\vec r\,'(t)\)", tex(integrando), 0)
        # ¿conservativo?
        if dim == 2:
            rot = _simp(sp.diff(F[1], x) - sp.diff(F[0], y))
            conservativo = rot == 0
        else:
            rot = [_simp(sp.diff(F[2], y) - sp.diff(F[1], z)), _simp(sp.diff(F[0], z) - sp.diff(F[2], x)),
                   _simp(sp.diff(F[1], x) - sp.diff(F[0], y))]
            conservativo = all(c == 0 for c in rot)
        if conservativo:
            r.extra('Campo conservativo', '', 'El rotor es nulo: la integral solo depende de los extremos (se puede usar la '
                    'función potencial: ∫ = φ(final) − φ(inicial)).')
        else:
            r.extra('Campo no conservativo', '', 'El rotor no es nulo: la integral depende del camino.')
    else:
        fe = leer(F_txt)
        r.entrada_latex = r'\int_C ' + tex(fe) + r'\,ds,\quad \vec r(' + tex(v) + ') = ' + _vector_tex(rv)
        fr_ = _simp(fe.subs(sub))
        rapidez = _raiz_simplificada(sum(c ** 2 for c in dr), vr)
        r.paso('Integral respecto de la longitud de arco', r"\(ds = \|\vec r\,'(t)\|\,dt\)",
               r"\|\vec r\,'(" + tex(v) + r')\| = ' + tex(rapidez), 0)
        integrando = _simp(fr_ * rapidez.subs(_plano(vr), vr))
        r.paso('Integrando', '', 'f(\\vec r(' + tex(v) + "))\\,\\|\\vec r\\,'\\| = " + tex(integrando), 0)
    val, exacto = _integrar_def(integrando, vr, a, b)
    val = _plano(val)
    r.paso('Integrar', '' if exacto else 'Se integra numéricamente.',
           r'\int_{' + tex(a) + '}^{' + tex(b) + '} ' + tex(integrando) + r'\,d' + tex(v) + ' = ' + _tex_num(val, 8), 0)
    r.valor = val
    r.valores['decimal'] = _decimal_txt(val)
    r.resultado_latex = r'\int_C = ' + _tex_num(val, 8)
    r.comparar = ('numero_o_expr', val, {})
    if dim == 2:
        try:
            r.grafico = G.grafico_parametrica(_plano(rv[0]), _plano(rv[1]), _plano(v), 'C', float(sp.N(a)), float(sp.N(b)))
        except Exception:  # noqa
            pass
    return r


def potencial(arg):
    uso = 'potencial F   (por ejemplo: potencial (2x*y, x^2)   o   potencial [y*z, x*z, x*y])'
    texto = arg.strip()
    if not _es_vector_texto(texto):
        partes = dividir(texto)
        if len(partes) in (2, 3):
            texto = '(' + texto + ')'
        else:
            raise _uso(uso)
    F = _leer_vector(texto, 'el campo F')
    if len(F) not in (2, 3):
        raise _uso(uso)
    vs = [x, y, z][:len(F)]
    r = Resultado('Análisis II', 'Función potencial')
    r.entrada_latex = r'\vec F = ' + _vector_tex(F)
    if len(F) == 2:
        P, Q = F
        Py, Qx = _simp(sp.diff(P, y)), _simp(sp.diff(Q, x))
        cons = _simp(Py - Qx) == 0
        r.paso('¿Es conservativo?', r'En el plano, \(\vec F = (P, Q)\) es conservativo (en un dominio simplemente conexo) '
               r'si \(\dfrac{\partial P}{\partial y} = \dfrac{\partial Q}{\partial x}\).',
               r'P_y = ' + tex(Py) + r',\quad Q_x = ' + tex(Qx) + (r'\ \Rightarrow\ \text{iguales}' if cons else r'\ \Rightarrow\ \text{distintas}'), 0)
    else:
        P, Q, R_ = F
        rot = [_simp(sp.diff(R_, y) - sp.diff(Q, z)), _simp(sp.diff(P, z) - sp.diff(R_, x)), _simp(sp.diff(Q, x) - sp.diff(P, y))]
        cons = all(c == 0 for c in rot)
        r.paso('¿Es conservativo?', r'\(\vec F\) es conservativo (en un dominio simplemente conexo) si su rotor es nulo.',
               r'\nabla\times\vec F = ' + _vector_tex(rot), 0)
    if not cons:
        r.paso('Conclusión', 'El campo no es conservativo: no existe función potencial.', '', 0)
        r.valor = 'no existe'
        r.resultado_latex = r'\text{El campo no es conservativo: no tiene función potencial}'
        r.comparar = ('texto', 'no existe', {})
        return r
    phi = sp.integrate(P, x)
    r.paso('Integrar P respecto de x', r'\(\varphi = \int P\,dx + g(y' + (', z' if len(F) == 3 else '') + r')\)',
           r'\varphi = ' + tex(phi) + ' + g(y' + (', z' if len(F) == 3 else '') + ')', 0)
    gp = _simp(Q - sp.diff(phi, y))
    r.paso('Derivar respecto de y e igualar a Q', '', r"\varphi_y = " + tex(_simp(sp.diff(phi, y))) + " + g_y = " + tex(Q)
           + r"\Rightarrow\ g_y = " + tex(gp), 0)
    g = sp.integrate(gp, y)
    phi = phi + g
    if len(F) == 3:
        hp = _simp(R_ - sp.diff(phi, z))
        r.paso('Derivar respecto de z e igualar a R', '', r'\varphi_z = ' + tex(_simp(sp.diff(phi, z))) + ' + h\'(z) = '
               + tex(R_) + r"\Rightarrow\ h'(z) = " + tex(hp), 0)
        phi = phi + sp.integrate(hp, z)
    phi = _simp(phi)
    r.paso('Función potencial', 'Se verifica que su gradiente es F.', r'\varphi = ' + tex(phi) + ' + C', 0)
    r.valor = phi
    r.resultado_latex = r'\varphi(' + ', '.join(map(str, vs)) + ') = ' + tex(phi) + ' + C'
    r.comparar = ('expr', phi, {})
    return r


def flujo(arg):
    uso = ('flujo F, (x(u,v), y(u,v), z(u,v)), u, a, b, v, c, d   o   flujo F, z=g(x,y), x, a, b, y, c, d   '
           '(por ejemplo: flujo (x, y, z), z=1-x-y, x, 0, 1, y, 0, 1-x)')
    partes = dividir(arg)
    if len(partes) != 8:
        raise _uso(uso)
    F = _leer_vector(partes[0], 'el campo F')
    if len(F) != 3:
        raise ErrorEntrada('El flujo se calcula para un campo de 3 componentes (P, Q, R).')
    # 'u' en la sintaxis general es el escalón: acá es un parámetro de la superficie
    par = {'u': sp.Symbol('u'), 'v': sp.Symbol('v')}
    u1, a1, b1 = leer(partes[2], par), leer(partes[3], par), leer(partes[4], par)
    u2, a2, b2 = leer(partes[5], par), leer(partes[6], par), leer(partes[7], par)
    if a1.has(u2) or b1.has(u2):  # los límites de la primera dependen de la segunda: se integra primero la primera
        pass
    elif a2.has(u1) or b2.has(u1):  # límites de la segunda dependen de la primera: la segunda va adentro
        u1, a1, b1, u2, a2, b2 = u2, a2, b2, u1, a1, b1
    r = Resultado('Análisis II', 'Flujo de un campo a través de una superficie')
    sup_txt = partes[1]
    if '=' in sup_txt:
        eq = leer_ecuacion(sup_txt)
        if not (isinstance(eq, sp.Equality) and eq.lhs == z):
            raise ErrorEntrada('Escribí la superficie como z = g(x, y).')
        gz = eq.rhs
        S = [x, y, gz]
        if {u1, u2} != {x, y}:
            raise ErrorEntrada('Para z = g(x, y) los parámetros son x e y.')
        r.paso('Parametrización', r'\(\vec r(x, y) = (x,\ y,\ g(x, y))\); normal hacia arriba: \(\vec N = (-g_x,\ -g_y,\ 1)\).',
               r'\vec r = ' + _vector_tex(S), 0)
    else:
        S = leer(sup_txt, par)
        S = list(S) if isinstance(S, (sp.Tuple, list, tuple)) else [S]
        if len(S) != 3:
            raise ErrorEntrada('La superficie debe tener 3 componentes.')
        r.paso('Parametrización', '', r'\vec r(' + tex(u1) + ',' + tex(u2) + ') = ' + _vector_tex(S), 0)
    # el orden del producto vectorial sigue el orden en que se escribieron los parámetros
    p1, p2 = (u1, u2) if partes[2].strip() == str(u1) else (u2, u1)
    if '=' in sup_txt:
        p1, p2 = x, y
    ru = [sp.diff(c, p1) for c in S]
    rv = [sp.diff(c, p2) for c in S]
    N = [_simp(ru[1] * rv[2] - ru[2] * rv[1]), _simp(ru[2] * rv[0] - ru[0] * rv[2]), _simp(ru[0] * rv[1] - ru[1] * rv[0])]
    r.paso('Vector normal', r'\(\vec N = \vec r_u \times \vec r_v\)', r'\vec N = ' + _vector_tex(N), 0)
    sub = {x: S[0], y: S[1], z: S[2]} if '=' not in sup_txt else {z: S[2]}
    FS = [_simp(c.subs(sub, simultaneous=True)) for c in F]
    integrando = _simp(sum(p * q for p, q in zip(FS, N)))
    r.entrada_latex = r'\iint_S \vec F\cdot d\vec S,\quad \vec F = ' + _vector_tex(F)
    r.paso('Integrando', r'\(\vec F(\vec r(u, v))\cdot\vec N\)', tex(integrando), 0)
    paso1 = _con_tiempo(lambda: sp.integrate(integrando, (u1, a1, b1)), 10)
    val = _con_tiempo(lambda: sp.integrate(paso1, (u2, a2, b2)), 10) if paso1 is not None else None
    if val is None or val.has(sp.Integral):
        raise ErrorEntrada('No pude calcular la integral doble del flujo en forma exacta.')
    val = _simp(val)
    r.paso('Integral doble', 'Se integra primero en ' + str(u1) + ' y luego en ' + str(u2) + '.',
           r'\int_{' + tex(a2) + '}^{' + tex(b2) + r'}\int_{' + tex(a1) + '}^{' + tex(b1) + '} ' + tex(integrando) + r'\,d'
           + tex(u1) + r'\,d' + tex(u2) + ' = ' + _tex_num(val, 8), 0)
    r.extra('Orientación', '', 'El signo corresponde a la normal r_u × r_v (para z = g(x, y), hacia arriba). '
            'Con la orientación opuesta el flujo cambia de signo.')
    r.valor = val
    r.valores['decimal'] = _decimal_txt(val)
    r.resultado_latex = r'\Phi = ' + _tex_num(val, 8)
    r.comparar = ('numero_o_expr', val, {})
    return r


# ----------------------------------------------------------------------------
# Curvatura y función vectorial
# ----------------------------------------------------------------------------
def _cruz(a_, b_):
    return [a_[1] * b_[2] - a_[2] * b_[1], a_[2] * b_[0] - a_[0] * b_[2], a_[0] * b_[1] - a_[1] * b_[0]]


def _leer_curva(arg, uso):
    partes = dividir(arg)
    if not partes:
        raise _uso(uso)
    if _es_vector_texto(partes[0]):
        rv = _leer_vector(partes[0], 'la función vectorial r(t)')
        t0 = leer(partes[1]) if len(partes) > 1 else None
        explicita = False
    elif len(partes) in (2, 3) and all(not p.strip().lstrip('-').replace('.', '').isdigit() for p in partes[:2]) \
            and len(partes) >= 2 and _variable_param(leer(partes[0]), leer(partes[1])).name == 't':
        rv = [leer(p) for p in partes[:2]] if len(partes) == 2 else [leer(p) for p in partes[:3]]
        t0 = None
        explicita = False
    else:
        f = leer(partes[0])
        vs = variables_de(f)
        v = vs[0] if vs else x
        rv = [v, f]
        t0 = leer(partes[1]) if len(partes) > 1 else None
        explicita = True
    if len(rv) not in (2, 3):
        raise _uso(uso)
    return rv, t0, explicita


def _datos_curva(rv):
    v = _variable_param(*rv)
    vr = _real(v)
    R = [c.subs(v, vr) for c in rv]
    if len(R) == 2:
        R = R + [sp.Integer(0)]
    d1 = [_simp(sp.diff(c, vr)) for c in R]
    d2 = [_simp(sp.diff(c, vr)) for c in d1]
    vel = _raiz_simplificada(sum(c ** 2 for c in d1), vr).subs(_plano(vr), vr)
    cr = [_simp(c) for c in _cruz(d1, d2)]
    ncr = _raiz_simplificada(sum(c ** 2 for c in cr), vr).subs(_plano(vr), vr)
    kappa = _simp(ncr / vel ** 3)
    return v, vr, R, d1, d2, vel, cr, ncr, kappa


def curvatura(arg):
    uso = 'curvatura r(t)[, t0]   (por ejemplo: curvatura (cos(t), sen(t), t))   o   curvatura f(x)[, x0]'
    rv, t0, explicita = _leer_curva(arg, uso)
    v, vr, R, d1, d2, vel, cr, ncr, kappa = _datos_curva(rv)
    dim = len(rv)
    r = Resultado('Análisis II', 'Curvatura')
    if explicita:
        f = rv[1]
        r.entrada_latex = r'\kappa\ \text{de}\ y = ' + tex(f)
        fp, fpp = _simp(sp.diff(f, v)), _simp(sp.diff(f, v, 2))
        r.paso('Fórmula para y = f(x)', '', r"\kappa = \frac{|f''|}{\left(1 + f'^2\right)^{3/2}}", 0)
        r.paso('Derivadas', '', "f' = " + tex(fp) + r",\quad f'' = " + tex(fpp), 0)
        kappa = _simp(sp.Abs(fpp) / (1 + fp ** 2) ** sp.Rational(3, 2))
        kappa_r = kappa
    else:
        r.entrada_latex = r'\kappa\ \text{de}\ \vec r(' + tex(v) + ') = ' + _vector_tex(rv)
        r.paso('Fórmula', '', r"\kappa = \frac{\|\vec r\,' \times \vec r\,''\|}{\|\vec r\,'\|^3}", 0)
        r.paso('Derivadas', '', r"\vec r\,' = " + _vector_tex(_plano(d1[:dim])) + r",\quad \vec r\,'' = " + _vector_tex(_plano(d2[:dim])), 0)
        r.paso('Producto vectorial y normas', '', r"\vec r\,'\times\vec r\,'' = " + _vector_tex(_plano(cr)) + r",\quad \|\vec r\,'\times\vec r\,''\| = "
               + tex(_plano(ncr)) + r",\quad \|\vec r\,'\| = " + tex(_plano(vel)), 0)
        kappa_r = kappa
        kappa = _plano(kappa)
    kappa = _plano(kappa)
    r.paso('Curvatura', '', r'\kappa(' + tex(v) + ') = ' + tex(kappa), 0)
    if t0 is not None:
        val = _simp(kappa.subs(v, t0))
        r.paso('Evaluar', '', r'\kappa(' + tex(t0) + ') = ' + _tex_num(val, 8), 0)
        if _finito(val) and val != 0:
            r.extra('Radio de curvatura', r'\rho = 1/\kappa = ' + _tex_num(_simp(1 / val), 6))
        r.valor = val
        r.valores['decimal'] = _decimal_txt(val)
        r.resultado_latex = r'\kappa(' + tex(t0) + ') = ' + _tex_num(val, 8)
        r.comparar = ('numero_o_expr', val, {})
    else:
        r.valor = kappa
        r.resultado_latex = r'\kappa(' + tex(v) + ') = ' + tex(kappa)
        r.comparar = ('expr', kappa, {})
        if kappa.is_number and kappa != 0:
            r.extra('Radio de curvatura', r'\rho = 1/\kappa = ' + tex(_simp(1 / kappa)))
    try:
        if explicita:
            r.grafico = G.grafico_2d([_plano(rv[1])], _plano(v))
        elif dim == 2:
            r.grafico = G.grafico_parametrica(_plano(rv[0]), _plano(rv[1]), _plano(v), 'curva')
    except Exception:  # noqa
        pass
    return r


def funcion_vectorial(arg):
    uso = 'funcion vectorial r(t)[, t0]   (por ejemplo: funcion vectorial (cos(t), sen(t), t))'
    partes = dividir(arg)
    if not partes or not _es_vector_texto(partes[0]):
        raise _uso(uso)
    rv = _leer_vector(partes[0], 'la función vectorial')
    t0 = leer(partes[1]) if len(partes) > 1 else None
    dim = len(rv)
    if dim not in (2, 3):
        raise _uso(uso)
    v, vr, R, d1, d2, vel, cr, ncr, kappa = _datos_curva(rv)
    r = Resultado('Análisis II', 'Función vectorial: triedro y curvatura')
    r.entrada_latex = r'\vec r(' + tex(v) + ') = ' + _vector_tex(rv)
    r.paso('Velocidad', r"\(\vec v = \vec r\,'(t)\)", r"\vec r\,'(" + tex(v) + ') = ' + _vector_tex(_plano(d1[:dim])), 0)
    r.paso('Aceleración', r"\(\vec a = \vec r\,''(t)\)", r"\vec r\,''(" + tex(v) + ') = ' + _vector_tex(_plano(d2[:dim])), 0)
    r.paso('Rapidez', '', r"\|\vec r\,'(" + tex(v) + r')\| = ' + tex(_plano(vel)), 0)
    T = [_simp(c / vel) for c in d1]
    r.paso('Vector tangente unitario', r"\(\vec T = \dfrac{\vec r\,'}{\|\vec r\,'\|}\)", r'\vec T = ' + _vector_tex(_plano(T[:dim])), 0)
    # N y B sin derivar T: B = (r' x r'')/|r' x r''|, N = B x T
    Nv = None
    if all(_simp(c) == 0 for c in cr):
        r.paso('Vector normal principal', r"\(\vec r\,'\times\vec r\,'' = \vec 0\): la curva es una recta y N no está definido.", '', 0)
    else:
        def calc_nb(sub):
            d1s = [c.subs(vr, sub) for c in d1] if sub is not None else d1
            crs = [c.subs(vr, sub) for c in cr] if sub is not None else cr
            ncs = sp.sqrt(sum(c ** 2 for c in crs)) if sub is not None else ncr
            vs_ = sp.sqrt(sum(c ** 2 for c in d1s)) if sub is not None else vel
            B_ = [_simp(c / ncs) for c in crs]
            N_ = [_simp(c / (ncs * vs_)) for c in _cruz(crs, d1s)]
            return N_, B_
        res_nb = _con_tiempo(lambda: calc_nb(None), 6)
        etiqueta = ''
        if res_nb is None and t0 is not None:
            res_nb = _con_tiempo(lambda: calc_nb(t0), 6)
            etiqueta = f' (en {tex(v)} = {tex(t0)})'
        if res_nb is not None:
            Nv, B = res_nb
            r.paso('Vector normal principal' + etiqueta,
                   r"\(\vec N = \dfrac{\vec T\,'}{\|\vec T\,'\|}\); se calcula como \(\vec N = \vec B\times\vec T\).",
                   r'\vec N = ' + _vector_tex(_plano(Nv[:dim])), 0)
            if dim == 3:
                r.paso('Vector binormal' + etiqueta, r"\(\vec B = \vec T \times \vec N = \dfrac{\vec r\,'\times\vec r\,''}{\|\vec r\,'\times\vec r\,''\|}\)",
                       r'\vec B = ' + _vector_tex(_plano(B)), 0)
                r.valores['binormal'] = _plano(B)
            if etiqueta:
                Nv = None
    r.paso('Curvatura', r"\(\kappa = \dfrac{\|\vec r\,'\times\vec r\,''\|}{\|\vec r\,'\|^3}\)", r'\kappa = ' + tex(_plano(kappa)), 0)
    if dim == 3:
        d3 = [_simp(sp.diff(c, vr)) for c in d2]
        den = _simp(sum(c ** 2 for c in cr))
        if den != 0:
            tau = _simp(sum(p * q for p, q in zip(cr, d3)) / den)
            r.paso('Torsión', r"\(\tau = \dfrac{(\vec r\,'\times\vec r\,'')\cdot\vec r\,'''}{\|\vec r\,'\times\vec r\,''\|^2}\)",
                   r'\tau = ' + tex(_plano(tau)), 0)
            r.valores['torsion'] = _plano(tau)
    T_out = _plano(T[:dim])
    r.valores['curvatura'] = _plano(kappa)
    r.valores['rapidez'] = _plano(vel)
    if Nv is not None:
        r.valores['normal'] = _plano(Nv[:dim])
    if t0 is not None:
        T0 = [_simp(c.subs(v, t0)) for c in T_out]
        k0 = _simp(_plano(kappa).subs(v, t0))
        r.paso(f'En {tex(v)} = {tex(t0)}', '', r'\vec T = ' + _vector_tex(T0) + r',\quad \kappa = ' + _tex_num(k0, 6), 0)
        r.valores['curvatura_t0'] = k0
        T_out = T0
    r.valor = T_out
    r.resultado_latex = r'\vec T = ' + _vector_tex(T_out) + r',\quad \kappa = ' + tex(_plano(kappa) if t0 is None else k0)
    r.comparar = ('vector', T_out, {})
    try:
        if dim == 2:
            r.grafico = G.grafico_parametrica(_plano(rv[0]), _plano(rv[1]), _plano(v), 'curva')
    except Exception:  # noqa
        pass
    return r


# ----------------------------------------------------------------------------
# Sistemas lineales de EDO
# ----------------------------------------------------------------------------
def _vector_entero(v):
    """Escala un autovector para que quede con números simples (primera componente no nula = 1, sin fracciones)."""
    v = sp.Matrix(v)
    piv = next((c for c in v if c != 0), 1)
    v = (v / piv).applyfunc(sp.nsimplify).applyfunc(sp.simplify)
    dens = [sp.fraction(sp.together(c))[1] for c in v if c.is_Rational]
    if dens and all(d.is_Integer for d in dens):
        m = sp.ilcm(*[int(d) for d in dens]) if len(dens) > 1 else int(dens[0])
        v = v * m
    return v


def sistema_edo(arg):
    uso = "sistema edo x'=..., y'=...[; x(0)=a; y(0)=b]   (por ejemplo: sistema edo x'=x+y, y'=4x+y)"
    texto = normalizar(arg).replace('’', "'")
    trozos = []
    for p in dividir(texto, ';'):
        trozos += dividir(p, ',')
    ecs, conds = [], []
    for p in trozos:
        p_ = p.strip()
        m = re.match(r"^([a-z]\d?)\s*'\s*(?:\(\s*t\s*\))?\s*=\s*(.+)$", p_) or re.match(r'^d([a-z]\d?)\s*/\s*dt\s*=\s*(.+)$', p_)
        if m:
            ecs.append((m.group(1), m.group(2)))
            continue
        m = re.match(r'^([a-z]\d?)\s*\(\s*([^)]+)\)\s*=\s*(.+)$', p_)
        if m:
            conds.append((m.group(1), m.group(2), m.group(3)))
            continue
        raise ErrorEntrada(f'No entendí «{p_}». ' + 'Uso: ' + uso)
    if len(ecs) < 2:
        raise _uso(uso)
    nombres = [nm for nm, _ in ecs]
    if len(set(nombres)) != len(nombres):
        raise ErrorEntrada('Hay dos ecuaciones para la misma variable.')
    vs = [sp.Symbol(nm) for nm in nombres]
    rhs = []
    for nm, txt in ecs:
        txt = re.sub(r'([a-z]\d?)\s*\(\s*t\s*\)', r'\1', txt)
        e = leer(txt, extra={nm_: sp.Symbol(nm_) for nm_ in nombres})
        extra = e.free_symbols - set(vs) - {t}
        if extra:
            raise ErrorEntrada('Aparecen variables que no son del sistema: ' + ', '.join(map(str, extra)))
        rhs.append(e)
    N = len(vs)
    A = sp.Matrix(N, N, lambda i, j: sp.diff(rhs[i], vs[j]))
    lineal = all(not c.has(*vs) for c in A)
    if not lineal:
        raise ErrorEntrada('El sistema no es lineal en ' + ', '.join(nombres) + '.')
    gvec = [_simp(rhs[i] - sum(A[i, j] * vs[j] for j in range(N))) for i in range(N)]
    r = Resultado('Ecuaciones diferenciales', f'Sistema lineal de {N} EDO')
    r.entrada_latex = r'\begin{cases}' + r'\\'.join(f"{nm}' = " + tex(e) for nm, e in zip(nombres, rhs)) + r'\end{cases}'
    Xs = sp.Matrix(vs)
    r.paso('Forma matricial', r"\(\vec X' = A\,\vec X\)" + (r' + \vec g(t)' if any(g != 0 for g in gvec) else ''),
           r"\vec X' = " + tex(A) + tex(Xs) + ((' + ' + tex(sp.Matrix(gvec))) if any(g != 0 for g in gvec) else ''), 0)
    t0 = None
    ics = {}
    for nm, tt, val in conds:
        if nm not in nombres:
            raise ErrorEntrada(f'La condición {nm}({tt}) no corresponde a una variable del sistema.')
        t0_ = leer(tt)
        if t0 is not None and t0_ != t0:
            raise ErrorEntrada('Todas las condiciones iniciales deben darse en el mismo instante.')
        t0 = t0_
        ics[sp.Symbol(nm)] = leer(val)
    C = sp.symbols(f'C1:{N + 1}')
    sol = None
    if A.has(t) or any(g != 0 for g in gvec):
        r.paso('Método', 'El sistema tiene coeficientes variables o término independiente: se resuelve con dsolve.', '', 0)
        fs = [sp.Function(nm)(t) for nm in nombres]
        subs_f = dict(zip(vs, fs))
        eqs = [sp.Eq(fs[i].diff(t), rhs[i].subs(subs_f)) for i in range(N)]
        ics_f = {sp.Function(str(k_))(t0): v_ for k_, v_ in ics.items()} if ics else None
        res = _con_tiempo(lambda: sp.dsolve(eqs, fs, ics=ics_f), 20)
        if not res:
            raise ErrorEntrada('No pude resolver el sistema en forma exacta.')
        sol = {sp.Symbol(nm): _simp(eq.rhs) for nm, eq in zip(nombres, res)}
    else:
        lam = sp.Symbol('lamda')
        pc = sp.factor((A - lam * sp.eye(N)).det())
        r.paso('Ecuación característica', r'\(\det(A - \lambda I) = 0\)', tex(pc) + ' = 0', 0)
        base = []
        try:
            eigs = sorted(A.eigenvects(), key=lambda e_: -float(sp.re(sp.N(e_[0]))))
        except Exception:  # noqa
            raise ErrorEntrada('No pude calcular los autovalores de la matriz.')
        for val, mult, vecs in eigs:
            if val.is_real is False or (val.is_real is None and sp.im(val) != 0):
                if sp.im(val) <= 0:
                    continue
                al, be = sp.re(val), sp.im(val)
                w = sp.Matrix(vecs[0])
                piv = next(c for c in w if c != 0)
                w = (w / piv).applyfunc(lambda q: sp.expand(sp.simplify(q)))
                ar = w.applyfunc(sp.re)
                ai = w.applyfunc(sp.im)
                r.paso(f'λ = {tex(val)} (complejo)', r'Con \(\lambda = \alpha \pm \beta i\) y \(\vec v = \vec a + i\vec b\): '
                       r'\(\vec X_1 = e^{\alpha t}(\vec a\cos\beta t - \vec b\sin\beta t)\), '
                       r'\(\vec X_2 = e^{\alpha t}(\vec a\sin\beta t + \vec b\cos\beta t)\).',
                       r'\vec v = ' + tex(w) + r',\ \alpha = ' + tex(al) + r',\ \beta = ' + tex(be), 1)
                base.append(sp.exp(al * t) * (ar * sp.cos(be * t) - ai * sp.sin(be * t)))
                base.append(sp.exp(al * t) * (ar * sp.sin(be * t) + ai * sp.cos(be * t)))
                continue
            vs_ = [_vector_entero(v_) for v_ in vecs]
            r.paso(f'λ = {tex(val)}' + (f' (multiplicidad {mult})' if mult > 1 else ''),
                   r'Autovector: se resuelve \((A - \lambda I)\,\vec v = \vec 0\).',
                   r',\ '.join(r'\vec v = ' + tex(v_) for v_ in vs_), 1)
            for v_ in vs_:
                base.append(v_ * sp.exp(val * t))
            if len(vs_) < mult:
                if mult - len(vs_) > 1:
                    raise ErrorEntrada('La matriz no es diagonalizable y la multiplicidad es mayor que 2: '
                                       'este caso no está soportado.')
                M = A - val * sp.eye(N)
                try:
                    wsol, params = M.gauss_jordan_solve(vs_[0])
                    wv = wsol.subs({p_: 0 for p_ in params})
                except Exception:  # noqa
                    raise ErrorEntrada('No pude hallar el vector generalizado.')
                r.paso('Autovalor defectivo', r'Falta un autovector: se busca \(\vec w\) con \((A - \lambda I)\,\vec w = \vec v\) '
                       r'y se usa \(\vec X_2 = e^{\lambda t}(t\,\vec v + \vec w)\).', r'\vec w = ' + tex(wv), 2)
                base.append((t * vs_[0] + wv) * sp.exp(val * t))
        if len(base) != N:
            raise ErrorEntrada('No pude armar una base de soluciones del sistema.')
        Xg = sp.zeros(N, 1)
        for c_, b_ in zip(C, base):
            Xg += c_ * b_
        Xg = Xg.applyfunc(lambda q: sp.collect(sp.expand(q), C))
        r.paso('Solución general', r'\(\vec X = \sum C_i\,\vec X_i\)',
               r'\\'.join(f'{nm}(t) = ' + tex(e) for nm, e in zip(nombres, Xg)), 0)
        sol = {sp.Symbol(nm): e for nm, e in zip(nombres, Xg)}
        if ics:
            if len(ics) != N:
                raise ErrorEntrada(f'Hacen falta {N} condiciones iniciales (una por variable).')
            ecs_c = [sp.Eq(sol[k_].subs(t, t0), v_) for k_, v_ in ics.items()]
            cs = sp.solve(ecs_c, C, dict=True)
            if not cs:
                raise ErrorEntrada('No se pudieron ajustar las constantes a las condiciones iniciales.')
            r.paso('Condiciones iniciales', '', r',\ '.join(tex(e.lhs) + ' = ' + tex(e.rhs) for e in ecs_c)
                   + r'\Rightarrow\ ' + r',\ '.join(f'{tex(k_)} = {tex(v_)}' for k_, v_ in cs[0].items()), 0)
            sol = {k_: sp.expand(e.subs(cs[0])) for k_, e in sol.items()}
    r.valor = sol
    r.resultado_latex = r'\begin{cases}' + r'\\'.join(f'{tex(k_)}(t) = ' + tex(e) for k_, e in sol.items()) + r'\end{cases}'
    r.comparar = ('dict', sol, {})
    if ics and all(not e.free_symbols - {t} for e in sol.values()):
        try:
            r.grafico = G.grafico_2d(list(sol.values()), t, etiquetas=[f'{k_}(t)' for k_ in sol], xmin=float(t0 or 0) - 0.5,
                                     xmax=float(t0 or 0) + 4)
        except Exception:  # noqa
            pass
    return r


# ----------------------------------------------------------------------------
# Trayectorias ortogonales
# ----------------------------------------------------------------------------
def trayectorias_ortogonales(arg):
    uso = 'trayectorias ortogonales familia   (por ejemplo: trayectorias ortogonales x^2+y^2=c   o   y=c*x^2)'
    texto = dividir(arg)[0] if dividir(arg) else ''
    if not texto:
        raise _uso(uso)
    eq = leer_ecuacion(texto)
    if not isinstance(eq, sp.Equality):
        raise ErrorEntrada('Escribí la familia como una ecuación con un parámetro, por ejemplo x^2+y^2=c.')
    F = eq.lhs - eq.rhs
    params = [s_ for s_ in F.free_symbols if s_ not in (x, y)]
    if len(params) != 1:
        raise ErrorEntrada('La familia debe tener exactamente un parámetro (c, k o a) además de x e y.')
    c = params[0]
    r = Resultado('Ecuaciones diferenciales', 'Trayectorias ortogonales')
    r.entrada_latex = tex(eq.lhs) + ' = ' + tex(eq.rhs)
    despeje = sp.solve(F, c)
    yp = sp.Symbol("y'")
    if len(despeje) >= 1:
        Gc = _simp(despeje[0])
        r.paso('Despejar el parámetro', '', tex(c) + ' = ' + tex(Gc), 0)
        pend = _simp(-sp.diff(Gc, x) / sp.diff(Gc, y))
        r.paso('Derivar (eliminando el parámetro)', r'Se deriva implícitamente \(G(x, y) = c\): \(G_x + G_y\,y\' = 0\).',
               r"y' = -\frac{G_x}{G_y} = " + tex(pend), 0)
    else:
        pend_c = _simp(-sp.diff(F, x) / sp.diff(F, y))
        cs = sp.solve(F, c)
        if not cs:
            raise ErrorEntrada('No pude eliminar el parámetro de la familia.')
        pend = _simp(pend_c.subs(c, cs[0]))
        r.paso('EDO de la familia', '', "y' = " + tex(pend), 0)
    orto = _simp(-1 / pend)
    r.paso('Condición de ortogonalidad', r'Las pendientes de curvas perpendiculares cumplen \(m_1\,m_2 = -1\): '
           r"se reemplaza \(y'\) por \(-1/y'\).", "y' = " + tex(orto), 0)
    Y = sp.Function('y')(x)
    ode = sp.Eq(Y.diff(x), orto.subs(y, Y))
    C1 = sp.Symbol('C1')
    Cc = sp.Symbol('C')
    implicita, explicitas = None, []
    # Variables separables (método de la cursada): y' = X(x) Y(y)  ->  ∫ dy / Y(y) = ∫ X(x) dx + C
    sep = _con_tiempo(lambda: sp.separatevars(orto, symbols=[x, y], dict=True), 4)
    if isinstance(sep, dict) and x in sep and y in sep:
        Xx, Yy = sep['coeff'] * sep[x], sep[y]
        Hy = _con_tiempo(lambda: sp.integrate(1 / Yy, y), 6)
        Kx = _con_tiempo(lambda: sp.integrate(Xx, x), 6)
        if Hy is not None and Kx is not None and not (Hy.has(sp.Integral) or Kx.has(sp.Integral)):
            r.paso('Separar variables', r'\(\dfrac{dy}{' + tex(Yy) + r'} = ' + tex(Xx) + r'\,dx\)',
                   r'\int \frac{dy}{' + tex(Yy) + r'} = \int ' + tex(Xx) + r'\,dx \Rightarrow ' + tex(Hy) + ' = ' + tex(Kx) + ' + C', 0)
            imp = sp.logcombine(sp.expand_log(Hy - Kx, force=True), force=True)
            if isinstance(imp, sp.log):
                imp = imp.args[0]  # ln(u) = C  ->  u = C
            imp = _simp(imp)
            cte, resto = sp.factor_terms(imp).as_coeff_Mul()
            if cte.is_Rational and cte != 0 and resto.has(x, y):
                imp = resto  # (x^2 + 2y^2)/2 = C  ->  x^2 + 2y^2 = C
            implicita = imp
            ex = _con_tiempo(lambda: sp.solve(sp.Eq(imp, C1), y), 5)
            if ex:
                explicitas = [_simp(e_) for e_ in ex]
    if not explicitas:
        sols = _con_tiempo(lambda: sp.dsolve(ode, Y, hint='separable'), 8) or _con_tiempo(lambda: sp.dsolve(ode, Y), 12)
        if sols is None:
            if implicita is None:
                raise ErrorEntrada('No pude resolver la EDO de las trayectorias ortogonales.')
        else:
            explicitas = [s_.rhs for s_ in (sols if isinstance(sols, list) else [sols])]
            r.paso('Resolver la EDO', 'Es una EDO de primer orden.', SALTO.join('y = ' + tex(e_) for e_ in explicitas), 0)
    if implicita is not None:
        r.paso('Familia ortogonal (forma implícita)', '', tex(implicita) + r' = C', 0)
        r.valores['implicita'] = implicita
    if explicitas and implicita is not None:
        r.paso('Forma explícita', 'Despejando y (la constante se renombra C₁).', SALTO.join('y = ' + tex(e_) for e_ in explicitas), 0)
    if explicitas:
        r.valor = explicitas if len(explicitas) > 1 else explicitas[0]
        r.resultado_latex = (SALTO.join('y = ' + tex(e_) for e_ in explicitas) if len(explicitas) <= 1 or implicita is None
                             else tex(implicita) + r' = C')
        r.comparar = ('edo', explicitas[0], {'eq': ode, 'ics': {}, 'v': x, 'orden': 1})
    else:
        r.valor = sp.Eq(implicita, Cc)
        r.resultado_latex = tex(implicita) + r' = C'
        r.comparar = None
    try:
        curvas = []
        for cv in (1, 4):
            curvas.append({'fn': G.js(F.subs(c, cv)), 'vars': ['x', 'y'], 'etiqueta': f'{c} = {cv}', 'color': G.COLORES[0]})
        if implicita is not None:
            for cv in (-1, 1, 2):
                curvas.append({'fn': G.js(implicita - cv), 'vars': ['x', 'y'], 'etiqueta': f'C = {cv}', 'color': G.COLORES[1]})
        curvas = [cv for cv in curvas if cv['fn']]
        if curvas:
            r.grafico = {'tipo': 'implicita', 'curvas': curvas, 'puntos': [],
                         'ventana': {'xmin': -4, 'xmax': 4, 'ymin': -4, 'ymax': 4}}
    except Exception:  # noqa
        pass
    return r


# ============================================================================
# Registro de comandos (se prueban ANTES que los base: prefijos específicos)
# ============================================================================
COMANDOS_EXTRA = [
    (('estudio completo', 'estudio de la funcion', 'estudio de funcion', 'estudio de funciones', 'analisis completo',
      'estudiar funcion', 'estudiar la funcion'), estudio_completo),
    (('asintotas', 'asintota'), asintotas),
    (('puntos de inflexion', 'punto de inflexion', 'inflexion', 'concavidad'), concavidad),
    (('dominio e imagen', 'dominio de', 'dominio'), dominio),
    (('recta normal', 'normal a la curva'), recta_normal),
    (('diferencial total',), diferencial_total),
    (('diferencial',), diferencial),
    (('aproximacion lineal', 'aproximar', 'aproximacion'), aproximar),
    (('linealizar', 'linealizacion'), linealizar),
    (('teorema del valor medio', 'valor medio', 'lagrange valor medio', 'tvm'), valor_medio),
    (('teorema de rolle', 'rolle'), rolle),
    (('funcion inversa', 'inversa de la funcion', 'inversa de f'), funcion_inversa),
    (('area polar', 'area en polares', 'area en coordenadas polares'), area_polar),
    (('longitud de arco', 'longitud de la curva', 'longitud de curva', 'longitud arco'), longitud_arco),
    (('volumen de revolucion', 'volumen revolucion', 'volumen de solido de revolucion', 'solido de revolucion'),
     volumen_revolucion),
    (('superficie de revolucion', 'area de superficie de revolucion', 'area de revolucion'), superficie_revolucion),
    (('derivada parametrica', 'derivar parametrica', 'derivada de parametrica', 'derivada de una parametrica'),
     derivada_parametrica),
    (('criterio serie', 'criterios serie', 'criterio de serie', 'criterio de la serie', 'convergencia de la serie',
      'convergencia serie', 'converge la serie', 'converge'), criterio_serie),
    (('radio de convergencia', 'intervalo de convergencia', 'serie de potencias'), radio_convergencia),
    (('derivada direccional', 'direccional'), derivada_direccional),
    (('limite doble', 'limite de dos variables', 'limite en dos variables', 'lim doble'), limite_doble),
    (('integral de linea', 'integral curvilinea', 'circulacion'), integral_linea),
    (('funcion potencial', 'potencial'), potencial),
    (('flujo',), flujo),
    (('curvatura',), curvatura),
    (('funcion vectorial', 'curva vectorial', 'triedro'), funcion_vectorial),
    (('sistema edo', 'sistema de edo', 'sistema de ecuaciones diferenciales', 'sistema diferencial'), sistema_edo),
    (('trayectorias ortogonales', 'trayectoria ortogonal', 'familia ortogonal'), trayectorias_ortogonales),
]
