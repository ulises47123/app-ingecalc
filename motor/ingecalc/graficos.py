"""Especificaciones de gráficos: la app los dibuja en un canvas evaluando las
funciones en JavaScript (así el zoom y el desplazamiento son fluidos)."""
import math

import sympy as sp
from sympy.printing.str import StrPrinter

from .nucleo import tex, x, y, t

COLORES = ['#2563eb', '#dc2626', '#059669', '#d97706', '#7c3aed', '#db2777', '#0891b2']

_NOMBRES_JS = {
    'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'cot': 'cot', 'sec': 'sec', 'csc': 'csc',
    'asin': 'asin', 'acos': 'acos', 'atan': 'atan', 'acot': 'acot', 'asec': 'asec', 'acsc': 'acsc',
    'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh', 'coth': 'coth', 'sech': 'sech', 'csch': 'csch',
    'asinh': 'asinh', 'acosh': 'acosh', 'atanh': 'atanh', 'exp': 'exp', 'log': 'ln',
    'Abs': 'abs', 'sign': 'sign', 'floor': 'floor', 'ceiling': 'ceil', 'Heaviside': 'H',
    'erf': 'erf', 'gamma': 'gamma', 'factorial': 'fact', 'Max': 'max', 'Min': 'min', 'atan2': 'atan2',
    're': 're_', 'im': 'im_', 'DiracDelta': 'zero', 'real_root': 'rroot',
}


class ImpresoraJS(StrPrinter):
    """Imprime una expresión SymPy como código JS que usa el objeto F de grafico.js."""

    def _print_Pow(self, e, rational=False):
        b, p = e.as_base_exp()
        if p == sp.Rational(1, 2):
            return f'sqrt({self._print(b)})'
        if p.is_Rational and p.q % 2 == 1 and p.q != 1:
            # raíz de índice impar: real también para base negativa
            return f'rpow({self._print(b)},{p.p},{p.q})'
        return f'pow({self._print(b)},{self._print(p)})'

    def _print_Rational(self, e):
        return f'({e.p}/{e.q})'

    def _print_Integer(self, e):
        return str(e.p)

    def _print_Float(self, e):
        return repr(float(e))

    def _print_Exp1(self, e):
        return 'E'

    def _print_Pi(self, e):
        return 'PI'

    def _print_ImaginaryUnit(self, e):
        raise ValueError('complejo')

    def _print_Infinity(self, e):
        return 'Infinity'

    def _print_NegativeInfinity(self, e):
        return '(-Infinity)'

    def _print_Function(self, e):
        nombre = type(e).__name__
        if nombre == 'log' and len(e.args) == 2:
            return f'(ln({self._print(e.args[0])})/ln({self._print(e.args[1])}))'
        if nombre not in _NOMBRES_JS:
            raise ValueError(f'función {nombre}')
        return f'{_NOMBRES_JS[nombre]}(' + ','.join(self._print(a) for a in e.args) + ')'

    _print_Abs = _print_Function
    _print_Heaviside = _print_Function

    def _print_Piecewise(self, e):
        partes = 'NaN'
        for expr, cond in reversed(e.args):
            partes = f'(({self._print_cond(cond)})?({self._print(expr)}):({partes}))'
        return partes

    def _print_cond(self, c):
        if c is sp.true or c == True:  # noqa
            return 'true'
        if isinstance(c, sp.And):
            return '&&'.join(f'({self._print_cond(a)})' for a in c.args)
        if isinstance(c, sp.Or):
            return '||'.join(f'({self._print_cond(a)})' for a in c.args)
        ops = {sp.StrictLessThan: '<', sp.LessThan: '<=', sp.StrictGreaterThan: '>',
               sp.GreaterThan: '>=', sp.Equality: '==', sp.Unequality: '!='}
        for cls, op in ops.items():
            if isinstance(c, cls):
                return f'{self._print(c.lhs)}{op}{self._print(c.rhs)}'
        raise ValueError('condición')

    def _print_Symbol(self, e):
        return str(e)


def js(expr):
    """Código JS de la expresión, o None si no se puede graficar en vivo."""
    try:
        expr = sp.sympify(expr)
        if expr.has(sp.I):
            return None
        return ImpresoraJS().doprint(expr)
    except Exception:  # noqa
        return None


def _f_num(expr, v):
    try:
        return sp.lambdify(v, expr, modules=['math', {'Heaviside': lambda a, h=0.5: 1.0 if a > 0 else (0.0 if a < 0 else h)}])
    except Exception:  # noqa
        return None


def _muestras(expr, v, a, b, n_=240):
    f = _f_num(expr, v)
    ys = []
    if f is None:
        return ys
    for i in range(n_ + 1):
        xv = a + (b - a) * i / n_
        try:
            yv = f(xv)
            if isinstance(yv, complex):
                if abs(yv.imag) > 1e-9:
                    continue
                yv = yv.real
            yv = float(yv)
            if math.isfinite(yv):
                ys.append(yv)
        except Exception:  # noqa
            continue
    return ys


def ventana(exprs, v=x, puntos=(), xmin=None, xmax=None):
    """Elige una ventana que muestre lo interesante (puntos marcados y forma de la curva)."""
    xs = [p['x'] for p in puntos if p.get('x') is not None and math.isfinite(p['x'])]
    if xmin is None or xmax is None:
        if xs:
            a, b = min(xs), max(xs)
            ancho = max(b - a, 2.0)
            xmin, xmax = a - 0.6 * ancho, b + 0.6 * ancho
            if xmin > -1 and xmax > 0:
                xmin = min(xmin, -1.0)
        else:
            xmin, xmax = -10.0, 10.0
            # funciones con periodo/escala chica: si todas son trigonométricas usar [-2pi, 2pi]
            if all(e.has(sp.sin, sp.cos, sp.tan) for e in exprs if e is not None) and exprs:
                xmin, xmax = -2 * math.pi, 2 * math.pi
    ys = [p['y'] for p in puntos if p.get('y') is not None and math.isfinite(p['y'])]
    for e in exprs:
        if e is None:
            continue
        m = _muestras(e, v, xmin, xmax)
        if m:
            m.sort()
            # descartar extremos (asíntotas) usando percentiles
            lo = m[int(len(m) * 0.05)]
            hi = m[int(len(m) * 0.95) - 1] if len(m) > 2 else m[-1]
            ys += [lo, hi]
    if not ys:
        ymin, ymax = -10.0, 10.0
    else:
        ymin, ymax = min(ys), max(ys)
        if ymax - ymin < 1e-9:
            ymin, ymax = ymin - 1, ymax + 1
        alto = ymax - ymin
        ymin, ymax = ymin - 0.15 * alto, ymax + 0.15 * alto
        ymin = min(ymin, -0.5) if ymin > -0.5 and ymax > 0 else ymin
    return {'xmin': float(xmin), 'xmax': float(xmax), 'ymin': float(ymin), 'ymax': float(ymax)}


def punto(px, py, etiqueta='', color='#111827'):
    try:
        fx, fy = float(sp.N(px)), float(sp.N(py))
    except Exception:  # noqa
        return None
    if not (math.isfinite(fx) and math.isfinite(fy)):
        return None
    return {'x': fx, 'y': fy, 'etiqueta': etiqueta, 'color': color}


def curva(expr, v=x, etiqueta=None, color=None, estilo='solido'):
    expr = sp.sympify(expr)
    c = {'fn': js(expr), 'var': str(v), 'etiqueta': etiqueta if etiqueta is not None else tex(expr),
         'color': color or COLORES[0], 'estilo': estilo}
    if c['fn'] is None:
        # sin JS: mandar muestras
        c['muestras'] = None
        c['expr_py'] = str(expr)
    return c


def grafico_2d(exprs, v=x, puntos=(), sombra=None, etiquetas=None, xmin=None, xmax=None, titulo=''):
    exprs = [sp.sympify(e) for e in exprs]
    curvas = []
    for i, e in enumerate(exprs):
        et = etiquetas[i] if etiquetas and i < len(etiquetas) else None
        curvas.append(curva(e, v, et, COLORES[i % len(COLORES)]))
    puntos = [p for p in puntos if p]
    vent = ventana(exprs, v, puntos, xmin, xmax)
    g = {'tipo': '2d', 'curvas': curvas, 'puntos': puntos, 'ventana': vent, 'titulo': titulo}
    for c, e in zip(curvas, exprs):
        if c['fn'] is None:
            c['muestras'] = _muestras_xy(e, v, vent['xmin'], vent['xmax'])
    if sombra:
        g['sombra'] = sombra
    return g


def _muestras_xy(expr, v, a, b, n_=400):
    f = _f_num(expr, v)
    pts = []
    if f is None:
        return pts
    for i in range(n_ + 1):
        xv = a + (b - a) * i / n_
        try:
            yv = complex(f(xv))
            if abs(yv.imag) < 1e-9 and math.isfinite(yv.real):
                pts.append([xv, yv.real])
            else:
                pts.append(None)
        except Exception:  # noqa
            pts.append(None)
    return pts


def sombra_entre(f_sup, f_inf, a, b, v=x):
    return {'arriba': js(f_sup), 'abajo': js(f_inf), 'var': str(v), 'desde': float(sp.N(a)), 'hasta': float(sp.N(b))}


def grafico_implicita(F, vars_=(x, y), etiqueta=''):
    """F(x, y) = 0"""
    return {'tipo': 'implicita', 'curvas': [{'fn': js(F), 'vars': [str(v) for v in vars_],
                                             'etiqueta': etiqueta, 'color': COLORES[0]}],
            'ventana': {'xmin': -10, 'xmax': 10, 'ymin': -10, 'ymax': 10}, 'puntos': []}


def grafico_polar(r, v=t, etiqueta='', desde=0, hasta=2 * math.pi):
    return {'tipo': 'polar', 'curvas': [{'fn': js(r), 'var': str(v), 'etiqueta': etiqueta, 'color': COLORES[0],
                                         'desde': desde, 'hasta': hasta}], 'puntos': []}


def grafico_parametrica(fx, fy, v=t, etiqueta='', desde=None, hasta=None):
    if desde is None:
        periodicas = all(e.has(sp.sin, sp.cos) for e in (fx, fy))
        desde, hasta = (0, 2 * math.pi) if periodicas else (-5, 5)
    return {'tipo': 'parametrica', 'curvas': [{'fx': js(fx), 'fy': js(fy), 'var': str(v), 'etiqueta': etiqueta,
                                               'color': COLORES[0], 'desde': float(desde), 'hasta': float(hasta)}],
            'puntos': []}


def grafico_3d(f, vars_=(x, y), etiqueta='', rango=3.0):
    return {'tipo': '3d', 'superficie': {'fn': js(f), 'vars': [str(v) for v in vars_], 'etiqueta': etiqueta,
                                         'rango': rango}}


def grafico_datos(pts, ajuste=None, etiqueta=''):
    puntos = [punto(a, b) for a, b in pts]
    g = {'tipo': 'datos', 'puntos': [p for p in puntos if p], 'curvas': []}
    if ajuste is not None:
        g['curvas'].append(curva(ajuste, x, etiqueta or tex(ajuste), COLORES[1]))
    xs = [p['x'] for p in g['puntos']]
    ys = [p['y'] for p in g['puntos']]
    if xs:
        ax, ay = (max(xs) - min(xs)) or 1, (max(ys) - min(ys)) or 1
        g['ventana'] = {'xmin': min(xs) - 0.2 * ax, 'xmax': max(xs) + 0.2 * ax,
                        'ymin': min(ys) - 0.2 * ay, 'ymax': max(ys) + 0.2 * ay}
    return g


def grafico_vectores(vectores):
    """Vectores en el plano desde el origen."""
    pts = []
    vs = []
    for i, (vec, et) in enumerate(vectores):
        a, b = float(vec[0]), float(vec[1])
        vs.append({'x': a, 'y': b, 'etiqueta': et, 'color': COLORES[i % len(COLORES)]})
        pts.append({'x': a, 'y': b})
    m = max([abs(p['x']) for p in pts] + [abs(p['y']) for p in pts] + [1]) * 1.3
    return {'tipo': '2d', 'curvas': [], 'puntos': [], 'vectores': vs,
            'ventana': {'xmin': -m, 'xmax': m, 'ymin': -m, 'ymax': m}}
