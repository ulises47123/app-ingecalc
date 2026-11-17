"""Núcleo de IngeCalc: lectura de la entrada (sintaxis amigable en español),
conversión a SymPy, LaTeX y estructura de la respuesta."""
import re
import unicodedata

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
    convert_xor, split_symbols_custom, implicit_multiplication, implicit_application,
    function_exponentiation, rationalize,
)

x, y, z, t, s, n, k = sp.symbols('x y z t s n k')
SIMBOLOS_FIJOS = {'x': x, 'y': y, 'z': z, 't': t, 's': s, 'n': n, 'k': k}


class ErrorEntrada(Exception):
    """Error de lectura o de uso, con un mensaje claro para el alumno."""


# ----------------------------------------------------------------------------
# Funciones de la sintaxis amigable
# ----------------------------------------------------------------------------
def _log10(a, b=10):
    return sp.log(a, b)


def _raiz(a, n_=2):
    # índice impar: raíz real (root(-8, 3) = -2), como en la cursada
    if n_ == 2:
        return sp.sqrt(a)
    if getattr(sp.sympify(n_), 'is_odd', False):
        return sp.real_root(a, n_)
    return sp.root(a, n_)


def _cbrt(a):
    # raíz cúbica real (la que se usa en la cursada, ej. cbrt(-8) = -2)
    return sp.real_root(a, 3)


FUNCIONES = {
    'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'cot': sp.cot, 'sec': sp.sec, 'csc': sp.csc,
    'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan, 'acot': sp.acot, 'asec': sp.asec, 'acsc': sp.acsc,
    'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh, 'coth': sp.coth, 'sech': sp.sech, 'csch': sp.csch,
    'asinh': sp.asinh, 'acosh': sp.acosh, 'atanh': sp.atanh,
    'exp': sp.exp, 'ln': sp.log, 'logn_': sp.log, 'log10_': _log10, 'sqrt': sp.sqrt, 'root': _raiz,
    'cbrt': _cbrt, 'abs': sp.Abs, 'sign': sp.sign, 'sgn': sp.sign,
    'floor': sp.floor, 'ceiling': sp.ceiling, 'factorial': sp.factorial, 'binomial': sp.binomial,
    'gamma': sp.gamma, 'erf': sp.erf, 'heaviside': sp.Heaviside, 'u': sp.Heaviside,
    'delta': sp.DiracDelta, 're': sp.re, 'im': sp.im, 'conjugate': sp.conjugate, 'conj': sp.conjugate,
    'arg': sp.arg, 'max': sp.Max, 'min': sp.Min, 'atan2': sp.atan2,
    'Rational': sp.Rational, 'Interval': sp.Interval, 'Union': sp.Union, 'FiniteSet': sp.FiniteSet,
    'Integer': sp.Integer, 'Float': sp.Float, 'Symbol': sp.Symbol, 'Matrix': sp.Matrix,
}
CONSTANTES = {'e': sp.E, 'pi': sp.pi, 'i': sp.I, 'oo': sp.oo, 'E': sp.E, 'I': sp.I}

# Reemplazos de palabras en español (orden importa)
_PALABRAS_BASE = [
    ('arcsen', 'asin'), ('arcsin', 'asin'), ('arccos', 'acos'), ('arctg', 'atan'),
    ('arctan', 'atan'), ('arcctg', 'acot'), ('arccot', 'acot'), ('senh', 'sinh'), ('tgh', 'tanh'),
    ('ctgh', 'coth'), ('sen', 'sin'), ('tg', 'tan'), ('cotg', 'cot'), ('ctg', 'cot'),
    ('cosec', 'csc'), ('raizcubica', 'cbrt'), ('raiz', 'sqrt'), ('infinito', 'oo'), ('infty', 'oo'),
    ('inf', 'oo'), ('escalon', 'heaviside'), ('log', 'log10_'), ('ln', 'logn_'), ('lambda', 'lamda'),
]
# Una palabra cuenta si no tiene letras pegadas (sí puede tener un número delante: 3sen(x))
_PALABRAS = [(r'(?<![A-Za-z_])' + a + r'(?![A-Za-z_])', b) for a, b in _PALABRAS_BASE]

_PROTEGIDOS = set(FUNCIONES) | set(CONSTANTES) | {
    'alpha', 'beta', 'gamma', 'theta', 'lambda', 'lamda', 'mu', 'sigma', 'omega', 'phi', 'rho', 'tau',
    'epsilon', 'delta', 'psi', 'xi', 'eta', 'zeta', 'nu', 'kappa', 'chi', 'C1', 'C2', 'C3', 'C4',
}


def sin_tildes(texto):
    out = []
    for ch in unicodedata.normalize('NFD', texto):
        if unicodedata.category(ch) == 'Mn':
            continue
        out.append(ch)
    return unicodedata.normalize('NFC', ''.join(out))


def _cerrar_parentesis(s_, i):
    """Índice del paréntesis que cierra al que abre en s_[i]."""
    pares = {'(': ')', '[': ']', '{': '}'}
    abre = s_[i]
    cierra = pares[abre]
    nivel = 0
    for j in range(i, len(s_)):
        if s_[j] == abre:
            nivel += 1
        elif s_[j] == cierra:
            nivel -= 1
            if nivel == 0:
                return j
    raise ErrorEntrada('Falta cerrar un paréntesis.')


def _funcion_potencia(s_):
    """sin^2(x) -> (sin(x))^2 ; sin^2 x -> (sin(x))^2"""
    patron = re.compile(r'\b([a-z]+)\^(\(?-?\d+\)?)\s*')
    i = 0
    while True:
        m = patron.search(s_, i)
        if not m:
            return s_
        nombre = m.group(1)
        if nombre not in FUNCIONES:
            i = m.end()
            continue
        resto = s_[m.end():]
        if resto.startswith('('):
            fin = _cerrar_parentesis(s_, m.end())
            arg = s_[m.end() + 1:fin]
            nuevo = f'({nombre}({arg}))^{m.group(2)}'
            s_ = s_[:m.start()] + nuevo + s_[fin + 1:]
        else:
            m2 = re.match(r'[a-z0-9.]+', resto)
            if not m2:
                i = m.end()
                continue
            nuevo = f'({nombre}({m2.group(0)}))^{m.group(2)}'
            s_ = s_[:m.start()] + nuevo + s_[m.end() + m2.end():]
        i = m.start() + 1


def _barras_abs(s_):
    # |x| -> abs(x)  (sin anidar)
    previo = None
    while previo != s_:
        previo = s_
        s_ = re.sub(r'\|([^|]+)\|', r'(abs(\1))', s_)
    return s_


def normalizar(texto):
    """Normaliza símbolos y palabras de la sintaxis amigable (no interpreta comandos)."""
    s_ = texto.strip()
    reempl = {
        '²': '^2', '³': '^3', '¹': '^1', '⁴': '^4', '·': '*', '×': '*', '÷': '/', '−': '-',
        '–': '-', 'π': 'pi', '∞': 'oo', '→': '->', '≤': '<=', '≥': '>=', '≠': '!=',
        '∠': '@', '∫': ' integrar ', 'θ': 'theta', 'λ': 'lambda', 'α': 'alpha', 'β': 'beta',
        'ω': 'omega', 'μ': 'mu', 'σ': 'sigma', 'φ': 'phi', 'Δ': 'delta', '’': "'", '´': "'",
        '"': "''", ' ': ' ', ';;': ';',
    }
    for a, b in reempl.items():
        s_ = s_.replace(a, b)
    s_ = re.sub(r'√\s*\(', 'sqrt(', s_)
    s_ = re.sub(r'√\s*([A-Za-z0-9.]+)', r'sqrt(\1)', s_)
    s_ = re.sub(r'∛\s*\(', 'cbrt(', s_)
    s_ = re.sub(r'∛\s*([A-Za-z0-9.]+)', r'cbrt(\1)', s_)
    return s_


_MULTI_ARG = {'root', 'log', 'atan2', 'binomial', 'max', 'min', 'Interval', 'Matrix', 'mcd', 'mcm', 'gcd', 'lcm'}


def coma_decimal(texto):
    """0,726 -> 0.726: coma entre dígitos y sin espacios.
    Sigue siendo separador dentro de [..], de coordenadas (0,1) y de funciones de varios argumentos root(8,3);
    es decimal en paréntesis de agrupación 2*(0,5+1) y en funciones de un argumento cbrt(0,08)."""
    out, pila = [], []
    for i, ch in enumerate(texto):
        if ch in '([{':
            if ch != '(':
                pila.append('sep')
            else:
                m = re.search(r'([A-Za-z_][A-Za-z_0-9]*)$', texto[:i])
                previo = texto[:i].rstrip()
                if m:
                    pila.append('sep' if m.group(1) in _MULTI_ARG else 'dec')
                elif previo and (previo[-1] in '+-*/^:·×÷√∛' or previo[-1].isdigit() or previo[-1] == ')'):
                    pila.append('dec')
                else:
                    pila.append('sep')
        elif ch in ')]}':
            if pila:
                pila.pop()
        decimal_ok = (not pila) or pila[-1] == 'dec'
        if (ch == ',' and decimal_ok and 0 < i < len(texto) - 1 and texto[i - 1].isdigit()
                and texto[i + 1].isdigit()):
            out.append('.')
        else:
            out.append(ch)
    return ''.join(out)


def _notacion_escolar(s_):
    """Notación de la escuela/ingreso: 1 1/4 (mixto), 20 % de 800, 144 : 12, 'mas', 'por'..."""
    # número mixto: 1 1/4 -> (1+1/4)   (solo un número suelto, no después de otra cosa)
    s_ = re.sub(r'(?<![\w.)\]])(\d+)\s+(\d+)\s*/\s*(\d+)(?![\w.(])', r'(\1+\2/\3)', s_)
    s_ = s_.replace('%', '*(1/100)')
    s_ = s_.replace(':', '/')
    reglas = [(r'dividido\s+(?:en|por)', '/'), (r'al\s+cuadrado', '^2'), (r'al\s+cubo', '^3'), (r'mas', '+'),
              (r'menos', '-'), (r'por', '*'), (r'sobre', '/'), (r'del?', '*'), (r'veces', '*')]
    for pal, rep in reglas:
        s_ = re.sub(r'(?<![A-Za-z_])' + pal + r'(?![A-Za-z_])', f' {rep} ', s_, flags=re.I)
    return s_


def preparar_expresion(s_):
    """Texto de una expresión -> texto que entiende parse_expr."""
    s_ = normalizar(s_)
    s_ = sin_tildes(s_)
    s_ = _notacion_escolar(s_)
    # Constantes C1.. se mantienen en mayúscula; el resto en minúscula
    s_ = re.sub(r'(?<![A-Za-z_])([A-Za-z_][A-Za-z_0-9]*)',
                lambda m: m.group(1) if re.fullmatch(r'C\d+', m.group(1))
                or m.group(1) in ('Interval', 'Union', 'FiniteSet', 'Rational', 'Matrix') else m.group(1).lower(), s_)
    for patron, rep in _PALABRAS:
        s_ = re.sub(patron, rep, s_)
    # e^x con exponente "2x" -> el usuario quiere e^(2x)
    s_ = re.sub(r'(?<![A-Za-z_])e\s*\^\s*(-?\d+(?:\.\d+)?[a-z][a-z0-9]*)', r'e^(\1)', s_)
    # 2e^x, 3e -> 2*e^x (evita que se lea como notación científica)
    s_ = re.sub(r'(\d)e(?![0-9+\-])', r'\1*e', s_)
    # sen x cos x -> sin(x) cos(x)  (argumento simple sin paréntesis)
    nombres = '|'.join(sorted((f for f in FUNCIONES if f.replace('_', '').isalnum()), key=len, reverse=True))
    s_ = re.sub(r'(?<![A-Za-z_])(' + nombres + r')\s+(-?\d*\.?\d*[a-z]?)(?![A-Za-z_0-9(^*])',
                lambda m: f'{m.group(1)}({m.group(2)})' if m.group(2) else m.group(0), s_)
    s_ = _funcion_potencia(s_)
    s_ = _barras_abs(s_)
    # números complejos estilo ingeniería 3+4j
    s_ = re.sub(r'(\d)j(?![A-Za-z_0-9])', r'\1*i', s_)
    s_ = s_.replace('^', '**')
    return s_


def _predicado_division(nombre):
    """Decide si un símbolo de varias letras se parte en producto (xy -> x*y)."""
    if nombre in _PROTEGIDOS or len(nombre) > 4 or not nombre.isalpha():
        return False
    if '_' in nombre:
        return False
    return True


_TRANSF = standard_transformations + (
    split_symbols_custom(_predicado_division), implicit_multiplication,
    implicit_application, function_exponentiation, convert_xor, rationalize)


_LETRAS_PRODUCTO = set('xyztabcrsuvwknmpqhe')


def _controlar_palabras(s_, loc, original):
    """Evita que una palabra desconocida se lea en silencio como producto de letras ('de' -> d*e)."""
    for w in re.findall(r'(?<![A-Za-z_0-9.])[A-Za-z_][A-Za-z_0-9]*', s_):
        if w in loc or len(w) == 1 or any(ch.isdigit() for ch in w) or '_' in w:
            continue
        if w in _PROTEGIDOS or w in ('Interval', 'Union', 'FiniteSet', 'Rational', 'Matrix', 'Symbol', 'Integer', 'Float'):
            continue
        if len(w) <= 4 and set(w) <= _LETRAS_PRODUCTO:
            continue  # producto implícito: xy, 2xz, ab
        raise ErrorEntrada(f'No reconozco «{w}» en «{str(original).strip()}». Si es un producto de variables, '
                           'separalas con * (x*y); si es una función, revisá el nombre (sen, cos, ln, sqrt...).')


def leer(texto, extra=None, real=True):
    """Convierte texto amigable a una expresión SymPy."""
    if texto is None or not str(texto).strip():
        raise ErrorEntrada('Falta la expresión.')
    s_ = preparar_expresion(str(texto))
    loc = dict(FUNCIONES)
    loc.update(CONSTANTES)
    loc.update(SIMBOLOS_FIJOS)
    for nombre in ('C1', 'C2', 'C3', 'C4'):
        loc[nombre] = sp.Symbol(nombre)
    for nombre in ('theta', 'alpha', 'beta', 'lamda', 'omega', 'mu', 'sigma', 'phi', 'tau', 'rho'):
        loc[nombre] = sp.Symbol(nombre)
    if extra:
        loc.update(extra)
    _controlar_palabras(s_, loc, texto)
    try:
        expr = parse_expr(s_, local_dict=loc, transformations=_TRANSF, evaluate=True)
    except ErrorEntrada:
        raise
    except Exception as exc:  # noqa
        raise ErrorEntrada(f'No pude entender «{texto.strip()}». Revisá paréntesis y operadores.') from exc
    if isinstance(expr, tuple):
        expr = sp.Tuple(*expr)
    return expr


def leer_sin_evaluar(texto, extra=None):
    """Como leer() pero SIN simplificar: para mostrar el ejercicio tal como se escribió (8/8·x·5, no 5x)."""
    s_ = preparar_expresion(str(texto))
    loc = dict(FUNCIONES)
    loc.update(CONSTANTES)
    loc.update(SIMBOLOS_FIJOS)
    for nombre in ('C1', 'C2', 'C3', 'C4', 'theta', 'alpha', 'beta', 'lamda', 'omega', 'mu', 'sigma', 'phi', 'tau', 'rho'):
        loc[nombre] = sp.Symbol(nombre)
    loc['cbrt'] = lambda a, **kw: sp.Pow(a, sp.Rational(1, 3), evaluate=False)
    loc['root'] = lambda a, n_=2, **kw: sp.Pow(a, sp.Rational(1, int(n_)), evaluate=False)
    loc['log10_'] = lambda a, b=10, **kw: sp.log(a, b, evaluate=False)
    if extra:
        loc.update(extra)
    try:
        with sp.evaluate(False):
            # sin 'rationalize': los decimales se muestran como decimales
            e = parse_expr(s_, local_dict=loc, transformations=_TRANSF[:-1], evaluate=False)
        if isinstance(e, tuple):
            e = sp.Tuple(*e)
        return e
    except Exception:  # noqa
        return leer(texto, extra)


def leer_ecuacion(texto, extra=None):
    """'a = b' -> Eq ; 'a < b' -> relacional ; 'a' -> a (igualado a 0)."""
    s_ = normalizar(texto)
    ops = re.findall(r'<=|>=|<|>', s_)
    if len(ops) == 2:  # a < x <= b
        partes = re.split(r'<=|>=|<|>', s_)
        a, m_, b = (leer(p, extra) for p in partes)
        rel = {'<': sp.Lt, '<=': sp.Le, '>': sp.Gt, '>=': sp.Ge}
        return sp.And(rel[ops[0]](a, m_), rel[ops[1]](m_, b))
    for op in ('<=', '>=', '!=', '==', '<', '>', '='):
        if op in s_ and not (op == '=' and ('<=' in s_ or '>=' in s_ or '!=' in s_ or '==' in s_)):
            partes = s_.split(op)
            if len(partes) == 2:
                a, b = leer(partes[0], extra), leer(partes[1], extra)
                return {'<=': sp.Le, '>=': sp.Ge, '!=': sp.Ne, '==': sp.Eq, '<': sp.Lt,
                        '>': sp.Gt, '=': sp.Eq}[op](a, b, evaluate=False)
            if len(partes) == 3 and op in ('<', '<='):
                # a < x < b
                a, m_, b = (leer(p, extra) for p in partes)
                rel = sp.Lt if op == '<' else sp.Le
                return sp.And(rel(a, m_, evaluate=False), rel(m_, b, evaluate=False))
            if len(partes) == 3 and op in ('>', '>='):
                a, m_, b = (leer(p, extra) for p in partes)
                rel = sp.Gt if op == '>' else sp.Ge
                return sp.And(rel(a, m_, evaluate=False), rel(m_, b, evaluate=False))
    return leer(s_, extra)


def dividir(texto, sep=','):
    """Divide por sep en el nivel superior (respetando paréntesis y corchetes)."""
    partes, nivel, actual = [], 0, ''
    for ch in texto:
        if ch in '([{':
            nivel += 1
        elif ch in ')]}':
            nivel -= 1
        if ch == sep and nivel == 0:
            partes.append(actual)
            actual = ''
        else:
            actual += ch
    partes.append(actual)
    return [p.strip() for p in partes if p.strip() != '']


def leer_matriz(texto):
    expr = leer(texto)
    if isinstance(expr, (list, sp.Tuple)) or hasattr(expr, 'tolist'):
        return sp.Matrix(expr)
    raise ErrorEntrada('Escribí la matriz por filas, por ejemplo [[1,2],[3,4]].')


def es_matriz_texto(texto):
    return texto.strip().startswith('[[')


def leer_lista(expr_o_texto):
    """'[1,2,3]' -> lista de expresiones"""
    e = leer(expr_o_texto) if isinstance(expr_o_texto, str) else expr_o_texto
    if isinstance(e, (list, tuple, sp.Tuple)):
        return [sp.sympify(v) for v in e]
    if isinstance(e, sp.MatrixBase):
        return list(e)
    raise ErrorEntrada('Se esperaba un vector, por ejemplo [1, 2, 3].')


def leer_numero(texto):
    v = leer(texto)
    if not v.is_number:
        raise ErrorEntrada(f'«{texto}» debería ser un número.')
    return v


def variables_de(expr, preferidas=('x', 'y', 'z', 't', 's')):
    """Variables libres ordenadas: primero x, y, z, t; luego alfabéticamente."""
    libres = [v for v in getattr(expr, 'free_symbols', set()) if not str(v).startswith('C')]
    orden = {nombre: i for i, nombre in enumerate(preferidas)}
    return sorted(libres, key=lambda v: (orden.get(str(v), 99), str(v)))


def variable_principal(expr, defecto=x):
    vs = variables_de(expr)
    return vs[0] if vs else defecto


# ----------------------------------------------------------------------------
# Salida
# ----------------------------------------------------------------------------
def tex(obj):
    """LaTeX en notación de la cursada (ln, i)."""
    if isinstance(obj, (list, tuple)):
        return r',\ '.join(tex(o) for o in obj)
    if isinstance(obj, dict):
        return r',\ '.join(f'{tex(k_)} = {tex(v)}' for k_, v in obj.items())
    if isinstance(obj, str):
        return obj
    try:
        return sp.latex(obj, ln_notation=True, imaginary_unit='i', mul_symbol=None,
                        decimal_separator='period')
    except Exception:  # noqa
        return sp.latex(obj)


def texto_valor(obj):
    """Valor en texto parseable por sympify."""
    if obj is None:
        return None
    if isinstance(obj, sp.MatrixBase):
        return str(obj.tolist())
    if isinstance(obj, dict):
        return '{' + ', '.join(f'{k_}: {texto_valor(v)}' for k_, v in obj.items()) + '}'
    if isinstance(obj, (list, tuple)):
        return '[' + ', '.join(texto_valor(v) for v in obj) + ']'
    if isinstance(obj, bool):
        return str(obj)
    return sp.sstr(obj)


def decimal(v, digitos=10):
    try:
        f = sp.N(v, digitos + 5)
        if f.is_real:
            return float(f)
        return complex(f)
    except Exception:  # noqa
        return None


def fmt_num(v, digitos=10):
    """Número para mostrar en tablas."""
    try:
        f = float(v)
    except Exception:  # noqa
        try:
            c = complex(sp.N(v))
            return f'{c.real:.6g}{c.imag:+.6g}i'
        except Exception:  # noqa
            return str(v)
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return f'{f:.{digitos}g}'


class Resultado:
    """Acumula título, pasos, tablas y gráfico de una resolución."""

    def __init__(self, tema, titulo):
        self.tema = tema
        self.titulo = titulo
        self.entrada_latex = ''
        self.resultado_latex = ''
        self.valor = None
        self.valores = {}
        self.pasos = []
        self.tabla = None
        self.grafico = None
        self.extras = []
        self.comparar = None  # (modo, referencia, contexto)
        self.avisos = []

    def paso(self, titulo, texto='', latex='', nivel=0):
        self.pasos.append({'titulo': titulo, 'texto': texto, 'latex': latex, 'nivel': nivel})

    def extra(self, titulo, latex='', texto=''):
        self.extras.append({'titulo': titulo, 'latex': latex, 'texto': texto})

    def a_dict(self):
        valores = {}
        for k_, v in self.valores.items():
            valores[k_] = v if isinstance(v, (str, int, float, bool)) or v is None else texto_valor(v)
        return {
            'ok': True, 'tema': self.tema, 'titulo': self.titulo,
            'entrada_latex': self.entrada_latex, 'resultado_latex': self.resultado_latex,
            'valor': texto_valor(self.valor) if not isinstance(self.valor, str) else self.valor,
            'valores': valores, 'pasos': self.pasos, 'tabla': self.tabla,
            'grafico': self.grafico, 'extras': self.extras, 'avisos': self.avisos,
            'error': None,
        }


def simplificar_bonito(e):
    """Simplifica eligiendo la forma más corta y legible."""
    if not isinstance(e, sp.Basic) or e.is_Number:
        return e
    candidatos = [e]
    for f in (sp.simplify, sp.factor, sp.expand, sp.cancel, sp.trigsimp):
        try:
            candidatos.append(f(e))
        except Exception:  # noqa
            pass
    return min(candidatos, key=lambda c: (sp.count_ops(c), len(str(c))))


def _es_letra(o):
    """Variable, función o potencia de variable: se escribe pegada (2x, 3sen x)."""
    if o is None:
        return False
    if isinstance(o, sp.Pow) and o.exp.is_Rational and not o.exp.is_Integer:
        return True  # radical: 3√6
    if isinstance(o, sp.Pow):
        o = o.base
    return bool(o.is_Symbol or isinstance(o, sp.Function))


def tex_crudo(e):
    """LaTeX de una expresión SIN simplificar, en el orden en que se escribió:
    48 : 6 · 2  ->  \frac{48}{6}\cdot 2   (a·b^-1 se muestra como fracción)."""
    def p(a, padre=None):
        if isinstance(a, sp.Add):
            out = ''
            for i, t_ in enumerate(a.args):
                neg = False
                if isinstance(t_, sp.Mul) and t_.args and t_.args[0] == -1:
                    neg, t_ = True, sp.Mul(*t_.args[1:], evaluate=False)
                elif t_.is_Number and t_ < 0:
                    neg, t_ = True, -t_
                s_ = p(t_, 'add')
                out += (' - ' if neg else (' + ' if i else '')) + s_
                if i == 0 and neg:
                    out = '-' + s_
            return r'\left(' + out + r'\right)' if padre in ('mul', 'pow') else out
        if isinstance(a, sp.Mul):
            piezas = []  # (latex, es_numero, objeto)
            for f in a.args:
                if isinstance(f, sp.Pow) and f.exp == -1 and piezas:
                    _, _, obj = piezas.pop()
                    num = p(obj) if obj is not None else _
                    piezas.append((r'\frac{' + num + '}{' + p(f.base) + '}', False, None))
                elif f.is_Rational and not f.is_Integer and f.p == 1 and piezas:
                    _, _, obj = piezas.pop()
                    num = p(obj) if obj is not None else _
                    piezas.append((r'\frac{' + num + '}{' + str(f.q) + '}', False, None))
                elif f == -1 and not piezas:
                    piezas.append(('-', None, None))
                else:
                    piezas.append((p(f, 'mul'), bool(f.is_Number), f))
            out = ''
            for s_, es_num, obj_actual in piezas:
                if s_ == '-':
                    out += '-'
                    continue
                if out and out != '-':
                    # punto solo antes de un número (2x, x·5, 3·4); si no, yuxtaposición
                    out += r'\,' if _es_letra(obj_actual) else r' \cdot '
                out += s_
            return r'\left(' + out + r'\right)' if padre == 'pow' else out
        if isinstance(a, sp.Pow):
            if a.exp == sp.Rational(1, 2):
                return r'\sqrt{' + p(a.base) + '}'
            if a.exp == -1:
                return r'\frac{1}{' + p(a.base) + '}'
            if a.exp.is_Rational and a.exp.p == 1 and a.exp.q > 2:
                return r'\sqrt[' + str(a.exp.q) + ']{' + p(a.base) + '}'
            return '{' + p(a.base, 'pow') + '}^{' + p(a.exp) + '}'
        if a.is_Rational and not a.is_Integer:
            return (r'-' if a < 0 else '') + r'\frac{' + str(abs(a.p)) + '}{' + str(a.q) + '}'
        if a.args and not a.is_Atom and isinstance(a, sp.Function):
            try:
                return tex(a.func(*a.args))
            except Exception:  # noqa
                return tex(a)
        return tex(a)
    try:
        return p(e)
    except Exception:  # noqa
        return tex(e)


def pasos_jerarquia(crudo, max_pasos=25):
    """Resuelve una cuenta numérica operación por operación respetando la jerarquía:
    potencias y raíces -> multiplicaciones y divisiones -> sumas y restas (de adentro hacia afuera).
    Devuelve [(descripcion, latex)]."""
    nombres = {0: 'Potencias y raíces', 1: 'Multiplicaciones y divisiones', 2: 'Sumas y restas'}
    pasos = []
    actual = crudo
    previo = tex_crudo(actual)
    inertes = set()  # raíces que ya no se pueden simplificar (√6)
    for _ in range(max_pasos * 3):
        candidatos = []
        def _num(a):
            return a.is_Number or (isinstance(a, sp.Pow) and a.exp == -1 and a.base.is_Number)
        for i, nodo in enumerate(sp.preorder_traversal(actual)):
            if nodo.is_Atom or not nodo.args or not all(_num(a) for a in nodo.args):
                continue
            if isinstance(nodo, sp.Pow) and nodo.exp == -1:
                continue  # una división se resuelve junto con su dividendo
            if nodo in inertes:
                continue
            if isinstance(nodo, sp.Add):
                prio = 2
            elif isinstance(nodo, sp.Mul):
                prio = 1
            else:
                prio = 0
            candidatos.append((prio, i, nodo))
        if not candidatos:
            break
        prio, _, nodo = min(candidatos, key=lambda c: (c[0], c[1]))
        es_raiz = nodo == actual
        with sp.evaluate(True):
            valor = sp.sympify(nodo.func(*nodo.args))
            if valor.is_Float:
                v = float(valor)
                valor = sp.Integer(round(v)) if abs(v - round(v)) < 1e-9 * max(1.0, abs(v)) else sp.Float(round(v, 10), 12)
        if not valor.is_Number:
            # raíz no exacta de un entero: se extraen los factores (√54 = √(3²·6) = 3√6)
            if (isinstance(nodo, sp.Pow) and nodo.base.is_Integer and nodo.base > 0 and nodo.exp.is_Rational
                    and nodo.exp.p == 1 and nodo.exp.q >= 2):
                q = nodo.exp.q
                fuera, dentro = 1, 1
                for primo, mult in sp.factorint(int(nodo.base)).items():
                    fuera *= primo ** (mult // q)
                    dentro *= primo ** (mult % q)
                if fuera == 1:
                    inertes.add(nodo)
                    continue
                if fuera > 1:
                    indice = '' if q == 2 else '[' + str(q) + ']'
                    pasos.append(('Simplificar la raíz',
                                  r'\sqrt' + indice + '{' + str(nodo.base) + r'} = \sqrt' + indice + '{' + str(fuera) + '^{' + str(q)
                                  + r'}\cdot ' + str(dentro) + '} = ' + tex(valor)))
                with sp.evaluate(False):
                    actual = actual.xreplace({nodo: valor})
                previo = tex_crudo(actual)
                continue
            break
        with sp.evaluate(False):
            actual = actual.xreplace({nodo: valor})
        nuevo = tex_crudo(actual)
        if nuevo != previo:
            pasos.append(('Paréntesis' if prio == 2 and not es_raiz else nombres[prio], nuevo))
            previo = nuevo
        if actual.is_Number or len(pasos) >= max_pasos:
            break
    return pasos
