"""Geometría analítica (rectas, planos, cónicas, cuádricas), complementos de álgebra lineal
(núcleo, imagen, independencia, bases, diagonalización, Cramer, polinomios) y matemática discreta
(primos, congruencias, Euclides, diofánticas, lógica proposicional, álgebra de Boole, combinatoria).

Se carga desde api.py a través de COMANDOS_EXTRA. Solo usa sympy (sin numpy/scipy)."""
import math
import re
from functools import reduce
from itertools import product as _producto

import sympy as sp
from sympy.logic.boolalg import And, Or, Not, Implies, Equivalent, Xor

from . import graficos as G
from .lineal import gauss_jordan_pasos
from .nucleo import (ErrorEntrada, Resultado, dividir, leer, leer_ecuacion, leer_lista, leer_matriz, tex, fmt_num,
                     normalizar, sin_tildes, variables_de, x, y, z, t)

X3 = (x, y, z)
TEMA_GEO = 'Geometría analítica'
TEMA_LIN = 'Álgebra lineal'
TEMA_DIS = 'Matemática discreta'


# ============================================================================
# Utilidades generales
# ============================================================================
def _ok(msg='¡Correcto!', parcial=False):
    d = {'correcto': True, 'mensaje': msg}
    if parcial:
        d['parcial'] = True
    return d


def _mal(msg, esperado='', pista=''):
    d = {'correcto': False, 'mensaje': msg}
    if esperado:
        d['esperado'] = esperado
    if pista:
        d['pista'] = pista
    return d


def _seguro(fn):
    def envoltura(texto):
        try:
            return fn(str(texto).strip())
        except ErrorEntrada as e:
            return {'correcto': None, 'mensaje': str(e)}
        except Exception as e:  # noqa
            return {'correcto': None, 'mensaje': 'No pude interpretar tu respuesta: ' + str(e)[:120]}
    return envoltura


def _propio(r, ref, fn):
    """Comparación con una función propia: comprobar.py la usa si ctx trae 'comprobar' (ver informe)."""
    r.comparar = ('propio', ref, {'comprobar': _seguro(fn)})


def _simp(e):
    try:
        return sp.simplify(e)
    except Exception:  # noqa
        return e


def _es_cero(e):
    try:
        return _simp(e) == 0
    except Exception:  # noqa
        return False


def _tt(v):
    """(a, b, c) en LaTeX."""
    return r'\left(' + ',\\ '.join(tex(_simp(c)) for c in v) + r'\right)'


def _txt_tupla(v):
    return '(' + ', '.join(sp.sstr(_simp(c)) for c in v) + ')'


def _enteros(vals, positivo=True):
    """Escala una lista para que quede con enteros coprimos (y el primer no nulo positivo)."""
    vals = [sp.nsimplify(_simp(v)) for v in vals]
    if all(v.is_Rational for v in vals) and any(v != 0 for v in vals):
        den = reduce(sp.ilcm, [int(v.q) for v in vals], 1)
        ints = [int(v * den) for v in vals]
        g = reduce(math.gcd, [abs(i) for i in ints], 0) or 1
        vals = [sp.Integer(i // g) for i in ints]
    if positivo:
        primero = next((v for v in vals if v != 0), None)
        try:
            if primero is not None and primero.is_number and primero < 0:
                vals = [-v for v in vals]
        except TypeError:
            pass
    return vals


def _vec_entero(v):
    """Vector con componentes enteras sin cambiar el sentido."""
    return sp.Matrix(_enteros(list(v), positivo=False))


def _paralelos(u, v):
    return sp.Matrix.hstack(sp.Matrix(u), sp.Matrix(v)).rank(simplify=True) <= 1


def _norma(v):
    return _simp(sp.sqrt(sum(c ** 2 for c in v)))


_PALABRAS = {'entre', 'de', 'del', 'la', 'el', 'los', 'las', 'al', 'a', 'y', 'e', 'punto', 'puntos', 'recta',
             'rectas', 'plano', 'planos', 'que', 'pasa', 'pasan', 'por', 'con', 'vector', 'vectores', 'director',
             'un', 'una', 'dos', 'tres', 'desde', 'hasta', 'hacia', 'sus', 'ecuacion', 'ecuaciones', 'dados', 'dado',
             'dadas', 'dada', 'en', 'base', 'polinomios', 'polinomio', 'matriz', 'conjunto', 'subespacio', 'generado',
             'son', 'es', 'los vectores', 'calcular', 'hallar', 'determinar', 'paralela', 'perpendicular', 'sistema',
             'numero', 'numeros', 'n', 'el numero', 'entero', 'enteros', 'formula', 'expresion', 'funcion', 'para'}


def _limpiar(s):
    """Quita palabras sueltas al comienzo ('entre los puntos', 'de la recta'...)."""
    s = s.strip().rstrip('.?')
    while True:
        m = re.match(r"^\s*([A-Za-zÁÉÍÓÚáéíóúÑñ]+)\b(?!\s*[=+\-*/^'<>≡])\s*:?\s*", s)
        if not m or sin_tildes(m.group(1).lower()) not in _PALABRAS or m.end() == 0:
            return s.strip()
        s = s[m.end():]


def _dividir_palabras(s):
    """'(1,2) y (3,4)' -> ['(1,2)', '(3,4)'] (conectores entre objetos, no dentro de ecuaciones)."""
    partes = re.split(r'(?<=[\w)\]])\s+(?:y|e|a|al|con|hasta|por|que\s+pasa\s+por|pasando\s+por|respecto\s+(?:a|de|del))\s+'
                      r'(?=[\w(\[\-])', s)
    return [p.strip() for p in partes if p.strip()]


def _partes(arg, sep=','):
    s = _limpiar(normalizar(arg))
    partes = dividir(s, sep)
    if sep == ',':
        partes = [q_ for p in partes for q_ in _dividir_palabras(p)]
    return [_limpiar(p) for p in partes]


def _entero(texto, nombre='el número'):
    try:
        v = sp.nsimplify(leer(texto))
    except ErrorEntrada:
        raise ErrorEntrada(f'No entendí {nombre}: «{str(texto).strip()}».')
    if not v.is_Integer:
        raise ErrorEntrada(f'{nombre[0].upper() + nombre[1:]} debe ser entero: «{str(texto).strip()}» no lo es.')
    return int(v)


def _sup(n):
    return str(n).translate(str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻'))


def _celda(v):
    v = sp.sympify(v)
    if v.is_Integer:
        return str(v)
    return r'\(' + tex(v) + r'\)'


# ============================================================================
# Lectura de objetos geométricos: puntos, vectores, rectas y planos
# ============================================================================
_FUNC_MULTI = {'root', 'log', 'atan2', 'binomial', 'max', 'min', 'matrix', 'sqrt', 'sin', 'cos', 'tan', 'sen', 'tg',
               'exp', 'ln', 'abs', 'raiz'}


def _cierre(s, i):
    pares = {'(': ')', '[': ']'}
    abre, cierra = s[i], pares[s[i]]
    nivel = 0
    for j in range(i, len(s)):
        if s[j] == abre:
            nivel += 1
        elif s[j] == cierra:
            nivel -= 1
            if nivel == 0:
                return j
    raise ErrorEntrada('Falta cerrar un paréntesis o corchete.')


def _tuplas_a_matriz(s):
    """'(1,2,3)+t(1,0,1)' -> 'Matrix([1,2,3])+t*Matrix([1,0,1])' para operar como vectores."""
    out, i = '', 0
    while i < len(s):
        ch = s[i]
        if ch in '([':
            j = _cierre(s, i)
            interno = _tuplas_a_matriz(s[i + 1:j])
            m = re.search(r'([A-Za-z_][A-Za-z_0-9]*)\s*$', out)
            nombre = m.group(1).lower() if m else None
            if len(dividir(interno)) >= 2 and nombre not in _FUNC_MULTI:
                pre = '*' if re.search(r'[\w)\]]\s*$', out) else ''
                out += pre + 'Matrix([' + interno + '])'
            else:
                out += ch + interno + (')' if ch == '(' else ']')
            i = j + 1
        else:
            out += ch
            i += 1
    return out


def _o_punto(v, es_vector=False):
    v = sp.Matrix([_simp(c) for c in v])
    return {'tipo': 'punto', 'P': v, 'dim': len(v), 'vector': es_vector}


def _o_recta(P, D):
    P = sp.Matrix([_simp(c) for c in P])
    D = sp.Matrix([_simp(c) for c in D])
    if all(_es_cero(c) for c in D):
        raise ErrorEntrada('El vector director no puede ser nulo.')
    o = {'tipo': 'recta', 'P': P, 'D': D, 'dim': len(P)}
    if len(P) == 2:
        a, b = D[1], -D[0]
        c = -(a * P[0] + b * P[1])
        o['abc'] = _enteros([a, b, c])
    return o


def _o_plano(n, d):
    vals = _enteros(list(n) + [d])
    n, d = sp.Matrix(vals[:3]), vals[3]
    if all(c == 0 for c in n):
        raise ErrorEntrada('La ecuación no representa un plano (coeficientes de x, y, z nulos).')
    return {'tipo': 'plano', 'n': n, 'd': d, 'dim': 3}


def _ecuacion_lineal(texto):
    """'a x + b y + c z = d' -> (a, b, c, d, usa_z)"""
    eq = leer_ecuacion(texto)
    if not isinstance(eq, sp.Equality):
        raise ErrorEntrada(f'Se esperaba una ecuación con «=»: «{texto}».')
    if not all(isinstance(m_, sp.Expr) for m_ in (eq.lhs, eq.rhs)):
        raise ErrorEntrada(f'No entendí «{texto}»: separá cada objeto con una coma.')
    e = sp.expand(eq.lhs - eq.rhs)
    if not e.free_symbols <= set(X3):
        raise ErrorEntrada('En rectas y planos usá las variables x, y, z.')
    if not e.free_symbols:
        raise ErrorEntrada(f'«{texto}» no tiene variables.')
    try:
        grado = sp.Poly(e, *X3).total_degree()
    except sp.PolynomialError:
        grado = 99
    if grado != 1:
        raise ErrorEntrada(f'«{texto}» no es una ecuación lineal (de primer grado): no es recta ni plano.')
    a, b, c = (e.coeff(v) for v in X3)
    d = -e.subs({x: 0, y: 0, z: 0})
    return a, b, c, d, (z in e.free_symbols or bool(re.search(r'(?<![a-z])z(?![a-z])', texto)))


def _recta_de_planos(p1, p2):
    D = p1['n'].cross(p2['n'])
    if all(_es_cero(c) for c in D):
        raise ErrorEntrada('Los dos planos son paralelos: no determinan una recta.')
    D = _vec_entero(D)
    i = next(k for k in range(3) if not _es_cero(D[k]))
    otras = [v for k, v in enumerate(X3) if k != i]
    sis = [p['n'].dot(sp.Matrix(X3)) - p['d'] for p in (p1, p2)]
    sis = [e.subs(X3[i], 0) for e in sis]
    sol = sp.solve(sis, otras, dict=True)
    if not sol:
        raise ErrorEntrada('No pude encontrar un punto común a los dos planos.')
    P = sp.Matrix([sol[0].get(v, 0) if v != X3[i] else 0 for v in X3])
    o = _o_recta(P, D)
    o['planos'] = (p1, p2)
    return o


def _param_a_objeto(vec, es_tupla=True):
    """Matriz con 0, 1 o 2 parámetros -> punto, recta o plano."""
    vec = sp.Matrix(vec)
    params = sorted(vec.free_symbols - (set(X3) if len(vec) == 3 else set()), key=str)
    if any(s in vec.free_symbols for s in X3) and len(vec) == 3:
        params = sorted(vec.free_symbols, key=str)
    if not params:
        return _o_punto(vec, not es_tupla)
    if len(params) == 1:
        p = params[0]
        P, D = vec.subs(p, 0), vec.diff(p)
        if any(D_.has(p) for D_ in D):
            raise ErrorEntrada('La ecuación paramétrica debe ser lineal en el parámetro.')
        return _o_recta(P, D)
    if len(params) == 2 and len(vec) == 3:
        s1, s2 = params
        P, u, v = vec.subs({s1: 0, s2: 0}), vec.diff(s1), vec.diff(s2)
        n = u.cross(v)
        if all(_es_cero(c) for c in n):
            raise ErrorEntrada('Los dos vectores directores del plano son paralelos.')
        return _o_plano(n, n.dot(P))
    raise ErrorEntrada('Demasiados parámetros en la ecuación vectorial.')


def _simetrica(s):
    """(x-1)/2 = (y-2)/3 = (z-3)/1 -> recta"""
    trozos = s.split('=')
    vs, P, D = [], {}, {}
    for tr in trozos:
        e = leer(tr)
        libres = e.free_symbols & set(X3)
        if len(libres) != 1 or e.free_symbols - set(X3):
            raise ErrorEntrada('En la forma simétrica cada miembro debe tener una sola variable: (x-1)/2 = (y-2)/3.')
        v = libres.pop()
        k = sp.diff(e, v)
        if k == 0 or sp.diff(e, v, 2) != 0:
            raise ErrorEntrada('La forma simétrica debe ser lineal: (x-x0)/a = (y-y0)/b.')
        P[v] = -e.subs(v, 0) / k
        D[v] = 1 / k
        vs.append(v)
    if len(vs) == 2 and set(vs) == {x, y}:
        return _o_recta([P[x], P[y]], [D[x], D[y]])
    if len(vs) == 3:
        return _o_recta([P[v] for v in X3], _enteros([D[v] for v in X3], False))
    raise ErrorEntrada('No entendí la forma simétrica de la recta.')


def _leer_objeto(texto):
    if isinstance(texto, tuple) and texto[0] == 'param':
        return _param_a_objeto([leer(c) for c in texto[1]])
    s0 = normalizar(str(texto)).strip()
    s0 = re.sub(r'^\s*(?:[A-Za-z]{1,2}\d?|pi\d?)\s*:\s*(?=\S)', '', s0)  # etiquetas "r: ..." o "π1: ..."
    s0 = _limpiar(s0)
    if not s0:
        raise ErrorEntrada('Falta un punto, recta o plano.')
    if ',' in s0 and s0.count('=') >= 2:
        grupo = _agrupar_parametricas(dividir(s0, ','))
        if len(grupo) == 1 and isinstance(grupo[0], tuple):
            return _leer_objeto(grupo[0])
    if ';' in s0:
        partes = dividir(s0, ';')
        grupo = _agrupar_parametricas(partes)
        if len(grupo) == 1 and isinstance(grupo[0], tuple):
            return _leer_objeto(grupo[0])
        if len(partes) == 2:
            o1, o2 = (_leer_objeto(p) for p in partes)
            o1, o2 = (_como_plano(o) for o in (o1, o2))
            return _recta_de_planos(o1, o2)
        raise ErrorEntrada('Para una recta en R³ como intersección de planos escribí dos ecuaciones separadas por «;».')
    m = re.match(r'^\s*[\(\[]\s*x\s*,\s*y\s*(?:,\s*z\s*)?[\)\]]\s*=\s*', s0)
    if m:
        s0 = s0[m.end():]
    elif s0.count('=') >= 2:
        return _simetrica(s0)
    elif '=' in s0:
        a, b, c, d, usa_z = _ecuacion_lineal(s0)
        return {'tipo': 'ecuacion', 'coef': (a, b, c), 'd': d, 'dim': 3 if usa_z else 2, 'texto': s0}
    es_tupla = s0.lstrip().startswith('(')
    e = leer(_tuplas_a_matriz(s0))
    if isinstance(e, (list, tuple, sp.Tuple)):
        e = sp.Matrix(list(e))
    if not isinstance(e, sp.MatrixBase):
        raise ErrorEntrada(f'No entendí «{s0}» como punto, recta o plano. Ejemplos: (1,2), 3x+4y-5=0, '
                           '(x,y,z)=(1,0,2)+t(1,1,1), x+y+z=1.')
    return _param_a_objeto(e, es_tupla)


def _agrupar_parametricas(partes):
    """['x=1+t', 'y=2-t', 'z=3t'] -> [('param', ['1+t', '2-t', '3t'])]"""
    out, grupo = [], []

    def cerrar():
        if not grupo:
            return
        nombres = [g_[0] for g_ in grupo]
        if nombres in (['x', 'y'], ['x', 'y', 'z']):
            out.append(('param', [g_[1] for g_ in grupo]))
        else:
            out.extend(f'{a}={b}' for a, b in grupo)
        grupo.clear()

    for p in partes:
        if not isinstance(p, str):
            cerrar()
            out.append(p)
            continue
        m = re.match(r'^\s*([xyz])\s*=\s*(.+)$', p)
        es_param = False
        if m:
            try:
                libres = leer(m.group(2)).free_symbols
                es_param = bool(libres) and not (libres & set(X3))
            except ErrorEntrada:
                es_param = False
        if es_param:
            grupo.append((m.group(1), m.group(2)))
        else:
            cerrar()
            out.append(p)
    cerrar()
    return out


def _como_plano(o):
    if o['tipo'] == 'plano':
        return o
    if o['tipo'] == 'ecuacion':
        a, b, c = o['coef']
        return _o_plano(sp.Matrix([a, b, c]), o['d'])
    raise ErrorEntrada('Se esperaba la ecuación de un plano, por ejemplo x+y+z=1.')


def _contexto(objs):
    """Decide si las ecuaciones sin z son rectas del plano o planos del espacio."""
    dim = 3 if any(o['dim'] == 3 for o in objs) else 2
    out = []
    for o in objs:
        if o['tipo'] == 'ecuacion':
            a, b, c = o['coef']
            if dim == 3:
                o = _o_plano(sp.Matrix([a, b, c]), o['d'])
            else:
                if a == 0 and b == 0:
                    raise ErrorEntrada('La ecuación no representa una recta.')
                D = sp.Matrix([-b, a])
                P = sp.Matrix([0, o['d'] / b]) if b != 0 else sp.Matrix([o['d'] / a, 0])
                o = _o_recta(P, _vec_entero(D))
        elif o['dim'] != dim:
            raise ErrorEntrada('Mezclaste objetos del plano (2 coordenadas) y del espacio (3 coordenadas).')
        out.append(o)
    return out


def _objetos(arg, cantidad=None):
    partes = _agrupar_parametricas(_partes(arg))
    objs = _contexto([_leer_objeto(p) for p in partes])
    if cantidad is not None and len(objs) != cantidad:
        raise ErrorEntrada(f'Se esperaban {cantidad} objetos (puntos, rectas o planos) separados por coma.')
    return objs


def _punto_de(texto):
    o = _leer_objeto(texto)
    if o['tipo'] != 'punto':
        raise ErrorEntrada(f'«{texto}» debería ser un punto, por ejemplo (1, 2) o (1, 0, 2).')
    return o['P']


# ---------------------------------------------------------------------------
# Escritura de rectas y planos
# ---------------------------------------------------------------------------
def _tex_plano(o):
    n = o['n']
    return tex(n[0] * x + n[1] * y + n[2] * z) + ' = ' + tex(o['d'])


def _tex_general2(abc):
    a, b, c = abc
    return tex(a * x + b * y + c) + ' = 0'


def _tex_recta(o):
    if o['dim'] == 2:
        return _tex_general2(o['abc'])
    return r'(x, y, z) = ' + _tt(o['P']) + r' + t\,' + _tt(o['D'])


def _tex_obj(o):
    if o['tipo'] == 'punto':
        return _tt(o['P'])
    if o['tipo'] == 'plano':
        return _tex_plano(o)
    return _tex_recta(o)


def _explicita(abc):
    """y = m x + h (o None si es vertical)"""
    a, b, c = abc
    if b == 0:
        return None
    return sp.expand(-(a * x + c) / b)


def _ec_recta_dir(P, d):
    """Ecuación (en LaTeX) de la recta del plano por P con dirección d; y su expresión implícita."""
    d0, d1 = _simp(d[0]), _simp(d[1])
    if _es_cero(d0):
        return 'x = ' + tex(_simp(P[0])), x - P[0]
    m = _simp(d1 / d0)
    yy = sp.expand(_simp(P[1] + m * (x - P[0])))
    return 'y = ' + tex(yy), y - yy


# ============================================================================
# Comparadores propios
# ============================================================================
def _resp_objeto(texto, dim):
    t_ = re.sub(r'^\s*(?:r|L|recta|plano|pi\d?|π\d?)\s*:\s*', '', texto.strip(), flags=re.I)
    o = _leer_objeto(t_)
    if o['tipo'] == 'ecuacion':
        o = _contexto([o, {'tipo': 'punto', 'dim': dim, 'P': sp.zeros(dim, 1)}])[0]
    return o


def _chk_recta(ref):
    def f(texto):
        o = _resp_objeto(texto, ref['dim'])
        if o['tipo'] == 'plano':
            return _mal('Una sola ecuación lineal en R³ es un plano, no una recta. Escribí la recta en forma '
                        'vectorial (x,y,z) = P + t·d o como dos ecuaciones separadas por «;».', _tex_recta(ref))
        if o['tipo'] != 'recta' or o['dim'] != ref['dim']:
            return _mal('Tu respuesta no es una recta de la misma dimensión.', _tex_recta(ref))
        if not _paralelos(o['D'], ref['D']):
            return _mal('La dirección de tu recta no coincide.', _tex_recta(ref),
                        'Compará los vectores directores: deben ser proporcionales.')
        if not _paralelos(o['P'] - ref['P'], ref['D']):
            return _mal('Tu recta es paralela a la correcta pero no pasa por los mismos puntos.', _tex_recta(ref))
        return _ok('¡Correcto! Es la misma recta.')
    return f


def _chk_plano(ref):
    def f(texto):
        o = _resp_objeto(texto, 3)
        if o['tipo'] != 'plano':
            return _mal('Tu respuesta no es la ecuación de un plano.', _tex_plano(ref))
        if not _paralelos(o['n'], ref['n']):
            return _mal('El vector normal de tu plano no coincide.', _tex_plano(ref))
        k = next(i for i in range(3) if ref['n'][i] != 0)
        if not _es_cero(o['d'] * ref['n'][k] - ref['d'] * o['n'][k]):
            return _mal('Tu plano es paralelo al correcto, pero el término independiente no coincide.', _tex_plano(ref))
        return _ok('¡Correcto! Es el mismo plano.')
    return f


def _chk_palabras(correcta, grupos, esperado=''):
    """grupos: {clave: [palabras]}; correcta: clave."""
    def f(texto):
        t_ = sin_tildes(texto.lower())
        for clave, palabras in grupos.items():
            if any(re.search(p, t_) for p in palabras):
                if clave == correcta:
                    return _ok()
                return _mal('No es correcto.', esperado or correcta)
        return {'correcto': None, 'mensaje': 'Respondé con palabras, por ejemplo: ' + ', '.join(grupos) + '.'}
    return f


def _vectores_resp(texto):
    t_ = normalizar(texto)
    t_ = re.sub(r'(?i)\b(gen|span|base|bases|b)\b\s*(=)?', ' ', t_)
    t_ = t_.replace('⟨', '{').replace('⟩', '}').replace('<', '{').replace('>', '}')
    t_ = t_.strip().strip('=').strip()
    if t_.startswith('{') and t_.endswith('}'):
        t_ = t_[1:-1]
    partes = dividir(t_, ';') if ';' in t_ else dividir(t_, ',')
    vecs = []
    for p in partes:
        e = leer(p)
        if isinstance(e, (list, tuple, sp.Tuple)):
            vecs.append(sp.Matrix(list(e)))
        elif isinstance(e, sp.MatrixBase):
            vecs.append(sp.Matrix(list(e)))
        else:
            vecs.append(e)
    return vecs


def _chk_base(base, contiene, n):
    dim = len(base)

    def f(texto):
        t_ = sin_tildes(texto.lower())
        if dim == 0 and re.search(r'\{\s*0\s*\}|nulo|vacio|solo el (cero|0)|^\s*0\s*$', t_):
            return _ok('¡Correcto! El subespacio es {0}: dimensión 0, base vacía.')
        vecs = _vectores_resp(texto)
        if len(vecs) == 1 and not isinstance(vecs[0], sp.MatrixBase):
            if sp.sympify(vecs[0]) == dim:
                return _ok(f'¡Correcto! La dimensión es {dim}.')
            return _mal(f'La dimensión no es {tex(vecs[0])}.', str(dim))
        vecs = [v for v in vecs if isinstance(v, sp.MatrixBase)]
        if any(len(v) != n for v in vecs):
            return _mal(f'Los vectores deben tener {n} componentes.')
        if dim == 0:
            if all(all(c == 0 for c in v) for v in vecs):
                return _ok('¡Correcto! El subespacio es {0}.')
            return _mal('El subespacio es solo {0}.', r'\{0\}')
        fuera = [v for v in vecs if not contiene(v)]
        if fuera:
            return _mal('Hay vectores que no pertenecen al subespacio: ' + ', '.join(_tt(v) for v in fuera) + '.',
                        r'\operatorname{gen}\{' + ', '.join(_tt(v) for v in base) + r'\}')
        rg = sp.Matrix.hstack(*vecs).rank(simplify=True) if vecs else 0
        if rg < len(vecs):
            return _mal('Tus vectores son linealmente dependientes: una base no puede tener vectores de más.',
                        r'\operatorname{gen}\{' + ', '.join(_tt(v) for v in base) + r'\}')
        if len(vecs) < dim:
            return _mal(f'Te faltan vectores: la dimensión es {dim}.',
                        r'\operatorname{gen}\{' + ', '.join(_tt(v) for v in base) + r'\}')
        return _ok('¡Correcto! Es una base válida (puede diferir de la del motor y estar bien).')
    return f


# ============================================================================
# 1. GEOMETRÍA ANALÍTICA
# ============================================================================
def _informe_recta(r, P, D, puntos=()):
    o = _o_recta(P, D)
    P, D = o['P'], o['D']
    if o['dim'] == 2:
        a, b, c = o['abc']
        r.paso('Ecuación vectorial', r'Punto de paso \(P_0\) más múltiplos del vector director.',
               r'(x, y) = ' + _tt(P) + r' + t\,' + _tt(D), 0)
        r.paso('Ecuaciones paramétricas', '', r'\begin{cases} x = ' + tex(sp.expand(P[0] + D[0] * t)) + r' \\ y = '
               + tex(sp.expand(P[1] + D[1] * t)) + r'\end{cases}', 0)
        if not _es_cero(D[0]) and not _es_cero(D[1]):
            r.paso('Ecuación simétrica (continua)', 'Se despeja t de cada paramétrica y se iguala.',
                   r'\frac{' + tex(x - P[0]) + '}{' + tex(D[0]) + r'} = \frac{' + tex(y - P[1]) + '}{' + tex(D[1]) + '}', 0)
        r.paso('Ecuación general (implícita)', r'\(ax + by + c = 0\) con \((a, b)\) normal a la recta.',
               _tex_general2((a, b, c)), 0)
        h = _explicita((a, b, c))
        if h is not None:
            m = _simp(-a / b)
            oo_ = _simp(-c / b)
            r.paso('Ecuación explícita', r'Se despeja y: \(y = mx + b\).', r'y = ' + tex(h) + r',\quad m = ' + tex(m)
                   + r',\ b = ' + tex(oo_), 0)
            r.valores['pendiente'] = m
            r.valores['ordenada'] = oo_
            if c != 0 and a != 0:
                p_, q_ = _simp(-c / a), _simp(-c / b)
                r.paso('Ecuación segmentaria', 'Se divide por el término independiente: corta a los ejes en p y q.',
                       r'\frac{x}{' + tex(p_) + r'} + \frac{y}{' + tex(q_) + '} = 1', 0)
                r.valores['abscisa_origen'] = p_
            r.valor = h
            r.resultado_latex = 'y = ' + tex(h)
        else:
            r.paso('Recta vertical', 'El vector director es vertical: la pendiente no existe.', 'x = ' + tex(P[0]), 0)
            r.valor = sp.Eq(x, P[0])
            r.resultado_latex = 'x = ' + tex(P[0])
        r.valores['general'] = sp.sstr(a * x + b * y + c) + ' = 0'
        r.valores['director'] = list(D)
        pts = [G.punto(p[0], p[1], et + ' ' + _txt_tupla(p)) for p, et in puntos]
        if h is not None:
            r.grafico = G.grafico_2d([h], x, pts)
        else:
            g = G.grafico_implicita(x - P[0], (x, y), r.resultado_latex)
            g['puntos'] = [p for p in pts if p]
            r.grafico = g
    else:
        r.paso('Ecuación vectorial', r'Punto de paso \(P_0\) más múltiplos del vector director.',
               r'(x, y, z) = ' + _tt(P) + r' + t\,' + _tt(D), 0)
        r.paso('Ecuaciones paramétricas', '', r'\begin{cases} x = ' + tex(sp.expand(P[0] + D[0] * t)) + r' \\ y = '
               + tex(sp.expand(P[1] + D[1] * t)) + r' \\ z = ' + tex(sp.expand(P[2] + D[2] * t)) + r'\end{cases}', 0)
        no_nulas = [i for i in range(3) if not _es_cero(D[i])]
        fr = [r'\frac{' + tex(X3[i] - P[i]) + '}{' + tex(D[i]) + '}' for i in no_nulas]
        fijas = [tex(X3[i]) + ' = ' + tex(P[i]) for i in range(3) if i not in no_nulas]
        r.paso('Ecuaciones simétricas (continuas)', 'Si una componente del director es 0, esa coordenada queda fija.',
               ' = '.join(fr) + (r',\quad ' + r',\ '.join(fijas) if fijas else ''), 0)
        normales = [_vec_entero(v) for v in sp.Matrix(D).T.nullspace()]
        ecs = [tex(nv[0] * x + nv[1] * y + nv[2] * z) + ' = ' + tex(_simp(nv.dot(P))) for nv in normales[:2]]
        r.paso('Ecuaciones cartesianas', 'La recta como intersección de dos planos que la contienen.',
               r'\begin{cases} ' + r' \\ '.join(ecs) + r'\end{cases}', 0)
        r.valor = [sp.expand(P[i] + D[i] * t) for i in range(3)]
        r.resultado_latex = r'(x, y, z) = ' + _tt(P) + r' + t\,' + _tt(D)
        r.valores['director'] = list(D)
        r.valores['punto'] = list(P)
    _propio(r, r.valor, _chk_recta(o))
    return r


def recta_por(arg):
    partes = _partes(arg)
    if len(partes) != 2:
        raise ErrorEntrada('Uso: recta por (x1, y1), (x2, y2)   — dos puntos del plano o del espacio.')
    A, B = _punto_de(partes[0]), _punto_de(partes[1])
    if len(A) != len(B) or len(A) not in (2, 3):
        raise ErrorEntrada('Los dos puntos deben tener 2 coordenadas (plano) o 3 (espacio).')
    if all(_es_cero(a - b) for a, b in zip(A, B)):
        raise ErrorEntrada('Los dos puntos son iguales: por un solo punto pasan infinitas rectas.')
    D = B - A
    r = Resultado(TEMA_GEO, 'Recta que pasa por dos puntos')
    r.entrada_latex = 'A = ' + _tt(A) + r',\quad B = ' + _tt(B)
    r.paso('Vector director', r'\(\vec d = \overrightarrow{AB} = B - A\)', r'\vec d = ' + _tt(B) + ' - ' + _tt(A) + ' = '
           + _tt(D), 0)
    if len(A) == 2 and not _es_cero(D[0]):
        r.paso('Pendiente', r'\(m = \frac{y_2 - y_1}{x_2 - x_1}\)', r'm = \frac{' + tex(B[1]) + ' - ' + tex(A[1]) + '}{'
               + tex(B[0]) + ' - ' + tex(A[0]) + '} = ' + tex(_simp(D[1] / D[0])), 0)
    Dn = _vec_entero(D)
    if list(Dn) != list(D):
        r.paso('Director simplificado', 'Cualquier múltiplo no nulo sirve como director.', r'\vec d = ' + _tt(Dn), 1)
    return _informe_recta(r, A, Dn, [(A, 'A'), (B, 'B')])


def recta_pendiente(arg):
    partes = _partes(arg)
    if len(partes) != 2:
        raise ErrorEntrada('Uso: recta pendiente punto m, (x0, y0)   — por ejemplo: recta pendiente punto 2, (1, 3)')
    pt, m = None, None
    for p in partes:
        e = leer(_tuplas_a_matriz(p))
        if isinstance(e, sp.MatrixBase) or isinstance(e, (list, sp.Tuple)):
            pt = sp.Matrix(list(e))
        else:
            m = e
    if pt is None or m is None or len(pt) != 2:
        raise ErrorEntrada('Indicá la pendiente (un número) y un punto del plano (x0, y0).')
    r = Resultado(TEMA_GEO, 'Recta dada la pendiente y un punto')
    r.entrada_latex = 'm = ' + tex(m) + r',\quad P_0 = ' + _tt(pt)
    r.paso('Ecuación punto-pendiente', r'\(y - y_0 = m\,(x - x_0)\)', tex(y - pt[1]) + ' = ' + tex(m) + r'\,('
           + tex(x - pt[0]) + ')', 0)
    return _informe_recta(r, pt, _vec_entero(sp.Matrix([1, m])), [(pt, 'P₀')])


def recta_relativa(arg, como):
    objs = _objetos(arg, 2)
    pt = next((o for o in objs if o['tipo'] == 'punto'), None)
    ref = next((o for o in objs if o['tipo'] != 'punto'), None)
    if pt is None or ref is None:
        raise ErrorEntrada(f'Uso: recta {como} <recta o plano>, <punto>   — ej.: recta {como} 2x+y=3, (1, 1)')
    P = pt['P']
    r = Resultado(TEMA_GEO, f'Recta {como}')
    r.entrada_latex = r'\text{' + como + ' a } ' + _tex_obj(ref) + r'\ \text{por } ' + _tt(P)
    if ref['tipo'] == 'plano':
        if como == 'paralela':
            raise ErrorEntrada('Hay infinitas rectas paralelas a un plano por un punto; pedí la perpendicular.')
        D = ref['n']
        r.paso('Dirección', 'Una recta perpendicular a un plano tiene la dirección de su vector normal.',
               r'\vec d = \vec n = ' + _tt(D), 0)
    elif ref['dim'] == 2:
        a, b, _ = ref['abc']
        D = ref['D'] if como == 'paralela' else sp.Matrix([a, b])
        r.paso('Dirección', 'Paralela: mismo director. Perpendicular: el director es el normal (a, b) de la recta dada.',
               r'\vec d = ' + _tt(D), 0)
    else:
        if como == 'paralela':
            D = ref['D']
            r.paso('Dirección', 'Rectas paralelas tienen directores proporcionales.', r'\vec d = ' + _tt(D), 0)
        else:
            Q0, Dr = ref['P'], ref['D']
            lam = _simp((P - Q0).dot(Dr) / Dr.dot(Dr))
            pie = (Q0 + lam * Dr).applyfunc(_simp)
            D = pie - P
            if all(_es_cero(c) for c in D):
                raise ErrorEntrada('El punto está sobre la recta: hay infinitas perpendiculares.')
            r.paso('Pie de la perpendicular', r'Punto Q de la recta con \(\overrightarrow{PQ}\cdot\vec d = 0\).',
                   'Q = ' + _tt(pie), 0)
            D = _vec_entero(D)
    return _informe_recta(r, P, _vec_entero(D), [(P, 'P')] if len(P) == 2 else [])


def recta_auto(arg):
    partes = _partes(arg)
    if len(partes) == 2 and all(p.strip().startswith('(') for p in partes):
        return recta_por(arg)
    return recta_pendiente(arg)


def plano_por(arg):
    s = normalizar(arg)
    trozos = _partes(s)
    normal_idx = [i for i, p in enumerate(dividir(_limpiar(s), ',')) if re.match(r'^\s*(?:con\s+)?(?:vector\s+)?normal', sin_tildes(p.lower()))]
    trozos = [re.sub(r'^\s*(?:vector\s+)?normal\s*(?:n\s*=)?\s*', '', p, flags=re.I) for p in trozos]
    objs = _contexto([_leer_objeto(p) for p in _agrupar_parametricas(trozos)] +
                     [{'tipo': 'punto', 'dim': 3, 'P': sp.zeros(3, 1)}])[:-1]
    r = Resultado(TEMA_GEO, 'Ecuación del plano')
    puntos = [o for o in objs if o['tipo'] == 'punto' and not o['vector']]
    vecs = [o for o in objs if o['tipo'] == 'punto' and o['vector']]
    rectas = [o for o in objs if o['tipo'] == 'recta']
    if any(o['dim'] != 3 for o in objs):
        raise ErrorEntrada('Un plano necesita puntos del espacio (x, y, z).')
    if len(puntos) == 3 and not vecs and not rectas:
        A, B, C = (o['P'] for o in puntos)
        u, v = B - A, C - A
        n = u.cross(v)
        r.entrada_latex = 'A = ' + _tt(A) + r',\ B = ' + _tt(B) + r',\ C = ' + _tt(C)
        r.paso('Dos vectores del plano', '', r'\overrightarrow{AB} = ' + _tt(u) + r',\quad \overrightarrow{AC} = ' + _tt(v), 0)
        r.paso('Vector normal', r'\(\vec n = \overrightarrow{AB}\times\overrightarrow{AC}\)', r'\vec n = ' + _tt(n), 0)
        if all(_es_cero(c) for c in n):
            raise ErrorEntrada('Los tres puntos están alineados: no determinan un único plano.')
        P0 = A
    elif len(puntos) + len(vecs) >= 1 and rectas and len(objs) == 2:
        P0 = (puntos or vecs)[0]['P']
        L = rectas[0]
        n = (P0 - L['P']).cross(L['D'])
        r.entrada_latex = 'P = ' + _tt(P0) + r',\quad r: ' + _tex_recta(L)
        r.paso('Vector normal', r'Se cruza el director de la recta con un vector que va de la recta al punto.',
               r'\vec n = ' + _tt(P0 - L['P']) + r'\times' + _tt(L['D']) + ' = ' + _tt(n), 0)
        if all(_es_cero(c) for c in n):
            raise ErrorEntrada('El punto está sobre la recta: hay infinitos planos que la contienen.')
    elif len(rectas) == 2 and len(objs) == 2:
        L1, L2 = rectas
        if _paralelos(L1['D'], L2['D']):
            n = (L2['P'] - L1['P']).cross(L1['D'])
        else:
            n = L1['D'].cross(L2['D'])
            if not _es_cero((L2['P'] - L1['P']).dot(n)):
                raise ErrorEntrada('Las rectas se cruzan (son alabeadas): no hay un plano que contenga a ambas.')
        if all(_es_cero(c) for c in n):
            raise ErrorEntrada('Las rectas son coincidentes: hay infinitos planos que las contienen.')
        P0 = L1['P']
        r.entrada_latex = r'r_1: ' + _tex_recta(L1) + r',\quad r_2: ' + _tex_recta(L2)
        r.paso('Vector normal', 'Perpendicular a las dos rectas (o a una y al vector que las une si son paralelas).',
               r'\vec n = ' + _tt(n), 0)
    elif len(puntos) + len(vecs) == 2 and len(objs) == 2:
        P0 = (puntos or vecs)[0]['P']
        n = (vecs[-1] if vecs else puntos[1])['P'] if (vecs and puntos) or normal_idx else None
        if n is None:
            raise ErrorEntrada('Con un punto y un vector escribí: plano por (1,2,3), normal [1,1,1].')
        r.entrada_latex = r'P_0 = ' + _tt(P0) + r',\quad \vec n = ' + _tt(n)
        r.paso('Ecuación punto-normal', r'\(\vec n\cdot (X - P_0) = 0\)', _tt(n) + r'\cdot (x - ' + tex(P0[0]) + ', y - '
               + tex(P0[1]) + ', z - ' + tex(P0[2]) + ') = 0', 0)
    elif len(puntos) == 1 and len(vecs) == 2:
        P0 = puntos[0]['P']
        u, v = vecs[0]['P'], vecs[1]['P']
        n = u.cross(v)
        r.entrada_latex = r'P_0 = ' + _tt(P0) + r',\ \vec u = ' + _tt(u) + r',\ \vec v = ' + _tt(v)
        r.paso('Vector normal', r'\(\vec n = \vec u\times\vec v\)', r'\vec n = ' + _tt(n), 0)
        if all(_es_cero(c) for c in n):
            raise ErrorEntrada('Los vectores directores son paralelos.')
    else:
        raise ErrorEntrada('Uso: plano por (1,0,0), (0,1,0), (0,0,1)  ·  plano por (1,2,3), normal [1,1,1]  ·  '
                           'plano por (1,2,3), (x,y,z)=(0,0,0)+t(1,1,0)')
    return _informe_plano(r, n, P0)


def _informe_plano(r, n, P0):
    o = _o_plano(n, sp.Matrix(n).dot(P0))
    a, b, c = o['n']
    d = o['d']
    if list(o['n']) != list(n):
        r.paso('Normal simplificado', 'Se puede usar cualquier múltiplo del vector normal.', r'\vec n = ' + _tt(o['n']), 1)
    r.paso('Ecuación general', r'\(ax + by + cz = d\), con \(d = \vec n\cdot P_0\).', _tex_plano(o), 0)
    r.paso('Forma implícita', '', tex(a * x + b * y + c * z - d) + ' = 0', 1)
    if d != 0 and a != 0 and b != 0 and c != 0:
        r.paso('Ecuación segmentaria', 'Cortes con los ejes: se divide por d.', r'\frac{x}{' + tex(d / a) + r'} + \frac{y}{'
               + tex(d / b) + r'} + \frac{z}{' + tex(d / c) + '} = 1', 0)
    base = [_vec_entero(v) for v in sp.Matrix(o['n']).T.nullspace()]
    if len(base) == 2:
        r.paso('Ecuación vectorial (paramétrica)', 'Dos vectores perpendiculares a la normal generan el plano.',
               r'(x, y, z) = ' + _tt(P0) + r' + s\,' + _tt(base[0]) + r' + t\,' + _tt(base[1]), 0)
    r.valor = sp.Eq(a * x + b * y + c * z, d)
    r.resultado_latex = _tex_plano(o)
    r.valores['normal'] = list(o['n'])
    r.valores['d'] = d
    _propio(r, r.valor, _chk_plano(o))
    if c != 0:
        zf = (d - a * x - b * y) / c
        r.grafico = G.grafico_3d(zf, (x, y), r'z = ' + tex(sp.expand(zf)), rango=4.0)
    return r


# ---------------------------------------------------------------------------
# Distancias, ángulos, posiciones relativas, intersecciones
# ---------------------------------------------------------------------------
def _orden(objs, *tipos):
    """Reordena dos objetos para que coincidan con los tipos pedidos (o None)."""
    a, b = objs
    if (a['tipo'], b['tipo']) == tipos:
        return a, b
    if (b['tipo'], a['tipo']) == tipos:
        return b, a
    return None


def _dist_punto_recta(P, L, r):
    if L['dim'] == 2:
        a, b, c = L['abc']
        num = a * P[0] + b * P[1] + c
        d = _simp(sp.Abs(num) / sp.sqrt(a ** 2 + b ** 2))
        r.paso('Fórmula', r'\(d(P, r) = \frac{|a x_0 + b y_0 + c|}{\sqrt{a^2 + b^2}}\)',
               r'd = \frac{|' + tex(a) + r'\cdot ' + tex(P[0]) + ' + ' + tex(b) + r'\cdot ' + tex(P[1]) + ' + ' + tex(c)
               + r'|}{\sqrt{' + tex(a) + '^2 + ' + tex(b) + '^2}} = \\frac{' + tex(sp.Abs(_simp(num))) + '}{'
               + tex(sp.sqrt(a ** 2 + b ** 2)) + '} = ' + tex(d), 0)
        return d
    AP = P - L['P']
    cr = AP.cross(L['D'])
    d = _simp(_norma(cr) / _norma(L['D']))
    r.paso('Fórmula', r'\(d(P, r) = \frac{\|\overrightarrow{AP}\times\vec d\|}{\|\vec d\|}\), con A un punto de la recta.',
           r'\overrightarrow{AP} = ' + _tt(AP) + r',\quad \overrightarrow{AP}\times\vec d = ' + _tt(cr), 0)
    r.paso('Cálculo', '', r'd = \frac{' + tex(_norma(cr)) + '}{' + tex(_norma(L['D'])) + '} = ' + tex(d), 1)
    return d


def _dist_punto_plano(P, pl, r):
    a, b, c = pl['n']
    num = a * P[0] + b * P[1] + c * P[2] - pl['d']
    d = _simp(sp.Abs(num) / _norma(pl['n']))
    r.paso('Fórmula', r'\(d(P, \pi) = \frac{|a x_0 + b y_0 + c z_0 - d|}{\sqrt{a^2 + b^2 + c^2}}\)',
           r'd = \frac{|' + tex(a) + r'\cdot ' + tex(P[0]) + ' + ' + tex(b) + r'\cdot ' + tex(P[1]) + ' + ' + tex(c)
           + r'\cdot ' + tex(P[2]) + ' - ' + tex(pl['d']) + r'|}{\sqrt{' + tex(a ** 2 + b ** 2 + c ** 2) + '}} = '
           + r'\frac{' + tex(sp.Abs(_simp(num))) + '}{' + tex(_norma(pl['n'])) + '} = ' + tex(d), 0)
    return d


def distancia(arg):
    objs = _objetos(arg, 2)
    r = Resultado(TEMA_GEO, 'Distancia')
    r.entrada_latex = r'd\left(' + _tex_obj(objs[0]) + r',\ ' + _tex_obj(objs[1]) + r'\right)'
    par = _orden(objs, 'punto', 'punto')
    puntos_g = []
    if par:
        A, B = par[0]['P'], par[1]['P']
        dif = B - A
        d = _norma(dif)
        r.titulo = 'Distancia entre dos puntos'
        r.paso('Fórmula', r'\(d(A, B) = \|B - A\| = \sqrt{\sum (b_i - a_i)^2}\)',
               r'd = \sqrt{' + ' + '.join('(' + tex(c) + ')^2' for c in dif) + '} = ' + tex(d), 0)
        if len(A) == 2:
            r.grafico = G.grafico_2d([], x, [G.punto(A[0], A[1], 'A ' + _txt_tupla(A)), G.punto(B[0], B[1], 'B ' + _txt_tupla(B))])
    elif _orden(objs, 'punto', 'recta'):
        Pt, L = _orden(objs, 'punto', 'recta')
        r.titulo = 'Distancia de un punto a una recta'
        d = _dist_punto_recta(Pt['P'], L, r)
        if L['dim'] == 2:
            a, b, c = L['abc']
            P = Pt['P']
            k = (a * P[0] + b * P[1] + c) / (a ** 2 + b ** 2)
            pie = [_simp(P[0] - k * a), _simp(P[1] - k * b)]
            r.extra('Pie de la perpendicular', _tt(pie))
            h = _explicita(L['abc'])
            puntos_g = [G.punto(P[0], P[1], 'P ' + _txt_tupla(P)), G.punto(pie[0], pie[1], 'pie')]
            if h is not None:
                r.grafico = G.grafico_2d([h], x, puntos_g)
    elif _orden(objs, 'punto', 'plano'):
        Pt, pl = _orden(objs, 'punto', 'plano')
        r.titulo = 'Distancia de un punto a un plano'
        d = _dist_punto_plano(Pt['P'], pl, r)
        k = (pl['n'].dot(Pt['P']) - pl['d']) / pl['n'].dot(pl['n'])
        r.extra('Proyección del punto sobre el plano', _tt((Pt['P'] - k * pl['n']).applyfunc(_simp)))
    elif _orden(objs, 'recta', 'recta'):
        L1, L2 = objs
        r.titulo = 'Distancia entre rectas'
        if _paralelos(L1['D'], L2['D']):
            r.paso('Rectas paralelas', 'Se mide la distancia de un punto de una recta a la otra.', '', 0)
            d = _dist_punto_recta(L2['P'], L1, r)
        elif L1['dim'] == 2:
            r.paso('Rectas secantes', 'Dos rectas no paralelas del plano se cortan: la distancia es 0.', '', 0)
            d = sp.Integer(0)
        else:
            n = L1['D'].cross(L2['D'])
            w = L2['P'] - L1['P']
            d = _simp(sp.Abs(w.dot(n)) / _norma(n))
            r.paso('Fórmula (rectas no paralelas)', r'\(d = \frac{|\overrightarrow{P_1P_2}\cdot(\vec d_1\times\vec d_2)|}'
                   r'{\|\vec d_1\times\vec d_2\|}\)', r'\vec d_1\times\vec d_2 = ' + _tt(n) + r',\quad \overrightarrow{P_1P_2} = '
                   + _tt(w) + r'\Rightarrow d = \frac{' + tex(sp.Abs(_simp(w.dot(n)))) + '}{' + tex(_norma(n)) + '} = ' + tex(d), 0)
            r.extra('Posición', '', 'Se cortan (secantes).' if d == 0 else 'Se cruzan (alabeadas).')
    elif _orden(objs, 'plano', 'plano'):
        p1, p2 = objs
        r.titulo = 'Distancia entre planos'
        if _paralelos(p1['n'], p2['n']):
            k = next(i for i in range(3) if p1['n'][i] != 0)
            esc = p2['n'][k] / p1['n'][k]
            d2 = p2['d'] / esc
            d = _simp(sp.Abs(d2 - p1['d']) / _norma(p1['n']))
            r.paso('Planos paralelos', r'Con la misma normal: \(d = \frac{|d_2 - d_1|}{\|\vec n\|}\)',
                   r'd = \frac{|' + tex(d2) + ' - ' + tex(p1['d']) + r'|}{' + tex(_norma(p1['n'])) + '} = ' + tex(d), 0)
        else:
            r.paso('Planos secantes', 'Los planos no son paralelos: se cortan y la distancia es 0.', '', 0)
            d = sp.Integer(0)
    elif _orden(objs, 'recta', 'plano'):
        L, pl = _orden(objs, 'recta', 'plano')
        r.titulo = 'Distancia entre recta y plano'
        if _es_cero(L['D'].dot(pl['n'])):
            r.paso('Recta paralela al plano', r'\(\vec d\cdot\vec n = 0\): se usa un punto de la recta.', '', 0)
            d = _dist_punto_plano(L['P'], pl, r)
        else:
            r.paso('Recta secante al plano', r'\(\vec d\cdot\vec n \ne 0\): la recta corta al plano, la distancia es 0.', '', 0)
            d = sp.Integer(0)
    else:
        raise ErrorEntrada('Combinación no soportada. Ejemplos: distancia (1,2), 3x+4y-5=0 · distancia (1,1,1), x+y+z=1')
    d = _simp(d)
    r.valor = d
    r.valores['decimal'] = fmt_num(sp.N(d), 10)
    r.resultado_latex = 'd = ' + tex(d) + (r'\approx ' + fmt_num(sp.N(d), 6) if not d.is_Integer else '')
    r.comparar = ('numero_o_expr', d, {})
    return r


def angulo_entre(arg):
    partes = _partes(arg)
    if len(partes) == 2 and all(p.strip().startswith('[') for p in partes):
        from .lineal import angulo
        return angulo(', '.join(partes))
    objs = _objetos(arg, 2)
    r = Resultado(TEMA_GEO, 'Ángulo')
    r.entrada_latex = r'\angle\left(' + _tex_obj(objs[0]) + r',\ ' + _tex_obj(objs[1]) + r'\right)'
    if _orden(objs, 'plano', 'plano'):
        u, v = objs[0]['n'], objs[1]['n']
        r.titulo = 'Ángulo entre planos'
        r.paso('Vectores normales', 'El ángulo entre planos es el (agudo) que forman sus normales.',
               r'\vec n_1 = ' + _tt(u) + r',\quad \vec n_2 = ' + _tt(v), 0)
        nombre, fun = r'\cos\theta', sp.acos
    elif _orden(objs, 'recta', 'recta'):
        u, v = objs[0]['D'], objs[1]['D']
        r.titulo = 'Ángulo entre rectas'
        r.paso('Vectores directores', 'Se toma el ángulo agudo entre los directores.',
               r'\vec d_1 = ' + _tt(u) + r',\quad \vec d_2 = ' + _tt(v), 0)
        nombre, fun = r'\cos\theta', sp.acos
    elif _orden(objs, 'recta', 'plano'):
        L, pl = _orden(objs, 'recta', 'plano')
        u, v = L['D'], pl['n']
        r.titulo = 'Ángulo entre recta y plano'
        r.paso('Director y normal', r'El ángulo recta-plano es el complemento del que forma \(\vec d\) con \(\vec n\): '
               r'\(\sin\theta = \frac{|\vec d\cdot\vec n|}{\|\vec d\|\|\vec n\|}\).', r'\vec d = ' + _tt(u) + r',\quad \vec n = ' + _tt(v), 0)
        nombre, fun = r'\sin\theta', sp.asin
    else:
        raise ErrorEntrada('Uso: angulo entre planos π1, π2 · angulo entre rectas r1, r2 · angulo entre recta y plano r, π')
    pu = _simp(u.dot(v))
    val = _simp(sp.Abs(pu) / (_norma(u) * _norma(v)))
    th = _simp(fun(val))
    r.paso('Fórmula', '', nombre + r' = \frac{|' + tex(pu) + '|}{' + tex(_norma(u)) + r'\cdot ' + tex(_norma(v)) + '} = ' + tex(val), 0)
    grados = sp.N(th * 180 / sp.pi)
    r.paso('Ángulo', '', r'\theta = ' + tex(th) + r'\approx ' + fmt_num(grados, 8) + r'^\circ', 0)
    if th == sp.pi / 2:
        r.extra('Conclusión', '', 'Son perpendiculares.')
    elif th == 0:
        r.extra('Conclusión', '', 'Son paralelos (ángulo nulo).')
    r.valor = th
    r.valores['grados'] = fmt_num(grados, 12)
    r.resultado_latex = r'\theta = ' + tex(th) + r'\approx ' + fmt_num(grados, 8) + r'^\circ'
    r.comparar = ('numero_o_expr', th, {})
    return r


def _interseccion_rectas(L1, L2):
    s1, s2 = sp.symbols('s1 s2')
    ecs = list(L1['P'] + s1 * L1['D'] - L2['P'] - s2 * L2['D'])
    sol = sp.solve(ecs, [s1, s2], dict=True)
    if not sol:
        return None
    return (L1['P'] + sol[0].get(s1, 0) * L1['D']).applyfunc(_simp)


def _interseccion_recta_plano(L, pl):
    den = pl['n'].dot(L['D'])
    if _es_cero(den):
        return None, den
    lam = _simp((pl['d'] - pl['n'].dot(L['P'])) / den)
    return (L['P'] + lam * L['D']).applyfunc(_simp), lam


def posicion_relativa(arg):
    objs = _objetos(arg, 2)
    r = Resultado(TEMA_GEO, 'Posición relativa')
    r.entrada_latex = _tex_obj(objs[0]) + r'\quad\text{y}\quad ' + _tex_obj(objs[1])
    grupos = {'coincidentes': [r'coincid', r'iguales', r'misma'], 'paralelos': [r'paralel'],
              'secantes': [r'secant', r'se cortan', r'perpendic', r'concurr'], 'alabeadas': [r'alabead', r'se cruzan'],
              'contenida': [r'contenid', r'incluid', r'pertenece'], 'exterior': [r'no pertenece', r'exterior', r'fuera']}
    if _orden(objs, 'recta', 'recta'):
        L1, L2 = objs
        r.paso('Directores', '', r'\vec d_1 = ' + _tt(L1['D']) + r',\quad \vec d_2 = ' + _tt(L2['D']), 0)
        if _paralelos(L1['D'], L2['D']):
            if _paralelos(L2['P'] - L1['P'], L1['D']):
                pos, txt = 'coincidentes', 'Directores proporcionales y un punto de una está en la otra: son la misma recta.'
            else:
                pos, txt = 'paralelos', 'Directores proporcionales pero sin puntos comunes: paralelas (distintas).'
                d = _dist_punto_recta(L2['P'], L1, Resultado('', ''))
                r.extra('Distancia entre ellas', 'd = ' + tex(d))
        else:
            Q = _interseccion_rectas(L1, L2)
            if Q is not None:
                perp = _es_cero(L1['D'].dot(L2['D']))
                pos = 'secantes'
                txt = 'Directores no proporcionales y hay un punto común: secantes' + (' y perpendiculares.' if perp else '.')
                r.extra('Punto de intersección', _tt(Q))
                r.valores['punto'] = list(Q)
            else:
                pos, txt = 'alabeadas', 'No son paralelas y no se cortan: se cruzan (alabeadas).'
                n = L1['D'].cross(L2['D'])
                d = _simp(sp.Abs((L2['P'] - L1['P']).dot(n)) / _norma(n))
                r.extra('Distancia entre ellas', 'd = ' + tex(d))
        r.paso('Análisis', txt, '', 0)
    elif _orden(objs, 'plano', 'plano'):
        p1, p2 = objs
        r.paso('Normales', '', r'\vec n_1 = ' + _tt(p1['n']) + r',\quad \vec n_2 = ' + _tt(p2['n']), 0)
        if _paralelos(p1['n'], p2['n']):
            if p1['d'] == p2['d'] and list(p1['n']) == list(p2['n']):
                pos, txt = 'coincidentes', 'Ecuaciones proporcionales: es el mismo plano.'
            else:
                pos, txt = 'paralelos', 'Normales proporcionales pero distinto término independiente: paralelos.'
        else:
            perp = _es_cero(p1['n'].dot(p2['n']))
            pos = 'secantes'
            txt = 'Normales no proporcionales: se cortan en una recta' + (' (y son perpendiculares).' if perp else '.')
            L = _recta_de_planos(p1, p2)
            r.extra('Recta de intersección', _tex_recta(L))
        r.paso('Análisis', txt, '', 0)
    elif _orden(objs, 'recta', 'plano'):
        L, pl = _orden(objs, 'recta', 'plano')
        prod = _simp(L['D'].dot(pl['n']))
        r.paso('Director por normal', r'Si \(\vec d\cdot\vec n = 0\) la recta es paralela al plano o está contenida.',
               r'\vec d\cdot\vec n = ' + tex(prod), 0)
        if prod == 0:
            if _es_cero(pl['n'].dot(L['P']) - pl['d']):
                pos, txt = 'contenida', 'Además un punto de la recta verifica la ecuación del plano: está contenida.'
            else:
                pos, txt = 'paralelos', 'Un punto de la recta no está en el plano: la recta es paralela al plano.'
        else:
            Q, _ = _interseccion_recta_plano(L, pl)
            pos, txt = 'secantes', 'La recta corta al plano en un punto' + (' (perpendicular).' if _paralelos(L['D'], pl['n']) else '.')
            r.extra('Punto de intersección', _tt(Q))
            r.valores['punto'] = list(Q)
        r.paso('Análisis', txt, '', 0)
    elif _orden(objs, 'punto', 'recta') or _orden(objs, 'punto', 'plano'):
        Pt, O = _orden(objs, 'punto', 'recta') or _orden(objs, 'punto', 'plano')
        if O['tipo'] == 'plano':
            dentro = _es_cero(O['n'].dot(Pt['P']) - O['d'])
        else:
            dentro = _paralelos(Pt['P'] - O['P'], O['D'])
        pos = 'contenida' if dentro else 'exterior'
        txt = 'El punto pertenece.' if dentro else 'El punto no pertenece (es exterior).'
        r.paso('Se reemplaza el punto', txt, '', 0)
    else:
        raise ErrorEntrada('Uso: posicion relativa <recta o plano>, <recta o plano>')
    nombres = {'coincidentes': 'coincidentes', 'paralelos': 'paralelos/as', 'secantes': 'secantes', 'alabeadas':
               'alabeadas (se cruzan)', 'contenida': 'contenido/a (pertenece)', 'exterior': 'no pertenece'}
    r.valor = pos
    r.resultado_latex = r'\text{' + nombres[pos] + '}'
    _propio(r, pos, _chk_palabras(pos, grupos, nombres[pos]))
    return r


def interseccion(arg):
    objs = _objetos(arg)
    r = Resultado(TEMA_GEO, 'Intersección')
    r.entrada_latex = r'\cap\ '.join(_tex_obj(o) for o in objs) if len(objs) <= 3 else ''
    tipos = sorted(o['tipo'] for o in objs)
    if tipos == ['plano', 'plano']:
        p1, p2 = objs
        r.titulo = 'Intersección de dos planos'
        D = p1['n'].cross(p2['n'])
        r.paso('Dirección de la recta', r'Es perpendicular a ambas normales: \(\vec d = \vec n_1\times\vec n_2\).',
               r'\vec d = ' + _tt(p1['n']) + r'\times' + _tt(p2['n']) + ' = ' + _tt(D), 0)
        if all(_es_cero(c) for c in D):
            return _sin_interseccion(r, 'Los planos son paralelos (o coincidentes): no se cortan en una recta.')
        L = _recta_de_planos(p1, p2)
        i = next(k for k in range(3) if not _es_cero(L['D'][k]))
        r.paso('Un punto común', f'Se fija {X3[i]} = 0 y se resuelve el sistema de las dos ecuaciones.', 'P_0 = ' + _tt(L['P']), 0)
        return _informe_recta(r, L['P'], L['D'])
    if tipos == ['plano', 'plano', 'plano']:
        r.titulo = 'Intersección de tres planos'
        A = sp.Matrix([list(o['n']) for o in objs])
        b = sp.Matrix([o['d'] for o in objs])
        r.paso('Sistema', 'Se resuelve el sistema de las tres ecuaciones.', r'\begin{cases}' + r'\\'.join(_tex_plano(o) for o in objs)
               + r'\end{cases}', 0)
        red, pasos = gauss_jordan_pasos(A.row_join(b), aumentada=True)
        r.pasos.extend(pasos)
        sol = sp.linsolve((A, b), *X3)
        if not sol:
            return _sin_interseccion(r, 'El sistema es incompatible: los tres planos no tienen puntos en común.')
        s = list(sol)[0]
        libres = set().union(*[sp.sympify(c).free_symbols for c in s]) & set(X3)
        if not libres:
            Q = sp.Matrix(s)
            r.valor = list(Q)
            r.resultado_latex = 'P = ' + _tt(Q)
            r.comparar = ('vector', list(Q), {})
            r.paso('Solución única', 'Los tres planos se cortan en un punto.', r.resultado_latex, 0)
            return r
        if len(libres) == 1:
            p = libres.pop()
            vec = sp.Matrix(s).subs(p, t)
            return _informe_recta(r, vec.subs(t, 0), _vec_entero(vec.diff(t)))
        r.valor = 'plano'
        r.resultado_latex = r'\text{Los tres planos coinciden}'
        return r
    if tipos == ['plano', 'recta']:
        L, pl = _orden(objs, 'recta', 'plano')
        r.titulo = 'Intersección de recta y plano'
        r.paso('Se reemplazan las paramétricas en el plano', r'\(\vec n\cdot(P_0 + t\,\vec d) = d\)',
               tex(sp.expand(pl['n'].dot(L['P'] + t * L['D'])) ) + ' = ' + tex(pl['d']), 0)
        Q, lam = _interseccion_recta_plano(L, pl)
        if Q is None:
            if _es_cero(pl['n'].dot(L['P']) - pl['d']):
                r.valor = 'recta'
                r.resultado_latex = r'\text{La recta está contenida en el plano}'
                return r
            return _sin_interseccion(r, 'La recta es paralela al plano y no lo toca.')
        r.paso('Parámetro', '', 't = ' + tex(lam), 1)
        r.valor = list(Q)
        r.resultado_latex = 'P = ' + _tt(Q)
        r.comparar = ('vector', list(Q), {})
        return r
    if tipos == ['recta', 'recta']:
        L1, L2 = objs
        r.titulo = 'Intersección de rectas'
        if _paralelos(L1['D'], L2['D']):
            if _paralelos(L2['P'] - L1['P'], L1['D']):
                r.valor = 'coincidentes'
                r.resultado_latex = r'\text{Son la misma recta}'
                return r
            return _sin_interseccion(r, 'Las rectas son paralelas: no se cortan.')
        r.paso('Se igualan las paramétricas', r'\(P_1 + s\,\vec d_1 = P_2 + u\,\vec d_2\)', '', 0)
        Q = _interseccion_rectas(L1, L2)
        if Q is None:
            return _sin_interseccion(r, 'Las rectas se cruzan (alabeadas): no tienen punto común.')
        r.valor = list(Q)
        r.resultado_latex = 'P = ' + _tt(Q)
        r.comparar = ('vector', list(Q), {})
        if L1['dim'] == 2:
            curvas = [h for h in (_explicita(L1['abc']), _explicita(L2['abc'])) if h is not None]
            if len(curvas) == 2:
                r.grafico = G.grafico_2d(curvas, x, [G.punto(Q[0], Q[1], 'P ' + _txt_tupla(Q))])
        return r
    raise ErrorEntrada('Uso: interseccion de planos π1, π2[, π3] · interseccion de rectas r1, r2 · interseccion recta plano r, π')


def _sin_interseccion(r, texto):
    r.paso('Conclusión', texto, '', 0)
    r.valor = 'vacio'
    r.resultado_latex = r'\varnothing'
    r.comparar = ('texto', 'no existe', {})
    return r


# ---------------------------------------------------------------------------
# Cónicas
# ---------------------------------------------------------------------------
def _leer_F(arg, vars_):
    s = _limpiar(normalizar(arg))
    eq = leer_ecuacion(s) if '=' in s else None
    if eq is not None and not isinstance(eq, sp.Equality):
        raise ErrorEntrada('Escribí una ecuación, por ejemplo x^2+4y^2-2x-3=0.')
    F = sp.expand(eq.lhs - eq.rhs) if eq is not None else sp.expand(leer(s))
    ent = (tex(eq.lhs) + ' = ' + tex(eq.rhs)) if eq is not None else tex(F) + ' = 0'
    if not F.free_symbols <= set(vars_):
        raise ErrorEntrada('Usá solo las variables ' + ', '.join(str(v) for v in vars_) + '.')
    try:
        P = sp.Poly(F, *vars_)
    except sp.PolynomialError:
        raise ErrorEntrada('La ecuación debe ser polinómica de segundo grado.')
    if P.total_degree() != 2:
        raise ErrorEntrada('La ecuación debe ser de segundo grado (aparecer x², y², xy...).')
    return F, P, ent


def _rot(u):
    return sp.Matrix([-u[1], u[0]])


def _unit(v):
    n_ = _norma(v)
    return sp.Matrix([_simp(c / n_) for c in v])


def _signo(e):
    v = sp.N(e, 30)
    if abs(v) < 1e-20:
        return 0
    return 1 if v > 0 else -1


def conica(arg):
    F, P, ent = _leer_F(arg, (x, y))
    A, B, C = P.coeff_monomial(x ** 2), P.coeff_monomial(x * y), P.coeff_monomial(y ** 2)
    D, E, F0 = P.coeff_monomial(x), P.coeff_monomial(y), P.coeff_monomial(1)
    r = Resultado(TEMA_GEO, 'Cónica')
    r.entrada_latex = ent
    r.paso('Coeficientes', r'Forma general \(Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0\).',
           f'A = {tex(A)},\\ B = {tex(B)},\\ C = {tex(C)},\\ D = {tex(D)},\\ E = {tex(E)},\\ F = {tex(F0)}', 0)
    disc = _simp(B ** 2 - 4 * A * C)
    gen = 'elíptico' if _signo(disc) < 0 else ('parabólico' if _signo(disc) == 0 else 'hiperbólico')
    r.paso('Discriminante', r'\(B^2 - 4AC < 0\): tipo elipse · \(= 0\): tipo parábola · \(> 0\): tipo hipérbola.',
           r'B^2 - 4AC = ' + tex(disc) + r'\Rightarrow \text{género ' + gen + '}', 0)
    Q = sp.Matrix([[A, B / 2], [B / 2, C]])
    Lv = sp.Matrix([D, E])
    rotada = B != 0
    if not rotada:
        ejes = [(A, sp.Matrix([1, 0])), (C, sp.Matrix([0, 1]))]
    else:
        pares = []
        for lam, mult, vecs in Q.eigenvects():
            for v in vecs:
                pares.append((_simp(lam), _unit(v)))
        pares.sort(key=lambda p: float(sp.N(p[0])))
        if _signo(pares[0][0]) == 0:
            pares = [pares[1], pares[0]]
        lam1, u1 = pares[0]
        if _signo(u1[0]) < 0 or (_signo(u1[0]) == 0 and _signo(u1[1]) < 0):
            u1 = -u1
        u2 = _rot(u1)
        lam2 = _simp((u2.T * Q * u2)[0])
        ejes = [(lam1, u1), (lam2, u2)]
        th = _simp(sp.atan2(u1[1], u1[0]))
        r.paso('Rotación de ejes', r'Hay término \(xy\): se rotan los ejes según los autovectores de la matriz de la forma '
               r'cuadrática \(\begin{pmatrix} A & B/2 \\ B/2 & C\end{pmatrix}\). Con \(x = x\'\cos\theta - y\'\sin\theta\), '
               r'\(y = x\'\sin\theta + y\'\cos\theta\) desaparece el término cruzado.',
               r'\lambda_1 = ' + tex(lam1) + r',\ \lambda_2 = ' + tex(lam2) + r',\quad \theta = ' + tex(th) + r'\approx '
               + fmt_num(sp.N(th * 180 / sp.pi), 6) + r'^\circ', 0)
    puntos, curvas_extra = [], []
    tipo = None
    escala = 1
    centro_g = (0, 0)
    if not _es_cero(Q.det()):
        c = (-(Q.inv() * Lv) / 2).applyfunc(_simp)
        K = _simp(F0 + (Lv.T * c)[0] / 2)
        R = _simp(-K)
        (l1, u1), (l2, u2) = ejes
        if not rotada:
            r.paso('Completar cuadrados', 'Se agrupan las x y las y y se completa cada cuadrado.',
                   tex(A) + '(' + tex(x - c[0]) + ')^2 + ' + tex(C) + '(' + tex(y - c[1]) + ')^2 = ' + tex(R), 0)
        else:
            r.paso('Traslación al centro', r'El centro resuelve \(\nabla F = 0\); en los ejes rotados queda:',
                   r'C = ' + _tt(c) + r',\quad ' + tex(l1) + r"\,x'^2 + " + tex(l2) + r"\,y'^2 = " + tex(R), 0)
        centro_g = (c[0], c[1])
        r.valores['centro'] = list(c)
        if R == 0:
            if _signo(l1) == _signo(l2):
                tipo = 'punto (cónica degenerada)'
                r.paso('Cónica degenerada', 'Suma de cuadrados igual a 0: solo el centro.', _tt(c), 0)
            else:
                tipo = 'par de rectas secantes (cónica degenerada)'
                m = sp.sqrt(-l1 / l2)
                for sg in (1, -1):
                    ec, imp = _ec_recta_dir(c, u1 + sg * m * u2)
                    curvas_extra.append((imp, ec))
                r.paso('Cónica degenerada', 'Diferencia de cuadrados igual a 0: dos rectas que se cortan en el centro.',
                       r',\quad '.join(e for _, e in curvas_extra), 0)
        else:
            al, be = _simp(R / l1), _simp(R / l2)
            if rotada:
                can = r"\frac{x'^2}{" + tex(al) + r"} + \frac{y'^2}{" + tex(be) + '} = 1'
            else:
                can = r'\frac{' + tex((x - c[0]) ** 2) + '}{' + tex(al) + r'} + \frac{' + tex((y - c[1]) ** 2) + '}{' + tex(be) + '} = 1'
            if _signo(al) > 0 and _signo(be) > 0:
                if _es_cero(al - be):
                    tipo = 'circunferencia'
                    rad = _simp(sp.sqrt(al))
                    r.paso('Forma canónica', 'Se divide por el término independiente.',
                           tex((x - c[0]) ** 2) + ' + ' + tex((y - c[1]) ** 2) + ' = ' + tex(al), 0)
                    r.paso('Elementos', '', r'C = ' + _tt(c) + r',\quad r = ' + tex(rad), 0)
                    r.valores.update({'radio': rad, 'a': rad, 'b': rad, 'c': 0, 'excentricidad': 0})
                    puntos.append(G.punto(c[0], c[1], 'C ' + _txt_tupla(c)))
                    escala = rad
                else:
                    tipo = 'elipse'
                    if _signo(al - be) >= 0:
                        a2, b2, um, un = al, be, u1, u2
                    else:
                        a2, b2, um, un = be, al, u2, u1
                    a, b = _simp(sp.sqrt(a2)), _simp(sp.sqrt(b2))
                    cf = _simp(sp.sqrt(a2 - b2))
                    e = _simp(cf / a)
                    focos = [(c + s_ * cf * um).applyfunc(_simp) for s_ in (-1, 1)]
                    verts = [(c + s_ * a * um).applyfunc(_simp) for s_ in (-1, 1)]
                    covs = [(c + s_ * b * un).applyfunc(_simp) for s_ in (-1, 1)]
                    r.paso('Forma canónica', 'Se divide por el término independiente.', can, 0)
                    r.paso('Semiejes', r'\(a\) es el semieje mayor, \(b\) el menor y \(c^2 = a^2 - b^2\).',
                           f'a = {tex(a)},\\ b = {tex(b)},\\ c = \\sqrt{{{tex(a2)} - {tex(b2)}}} = {tex(cf)}', 0)
                    eje_f, _ = _ec_recta_dir(c, um)
                    eje_s, _ = _ec_recta_dir(c, un)
                    r.paso('Elementos', '', r'C = ' + _tt(c) + r',\quad F_{1,2} = ' + _tt(focos[0]) + ',\\ ' + _tt(focos[1])
                           + r',\quad V_{1,2} = ' + _tt(verts[0]) + ',\\ ' + _tt(verts[1]), 0)
                    r.paso('Covértices, ejes y excentricidad', '', r'B_{1,2} = ' + _tt(covs[0]) + ',\\ ' + _tt(covs[1])
                           + r',\quad \text{eje focal: } ' + eje_f + r',\ \text{eje menor: } ' + eje_s
                           + r',\quad e = \frac{c}{a} = ' + tex(e), 0)
                    r.valores.update({'a': a, 'b': b, 'c': cf, 'excentricidad': e, 'focos': [list(f) for f in focos],
                                      'vertices': [list(v) for v in verts], 'covertices': [list(v) for v in covs],
                                      'eje_focal': eje_f, 'eje_menor': eje_s})
                    r.extra('Lado recto', r'\frac{2b^2}{a} = ' + tex(_simp(2 * b2 / a)))
                    puntos += [G.punto(c[0], c[1], 'C')] + [G.punto(f[0], f[1], f'F{i + 1}', '#dc2626') for i, f in enumerate(focos)]
                    puntos += [G.punto(v[0], v[1], f'V{i + 1}', '#059669') for i, v in enumerate(verts)]
                    escala = a
            elif _signo(al) < 0 and _signo(be) < 0:
                tipo = 'vacío (elipse imaginaria)'
                r.paso('Sin puntos reales', 'Una suma de cuadrados no puede dar un número negativo.', can, 0)
            else:
                tipo = 'hipérbola'
                if _signo(al) > 0:
                    a2, b2, ut, uc = al, -be, u1, u2
                else:
                    a2, b2, ut, uc = be, -al, u2, u1
                a, b = _simp(sp.sqrt(a2)), _simp(sp.sqrt(b2))
                cf = _simp(sp.sqrt(a2 + b2))
                e = _simp(cf / a)
                if rotada:
                    can = (r"\frac{x'^2}{" + tex(a2) + r"} - \frac{y'^2}{" + tex(b2) + '} = 1') if ut is u1 else \
                          (r"\frac{y'^2}{" + tex(a2) + r"} - \frac{x'^2}{" + tex(b2) + '} = 1')
                else:
                    X_, Y_ = (x - c[0], y - c[1]) if ut is u1 else (y - c[1], x - c[0])
                    can = r'\frac{' + tex(X_ ** 2) + '}{' + tex(a2) + r'} - \frac{' + tex(Y_ ** 2) + '}{' + tex(b2) + '} = 1'
                focos = [(c + s_ * cf * ut).applyfunc(_simp) for s_ in (-1, 1)]
                verts = [(c + s_ * a * ut).applyfunc(_simp) for s_ in (-1, 1)]
                r.paso('Forma canónica', 'Se divide por el término independiente; el término positivo indica el eje focal.', can, 0)
                r.paso('Semiejes', r'\(c^2 = a^2 + b^2\).', f'a = {tex(a)},\\ b = {tex(b)},\\ c = {tex(cf)}', 0)
                asin = []
                for sg in (1, -1):
                    ec, imp = _ec_recta_dir(c, a * ut + sg * b * uc)
                    asin.append(ec)
                    curvas_extra.append((imp, ec))
                eje_f, _ = _ec_recta_dir(c, ut)
                r.paso('Elementos', '', r'C = ' + _tt(c) + r',\quad F_{1,2} = ' + _tt(focos[0]) + ',\\ ' + _tt(focos[1])
                       + r',\quad V_{1,2} = ' + _tt(verts[0]) + ',\\ ' + _tt(verts[1]), 0)
                r.paso('Asíntotas, eje focal y excentricidad', '', r'\text{asíntotas: } ' + r',\ '.join(asin)
                       + r',\quad \text{eje focal: } ' + eje_f + r',\quad e = \frac{c}{a} = ' + tex(e), 0)
                if _es_cero(a - b):
                    r.extra('Hipérbola equilátera', '', 'a = b: las asíntotas son perpendiculares.')
                r.valores.update({'a': a, 'b': b, 'c': cf, 'excentricidad': e, 'focos': [list(f) for f in focos],
                                  'vertices': [list(v) for v in verts], 'asintotas': asin, 'eje_focal': eje_f})
                puntos += [G.punto(c[0], c[1], 'C')] + [G.punto(f[0], f[1], f'F{i + 1}', '#dc2626') for i, f in enumerate(focos)]
                puntos += [G.punto(v[0], v[1], f'V{i + 1}', '#059669') for i, v in enumerate(verts)]
                escala = cf
    else:
        (l1, u1), (l2, u2) = ejes
        if _signo(l1) == 0:
            (l1, u1), (l2, u2) = (l2, u2), (l1, u1)
        if not rotada:
            u2 = sp.Matrix([1, 0]) if u1[1] == 1 else sp.Matrix([0, 1])
        L1, L2 = _simp(Lv.dot(u1)), _simp(Lv.dot(u2))
        X0 = _simp(-L1 / (2 * l1))
        if _es_cero(L2):
            disc2 = _simp(L1 ** 2 - 4 * l1 * F0)
            vv = 'x' if u1[0] != 0 and not rotada else ('y' if not rotada else "x'")
            if _signo(disc2) > 0:
                tipo = 'par de rectas paralelas (cónica degenerada)'
            elif _signo(disc2) == 0:
                tipo = 'recta doble (cónica degenerada)'
            else:
                tipo = 'vacío (sin puntos reales)'
            r.paso('Cónica degenerada', f'Solo aparece la variable {vv}: la ecuación de segundo grado en {vv} da '
                   'dos, una o ninguna recta.', r'\Delta = ' + tex(disc2), 0)
        else:
            tipo = 'parábola'
            Y0 = _simp((L1 ** 2 / (4 * l1) - F0) / L2)
            p = _simp(-L2 / (4 * l1))
            V = (X0 * u1 + Y0 * u2).applyfunc(_simp)
            Fo = (V + p * u2).applyfunc(_simp)
            dirp = (V - p * u2)
            dir_ec, dir_imp = _ec_recta_dir(dirp, u1)
            eje_ec, _ = _ec_recta_dir(V, u2)
            if not rotada:
                vX, vY = (x, y) if u1[0] == 1 else (y, x)
                r.paso('Completar cuadrados', f'Se completa el cuadrado en {vX} y se despeja el término lineal en {vY}.',
                       tex((vX - X0) ** 2) + ' = ' + tex(4 * p) + r'\,(' + tex(vY - Y0) + ')', 0)
            else:
                r.paso('Forma canónica en ejes rotados', '', r"(x' - " + tex(X0) + r")^2 = " + tex(4 * p) + r"\,(y' - " + tex(Y0) + ')', 0)
            r.paso('Parámetro', r'Comparando con \((X - h)^2 = 4p\,(Y - k)\).', r'4p = ' + tex(4 * p) + r'\Rightarrow p = ' + tex(p), 0)
            r.paso('Elementos', 'El foco está a distancia |p| del vértice, hacia donde abre; la directriz, del otro lado.',
                   r'V = ' + _tt(V) + r',\quad F = ' + _tt(Fo) + r',\quad \text{directriz: } ' + dir_ec
                   + r',\quad \text{eje: } ' + eje_ec, 0)
            r.valores.update({'vertice': list(V), 'foco': list(Fo), 'directriz': dir_ec, 'eje': eje_ec, 'p': p,
                              'excentricidad': 1})
            r.extra('Lado recto', '|4p| = ' + tex(sp.Abs(4 * p)))
            puntos += [G.punto(V[0], V[1], 'V ' + _txt_tupla(V), '#059669'), G.punto(Fo[0], Fo[1], 'F ' + _txt_tupla(Fo), '#dc2626')]
            curvas_extra.append((dir_imp, dir_ec))
            centro_g = (V[0], V[1])
            escala = max(float(sp.N(sp.Abs(p))) * 4, 2)
    r.valores['tipo'] = tipo
    r.valor = tipo
    r.resultado_latex = r'\text{' + tipo + '}'
    g = G.grafico_implicita(F, (x, y), ent)
    for i, (imp, et) in enumerate(curvas_extra, 1):
        cc = G.grafico_implicita(imp, (x, y), et)['curvas'][0]
        cc['color'] = G.COLORES[i % len(G.COLORES)]
        g['curvas'].append(cc)
    g['puntos'] = [p for p in puntos if p]
    try:
        cx, cy, R_ = float(sp.N(centro_g[0])), float(sp.N(centro_g[1])), float(sp.N(escala)) * 1.6 + 1
        g['ventana'] = {'xmin': cx - R_, 'xmax': cx + R_, 'ymin': cy - R_, 'ymax': cy + R_}
    except Exception:  # noqa
        pass
    r.grafico = g
    _propio(r, tipo, _chk_conica(r.valores))
    return r


def _chk_conica(vals):
    tipo = sin_tildes(vals.get('tipo', ''))
    claves = {'circunferencia': ['circunf', 'circulo'], 'elipse': ['elipse'], 'hiperbola': ['hiperbola'],
              'parabola': ['parabola']}

    def f(texto):
        t_ = sin_tildes(texto.lower())
        for nombre, pals in claves.items():
            if any(p in t_ for p in pals):
                return _ok('¡Correcto!') if tipo.startswith(nombre) else _mal('El tipo de cónica no es ese.', vals.get('tipo', ''))
        partes = dividir(normalizar(texto).strip().strip('{}[]'), ',') if not texto.strip().startswith('(') or \
            len(dividir(texto)) > 1 else [texto]
        valores = [leer(p) for p in partes if p.strip()]
        if not valores:
            return {'correcto': None, 'mensaje': 'Escribí el tipo, un punto (centro, foco, vértice) o un número (a, b, c, e).'}
        if all(isinstance(v, (sp.Tuple, tuple, list)) for v in valores):
            alumno = [[sp.sympify(c) for c in v] for v in valores]
            candidatos = {k_: v for k_, v in vals.items() if k_ in ('centro', 'focos', 'vertices', 'covertices', 'foco', 'vertice')}
            for nombre, ref in candidatos.items():
                ref_l = ref if ref and isinstance(ref[0], list) else [ref]
                if len(ref_l) == len(alumno) and all(any(len(a_) == len(b_) and all(_es_cero(p - q) for p, q in zip(a_, b_))
                                                         for b_ in ref_l) for a_ in alumno):
                    return _ok(f'¡Correcto! ({nombre}).')
            return _mal('Esos puntos no coinciden con el centro, los focos ni los vértices.')
        if len(valores) == 1 and valores[0].is_number:
            for nombre in ('a', 'b', 'c', 'excentricidad', 'radio', 'p'):
                if nombre in vals and _es_cero(valores[0] - sp.sympify(vals[nombre])):
                    return _ok(f'¡Correcto! ({nombre} = {tex(vals[nombre])}).')
            return _mal('Ese número no coincide con a, b, c, la excentricidad ni p.')
        return {'correcto': None, 'mensaje': 'No pude comparar tu respuesta con los elementos de la cónica.'}
    return f


# ---------------------------------------------------------------------------
# Cuádricas
# ---------------------------------------------------------------------------
def _signos_autovalores(Q):
    """(positivos, negativos) contados exactamente con la regla de Descartes (raíces reales)."""
    lam = sp.Symbol('lamda')
    p = sp.Poly((lam * sp.eye(3) - Q).det(), lam)
    while p.degree() > 0 and p.eval(0) == 0:
        p = sp.Poly(sp.cancel(p.as_expr() / lam), lam)

    def cambios(pol):
        cs = [c for c in pol.all_coeffs() if c != 0]
        return sum(1 for a, b in zip(cs, cs[1:]) if _signo(a) * _signo(b) < 0)
    pos = cambios(p)
    neg = cambios(sp.Poly(p.as_expr().subs(lam, -lam), lam))
    return pos, neg


def cuadrica(arg):
    F, P, ent = _leer_F(arg, X3)
    co = {m: P.coeff_monomial(m) for m in (x ** 2, y ** 2, z ** 2, x * y, x * z, y * z, x, y, z, 1)}
    Q = sp.Matrix([[co[x ** 2], co[x * y] / 2, co[x * z] / 2], [co[x * y] / 2, co[y ** 2], co[y * z] / 2],
                   [co[x * z] / 2, co[y * z] / 2, co[z ** 2]]])
    Lv = sp.Matrix([co[x], co[y], co[z]])
    K0 = co[1]
    r = Resultado(TEMA_GEO, 'Superficie cuádrica')
    r.entrada_latex = ent
    r.paso('Forma matricial', r'\(F = X^T Q X + L\cdot X + K\), con Q la matriz de la parte cuadrática.',
           'Q = ' + tex(Q) + r',\quad L = ' + _tt(Lv) + r',\quad K = ' + tex(K0), 0)
    rq = Q.rank(simplify=True)
    pos, neg = _signos_autovalores(Q)
    r.paso('Autovalores de Q', 'Se cuentan los autovalores positivos y negativos (la forma cuadrática '
           'diagonalizada).', r'\text{rango}(Q) = ' + str(rq) + r',\ \text{positivos: } ' + str(pos)
           + r',\ \text{negativos: } ' + str(neg), 0)
    diagonal = all(Q[i, j] == 0 for i in range(3) for j in range(3) if i != j)
    if diagonal:
        terminos, lineal_txt, K = [], [], K0
        for i, v in enumerate(X3):
            a = Q[i, i]
            if a != 0:
                h = -Lv[i] / (2 * a)
                K -= a * h ** 2
                terminos.append(tex(a) + '(' + tex(v - h) + ')^2')
            elif Lv[i] != 0:
                lineal_txt.append(tex(Lv[i] * v))
        expr = ' + '.join(terminos + lineal_txt).replace('+ -', '- ')
        r.paso('Completar cuadrados', '', expr + (' + ' + tex(K) if K != 0 else '') + ' = 0', 0)
    tipo = None
    if rq == 3:
        c = (-(Q.inv() * Lv) / 2).applyfunc(_simp)
        K = _simp(K0 + (Lv.T * c)[0] / 2)
        r.paso('Centro', r'Se resuelve \(2QX + L = 0\) y se evalúa F en el centro.', 'C = ' + _tt(c) + r',\quad F(C) = ' + tex(K), 0)
        sk = _signo(K)
        if sk == 0:
            tipo = 'punto (cuádrica degenerada)' if (pos == 3 or neg == 3) else 'cono'
        else:
            # denominadores -K/λ: positivos = autovalores con signo opuesto a K
            npos = neg if sk > 0 else pos
            tipo = {3: 'elipsoide', 2: 'hiperboloide de una hoja', 1: 'hiperboloide de dos hojas',
                    0: 'vacío (elipsoide imaginario)'}[npos]
            if tipo == 'elipsoide' and Q == Q[0, 0] * sp.eye(3):
                tipo = 'esfera'
                r.extra('Radio', tex(_simp(sp.sqrt(-K / Q[0, 0]))))
        r.valores['centro'] = list(c)
    elif rq == 2:
        u3 = Q.nullspace()[0]
        L3 = _simp(Lv.dot(u3))
        if L3 != 0:
            tipo = 'paraboloide elíptico' if (pos == 2 or neg == 2) else 'paraboloide hiperbólico (silla de montar)'
            r.paso('Término lineal', 'Hay un término lineal en la dirección sin término cuadrático: paraboloide.', '', 0)
        else:
            sol = sp.linsolve((2 * Q, -Lv), *X3)
            c = sp.Matrix(list(list(sol)[0])).subs({v: 0 for v in X3})
            K = _simp(K0 + (Lv.T * c)[0] / 2)
            sk = _signo(K)
            if sk == 0:
                tipo = 'recta (cuádrica degenerada)' if (pos == 2 or neg == 2) else 'par de planos secantes'
            else:
                npos = neg if sk > 0 else pos
                tipo = {2: 'cilindro elíptico', 1: 'cilindro hiperbólico', 0: 'vacío (cilindro imaginario)'}[npos]
                if tipo == 'cilindro elíptico':
                    ev = [e for e in Q.eigenvals() if e != 0]
                    if len(ev) == 1:
                        tipo = 'cilindro circular'
    elif rq == 1:
        ev = [e for e in Q.eigenvals() if e != 0][0]
        u1 = Q.columnspace()[0]
        u1 = u1 / _norma(u1)
        perp = (Lv - Lv.dot(u1) * u1).applyfunc(_simp)
        if any(c != 0 for c in perp):
            tipo = 'cilindro parabólico'
        else:
            l1 = _simp(Lv.dot(u1))
            disc = _simp(l1 ** 2 - 4 * ev * K0)
            tipo = 'par de planos paralelos' if _signo(disc) > 0 else ('plano doble' if _signo(disc) == 0 else 'vacío')
    r.paso('Clasificación', '', r'\text{' + tipo + '}', 0)
    r.valor = tipo
    r.valores['tipo'] = tipo
    r.resultado_latex = r'\text{' + tipo + '}'
    if sp.degree(F, z) == 1:
        zf = sp.solve(F, z)[0]
        r.grafico = G.grafico_3d(zf, (x, y), 'z = ' + tex(zf))
    claves = {'elipsoide': ['elipsoide'], 'esfera': ['esfera'], 'hiperboloide de una hoja': ['una hoja', '1 hoja'],
              'hiperboloide de dos hojas': ['dos hojas', '2 hojas'], 'cono': ['cono'],
              'paraboloide eliptico': ['paraboloide elip', 'paraboloide circ'],
              'paraboloide hiperbolico': ['paraboloide hip', 'silla'], 'cilindro': ['cilindro']}

    def chk(texto):
        t_ = sin_tildes(texto.lower())
        ref = sin_tildes(tipo)
        for nombre, pals in claves.items():
            if any(p in t_ for p in pals):
                ok = ref.startswith(nombre) or (nombre == 'cilindro' and ref.startswith('cilindro') and
                                                 all(w in ref for w in t_.split() if w in ('eliptico', 'hiperbolico', 'parabolico', 'circular')))
                return _ok() if ok else _mal('No es ese tipo de cuádrica.', tipo)
        return {'correcto': None, 'mensaje': 'Respondé con el nombre de la superficie (elipsoide, cono, paraboloide...).'}
    _propio(r, tipo, chk)
    return r


# ============================================================================
# 2. ÁLGEBRA LINEAL
# ============================================================================
def producto_mixto(arg):
    partes = _partes(arg)
    if len(partes) != 3:
        raise ErrorEntrada('Uso: producto mixto u, v, w   — tres vectores de R³, ej.: producto mixto [1,2,3], [4,5,6], [7,8,10]')
    u, v, w = (sp.Matrix(leer_lista(p)) for p in partes)
    if not all(len(a) == 3 for a in (u, v, w)):
        raise ErrorEntrada('El producto mixto se calcula con tres vectores de R³.')
    r = Resultado('Vectores', 'Producto mixto')
    r.entrada_latex = r'\vec u\cdot(\vec v\times\vec w),\quad \vec u = ' + _tt(u) + r',\ \vec v = ' + _tt(v) + r',\ \vec w = ' + _tt(w)
    M = sp.Matrix([list(u), list(v), list(w)])
    cr = v.cross(w)
    val = _simp(u.dot(cr))
    r.paso('Como determinante', r'\([\vec u, \vec v, \vec w] = \det\) de la matriz con los vectores como filas.',
           r'\begin{vmatrix}' + r'\\'.join(' & '.join(tex(c) for c in fila) for fila in M.tolist()) + r'\end{vmatrix} = ' + tex(val), 0)
    r.paso('Verificación', r'Primero \(\vec v\times\vec w\), luego producto escalar con \(\vec u\).',
           r'\vec v\times\vec w = ' + _tt(cr) + r',\quad \vec u\cdot(\vec v\times\vec w) = '
           + ' + '.join(f'({tex(a)})({tex(b)})' for a, b in zip(u, cr)) + ' = ' + tex(val), 0)
    r.valor = val
    r.resultado_latex = r'[\vec u, \vec v, \vec w] = ' + tex(val)
    r.extra('Volumen del paralelepípedo', r'V = |[\vec u, \vec v, \vec w]| = ' + tex(sp.Abs(val)))
    r.extra('Volumen del tetraedro', r'V = \frac{|[\vec u, \vec v, \vec w]|}{6} = ' + tex(sp.Abs(val) / 6))
    r.extra('Coplanaridad', '', 'Los vectores son coplanares (linealmente dependientes).' if val == 0 else
            'Los vectores no son coplanares (forman una base de R³).')
    r.comparar = ('numero_o_expr', val, {})
    return r


def _matriz_de(arg):
    """Matriz [[..]] o transformación T(x,y)=(x+2y, 2x+4y) -> (A, variables o None)"""
    s = normalizar(arg).strip()
    if s.startswith('[['):
        return leer_matriz(s), None
    m = re.match(r'^\s*[A-Za-z]?\s*\(([^()]*)\)\s*=\s*(.+)$', s)
    if m:
        vars_ = [leer(v) for v in dividir(m.group(1))]
        rhs = m.group(2)
    elif s.startswith('(') or s.startswith('['):
        vars_, rhs = None, s
    else:
        raise ErrorEntrada('Escribí la matriz por filas, [[1,2],[2,4]], o la transformación T(x,y) = (x+2y, 2x+4y).')
    comps = leer_lista(leer(rhs))
    if vars_ is None:
        vars_ = variables_de(sp.Tuple(*comps)) or [x]
    A = sp.Matrix([[sp.diff(c, v) for v in vars_] for c in comps])
    for c, fila in zip(comps, A.tolist()):
        if not _es_cero(c - sum(a * v for a, v in zip(fila, vars_))) or any(a.free_symbols & set(vars_) for a in fila):
            raise ErrorEntrada('La transformación no es lineal (o tiene término independiente).')
    return A, vars_


def _nombres_vars(n_, vars_):
    if vars_:
        return list(vars_)
    if n_ <= 3:
        return list(X3[:n_])
    return list(sp.symbols(f'x1:{n_ + 1}'))


def _resolver_homogeneo(r, A, vars_):
    """Gauss sobre A v = 0, devuelve la base del núcleo (vectores enteros)."""
    n_ = A.shape[1]
    vs = _nombres_vars(n_, vars_)
    r.paso('Sistema homogéneo', r'\(A\,v = 0\)', r'\begin{cases}' + r'\\'.join(
        tex(sum(A[i, j] * vs[j] for j in range(n_))) + ' = 0' for i in range(A.shape[0])) + r'\end{cases}', 0)
    red, pasos = gauss_jordan_pasos(A)
    r.pasos.extend(pasos)
    rref, piv = A.rref()
    libres = [j for j in range(n_) if j not in piv]
    if libres:
        ecs = []
        for i, pj in enumerate(piv):
            ecs.append(tex(vs[pj]) + ' = ' + tex(-sum(rref[i, j] * vs[j] for j in libres)))
        r.paso('Variables libres', 'Las columnas sin pivote dan variables libres; se despejan las principales.',
               r',\ '.join(ecs) + (r',\quad ' if ecs else '') + r'\text{libres: } ' + ', '.join(tex(vs[j]) for j in libres), 0)
    base = [_vec_entero(b) for b in A.nullspace()]
    return base, vs


def _gen(base):
    if not base:
        return r'\{\vec 0\}'
    return r'\operatorname{gen}\left\{' + ',\\ '.join(_tt(b) for b in base) + r'\right\}'


def nucleo(arg):
    A, vars_ = _matriz_de(arg)
    r = Resultado(TEMA_LIN, 'Núcleo')
    r.entrada_latex = r'\operatorname{Nu}(A),\quad A = ' + tex(A)
    base, vs = _resolver_homogeneo(r, A, vars_)
    dim = len(base)
    r.paso('Base y dimensión', 'Un vector por cada variable libre (se da valor 1 a una y 0 a las demás).',
           r'\operatorname{Nu}(A) = ' + _gen(base) + r',\quad \dim = ' + str(dim), 0)
    r.valor = [list(b) for b in base]
    r.valores['dimension'] = dim
    r.resultado_latex = r'\dim = ' + str(dim) + r',\quad \operatorname{Nu}(A) = ' + _gen(base)
    rg = A.rank(simplify=True)
    r.extra('Teorema de la dimensión', f'{A.shape[1]} = \\dim \\operatorname{{Nu}} + \\operatorname{{rg}} A = {dim} + {rg}')
    r.extra('Inyectividad', '', 'La transformación es inyectiva (núcleo nulo).' if dim == 0 else 'No es inyectiva (núcleo no nulo).')
    _propio(r, r.valor, _chk_base(base, lambda v: all(_es_cero(c) for c in A * v), A.shape[1]))
    return r


def imagen(arg):
    A, vars_ = _matriz_de(arg)
    r = Resultado(TEMA_LIN, 'Imagen')
    r.entrada_latex = r'\operatorname{Im}(A),\quad A = ' + tex(A)
    r.paso('Generadores', 'La imagen está generada por las columnas de A.', ',\\ '.join(_tt(A[:, j]) for j in range(A.shape[1])), 0)
    red, pasos = gauss_jordan_pasos(A)
    r.pasos.extend(pasos)
    _, piv = A.rref()
    base = [_vec_entero(A[:, j]) for j in piv]
    r.paso('Columnas pivote', 'Las columnas de A (originales) donde hay pivotes forman una base.',
           r'\text{pivotes en las columnas } ' + ', '.join(str(j + 1) for j in piv), 0)
    dim = len(base)
    r.paso('Base y dimensión', '', r'\operatorname{Im}(A) = ' + _gen(base) + r',\quad \dim = \operatorname{rg} A = ' + str(dim), 0)
    m_ = A.shape[0]
    normales = [_vec_entero(v) for v in A.T.nullspace()]
    if normales and dim > 0:
        vs = _nombres_vars(m_, None)
        r.extra('Ecuaciones implícitas de la imagen', r',\ '.join(tex(sum(c * v for c, v in zip(nv, vs))) + ' = 0' for nv in normales))
    r.extra('Sobreyectividad', '', 'Es sobreyectiva (la imagen es todo el codominio).' if dim == m_ else 'No es sobreyectiva.')
    r.valor = [list(b) for b in base]
    r.valores['dimension'] = dim
    r.resultado_latex = r'\dim = ' + str(dim) + r',\quad \operatorname{Im}(A) = ' + _gen(base)
    rg = A.rank(simplify=True)
    _propio(r, r.valor, _chk_base(base, lambda v: A.row_join(sp.Matrix(v)).rank(simplify=True) == rg, m_))
    return r


def _a_vectores(partes):
    """Vectores, polinomios o matrices -> (lista de columnas, tipo, función para mostrar un vector)."""
    objs = [leer(p) for p in partes]
    if all(isinstance(o, (list, tuple, sp.Tuple, sp.MatrixBase)) for o in objs):
        if all(isinstance(o, list) and o and isinstance(o[0], list) for o in objs) or all(isinstance(o, sp.MatrixBase) and o.shape[1] > 1 for o in objs):
            ms = [sp.Matrix(o) for o in objs]
            forma = ms[0].shape
            if any(m.shape != forma for m in ms):
                raise ErrorEntrada('Todas las matrices deben tener el mismo tamaño.')
            return [sp.Matrix(list(m)) for m in ms], 'matrices', lambda v: tex(sp.Matrix(forma[0], forma[1], list(v)))
        vs = [sp.Matrix(list(o)) for o in objs]
        if len({len(v) for v in vs}) != 1:
            raise ErrorEntrada('Todos los vectores deben tener la misma cantidad de componentes.')
        return vs, 'vectores', _tt
    var = variables_de(sp.Tuple(*objs))
    if len(var) > 1:
        raise ErrorEntrada('Escribí vectores entre corchetes [1,2,3] o polinomios en una sola variable.')
    v = var[0] if var else x
    if not all(sp.sympify(o).is_polynomial(v) for o in objs):
        raise ErrorEntrada('Escribí vectores entre corchetes, por ejemplo [1, 2, 3], o polinomios.')
    grado = max(sp.degree(o, v) if o != 0 else 0 for o in objs)
    vecs = [sp.Matrix([sp.Poly(o, v).coeff_monomial(v ** k) for k in range(grado + 1)]) for o in objs]
    return vecs, 'polinomios', lambda c: tex(sum(c[k] * v ** k for k in range(len(c))))


def independencia(arg):
    partes = _partes(arg)
    if len(partes) < 1:
        raise ErrorEntrada('Uso: independencia [1,2,3], [4,5,6], [7,8,9]')
    vecs, clase, mostrar = _a_vectores(partes)
    k_ = len(vecs)
    M = sp.Matrix.hstack(*vecs)
    r = Resultado(TEMA_LIN, 'Independencia lineal')
    r.entrada_latex = r'\left\{' + ',\\ '.join(mostrar(v) for v in vecs) + r'\right\}'
    if clase != 'vectores':
        r.paso('Coordenadas', f'Se trabaja con las coordenadas de los {clase} en la base canónica.',
               ',\\ '.join(_tt(v) for v in vecs), 0)
    cs = sp.symbols(f'c1:{k_ + 1}')
    r.paso('Combinación lineal nula', r'Son L.I. si \(c_1 v_1 + \dots + c_k v_k = 0\) solo tiene la solución trivial.',
           ' + '.join(f'{tex(c)}\\,{_tt(v)}' for c, v in zip(cs, vecs)) + r' = \vec 0', 0)
    r.paso('Matriz con los vectores como columnas', 'Se escalona: el rango es la cantidad de vectores independientes.', tex(M), 0)
    red, pasos = gauss_jordan_pasos(M)
    r.pasos.extend(pasos)
    rg = M.rank(simplify=True)
    if rg == k_:
        res = 'independientes'
        r.paso('Conclusión', f'rango = {rg} = cantidad de vectores: solo la solución trivial.', r'\text{Son linealmente independientes}', 0)
    else:
        res = 'dependientes'
        coef = _vec_entero(M.nullspace()[0])
        j = max(i for i in range(k_) if coef[i] != 0)
        comb = sum(_simp(-coef[i] / coef[j]) * sp.Symbol(f'v_{i + 1}') for i in range(k_) if i != j)
        r.paso('Conclusión', f'rango = {rg} < {k_}: hay soluciones no triviales.',
               ' + '.join(f'({tex(coef[i])})\\,v_{{{i + 1}}}' for i in range(k_) if coef[i] != 0) + r' = \vec 0', 0)
        r.paso('Un vector como combinación de los otros', '', f'v_{{{j + 1}}} = ' + tex(comb), 1)
        r.valores['combinacion'] = [str(c) for c in coef]
        if k_ > len(vecs[0]):
            r.extra('Observación', '', f'Hay más vectores ({k_}) que la dimensión del espacio ({len(vecs[0])}): '
                    'siempre son dependientes.')
    r.valor = res
    r.valores['rango'] = rg
    r.resultado_latex = r'\text{Linealmente ' + res + r'}\ (\operatorname{rg} = ' + str(rg) + ')'
    _propio(r, res, _chk_palabras(res, {'independientes': [r'independ', r'\bl\.?i\.?\b'],
                                        'dependientes': [r'dependient', r'\bl\.?d\.?\b']}, 'linealmente ' + res))
    return r


def _subespacio_ecuaciones(texto):
    s = normalizar(texto)
    n_fijo = None
    m = re.search(r'(?:en\s*)?R\s*\^?\s*(\d)', s)
    if m:
        n_fijo = int(m.group(1))
        s = s[:m.start()] + s[m.end():]
    m = re.match(r'^\s*\{?\s*\(([^)]*)\)\s*(?:∈|en)?\s*[/|:]\s*(.+?)\}?\s*$', s)
    vars_ = None
    if m:
        vars_ = [leer(v) for v in dividir(m.group(1))]
        s = m.group(2)
    s = s.strip().strip('{}')
    partes = dividir(s, ';')
    if len(partes) == 1:
        partes = dividir(s, ',')
    partes = [re.sub(r'^\s*(?:y|e)\s+', '', p) for p in partes]
    ecs = []
    for p in partes:
        eq = leer_ecuacion(p)
        if not isinstance(eq, sp.Equality):
            raise ErrorEntrada(f'«{p}» debería ser una ecuación.')
        ecs.append(sp.expand(eq.lhs - eq.rhs))
    if vars_ is None:
        usadas = variables_de(sp.Tuple(*ecs))
        if set(usadas) <= {x, y, z}:
            n_ = n_fijo or max(X3.index(v) for v in usadas) + 1
            vars_ = list(X3[:n_]) if n_ <= 3 else list(X3) + list(sp.symbols(f'x4:{n_ + 1}'))
        else:
            vars_ = usadas
            if n_fijo and n_fijo > len(vars_):
                raise ErrorEntrada('Nombrá todas las variables, por ejemplo {(x,y,z,w) / x+y=0}.')
    A, b = sp.linear_eq_to_matrix(ecs, vars_)
    return A, b, vars_


def base_dimension(arg):
    r = Resultado(TEMA_LIN, 'Base y dimensión')
    s = normalizar(arg)
    if '=' in s:
        A, b, vars_ = _subespacio_ecuaciones(s)
        r.entrada_latex = r'S = \left\{' + _tt(vars_) + r' :\ ' + r',\ '.join(
            tex(sum(A[i, j] * vars_[j] for j in range(len(vars_)))) + ' = ' + tex(b[i]) for i in range(A.shape[0])) + r'\right\}'
        if any(c != 0 for c in b):
            r.paso('No es subespacio', 'Las ecuaciones no son homogéneas: el vector nulo no las verifica.', '', 0)
            r.valor = 'no es subespacio'
            r.resultado_latex = r'\text{No es subespacio (no contiene al } \vec 0)'
            r.comparar = ('texto', 'no existe', {})
            return r
        base, _ = _resolver_homogeneo(r, A, vars_)
        n_ = len(vars_)
        contiene = (lambda v: all(_es_cero(c) for c in A * v))
        mostrar = _tt
        r.paso('Base y dimensión', r'\(\dim S = n - \operatorname{rg}(A)\).', 'B = ' + _gen(base) + r',\quad \dim S = '
               + f'{n_} - {A.rank()} = {len(base)}', 0)
    else:
        partes = _partes(s)
        vecs, clase, mostrar = _a_vectores(partes)
        M = sp.Matrix.hstack(*vecs)
        r.entrada_latex = r'S = \operatorname{gen}\left\{' + ',\\ '.join(mostrar(v) for v in vecs) + r'\right\}'
        r.paso('Matriz con los generadores como columnas', 'Se escalona para ver qué generadores son independientes.', tex(M), 0)
        red, pasos = gauss_jordan_pasos(M)
        r.pasos.extend(pasos)
        _, piv = M.rref()
        base = [vecs[j] for j in piv]
        n_ = len(vecs[0])
        rg = len(base)
        contiene = (lambda v: M.row_join(sp.Matrix(v)).rank(simplify=True) == rg)
        r.paso('Generadores independientes', 'Los vectores en las columnas pivote forman una base.',
               r'\text{columnas } ' + ', '.join(str(j + 1) for j in piv) + r'\Rightarrow B = \left\{' + ',\\ '.join(mostrar(v) for v in base)
               + r'\right\},\ \dim S = ' + str(rg), 0)
        if len(vecs) > rg:
            r.extra('Generadores sobrantes', '', f'{len(vecs) - rg} generador(es) son combinación lineal de los demás.')
    dim = len(base)
    r.valor = [list(v) for v in base]
    r.valores['dimension'] = dim
    r.resultado_latex = r'\dim = ' + str(dim) + r',\quad B = \left\{' + ',\\ '.join(mostrar(v) for v in base) + r'\right\}' if base else \
        r'S = \{\vec 0\},\ \dim = 0'
    _propio(r, r.valor, _chk_base(base, contiene, n_))
    return r


def diagonalizar(arg):
    A = leer_matriz(arg)
    if not A.is_square:
        raise ErrorEntrada('Solo se diagonalizan matrices cuadradas.')
    n_ = A.shape[0]
    lam = sp.Symbol('lamda')
    r = Resultado(TEMA_LIN, 'Diagonalización')
    r.entrada_latex = 'A = ' + tex(A)
    pc = sp.factor((A - lam * sp.eye(n_)).det())
    r.paso('Polinomio característico', r'\(p(\lambda) = \det(A - \lambda I)\)', 'p(\\lambda) = ' + tex(sp.expand(pc)) + ' = ' + tex(pc), 0)
    datos = []
    for val, mult, base in A.eigenvects():
        datos.append((_simp(val), mult, [_vec_entero(b) for b in base]))
    datos.sort(key=lambda d: (float(sp.re(sp.N(d[0]))), float(sp.im(sp.N(d[0])))))
    r.paso('Autovalores', 'Raíces del polinomio característico (con su multiplicidad algebraica).',
           r',\ '.join(r'\lambda = ' + tex(v) + (f'\\ (m = {m})' if m > 1 else '') for v, m, _ in datos), 0)
    diag = True
    for val, mult, base in datos:
        r.paso(f'Autoespacio de λ = {tex(val)}', r'Se resuelve \((A - \lambda I)v = 0\).',
               r'E_{' + tex(val) + '} = ' + _gen(base) + f',\\ \\text{{mult. algebraica {mult}, geométrica {len(base)}}}', 1)
        if len(base) < mult:
            diag = False
    complejos = any(not sp.sympify(v).is_real for v, _, _ in datos)
    if not diag:
        r.paso('Conclusión', 'Algún autovalor tiene multiplicidad geométrica menor que la algebraica: '
               'no hay una base de autovectores.', r'\text{A no es diagonalizable}', 0)
        r.valor = 'no diagonalizable'
        r.resultado_latex = r'\text{A no es diagonalizable}'
        _propio(r, r.valor, _chk_palabras('no', {'no': [r'\bno\b'], 'si': [r'\bsi\b', r'diagonalizable']},
                                          'no es diagonalizable'))
        return r
    cols, dd = [], []
    for val, mult, base in datos:
        for b in base:
            cols.append(b)
            dd.append(val)
    P = sp.Matrix.hstack(*cols)
    D = sp.diag(*dd)
    Pi = P.inv().applyfunc(_simp)
    r.paso('Matrices P y D', 'P tiene los autovectores como columnas; D los autovalores en el mismo orden.',
           'P = ' + tex(P) + r',\quad D = ' + tex(D), 0)
    r.paso('Verificación', r'\(P^{-1} A P = D\)', 'P^{-1} = ' + tex(Pi) + r',\quad P^{-1}AP = ' + tex((Pi * A * P).applyfunc(_simp)), 0)
    if complejos:
        r.extra('Observación', '', 'Hay autovalores complejos: A es diagonalizable sobre ℂ pero no sobre ℝ.')
    if A == A.T and not complejos:
        qs = []
        for val, mult, base in datos:
            qs += sp.GramSchmidt([sp.Matrix(b) for b in base], True)
        Qm = sp.Matrix.hstack(*qs).applyfunc(_simp)
        r.extra('Diagonalización ortogonal (A simétrica)', r'Q = ' + tex(Qm) + r',\quad Q^T A Q = D')
    r.valor = D
    r.valores['P'] = P
    r.valores['P_inversa'] = Pi
    r.valores['autovalores'] = dd
    r.resultado_latex = 'D = ' + tex(D)
    r.extra('Matriz de paso', 'P = ' + tex(P) + r',\quad P^{-1} = ' + tex(Pi))
    _propio(r, D, _chk_diagonal(A, dd))
    return r


def _chk_diagonal(A, dd):
    def f(texto):
        M = sp.Matrix(leer(texto.split('=')[-1] if re.match(r'^\s*[DP]\s*=', texto) else texto))
        if M.shape != A.shape:
            return _mal('La matriz debe tener el mismo tamaño que A.')
        if M.is_diagonal():
            diag = [M[i, i] for i in range(M.rows)]
            resto = list(dd)
            for v in diag:
                k_ = next((i for i, w in enumerate(resto) if _es_cero(v - w)), None)
                if k_ is None:
                    return _mal('La diagonal debe tener los autovalores de A.', tex(sp.diag(*dd)))
                resto.pop(k_)
            return _ok('¡Correcto! D tiene los autovalores (el orden depende de cómo armes P).')
        if M.det() == 0:
            return _mal('P debe ser invertible.')
        R = (M.inv() * A * M).applyfunc(_simp)
        if R.is_diagonal():
            return _ok('¡Correcto! Es una matriz P válida: P⁻¹AP es diagonal.')
        return _mal('Con esa P, P⁻¹AP no es diagonal: las columnas deben ser autovectores.')
    return f


def _sistema(arg):
    s = normalizar(arg).strip()
    if s.startswith('[['):
        M = leer_matriz(s)
        A, b = M[:, :-1], M[:, -1]
        vs = _nombres_vars(A.shape[1], None)
        return A, b, vs
    partes = dividir(s, ';')
    if len(partes) == 1:
        partes = dividir(s, ',')
    ecs = []
    for p in partes:
        eq = leer_ecuacion(p)
        if not isinstance(eq, sp.Equality):
            raise ErrorEntrada(f'«{p}» no es una ecuación. Separá las ecuaciones con «;».')
        ecs.append(eq.lhs - eq.rhs)
    vs = variables_de(sp.Tuple(*ecs))
    try:
        A, b = sp.linear_eq_to_matrix(ecs, vs)
    except Exception:  # noqa
        raise ErrorEntrada('El sistema debe ser lineal.')
    return A, b, vs


def cramer(arg):
    A, b, vs = _sistema(arg)
    if not A.is_square:
        raise ErrorEntrada(f'La regla de Cramer necesita tantas ecuaciones como incógnitas ({A.shape[0]} ecuaciones, '
                           f'{A.shape[1]} incógnitas).')
    r = Resultado(TEMA_LIN, 'Regla de Cramer')
    r.entrada_latex = r'\begin{cases}' + r'\\'.join(tex(sum(A[i, j] * vs[j] for j in range(len(vs)))) + ' = ' + tex(b[i])
                                                  for i in range(A.shape[0])) + r'\end{cases}'
    dA = _simp(A.det())
    r.paso('Determinante del sistema', r'\(\Delta = \det A\); si es 0 no se puede usar Cramer.', r'\Delta = \det' + tex(A) + ' = ' + tex(dA), 0)
    if dA == 0:
        ra, rb = A.rank(), A.row_join(b).rank()
        txt = 'incompatible (rg A ≠ rg A|b)' if ra != rb else 'compatible indeterminado (infinitas soluciones)'
        r.paso('Δ = 0', 'Cramer no se aplica. Por el teorema de Rouché–Frobenius el sistema es ' + txt + '.',
               f'\\operatorname{{rg}}A = {ra},\\ \\operatorname{{rg}}(A|b) = {rb}', 0)
        r.valor = 'incompatible' if ra != rb else 'indeterminado'
        r.resultado_latex = r'\Delta = 0:\ \text{sistema ' + txt + '}'
        if ra == rb:
            sol = sp.linsolve((A, b), *vs)
            r.extra('Solución general', tex(list(sol)[0]) if sol else '')
        return r
    sol = {}
    for j, v in enumerate(vs):
        Aj = A.copy()
        Aj[:, j] = b
        dj = _simp(Aj.det())
        sol[v] = _simp(dj / dA)
        r.paso(f'Incógnita {v}', f'Se reemplaza la columna {j + 1} por los términos independientes.',
               r'\Delta_{' + tex(v) + r'} = \det' + tex(Aj) + ' = ' + tex(dj) + r'\Rightarrow ' + tex(v) + r' = \frac{'
               + tex(dj) + '}{' + tex(dA) + '} = ' + tex(sol[v]), 1)
    r.valor = sol
    r.resultado_latex = r',\ '.join(f'{tex(v)} = {tex(c)}' for v, c in sol.items())
    r.comparar = ('dict', sol, {})
    return r


# ---------------------------------------------------------------------------
# Polinomios: división y Ruffini
# ---------------------------------------------------------------------------
def _cols_potencias(v, grado):
    return [f'{v}{_sup(k)}' if k > 1 else (str(v) if k == 1 else 'indep.') for k in range(grado, -1, -1)]


def _ruffini_tabla(coefs, a, v):
    fila2, acc = [''], [coefs[0]]
    for c in coefs[1:]:
        prod = _simp(acc[-1] * a)
        fila2.append(_celda(prod))
        acc.append(_simp(c + prod))
    columnas = [''] + _cols_potencias(v, len(coefs) - 1)
    filas = [['Coeficientes'] + [_celda(c) for c in coefs], [f'a = {sp.sstr(a)}'] + fila2,
             ['Resultado'] + [_celda(c) for c in acc]]
    return {'columnas': columnas, 'filas': filas}, acc


def division(arg, solo_ruffini=False):
    s = normalizar(arg)
    s = re.sub(r'\s+(?:entre|dividido(?:\s+(?:por|en))?|por|sobre)\s+', ', ', s)
    partes = _partes(s)
    if len(partes) == 1:
        e = leer(partes[0])
        num, den = sp.fraction(e)
        if den == 1:
            raise ErrorEntrada('Uso: division de polinomios P(x), Q(x)   — ej.: division de polinomios x^3-2x+1, x-1')
        Pp, Qp = num, den
    elif len(partes) == 2:
        Pp, Qp = leer(partes[0]), leer(partes[1])
    else:
        raise ErrorEntrada('Uso: division de polinomios P(x), Q(x)')
    if Pp.is_Integer and Qp.is_Integer and not solo_ruffini:
        return _division_entera(int(Pp), int(Qp))
    v = variables_de(sp.Tuple(Pp, Qp))
    v = v[0] if v else x
    if solo_ruffini and Qp.is_number:
        a = Qp
        Qp = v - a
    if not (Pp.is_polynomial(v) and Qp.is_polynomial(v)):
        raise ErrorEntrada('Dividendo y divisor deben ser polinomios.')
    if Qp == 0:
        raise ErrorEntrada('No se puede dividir por el polinomio nulo.')
    Pp, Qp = sp.expand(Pp), sp.expand(Qp)
    q, resto = sp.div(Pp, Qp, v)
    r = Resultado('Álgebra', 'Regla de Ruffini' if solo_ruffini else 'División de polinomios')
    r.entrada_latex = r'\frac{' + tex(Pp) + '}{' + tex(Qp) + '}'
    gq = sp.degree(Qp, v)
    es_lineal = gq == 1
    if es_lineal:
        k_ = sp.Poly(Qp, v).LC()
        a = _simp(-Qp.subs(v, 0) / k_)
        coefs = sp.Poly(Pp, v).all_coeffs()
        tabla, acc = _ruffini_tabla(coefs, a, v)
        r.tabla = tabla
        r.paso('Regla de Ruffini', f'Divisor {tex(Qp)}: se usa a = {sp.sstr(a)}. Se baja el primer coeficiente; se multiplica '
               'por a, se suma con el siguiente, y así sucesivamente.', r'\text{coeficientes: } ' + ',\\ '.join(tex(c) for c in coefs), 0)
        r.paso('Lectura de la tabla', 'Los primeros números son los coeficientes del cociente (un grado menos); el último, el resto.',
               r'\text{fila final: } ' + ',\\ '.join(tex(c) for c in acc), 1)
        if k_ != 1:
            r.paso('Divisor no mónico', f'Como el divisor es {tex(k_)}·({tex(v - a)}), el cociente de Ruffini se divide por {tex(k_)}.',
                   tex(sum(c * v ** i for i, c in enumerate(reversed(acc[:-1])))) + r'\ \div\ ' + tex(k_) + ' = ' + tex(q), 1)
        r.paso('Teorema del resto', f'El resto de dividir por {tex(v - a)} es P({sp.sstr(a)}).',
               f'P({tex(a)}) = ' + tex(_simp(Pp.subs(v, a))), 0)
    if not es_lineal or not solo_ruffini:
        filas, R = [], Pp
        lcq = sp.LT(Qp, v)
        pasos_txt = []
        while R != 0 and sp.degree(R, v) >= gq:
            term = sp.expand(sp.LT(R, v) / lcq)
            resta = sp.expand(term * Qp)
            R = sp.expand(R - resta)
            filas.append([_celda(term), _celda(resta), _celda(R)])
            pasos_txt.append(r'\frac{' + tex(sp.LT(R + resta, v)) + '}{' + tex(lcq) + '} = ' + tex(term) + r',\quad '
                             + tex(term) + r'\cdot(' + tex(Qp) + ') = ' + tex(resta) + r',\quad \text{queda } ' + tex(R))
        r.paso('División larga', 'Se divide el término de mayor grado del resto parcial por el del divisor, se multiplica y se resta.', '', 0)
        for ptx in pasos_txt[:12]:
            r.paso('Paso', '', ptx, 1)
        if not es_lineal:
            r.tabla = {'columnas': ['Término del cociente', 'Se resta', 'Resto parcial'], 'filas': filas}
    r.paso('Resultado', r'\(P(x) = Q(x)\cdot C(x) + R(x)\)', tex(Pp) + ' = (' + tex(Qp) + ')(' + tex(q) + ') + ' + tex(resto), 0)
    r.valor = [q, resto]
    r.valores['cociente'] = q
    r.valores['resto'] = resto
    r.resultado_latex = r'R = ' + tex(resto) + r',\quad C(' + str(v) + ') = ' + tex(q)
    if resto == 0:
        r.extra('Divisibilidad', '', f'La división es exacta: {sp.sstr(Qp)} es factor de {sp.sstr(Pp)}.')
    _propio(r, r.valor, _chk_division(q, resto))
    return r


def _chk_division(q, resto):
    def f(texto):
        t_ = sin_tildes(texto.lower())
        t_ = re.sub(r'(cociente|c\(x\)|q\(x\)|q)\s*[:=]?\s*', ' ', t_)
        t_ = re.sub(r'(resto|r\(x\)|r)\s*[:=]?\s*', ',', t_)
        partes = dividir(t_.replace(';', ','), ',')
        vals = [leer(p) for p in partes if p.strip()]
        if not vals:
            return {'correcto': None, 'mensaje': 'Escribí el cociente y el resto separados por coma.'}
        ok_q = _es_cero(vals[0] - q)
        if len(vals) == 1:
            if ok_q:
                return _ok('¡Correcto! (el cociente; el resto es ' + sp.sstr(resto) + ').')
            return _mal('El cociente no coincide.', tex(q))
        ok_r = _es_cero(vals[1] - resto)
        if ok_q and ok_r:
            return _ok()
        return _mal('Revisá ' + ('el cociente' if not ok_q else 'el resto') + '.', f'C = {tex(q)},\\ R = {tex(resto)}')
    return f


def _division_entera(a, b):
    if b == 0:
        raise ErrorEntrada('No se puede dividir por 0.')
    q, r_ = divmod(a, b)
    if r_ < 0:
        q, r_ = q + 1, r_ - b
    r = Resultado(TEMA_DIS, 'División entera')
    r.entrada_latex = f'{a} \\div {b}'
    r.paso('Algoritmo de la división', r'\(a = b\cdot q + r\) con \(0 \le r < |b|\).', f'{a} = {b}\\cdot ({q}) + {r_}', 0)
    r.valor = [sp.Integer(q), sp.Integer(r_)]
    r.valores.update({'cociente': q, 'resto': r_})
    r.resultado_latex = f'q = {q},\\quad r = {r_}'
    _propio(r, r.valor, _chk_division(sp.Integer(q), sp.Integer(r_)))
    return r


def ruffini(arg):
    return division(arg, solo_ruffini=True)


# ---------------------------------------------------------------------------
# Coordenadas y cambio de base
# ---------------------------------------------------------------------------
def _separar_base(s):
    m = re.search(r'\s+(?:en|respecto\s+(?:a|de))\s+(?:la\s+)?base\s+(?:B\s*=\s*)?|;', s, flags=re.I)
    if not m:
        raise ErrorEntrada('Uso: coordenadas [3,1] en base [1,1], [1,-1]')
    return s[:m.start()], s[m.end():]


def coordenadas(arg):
    s = normalizar(arg)
    vtxt, btxt = _separar_base(s)
    vparts = _partes(vtxt)
    bparts = _partes(btxt.strip().strip('{}'))
    vecs, clase, mostrar = _a_vectores(vparts + bparts)
    if clase == 'polinomios':
        largo = max(len(v) for v in vecs)
        vecs = [sp.Matrix(list(v) + [0] * (largo - len(v))) for v in vecs]
    v, base = vecs[0], vecs[1:]
    B = sp.Matrix.hstack(*base)
    r = Resultado(TEMA_LIN, 'Coordenadas en una base')
    r.entrada_latex = r'[v]_B,\quad v = ' + mostrar(v) + r',\ B = \left\{' + ',\\ '.join(mostrar(b) for b in base) + r'\right\}'
    if B.rank() < len(base):
        raise ErrorEntrada('Los vectores dados no forman una base (son linealmente dependientes).')
    cs = sp.symbols(f'c1:{len(base) + 1}')
    r.paso('Planteo', r'Se buscan \(c_i\) con \(v = c_1 b_1 + \dots + c_n b_n\).',
           mostrar(v) + ' = ' + ' + '.join(f'{tex(c)}\\,{mostrar(b)}' for c, b in zip(cs, base)), 0)
    red, pasos = gauss_jordan_pasos(B.row_join(v), aumentada=True)
    r.paso('Sistema [B | v]', 'Se resuelve por Gauss-Jordan.', tex(B.row_join(v)), 0)
    r.pasos.extend(pasos)
    sol = sp.linsolve((B, v), *cs)
    if not sol:
        raise ErrorEntrada('El vector no pertenece al subespacio generado por esos vectores.')
    c = [_simp(a) for a in list(sol)[0]]
    r.paso('Coordenadas', '', r'[v]_B = ' + _tt(c), 0)
    r.valor = c
    r.resultado_latex = r'[v]_B = ' + _tt(c)
    r.comparar = ('vector', c, {})
    return r


def cambio_base(arg):
    s = normalizar(arg)
    s = re.sub(r'^\s*(?:de\s+)?(?:la\s+)?(?:base\s+)?', '', s, flags=re.I)
    trozos = [p for p in re.split(r';|\s+(?:a|hacia)\s+(?:la\s+)?(?:base\s+)?', s) if p.strip()]
    bases = []
    for tr in trozos:
        tr = re.sub(r'^\s*[A-Za-z]\d?\s*=\s*', '', tr.strip()).strip().strip('{}')
        vecs, clase, mostrar = _a_vectores(_partes(tr))
        bases.append(sp.Matrix.hstack(*vecs))
    r = Resultado(TEMA_LIN, 'Matriz de cambio de base')
    for M in bases:
        if not M.is_square or M.det() == 0:
            raise ErrorEntrada('Cada base debe tener n vectores independientes de Rⁿ.')
    if len(bases) == 1:
        B = bases[0]
        r.entrada_latex = r'B = \left\{' + ',\\ '.join(_tt(B[:, j]) for j in range(B.cols)) + r'\right\}\to E'
        r.paso('De B a la canónica', 'Las columnas son los vectores de B (sus coordenadas canónicas).', r'C_{B\to E} = ' + tex(B), 0)
        Ci = B.inv()
        r.paso('De la canónica a B', 'Es la inversa.', r'C_{E\to B} = ' + tex(Ci), 0)
        C = B
    elif len(bases) == 2:
        B1, B2 = bases
        if B1.shape != B2.shape:
            raise ErrorEntrada('Las dos bases deben ser del mismo espacio.')
        r.entrada_latex = r'B = \left\{' + ',\\ '.join(_tt(B1[:, j]) for j in range(B1.cols)) + r'\right\}\to B\' = \left\{' \
            + ',\\ '.join(_tt(B2[:, j]) for j in range(B2.cols)) + r'\right\}'
        r.paso('Idea', r'Cada columna de \(C_{B\to B\'}\) son las coordenadas de un vector de B en la base B\'.',
               r'C_{B\to B\'} = (B\')^{-1} B', 0)
        C = (B2.inv() * B1).applyfunc(_simp)
        for j in range(B1.cols):
            r.paso(f'Vector {j + 1} de B', '', r'[' + _tt(B1[:, j]) + r"]_{B'} = " + _tt(C[:, j]), 1)
        Ci = C.inv()
    else:
        raise ErrorEntrada("Uso: cambio de base [1,0],[0,1]; [1,1],[1,-1]   (de B a B', separadas por «;»)")
    r.valor = C
    r.valores['inversa'] = Ci
    r.resultado_latex = r'C = ' + tex(C)
    r.extra('Cambio inverso', tex(Ci))
    r.comparar = ('matriz', C, {})
    return r


# ============================================================================
# 3. MATEMÁTICA DISCRETA / ÁLGEBRA I-II
# ============================================================================
def _tex_factores(fs):
    if not fs:
        return '1'
    return r'\cdot '.join(f'{p}^{{{e}}}' if e > 1 else str(p) for p, e in sorted(fs.items()))


def factores_primos(arg):
    n_ = _entero(_limpiar(arg), 'el número')
    if abs(n_) < 2:
        raise ErrorEntrada('Ingresá un entero mayor que 1 (o menor que -1).')
    m = abs(n_)
    fs = sp.factorint(m)
    r = Resultado(TEMA_DIS, 'Descomposición en factores primos')
    r.entrada_latex = str(n_)
    filas, actual = [], m
    for p in sorted(fs):
        for _ in range(fs[p]):
            filas.append([str(actual), str(p)])
            actual //= p
    filas.append(['1', ''])
    r.tabla = {'columnas': ['Número', 'Divisor primo'], 'filas': filas[:80]}
    r.paso('Divisiones sucesivas', 'Se divide por el menor primo posible hasta llegar a 1.',
           r'\begin{array}{r|l}' + r'\\'.join(f'{a} & {b}' for a, b in filas[:25]) + r'\end{array}', 0)
    fac = _tex_factores(fs)
    signo = '-' if n_ < 0 else ''
    r.paso('Resultado', 'Se agrupan los primos repetidos como potencias.', f'{n_} = {signo}{fac}', 0)
    ndiv = reduce(lambda a, b: a * b, [e + 1 for e in fs.values()], 1)
    r.extra('Cantidad de divisores positivos', r'\tau(n) = ' + r'\cdot '.join(f'({e}+1)' for e in fs.values()) + f' = {ndiv}')
    r.extra('Suma de divisores', r'\sigma(n) = ' + str(sp.divisor_sigma(m)))
    r.extra('Función φ de Euler', r'\varphi(n) = ' + str(sp.totient(m)))
    if len(fs) == 1 and list(fs.values())[0] == 1:
        r.extra('Primalidad', '', f'{m} es primo.')
    texto = ('-' if n_ < 0 else '') + '*'.join(f'{p}**{e}' if e > 1 else str(p) for p, e in sorted(fs.items()))
    r.valor = texto
    r.valores['factores'] = '{' + ', '.join(f'{p}: {e}' for p, e in sorted(fs.items())) + '}'
    r.valores['divisores'] = ndiv
    r.resultado_latex = f'{n_} = {signo}{fac}'

    def chk(t_):
        s = normalizar(t_).replace('x', '*').replace('.', '*')
        s = s.split('=')[-1]
        trozos = [p.strip() for p in s.split('*') if p.strip()]
        prod, alumno = 1, {}
        for tr in trozos:
            mm = re.fullmatch(r'\(?\s*(-?\d+)\s*\)?\s*(?:(?:\^|\*\*)\s*(\d+))?', tr)
            if not mm:
                return {'correcto': None, 'mensaje': 'Escribí la factorización como 2^3*3^2*5.'}
            b, e = int(mm.group(1)), int(mm.group(2) or 1)
            if b != -1:
                alumno[abs(b)] = alumno.get(abs(b), 0) + e
            prod *= b ** e
        if prod != n_:
            return _mal(f'Ese producto da {prod}, no {n_}.', fac)
        no_primos = [b for b in alumno if not sp.isprime(b)]
        if no_primos:
            return _mal('El producto es correcto, pero ' + ', '.join(map(str, no_primos)) +
                        (' no son primos' if len(no_primos) > 1 else ' no es primo') + ': seguí descomponiendo.', fac)
        return _ok()
    _propio(r, texto, chk)
    return r


def divisores(arg):
    n_ = abs(_entero(_limpiar(arg), 'el número'))
    if n_ == 0:
        raise ErrorEntrada('El 0 tiene infinitos divisores.')
    ds = sp.divisors(n_)
    r = Resultado(TEMA_DIS, 'Divisores')
    r.entrada_latex = r'\operatorname{Div}(' + str(n_) + ')'
    fs = sp.factorint(n_)
    r.paso('Factorización', 'Cada divisor elige un exponente entre 0 y el de cada primo.', f'{n_} = ' + _tex_factores(fs), 0)
    r.paso('Divisores positivos', f'Son {len(ds)}.', ', '.join(map(str, ds)), 0)
    r.valor = [sp.Integer(d) for d in ds]
    r.valores['cantidad'] = len(ds)
    r.resultado_latex = r'\{' + ', '.join(map(str, ds)) + r'\}' if len(ds) <= 40 else f'{len(ds)}\\ \\text{{divisores}}'
    r.comparar = ('conjunto', r.valor, {})
    return r


def es_primo(arg):
    n_ = _entero(_limpiar(arg), 'el número')
    r = Resultado(TEMA_DIS, '¿Es primo?')
    r.entrada_latex = str(n_)
    pr = sp.isprime(n_)
    raiz = int(math.isqrt(abs(n_))) if n_ > 0 else 0
    if pr:
        r.paso('Prueba de divisores', f'Basta probar los primos hasta √{n_} ≈ {raiz}: ninguno lo divide.',
               ', '.join(str(p) for p in sp.primerange(2, raiz + 1))[:300] or r'\text{(no hay)}', 0)
    elif n_ > 1:
        p = min(sp.factorint(n_))
        r.paso('Divisor encontrado', '', f'{n_} = {p}\\cdot {n_ // p}', 0)
    r.valor = 'primo' if pr else 'compuesto'
    r.resultado_latex = r'\text{' + str(n_) + (' es primo' if pr else ' no es primo') + '}'
    _propio(r, r.valor, _chk_palabras('primo' if pr else 'compuesto', {'compuesto': [r'compuest', r'no es primo', r'^\s*no\b'],
                                                                        'primo': [r'primo', r'^\s*si\b']}))
    return r


def _bin_pot(b, e, m):
    """Exponenciación modular por cuadrados sucesivos: filas de la tabla."""
    filas, pot, k_, res, usados = [], b % m, 0, 1, []
    ee = e
    while ee > 0:
        bit = ee & 1
        filas.append([str(k_), f'2{_sup(k_)} = {2 ** k_}', str(pot), str(bit)])
        if bit:
            res = (res * pot) % m
            usados.append((2 ** k_, pot))
        pot = (pot * pot) % m
        ee >>= 1
        k_ += 1
    return filas, res, usados


def modulo(arg):
    s = normalizar(arg).strip()
    tiene_mod = re.search(r'(?i)\bmod(?:ulo)?\b', s) or ',' in s or re.search(r'\s(?:entre|por)\s', s)
    if s.startswith('[') or not tiene_mod:
        from .lineal import norma
        return norma(arg)
    s = re.sub(r'(?i)\s*\bmod(?:ulo)?\b\s*', ',', s)
    s = re.sub(r'\s+(?:entre|por)\s+', ',', s)
    partes = [p for p in dividir(s, ',') if p.strip()]
    if len(partes) != 2:
        raise ErrorEntrada('Uso: modulo a, m   o   modulo 2^100 mod 7')
    m = _entero(partes[1], 'el módulo')
    if m == 0:
        raise ErrorEntrada('El módulo no puede ser 0.')
    m = abs(m)
    r = Resultado(TEMA_DIS, 'Aritmética modular')
    pot = re.fullmatch(r'\s*\(?\s*(-?\d+)\s*\)?\s*(?:\^|\*\*)\s*\(?\s*(\d+)\s*\)?\s*', partes[0])
    if pot:
        b, e = int(pot.group(1)), int(pot.group(2))
        r.titulo = 'Potencia modular'
        r.entrada_latex = f'{b}^{{{e}}} \\bmod {m}'
        res = pow(b, e, m)
        filas, res2, usados = _bin_pot(b, e, m)
        r.paso('Exponente en binario', 'Se escribe el exponente como suma de potencias de 2.',
               f'{e} = ' + ' + '.join(str(u) for u, _ in usados) + f' = ({bin(e)[2:]})_2', 0)
        r.paso('Cuadrados sucesivos', f'Cada fila es el cuadrado de la anterior, reducido módulo {m}.', '', 0)
        r.tabla = {'columnas': ['k', 'potencia de 2', f'{b}^(2^k) mod {m}', 'bit'], 'filas': filas[:64]}
        r.paso('Producto de las filas con bit 1', '', f'{b}^{{{e}}} \\equiv ' + r'\cdot '.join(str(p) for _, p in usados)
               + f' \\equiv {res} \\pmod{{{m}}}', 0)
        if math.gcd(b, m) == 1 and m > 1:
            ph = int(sp.totient(m))
            nombre = 'pequeño teorema de Fermat' if sp.isprime(m) else 'teorema de Euler'
            red_e = e % ph
            r.paso('Atajo: ' + nombre, f'Como mcd({b}, {m}) = 1, se cumple {b}^φ({m}) ≡ 1 (mod {m}) con φ({m}) = {ph}; '
                   'se reduce el exponente módulo φ.', f'{e} = {ph}\\cdot {e // ph} + {red_e}\\Rightarrow {b}^{{{e}}} \\equiv '
                   f'{b}^{{{red_e}}} \\equiv {pow(b, red_e, m)} \\pmod{{{m}}}', 1)
            ciclo, p_ = [], b % m
            for _ in range(min(ph, 30)):
                ciclo.append(p_)
                if p_ == 1:
                    break
                p_ = (p_ * b) % m
            if ciclo and ciclo[-1] == 1 and len(ciclo) <= 30:
                per = len(ciclo)
                r.paso('Ciclo de restos', f'Las potencias de {b} módulo {m} se repiten cada {per}.',
                       ',\\ '.join(f'{b}^{{{i + 1}}}\\equiv {c}' for i, c in enumerate(ciclo)) + f'\\Rightarrow {e} \\bmod {per} = {e % per}', 1)
        r.valor = sp.Integer(res)
        r.resultado_latex = f'{b}^{{{e}}} \\equiv {res} \\pmod{{{m}}}'
    else:
        a = _entero(partes[0], 'el número')
        q, res = a // m, a % m
        r.entrada_latex = f'{a} \\bmod {m}'
        r.paso('Algoritmo de la división', r'\(a = m\cdot q + r\) con \(0 \le r < m\).', f'{a} = {m}\\cdot ({q}) + {res}', 0)
        r.valor = sp.Integer(res)
        r.resultado_latex = f'{a} \\equiv {res} \\pmod{{{m}}}'
    r.valores['modulo'] = m
    r.comparar = ('numero_o_expr', r.valor, {})
    return r


def _inverso_txt(a, m):
    """Inverso de a módulo m (a, m coprimos) con su justificación."""
    s_, t_, g = sp.gcdex(a, m)
    inv = int(s_) % m
    return inv, f'{a}\\cdot ({s_}) + {m}\\cdot ({t_}) = 1 \\Rightarrow {a}^{{-1}} \\equiv {inv} \\pmod{{{m}}}'


def _leer_congruencia(texto):
    s = normalizar(texto).replace('≡', '=').replace('==', '=').replace('≅', '=')
    m = re.search(r'\(?\s*\bmod(?:ulo)?\b\s*([^)]+?)\s*\)?\s*$', s, flags=re.I)
    if not m:
        raise ErrorEntrada('Indicá el módulo, por ejemplo: 3x ≡ 4 (mod 7).')
    mod = abs(_entero(m.group(1), 'el módulo'))
    eq = s[:m.start()].strip()
    if '=' not in eq:
        raise ErrorEntrada('Escribí la congruencia con ≡ (o =): 3x ≡ 4 (mod 7).')
    lhs, rhs = eq.split('=', 1)
    e = sp.expand(leer(lhs) - leer(rhs))
    vs = list(e.free_symbols)
    if len(vs) != 1:
        raise ErrorEntrada('La congruencia debe tener una sola incógnita.')
    v = vs[0]
    Pl = sp.Poly(e, v)
    if Pl.degree() != 1 or not all(c.is_Integer for c in Pl.all_coeffs()):
        raise ErrorEntrada('Solo resuelvo congruencias lineales a·x ≡ b (mod m) con coeficientes enteros.')
    a, c = (int(cc) for cc in Pl.all_coeffs())
    return a, -c, mod, v


def _resolver_lineal_cong(r, a, b, m, v, nivel=0):
    """a v ≡ b (mod m) -> (x0, m') o None"""
    a0, b0 = a % m, b % m
    if (a0, b0) != (a, b):
        r.paso('Reducir coeficientes', f'Se reducen módulo {m}.', f'{a0}{v} \\equiv {b0} \\pmod{{{m}}}', nivel)
    g = math.gcd(a0, m)
    r.paso('Condición de existencia', f'Tiene solución si mcd(a, m) divide a b.', f'\\operatorname{{mcd}}({a0}, {m}) = {g}'
           + (f',\\ {g} \\mid {b0}' if b0 % g == 0 else f',\\ {g} \\nmid {b0}'), nivel)
    if b0 % g:
        return None
    a1, b1, m1 = a0 // g, b0 // g, m // g
    if g > 1:
        r.paso('Simplificar', f'Se divide todo por {g}.', f'{a1}{v} \\equiv {b1} \\pmod{{{m1}}}', nivel)
    if m1 == 1:
        return 0, 1
    inv, txt = _inverso_txt(a1, m1)
    r.paso('Inverso modular', 'Con el algoritmo de Euclides extendido (identidad de Bézout).', txt, nivel)
    x0 = (inv * b1) % m1
    r.paso('Despeje', f'Se multiplica por el inverso.', f'{v} \\equiv {inv}\\cdot {b1} \\equiv {x0} \\pmod{{{m1}}}', nivel)
    return x0, m1


def congruencia(arg):
    partes = [p for p in dividir(normalizar(_limpiar(arg)), ';') if p.strip()]
    if len(partes) == 1 and ',' in partes[0] and len(re.findall(r'mod', partes[0])) > 1:
        partes = dividir(partes[0], ',')
    r = Resultado(TEMA_DIS, 'Congruencia lineal' if len(partes) == 1 else 'Sistema de congruencias')
    datos = [_leer_congruencia(p) for p in partes]
    r.entrada_latex = r'\begin{cases}' + r'\\'.join(f'{a}{v} \\equiv {b} \\pmod{{{m}}}' for a, b, m, v in datos) + r'\end{cases}' \
        if len(datos) > 1 else f'{datos[0][0]}{datos[0][3]} \\equiv {datos[0][1]} \\pmod{{{datos[0][2]}}}'
    v = datos[0][3]
    sols = []
    for i, (a, b, m, vv) in enumerate(datos):
        if len(datos) > 1:
            r.paso(f'Congruencia {i + 1}', '', f'{a}{v} \\equiv {b} \\pmod{{{m}}}', 0)
        s_ = _resolver_lineal_cong(r, a, b, m, v, 1 if len(datos) > 1 else 0)
        if s_ is None:
            r.paso('Sin solución', 'mcd(a, m) no divide a b.', '', 0)
            r.valor = 'sin solucion'
            r.resultado_latex = r'\text{No tiene solución}'
            r.comparar = ('texto', 'no existe', {})
            return r
        sols.append(s_)
    x0, M = sols[0]
    k_ = sp.Symbol('k')
    for (x1, m1) in sols[1:]:
        r.paso('Combinar (teorema chino del resto)', f'De {v} ≡ {x0} (mod {M}): {v} = {x0} + {M}k; se reemplaza en '
               f'{v} ≡ {x1} (mod {m1}).', f'{M}k \\equiv {(x1 - x0) % m1} \\pmod{{{m1}}}', 0)
        s_ = _resolver_lineal_cong(r, M, x1 - x0, m1, k_, 1)
        if s_ is None:
            r.paso('Sin solución', 'Las congruencias son incompatibles.', '', 0)
            r.valor = 'sin solucion'
            r.resultado_latex = r'\text{El sistema no tiene solución}'
            r.comparar = ('texto', 'no existe', {})
            return r
        k0, mk = s_
        x0, M = (x0 + M * k0) % (M * mk), M * mk
        r.paso('Nueva congruencia', '', f'{v} \\equiv {x0} \\pmod{{{M}}}', 1)
    r.paso('Solución', f'Todos los enteros de la forma {x0} + {M}k.', f'{v} \\equiv {x0} \\pmod{{{M}}}', 0)
    if len(datos) == 1 and datos[0][2] != M:
        mm = datos[0][2]
        lista = [x0 + M * j for j in range(mm // M)]
        r.extra(f'Soluciones módulo {mm}', ', '.join(map(str, lista)))
        r.valores['soluciones'] = lista
    r.valor = sp.Integer(x0)
    r.valores['modulo'] = M
    r.valores['general'] = f'{x0} + {M}*k'
    r.resultado_latex = f'{v} \\equiv {x0} \\pmod{{{M}}}'

    def chk(texto):
        s = normalizar(texto).replace('≡', '=')
        mm = re.search(r'\(?\s*\bmod(?:ulo)?\b\s*(\d+)\s*\)?', s)
        if mm:
            if int(mm.group(1)) % M and M % int(mm.group(1)):
                return _mal(f'El módulo de la solución es {M}.', r.resultado_latex)
            s = s[:mm.start()]
        s = re.sub(r'^\s*[a-z]\s*=', '', s.strip())
        e = leer(s)
        ks = list(e.free_symbols)
        if not ks:
            if mm and int(mm.group(1)) != M:
                return _mal(f'El módulo de la solución es {M}.', r.resultado_latex)
            return _ok() if (int(e) - x0) % M == 0 else _mal('No es solución.', r.resultado_latex)
        kk = ks[0]
        e0, ck = e.subs(kk, 0), sp.diff(e, kk)
        if (int(e0) - x0) % M == 0 and abs(int(ck)) == M:
            return _ok()
        return _mal('La familia de soluciones no coincide.', r.resultado_latex)
    _propio(r, r.valor, chk)
    return r


def euclides(arg):
    partes = _partes(arg)
    if len(partes) != 2:
        raise ErrorEntrada('Uso: euclides a, b   — ej.: euclides 252, 198')
    a, b = (_entero(p, 'cada número') for p in partes)
    if a == 0 and b == 0:
        raise ErrorEntrada('mcd(0, 0) no está definido.')
    A, B = abs(a), abs(b)
    r = Resultado(TEMA_DIS, 'Algoritmo de Euclides')
    r.entrada_latex = f'\\operatorname{{mcd}}({a}, {b})'
    filas, divs = [], []
    r0, r1, s0, s1, t0, t1 = A, B, 1, 0, 0, 1
    filas.append(['', str(r0), '', '1', '0'])
    filas.append(['', str(r1), '', '0', '1'])
    while r1 != 0:
        q = r0 // r1
        divs.append(f'{r0} = {r1}\\cdot {q} + {r0 - q * r1}')
        r0, r1 = r1, r0 - q * r1
        s0, s1 = s1, s0 - q * s1
        t0, t1 = t1, t0 - q * t1
        filas.append([str(q), str(r1), '', str(s1), str(t1)])
    g = r0
    s_, t_ = s0 * (1 if a >= 0 else -1), t0 * (1 if b >= 0 else -1)
    for f in filas:
        f.pop(2)
    r.paso('Divisiones sucesivas', 'Se divide el mayor por el menor, luego el divisor por el resto, hasta que el resto es 0. '
           'El último resto no nulo es el mcd.', r'\begin{aligned}' + r'\\'.join(divs[:30]) + r'\end{aligned}', 0)
    r.tabla = {'columnas': ['cociente q', 'resto r', 's', 't'], 'filas': filas[:60]}
    r.paso('Euclides extendido', r'Cada resto se escribe como \(r = a\,s + b\,t\): \(s_{i+1} = s_{i-1} - q\,s_i\) '
           r'(igual para t). La fila del mcd da los coeficientes de Bézout.', '', 0)
    r.paso('Identidad de Bézout', '', f'{g} = {a}\\cdot ({s_}) + {b}\\cdot ({t_})', 0)
    mcm = A * B // g if g else 0
    r.extra('mcm', r'\operatorname{mcm} = \frac{|a\,b|}{\operatorname{mcd}} = ' + str(mcm))
    if g == 1:
        r.extra('Coprimos', '', f'{a} y {b} son coprimos.')
    r.valor = sp.Integer(g)
    r.valores.update({'mcd': g, 's': s_, 't': t_, 'mcm': mcm})
    r.resultado_latex = f'\\operatorname{{mcd}}({a}, {b}) = {g} = {a}\\cdot ({s_}) + {b}\\cdot ({t_})'
    r.comparar = ('numero_o_expr', sp.Integer(g), {})
    return r


def diofantica(arg):
    eq = leer_ecuacion(_limpiar(normalizar(arg)))
    if not isinstance(eq, sp.Equality):
        raise ErrorEntrada('Escribí una ecuación, por ejemplo 6x+9y=21.')
    e = sp.expand(eq.lhs - eq.rhs)
    vs = variables_de(e)
    if len(vs) != 2:
        raise ErrorEntrada('La ecuación diofántica lineal debe tener dos incógnitas: ax + by = c.')
    v1, v2 = vs
    Pl = sp.Poly(e, v1, v2)
    if Pl.total_degree() != 1 or not all(c.is_Integer for c in Pl.coeffs()):
        raise ErrorEntrada('Debe ser lineal con coeficientes enteros: ax + by = c.')
    a, b, c = int(e.coeff(v1)), int(e.coeff(v2)), int(-e.subs({v1: 0, v2: 0}))
    if a == 0 or b == 0:
        raise ErrorEntrada('Los dos coeficientes deben ser no nulos.')
    r = Resultado(TEMA_DIS, 'Ecuación diofántica lineal')
    r.entrada_latex = tex(a * v1 + b * v2) + ' = ' + str(c)
    g = math.gcd(a, b)
    s_, t_, _ = (int(w) for w in sp.gcdex(a, b))
    r.paso('mcd de los coeficientes', 'Por el algoritmo de Euclides.', f'\\operatorname{{mcd}}({a}, {b}) = {g}', 0)
    if c % g:
        r.paso('Sin solución', f'{g} no divide a {c}: la ecuación no tiene soluciones enteras.', f'{g} \\nmid {c}', 0)
        r.valor = 'sin solucion'
        r.resultado_latex = r'\text{No tiene soluciones enteras}'
        r.comparar = ('texto', 'no existe', {})
        return r
    r.paso('Existencia', f'{g} divide a {c}: hay infinitas soluciones.', f'{g} \\mid {c}', 0)
    r.paso('Identidad de Bézout', '', f'{a}\\cdot ({s_}) + {b}\\cdot ({t_}) = {g}', 0)
    k_ = c // g
    xp, yp = s_ * k_, t_ * k_
    r.paso('Solución particular', f'Se multiplica la identidad por {c}/{g} = {k_}.', f'{v1}_0 = {xp},\\ {v2}_0 = {yp}', 0)
    db, da = b // g, a // g
    sh = (xp % abs(db) - xp) // db
    x0, y0 = xp + db * sh, yp - da * sh
    par = sp.Symbol('t') if t not in (v1, v2) else sp.Symbol('k')
    X_, Y_ = x0 + db * par, y0 - da * par
    if (x0, y0) != (xp, yp):
        r.paso('Particular más simple', f'Se corre el parámetro para que {v1}₀ quede entre 0 y {abs(db) - 1}.',
               f'{v1}_0 = {x0},\\ {v2}_0 = {y0}', 1)
    r.paso('Solución general', r'\(x = x_0 + \frac{b}{d}t,\ y = y_0 - \frac{a}{d}t,\ t\in\mathbb Z\)',
           f'{tex(v1)} = {tex(X_)},\\quad {tex(v2)} = {tex(Y_)},\\quad {tex(par)}\\in\\mathbb{{Z}}', 0)
    r.paso('Verificación', '', f'{a}({tex(X_)}) + {b}({tex(Y_)}) = {tex(sp.expand(a * X_ + b * Y_))}', 1)
    if (a > 0) == (b > 0):
        # x = x0 + db t >= 0 , y = y0 - da t >= 0  (a, b del mismo signo: finitas)
        cotas = sorted([sp.Rational(-x0, db), sp.Rational(y0, da)], key=float)
        desde, hasta = int(sp.ceiling(cotas[0])), int(sp.floor(cotas[1]))
        nat = [(x0 + db * tt_, y0 - da * tt_) for tt_ in range(desde, min(hasta, desde + 25) + 1)
               if x0 + db * tt_ >= 0 and y0 - da * tt_ >= 0]
        def valida(tt_):
            return x0 + db * tt_ >= 0 and y0 - da * tt_ >= 0
        if hasta - desde > 25 and valida(desde) and valida(hasta):
            r.extra('Soluciones con x, y ≥ 0', '', f'Hay {hasta - desde + 1} (para {par} entre {desde} y {hasta}).')
        elif nat:
            r.extra('Soluciones con x, y ≥ 0', ', '.join(f'({p}, {q})' for p, q in nat))
        elif not nat:
            r.extra('Soluciones con x, y ≥ 0', '', 'No hay.')
    sol = {v1: X_, v2: Y_}
    r.valor = sol
    r.valores.update({'mcd': g, 'particular': [x0, y0]})
    r.resultado_latex = f'{tex(v1)} = {tex(X_)},\\ {tex(v2)} = {tex(Y_)}'

    def chk(texto):
        d = {}
        for p in re.split(r'[,;]', normalizar(texto)):
            if '=' in p:
                k1, k2 = p.split('=', 1)
                d[leer(k1)] = leer(k2)
        if v1 not in d or v2 not in d:
            return {'correcto': None, 'mensaje': f'Escribí {v1} = ..., {v2} = ... en función de un parámetro.'}
        ex, ey = d[v1], d[v2]
        if not _es_cero(sp.expand(a * ex + b * ey - c)):
            return _mal('No verifica la ecuación.', r.resultado_latex)
        ps = (ex.free_symbols | ey.free_symbols)
        if not ps:
            return _ok('Es una solución particular correcta; falta la solución general con parámetro.', True)
        pp = ps.pop()
        if abs(sp.diff(ex, pp)) != abs(db):
            return _mal('Verifica la ecuación, pero no describe todas las soluciones (revisá el coeficiente del parámetro).',
                        r.resultado_latex)
        return _ok()
    _propio(r, sol, chk)
    return r


# ---------------------------------------------------------------------------
# Lógica proposicional y álgebra de Boole
# ---------------------------------------------------------------------------
_KW = {'y': 'AND', 'and': 'AND', 'o': 'OR', 'or': 'OR', 'no': 'NOT', 'not': 'NOT', 'xor': 'XOR',
       'verdadero': 'T', 'true': 'T', 'falso': 'F', 'false': 'F'}
_SIMB = [('<->', 'EQ'), ('<=>', 'EQ'), ('↔', 'EQ'), ('⇔', 'EQ'), ('->', 'IMP'), ('=>', 'IMP'), ('→', 'IMP'), ('⇒', 'IMP'),
         ('⊕', 'XOR'), ('⊻', 'XOR'), ('∧', 'AND'), ('&&', 'AND'), ('&', 'AND'), ('*', 'AND'), ('·', 'AND'), ('.', 'AND'),
         ('∨', 'OR'), ('||', 'OR'), ('|', 'OR'), ('+', 'OR'), ('¬', 'NOT'), ('~', 'NOT'), ('!', 'NOT'),
         ("'", 'PRIME'), ('’', 'PRIME'), ('(', '('), (')', ')'), ('[', '('), (']', ')'), ('1', 'T'), ('0', 'F')]


def _tokens(s, modo):
    out, i = [], 0
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        for sim, tipo in _SIMB:
            if s.startswith(sim, i):
                out.append((tipo, sim))
                i += len(sim)
                break
        else:
            if s[i] == '^':
                out.append(('AND' if modo == 'logica' else 'XOR', '^'))
                i += 1
                continue
            m = re.match(r'[A-Za-z_][A-Za-z0-9_]*', s[i:])
            if not m:
                raise ErrorEntrada(f'No reconozco el símbolo «{s[i]}» en la expresión lógica.')
            w = m.group(0)
            i += len(w)
            wl = w.lower()
            if wl in _KW:
                out.append((_KW[wl], w))
            elif modo == 'logica' and wl == 'v':
                out.append(('OR', w))
            elif re.fullmatch(r'[A-Za-z]\d*', w) or '_' in w:
                out.append(('VAR', w))
            else:
                for pieza in re.findall(r'[A-Za-z]\d*', w):
                    out.append(('VAR', pieza))
    return out


class _ParserLogico:
    def __init__(self, toks):
        self.t, self.i, self.sub = toks, 0, []

    def ver(self):
        return self.t[self.i][0] if self.i < len(self.t) else None

    def tomar(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def reg(self, e):
        if not e.is_Atom and e not in self.sub:
            self.sub.append(e)
        return e

    def todo(self):
        e = self.equiv()
        if self.i < len(self.t):
            raise ErrorEntrada(f'Sobra algo cerca de «{self.t[self.i][1]}» en la expresión lógica.')
        return e

    def equiv(self):
        a = self.impl()
        while self.ver() == 'EQ':
            self.tomar()
            a = self.reg(Equivalent(a, self.impl(), evaluate=False))
        return a

    def impl(self):
        a = self.xor()
        if self.ver() == 'IMP':
            self.tomar()
            a = self.reg(Implies(a, self.impl(), evaluate=False))
        return a

    def xor(self):
        a = self.or_()
        while self.ver() == 'XOR':
            self.tomar()
            a = self.reg(Xor(a, self.or_(), evaluate=False))
        return a

    def or_(self):
        a = self.and_()
        while self.ver() == 'OR':
            self.tomar()
            a = self.reg(Or(a, self.and_(), evaluate=False))
        return a

    def and_(self):
        a = self.not_()
        while True:
            v = self.ver()
            if v == 'AND':
                self.tomar()
                b = self.not_()
            elif v in ('VAR', '(', 'NOT', 'T', 'F'):
                b = self.not_()
            else:
                return a
            a = self.reg(And(a, b, evaluate=False))

    def not_(self):
        if self.ver() == 'NOT':
            self.tomar()
            return self.reg(Not(self.not_(), evaluate=False))
        a = self.atomo()
        while self.ver() == 'PRIME':
            self.tomar()
            a = self.reg(Not(a, evaluate=False))
        return a

    def atomo(self):
        if self.ver() is None:
            raise ErrorEntrada('La expresión lógica está incompleta.')
        tipo, txt = self.tomar()
        if tipo == 'VAR':
            return sp.Symbol(txt)
        if tipo == 'T':
            return sp.true
        if tipo == 'F':
            return sp.false
        if tipo == '(':
            a = self.equiv()
            if self.ver() != ')':
                raise ErrorEntrada('Falta cerrar un paréntesis en la expresión lógica.')
            self.tomar()
            return a
        raise ErrorEntrada(f'Falta un operando cerca de «{txt}».')


def leer_logica(texto, modo='logica'):
    s = texto.replace('−', '-').replace('“', '"').strip()
    toks = _tokens(s, modo)
    if not toks:
        raise ErrorEntrada('Falta la expresión lógica.')
    p = _ParserLogico(toks)
    e = p.todo()
    return e, p.sub


_PREC = {Equivalent: 1, Implies: 2, Xor: 3, Or: 4, And: 5, Not: 6}


def _fmt_log(e, L=True):
    sim = {And: r'\land ' if L else ' ∧ ', Or: r'\lor ' if L else ' ∨ ', Xor: r'\veebar ' if L else ' ⊻ ',
           Implies: r'\Rightarrow ' if L else ' ⇒ ', Equivalent: r'\Leftrightarrow ' if L else ' ⇔ '}
    if e is sp.true or e == True:  # noqa
        return 'V'
    if e is sp.false or e == False:  # noqa
        return 'F'
    if isinstance(e, sp.Symbol):
        return str(e)
    if isinstance(e, Not):
        a = e.args[0]
        inner = _fmt_log(a, L)
        if not (isinstance(a, sp.Symbol) or isinstance(a, Not)):
            inner = (r'\left(' + inner + r'\right)') if L else '(' + inner + ')'
        return (r'\neg ' if L else '¬') + inner
    tipo = type(e)
    pr = _PREC.get(tipo, 0)
    partes = []
    for a in e.args:
        s_ = _fmt_log(a, L)
        pa = _PREC.get(type(a), 9)
        if pa <= pr and not isinstance(a, (sp.Symbol, Not)):
            s_ = (r'\left(' + s_ + r'\right)') if L else '(' + s_ + ')'
        partes.append(s_)
    return sim.get(tipo, ' ? ').join(partes)


def _ordenar_lit(a):
    s_ = a.args[0] if isinstance(a, Not) else a
    return str(s_)


def _fmt_boole(e, L=False):
    if e is sp.true or e == True:  # noqa
        return '1'
    if e is sp.false or e == False:  # noqa
        return '0'
    if isinstance(e, sp.Symbol):
        return str(e)
    if isinstance(e, Not):
        a = e.args[0]
        if isinstance(a, sp.Symbol):
            return (r'\overline{' + str(a) + '}') if L else str(a) + "'"
        return (r'\overline{' + _fmt_boole(a, L) + '}') if L else '(' + _fmt_boole(a, L) + ")'"
    if isinstance(e, And):
        fs = []
        for a in sorted(e.args, key=_ordenar_lit):
            s_ = _fmt_boole(a, L)
            if isinstance(a, (Or, Xor)):
                s_ = '(' + s_ + ')'
            fs.append(s_)
        return ('' if all(isinstance(a, (sp.Symbol, Not)) for a in e.args) else r'\cdot ' if L else '·').join(fs)
    if isinstance(e, Or):
        if all(isinstance(a, (sp.Symbol, Not)) for a in e.args):
            return ' + '.join(_fmt_boole(a, L) for a in sorted(e.args, key=_ordenar_lit))
        return ' + '.join(_fmt_boole(a, L) for a in sorted(e.args, key=lambda a: (len(str(a)), str(a))))
    if isinstance(e, Xor):
        return (r' \oplus ' if L else ' ⊕ ').join('(' + _fmt_boole(a, L) + ')' if isinstance(a, (Or, And)) else _fmt_boole(a, L) for a in e.args)
    return _fmt_log(e, L)


def _vars_log(e):
    return sorted(e.free_symbols, key=str)


def _valor_bool(e, asig):
    v = e.subs(asig)
    return bool(v == True)  # noqa


def tabla_verdad(arg):
    s = _limpiar(arg)
    if not s:
        raise ErrorEntrada('Uso: tabla de verdad p -> (q y r)')
    textos = [p for p in dividir(s.replace('≡', ','), ',') if p.strip()]
    formulas, subs = [], []
    for tx in textos:
        e, sub = leer_logica(tx, 'logica')
        formulas.append(e)
        for q_ in sub:
            if q_ not in subs:
                subs.append(q_)
    for f_ in formulas:
        if f_ not in subs and not f_.is_Atom:
            subs.append(f_)
    vs = sorted(set().union(*[f_.free_symbols for f_ in formulas]), key=str)
    if len(vs) > 6:
        raise ErrorEntrada('Con más de 6 variables la tabla es demasiado grande (más de 64 filas).')
    r = Resultado('Lógica', 'Tabla de verdad')
    r.entrada_latex = r',\quad '.join(_fmt_log(f_) for f_ in formulas)
    columnas = [str(v) for v in vs] + [_fmt_log(q_, False) for q_ in subs]
    filas, finales = [], [[] for _ in formulas]
    for valores in _producto([True, False], repeat=len(vs)):
        asig = dict(zip(vs, valores))
        fila = ['V' if b else 'F' for b in valores]
        for q_ in subs:
            fila.append('V' if _valor_bool(q_, asig) else 'F')
        for i, f_ in enumerate(formulas):
            finales[i].append(_valor_bool(f_, asig))
        filas.append(fila)
    r.tabla = {'columnas': columnas, 'filas': filas}
    r.paso('Variables', f'{len(vs)} variable(s): 2^{len(vs)} = {2 ** len(vs)} filas (V = verdadero, F = falso).',
           ', '.join(str(v) for v in vs), 0)
    r.paso('Columnas intermedias', 'Se evalúan las subfórmulas de adentro hacia afuera, respetando la precedencia '
           '¬, ∧, ∨, ⇒, ⇔.', r',\quad '.join(_fmt_log(q_) for q_ in subs[:12]), 0)
    col = finales[0]
    clas = 'tautología' if all(col) else ('contradicción' if not any(col) else 'contingencia')
    r.paso('Clasificación', 'Tautología: siempre V. Contradicción: siempre F. Contingencia: a veces V y a veces F.',
           r'\text{' + clas + '}', 0)
    try:
        simp = sp.simplify_logic(formulas[0], form='dnf', force=True)
        r.extra('Forma simplificada equivalente', _fmt_log(simp))
    except Exception:  # noqa
        pass
    if len(formulas) == 2:
        eqv = finales[0] == finales[1]
        r.paso('Equivalencia', 'Dos fórmulas son equivalentes si sus columnas finales coinciden.',
               _fmt_log(formulas[0]) + (r'\equiv ' if eqv else r'\not\equiv ') + _fmt_log(formulas[1]), 0)
        r.valores['equivalentes'] = 'si' if eqv else 'no'
    r.valor = clas
    r.valores['clasificacion'] = clas
    r.valores['columna'] = ''.join('V' if b else 'F' for b in col)
    r.resultado_latex = r'\text{' + clas + '}' + (r'\ (' + _fmt_log(formulas[0]) + ')' if len(r.entrada_latex) < 60 else '')
    f0 = formulas[0]

    def chk(texto):
        t_ = sin_tildes(texto.lower()).strip()
        for nombre, pals in (('tautologia', ['tautolog']), ('contradiccion', ['contradic']), ('contingencia', ['contingen'])):
            if any(p in t_ for p in pals):
                return _ok() if sin_tildes(clas).startswith(nombre) else _mal('No es ' + nombre + '.', clas)
        sec = re.sub(r'[\s,;]', '', t_.upper())
        if sec and re.fullmatch(r'[VF10T]+', sec) and len(sec) == len(col):
            alumno = [ch in 'V1T' for ch in sec]
            if alumno == col:
                return _ok('¡Correcto! La columna final coincide.')
            malas = [str(i + 1) for i, (a_, b_) in enumerate(zip(alumno, col)) if a_ != b_]
            return _mal('Revisá las filas ' + ', '.join(malas) + '.', r.valores['columna'])
        e, _ = leer_logica(texto, 'logica')
        if sp.satisfiable(Xor(e, f0)) is False:
            return _ok('¡Correcto! Es una fórmula equivalente.')
        return _mal('No es equivalente.', _fmt_log(f0))
    _propio(r, clas, chk)
    return r


def booleana(arg):
    s = _limpiar(arg)
    s = re.sub(r'^\s*[A-Za-z]\s*(\([^)]*\))?\s*=\s*', '', s)
    r = Resultado('Lógica', 'Simplificación booleana')
    m = re.match(r'^\s*(?:Σ|∑|sum|sigma)?\s*m\s*\(([\d,\s]+)\)\s*(?:\+?\s*d\s*\(([\d,\s]+)\))?\s*$', s, flags=re.I)
    if m:
        mins = sorted({int(v) for v in m.group(1).split(',') if v.strip()})
        dc = sorted({int(v) for v in (m.group(2) or '').split(',') if v.strip()})
        nv = max(2, (max(mins + dc + [1])).bit_length())
        vs = list(sp.symbols('a b c d e f g h')[:nv])
        e = sp.SOPform(vs, [list(map(int, format(k, f'0{nv}b'))) for k in mins],
                       [list(map(int, format(k, f'0{nv}b'))) for k in dc]) if mins else sp.false
        r.entrada_latex = r'\sum m(' + ', '.join(map(str, mins)) + ')' + (r' + d(' + ', '.join(map(str, dc)) + ')' if dc else '')
        r.paso('Minitérminos', f'{nv} variables ({", ".join(map(str, vs))}); cada número en binario es una fila donde F = 1.',
               r',\ '.join(f'm_{{{k}}} = {format(k, f"0{nv}b")}' for k in mins), 0)
        original = e
    else:
        original, _ = leer_logica(s, 'boole')
        vs = _vars_log(original)
        if len(vs) > 8:
            raise ErrorEntrada('Demasiadas variables (máximo 8).')
        r.entrada_latex = 'F = ' + _fmt_boole(original, True)
        r.paso('Lectura', 'Producto (o letras juntas) = Y lógico; + = O lógico; \' = complemento.',
               _fmt_log(original), 0)
        mins = [k for k, val in enumerate(_producto([False, True], repeat=len(vs))) if _valor_bool(original, dict(zip(vs, val)))]
        dc = []
        r.paso('Tabla de verdad: minitérminos', 'Filas (en binario, con ' + ', '.join(map(str, vs)) + ') donde F = 1.',
               r'F = \sum m(' + ', '.join(map(str, mins)) + ')' if mins else r'F = 0', 0)
        filas = []
        for k, val in enumerate(_producto([0, 1], repeat=len(vs))):
            if k >= 64:
                break
            filas.append([str(k)] + [str(b) for b in val] + ['1' if k in mins else '0'])
        r.tabla = {'columnas': ['m'] + [str(v) for v in vs] + ['F'], 'filas': filas}
    sop = sp.simplify_logic(original, form='dnf', force=True) if not dc else original
    pos = sp.simplify_logic(original, form='cnf', force=True) if not dc else sp.POSform(
        vs, [list(map(int, format(k, f'0{len(vs)}b'))) for k in mins], [list(map(int, format(k, f'0{len(vs)}b'))) for k in dc])
    if not mins:
        sop = pos = sp.false
    r.paso('Simplificación (Quine–McCluskey / Karnaugh)', 'Se agrupan minitérminos adyacentes (que difieren en una sola variable): '
           'en cada grupo desaparece la variable que cambia.', 'F = ' + _fmt_boole(sop, True), 0)
    if isinstance(sop, Or):
        for term in sop.args:
            cub = [k for k in mins if _valor_bool(term, dict(zip(vs, [bool(int(b)) for b in format(k, f'0{len(vs)}b')])))]
            r.paso('Implicante', '', _fmt_boole(term, True) + r'\ \text{cubre } ' + ', '.join(f'm_{{{k}}}' for k in cub), 1)
    r.extra('Producto de sumas (forma mínima)', 'F = ' + _fmt_boole(pos, True))
    r.valor = sp.sstr(sop)
    r.valores['sop'] = _fmt_boole(sop)
    r.valores['pos'] = _fmt_boole(pos)
    r.valores['minterminos'] = mins
    r.resultado_latex = 'F = ' + _fmt_boole(sop, True)

    def literales(e):
        return len(re.findall(r'[A-Za-z]\d*', _fmt_boole(e)))

    def chk(texto):
        tt_ = re.sub(r'^\s*F\s*=\s*', '', texto.strip())
        e, _ = leer_logica(tt_, 'boole')
        ref = sop if not dc else original
        if sp.satisfiable(Xor(e, ref)) is not False:
            return _mal('No es equivalente a la función dada.', _fmt_boole(sop, True))
        if literales(e) > literales(sop):
            return _ok('Es equivalente, pero se puede simplificar más: ' + _fmt_boole(sop), True)
        return _ok()
    _propio(r, r.valor, chk)
    return r


# ---------------------------------------------------------------------------
# Combinatoria
# ---------------------------------------------------------------------------
def _fact_txt(n_):
    if n_ <= 1:
        return '1'
    if n_ <= 8:
        return r'\cdot '.join(str(i) for i in range(n_, 0, -1))
    return str(n_) + '!'


def combinatoria(arg, tipo):
    s = _limpiar(arg)
    if tipo == 'PR' and re.fullmatch(r'\s*[A-Za-zÁÉÍÓÚáéíóúÑñ]+\s*', s):
        palabra = sin_tildes(s.strip().upper())
        cuenta = {}
        for ch in palabra:
            cuenta[ch] = cuenta.get(ch, 0) + 1
        ns = [len(palabra)] + list(cuenta.values())
        etiqueta = f'Anagramas de {palabra}: ' + ', '.join(f'{k}×{v}' for k, v in cuenta.items())
    else:
        ns = [_entero(p, 'cada dato') for p in dividir(s.replace(';', ','), ',')]
        etiqueta = ''
    if any(v < 0 for v in ns):
        raise ErrorEntrada('Los datos deben ser enteros no negativos.')
    r = Resultado(TEMA_DIS, 'Combinatoria')
    if tipo == 'P' and len(ns) == 2:
        tipo = 'V'
        r.avisos.append('Con dos datos se interpreta como variaciones (permutaciones de n tomadas de a k).')
    if tipo in ('C', 'CR', 'V', 'VR', 'todo') and len(ns) != 2:
        raise ErrorEntrada('Indicá n y k separados por coma, por ejemplo: combinaciones 10, 3')
    if tipo in ('P', 'PC') and len(ns) != 1:
        raise ErrorEntrada('Indicá solo n, por ejemplo: permutaciones 5')
    if tipo == 'todo':
        n_, k_ = ns
        filas = [['Variaciones (sin rep.)', 'V(n,k) = n!/(n−k)!', str(sp.ff(n_, k_)) if k_ <= n_ else '0'],
                 ['Variaciones con repetición', 'nᵏ', str(n_ ** k_)],
                 ['Combinaciones (sin rep.)', 'C(n,k) = n!/(k!(n−k)!)', str(sp.binomial(n_, k_))],
                 ['Combinaciones con repetición', 'C(n+k−1, k)', str(sp.binomial(n_ + k_ - 1, k_))],
                 ['Permutaciones de n', 'n!', str(sp.factorial(n_))]]
        r.tabla = {'columnas': ['Tipo', 'Fórmula', 'Valor'], 'filas': filas}
        r.entrada_latex = f'n = {n_},\\ k = {k_}'
        r.paso('¿Cuál usar?', 'Importa el orden → variaciones; no importa → combinaciones. Se pueden repetir elementos → '
               '«con repetición». Se usan todos los elementos → permutaciones.', '', 0)
        val = sp.binomial(n_, k_)
        r.valor = val
        r.resultado_latex = r'\binom{' + str(n_) + '}{' + str(k_) + '} = ' + str(val)
        r.comparar = ('numero_o_expr', val, {})
        return r
    if tipo == 'C':
        n_, k_ = ns
        r.titulo = 'Combinaciones sin repetición'
        val = sp.binomial(n_, k_)
        r.entrada_latex = r'C_{' + str(n_) + ',' + str(k_) + r'} = \binom{' + str(n_) + '}{' + str(k_) + '}'
        r.paso('Cuándo', 'No importa el orden y no se repiten elementos.', r'C_{n,k} = \binom{n}{k} = \frac{n!}{k!\,(n-k)!}', 0)
        if k_ <= n_:
            r.paso('Cálculo', 'Se simplifica (n−k)! con el factorial de arriba.', r'\frac{' + str(n_) + '!}{' + str(k_) + r'!\,'
                   + str(n_ - k_) + '!} = ' + r'\frac{' + (r'\cdot '.join(str(i) for i in range(n_, n_ - k_, -1)) or '1') + '}{'
                   + _fact_txt(k_) + '} = ' + str(val), 0)
    elif tipo == 'CR':
        n_, k_ = ns
        r.titulo = 'Combinaciones con repetición'
        val = sp.binomial(n_ + k_ - 1, k_)
        r.entrada_latex = r"C'_{" + str(n_) + ',' + str(k_) + '}'
        r.paso('Cuándo', 'No importa el orden y los elementos se pueden repetir.', r"C'_{n,k} = \binom{n+k-1}{k}", 0)
        r.paso('Cálculo', '', r'\binom{' + str(n_ + k_ - 1) + '}{' + str(k_) + '} = ' + str(val), 0)
    elif tipo == 'V':
        n_, k_ = ns
        r.titulo = 'Variaciones sin repetición'
        val = sp.ff(n_, k_) if k_ <= n_ else sp.Integer(0)
        r.entrada_latex = r'V_{' + str(n_) + ',' + str(k_) + '}'
        r.paso('Cuándo', 'Importa el orden y no se repiten elementos.', r'V_{n,k} = \frac{n!}{(n-k)!} = n(n-1)\cdots(n-k+1)', 0)
        if k_ <= n_:
            r.paso('Cálculo', f'Producto de {k_} factores decrecientes desde {n_}.',
                   (r'\cdot '.join(str(i) for i in range(n_, n_ - k_, -1)) or '1') + ' = ' + str(val), 0)
    elif tipo == 'VR':
        n_, k_ = ns
        r.titulo = 'Variaciones con repetición'
        val = sp.Integer(n_) ** k_
        r.entrada_latex = r"V'_{" + str(n_) + ',' + str(k_) + '}'
        r.paso('Cuándo', 'Importa el orden y se pueden repetir elementos.', r"V'_{n,k} = n^k", 0)
        r.paso('Cálculo', '', f'{n_}^{{{k_}}} = {val}', 0)
    elif tipo == 'P':
        n_ = ns[0]
        r.titulo = 'Permutaciones sin repetición'
        val = sp.factorial(n_)
        r.entrada_latex = f'P_{{{n_}}} = {n_}!'
        r.paso('Cuándo', 'Se ordenan todos los elementos (distintos).', r'P_n = n!', 0)
        r.paso('Cálculo', '', _fact_txt(n_) + ' = ' + str(val), 0)
    elif tipo == 'PC':
        n_ = ns[0]
        r.titulo = 'Permutaciones circulares'
        val = sp.factorial(max(n_ - 1, 0))
        r.entrada_latex = f'PC_{{{n_}}} = ({n_}-1)!'
        r.paso('Cuándo', 'Se ordenan en círculo: las rotaciones son la misma disposición.', r'PC_n = (n-1)!', 0)
        r.paso('Cálculo', '', f'{n_ - 1}! = {val}', 0)
    else:  # PR
        if len(ns) < 2:
            raise ErrorEntrada('Uso: permutaciones con repeticion n, a, b, c   (n total y cuántas veces se repite cada elemento)')
        if sum(ns[1:]) == ns[0]:
            n_, reps = ns[0], ns[1:]
        else:
            n_, reps = sum(ns), ns
        val = sp.factorial(n_) / reduce(lambda a, b: a * b, [sp.factorial(v) for v in reps], 1)
        r.titulo = 'Permutaciones con repetición'
        r.entrada_latex = f'P_{{{n_}}}^{{' + ','.join(map(str, reps)) + '}'
        r.paso('Cuándo', 'Se ordenan n elementos donde algunos son iguales.' + (' ' + etiqueta if etiqueta else ''),
               r'P_n^{a,b,\dots} = \frac{n!}{a!\,b!\cdots}', 0)
        r.paso('Cálculo', '', r'\frac{' + str(n_) + '!}{' + r'\,'.join(f'{v}!' for v in reps) + '} = ' + str(val), 0)
    r.valor = sp.Integer(val)
    r.resultado_latex = r.entrada_latex.split('=')[0] + ' = ' + str(val)
    r.comparar = ('numero_o_expr', sp.Integer(val), {})
    return r


# ============================================================================
# Registro de comandos (api.py los prueba ANTES que los base)
# ============================================================================
def _otro_comando(texto, propia):
    """Busca un comando que coincida con el texto y esté DESPUÉS del propio en api.COMANDOS."""
    from . import api
    clave = re.sub(r'\s+', ' ', sin_tildes(texto.strip().lower()))
    despues = False
    for prefijos, f in api.COMANDOS:
        if f is propia:
            despues = True
            continue
        if not despues:
            continue
        for p in prefijos:
            if clave.startswith(p) and (len(clave) == len(p) or not clave[len(p)].isalnum()):
                resto = api._resto_original(texto.strip(), p).lstrip(' :')
                return lambda: f(resto)
    return None


def _con_respaldo(prefijo, fn):
    """Para prefijos genéricos ('imagen', 'distancia'...): si la entrada no es de este módulo, se delega al
    siguiente comando que coincida (p. ej. de otro módulo de extensión) en lugar de taparlo."""
    def envoltura(arg):
        try:
            return fn(arg)
        except ErrorEntrada:
            otro = _otro_comando(prefijo + ' ' + arg, envoltura)
            if otro is None:
                raise
            return otro()
    return envoltura


COMANDOS_EXTRA = [
    # Geometría analítica
    (('recta que pasa por', 'recta por los puntos', 'recta por', 'recta entre los puntos'), recta_por),
    (('recta pendiente punto', 'recta punto pendiente', 'recta de pendiente', 'recta con pendiente', 'recta pendiente'),
     recta_pendiente),
    (('recta paralela',), lambda a: recta_relativa(a, 'paralela')),
    (('recta perpendicular',), lambda a: recta_relativa(a, 'perpendicular')),
    (('ecuacion de la recta', 'ecuacion recta'), recta_auto),
    (('plano que pasa por', 'plano por', 'ecuacion del plano', 'plano punto normal'), plano_por),
    (('distancia entre', 'distancia punto', 'distancia recta', 'distancia plano', 'distancia del', 'distancia de',
      'distancia'), _con_respaldo('distancia', distancia)),
    (('angulo entre', 'angulo recta', 'angulo plano'), angulo_entre),
    (('posicion relativa', 'posiciones relativas'), posicion_relativa),
    (('interseccion de planos', 'interseccion entre planos', 'interseccion de los planos', 'interseccion de tres planos',
      'interseccion de rectas', 'interseccion entre rectas', 'interseccion de las rectas', 'interseccion recta plano',
      'interseccion de recta y plano', 'interseccion entre recta y plano', 'interseccion de la recta y el plano',
      'interseccion plano recta'), interseccion),
    (('identificar conica', 'elementos de la conica', 'seccion conica', 'conicas', 'conica'), conica),
    (('superficie cuadrica', 'identificar cuadrica', 'cuadricas', 'cuadrica'), cuadrica),
    # Álgebra lineal
    (('producto mixto', 'triple producto escalar', 'producto triple'), producto_mixto),
    (('nucleo', 'kernel', 'ker'), _con_respaldo('nucleo', nucleo)),
    (('imagen',), _con_respaldo('imagen', imagen)),
    (('independencia lineal', 'dependencia lineal', 'independencia', 'son linealmente independientes',
      'linealmente independientes'), independencia),
    (('base y dimension', 'dimension y base', 'base del subespacio', 'dimension del subespacio', 'base y dim'), base_dimension),
    (('diagonalizar', 'diagonalizacion', 'diagonalizable'), diagonalizar),
    (('regla de cramer', 'metodo de cramer', 'cramer'), cramer),
    (('division de polinomios', 'dividir polinomios', 'division polinomica', 'division entera', 'division', 'dividir'),
     _con_respaldo('division', division)),
    (('regla de ruffini', 'ruffini'), ruffini),
    (('vector de coordenadas', 'coordenadas'), _con_respaldo('coordenadas', coordenadas)),
    (('matriz de cambio de base', 'cambio de base', 'matriz de pasaje', 'matriz de transicion'), cambio_base),
    # Discreta
    (('descomponer en factores primos', 'descomposicion en factores primos', 'factores primos', 'factorizacion prima',
      'descomponer en primos', 'factorizar entero', 'descomposicion factorial'), factores_primos),
    (('divisores de', 'divisores'), divisores),
    (('es primo', 'primalidad'), es_primo),
    (('potencia modular', 'exponenciacion modular', 'resto de', 'modulo', 'mod'), modulo),
    (('sistema de congruencias', 'teorema chino del resto', 'resto chino', 'resolver congruencia', 'congruencias',
      'congruencia'), congruencia),
    (('algoritmo de euclides', 'euclides extendido', 'euclides', 'mcd extendido', 'identidad de bezout', 'bezout'), euclides),
    (('ecuacion diofantica', 'ecuacion diofantina', 'diofantica', 'diofantina'), diofantica),
    (('tabla de verdad', 'tabla de la verdad', 'tabla verdad', 'es tautologia', 'tautologia', 'equivalencia logica'), tabla_verdad),
    (('simplificar funcion booleana', 'simplificar expresion booleana', 'simplificar booleana', 'simplificar logica',
      'simplificar boole', 'algebra de boole', 'mapa de karnaugh', 'karnaugh', 'booleana', 'minterminos'), booleana),
    (('combinaciones con repeticion',), lambda a: combinatoria(a, 'CR')),
    (('combinaciones sin repeticion', 'combinaciones', 'combinacion', 'numero combinatorio'), lambda a: combinatoria(a, 'C')),
    (('variaciones con repeticion',), lambda a: combinatoria(a, 'VR')),
    (('variaciones sin repeticion', 'variaciones', 'variacion'), lambda a: combinatoria(a, 'V')),
    (('permutaciones con repeticion', 'anagramas de', 'anagramas'), lambda a: combinatoria(a, 'PR')),
    (('permutaciones circulares',), lambda a: combinatoria(a, 'PC')),
    (('permutaciones sin repeticion', 'permutaciones', 'permutacion'), lambda a: combinatoria(a, 'P')),
    (('combinatoria',), lambda a: combinatoria(a, 'todo')),
]
