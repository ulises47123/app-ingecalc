"""Comprobación de la respuesta del alumno contra la resolución del motor."""
import random
import re

import sympy as sp

from .nucleo import ErrorEntrada, coma_decimal, dividir, leer, leer_ecuacion, normalizar, sin_tildes, tex, x


def _decimales(texto):
    ds = [len(m) for m in re.findall(r'\d\.(\d+)', texto)]
    return max(ds) if ds else None


def _quitar_prefijo(texto):
    """'y = 2x' -> '2x' ; "f'(x) = ..." -> '...' ; 'x = 3' se mantiene para conjuntos."""
    t_ = texto.strip()
    m = re.match(r"^\s*(y|f|F|f'|f''|y'|dy/dx|P\(x\)|P|z|A|I|r|S|y\(x\)|y\(t\)|f\(x\)|F\(s\)|f\(t\)|F\(x\)|f'\(x\)|L|R|Req|R_eq)\s*=\s*(.+)$", t_)
    if m:
        return m.group(2)
    return t_


def _num_iguales(a, b, vars_, tol=1e-8, intentos=8):
    """Compara numéricamente en puntos al azar (respaldo de simplify)."""
    vars_ = list(vars_)
    try:
        fa = sp.lambdify(vars_, a, 'mpmath')
        fb = sp.lambdify(vars_, b, 'mpmath')
    except Exception:  # noqa
        return None
    ok = 0
    for _ in range(intentos * 4):
        pts = [random.uniform(0.3, 2.3) for _ in vars_]
        try:
            va, vb = complex(fa(*pts)), complex(fb(*pts))
        except Exception:  # noqa
            continue
        if abs(va - vb) > tol * (1 + abs(vb)):
            return False
        ok += 1
        if ok >= intentos:
            return True
    return None if ok == 0 else True


def equivalentes(a, b, tol=1e-9):
    a, b = sp.sympify(a), sp.sympify(b)
    if a == b:
        return True
    d = a - b
    try:
        if sp.simplify(d) == 0:
            return True
    except Exception:  # noqa
        pass
    try:
        if sp.expand(sp.expand_log(sp.expand_trig(d), force=True)) == 0:
            return True
    except Exception:  # noqa
        pass
    vs = d.free_symbols
    if not vs:
        try:
            return abs(complex(sp.N(d, 20))) < tol * (1 + abs(complex(sp.N(b, 20))))
        except Exception:  # noqa
            return False
    r = _num_iguales(a, b, vs, tol=max(tol, 1e-9))
    return bool(r)


def _diagnostico_expr(alumno, ref):
    """Pistas típicas: signo, constante sumada, factor multiplicativo."""
    try:
        if equivalentes(alumno, -ref):
            return 'Tiene el signo cambiado. Revisá un signo menos (por ejemplo la derivada del coseno o una resta).'
        dif = sp.simplify(alumno - ref)
        if dif.is_number and dif != 0:
            return f'Difiere en una constante ({tex(dif)}).'
        coc = sp.simplify(alumno / ref)
        if coc.is_number and coc != 0:
            return (f'Es la respuesta correcta multiplicada por {tex(coc)}. Puede faltar un factor de la regla de la '
                    'cadena o una constante.')
        vs = (alumno - ref).free_symbols
        if len(vs) == 1:
            v = list(vs)[0]
            for prueba in (sp.Rational(1, 2), 1, 2):
                va, vr = sp.N(alumno.subs(v, prueba)), sp.N(ref.subs(v, prueba))
                if va.is_number and vr.is_number:
                    return f'Por ejemplo en {v} = {prueba}: tu respuesta da {va:.6g} y debería dar {vr:.6g}.'
    except Exception:  # noqa
        pass
    return ''


def _leer_respuesta(texto):
    return leer(_quitar_prefijo(texto))


def _a_lista(texto):
    """'2, 3' | '[2, 3]' | 'x=2, x=3' | 'x=2 o x=3' | '{2,3}' -> lista de expresiones"""
    t_ = sin_tildes(normalizar(texto)).strip()
    t_ = re.sub(r'\s+(o|y|u)\s+', ', ', t_)
    t_ = t_.strip()
    if t_.startswith('{') and t_.endswith('}') and ':' not in t_:
        t_ = t_[1:-1]
    if t_.startswith('[') and t_.endswith(']') and not t_.startswith('[['):
        t_ = t_[1:-1]
    partes = dividir(t_, ',') if ';' not in t_ else dividir(t_, ';')
    vals = []
    for p in partes:
        p = re.sub(r'^\s*[a-z]\s*(_?\d)?\s*=\s*', '', p)
        if p.strip() in ('', 'no tiene', 'ninguna', 'vacio', 'ø', '∅'):
            continue
        vals.append(leer(p))
    return vals


def _leer_dict(texto):
    t_ = normalizar(texto).strip()
    if t_.startswith('{') and t_.endswith('}'):
        t_ = t_[1:-1]
    d = {}
    for p in dividir(t_.replace(';', ','), ','):
        if '=' in p:
            a, b = p.split('=', 1)
        elif ':' in p:
            a, b = p.split(':', 1)
        else:
            continue
        d[leer(a)] = leer(b)
    return d


def _leer_intervalo(texto, var):
    t_ = normalizar(texto).replace('∪', ' U ').replace(' u ', ' U ').strip()
    t_ = re.sub(r'^\s*[a-z]\s*(∈|en|in)\s*', '', t_)
    if re.search(r'[<>]', t_):
        rel = leer_ecuacion(t_)
        vr = sp.Symbol(str(var), real=True)
        if isinstance(rel, sp.Or):
            return sp.Union(*[sp.solveset(a.subs(var, vr), vr, sp.S.Reals) for a in rel.args])
        if isinstance(rel, sp.And):
            return sp.Intersection(*[sp.solveset(a.subs(var, vr), vr, sp.S.Reals) for a in rel.args])
        return sp.solveset(rel.subs(var, vr), vr, sp.S.Reals)
    if 'Interval' in t_ or 'Union' in t_:
        return leer(t_)
    if t_ in ('R', 'reales', 'todos los reales'):
        return sp.S.Reals
    piezas = [p.strip() for p in re.split(r'\s+U\s+|\bU\b', t_) if p.strip()]
    conj = []
    for p in piezas:
        m = re.match(r'^([\(\[])\s*(.+?)\s*,\s*(.+?)\s*([\)\]])$', p)
        if not m:
            raise ErrorEntrada(f'No entendí el intervalo «{p}». Escribí por ejemplo (-2, 2) o [1, oo).')
        a, b = leer(m.group(2)), leer(m.group(3))
        conj.append(sp.Interval(a, b, m.group(1) == '(', m.group(4) == ')'))
    return sp.Union(*conj)


def comprobar(res, respuesta):
    """res: Resultado con .comparar = (modo, referencia, ctx)."""
    if not res.comparar:
        return {'correcto': None, 'mensaje': 'Este tipo de ejercicio no tiene un resultado único para comparar.'}
    modo, ref, ctx = res.comparar
    if callable((ctx or {}).get('comprobar')):
        # comparadores propios de un módulo (rectas en otra forma, bases equivalentes, congruencias...)
        try:
            return ctx['comprobar'](coma_decimal(str(respuesta).strip()))
        except Exception as e:  # noqa
            return {'correcto': None, 'mensaje': 'No pude interpretar tu respuesta: ' + str(e)[:120]}
    texto = coma_decimal(str(respuesta).strip())
    if not texto:
        return {'correcto': None, 'mensaje': 'Escribí tu respuesta para compararla.'}
    dec = _decimales(texto)
    try:
        if modo in ('expr', 'expr_forma', 'numero_o_expr', 'numero', 'expr_num'):
            alumno = _leer_respuesta(texto)
            if isinstance(ref, (list, tuple)):
                return comprobar_lista(alumno, ref)
            tol = ctx.get('tol', 1e-9)
            if dec is not None and sp.sympify(ref).is_number:
                return _comparar_decimal(alumno, ref, dec)
            if modo in ('numero', 'expr_num') and sp.sympify(ref).is_number:
                ok = abs(complex(sp.N(alumno - ref))) <= max(tol, 1e-9) * (1 + abs(complex(sp.N(ref))))
            elif modo == 'expr_num':
                ok = equivalentes(alumno, ref, tol=1e-4)
            else:
                ok = equivalentes(alumno, ref)
            if ok:
                msg = '¡Correcto!'
                if modo == 'expr_forma':
                    forma = ctx.get('forma')
                    if forma == 'factorizada' and sp.factor(alumno) != alumno and not alumno.is_Mul and not alumno.is_Pow:
                        msg = 'Es equivalente, pero no está factorizada del todo: ' + tex(ref)
                        return {'correcto': True, 'mensaje': msg, 'parcial': True}
                    if forma == 'expandida' and sp.expand(alumno) != alumno:
                        return {'correcto': True, 'mensaje': 'Es equivalente, pero falta expandir: ' + tex(ref), 'parcial': True}
                return {'correcto': True, 'mensaje': msg}
            return {'correcto': False, 'mensaje': 'No coincide con el resultado.',
                    'pista': _diagnostico_expr(alumno, sp.sympify(ref)), 'esperado': tex(ref)}
        if modo == 'antiderivada':
            alumno = _leer_respuesta(texto)
            alumno = alumno.subs({s_: 0 for s_ in alumno.free_symbols if str(s_) in ('C', 'c', 'C1', 'k_')})
            v = ctx['var']
            f = ctx['integrando']
            d = sp.diff(alumno, v)
            if equivalentes(d, f):
                return {'correcto': True, 'mensaje': '¡Correcto! Al derivar tu respuesta se obtiene el integrando '
                        '(recordá sumar + C).'}
            pista = _diagnostico_expr(d, f)
            return {'correcto': False, 'mensaje': 'Al derivar tu respuesta no se obtiene el integrando.',
                    'pista': (r'La derivada de tu respuesta es \(' + tex(sp.simplify(d)) + r'\). ' + pista).strip(),
                    'esperado': tex(ref) + ' + C'}
        if modo == 'conjunto':
            alumno = _a_lista(texto)
            return comprobar_lista(alumno, ref, ctx.get('multiplicidad', False))
        if modo == 'conjunto_periodico':
            alumno = _a_lista(texto)
            princ = ctx.get('principales') or []
            if princ and comprobar_lista(alumno, princ)['correcto']:
                return {'correcto': True, 'mensaje': '¡Correcto! (soluciones en [0, 2π); la general agrega + 2nπ o + nπ).'}
            return {'correcto': False, 'mensaje': 'No coincide con las soluciones en [0, 2π).', 'esperado': tex(princ)}
        if modo == 'dict':
            alumno = _leer_dict(texto)
            faltan = [k_ for k_ in ref if k_ not in alumno]
            if faltan:
                return {'correcto': False, 'mensaje': 'Faltan valores para: ' + ', '.join(str(f) for f in faltan) + '.'}
            mal = [k_ for k_ in ref if not equivalentes(alumno[k_], ref[k_], 1e-9 if dec is None else 10 ** -dec)]
            if not mal:
                return {'correcto': True, 'mensaje': '¡Correcto!'}
            return {'correcto': False, 'mensaje': 'Revisá el valor de ' + ', '.join(str(m) for m in mal) + '.',
                    'esperado': ', '.join(f'{tex(k_)} = {tex(v)}' for k_, v in ref.items())}
        if modo == 'lista_dict':
            return {'correcto': None, 'mensaje': 'El sistema tiene varias soluciones: ' + res.resultado_latex}
        if modo == 'intervalo':
            alumno = _leer_intervalo(texto, ctx.get('var', x))
            if sp.simplify(sp.SymmetricDifference(alumno, ref)) == sp.EmptySet or alumno == ref:
                return {'correcto': True, 'mensaje': '¡Correcto!'}
            return {'correcto': False, 'mensaje': 'El conjunto solución no coincide. Revisá si los extremos van abiertos '
                    'o cerrados y los puntos donde se anula el denominador.', 'esperado': tex(ref)}
        if modo in ('matriz', 'vector'):
            alumno = _leer_respuesta(texto)
            A = sp.Matrix(alumno) if not isinstance(alumno, sp.MatrixBase) else alumno
            B = sp.Matrix(ref)
            if A.shape != B.shape:
                if A.shape == B.T.shape:
                    A = A.T
                else:
                    return {'correcto': False, 'mensaje': f'Las dimensiones no coinciden: {A.shape} vs {B.shape}.', 'esperado': tex(B)}
            if dec is None:
                malas = [(i, j) for i in range(B.rows) for j in range(B.cols) if not equivalentes(A[i, j], B[i, j])]
            else:  # con decimales: error absoluto acorde a los decimales que escribió el alumno
                def _cerca(a_, b_):
                    try:
                        return abs(complex(sp.N(a_ - b_))) <= 0.6 * 10 ** -dec + 1e-12
                    except Exception:  # noqa
                        return False
                malas = [(i, j) for i in range(B.rows) for j in range(B.cols) if not _cerca(A[i, j], B[i, j])]
            if not malas:
                return {'correcto': True, 'mensaje': '¡Correcto!'}
            return {'correcto': False, 'mensaje': 'Revisá ' + ('las componentes ' if modo == 'vector' else 'los elementos ')
                    + ', '.join(f'({i + 1},{j + 1})' if modo == 'matriz' else str(i + j + 1) for i, j in malas[:6]) + '.',
                    'esperado': tex(B)}
        if modo == 'edo':
            alumno = _leer_respuesta(texto)
            eq, ics, v, orden = ctx['eq'], ctx['ics'], ctx['v'], ctx['orden']
            Y = sp.Function('y')(v)
            try:
                ok = sp.checkodesol(eq, sp.Eq(Y, alumno))[0]
            except Exception:  # noqa
                ok = False
            if not ok:
                return {'correcto': False, 'mensaje': 'Tu función no satisface la ecuación diferencial.',
                        'pista': 'Reemplazá y, y\' e y\'\' en la ecuación para ver qué término falla.', 'esperado': 'y = ' + tex(ref)}
            consts = [c for c in alumno.free_symbols if str(c).upper().startswith('C') or str(c) in ('A', 'B', 'K', 'k')]
            if ics:
                for k_, b in ics.items():
                    if isinstance(k_, sp.Subs) or k_.has(sp.Derivative):
                        pto = list(k_.point)[0] if isinstance(k_, sp.Subs) else k_.args[0]
                        d_ord = k_.expr.derivative_count if hasattr(k_, 'expr') else 1
                        val = sp.diff(alumno, v, d_ord).subs(v, pto)
                    else:
                        val = alumno.subs(v, k_.args[0])
                    if not equivalentes(val, b):
                        return {'correcto': False, 'mensaje': 'Satisface la ecuación pero no la condición inicial '
                                + tex(k_) + ' = ' + tex(b) + '.', 'esperado': 'y = ' + tex(ref)}
                return {'correcto': True, 'mensaje': '¡Correcto! Satisface la ecuación y las condiciones iniciales.'}
            if len(consts) < orden:
                return {'correcto': False, 'mensaje': f'Satisface la ecuación, pero una solución general de orden {orden} '
                        f'necesita {orden} constante(s) arbitraria(s).', 'esperado': 'y = ' + tex(ref)}
            return {'correcto': True, 'mensaje': '¡Correcto! Es solución general de la ecuación.'}
        if modo == 'convergencia':
            resp = sin_tildes(texto.lower()).strip()
            refn = sin_tildes(str(ref).lower())
            conv = resp.startswith('conv') or resp == 'c'
            div = resp.startswith('diverg') or 'no converge' in resp
            ok = (conv and refn.startswith('converge')) or (div and refn.startswith('diverge'))
            for tipo in ('absolutamente', 'condicionalmente'):
                if tipo in resp and tipo not in refn:
                    ok = False
            return {'correcto': ok, 'mensaje': '¡Correcto!' if ok else 'La serie ' + str(ref) + '.'}
        if modo == 'texto':
            ok = sin_tildes(texto.lower()).strip() in ('no existe', 'ne', 'no tiene', 'diverge', 'no hay')
            return {'correcto': ok, 'mensaje': '¡Correcto!' if ok else 'El resultado es: ' + str(ref)}
        if modo == 'autovectores':
            A = ctx['A']
            vec = sp.Matrix(_leer_respuesta(texto))
            for val, mult, base in ref:
                if (A * vec - val * vec).applyfunc(sp.simplify) == sp.zeros(*vec.shape) and any(c != 0 for c in vec):
                    return {'correcto': True, 'mensaje': f'¡Correcto! Es autovector de λ = {val}.'}
            return {'correcto': False, 'mensaje': 'No es autovector de A (A·v no es múltiplo de v).'}
    except ErrorEntrada as e:
        return {'correcto': None, 'mensaje': str(e)}
    except Exception as e:  # noqa
        return {'correcto': None, 'mensaje': 'No pude interpretar tu respuesta: ' + str(e)[:120]}
    return {'correcto': None, 'mensaje': 'No se puede comparar este resultado.'}


def _comparar_decimal(alumno, ref, dec):
    try:
        va, vr = complex(sp.N(alumno, 20)), complex(sp.N(ref, 20))
    except Exception:  # noqa
        return {'correcto': False, 'mensaje': 'No pude evaluar tu respuesta.'}
    err = abs(va - vr)
    if err <= 0.6 * 10 ** -dec + 1e-12 or err <= 1e-3 * abs(vr):
        return {'correcto': True, 'mensaje': f'¡Correcto! (aproximado a {dec} decimales; valor exacto: \\({tex(ref)}\\))'}
    return {'correcto': False, 'mensaje': f'No coincide: el valor es aproximadamente {sp.N(ref, 10)}.', 'esperado': tex(ref)}


def comprobar_lista(alumno, ref, multiplicidad=False):
    if not isinstance(alumno, (list, tuple)):
        alumno = [alumno]
    if isinstance(alumno, (list, tuple)) and len(alumno) == 1 and isinstance(alumno[0], (sp.Tuple, list, tuple)) \
            and ref and not isinstance(ref[0], (sp.Tuple, list, tuple)):
        alumno = list(alumno[0])
    ref = list(ref)
    pendientes = list(ref)
    sobran = []
    for a in alumno:
        hallado = None
        for i, b in enumerate(pendientes):
            try:
                if isinstance(b, (sp.Tuple, tuple, list)):
                    if isinstance(a, (sp.Tuple, tuple, list)) and len(a) == len(b) and all(equivalentes(p, q) for p, q in zip(a, b)):
                        hallado = i
                        break
                elif equivalentes(a, b, 1e-8):
                    hallado = i
                    break
            except Exception:  # noqa
                continue
        if hallado is None:
            if not multiplicidad and any(equivalentes(a, b) for b in ref if not isinstance(b, (sp.Tuple, tuple, list))):
                continue  # repetida
            sobran.append(a)
        else:
            pendientes.pop(hallado)
    if not pendientes and not sobran:
        return {'correcto': True, 'mensaje': '¡Correcto!'}
    msg = []
    if pendientes:
        msg.append(f'Te falta(n) {len(pendientes)} solución(es).')
    if sobran:
        msg.append('No son solución: ' + ', '.join(tex(s_) for s_ in sobran) + '.')
    return {'correcto': False, 'mensaje': ' '.join(msg), 'esperado': tex(ref),
            'pista': 'Revisá si descartaste raíces (por ejemplo al dividir por x) o si alguna anula un denominador.'}
