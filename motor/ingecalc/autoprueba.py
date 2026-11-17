"""Corre el banco de ejercicios (pruebas/banco_500.json) contra el motor y compara con lo esperado.
Se usa igual en la PC (pruebas/correr_banco.py) y en el celular (Autoprueba de la app)."""
import json
import re
import time

import sympy as sp

from .api import resolver
from .comprobar import equivalentes

_LOC = {nombre: sp.Symbol(nombre) for nombre in ('x', 'y', 'z', 't', 's', 'n', 'k', 'C1', 'C2', 'C3', 'lamda')}
_LOC.update({'E': sp.E, 'I': sp.I, 'pi': sp.pi, 'oo': sp.oo, 'Interval': sp.Interval, 'Union': sp.Union,
             'FiniteSet': sp.FiniteSet, 'EmptySet': sp.EmptySet, 'Reals': sp.S.Reals})


def S(texto):
    if isinstance(texto, (int, float)):
        return sp.nsimplify(texto) if isinstance(texto, int) else sp.Float(texto)
    if isinstance(texto, (list, tuple)):
        return [S(a) for a in texto]
    if isinstance(texto, dict):
        return {S(k): S(v) for k, v in texto.items()}
    return sp.sympify(texto, locals=_LOC)


def _num(a, b, tol):
    try:
        va, vb = complex(sp.N(a, 20)), complex(sp.N(b, 20))
    except Exception:  # noqa
        return False
    return abs(va - vb) <= tol * max(1.0, abs(vb))


def _iguales(a, b, tol=None):
    if isinstance(a, (list, tuple, sp.Tuple)) or isinstance(b, (list, tuple, sp.Tuple)):
        a, b = list(a), list(b)
        return len(a) == len(b) and all(_iguales(p, q, tol) for p, q in zip(a, b))
    if tol is not None and (sp.sympify(b).is_number):
        return _num(a, b, tol)
    return equivalentes(a, b, 1e-9)


def _multiset(a, b, tol=None):
    a, b = list(a), list(b)
    if len(a) != len(b):
        return False
    resto = list(b)
    for p in a:
        for i, q in enumerate(resto):
            try:
                if _iguales(p, q, tol):
                    resto.pop(i)
                    break
            except Exception:  # noqa
                continue
        else:
            return False
    return True


def comparar(caso, d):
    """True/False y motivo."""
    if not d.get('ok'):
        return False, 'error: ' + str(d.get('error'))
    modo = caso['modo']
    esp = caso['esperado']
    tol = caso.get('tol')
    try:
        if modo == 'grafico':
            tipo = (d.get('grafico') or {}).get('tipo')
            return tipo == esp, f'tipo {tipo}'
        if modo == 'clave':
            val = (d.get('valores') or {}).get(caso['clave'])
            if val is None:
                return False, f'falta valores[{caso["clave"]}]'
            v = S(val)
            e = S(esp)
            if isinstance(e, list):
                return _multiset(v if isinstance(v, (list, tuple, sp.Tuple)) else [v], e, tol), str(val)
            return (_num(v, e, tol) if tol else _iguales(v, e)), str(val)
        valor = d.get('valor')
        if valor is None:
            return False, 'sin valor'
        v = S(valor) if not isinstance(valor, str) or valor != 'no existe' else valor
        if isinstance(v, str):
            return str(esp).strip().lower() == v, v
        e = S(esp) if not (isinstance(esp, str) and esp.strip().lower() in ('no existe',)) else 'no existe'
        if isinstance(e, str):
            return False, f'se esperaba "no existe", dio {valor}'
        if modo in ('expr',):
            if isinstance(e, list):
                return _iguales(v, e, tol), str(valor)
            return (_num(v, e, tol) if tol and e.is_number else equivalentes(v, e)), str(valor)
        if modo == 'numero':
            vv = v[0] if isinstance(v, list) and len(v) == 1 else v
            return _num(vv, e, tol or 1e-9), str(valor)
        if modo == 'antiderivada':
            var = sp.Symbol(caso.get('var', 'x'))
            libres = (v.free_symbols | e.free_symbols)
            if var not in libres and libres:
                var = sorted(libres, key=str)[0]
            dif = sp.simplify(sp.diff(v - e, var))
            if dif == 0:
                return True, str(valor)
            return equivalentes(sp.diff(v, var), sp.diff(e, var)), str(valor)
        if modo == 'conjunto':
            vl = v if isinstance(v, (list, tuple, sp.Tuple)) else [v]
            el = e if isinstance(e, (list, tuple, sp.Tuple)) else [e]
            vl = [sp.Tuple(*a) if isinstance(a, list) else a for a in vl]
            el = [sp.Tuple(*a) if isinstance(a, list) else a for a in el]
            if isinstance(v, (sp.Set,)):
                return v == sp.FiniteSet(*el), str(valor)
            return _multiset(vl, el, tol), str(valor)
        if modo == 'dict':
            if isinstance(v, list) and len(v) == 1:
                v = v[0]
            if not isinstance(v, dict):
                return False, f'no es dict: {valor}'
            vk = {str(k): val for k, val in v.items()}
            ek = {str(k): val for k, val in e.items()}
            return set(vk) == set(ek) and all(_iguales(vk[k], ek[k], tol) for k in ek), str(valor)
        if modo in ('matriz', 'vector'):
            A = sp.Matrix(v)
            B = sp.Matrix(e)
            if A.shape != B.shape and A.shape == B.T.shape:
                A = A.T
            return A.shape == B.shape and all(_iguales(a, b, tol) for a, b in zip(A, B)), str(valor)
        if modo == 'intervalo':
            dif = sp.SymmetricDifference(v, e)
            return (v == e or sp.simplify(dif) == sp.EmptySet), str(valor)
        if modo == 'edo_general':
            from .edo import leer_edo
            eq_txt = caso.get('edo')
            xs = sp.Symbol('t') if re.search(r'y\(t\)', eq_txt or '') else sp.Symbol('x')
            Y = sp.Function('y')
            eq = sp.sympify(eq_txt, locals={'y': Y, str(xs): xs, 'Eq': sp.Eq, 'Derivative': sp.Derivative, 'E': sp.E,
                                            'exp': sp.exp, 'sin': sp.sin, 'cos': sp.cos, 'log': sp.log})
            if not isinstance(eq, sp.Equality):
                eq = sp.Eq(eq, 0)
            fun = Y(xs)
            sols = v if isinstance(v, list) else [v]
            ok = all(sp.checkodesol(eq, sp.Eq(fun, s_))[0] for s_ in sols)
            orden = sp.ode_order(eq, fun)
            consts = {c for s_ in sols for c in s_.free_symbols if str(c).startswith('C')}
            return ok and len(consts) >= orden, str(valor)
    except Exception as ex:  # noqa
        return False, f'excepción al comparar: {ex!r}'[:200]
    return False, f'modo desconocido {modo}'


def correr(ruta, salida=None, tiempo_max=40, desde=1, hasta=10 ** 9, eco=print):
    casos = json.load(open(ruta, encoding='utf-8'))
    resultados = []
    t0 = time.time()
    bien = 0
    for c in casos:
        if not (desde <= c['id'] <= hasta):
            continue
        d = resolver(c['entrada'], tiempo_max=tiempo_max)
        ok, motivo = comparar(c, d)
        # Además: la comprobación de respuestas debe aceptar el esperado del banco (modos simples)
        bien += ok
        resultados.append({'id': c['id'], 'nivel': c['nivel'], 'tema': c['tema'], 'entrada': c['entrada'], 'ok': ok,
                           'motivo': motivo if not ok else '', 'esperado': c['esperado'], 'tiempo': d.get('tiempo'),
                           'error': d.get('error'), 'detalle': (d.get('detalle') or '')[-600:]})
        eco(f"{'OK ' if ok else 'MAL'} #{c['id']:3d} n{c['nivel']:2d} {d.get('tiempo', 0):6.2f}s {c['entrada'][:70]}"
            + ('' if ok else f"  -> {motivo[:120]} | esperado {str(c['esperado'])[:60]}"))
    total = len(resultados)
    resumen = {'total': total, 'bien': bien, 'mal': total - bien, 'segundos': round(time.time() - t0, 1)}
    por_tema = {}
    for r in resultados:
        pt = por_tema.setdefault(r['tema'], [0, 0])
        pt[0] += r['ok']
        pt[1] += 1
    resumen['por_tema'] = {k: f'{a}/{b}' for k, (a, b) in sorted(por_tema.items())}
    eco(f"RESUMEN {bien}/{total} correctos en {resumen['segundos']} s")
    if salida:
        json.dump({'resumen': resumen, 'resultados': resultados}, open(salida, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
    return resumen


def _a_entrada_alumno(esp):
    """Esperado (sintaxis SymPy) -> como lo escribiría un alumno (ln natural, ^)."""
    t_ = str(esp)
    t_ = re.sub(r'(?<![a-z_])log\(', 'ln(', t_)
    t_ = t_.replace('**', '^')
    return t_


def correr_comprobacion(ruta, eco=print, tiempo_max=40):
    """Para cada caso: la respuesta correcta debe aceptarse y una alterada (+1 / x2) rechazarse."""
    casos = json.load(open(ruta, encoding='utf-8'))
    modos = ('expr', 'antiderivada', 'numero', 'conjunto', 'matriz', 'vector', 'dict', 'intervalo')
    acepta = rechaza = total = 0
    fallos = []
    for c in casos:
        if c['modo'] not in modos or 'Heaviside' in str(c['esperado']):
            continue
        esp = c['esperado']
        if c['modo'] == 'dict':
            e = S(esp)
            bien = ', '.join(f'{k} = {_a_entrada_alumno(v)}' for k, v in e.items())
            k0 = list(e)[0]
            mal = ', '.join(f'{k} = {_a_entrada_alumno(v + (1 if k == k0 else 0))}' for k, v in e.items())
        elif c['modo'] in ('conjunto',):
            lista = S(esp) if isinstance(esp, (list, str)) else esp
            lista = lista if isinstance(lista, list) else [lista]
            if not lista or any(isinstance(a, (list, tuple, sp.Tuple)) for a in lista):
                continue
            bien = ', '.join(_a_entrada_alumno(a) for a in lista)
            mal = ', '.join(_a_entrada_alumno(a + 1) for a in lista)
        elif c['modo'] in ('matriz', 'vector'):
            bien = _a_entrada_alumno(esp)
            M = sp.Matrix(S(esp))
            M[0] = M[0] + 1
            mal = _a_entrada_alumno(M.tolist() if c['modo'] == 'matriz' else list(M))
        elif c['modo'] == 'intervalo':
            bien = _a_entrada_alumno(esp)
            mal = None
        else:
            e = S(esp)
            bien = _a_entrada_alumno(esp)
            if c['modo'] == 'numero' and e.is_number and not e.is_Rational:
                bien = str(sp.N(e, 12))  # el alumno escribe el decimal
            if e.is_number and e.is_infinite:
                mal = None
            elif c['modo'] == 'antiderivada':
                libres = sorted(e.free_symbols, key=str)
                var = sp.Symbol('t') if sp.Symbol('t') in e.free_symbols and sp.Symbol('x') not in e.free_symbols else (libres[0] if libres else sp.Symbol('x'))
                mal = _a_entrada_alumno(e + var ** 2)
            else:
                mal = _a_entrada_alumno(sp.N(e + 1, 12) if c['modo'] == 'numero' and not e.is_Rational else e + 1)
        total += 1
        d1 = resolver(c['entrada'], bien, tiempo_max=tiempo_max)
        ok1 = (d1.get('comprobacion') or {}).get('correcto') is True
        acepta += ok1
        ok2 = True
        if mal is not None:
            d2 = resolver(c['entrada'], mal, tiempo_max=tiempo_max)
            ok2 = (d2.get('comprobacion') or {}).get('correcto') is False
            rechaza += ok2
        if not (ok1 and ok2):
            fallos.append(c['id'])
            eco(f"FALLA #{c['id']} {c['entrada'][:50]} | bien={bien[:40]} -> {ok1} {(d1.get('comprobacion') or {}).get('mensaje', d1.get('error'))!s:.80} | mal -> {ok2}")
    eco(f'COMPROBACION: acepta {acepta}/{total} correctas, rechaza {rechaza} alteradas; fallos {len(fallos)}')
    return fallos
