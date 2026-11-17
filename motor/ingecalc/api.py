"""Punto de entrada del motor: resolver(texto, respuesta) -> dict / JSON."""
import ctypes
import json
import re
import threading
import time
import traceback

import sympy as sp

from . import algebra, basico, calculo, edo, lineal, multivar, numerico, varios
from .comprobar import comprobar
from .nucleo import (ErrorEntrada, coma_decimal, dividir, leer, leer_matriz, leer_sin_evaluar, normalizar, sin_tildes,
                     tex_crudo, variables_de, x, y)

VERSION = '1.0.0'


class TiempoAgotado(Exception):
    pass


def _area(arg):
    if basico.es_figura(arg):
        return basico.geometria(arg, 'area')
    return calculo.area(arg)


def _extremos(arg):
    f = leer(dividir(arg)[0])
    vs = variables_de(f)
    if len(vs) >= 2:
        return multivar.extremos_2v(f, vs[:3])
    return calculo.extremos_1v(f, vs[0] if vs else x)


# (prefijos, función). Se prueban en orden: los más largos primero.
COMANDOS = [
    (('fracciones parciales', 'fraccion parcial', 'fracciones simples', 'apart'), algebra.fracciones_parciales),
    (('derivar implicita', 'derivada implicita', 'implicita'), calculo.derivar_implicita),
    (('recta tangente', 'tangente'), calculo.recta_tangente),
    (('laplace inversa', 'inversa de laplace', 'antitransformada', 'laplace^-1', 'ilaplace'), edo.laplace_inversa),
    (('transformada de laplace', 'laplace'), edo.laplace),
    (('serie de fourier', 'fourier'), edo.fourier),
    (('integral doble', 'doble'), lambda a: multivar.integral_multiple(a, 2)),
    (('integral triple', 'triple'), lambda a: multivar.integral_multiple(a, 3)),
    (('punto fijo',), numerico.punto_fijo),
    (('newton raphson', 'newton-raphson', 'newton'), numerico.newton),
    (('biseccion',), numerico.biseccion),
    (('secante',), numerico.secante),
    (('trapecio', 'trapecios'), numerico.trapecio),
    (('simpson',), numerico.simpson),
    (('runge kutta', 'runge-kutta', 'rk4'), numerico.rk4),
    (('euler mejorado', 'heun'), numerico.heun),
    (('euler',), numerico.euler),
    (('interpolar', 'interpolacion', 'lagrange interpolacion'), numerico.interpolar),
    (('regresion', 'minimos cuadrados', 'ajuste lineal'), numerico.regresion),
    (('autovectores', 'vectores propios', 'eigenvectores'), lambda a: lineal.autovalores(a, True)),
    (('autovalores', 'valores propios', 'eigenvalores', 'eigenvalues'), lineal.autovalores),
    (('determinante', 'det'), lineal.det),
    (('inversa', 'inv'), lineal.inversa),
    (('transpuesta', 'traspuesta'), lineal.transpuesta),
    (('traza',), lineal.traza),
    (('rango',), lineal.rango),
    (('gauss jordan', 'gauss-jordan', 'gauss', 'escalonar'), lineal.gauss),
    (('producto vectorial', 'producto cruz', 'cruz'), lineal.producto_cruz),
    (('producto escalar', 'producto punto', 'punto'), lineal.producto_punto),
    (('norma', 'modulo'), lineal.norma),
    (('angulo',), lineal.angulo),
    (('proyeccion',), lineal.proyeccion),
    (('gradiente', 'grad'), multivar.gradiente),
    (('divergencia', 'div'), multivar.divergencia),
    (('rotacional', 'rotor', 'rot'), multivar.rotacional),
    (('laplaciano',), multivar.laplaciano),
    (('jacobiano', 'jacobiana'), multivar.jacobiano),
    (('hessiano', 'hessiana'), multivar.hessiano),
    (('lagrange', 'multiplicadores de lagrange'), multivar.lagrange),
    (('extremos', 'maximos y minimos', 'puntos criticos', 'optimizar'), _extremos),
    (('derivada de', 'derivar', 'derivada', 'deriva', "d/dx"), calculo.derivar),
    (('integral de', 'integrar', 'integral', 'integra', 'primitiva'), calculo.integrar),
    (('area entre', 'area del', 'area de', 'area', 'superficie del', 'superficie de', 'superficie'), _area),
    (('perimetro del', 'perimetro de', 'perimetro', 'longitud de la circunferencia'), lambda a: basico.geometria(a, 'perimetro')),
    (('volumen del', 'volumen de', 'volumen'), lambda a: basico.geometria(a, 'volumen')),
    (('convertir', 'conversion', 'pasar', 'convierte'), basico.convertir),
    (('regla de tres inversa', 'regla de 3 inversa', 'proporcion inversa'), lambda a: basico.regla_de_tres(a, True)),
    (('regla de tres directa', 'regla de tres', 'regla de 3', 'proporcion directa'), basico.regla_de_tres),
    (('limite de', 'limite', 'lim'), calculo.limite),
    (('taylor', 'mac laurin', 'maclaurin', 'mclaurin'), calculo.taylor),
    (('sumatoria', 'suma de', 'serie numerica', 'sumar'), calculo.sumatoria),
    (('edo', 'ecuacion diferencial', 'resolver edo'), edo.edo),
    (('resolver', 'resuelve', 'resolve', 'despejar', 'solve', 'ecuacion', 'sistema', 'inecuacion'), algebra.resolver),
    (('simplificar', 'simplifica', 'reducir'), algebra.simplificar),
    (('evaluar', 'calcular', 'calcula', 'cuanto es'), algebra.evaluar),
    (('expandir', 'desarrollar', 'expand'), algebra.expandir),
    (('factorizar', 'factoriza', 'factor'), algebra.factorizar),
    (('mcd', 'maximo comun divisor'), lambda a: algebra.mcd_mcm(a, 'mcd')),
    (('mcm', 'minimo comun multiplo'), lambda a: algebra.mcd_mcm(a, 'mcm')),
    (('forma polar', 'polar'), algebra.polar),
    (('fasores', 'fasor'), algebra.fasor),
    (('paralelo',), lambda a: algebra.resistencias(a, 'paralelo')),
    (('serie',), lambda a: algebra.resistencias(a, 'serie')),
    (('estadistica', 'estadisticas', 'datos estadisticos'), varios.estadistica),
    (('binomial',), varios.binomial),
    (('poisson',), varios.poisson),
    (('normal',), varios.normal),
    (('graficar', 'grafica', 'grafico', 'plot', 'dibujar'), varios.graficar),
]


# Módulos de extensión: cada uno define COMANDOS_EXTRA = [(prefijos, funcion), ...] que se prueban ANTES que los
# comandos base (por eso deben usar prefijos específicos que no tapen a los existentes).
EXTENSIONES = ('calculo_extra', 'geo_lineal', 'estad_numerico')
for _nombre in EXTENSIONES:
    try:
        _mod = __import__(f'ingecalc.{_nombre}', fromlist=['COMANDOS_EXTRA'])
        COMANDOS = list(getattr(_mod, 'COMANDOS_EXTRA', [])) + COMANDOS
    except Exception as _e:  # noqa  (un módulo de extensión roto no debe tumbar la app)
        if not (isinstance(_e, ImportError) and _nombre in str(_e)):
            print(f'[IngeCalc] extensión {_nombre} no cargada: {_e!r}')


def _detectar(texto):
    """Devuelve (funcion, argumento)."""
    t_ = texto.strip()
    clave = sin_tildes(t_.lower())
    clave = re.sub(r'\s+', ' ', clave)
    for prefijos, fn in COMANDOS:
        for p in prefijos:
            if clave.startswith(p) and (len(clave) == len(p) or not clave[len(p)].isalnum() or p.endswith(' ')):
                resto = t_[len(p):] if len(t_) >= len(p) else ''
                # respetar mayúsculas originales del resto: buscar por longitud equivalente
                resto = _resto_original(t_, p)
                resto = resto.lstrip(' :')
                return fn, resto
    return None, t_


def _resto_original(original, prefijo):
    """Corta el prefijo del texto original aunque tenga tildes/espacios distintos."""
    norm = ''
    for i, ch in enumerate(original):
        norm = sin_tildes(original[:i + 1].lower())
        norm = re.sub(r'\s+', ' ', norm)
        if norm == prefijo or norm.rstrip() == prefijo and i + 1 < len(original) and not original[i + 1].isalnum():
            return original[i + 1:]
    return original[len(prefijo):]


def _envolver_matrices(texto):
    """[[..]] con operadores -> Matrix([[..]]) para operar."""
    salida, i = '', 0
    while i < len(texto):
        if texto.startswith('[[', i):
            nivel, j = 0, i
            while j < len(texto):
                if texto[j] == '[':
                    nivel += 1
                elif texto[j] == ']':
                    nivel -= 1
                    if nivel == 0:
                        break
                j += 1
            salida += 'Matrix(' + texto[i:j + 1] + ')'
            i = j + 1
        else:
            salida += texto[i]
            i += 1
    return salida


def _automatico(texto):
    """Sin comando: adivina el tema."""
    t_ = coma_decimal(normalizar(texto).strip())
    low = sin_tildes(t_.lower())
    if re.search(r"y\s*'|dy/d[xt]", low):
        return edo.edo(t_)
    if low.startswith('d/dx'):
        return calculo.derivar(t_[4:].strip().strip('()') if t_[4:].strip().startswith('(') else t_[4:])
    if low.startswith('integrar'):
        return calculo.integrar(t_[8:])
    if '@' in t_:
        return algebra.fasor(t_)
    if '≡' in t_ or re.search(r'\(\s*mod', low):
        from .geo_lineal import congruencia
        return congruencia(t_)
    if re.search(r'(?<![a-z])mod(?![a-z])', low):
        from .geo_lineal import modulo
        return modulo(t_)
    if re.match(r'^[\d.,/ ]+\s*[a-zA-Z°²³/]+[a-zA-Z0-9°²³/^]*\s+(a|en)\s+[a-zA-Z°²³/][a-zA-Z0-9°²³/^]*\s*$', t_):
        return basico.convertir(t_)
    if '[[' in t_:
        resto = t_[t_.rfind(']]') + 2:].strip() if t_.count('[[') == 1 and t_.startswith('[[') else 'x'
        if not resto:
            return lineal.resumen_matriz(leer_matriz(t_))
        e = leer(_envolver_matrices(t_), extra={'Matrix': sp.Matrix})
        return lineal.operar_matrices(e, t_)
    if ';' in t_:
        return algebra.resolver(t_)
    if re.search(r'[<>]|(?<![<>!=])=(?!=)', t_):
        partes = dividir(t_)
        if len(partes) == 1:
            eq_vars = variables_de(sp.Tuple(*[leer(p) for p in re.split(r'<=|>=|=|<|>', t_) if p.strip()]))
            if set(eq_vars) == {x, y} and '=' in t_ and not re.search(r'[<>]', t_):
                return varios.graficar(t_)
        return algebra.resolver(t_)
    return algebra.simplificar(t_)


def _resolver_interno(texto, respuesta=None):
    texto = (texto or '').strip()
    if not texto:
        raise ErrorEntrada('Escribí un ejercicio.')
    fn, arg = _detectar(texto)
    if fn is None:
        res = _automatico(texto)
    else:
        if not arg.strip() and fn not in ():
            raise ErrorEntrada('Falta la expresión después del comando.')
        res = fn(arg)
    d = res.a_dict()
    if respuesta is not None and str(respuesta).strip():
        d['comprobacion'] = comprobar(res, respuesta)
    return d


def _con_limite(fn, segundos):
    caja = {}

    def objetivo():
        try:
            caja['r'] = fn()
        except BaseException as e:  # noqa
            caja['e'] = e
            caja['tb'] = traceback.format_exc()

    hilo = threading.Thread(target=objetivo, daemon=True)
    hilo.start()
    hilo.join(segundos)
    if hilo.is_alive():
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(hilo.ident), ctypes.py_object(TiempoAgotado))
        hilo.join(3)
        raise TiempoAgotado()
    if 'e' in caja:
        e = caja['e']
        e.tb = caja.get('tb')
        raise e
    return caja['r']


def resolver(texto, respuesta=None, tiempo_max=30):
    t0 = time.time()
    try:
        d = _con_limite(lambda: _resolver_interno(texto, respuesta), tiempo_max) if tiempo_max else \
            _resolver_interno(texto, respuesta)
    except TiempoAgotado:
        d = {'ok': False, 'error': f'El cálculo tardó más de {tiempo_max} s. Probá simplificar el ejercicio o '
                                   'dividirlo en partes.'}
    except ErrorEntrada as e:
        d = {'ok': False, 'error': str(e)}
    except (sp.SympifyError, SyntaxError, TypeError, ValueError, AttributeError, IndexError, KeyError,
            NotImplementedError, ZeroDivisionError) as e:
        d = {'ok': False, 'error': 'No pude resolverlo: revisá la escritura (' + type(e).__name__ + ').',
             'detalle': (getattr(e, 'tb', None) or str(e))[-1500:]}
    except Exception as e:  # noqa
        d = {'ok': False, 'error': 'Error interno al resolver.', 'detalle': (getattr(e, 'tb', None) or repr(e))[-1500:]}
    d['tiempo'] = round(time.time() - t0, 3)
    d['entrada'] = texto
    return d


def _json_default(o):
    if isinstance(o, sp.Basic):
        return str(o)
    if isinstance(o, complex):
        return str(o)
    return str(o)


def _limpiar(o):
    """Reemplaza NaN/Infinity (JSON inválido) por null."""
    if isinstance(o, float):
        return o if o == o and o not in (float('inf'), float('-inf')) else None
    if isinstance(o, dict):
        return {k_: _limpiar(v) for k_, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_limpiar(v) for v in o]
    return o


def resolver_json(texto, respuesta='', tiempo_max=30):
    d = resolver(texto, respuesta or None, tiempo_max)
    return json.dumps(_limpiar(d), ensure_ascii=False, default=_json_default)


def calentar():
    """Precarga módulos pesados (primer uso más rápido en el celular)."""
    t0 = time.time()
    resolver('derivar x^2*sen(x)', tiempo_max=0)
    resolver('integrar x*e^x', tiempo_max=0)
    return round(time.time() - t0, 2)


NOMBRES_COMANDO = {}


def previa(texto):
    """Vista previa rápida: interpreta la entrada y devuelve su LaTeX (sin resolver)."""
    from .nucleo import tex
    try:
        fn, arg = _detectar(texto)
        comando = ''
        if fn is not None:
            clave = sin_tildes(texto.strip().lower())
            for prefijos, f in COMANDOS:
                if f is fn:
                    comando = next((p for p in prefijos if clave.startswith(p)), prefijos[0])
                    break
        cuerpo = arg if fn is not None else texto
        if fn is not None and getattr(fn, '__module__', '').endswith('estad_numerico'):
            # parámetros con nombre (x̄=50, σ=5, P(B|A)=0.95): se muestran tal cual
            return {'latex': r'\text{' + re.sub(r'[\\{}$#%&_^~]', '', cuerpo.strip())[:120] + '}', 'comando': comando}
        partes = dividir(normalizar(cuerpo), ';')
        trozos = []
        for p in partes[:4]:
            sub = dividir(p)
            principal = sub[0] if sub else p
            principal = principal.replace("y'''", ' yprim3 ').replace("y''", ' yprim2 ').replace("y'", ' yprim1 ')
            if re.match(r"\s*y\s*'*\s*\(", p) and trozos:
                trozos.append('\\text{' + p + '}')
                continue
            if '@' in principal:
                trozos.append(principal.replace('@', r'\angle '))
                continue
            if '=' in principal or '<' in principal or '>' in principal:
                from .nucleo import leer_ecuacion
                if principal.count('=') == 1 and not re.search(r'[<>!]', principal):
                    izq, der = principal.split('=')
                    trozos.append(tex_crudo(leer_sin_evaluar(coma_decimal(izq))) + ' = '
                                  + tex_crudo(leer_sin_evaluar(coma_decimal(der))))
                else:
                    e = leer_ecuacion(principal)
                    trozos.append(tex(e.lhs) + ' = ' + tex(e.rhs) if isinstance(e, sp.Equality) else tex(e))
            elif principal.strip().startswith('[['):
                trozos.append(tex(sp.Matrix(leer(principal))))
            else:
                trozos.append(tex_crudo(leer_sin_evaluar(coma_decimal(principal))))
            if len(sub) > 1:
                def _t(q):
                    if '->' in q:
                        return q.replace('->', r'\to ')
                    try:
                        return tex(leer(q))
                    except Exception:  # noqa  (ej. "eje y", "x'=x+y": se muestran como texto)
                        return r'\text{' + re.sub(r'[\\{}$]', '', q) + '}'
                trozos[-1] += r',\ ' + r',\ '.join(_t(q) for q in sub[1:6])
        latex = r';\quad '.join(trozos)
        for a, b in (('yprim_{3}', "y'''"), ('yprim_{2}', "y''"), ('yprim_{1}', "y'")):
            latex = latex.replace(a, b)
        return {'latex': latex, 'comando': comando}
    except Exception:  # noqa
        # si no se puede interpretar como fórmula, se muestra el texto tal cual (con el comando reconocido)
        try:
            fn, arg = _detectar(texto)
            cuerpo = (arg if fn is not None else texto).strip()
            comando = '' if fn is None else sin_tildes(texto.strip().lower())[:len(texto.strip()) - len(cuerpo)].strip()
            return {'latex': r'\text{' + re.sub(r'[\\{}$#%&_^~]', '', cuerpo)[:120] + '}', 'comando': comando}
        except Exception:  # noqa
            return {'latex': '', 'comando': ''}


def llamar(funcion, args_json):
    """Puente para la app: funcion en {'resolver', 'previa', 'calentar'}; devuelve JSON."""
    args = json.loads(args_json) if args_json else []
    if funcion == 'resolver':
        return resolver_json(args[0], args[1] if len(args) > 1 else '', 40)
    if funcion == 'previa':
        return json.dumps(_limpiar(previa(args[0])), ensure_ascii=False, default=_json_default)
    if funcion == 'rapido':  # resultado en vivo mientras se escribe: límite de tiempo corto
        return resolver_json(args[0], '', 8)
    if funcion == 'sugerencias':
        return json.dumps(sugerencias(args[0]), ensure_ascii=False)
    if funcion == 'calentar':
        return json.dumps({'segundos': calentar()})
    return json.dumps({'ok': False, 'error': 'Función desconocida'})


def sugerencias(texto):
    """Qué tarjetas mostrar para una entrada (como varias formas de resolverla).
    Con comando explícito: una sola. Sin comando: según el tipo de expresión."""
    t_ = (texto or '').strip()
    if not t_:
        return []
    fn, _ = _detectar(t_)
    if fn is not None:
        return [{'titulo': '', 'entrada': t_}]
    base = coma_decimal(normalizar(t_))
    try:
        low = sin_tildes(base.lower())
        if re.search(r"y\s*'|dy/d[xt]", low) or '@' in base or ';' in base:
            return [{'titulo': '', 'entrada': t_}]
        if base.startswith('[[') and base.endswith(']]') and base.count('[[') == 1:
            return [{'titulo': 'Determinante', 'entrada': 'det ' + t_}, {'titulo': 'Matriz inversa', 'entrada': 'inversa ' + t_},
                    {'titulo': 'Autovalores y autovectores', 'entrada': 'autovectores ' + t_},
                    {'titulo': 'Diagonalizar', 'entrada': 'diagonalizar ' + t_}, {'titulo': 'Rango', 'entrada': 'rango ' + t_},
                    {'titulo': 'Núcleo', 'entrada': 'nucleo ' + t_}, {'titulo': 'Imagen', 'entrada': 'imagen ' + t_}]
        if '[[' in base:
            return [{'titulo': '', 'entrada': t_}]
        if re.search(r'[<>]|(?<![<>!=])=(?!=)', base):
            partes = [p for p in re.split(r'<=|>=|=|<|>', base) if p.strip()]
            vs = variables_de(sp.Tuple(*[leer(p) for p in partes]))
            if set(vs) == {x, y} and '=' in base and not re.search(r'[<>]', base):
                return [{'titulo': 'Graficar la curva', 'entrada': 'graficar ' + t_},
                        {'titulo': 'Derivada implícita', 'entrada': 'derivar implicita ' + t_},
                        {'titulo': 'Despejar y', 'entrada': 'resolver ' + t_ + ', y'}]
            sal = [{'titulo': 'Resolver', 'entrada': 'resolver ' + t_}]
            if len(vs) == 1 and not re.search(r'[<>]', base):
                lhs, rhs = base.split('=', 1)
                sal.append({'titulo': 'Graficar ambos miembros', 'entrada': f'graficar {lhs}, {rhs}'})
            return sal
        e = leer(base)
        vs = variables_de(e)
        if not vs:
            return [{'titulo': 'Calcular', 'entrada': t_}]
        if len(vs) == 1:
            v = vs[0]
            es_pol = e.is_polynomial(v)
            es_rac = e.is_rational_function(v)
            # un polinomio ya está "simplificado": se ofrece directamente factorizar/raíces
            ya_simple = (es_pol and sp.expand(e) == e and isinstance(e, sp.Add)
                         and not re.search(r'[/()]|\d\s*\*\s*\d', base))
            sal = [] if ya_simple else [{'titulo': 'Simplificar', 'entrada': t_}]
            if es_pol and sp.degree(e, v) >= 2:
                sal.append({'titulo': 'Factorizar', 'entrada': 'factorizar ' + t_})
            if es_pol and sp.degree(e, v) >= 1 and isinstance(e, sp.Add):
                sal.append({'titulo': 'Raíces (igualar a 0)', 'entrada': f'resolver {t_} = 0'})
            elif es_rac and not es_pol:
                sal.append({'titulo': 'Fracciones parciales', 'entrada': 'fracciones parciales ' + t_})
            sal.append({'titulo': 'Derivada', 'entrada': 'derivar ' + t_})
            sal.append({'titulo': 'Integral', 'entrada': 'integrar ' + t_})
            sal.append({'titulo': 'Estudio completo y gráfico', 'entrada': 'estudio completo ' + t_})
            if es_pol and sp.degree(e, v) >= 2:
                sal.append({'titulo': 'Extremos', 'entrada': 'extremos ' + t_})
            return sal
        sal = [{'titulo': 'Simplificar', 'entrada': t_}, {'titulo': 'Gradiente', 'entrada': 'gradiente ' + t_}]
        if len(vs) == 2:
            sal += [{'titulo': 'Extremos', 'entrada': 'extremos ' + t_}, {'titulo': 'Superficie 3D', 'entrada': 'graficar 3d ' + t_}]
        return sal
    except Exception:  # noqa
        return [{'titulo': '', 'entrada': t_}]
