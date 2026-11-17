"""Integración con explicación paso a paso (traduce las reglas de manualintegrate)."""
import sympy as sp
from sympy.integrals import manualintegrate as mi

from .nucleo import tex


def _int(f, v):
    return r'\int ' + tex(f) + r'\,d' + tex(v)


def _eval(regla):
    try:
        return regla.eval()
    except Exception:  # noqa
        return mi.manualintegrate(regla.integrand, regla.variable)


class Explicador:
    def __init__(self, max_pasos=50):
        self.pasos = []
        self.max_pasos = max_pasos
        self.u_nombres = iter(['u', 'w', 'p', 'q', 'r'])

    def paso(self, titulo, texto, latex, nivel):
        if len(self.pasos) < self.max_pasos:
            self.pasos.append({'titulo': titulo, 'texto': texto, 'latex': latex, 'nivel': nivel})

    def explicar(self, r, nivel=0):
        nombre = type(r).__name__
        f, v = r.integrand, r.variable
        res = _eval(r)
        igual = _int(f, v) + ' = ' + tex(res)

        if nombre == 'ConstantRule':
            self.paso('Integral de una constante', r'\(\int c\,dx = c\,x\).', igual, nivel)
        elif nombre == 'PowerRule':
            if r.exp == -1:
                self.paso('Integral de 1/x', r'\(\int \frac{1}{x}\,dx = \ln|x|\).', igual, nivel)
            else:
                self.paso('Regla de la potencia', r'\(\int x^n\,dx = \frac{x^{n+1}}{n+1}\), con \(n = ' + tex(r.exp) + r'\).', igual, nivel)
        elif nombre == 'ReciprocalRule':
            self.paso('Integral de 1/u', r'\(\int \frac{1}{u}\,du = \ln|u|\).', igual, nivel)
        elif nombre == 'ExpRule':
            texto = r'\(\int e^x\,dx = e^x\).' if r.base == sp.E else r'\(\int a^x\,dx = \frac{a^x}{\ln a}\).'
            self.paso('Integral exponencial', texto, igual, nivel)
        elif nombre in ('SinRule', 'CosRule', 'Sec2Rule', 'SecTanRule', 'Csc2Rule', 'CscCotRule',
                        'SinhRule', 'CoshRule', 'TrigRule', 'HyperbolicRule'):
            self.paso('Integral inmediata (tabla)', 'Se usa la tabla de integrales trigonométricas/hiperbólicas.', igual, nivel)
        elif nombre in ('ArctanRule',):
            self.paso('Integral de arcotangente', r'\(\int \frac{dx}{a^2+x^2} = \frac{1}{a}\arctan\frac{x}{a}\).', igual, nivel)
        elif nombre in ('ArcsinRule', 'ArcsinhRule', 'ReciprocalSqrtQuadraticRule', 'SqrtQuadraticRule',
                        'SqrtQuadraticDenomRule'):
            self.paso('Integral inmediata (tabla)', r'Forma \(\int \frac{dx}{\sqrt{a^2 \pm x^2}}\) de la tabla.', igual, nivel)
        elif nombre == 'ConstantTimesRule':
            self.paso('Constante fuera de la integral', r'\(\int c\,f(x)\,dx = c\int f(x)\,dx\).',
                      _int(f, v) + ' = ' + tex(r.constant) + _int(r.other, v), nivel)
            self.explicar(r.substep, nivel + 1)
        elif nombre == 'AddRule':
            self.paso('Linealidad', 'La integral de una suma es la suma de las integrales.',
                      _int(f, v) + ' = ' + ' + '.join(_int(sr.integrand, v) for sr in r.substeps), nivel)
            for sr in r.substeps:
                self.explicar(sr, nivel + 1)
        elif nombre == 'URule':
            u = sp.Symbol(next(self.u_nombres, 'u'))
            sub = r.substep
            g = sub.integrand.subs(r.u_var, u)
            du = sp.diff(r.u_func, v)
            self.paso('Sustitución', r'Sea \(' + tex(u) + ' = ' + tex(r.u_func) + r'\), entonces \(d' + tex(u) + ' = '
                      + tex(du) + r'\,d' + tex(v) + r'\).',
                      _int(f, v) + ' = ' + _int(g, u), nivel)
            sub2 = _renombrar(sub, r.u_var, u)
            self.explicar(sub2, nivel + 1)
            self.paso('Volver a la variable original', r'Se reemplaza \(' + tex(u) + ' = ' + tex(r.u_func) + r'\).', igual, nivel)
        elif nombre == 'PartsRule':
            du = sp.diff(r.u, v)
            vv = _eval(r.v_step)
            self.paso('Integración por partes', r'\(\int u\,dv = u\,v - \int v\,du\) con \(u = ' + tex(r.u)
                      + r'\), \(dv = ' + tex(r.dv) + r'\,d' + tex(v) + r'\) \(\Rightarrow du = ' + tex(du) + r'\,d' + tex(v)
                      + r'\), \(v = ' + tex(vv) + r'\).',
                      _int(f, v) + ' = ' + tex(r.u * vv) + ' - ' + _int(sp.simplify(vv * du), v), nivel)
            if r.second_step is not None:
                self.explicar(r.second_step, nivel + 1)
        elif nombre == 'CyclicPartsRule':
            self.paso('Partes cíclicas', 'Integrando por partes dos veces vuelve a aparecer la integral original; '
                      'se despeja como una ecuación.', igual, nivel)
        elif nombre == 'RewriteRule':
            self.paso('Reescribir el integrando', 'Se transforma algebraicamente antes de integrar.',
                      _int(f, v) + ' = ' + _int(r.rewritten, v), nivel)
            self.explicar(r.substep, nivel + 1)
        elif nombre == 'CompleteSquareRule':
            self.paso('Completar cuadrados', 'Se completa el cuadrado en el denominador.',
                      _int(f, v) + ' = ' + _int(r.rewritten, v), nivel)
            self.explicar(r.substep, nivel + 1)
        elif nombre == 'TrigSubstitutionRule':
            th = r.theta
            self.paso('Sustitución trigonométrica', r'Sea \(' + tex(v) + ' = ' + tex(r.func) + r'\).',
                      _int(f, v) + ' = ' + _int(r.substep.integrand, th), nivel)
            self.explicar(r.substep, nivel + 1)
            self.paso('Volver a la variable original', 'Se usa el triángulo rectángulo para expresar el resultado en '
                      + tex(v) + '.', igual, nivel)
        elif nombre == 'AlternativeRule':
            # elegir la alternativa con menos pasos
            mejor = min(r.alternatives, key=lambda a: len(str(a)))
            self.explicar(mejor, nivel)
        elif nombre == 'PiecewiseRule':
            self.paso('Por tramos', 'El resultado depende del valor de los parámetros.', igual, nivel)
        elif nombre == 'NestedPowRule':
            self.paso('Potencia anidada', 'Se simplifica la potencia y se integra.', igual, nivel)
        elif nombre in ('HeavisideRule', 'DiracDeltaRule'):
            self.paso('Escalón / Delta', 'Integral de la función escalón o delta de Dirac.', igual, nivel)
        elif nombre in ('ErfRule', 'SiRule', 'CiRule', 'EiRule', 'LiRule', 'FresnelSRule', 'FresnelCRule',
                        'ShiRule', 'ChiRule', 'UpperGammaRule', 'EllipticFRule', 'EllipticERule', 'PolylogRule'):
            self.paso('Función especial', 'Esta integral no tiene primitiva elemental; se expresa con una función '
                      'especial (' + nombre.replace('Rule', '') + ').', igual, nivel)
        elif nombre == 'DerivativeRule':
            self.paso('Integral de una derivada', 'La integral deshace la derivada.', igual, nivel)
        else:
            self.paso('Integral', 'Se resuelve con la tabla de integrales.', igual, nivel)
        return res


def _renombrar(regla, viejo, nuevo):
    """Reemplaza la variable de sustitución interna (_u) por un nombre legible."""
    try:
        import dataclasses
        campos = {}
        for c in dataclasses.fields(regla):
            val = getattr(regla, c.name)
            if isinstance(val, mi.Rule):
                val = _renombrar(val, viejo, nuevo)
            elif isinstance(val, list):
                val = [_renombrar(a, viejo, nuevo) if isinstance(a, mi.Rule) else
                       (a.subs(viejo, nuevo) if isinstance(a, sp.Basic) else a) for a in val]
            elif isinstance(val, sp.Basic):
                val = val.subs(viejo, nuevo)
            campos[c.name] = val
        return type(regla)(**campos)
    except Exception:  # noqa
        return regla


def _hay_desconocido(r):
    if r is None:
        return True
    if type(r).__name__ == 'DontKnowRule':
        return True
    try:
        import dataclasses
        for c in dataclasses.fields(r):
            val = getattr(r, c.name)
            if isinstance(val, mi.Rule) and _hay_desconocido(val):
                return True
            if isinstance(val, list):
                for a in val:
                    if isinstance(a, mi.Rule) and _hay_desconocido(a):
                        if type(r).__name__ != 'AlternativeRule':
                            return True
    except Exception:  # noqa
        return False
    return False


def fracciones_parciales_pasos(f, v, ex, nivel=0):
    """Si f es racional propia/impropia, explica la descomposición."""
    num, den = sp.fraction(sp.cancel(f))
    if not (num.is_polynomial(v) and den.is_polynomial(v)) or sp.degree(den, v) < 2:
        return None
    ap = sp.apart(f, v)
    if ap == f or not ap.is_Add:
        return None
    ex.paso('Fracciones parciales', r'El denominador se factoriza como \(' + tex(sp.factor(den)) + r'\) y el integrando '
            'se descompone en fracciones simples.', tex(f) + ' = ' + tex(ap), nivel)
    return ap


def integrar_con_pasos(f, v):
    """Devuelve (primitiva, pasos, metodo)."""
    ex = Explicador()
    # 1) Racionales: fracciones parciales explícitas (como en la cursada)
    ap = None
    try:
        if f.is_rational_function(v) and not f.is_polynomial(v):
            ap = fracciones_parciales_pasos(f, v, ex)
    except Exception:  # noqa
        ap = None
    objetivo = ap if ap is not None else f
    regla = None
    try:
        regla = mi.integral_steps(objetivo, v)
    except Exception:  # noqa
        regla = None
    if regla is not None and not _hay_desconocido(regla):
        try:
            prim = ex.explicar(regla, 1 if ap is not None else 0)
            prim = sp.simplify(prim) if sp.count_ops(prim) > 40 else prim
            if sp.simplify(sp.diff(prim, v) - f) == 0 or _num_igual(sp.diff(prim, v), f, v):
                return prim, ex.pasos, 'manual'
        except Exception:  # noqa
            pass
    # 2) Algoritmo general de SymPy (Risch / Meijer G), sin pasos detallados
    prim = sp.integrate(f, v)
    if prim.has(sp.Integral):
        return prim, ex.pasos, 'sin_primitiva'
    ex.pasos = ex.pasos[:1] if ap is not None else []
    ex.paso('Integración por algoritmo', 'No es una integral de tabla ni sale por sustitución/partes directa; '
            'se resolvió con el algoritmo de integración simbólica (Risch). Verificá derivando el resultado.',
            _int(f, v) + ' = ' + tex(prim), 0)
    return prim, ex.pasos, 'algoritmo'


def _num_igual(a, b, v, intentos=6):
    import random
    try:
        fa = sp.lambdify(v, a - b, 'mpmath')
        ok = 0
        for _ in range(intentos * 3):
            val = random.uniform(0.15, 2.7)
            try:
                d = complex(fa(val))
            except Exception:  # noqa
                continue
            if abs(d) > 1e-7 * (1 + abs(complex(sp.N(b.subs(v, val))))):
                return False
            ok += 1
            if ok >= intentos:
                return True
        return ok > 0
    except Exception:  # noqa
        return False
