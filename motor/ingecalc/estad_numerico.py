"""Probabilidad y Estadística (cuantiles, distribuciones, Bayes, descriptiva, inferencia), complementos de
Cálculo Numérico, Electrotecnia (potencia) e Investigación Operativa (programación lineal, asignación, colas).

Sin numpy ni scipy: las funciones de distribución y sus inversas se calculan con mpmath
(ncdf, erfinv, betainc, gammainc) y Newton protegido con bisección.
"""
import cmath
import math
import re

import mpmath as mp
import sympy as sp

from . import graficos as G
from .nucleo import (ErrorEntrada, Resultado, dividir, fmt_num, leer, leer_matriz, normalizar, sin_tildes, tex,
                     variable_principal, variables_de, x, y)
from .numerico import _cierre, _cierre_integral, _leer_edo1, _puntos

TEMA_P = 'Probabilidad'
TEMA_E = 'Estadística'
TEMA_N = 'Cálculo numérico'
TEMA_EL = 'Electrotecnia'
TEMA_IO = 'Investigación operativa'
DPS = 25


# ============================================================================
# Utilidades
# ============================================================================
def _val(texto):
    """Número exacto (los decimales se leen como racionales: 0.01 -> 1/100)."""
    s_ = str(texto).strip().replace('°', '')
    if not s_:
        raise ErrorEntrada('Falta un valor numérico.')
    v = leer(s_)
    if not (isinstance(v, sp.Basic) and v.is_number):
        raise ErrorEntrada(f'«{s_}» debería ser un número.')
    if v.has(sp.I):
        raise ErrorEntrada(f'«{s_}» debería ser un número real.')
    return v


def _fl(v):
    """float de un texto o de un número."""
    if isinstance(v, str):
        v = _val(v)
    try:
        return float(v)
    except Exception as exc:  # noqa
        raise ErrorEntrada(f'«{v}» debería ser un número real.') from exc


def _entero(texto, nombre='n', minimo=None):
    v = _val(texto) if isinstance(texto, str) else sp.sympify(texto)
    if not v.is_integer:
        try:
            fv = float(v)
        except Exception:  # noqa
            fv = None
        if fv is None or abs(fv - round(fv)) > 1e-12:
            raise ErrorEntrada(f'{nombre} debe ser un número entero (se leyó {texto}).')
        v = sp.Integer(round(fv))
    v = int(v)
    if minimo is not None and v < minimo:
        raise ErrorEntrada(f'{nombre} debe ser al menos {minimo}.')
    return v


def _F(v, dig=12):
    """Float de SymPy con dig cifras significativas (para el valor que se compara)."""
    f = float(v)
    if f == 0:
        return sp.Float(0)
    return sp.Float(float(f'{f:.{dig}g}'), dig)


def _n(v, dig=6):
    """Texto de un número para mostrar."""
    try:
        return fmt_num(float(v), dig)
    except Exception:  # noqa
        return fmt_num(v, dig)


def _nk(k):
    """Normaliza el nombre de un dato: 'x̄' -> 'x', 'σ²' -> 'sigma2', 'μ₀' -> 'mu0', 'fp₁' -> 'fp1'."""
    k = k.translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789'))
    k = sin_tildes(normalizar(k)).lower()
    return re.sub(r'[\s_^{}()\\.]', '', k)


def _kv(arg):
    """'x̄=50, σ=5, n=36, 95%' -> ({'x': '50', 'sigma': '5', 'n': '36'}, ['95%'])"""
    d, pos = {}, []
    for p in dividir(arg.replace(';', ',')):
        if '=' in p and not re.match(r'\s*[Pp]\s*\(', p) and not re.search(r'[<>]=|=[<>]', p):
            k_, v = p.split('=', 1)
            d[_nk(k_)] = v.strip()
        else:
            pos.append(p.strip())
    return d, pos


def _get(d, *claves):
    for c in claves:
        if c in d:
            return d[c]
    return None


def _prob(texto, nombre='p', abierta=True):
    """Probabilidad: acepta 0.975, 97.5 % o 97.5 (si es > 1 se toma como porcentaje)."""
    s_ = str(texto).strip()
    v = _fl(s_)
    if '%' not in s_ and 1 < v < 100:
        v = v / 100
    if abierta and not 0 < v < 1:
        raise ErrorEntrada(f'{nombre} debe estar entre 0 y 1 (o entre 0 % y 100 %); se leyó {s_}.')
    if not abierta and not 0 <= v <= 1:
        raise ErrorEntrada(f'{nombre} debe estar entre 0 y 1; se leyó {s_}.')
    return v


def _prob_exacta(texto, nombre='P'):
    v = _val(texto)
    if not (0 <= v <= 1):
        raise ErrorEntrada(f'{nombre} = {texto} no es una probabilidad (debe estar entre 0 y 1).')
    return v


def _ventana_densidad(g, ymax=None):
    try:
        vy = g['ventana']
        top = ymax if ymax is not None else vy['ymax']
        vy['ymin'] = -0.08 * top
        if ymax is not None:
            vy['ymax'] = 1.2 * ymax
    except Exception:  # noqa
        pass
    return g


def _graf_densidad(dens, xmin, xmax, a, b, puntos=(), ymax=None, arriba=None):
    """Densidad con el área entre a y b sombreada (o la función 'arriba' si se indica)."""
    try:
        g = G.grafico_2d([dens], x, [p for p in puntos if p], xmin=float(xmin), xmax=float(xmax))
        a_, b_ = max(float(a), float(xmin)), min(float(b), float(xmax))
        if b_ > a_:
            g['sombra'] = G.sombra_entre(arriba if arriba is not None else dens, sp.Integer(0), a_, b_, x)
        return _ventana_densidad(g, ymax)
    except Exception:  # noqa
        return None


def _curva_puntos(pts, etiqueta='', color=None, estilo='solido'):
    """Curva dibujada por puntos (sin fórmula): rectas verticales, soluciones numéricas."""
    return {'fn': None, 'var': 'x', 'etiqueta': etiqueta, 'color': color or G.COLORES[0], 'estilo': estilo,
            'muestras': [[float(p[0]), float(p[1])] if p is not None else None for p in pts]}


def _datos(texto):
    s_ = texto.strip()
    if s_.startswith('[') and s_.endswith(']') and not s_.startswith('[['):
        s_ = s_[1:-1]
    datos = []
    for p in dividir(s_.replace(';', ',')):
        datos.append(_val(p))
    return datos


def _extra(r, titulo, texto, latex):
    """Resultado.extra con los argumentos en orden (título, texto, latex)."""
    r.extra(titulo, latex, texto)


# ============================================================================
# Distribuciones continuas (mpmath)
# ============================================================================
def _phi(z):
    return mp.ncdf(z)


def _zinv(p):
    return mp.sqrt(2) * mp.erfinv(2 * mp.mpf(p) - 1)


def _t_cdf(v, nu):
    v, nu = mp.mpf(v), mp.mpf(nu)
    if mp.isinf(v):
        return mp.mpf(1) if v > 0 else mp.mpf(0)
    if v == 0:
        return mp.mpf(1) / 2
    cola = mp.betainc(nu / 2, mp.mpf(1) / 2, 0, nu / (nu + v * v), regularized=True) / 2
    return 1 - cola if v > 0 else cola


def _t_pdf(v, nu):
    v, nu = mp.mpf(v), mp.mpf(nu)
    return mp.exp(mp.loggamma((nu + 1) / 2) - mp.loggamma(nu / 2)) / mp.sqrt(nu * mp.pi) * (1 + v * v / nu) ** (-(nu + 1) / 2)


def _chi2_cdf(v, k):
    v, k = mp.mpf(v), mp.mpf(k)
    if v <= 0:
        return mp.mpf(0)
    if mp.isinf(v):
        return mp.mpf(1)
    return mp.gammainc(k / 2, 0, v / 2, regularized=True)


def _chi2_pdf(v, k):
    v, k = mp.mpf(v), mp.mpf(k)
    if v <= 0:
        return mp.mpf(0)
    return mp.exp((k / 2 - 1) * mp.log(v) - v / 2 - (k / 2) * mp.log(2) - mp.loggamma(k / 2))


def _f_cdf(v, d1, d2):
    v, d1, d2 = mp.mpf(v), mp.mpf(d1), mp.mpf(d2)
    if v <= 0:
        return mp.mpf(0)
    if mp.isinf(v):
        return mp.mpf(1)
    return mp.betainc(d1 / 2, d2 / 2, 0, d1 * v / (d1 * v + d2), regularized=True)


def _f_pdf(v, d1, d2):
    v, d1, d2 = mp.mpf(v), mp.mpf(d1), mp.mpf(d2)
    if v <= 0:
        return mp.mpf(0)
    lg = (d1 * mp.log(d1 * v) + d2 * mp.log(d2) - (d1 + d2) * mp.log(d1 * v + d2)) / 2 - mp.log(v) \
        - (mp.loggamma(d1 / 2) + mp.loggamma(d2 / 2) - mp.loggamma((d1 + d2) / 2))
    return mp.exp(lg)


def _invertir(cdf, pdf, p, x0, lo, hi):
    """Resuelve cdf(q) = p (cdf creciente) con Newton protegido por bisección."""
    p = mp.mpf(p)
    lo, hi = mp.mpf(lo), mp.mpf(hi)
    for _ in range(400):
        if cdf(hi) >= p:
            break
        lo, hi = hi, hi * 2
    for _ in range(400):
        if cdf(lo) <= p:
            break
        hi, lo = lo, (lo * 2 if lo < 0 else lo / 2)
    xk = mp.mpf(x0)
    if not lo < xk < hi:
        xk = (lo + hi) / 2
    for _ in range(200):
        c = cdf(xk) - p
        if c == 0:
            return xk
        if c < 0:
            lo = xk
        else:
            hi = xk
        d = pdf(xk)
        xn = xk - c / d if d > 0 else (lo + hi) / 2
        if not lo < xn < hi:
            xn = (lo + hi) / 2
        if abs(xn - xk) <= mp.mpf(10) ** -20 * (1 + abs(xk)):
            return xn
        xk = xn
    return xk


def _q_normal(p):
    with mp.workdps(DPS):
        return _zinv(p)


def _q_t(p, nu):
    with mp.workdps(DPS):
        return _invertir(lambda v: _t_cdf(v, nu), lambda v: _t_pdf(v, nu), p, _zinv(p), -1, 1)


def _q_chi2(p, k):
    with mp.workdps(DPS):
        z = _zinv(p)
        k_ = mp.mpf(k)
        x0 = k_ * (1 - 2 / (9 * k_) + z * mp.sqrt(2 / (9 * k_))) ** 3
        return _invertir(lambda v: _chi2_cdf(v, k), lambda v: _chi2_pdf(v, k), p, x0 if x0 > 0 else k_ / 2, 0, k_ + 10)


def _q_f(p, d1, d2):
    with mp.workdps(DPS):
        return _invertir(lambda v: _f_cdf(v, d1, d2), lambda v: _f_pdf(v, d1, d2), p, 1, 0, 4)


# Expresiones de densidad para graficar
def _dens_normal(mu=0, sg=1):
    return sp.exp(-(x - sp.Float(mu)) ** 2 / (2 * sp.Float(sg) ** 2)) / (sp.Float(sg) * sp.sqrt(2 * sp.pi))


def _dens_t(nu):
    with mp.workdps(20):
        c = float(_t_pdf(0, nu))
    return sp.Float(c) * (1 + x ** 2 / sp.Float(nu)) ** (-sp.Float((nu + 1) / 2))


def _dens_chi2(k):
    with mp.workdps(20):
        c = float(mp.exp(-(mp.mpf(k) / 2) * mp.log(2) - mp.loggamma(mp.mpf(k) / 2)))
    return sp.Piecewise((sp.Float(c) * x ** sp.Float(k / 2 - 1) * sp.exp(-x / 2), x > 0), (0, True))


def _dens_f(d1, d2):
    with mp.workdps(20):
        c = float(mp.exp((mp.mpf(d1) / 2) * mp.log(mp.mpf(d1) / d2) - (mp.loggamma(mp.mpf(d1) / 2) + mp.loggamma(mp.mpf(d2) / 2)
                                                                       - mp.loggamma(mp.mpf(d1 + d2) / 2))))
    return sp.Piecewise((sp.Float(c) * x ** sp.Float(d1 / 2 - 1) * (1 + sp.Float(d1 / d2) * x) ** sp.Float(-(d1 + d2) / 2), x > 0),
                        (0, True))


def _dist(nombre, params):
    """(cdf, pdf, densidad sympy, ventana, latex de la variable, descripción) de una distribución."""
    if nombre == 'normal':
        mu, sg = params
        return (lambda v: _phi((mp.mpf(v) - mu) / sg), lambda v: mp.npdf(v, mu, sg), _dens_normal(mu, sg),
                (mu - 4 * sg, mu + 4 * sg), 'Z' if (mu, sg) == (0, 1) else 'X',
                r'N(0,1)' if (mu, sg) == (0, 1) else r'N(' + _n(mu) + ',\ ' + _n(sg) + '^2)')
    if nombre == 't':
        nu, = params
        lim = max(4.0, float(_q_t(0.995, nu)) if nu < 30 else 4.0)
        return (lambda v: _t_cdf(v, nu), lambda v: _t_pdf(v, nu), _dens_t(nu), (-lim, lim), 'T',
                r't_{' + _n(nu) + '}')
    if nombre == 'chi2':
        k_, = params
        return (lambda v: _chi2_cdf(v, k_), lambda v: _chi2_pdf(v, k_), _dens_chi2(k_),
                (0, float(_q_chi2(0.999, k_)) * 1.05), r'\chi^2', r'\chi^2_{' + _n(k_) + '}')
    d1, d2 = params
    return (lambda v: _f_cdf(v, d1, d2), lambda v: _f_pdf(v, d1, d2), _dens_f(d1, d2),
            (0, float(_q_f(0.995, d1, d2)) * 1.05), 'F', r'F_{' + _n(d1) + ',\ ' + _n(d2) + '}')


def _ymax_densidad(pdf, a, b):
    with mp.workdps(15):
        m = 0.0
        for i in range(1, 200):
            try:
                v = float(pdf(a + (b - a) * i / 200))
            except Exception:  # noqa
                continue
            if math.isfinite(v):
                m = max(m, v)
    return m or None


# ============================================================================
# Cuantiles
# ============================================================================
def cuantil(arg, dist):
    usos = {'normal': 'cuantil normal p[, μ, σ]   (p = área a la izquierda)', 't': 'cuantil t p, gl',
            'chi2': 'cuantil chi2 p, gl', 'f': 'cuantil f p, gl1, gl2'}
    partes = dividir(arg)
    req = {'normal': 1, 't': 2, 'chi2': 2, 'f': 3}[dist]
    if len(partes) < req:
        raise ErrorEntrada('Uso: ' + usos[dist])
    p = _prob(partes[0], 'p')
    nombres = {'normal': 'normal', 't': 't de Student', 'chi2': 'chi cuadrado', 'f': 'F de Fisher-Snedecor'}
    r = Resultado(TEMA_P, 'Cuantil de la ' + ('distribución ' if dist != 'normal' else '') + nombres[dist])
    with mp.workdps(DPS):
        if dist == 'normal':
            mu, sg = (_fl(partes[1]), _fl(partes[2])) if len(partes) >= 3 else (0.0, 1.0)
            if sg <= 0:
                raise ErrorEntrada('σ debe ser positivo.')
            z = _q_normal(p)
            q = mu + sg * z
            params = (mu, sg)
            r.entrada_latex = r'P(Z \le z) = ' + _n(p, 10) + (r',\quad X \sim N(' + _n(mu) + r',\ ' + _n(sg) + '^2)'
                                                               if (mu, sg) != (0, 1) else '')
            r.paso('Definición', 'Se busca el valor que deja un área p a su izquierda bajo la normal estándar.',
                   r'\Phi(z) = P(Z \le z) = ' + _n(p, 10), 0)
            r.paso('Inversa de la normal', r'Se usa \(z = \sqrt{2}\,\operatorname{erf}^{-1}(2p-1)\) '
                   '(equivale a buscar el área p en el cuerpo de la tabla).',
                   r'z = \sqrt{2}\,\operatorname{erf}^{-1}(2\cdot ' + _n(p, 10) + ' - 1) = ' + _n(z, 10), 0)
            r.paso('Simetría', 'La normal es simétrica: el cuantil de 1 − p es el mismo con signo opuesto.',
                   r'z_{' + _n(1 - p, 10) + '} = ' + _n(-z, 10), 1)
            if (mu, sg) != (0, 1):
                r.paso('Desestandarizar', r'\(x = \mu + \sigma z\)', 'x = ' + _n(mu) + r' + ' + _n(sg) + r'\cdot ' + _n(z, 10)
                       + ' = ' + _n(q, 10), 0)
            sim = 'z' if (mu, sg) == (0, 1) else 'x'
            dist_g = 'normal'
        else:
            if dist == 'f':
                d1, d2 = _fl(partes[1]), _fl(partes[2])
                if d1 <= 0 or d2 <= 0:
                    raise ErrorEntrada('Los grados de libertad deben ser positivos.')
                q = _q_f(p, d1, d2)
                params = (d1, d2)
                sim = 'f'
                r.entrada_latex = r'P(F_{' + _n(d1) + ',' + _n(d2) + r'} \le f) = ' + _n(p, 10)
                cdf_txt = r'I_{\frac{d_1 f}{d_1 f + d_2}}\left(\tfrac{d_1}{2}, \tfrac{d_2}{2}\right)'
                r.paso('Recíproco', 'Propiedad útil con tablas que solo traen la cola derecha:',
                       r'f_{1-p}(d_1, d_2) = \frac{1}{f_{p}(d_2, d_1)}', 1)
            else:
                k_ = _fl(partes[1])
                if k_ <= 0:
                    raise ErrorEntrada('Los grados de libertad deben ser positivos.')
                params = (k_,)
                if dist == 't':
                    q = _q_t(p, k_)
                    sim = 't'
                    r.entrada_latex = r'P(T_{' + _n(k_) + r'} \le t) = ' + _n(p, 10)
                    cdf_txt = r'1 - \tfrac12 I_{\frac{\nu}{\nu+t^2}}\left(\tfrac{\nu}{2}, \tfrac12\right)\ (t>0)'
                else:
                    q = _q_chi2(p, k_)
                    sim = r'\chi^2'
                    r.entrada_latex = r'P(\chi^2_{' + _n(k_) + r'} \le x) = ' + _n(p, 10)
                    cdf_txt = r'P\left(\tfrac{k}{2}, \tfrac{x}{2}\right)\ \text{(gamma incompleta regularizada)}'
            r.paso('Definición', 'Se busca el valor que deja un área p a su izquierda (cuantil o percentil p).',
                   r.entrada_latex, 0)
            r.paso('Función de distribución', 'Se resuelve F(q) = p numéricamente (Newton protegido con bisección) usando',
                   'F = ' + cdf_txt, 1)
            r.paso('Uso de tablas', 'En una tabla de cola derecha buscá α = 1 − p = ' + _n(1 - p, 10) + '.', '', 1)
            if dist == 't':
                r.paso('Simetría', 'La t es simétrica respecto de 0:', r't_{' + _n(1 - p, 10) + '} = ' + _n(-q, 10), 1)
            dist_g = dist
    q = float(q)
    r.paso('Resultado', '', sim + r'_{' + _n(p, 10) + '} = ' + _n(q, 10), 0)
    r.valor = _F(q, 12)
    r.valores['decimal'] = _n(q, 12)
    r.resultado_latex = sim + r'_{' + _n(p, 10) + r'} \approx ' + _n(q, 10)
    r.comparar = ('numero', q, {'tol': 1e-5})
    cdf, pdf, dens, (a, b), _, _ = _dist(dist_g, params)
    r.grafico = _graf_densidad(dens, a, b, a, q, [G.punto(q, 0, f'{_n(q, 5)}', '#dc2626')],
                               ymax=_ymax_densidad(pdf, a, b))
    return r


def normal_inversa(arg):
    partes = dividir(arg)
    if len(partes) < 3:
        raise ErrorEntrada('Uso: normal inversa μ, σ, p   → x tal que P(X ≤ x) = p   (agregá «, derecha» para P(X > x) = p)')
    mu, sg = _fl(partes[0]), _fl(partes[1])
    if sg <= 0:
        raise ErrorEntrada('σ debe ser positivo.')
    p = _prob(partes[2], 'p')
    derecha = len(partes) > 3 and re.search(r'derech|mayor|>', sin_tildes(partes[3].lower()))
    pi = 1 - p if derecha else p
    r = Resultado(TEMA_P, 'Normal inversa')
    r.entrada_latex = r'X \sim N(' + _n(mu) + r',\ ' + _n(sg) + r'^2),\quad P(X ' + ('>' if derecha else r'\le') + ' x) = ' + _n(p, 10)
    if derecha:
        r.paso('Pasar a la cola izquierda', '', r'P(X \le x) = 1 - ' + _n(p, 10) + ' = ' + _n(pi, 10), 0)
    z = float(_q_normal(pi))
    q = mu + sg * z
    r.paso('Cuantil de la normal estándar', 'Se busca z con Φ(z) = p (tabla de la normal o z = √2·erf⁻¹(2p − 1)).',
           r'\Phi(z) = ' + _n(pi, 10) + r'\Rightarrow z = ' + _n(z, 10), 0)
    r.paso('Desestandarizar', r'\(x = \mu + z\,\sigma\)', 'x = ' + _n(mu) + ' + (' + _n(z, 10) + r')\cdot ' + _n(sg) + ' = ' + _n(q, 10), 0)
    r.valor = _F(q, 12)
    r.valores['z'] = _n(z, 12)
    r.resultado_latex = r'x \approx ' + _n(q, 10)
    r.comparar = ('numero', q, {'tol': 1e-5})
    dens = _dens_normal(mu, sg)
    a, b = (q, mu + 4 * sg) if derecha else (mu - 4 * sg, q)
    r.grafico = _graf_densidad(dens, mu - 4 * sg, mu + 4 * sg, a, b, [G.punto(q, 0, 'x ≈ ' + _n(q, 5), '#dc2626')])
    return r


# ============================================================================
# Distribuciones
# ============================================================================
def geometrica(arg):
    partes = dividir(arg)
    if len(partes) < 2:
        raise ErrorEntrada('Uso: geometrica p, k   → P(X = k), X = número de ensayo del primer éxito (k = 1, 2, ...)')
    p = _prob_exacta(partes[0], 'p')
    k_ = _entero(partes[1], 'k', 1)
    if p == 0:
        raise ErrorEntrada('p debe ser mayor que 0.')
    r = Resultado(TEMA_P, 'Distribución geométrica')
    r.entrada_latex = r'X \sim \text{Geom}(' + tex(p) + r'),\quad P(X = ' + str(k_) + ')'
    val = (1 - p) ** (k_ - 1) * p
    r.paso('Modelo', 'X = número del ensayo en que aparece el primer éxito: k − 1 fracasos y luego un éxito.',
           r'P(X = k) = (1-p)^{k-1}\,p', 0)
    r.paso('Reemplazo', '', r'P(X = ' + str(k_) + ') = (' + tex(1 - p) + ')^{' + str(k_ - 1) + r'}\cdot ' + tex(p) + ' = '
           + tex(val) + r'\approx ' + _n(val, 8), 0)
    acum = 1 - (1 - p) ** k_
    r.extra('Acumulada', r'P(X \le ' + str(k_) + ') = 1 - (1-p)^{k} = ' + tex(acum) + r'\approx ' + _n(acum, 8))
    r.extra('Media y varianza', r'E[X] = \frac1p = ' + tex(1 / p) + r',\quad V[X] = \frac{1-p}{p^2} = ' + tex((1 - p) / p ** 2))
    _extra(r, 'Otra convención', 'Si X cuenta los fracasos antes del primer éxito (k = 0, 1, ...):',
            r'P(X = ' + str(k_) + ') = (1-p)^{k}p = ' + tex((1 - p) ** k_ * p) + r'\approx ' + _n((1 - p) ** k_ * p, 8))
    r.valor = val
    r.valores.update({'decimal': _n(val, 12), 'acumulada': _n(acum, 12), 'media': _n(1 / p, 12),
                      'varianza': _n((1 - p) / p ** 2, 12)})
    r.resultado_latex = 'P(X = ' + str(k_) + ') = ' + tex(val) + r'\approx ' + _n(val, 8)
    r.comparar = ('numero', val, {'tol': 1e-6})
    tope = int(min(40, max(k_ + 3, math.ceil(4 / float(p)))))
    r.grafico = {'tipo': 'barras', 'barras': [{'x': i, 'alto': float((1 - p) ** (i - 1) * p), 'resaltar': i == k_}
                                              for i in range(1, tope + 1)]}
    return r


def hipergeometrica(arg):
    partes = dividir(arg)
    if len(partes) < 4:
        raise ErrorEntrada('Uso: hipergeometrica N, K, n, k   (N población, K éxitos en la población, n muestra, k éxitos)')
    N_, K_, n_, k_ = (_entero(p, nom, 0) for p, nom in zip(partes[:4], ('N', 'K', 'n', 'k')))
    if K_ > N_ or n_ > N_:
        raise ErrorEntrada('Debe ser K ≤ N y n ≤ N.')
    r = Resultado(TEMA_P, 'Distribución hipergeométrica')
    r.entrada_latex = r'X \sim H(N=' + str(N_) + ',\ K=' + str(K_) + ',\ n=' + str(n_) + r'),\quad P(X = ' + str(k_) + ')'
    kmin, kmax = max(0, n_ - (N_ - K_)), min(n_, K_)
    pmf = lambda i: sp.binomial(K_, i) * sp.binomial(N_ - K_, n_ - i) / sp.binomial(N_, n_)  # noqa
    val = pmf(k_) if kmin <= k_ <= kmax else sp.Integer(0)
    r.paso('Modelo', 'Extracción SIN reposición: casos favorables / casos posibles.',
           r'P(X = k) = \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}', 0)
    r.paso('Reemplazo', '', r'\frac{\binom{' + str(K_) + '}{' + str(k_) + r'}\binom{' + str(N_ - K_) + '}{' + str(n_ - k_) + r'}}{\binom{'
           + str(N_) + '}{' + str(n_) + '}} = ' + (r'\frac{' + tex(sp.binomial(K_, k_)) + r'\cdot ' + tex(sp.binomial(N_ - K_, n_ - k_))
                                                    + '}{' + tex(sp.binomial(N_, n_)) + '} = ' if kmin <= k_ <= kmax else '')
           + tex(val) + r'\approx ' + _n(val, 8), 0)
    if not kmin <= k_ <= kmax:
        r.avisos.append(f'k debe estar entre {kmin} y {kmax}: la probabilidad es 0.')
    acum = sum((pmf(i) for i in range(kmin, min(k_, kmax) + 1)), sp.Integer(0))
    media = sp.Rational(n_ * K_, N_)
    var = media * sp.Rational(N_ - K_, N_) * (sp.Rational(N_ - n_, N_ - 1) if N_ > 1 else 0)
    r.extra('Acumulada', r'P(X \le ' + str(k_) + r') \approx ' + _n(acum, 8))
    r.extra('Media y varianza', r'E[X] = n\frac{K}{N} = ' + tex(media) + r',\quad V[X] = n\frac{K}{N}\frac{N-K}{N}\frac{N-n}{N-1} = '
            + tex(var))
    r.valor = val
    r.valores.update({'decimal': _n(val, 12), 'acumulada': _n(acum, 12), 'media': _n(media, 12), 'varianza': _n(var, 12)})
    r.resultado_latex = 'P(X = ' + str(k_) + ') = ' + tex(val) + r'\approx ' + _n(val, 8)
    r.comparar = ('numero', val, {'tol': 1e-6})
    r.grafico = {'tipo': 'barras', 'barras': [{'x': i, 'alto': float(pmf(i)), 'resaltar': i == k_}
                                              for i in range(kmin, min(kmax, kmin + 60) + 1)]}
    return r


def _limites_ab(partes, i):
    """Lee a, b (con -oo / oo) desde partes[i:]; con un solo valor devuelve (-oo, valor)."""
    if len(partes) > i + 1:
        a, b = _val(partes[i]), _val(partes[i + 1])
    else:
        a, b = -sp.oo, _val(partes[i])
    if a > b:
        raise ErrorEntrada('El límite inferior a debe ser menor que b.')
    return a, b


def _txt_lim(v):
    if v == sp.oo:
        return r'\infty'
    if v == -sp.oo:
        return r'-\infty'
    return _n(v, 8)


def exponencial(arg):
    partes = dividir(arg)
    if len(partes) < 2:
        raise ErrorEntrada('Uso: exponencial λ, a, b   → P(a < X < b)   ;   exponencial λ, x   → P(X ≤ x)')
    lam = _val(partes[0])
    if lam <= 0:
        raise ErrorEntrada('λ debe ser positivo.')
    a, b = _limites_ab(partes, 1)
    a0 = max(a, 0) if a != -sp.oo else sp.Integer(0)
    r = Resultado(TEMA_P, 'Distribución exponencial')
    r.entrada_latex = r'X \sim \text{Exp}(\lambda = ' + tex(lam) + r'),\quad P(' + _txt_lim(a) + ' < X < ' + _txt_lim(b) + ')'
    r.paso('Modelo', r'Densidad \(f(x) = \lambda e^{-\lambda x}\) para \(x \ge 0\); acumulada \(F(x) = 1 - e^{-\lambda x}\).',
           r'P(a < X < b) = F(b) - F(a) = e^{-\lambda a} - e^{-\lambda b}', 0)
    if b <= 0:
        val = sp.Integer(0)
    else:
        val = sp.exp(-lam * a0) - (sp.exp(-lam * b) if b != sp.oo else 0)
    r.paso('Reemplazo', '', 'P = e^{-' + tex(lam) + r'\cdot ' + tex(a0) + '} - ' + ('e^{-' + tex(lam) + r'\cdot ' + tex(b) + '}' if b != sp.oo else '0')
           + ' = ' + tex(val) + r'\approx ' + _n(val, 8), 0)
    r.extra('Media, varianza y mediana', r'E[X] = \frac1\lambda = ' + tex(1 / lam) + r',\ V[X] = \frac{1}{\lambda^2} = ' + tex(1 / lam ** 2)
            + r',\ Me = \frac{\ln 2}{\lambda} \approx ' + _n(sp.log(2) / lam, 8))
    _extra(r, 'Falta de memoria', '', r'P(X > s + t \mid X > s) = P(X > t)')
    r.valor = val
    r.valores.update({'decimal': _n(val, 12), 'media': _n(1 / lam, 12), 'varianza': _n(1 / lam ** 2, 12)})
    r.resultado_latex = r'P \approx ' + _n(val, 10)
    r.comparar = ('numero', val, {'tol': 1e-6})
    dens = sp.Piecewise((sp.Float(lam) * sp.exp(-sp.Float(lam) * x), x >= 0), (0, True))
    xmax = float(5 / lam)
    if b != sp.oo:
        xmax = max(xmax, float(b) * 1.2)
    r.grafico = _graf_densidad(dens, -0.1 * xmax, xmax, float(a0), float(b) if b != sp.oo else xmax, ymax=float(lam))
    return r


def uniforme(arg):
    partes = dividir(arg)
    if len(partes) < 3:
        raise ErrorEntrada('Uso: uniforme a, b, c, d   → P(c < X < d) con X ~ U(a, b)   ;   uniforme a, b, c   → P(X ≤ c)')
    a, b = _val(partes[0]), _val(partes[1])
    if a >= b:
        raise ErrorEntrada('En U(a, b) debe ser a < b.')
    c, d = _limites_ab(partes, 2)
    r = Resultado(TEMA_P, 'Distribución uniforme')
    r.entrada_latex = r'X \sim U(' + tex(a) + ', ' + tex(b) + r'),\quad P(' + _txt_lim(c) + ' < X < ' + _txt_lim(d) + ')'
    ci, di = sp.Max(c, a), sp.Min(d, b)
    val = (di - ci) / (b - a) if di > ci else sp.Integer(0)
    r.paso('Modelo', r'Densidad constante \(f(x) = \frac{1}{b-a}\) en [a, b]: la probabilidad es proporcional a la longitud.',
           r'P(c < X < d) = \frac{\min(d,b) - \max(c,a)}{b - a}', 0)
    r.paso('Reemplazo', '', r'\frac{' + tex(di) + ' - ' + tex(ci) + '}{' + tex(b) + ' - ' + tex(a) + '} = ' + tex(val) + r'\approx ' + _n(val, 8), 0)
    r.extra('Media y varianza', r'E[X] = \frac{a+b}{2} = ' + tex((a + b) / 2) + r',\quad V[X] = \frac{(b-a)^2}{12} = ' + tex((b - a) ** 2 / 12))
    r.valor = val
    r.valores.update({'decimal': _n(val, 12), 'media': _n((a + b) / 2, 12), 'varianza': _n((b - a) ** 2 / 12, 12)})
    r.resultado_latex = 'P = ' + tex(val) + r'\approx ' + _n(val, 8)
    r.comparar = ('numero', val, {'tol': 1e-6})
    h = 1 / (b - a)
    dens = sp.Piecewise((h, (x >= a) & (x <= b)), (0, True))
    an = float(b - a)
    r.grafico = _graf_densidad(dens, float(a) - 0.3 * an, float(b) + 0.3 * an, float(ci), float(di), ymax=float(h))
    return r


def _prob_continua(arg, dist):
    """t student gl, a, b  /  chi2 gl, a, b  /  f snedecor d1, d2, a, b"""
    usos = {'t': 't student gl, a, b   → P(a < T < b)   ;   t student gl, t   → P(T ≤ t)',
            'chi2': 'chi2 gl, a, b   → P(a < X < b)   ;   chi2 gl, x   → P(X ≤ x)',
            'f': 'f snedecor gl1, gl2, a, b   → P(a < F < b)   ;   f snedecor gl1, gl2, f   → P(F ≤ f)'}
    partes = dividir(arg)
    npar = 2 if dist == 'f' else 1
    if len(partes) < npar + 1:
        raise ErrorEntrada('Uso: ' + usos[dist])
    params = tuple(_fl(p) for p in partes[:npar])
    if any(v <= 0 for v in params):
        raise ErrorEntrada('Los grados de libertad deben ser positivos.')
    a, b = _limites_ab(partes, npar)
    titulos = {'t': 'Distribución t de Student', 'chi2': 'Distribución chi cuadrado', 'f': 'Distribución F de Fisher-Snedecor'}
    r = Resultado(TEMA_P, titulos[dist])
    with mp.workdps(DPS):
        cdf, pdf, dens, (xa, xb), var, nombre = _dist(dist, params)
        Fa = mp.mpf(0) if a == -sp.oo else cdf(float(a))
        Fb = mp.mpf(1) if b == sp.oo else cdf(float(b))
        val = float(Fb - Fa)
    r.entrada_latex = var + r' \sim ' + nombre + r',\quad P(' + _txt_lim(a) + ' < ' + var + ' < ' + _txt_lim(b) + ')'
    formulas = {'t': r'F(t) = 1 - \tfrac12 I_{\nu/(\nu+t^2)}\left(\tfrac\nu2, \tfrac12\right)\ (t > 0),\quad F(-t) = 1 - F(t)',
                'chi2': r'F(x) = \frac{\gamma\left(\frac k2, \frac x2\right)}{\Gamma\left(\frac k2\right)}',
                'f': r'F(f) = I_{\frac{d_1 f}{d_1 f + d_2}}\left(\tfrac{d_1}{2}, \tfrac{d_2}{2}\right)'}
    r.paso('Función de distribución acumulada', 'Se calcula con funciones especiales (equivale a leer la tabla).', formulas[dist], 0)
    r.paso('Probabilidad', r'\(P(a < X < b) = F(b) - F(a)\)',
           'F(' + _txt_lim(b) + ') - F(' + _txt_lim(a) + ') = ' + _n(Fb, 8) + ' - ' + _n(Fa, 8) + r' \approx ' + _n(val, 10), 0)
    if dist == 't' and params[0] >= 30:
        r.paso('Aproximación', 'Con muchos grados de libertad la t se parece a la normal estándar.', '', 1)
    if dist == 't':
        nu = params[0]
        r.extra('Media y varianza', r'E[T] = 0' + (r',\ V[T] = \frac{\nu}{\nu-2} = ' + _n(nu / (nu - 2), 8) if nu > 2 else ''))
    elif dist == 'chi2':
        r.extra('Media y varianza', r'E = k = ' + _n(params[0]) + r',\ V = 2k = ' + _n(2 * params[0]))
    r.valor = _F(val, 12)
    r.valores['decimal'] = _n(val, 12)
    r.resultado_latex = r'P \approx ' + _n(val, 10)
    r.comparar = ('numero', val, {'tol': 1e-5})
    r.grafico = _graf_densidad(dens, xa, xb, max(float(a), xa) if a != -sp.oo else xa, min(float(b), xb) if b != sp.oo else xb,
                               ymax=_ymax_densidad(pdf, xa, xb))
    return r


# ============================================================================
# Bayes y probabilidad total
# ============================================================================
_PAT_P = re.compile(r'^\s*P\s*\(\s*([^|)]+?)\s*(?:\|\s*([^)]+?)\s*)?\)\s*(?:=\s*(.*))?$', re.I)


def _base_complemento(nombre):
    """'no A' / "A'" / 'A^c' / 'Ac' / '¬A' / '~A' / 'Ā' -> 'A' ; si no es complemento -> None."""
    s_ = nombre.strip()
    for pat in (r"^(?:no|not)\s+(.+)$", r"^[¬~!]\s*(.+)$", r"^(.+?)\s*'$", r"^(.+?)\s*\^\s*[cC]$", r"^(.+?)[̄̅]$",
                r"^(.+?)ᶜ$", r"^(.+?)c$"):
        m = re.match(pat, s_)
        if m:
            return m.group(1).strip()
    return None


def _tex_ev(nombre, comp=False):
    base = nombre if re.fullmatch(r'[A-Za-z][A-Za-z0-9]*', nombre) else r'\text{' + nombre + '}'
    if re.fullmatch(r'[A-Za-z]\d+', nombre):
        base = nombre[0] + '_{' + nombre[1:] + '}'
    return r'\overline{' + base + '}' if comp else base


def _leer_bayes(arg):
    """Devuelve (hipotesis[(nombre_latex, nombre_texto)], priors, verosimilitudes, evidencia_latex, objetivo_indice)."""
    texto = arg.strip()
    d_low = sin_tildes(texto.lower())
    # 1) Prueba diagnóstica con palabras
    if re.search(r'prevalencia|sensibilidad|especificidad|falsos? positivos?', d_low):
        d, _ = _kv(texto)
        prev = _get(d, 'prevalencia', 'p', 'priori')
        sens = _get(d, 'sensibilidad', 'verdaderospositivos', 'vp')
        esp = _get(d, 'especificidad')
        fp = _get(d, 'falsospositivos', 'falsopositivo', 'tasadefalsospositivos', 'fp')
        if prev is None or sens is None or (esp is None and fp is None):
            raise ErrorEntrada('Para una prueba diagnóstica indicá prevalencia, sensibilidad y especificidad (o falsos positivos). '
                               'Ej.: bayes prevalencia=0.01, sensibilidad=0.95, falsos positivos=0.05')
        pa = _prob_exacta(prev, 'prevalencia')
        lik = [_prob_exacta(sens, 'sensibilidad'), (1 - _prob_exacta(esp, 'especificidad')) if fp is None else _prob_exacta(fp, 'falsos positivos')]
        return [('E', 'E (tiene la condición)'), (r'\overline{E}', 'no E')], [pa, 1 - pa], lik, '+', 0
    partes = dividir(texto.replace(';', ','))
    # 2) Listas: [priors], [verosimilitudes]
    if len(partes) >= 2 and partes[0].strip().startswith('[') and partes[1].strip().startswith('['):
        pri = _datos(partes[0])
        lik = _datos(partes[1])
        if len(pri) != len(lik):
            raise ErrorEntrada('Las dos listas deben tener la misma cantidad de elementos: [P(A1), P(A2), ...], [P(B|A1), P(B|A2), ...].')
        hip = [(f'A_{{{i + 1}}}', f'A{i + 1}') for i in range(len(pri))]
        return hip, pri, lik, 'B', None
    # 3) Tres números: P(A), P(B|A), P(B|no A)
    if len(partes) == 3 and all(not _PAT_P.match(p) and '=' not in p for p in partes):
        pa, b1, b2 = (_prob_exacta(p) for p in partes)
        return [('A', 'A'), (r'\overline{A}', 'no A')], [pa, 1 - pa], [b1, b2], 'B', 0
    # 4) Notación P(A)=..., P(B|A)=...
    entradas, objetivo = [], None
    for p in partes:
        m = _PAT_P.match(p)
        if not m:
            raise ErrorEntrada(f'No entendí «{p}». Escribí, por ejemplo: bayes P(A)=0.01, P(B|A)=0.95, P(B|no A)=0.05')
        ev, cond, v = m.group(1), m.group(2), (m.group(3) or '').strip()
        if v in ('', '?'):
            objetivo = (ev.strip(), (cond or '').strip())
            continue
        entradas.append((ev.strip(), cond.strip() if cond else None, _prob_exacta(v, p.split('=')[0].strip())))
    nombres = set()
    for ev, cond, _ in entradas:
        nombres.add(ev)
        if cond:
            nombres.add(cond)

    def canon(nm):
        b = _base_complemento(nm)
        if b is None:
            return nm, False
        if b in nombres:
            return b, True
        if re.match(r'^.+c$', nm) and not re.search(r"'|\^|ᶜ|[¬~!]|^no\s|^not\s", nm):
            return nm, False  # 'Ac' solo es complemento si también aparece 'A'
        return b, True

    condicionales = [(canon(ev), canon(c), v) for ev, c, v in entradas if c]
    if not condicionales:
        raise ErrorEntrada('Faltan las probabilidades condicionales P(B|A) para aplicar Bayes.')
    evid = condicionales[0][0][0]
    hips = []
    for (e_, ce), (h, hc), v in condicionales:
        if e_ != evid:
            raise ErrorEntrada(f'Todas las condicionales deben ser del mismo suceso (P({evid}|·)); revisá P({e_}|...).')
        if (h, hc) not in hips:
            hips.append((h, hc))
    lik = {}
    for (e_, ce), hk, v in condicionales:
        lik[hk] = 1 - v if ce else v
    priors = {}
    for ev, c, v in entradas:
        if c is None:
            priors[canon(ev)] = v
    bases = {h for h, _ in hips}
    if len(bases) == 1 and len(hips) <= 2:
        h = hips[0][0]
        hk, hck = (h, False), (h, True)
        if hk not in priors and hck in priors:
            priors[hk] = 1 - priors[hck]
        if hk not in priors:
            raise ErrorEntrada(f'Falta P({h}).')
        priors[hck] = 1 - priors[hk]
        faltan = [k_ for k_ in (hk, hck) if k_ not in lik]
        if faltan:
            nom = faltan[0][0] if not faltan[0][1] else 'no ' + faltan[0][0]
            raise ErrorEntrada(f'Falta P({evid}|{nom}).')
        orden = [hk, hck]
    else:
        orden = hips
        faltan = [k_ for k_ in orden if k_ not in priors]
        if len(faltan) == 1:
            priors[faltan[0]] = 1 - sum(priors[k_] for k_ in orden if k_ in priors)
        elif faltan:
            raise ErrorEntrada('Faltan las probabilidades a priori de: ' + ', '.join(k_[0] for k_ in faltan) + '.')
    suma = sum(priors[k_] for k_ in orden)
    if abs(float(suma) - 1) > 1e-9:
        raise ErrorEntrada(f'Las probabilidades a priori deben sumar 1 (suman {_n(suma, 8)}): las hipótesis tienen que ser una partición.')
    hip = [(_tex_ev(h, c), ('no ' + h) if c else h) for h, c in orden]
    idx = None
    if objetivo:
        oh = canon(objetivo[0])
        idx = orden.index(oh) if oh in orden else None
    elif len(orden) == 2 and orden[1] == (orden[0][0], True):
        idx = 0
    return hip, [priors[k_] for k_ in orden], [lik[k_] for k_ in orden], _tex_ev(evid), idx


def _bayes_tabla(r, hip, pri, lik, ev):
    prod = [a * b for a, b in zip(pri, lik)]
    pb = sum(prod)
    if pb == 0:
        raise ErrorEntrada(f'P({ev}) = 0: el suceso observado es imposible con esos datos.')
    post = [p_ / pb for p_ in prod]
    r.tabla = {'columnas': ['Hipótesis', 'P(Hᵢ)', f'P({ev}|Hᵢ)', f'P(Hᵢ)·P({ev}|Hᵢ)', f'P(Hᵢ|{ev})'],
               'filas': [[h[1], _n(a, 8), _n(b, 8), _n(c, 8), _n(d, 8)] for h, a, b, c, d in zip(hip, pri, lik, prod, post)]
               + [['Total', _n(sum(pri), 8), '', _n(pb, 8), _n(sum(post), 8)]]}
    r.paso('Datos', 'Las hipótesis forman una partición (sus probabilidades suman 1).',
           r',\ '.join(f'P({h[0]}) = {tex(a)}' for h, a in zip(hip, pri)) + r';\quad '
           + r',\ '.join(f'P({ev}\\mid {h[0]}) = {tex(b)}' for h, b in zip(hip, lik)), 0)
    r.paso('Probabilidad total', r'\(P(B) = \sum_i P(H_i)\,P(B\mid H_i)\)',
           f'P({ev}) = ' + ' + '.join(f'{tex(a)}\\cdot {tex(b)}' for a, b in zip(pri, lik)) + ' = ' + tex(pb) + r'\approx ' + _n(pb, 10), 0)
    return prod, pb, post


def bayes(arg):
    hip, pri, lik, ev, idx = _leer_bayes(arg)
    r = Resultado(TEMA_P, 'Teorema de Bayes')
    prod, pb, post = _bayes_tabla(r, hip, pri, lik, ev)
    r.entrada_latex = r',\ '.join(f'P({h[0]}) = {tex(a)}' for h, a in zip(hip, pri)) + r';\ ' + \
        r',\ '.join(f'P({ev}\\mid {h[0]}) = {tex(b)}' for h, b in zip(hip, lik))
    r.paso('Teorema de Bayes', r'\(P(H_i\mid B) = \dfrac{P(H_i)\,P(B\mid H_i)}{P(B)}\)', '', 0)
    for i, (h, a, b, q) in enumerate(zip(hip, pri, lik, post)):
        r.paso(f'P({h[1]}|{ev})', '', f'P({h[0]}\\mid {ev}) = \\frac{{{tex(a)}\\cdot {tex(b)}}}{{{tex(pb)}}} = {tex(q)}\\approx {_n(q, 10)}',
               0 if (idx is None or i == idx) else 1)
    pnb = 1 - pb
    if pnb != 0:
        neg = [a * (1 - b) / pnb for a, b in zip(pri, lik)]
        r.extra('Si NO ocurre ' + ev, r',\ '.join(f'P({h[0]}\\mid \\overline{{{ev}}}) \\approx {_n(q, 8)}' for h, q in zip(hip, neg)))
    for h, q in zip(hip, post):
        r.valores[f'P({h[1]}|{ev})'] = _n(q, 12)
    r.valores['P(evidencia)'] = _n(pb, 12)
    if idx is not None:
        q = post[idx]
        r.valor = q
        r.valores['decimal'] = _n(q, 12)
        r.resultado_latex = f'P({hip[idx][0]}\\mid {ev}) = ' + tex(q) + r'\approx ' + _n(q, 10)
        r.comparar = ('numero', q, {'tol': 1e-6})
        if len(hip) == 2 and hip[0][0] == 'E':
            _extra(r, 'Interpretación', '', r'\text{Solo el } ' + _n(100 * float(q), 4) + r'\,\%\text{ de los positivos tiene realmente la condición.}')
    else:
        r.valor = list(post)
        r.resultado_latex = r',\ '.join(f'P({h[0]}\\mid {ev}) \\approx {_n(q, 8)}' for h, q in zip(hip, post))
        r.comparar = ('vector', list(post), {})
    r.grafico = {'tipo': 'barras', 'barras': [{'x': i + 1, 'alto': float(q), 'resaltar': i == idx} for i, q in enumerate(post)]}
    return r


def probabilidad_total(arg):
    hip, pri, lik, ev, _ = _leer_bayes(arg)
    r = Resultado(TEMA_P, 'Probabilidad total')
    prod, pb, post = _bayes_tabla(r, hip, pri, lik, ev)
    r.entrada_latex = r',\ '.join(f'P({h[0]}) = {tex(a)}' for h, a in zip(hip, pri)) + r';\ ' + \
        r',\ '.join(f'P({ev}\\mid {h[0]}) = {tex(b)}' for h, b in zip(hip, lik))
    r.extra('Bayes (a posteriori)', r',\ '.join(f'P({h[0]}\\mid {ev}) \\approx {_n(q, 8)}' for h, q in zip(hip, post)))
    r.valor = pb
    r.valores['decimal'] = _n(pb, 12)
    r.resultado_latex = f'P({ev}) = ' + tex(pb) + r'\approx ' + _n(pb, 10)
    r.comparar = ('numero', pb, {'tol': 1e-6})
    return r


# ============================================================================
# Estadística descriptiva extendida
# ============================================================================
def _perc_lineal(orden, p):
    """Percentil por interpolación lineal (posición 1 + (n−1)p/100; PERCENTIL.INC de Excel). p en [0, 100]."""
    n_ = len(orden)
    h = (n_ - 1) * sp.nsimplify(p) / 100
    lo = int(sp.floor(h))
    fr = h - lo
    if lo >= n_ - 1:
        return orden[-1], h, lo, fr
    return orden[lo] + fr * (orden[lo + 1] - orden[lo]), h, lo, fr


def _perc_n1(orden, p):
    """Método (n+1)p/100 (muy usado en los apuntes)."""
    n_ = len(orden)
    pos = (n_ + 1) * sp.nsimplify(p) / 100
    if pos <= 1:
        return orden[0], pos
    if pos >= n_:
        return orden[-1], pos
    lo = int(sp.floor(pos))
    return orden[lo - 1] + (pos - lo) * (orden[lo] - orden[lo - 1]), pos


def _ordenar(datos):
    return sorted(datos, key=lambda v: float(v))


def percentil(arg):
    partes = dividir(arg.replace(';', ','))
    if len(partes) < 2:
        raise ErrorEntrada('Uso: percentil p, d1, d2, ...   (p entre 0 y 100; ej.: percentil 90, 2, 4, 4, 5, 7, 9)')
    ptxt = re.sub(r'^[Pp]\s*', '', partes[0].strip()).replace('%', '')
    p = _val(ptxt)
    if 0 < p < 1 and not p.is_integer:
        p = p * 100
    if not 0 <= p <= 100:
        raise ErrorEntrada('El percentil debe estar entre 0 y 100.')
    datos = _datos(partes[1]) if partes[1].strip().startswith('[') else [_val(q) for q in partes[1:]]
    if len(datos) < 2:
        raise ErrorEntrada('Ingresá al menos 2 datos después del percentil.')
    orden = _ordenar(datos)
    n_ = len(orden)
    val, h, lo, fr = _perc_lineal(orden, p)
    r = Resultado(TEMA_E, f'Percentil {_n(p)}')
    r.entrada_latex = f'P_{{{_n(p)}}}' + r'\ \text{de}\ \{' + ', '.join(tex(d) for d in datos) + r'\}'
    r.paso('Ordenar los datos', f'n = {n_}', tex(orden), 0)
    r.paso('Posición (interpolación lineal)', r'Se usa la posición \(1 + (n-1)\frac{p}{100}\) (método lineal, como PERCENTIL.INC de Excel).',
           r'1 + (' + str(n_) + r' - 1)\cdot\frac{' + _n(p) + '}{100} = ' + tex(1 + h) + r'\approx ' + _n(1 + h, 8), 0)
    if fr != 0 and lo < n_ - 1:
        r.paso('Interpolar', f'Entre el dato {lo + 1}° ({tex(orden[lo])}) y el {lo + 2}° ({tex(orden[lo + 1])}):',
               f'P_{{{_n(p)}}} = {tex(orden[lo])} + {tex(fr)}\\cdot({tex(orden[lo + 1])} - {tex(orden[lo])}) = {tex(val)}', 0)
    else:
        r.paso('Valor', f'La posición es exacta: es el dato {min(lo, n_ - 1) + 1}°.', f'P_{{{_n(p)}}} = {tex(val)}', 0)
    alt, pos = _perc_n1(orden, p)
    _extra(r, 'Otro método', r'Con la posición \((n+1)\frac{p}{100}\) (otros apuntes y calculadoras):',
            r'\text{posición } ' + tex(pos) + r'\Rightarrow P = ' + tex(alt) + r'\approx ' + _n(alt, 8))
    r.valor = val
    r.valores['decimal'] = _n(val, 12)
    r.valores['metodo_n+1'] = _n(alt, 12)
    r.resultado_latex = f'P_{{{_n(p)}}} = ' + tex(val) + (r'\approx ' + _n(val, 8) if not val.is_Integer else '')
    r.comparar = ('numero', val, {'tol': 1e-6})
    r.grafico = {'tipo': 'barras', 'barras': [{'x': float(d), 'alto': c, 'resaltar': False}
                                              for d, c in sorted({d: orden.count(d) for d in orden}.items(), key=lambda q: float(q[0]))],
                 'media': float(val)}
    return r


def cuartiles(arg):
    datos = _datos(arg)
    if len(datos) < 3:
        raise ErrorEntrada('Uso: cuartiles d1, d2, d3, ...  (al menos 3 datos)')
    orden = _ordenar(datos)
    n_ = len(orden)
    q1, q2, q3 = (_perc_lineal(orden, p)[0] for p in (25, 50, 75))
    iqr = q3 - q1
    li, ls = q1 - sp.Rational(3, 2) * iqr, q3 + sp.Rational(3, 2) * iqr
    atip = [d for d in orden if d < li or d > ls]
    normales = [d for d in orden if li <= d <= ls]
    wmin, wmax = normales[0], normales[-1]
    r = Resultado(TEMA_E, 'Cuartiles y diagrama de caja')
    r.entrada_latex = r'\{' + ', '.join(tex(d) for d in datos) + r'\}'
    r.paso('Ordenar los datos', f'n = {n_}', tex(orden), 0)
    r.paso('Cuartiles (interpolación lineal)', r'Posición de \(Q_k\): \(1 + (n-1)\frac{k}{4}\)',
           r'Q_1 = ' + tex(q1) + r',\ Q_2 = Me = ' + tex(q2) + r',\ Q_3 = ' + tex(q3), 0)
    r.paso('Rango intercuartílico', '', r'RIC = Q_3 - Q_1 = ' + tex(iqr), 0)
    r.paso('Vallas (bigotes)', 'Los datos fuera de [Q₁ − 1,5·RIC; Q₃ + 1,5·RIC] son atípicos.',
           r'[' + tex(li) + r',\ ' + tex(ls) + r']\Rightarrow \text{atípicos: }' + (tex(atip) if atip else r'\text{ninguno}'), 0)
    r.paso('Resumen de cinco números', '', r'\min = ' + tex(orden[0]) + r',\ Q_1 = ' + tex(q1) + r',\ Me = ' + tex(q2)
           + r',\ Q_3 = ' + tex(q3) + r',\ \max = ' + tex(orden[-1]), 0)
    # Método de las mitades (Tukey), el de la cursada de muchos profesores
    mitad_inf, mitad_sup = orden[:n_ // 2], orden[(n_ + 1) // 2:]
    med = lambda L: L[len(L) // 2] if len(L) % 2 else (L[len(L) // 2 - 1] + L[len(L) // 2]) / 2  # noqa
    _extra(r, 'Otro método (medianas de cada mitad)', '', r'Q_1 = ' + tex(med(mitad_inf)) + r',\ Q_3 = ' + tex(med(mitad_sup)))
    r.valor = [q1, q2, q3]
    r.valores.update({'q1': _n(q1, 12), 'mediana': _n(q2, 12), 'q3': _n(q3, 12), 'ric': _n(iqr, 12),
                      'atipicos': ', '.join(_n(a, 10) for a in atip) or 'ninguno'})
    r.resultado_latex = r'Q_1 = ' + tex(q1) + r',\ Q_2 = ' + tex(q2) + r',\ Q_3 = ' + tex(q3) + r',\ RIC = ' + tex(iqr)
    r.comparar = ('vector', [q1, q2, q3], {})
    f = [float(v) for v in (wmin, q1, q2, q3, wmax)]
    trazos = [(f[0], 1), (f[1], 1), None, (f[3], 1), (f[4], 1), None,
              (f[1], 0.7), (f[1], 1.3), (f[3], 1.3), (f[3], 0.7), (f[1], 0.7), None,
              (f[2], 0.7), (f[2], 1.3), None, (f[0], 0.85), (f[0], 1.15), None, (f[4], 0.85), (f[4], 1.15)]
    curva = _curva_puntos(trazos, 'caja')
    ancho = (float(orden[-1]) - float(orden[0])) or 1.0
    r.grafico = {'tipo': '2d', 'curvas': [curva],
                 'puntos': [G.punto(a, 1, 'atípico', '#dc2626') for a in atip],
                 'sombra': G.sombra_entre(sp.Float(1.3), sp.Float(0.7), q1, q3),
                 'ventana': {'xmin': float(orden[0]) - 0.15 * ancho, 'xmax': float(orden[-1]) + 0.15 * ancho, 'ymin': 0.0, 'ymax': 2.0}}
    return r


def _momentos(datos):
    n_ = len(datos)
    media = sum(datos) / sp.Integer(n_)
    dev = [d - media for d in datos]
    sums = [sum(dv ** k_ for dv in dev) for k_ in (2, 3, 4)]
    return n_, media, dev, sums


def _tabla_momentos(r, datos, dev):
    r.tabla = {'columnas': ['xᵢ', 'xᵢ − x̄', '(xᵢ − x̄)²', '(xᵢ − x̄)³', '(xᵢ − x̄)⁴'],
               'filas': [[_n(d, 8), _n(e, 8), _n(e ** 2, 8), _n(e ** 3, 8), _n(e ** 4, 8)] for d, e in zip(datos, dev)]
               + [['Σ', _n(sum(dev), 6), _n(sum(e ** 2 for e in dev), 8), _n(sum(e ** 3 for e in dev), 8), _n(sum(e ** 4 for e in dev), 8)]]}


def asimetria(arg):
    datos = _datos(arg)
    if len(datos) < 3:
        raise ErrorEntrada('Uso: asimetria d1, d2, d3, ...  (al menos 3 datos)')
    n_, media, dev, (S2, S3, S4) = _momentos(datos)
    if S2 == 0:
        raise ErrorEntrada('Todos los datos son iguales: la asimetría no está definida.')
    s = sp.sqrt(S2 / (n_ - 1))
    m2, m3 = S2 / n_, S3 / n_
    g1 = m3 / m2 ** sp.Rational(3, 2)
    G1 = sp.nsimplify(n_) / ((n_ - 1) * (n_ - 2)) * S3 / s ** 3
    G1v = float(sp.N(G1, 20))
    r = Resultado(TEMA_E, 'Coeficiente de asimetría')
    r.entrada_latex = r'\{' + ', '.join(tex(d) for d in datos) + r'\}'
    _tabla_momentos(r, datos, dev)
    r.paso('Media y desvío muestral', '', r'\bar x = ' + tex(media) + r'\approx ' + _n(media, 8) + r',\quad s = \sqrt{\frac{\sum(x_i-\bar x)^2}{n-1}} = '
           + _n(s, 10), 0)
    r.paso('Coeficiente de asimetría muestral (ajustado)', 'Es el de Excel/Calc (COEFICIENTE.ASIMETRIA) y el de SPSS; equivale a '
           r'\(G_1 = g_1\frac{\sqrt{n(n-1)}}{n-2}\).',
           r'G_1 = \frac{n}{(n-1)(n-2)}\sum\left(\frac{x_i-\bar x}{s}\right)^3 = \frac{' + str(n_) + '}{' + str((n_ - 1) * (n_ - 2)) + r'}\cdot\frac{'
           + _n(S3, 8) + '}{' + _n(s ** 3, 8) + r'} \approx ' + _n(G1v, 10), 0)
    interp = ('positiva (cola a la derecha: media > mediana)' if G1v > 1e-12 else
              'negativa (cola a la izquierda)' if G1v < -1e-12 else 'nula (distribución simétrica)')
    r.paso('Interpretación', 'Asimetría ' + interp + '.', '', 0)
    _extra(r, 'Versión poblacional (sesgada)', r'Fisher-Pearson: \(g_1 = \dfrac{m_3}{m_2^{3/2}}\), con \(m_k = \frac1n\sum(x_i-\bar x)^k\)',
            r'g_1 \approx ' + _n(g1, 10))
    orden = _ordenar(datos)
    me = orden[n_ // 2] if n_ % 2 else (orden[n_ // 2 - 1] + orden[n_ // 2]) / 2
    _extra(r, 'Coeficiente de Pearson', '', r'A_P = \frac{3(\bar x - Me)}{s} \approx ' + _n(3 * (media - me) / s, 8))
    r.valor = _F(G1v, 12)
    r.valores.update({'G1': _n(G1v, 12), 'g1': _n(g1, 12), 'media': _n(media, 12), 's': _n(s, 12)})
    r.resultado_latex = r'G_1 \approx ' + _n(G1v, 10)
    r.comparar = ('numero', G1v, {'tol': 1e-6})
    return r


def curtosis(arg):
    datos = _datos(arg)
    if len(datos) < 4:
        raise ErrorEntrada('Uso: curtosis d1, d2, d3, d4, ...  (al menos 4 datos)')
    n_, media, dev, (S2, S3, S4) = _momentos(datos)
    if S2 == 0:
        raise ErrorEntrada('Todos los datos son iguales: la curtosis no está definida.')
    s2 = S2 / (n_ - 1)
    m2, m4 = S2 / n_, S4 / n_
    g2 = m4 / m2 ** 2 - 3
    G2 = sp.Rational(n_ * (n_ + 1), (n_ - 1) * (n_ - 2) * (n_ - 3)) * S4 / s2 ** 2 - sp.Rational(3 * (n_ - 1) ** 2, (n_ - 2) * (n_ - 3))
    G2v = float(sp.N(G2, 20))
    r = Resultado(TEMA_E, 'Curtosis (exceso)')
    r.entrada_latex = r'\{' + ', '.join(tex(d) for d in datos) + r'\}'
    _tabla_momentos(r, datos, dev)
    r.paso('Media y varianza muestral', '', r'\bar x = ' + tex(media) + r',\quad s^2 = \frac{\sum(x_i-\bar x)^2}{n-1} = ' + tex(s2)
           + r'\approx ' + _n(s2, 10), 0)
    r.paso('Exceso de curtosis muestral', 'Fórmula de Excel/Calc (CURTOSIS) y SPSS; vale 0 para la normal.',
           r'G_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)}\sum\left(\frac{x_i-\bar x}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}', 0)
    r.paso('Reemplazo', '', r'G_2 = \frac{' + str(n_ * (n_ + 1)) + '}{' + str((n_ - 1) * (n_ - 2) * (n_ - 3)) + r'}\cdot\frac{' + _n(S4, 8)
           + '}{' + _n(s2 ** 2, 8) + r'} - \frac{' + str(3 * (n_ - 1) ** 2) + '}{' + str((n_ - 2) * (n_ - 3)) + r'} \approx ' + _n(G2v, 10), 0)
    interp = ('leptocúrtica (más apuntada que la normal)' if G2v > 1e-12 else
              'platicúrtica (más achatada que la normal)' if G2v < -1e-12 else 'mesocúrtica (como la normal)')
    r.paso('Interpretación', 'Distribución ' + interp + '.', '', 0)
    _extra(r, 'Versión poblacional', r'\(g_2 = \dfrac{m_4}{m_2^2} - 3\); sin restar 3 es el coeficiente de curtosis \(\beta_2\).',
            r'g_2 \approx ' + _n(g2, 10) + r',\quad \beta_2 = \frac{m_4}{m_2^2} \approx ' + _n(g2 + 3, 10))
    r.valor = _F(G2v, 12)
    r.valores.update({'G2': _n(G2v, 12), 'g2': _n(g2, 12), 'beta2': _n(g2 + 3, 12)})
    r.resultado_latex = r'G_2 \approx ' + _n(G2v, 10)
    r.comparar = ('numero', G2v, {'tol': 1e-6})
    return r


def _decimales(v):
    for dd in range(0, 7):
        if (v * 10 ** dd).is_integer:
            return dd
    return 6


def tabla_frecuencias(arg):
    trozos = arg.split(';')
    cuerpo, k_cls, amplitud = [], None, None
    for tz in trozos:
        tl = sin_tildes(tz.lower()).strip()
        m = re.match(r'^(clases|k|intervalos|numero de clases)\s*[=:]?\s*(.+)$', tl)
        m2 = re.match(r'^(amplitud|ancho|c)\s*[=:]?\s*(.+)$', tl)
        if m:
            k_cls = _entero(m.group(2), 'la cantidad de clases', 1)
        elif m2:
            amplitud = _val(m2.group(2))
            if amplitud <= 0:
                raise ErrorEntrada('La amplitud debe ser positiva.')
        elif tz.strip():
            cuerpo.append(tz)
    datos = _datos(','.join(cuerpo))
    if len(datos) < 2:
        raise ErrorEntrada('Uso: tabla de frecuencias d1, d2, ...[; clases k][; amplitud c]')
    orden = _ordenar(datos)
    n_ = len(orden)
    r = Resultado(TEMA_E, 'Tabla de frecuencias')
    r.entrada_latex = r'\{' + ', '.join(tex(d) for d in datos[:40]) + (r',\ \dots' if n_ > 40 else '') + r'\},\ n = ' + str(n_)
    distintos = sorted(set(orden), key=float)
    filas, barras, F_, f_lista = [], [], 0, []
    if k_cls is None and amplitud is None and len(distintos) <= 12:
        r.paso('Datos discretos', f'Hay {len(distintos)} valores distintos: se cuenta cuántas veces aparece cada uno.', tex(orden), 0)
        for v in distintos:
            fi = orden.count(v)
            F_ += fi
            f_lista.append(fi)
            filas.append([_n(v, 8), str(fi), _n(sp.Rational(fi, n_), 6), str(F_), _n(sp.Rational(F_, n_), 6), _n(100 * sp.Rational(fi, n_), 5)])
            barras.append({'x': float(v), 'alto': fi})
        r.tabla = {'columnas': ['xᵢ', 'fᵢ', 'frᵢ', 'Fᵢ', 'Frᵢ', '%'], 'filas': filas + [['Total', str(n_), '1', '', '', '100']]}
        media = sum(v * orden.count(v) for v in distintos) / sp.Integer(n_)
        fmax = max(f_lista)
        _extra(r, 'Moda', '', 'Mo = ' + ', '.join(tex(v) for v, fi in zip(distintos, f_lista) if fi == fmax))
    else:
        mn, mx = orden[0], orden[-1]
        R = mx - mn
        if k_cls is None and amplitud is None:
            k_cls = max(1, int(round(1 + 3.322 * math.log10(n_))))
            r.paso('Cantidad de clases (Sturges)', '', r'k = 1 + 3{,}322\log_{10} n = 1 + 3{,}322\log_{10}' + str(n_) + r'\approx ' + str(k_cls), 0)
        dec = max(_decimales(d) for d in orden)
        if amplitud is None:
            c = sp.ceiling(R / k_cls * 10 ** dec) / sp.Integer(10 ** dec) if R != 0 else sp.Integer(1)
            if c == 0:
                c = sp.Integer(1) / 10 ** dec
            r.paso('Rango y amplitud', 'La amplitud se redondea hacia arriba a la precisión de los datos.',
                   r'R = ' + tex(mx) + ' - ' + tex(mn) + ' = ' + tex(R) + r',\quad c = \frac{R}{k} = \frac{' + tex(R) + '}{' + str(k_cls)
                   + r'} \to ' + tex(c), 0)
        else:
            c = amplitud
            k_cls = max(1, int(sp.ceiling(R / c))) if R != 0 else 1
            r.paso('Rango y clases', '', r'R = ' + tex(R) + r',\quad c = ' + tex(c) + r',\quad k = \lceil R/c \rceil = ' + str(k_cls), 0)
        while mn + k_cls * c < mx:
            k_cls += 1
        lims = [(mn + i * c, mn + (i + 1) * c) for i in range(k_cls)]
        marcas = []
        for i, (a, b) in enumerate(lims):
            ultima = i == k_cls - 1
            fi = sum(1 for d in orden if a <= d < b or (ultima and d == b))
            F_ += fi
            f_lista.append(fi)
            xm = (a + b) / 2
            marcas.append(xm)
            filas.append([f'[{_n(a, 8)}; {_n(b, 8)}' + (']' if ultima else ')'), _n(xm, 8), str(fi), _n(sp.Rational(fi, n_), 6), str(F_),
                          _n(sp.Rational(F_, n_), 6), _n(100 * sp.Rational(fi, n_), 5)])
            barras.append({'x': float(xm), 'alto': fi})
        r.paso('Clases', 'Intervalos cerrados a izquierda y abiertos a derecha (el último, cerrado).',
               ',\\ '.join(f'[{tex(a)}, {tex(b)})' for a, b in lims), 0)
        r.tabla = {'columnas': ['Clase', 'xᵢ (marca)', 'fᵢ', 'frᵢ', 'Fᵢ', 'Frᵢ', '%'], 'filas': filas + [['Total', '', str(n_), '1', '', '', '100']]}
        media = sum(m_ * fi for m_, fi in zip(marcas, f_lista)) / sp.Integer(n_)
        var = sum(fi * (m_ - media) ** 2 for m_, fi in zip(marcas, f_lista)) / sp.Integer(n_ - 1) if n_ > 1 else 0
        _extra(r, 'Medidas con datos agrupados', r'Usando las marcas de clase: \(\bar x = \frac{\sum x_i f_i}{n}\)',
                r'\bar x \approx ' + _n(media, 8) + r',\quad s^2 \approx ' + _n(var, 8))
        imax = f_lista.index(max(f_lista))
        r.extra('Clase modal', '', filas[imax][0])
    r.paso('Frecuencias', r'fᵢ: absoluta; frᵢ = fᵢ/n: relativa; Fᵢ: absoluta acumulada; Frᵢ: relativa acumulada.', '', 0)
    r.valor = f_lista
    r.valores.update({'n': n_, 'media': _n(media, 12), 'clases': len(f_lista)})
    r.resultado_latex = r'f = (' + ', '.join(str(fi) for fi in f_lista) + r'),\quad n = ' + str(n_)
    r.comparar = ('vector', f_lista, {})
    r.grafico = {'tipo': 'barras', 'barras': barras}
    return r


# ============================================================================
# Inferencia: intervalos de confianza y pruebas de hipótesis
# ============================================================================
def _nivel(d, pos, defecto=0.95):
    v = _get(d, 'conf', 'confianza', 'nc', 'nivel', 'niveldeconfianza', '1-alpha', '1-alfa', 'gamma')
    if v is not None:
        return _prob(v, 'El nivel de confianza')
    a = _get(d, 'alpha', 'alfa', 'significancia', 'niveldesignificancia')
    if a is not None:
        return 1 - _prob(a, 'α')
    for p in pos:
        if '%' in p:
            return _prob(p, 'El nivel de confianza')
    for p in pos:
        try:
            v = _fl(p)
        except ErrorEntrada:
            continue
        if 0 < v < 1 or 50 <= v < 100:
            return _prob(p, 'El nivel de confianza')
    return defecto


def _alpha(d, pos, defecto=0.05):
    a = _get(d, 'alpha', 'alfa', 'significancia', 'niveldesignificancia', 'a')
    if a is not None:
        return _prob(a, 'α')
    v = _get(d, 'conf', 'confianza', 'nc', 'nivel')
    if v is not None:
        return 1 - _prob(v, 'El nivel de confianza')
    for p in pos:
        if '%' in p:
            return _prob(p, 'α')
    return defecto


def _req(d, nombre, *claves, uso=''):
    v = _get(d, *claves)
    if v is None:
        raise ErrorEntrada(f'Falta {nombre}. Uso: {uso}')
    return _val(v)


def _graf_intervalo(dist, params, crit):
    with mp.workdps(20):
        cdf, pdf, dens, (a, b), _, _ = _dist(dist, params)
        ymax = _ymax_densidad(pdf, a, b)
    return _graf_densidad(dens, a, b, -crit, crit, [G.punto(-crit, 0, _n(-crit, 5), '#dc2626'), G.punto(crit, 0, _n(crit, 5), '#dc2626')],
                          ymax=ymax)


def _cierre_intervalo(r, li, ls, nombre_par, extra_valores=None):
    r.valor = [_F(li), _F(ls)]
    r.valores.update({'li': _n(li, 12), 'ls': _n(ls, 12)})
    if extra_valores:
        r.valores.update(extra_valores)
    r.resultado_latex = nombre_par + r'\in [' + _n(li, 8) + r',\ ' + _n(ls, 8) + ']'
    r.paso('Intervalo', '', r.resultado_latex, 0)
    r.comparar = ('vector', [float(li), float(ls)], {})


def intervalo_media(arg):
    uso = 'intervalo media x̄=50, σ=5, n=36, 95%   (σ conocido → z)   ;   intervalo media x̄=50, s=5, n=16, 95%   (s muestral → t)'
    d, pos = _kv(arg)
    if not d and len(pos) >= 3:
        d = {'x': pos[0], 'sigma': pos[1], 'n': pos[2]}
        pos = pos[3:]
    xb = _req(d, 'la media muestral x̄', 'x', 'xbarra', 'media', 'promedio', 'xm', uso=uso)
    n_ = _entero(_get(d, 'n') or '0', 'n', 2) if _get(d, 'n') else None
    if n_ is None:
        raise ErrorEntrada('Falta n (tamaño de la muestra). Uso: ' + uso)
    conf = _nivel(d, pos)
    alpha = 1 - conf
    sg = _get(d, 'sigma', 'desviopoblacional')
    sg2 = _get(d, 'sigma2', 'varianzapoblacional')
    s = _get(d, 's', 'desvio', 'desviomuestral', 'sd', 'de')
    s2 = _get(d, 's2', 'varianza', 'varianzamuestral')
    r = Resultado(TEMA_E, 'Intervalo de confianza para la media')
    if sg is not None or sg2 is not None:
        sv = _val(sg) if sg is not None else sp.sqrt(_val(sg2))
        crit = float(_q_normal(1 - alpha / 2))
        dist, params, simb, nombre_c = 'normal', (0, 1), r'\sigma', f'z_{{{_n(1 - alpha / 2, 6)}}}'
        r.paso('Método', 'σ poblacional conocido: se usa la normal estándar.', r'\bar x \pm z_{1-\alpha/2}\frac{\sigma}{\sqrt n}', 0)
    elif s is not None or s2 is not None:
        sv = _val(s) if s is not None else sp.sqrt(_val(s2))
        crit = float(_q_t(1 - alpha / 2, n_ - 1))
        dist, params, simb, nombre_c = 't', (n_ - 1,), 's', f't_{{{_n(1 - alpha / 2, 6)};\\,{n_ - 1}}}'
        r.paso('Método', f'σ desconocido (se usa s): t de Student con n − 1 = {n_ - 1} grados de libertad.',
               r'\bar x \pm t_{1-\alpha/2;\,n-1}\frac{s}{\sqrt n}', 0)
    else:
        raise ErrorEntrada('Falta el desvío: σ (poblacional, usa z) o s (muestral, usa t). Uso: ' + uso)
    if sv <= 0:
        raise ErrorEntrada('El desvío debe ser positivo.')
    r.entrada_latex = r'\bar x = ' + tex(xb) + r',\ ' + simb + ' = ' + tex(sv) + r',\ n = ' + str(n_) + r',\ 1-\alpha = ' + _n(conf, 6)
    ee = float(sv) / math.sqrt(n_)
    E = crit * ee
    li, ls = float(xb) - E, float(xb) + E
    r.paso('Valor crítico', f'α = {_n(alpha, 6)}, se deja α/2 en cada cola.', nombre_c + ' = ' + _n(crit, 8), 0)
    r.paso('Error estándar', '', r'\frac{' + simb + r'}{\sqrt n} = \frac{' + tex(sv) + r'}{\sqrt{' + str(n_) + '}} = ' + _n(ee, 8), 0)
    r.paso('Margen de error', '', r'E = ' + _n(crit, 8) + r'\cdot ' + _n(ee, 8) + ' = ' + _n(E, 8), 0)
    if dist == 't' and n_ >= 30:
        _extra(r, 'Muestra grande', 'Con n ≥ 30 muchos apuntes usan z en lugar de t:',
                r'\bar x \pm ' + _n(float(_q_normal(1 - alpha / 2)), 6) + r'\cdot ' + _n(ee, 6))
    _cierre_intervalo(r, li, ls, r'\mu', {'E': _n(E, 12), 'critico': _n(crit, 12)})
    r.grafico = _graf_intervalo(dist, params, crit)
    return r


def intervalo_varianza(arg):
    uso = 'intervalo varianza s2=25, n=16, 95%   (o s=5)'
    d, pos = _kv(arg)
    if not d and len(pos) >= 2:
        d = {'s2': pos[0], 'n': pos[1]}
        pos = pos[2:]
    s2 = _get(d, 's2', 'varianza', 'varianzamuestral')
    s = _get(d, 's', 'desvio', 'desviomuestral')
    if s2 is None and s is None:
        raise ErrorEntrada('Falta la varianza muestral s² (o el desvío s). Uso: ' + uso)
    s2v = _val(s2) if s2 is not None else _val(s) ** 2
    if _get(d, 'n') is None:
        raise ErrorEntrada('Falta n. Uso: ' + uso)
    n_ = _entero(_get(d, 'n'), 'n', 2)
    conf = _nivel(d, pos)
    alpha = 1 - conf
    gl = n_ - 1
    c_inf = float(_q_chi2(alpha / 2, gl))
    c_sup = float(_q_chi2(1 - alpha / 2, gl))
    r = Resultado(TEMA_E, 'Intervalo de confianza para la varianza')
    r.entrada_latex = r's^2 = ' + tex(s2v) + r',\ n = ' + str(n_) + r',\ 1-\alpha = ' + _n(conf, 6)
    r.paso('Estadístico', 'Para una población normal:', r'\frac{(n-1)s^2}{\sigma^2}\sim\chi^2_{n-1}', 0)
    r.paso('Valores críticos', f'gl = n − 1 = {gl}', r'\chi^2_{' + _n(alpha / 2, 6) + r'} = ' + _n(c_inf, 8) + r',\quad \chi^2_{'
           + _n(1 - alpha / 2, 6) + '} = ' + _n(c_sup, 8), 0)
    num = float(gl * s2v)
    li, ls = num / c_sup, num / c_inf
    r.paso('Fórmula', '', r'\left[\frac{(n-1)s^2}{\chi^2_{1-\alpha/2}},\ \frac{(n-1)s^2}{\chi^2_{\alpha/2}}\right] = \left[\frac{' + _n(num, 8)
           + '}{' + _n(c_sup, 8) + r'},\ \frac{' + _n(num, 8) + '}{' + _n(c_inf, 8) + r'}\right]', 0)
    _extra(r, 'Intervalo para σ', 'Raíz cuadrada de los extremos:', r'\sigma\in[' + _n(math.sqrt(li), 8) + r',\ ' + _n(math.sqrt(ls), 8) + ']')
    _cierre_intervalo(r, li, ls, r'\sigma^2', {'sigma_li': _n(math.sqrt(li), 12), 'sigma_ls': _n(math.sqrt(ls), 12)})
    with mp.workdps(20):
        cdf, pdf, dens, (a, b), _, _ = _dist('chi2', (gl,))
        r.grafico = _graf_densidad(dens, a, b, c_inf, c_sup, [G.punto(c_inf, 0, _n(c_inf, 5), '#dc2626'),
                                                             G.punto(c_sup, 0, _n(c_sup, 5), '#dc2626')], ymax=_ymax_densidad(pdf, a, b))
    return r


def intervalo_proporcion(arg):
    uso = 'intervalo proporcion p=0.4, n=100, 95%   (o x=40, n=100)'
    d, pos = _kv(arg)
    if not d and len(pos) >= 2:
        d = {'p': pos[0], 'n': pos[1]}
        pos = pos[2:]
    if _get(d, 'n') is None:
        raise ErrorEntrada('Falta n. Uso: ' + uso)
    n_ = _entero(_get(d, 'n'), 'n', 1)
    p = _get(d, 'p', 'phat', 'proporcion')
    if p is not None:
        pv = _prob_exacta(p, 'p̂')
    elif _get(d, 'x', 'exitos') is not None:
        pv = _val(_get(d, 'x', 'exitos')) / n_
    else:
        raise ErrorEntrada('Falta la proporción muestral p̂ (o la cantidad de éxitos x). Uso: ' + uso)
    conf = _nivel(d, pos)
    alpha = 1 - conf
    crit = float(_q_normal(1 - alpha / 2))
    r = Resultado(TEMA_E, 'Intervalo de confianza para una proporción')
    r.entrada_latex = r'\hat p = ' + tex(pv) + r',\ n = ' + str(n_) + r',\ 1-\alpha = ' + _n(conf, 6)
    ee = math.sqrt(float(pv * (1 - pv)) / n_)
    E = crit * ee
    r.paso('Método', 'Aproximación normal (Wald), válida si n·p̂ ≥ 5 y n·(1 − p̂) ≥ 5.',
           r'\hat p \pm z_{1-\alpha/2}\sqrt{\frac{\hat p(1-\hat p)}{n}}', 0)
    if n_ * pv < 5 or n_ * (1 - pv) < 5:
        r.avisos.append('n·p̂ o n·(1 − p̂) es menor que 5: la aproximación normal no es buena.')
    r.paso('Valor crítico', '', f'z_{{{_n(1 - alpha / 2, 6)}}} = ' + _n(crit, 8), 0)
    r.paso('Error estándar y margen', '', r'\sqrt{\frac{' + _n(pv, 6) + r'\cdot ' + _n(1 - pv, 6) + '}{' + str(n_) + '}} = ' + _n(ee, 8)
           + r',\quad E = ' + _n(E, 8), 0)
    _cierre_intervalo(r, float(pv) - E, float(pv) + E, 'p', {'E': _n(E, 12), 'critico': _n(crit, 12)})
    r.grafico = _graf_intervalo('normal', (0, 1), crit)
    return r


def tamano_muestra(arg):
    uso = 'tamano de muestra σ=5, E=1, 95%   (media)   ;   tamano de muestra p=0.5, E=0.03, 95%   (proporción)'
    d, pos = _kv(arg)
    E = _get(d, 'e', 'error', 'margen', 'margendeerror')
    if E is None:
        raise ErrorEntrada('Falta el margen de error E. Uso: ' + uso)
    Ev = _val(E)
    if Ev <= 0:
        raise ErrorEntrada('E debe ser positivo.')
    conf = _nivel(d, pos)
    alpha = 1 - conf
    z = float(_q_normal(1 - alpha / 2))
    r = Resultado(TEMA_E, 'Tamaño de muestra')
    sg = _get(d, 'sigma', 's', 'desvio')
    if sg is not None:
        sv = _val(sg)
        n0 = (z * float(sv) / float(Ev)) ** 2
        r.entrada_latex = r'\sigma = ' + tex(sv) + r',\ E = ' + tex(Ev) + r',\ 1-\alpha = ' + _n(conf, 6)
        r.paso('Fórmula (media)', r'De \(E = z\frac{\sigma}{\sqrt n}\):', r'n = \left(\frac{z_{1-\alpha/2}\,\sigma}{E}\right)^2 = \left(\frac{'
               + _n(z, 6) + r'\cdot ' + tex(sv) + '}{' + tex(Ev) + r'}\right)^2 = ' + _n(n0, 8), 0)
    else:
        p = _get(d, 'p', 'phat', 'proporcion')
        pv = _prob_exacta(p, 'p') if p is not None else sp.Rational(1, 2)
        if p is None:
            r.avisos.append('Sin estimación previa de p se usa p = 0,5 (caso más desfavorable).')
        n0 = z ** 2 * float(pv * (1 - pv)) / float(Ev) ** 2
        r.entrada_latex = r'\hat p = ' + tex(pv) + r',\ E = ' + tex(Ev) + r',\ 1-\alpha = ' + _n(conf, 6)
        r.paso('Fórmula (proporción)', '', r'n = \frac{z^2\,\hat p(1-\hat p)}{E^2} = \frac{' + _n(z, 6) + r'^2\cdot ' + _n(pv * (1 - pv), 6)
               + '}{' + tex(Ev) + '^2} = ' + _n(n0, 8), 0)
    nn = int(math.ceil(n0 - 1e-9))
    r.paso('Redondear hacia arriba', 'Siempre se redondea al entero siguiente para asegurar el margen pedido.', f'n = {nn}', 0)
    r.valor = sp.Integer(nn)
    r.valores['n_exacto'] = _n(n0, 12)
    r.resultado_latex = f'n = {nn}'
    r.comparar = ('numero', sp.Integer(nn), {})
    return r


def _cola(d, pos):
    h1 = _get(d, 'h1', 'ha', 'cola', 'alternativa', 'tipo', 'prueba')
    for tx in [h1 or ''] + list(pos):
        tl = sin_tildes(normalizar(tx)).lower()
        if re.search(r'bilateral|dos colas|distinto|!=|<>', tl):
            return 'bilateral'
        if '>' in tl or re.search(r'derech|mayor|superior', tl):
            return 'derecha'
        if '<' in tl or re.search(r'izquier|menor|inferior', tl):
            return 'izquierda'
    return 'bilateral'


def _hipotesis(r, cola, par_tex, h0_val):
    signo = {'bilateral': r'\ne', 'derecha': '>', 'izquierda': '<'}[cola]
    r.paso('Hipótesis', 'Prueba ' + ('bilateral (dos colas)' if cola == 'bilateral' else 'unilateral (cola ' + cola + ')')
           + '. Para otra cola agregá «cola derecha» o «cola izquierda».',
           r'H_0: ' + par_tex + ' = ' + h0_val + r'\qquad H_1: ' + par_tex + ' ' + signo + ' ' + h0_val, 0)


def _prueba(r, est, dist, params, alpha, cola, simb, par_tex, h0_val):
    """Completa una prueba de hipótesis: valor crítico, valor p, decisión y gráfico."""
    est_f = float(est)
    with mp.workdps(DPS):
        cdf, pdf, dens, (a, b), _, _ = _dist(dist, params)
        qf = (lambda p: _q_normal(p)) if dist == 'normal' else (lambda p: _q_t(p, params[0]))
        if cola == 'bilateral':
            crit = float(qf(1 - alpha / 2))
            pval = float(2 * (1 - cdf(abs(est_f))))
            rechaza = abs(est_f) > crit
            region = f'|{simb}| > {_n(crit, 6)}'
            arriba = sp.Piecewise((dens, (x <= -crit) | (x >= crit)), (0, True))
        elif cola == 'derecha':
            crit = float(qf(1 - alpha))
            pval = float(1 - cdf(est_f))
            rechaza = est_f > crit
            region = f'{simb} > {_n(crit, 6)}'
            arriba = sp.Piecewise((dens, x >= crit), (0, True))
        else:
            crit = -float(qf(1 - alpha))
            pval = float(cdf(est_f))
            rechaza = est_f < crit
            region = f'{simb} < {_n(crit, 6)}'
            arriba = sp.Piecewise((dens, x <= crit), (0, True))
        ymax = _ymax_densidad(pdf, a, b)
    r.paso('Región de rechazo', f'α = {_n(alpha, 6)}', r'\text{Se rechaza } H_0 \text{ si } ' + region, 0)
    r.paso('Valor p', 'Probabilidad de obtener un estadístico tan extremo como el observado si H₀ es cierta.',
           r'p = ' + _n(pval, 8), 0)
    decision = 'Se rechaza H0' if rechaza else 'No se rechaza H0'
    r.paso('Decisión', (f'{simb} = {_n(est_f, 6)} cae en la región de rechazo y p = {_n(pval, 4)} < α = {_n(alpha, 4)}.' if rechaza else
                        f'{simb} = {_n(est_f, 6)} no cae en la región de rechazo y p = {_n(pval, 4)} ≥ α = {_n(alpha, 4)}.'),
           r'\textbf{' + ('Se rechaza' if rechaza else 'No se rechaza') + r'}\ H_0', 0)
    r.valores.update({'p': _n(pval, 12), 'critico': _n(crit, 12), 'decision': decision, 'cola': cola})
    r.resultado_latex = simb + ' = ' + (tex(est) if isinstance(est, sp.Rational) else _n(est_f, 8)) + r',\quad p \approx ' + _n(pval, 6) \
        + r'\ \Rightarrow\ \text{' + ('Se rechaza' if rechaza else 'No se rechaza') + r'}\ H_0'
    r.valor = est if isinstance(est, sp.Rational) else _F(est_f, 12)
    r.comparar = ('numero', est, {'tol': 1e-5})
    puntos = [G.punto(est_f, 0, f'{simb} = {_n(est_f, 4)}', '#dc2626'), G.punto(crit, 0, _n(crit, 4), '#111827')]
    if cola == 'bilateral':
        puntos.append(G.punto(-crit, 0, _n(-crit, 4), '#111827'))
    lim = max(abs(a), abs(est_f) * 1.15)
    r.grafico = _graf_densidad(dens, -lim, lim, -lim, lim, puntos, ymax=ymax, arriba=arriba)


def prueba_media(arg, tipo):
    uso = (f'prueba {tipo} x̄=52, μ0=50, ' + ('σ=5' if tipo == 'z' else 's=5') + ', n=36, α=0.05[, cola derecha | cola izquierda]')
    d, pos = _kv(arg)
    if not d and len(pos) >= 4:
        d = {'x': pos[0], 'mu0': pos[1], 'sigma' if tipo == 'z' else 's': pos[2], 'n': pos[3]}
        pos = pos[4:]
    xb = _req(d, 'la media muestral x̄', 'x', 'xbarra', 'media', 'promedio', uso=uso)
    mu0 = _req(d, 'μ0 (valor de la hipótesis nula)', 'mu0', 'mu', 'm0', 'h0', uso=uso)
    if _get(d, 'n') is None:
        raise ErrorEntrada('Falta n. Uso: ' + uso)
    n_ = _entero(_get(d, 'n'), 'n', 2)
    if tipo == 'z':
        sg = _get(d, 'sigma', 's', 'desvio')
        if sg is None and _get(d, 'sigma2') is not None:
            sv = sp.sqrt(_val(_get(d, 'sigma2')))
        elif sg is None:
            raise ErrorEntrada('Falta σ. Uso: ' + uso)
        else:
            sv = _val(sg)
        simb_d = r'\sigma'
    else:
        sg = _get(d, 's', 'desvio', 'sigma')
        if sg is None and _get(d, 's2', 'varianza') is not None:
            sv = sp.sqrt(_val(_get(d, 's2', 'varianza')))
        elif sg is None:
            raise ErrorEntrada('Falta s. Uso: ' + uso)
        else:
            sv = _val(sg)
        simb_d = 's'
    if sv <= 0:
        raise ErrorEntrada('El desvío debe ser positivo.')
    alpha = _alpha(d, pos)
    cola = _cola(d, pos)
    r = Resultado(TEMA_E, 'Prueba ' + ('z' if tipo == 'z' else 't') + ' para la media')
    r.entrada_latex = r'\bar x = ' + tex(xb) + r',\ \mu_0 = ' + tex(mu0) + r',\ ' + simb_d + ' = ' + tex(sv) + r',\ n = ' + str(n_) \
        + r',\ \alpha = ' + _n(alpha, 6)
    est = sp.nsimplify((xb - mu0) / (sv / sp.sqrt(n_)))
    est_v = est if est.is_Rational else sp.Float(sp.N(est, 20), 15)
    simb = 'z' if tipo == 'z' else 't'
    _hipotesis(r, cola, r'\mu', tex(mu0))
    r.paso('Estadístico de prueba', ('σ conocido: normal estándar.' if tipo == 'z' else f't de Student con n − 1 = {n_ - 1} grados de libertad.'),
           simb + r' = \frac{\bar x - \mu_0}{' + simb_d + r'/\sqrt n} = \frac{' + tex(xb) + ' - ' + tex(mu0) + '}{' + tex(sv) + r'/\sqrt{' + str(n_)
           + '}} = ' + (tex(est) + r'\approx ' if not est.is_Integer else '') + _n(est_v, 8), 0)
    _prueba(r, est_v, 'normal' if tipo == 'z' else 't', (0, 1) if tipo == 'z' else (n_ - 1,), alpha, cola, simb, r'\mu', tex(mu0))
    return r


def prueba_proporcion(arg):
    uso = 'prueba proporcion p=0.45, p0=0.4, n=100, α=0.05[, cola derecha]   (o x=45 en lugar de p)'
    d, pos = _kv(arg)
    if _get(d, 'n') is None:
        raise ErrorEntrada('Falta n. Uso: ' + uso)
    n_ = _entero(_get(d, 'n'), 'n', 1)
    if _get(d, 'p', 'phat') is not None:
        pv = _prob_exacta(_get(d, 'p', 'phat'), 'p̂')
    elif _get(d, 'x', 'exitos') is not None:
        pv = _val(_get(d, 'x', 'exitos')) / n_
    else:
        raise ErrorEntrada('Falta p̂ (o x). Uso: ' + uso)
    p0 = _get(d, 'p0', 'pi0', 'h0')
    if p0 is None:
        raise ErrorEntrada('Falta p0 (valor de la hipótesis nula). Uso: ' + uso)
    p0 = _prob_exacta(p0, 'p0')
    if p0 in (0, 1):
        raise ErrorEntrada('p0 debe estar estrictamente entre 0 y 1.')
    alpha = _alpha(d, pos)
    cola = _cola(d, pos)
    r = Resultado(TEMA_E, 'Prueba z para una proporción')
    r.entrada_latex = r'\hat p = ' + tex(pv) + r',\ p_0 = ' + tex(p0) + r',\ n = ' + str(n_) + r',\ \alpha = ' + _n(alpha, 6)
    est = sp.nsimplify((pv - p0) / sp.sqrt(p0 * (1 - p0) / n_))
    est_v = est if est.is_Rational else sp.Float(sp.N(est, 20), 15)
    _hipotesis(r, cola, 'p', tex(p0))
    r.paso('Estadístico de prueba', 'Aproximación normal (n·p₀ ≥ 5 y n·(1 − p₀) ≥ 5).',
           r'z = \frac{\hat p - p_0}{\sqrt{p_0(1-p_0)/n}} = ' + _n(est_v, 8), 0)
    _prueba(r, est_v, 'normal', (0, 1), alpha, cola, 'z', 'p', tex(p0))
    return r


def _listas_chi2(arg):
    obs = esp = probs = None
    resto = []
    for p in dividir(arg.replace(';', ',')):
        pl = sin_tildes(p.lower()).strip()
        if '[' in p:
            clave = re.sub(r'[\s:=]+$', '', pl[:pl.index('[')]).strip()
            lista = _datos(p[p.index('['):].strip())
            if clave.startswith('obs') or clave in ('o', 'fo'):
                obs = lista
            elif clave.startswith('esp') or clave in ('e', 'fe'):
                esp = lista
            elif clave.startswith('prob') or clave.startswith('propor') or clave in ('p', 'pi'):
                probs = lista
            elif obs is None:
                obs = lista
            elif esp is None and probs is None:
                esp = lista
        else:
            resto.append(p)
    return obs, esp, probs, ','.join(resto)


def chi2_bondad(arg):
    uso = 'chi2 bondad observados [10,20,30], esperados [20,20,20][, α=0.05]   (o probabilidades [..]; sin esperados = uniforme)'
    obs, esp, probs, resto = _listas_chi2(arg)
    if not obs:
        raise ErrorEntrada('Faltan las frecuencias observadas. Uso: ' + uso)
    d, pos = _kv(resto)
    alpha = _alpha(d, pos)
    m_est = _entero(_get(d, 'm', 'estimados', 'parametrosestimados', 'parametros') or '0', 'm', 0)
    k_ = len(obs)
    N_ = sum(obs)
    r = Resultado(TEMA_E, 'Prueba chi cuadrado de bondad de ajuste')
    if probs is not None:
        if abs(float(sum(probs)) - 1) > 1e-9:
            raise ErrorEntrada(f'Las probabilidades deben sumar 1 (suman {_n(sum(probs), 8)}).')
        esp = [N_ * p_ for p_ in probs]
        r.paso('Frecuencias esperadas', r'\(E_i = N\,p_i\)', f'N = {tex(N_)}', 0)
    elif esp is None:
        esp = [N_ / sp.Integer(k_)] * k_
        r.paso('Frecuencias esperadas', 'Sin esperados se supone distribución uniforme: N/k en cada categoría.', r'E_i = ' + tex(esp[0]), 0)
    elif abs(float(sum(esp)) - 1) < 1e-9 and N_ != 1:
        esp = [N_ * p_ for p_ in esp]
        r.avisos.append('Los esperados suman 1: se tomaron como probabilidades (Eᵢ = N·pᵢ).')
    if len(esp) != k_:
        raise ErrorEntrada('Observados y esperados deben tener la misma cantidad de categorías.')
    if any(e <= 0 for e in esp):
        raise ErrorEntrada('Las frecuencias esperadas deben ser positivas.')
    if abs(float(sum(esp) - N_)) > 1e-9 * float(abs(N_) + 1):
        r.avisos.append(f'La suma de esperados ({_n(sum(esp), 8)}) no coincide con la de observados ({_n(N_, 8)}).')
    if any(e < 5 for e in esp):
        r.avisos.append('Hay frecuencias esperadas menores que 5: conviene agrupar categorías.')
    terms = [(o - e) ** 2 / e for o, e in zip(obs, esp)]
    chi = sum(terms)
    gl = k_ - 1 - m_est
    if gl < 1:
        raise ErrorEntrada('Los grados de libertad deben ser al menos 1 (k − 1 − parámetros estimados).')
    r.entrada_latex = r'O = (' + ', '.join(tex(o) for o in obs) + r'),\ E = (' + ', '.join(tex(e) for e in esp) + ')'
    r.tabla = {'columnas': ['i', 'Oᵢ', 'Eᵢ', 'Oᵢ − Eᵢ', '(Oᵢ − Eᵢ)²/Eᵢ'],
               'filas': [[str(i + 1), _n(o, 8), _n(e, 8), _n(o - e, 8), _n(tm, 8)] for i, (o, e, tm) in enumerate(zip(obs, esp, terms))]
               + [['Σ', _n(N_, 8), _n(sum(esp), 8), '', _n(chi, 10)]]}
    r.paso('Hipótesis', '', r'H_0:\ \text{los datos siguen la distribución propuesta}\qquad H_1:\ \text{no la siguen}', 0)
    r.paso('Estadístico', r'\(\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}\)', r'\chi^2 = ' + ' + '.join(tex(tm) for tm in terms[:8])
           + (r' + \dots' if k_ > 8 else '') + ' = ' + tex(chi) + (r'\approx ' + _n(chi, 8) if not chi.is_Integer else ''), 0)
    r.paso('Grados de libertad', 'k − 1 − (parámetros estimados)', f'gl = {k_} - 1' + (f' - {m_est}' if m_est else '') + f' = {gl}', 0)
    with mp.workdps(DPS):
        pval = float(1 - _chi2_cdf(float(chi), gl))
        crit = float(_q_chi2(1 - alpha, gl))
    rechaza = float(chi) > crit
    r.paso('Valor crítico y valor p', f'α = {_n(alpha, 6)}', r'\chi^2_{' + _n(1 - alpha, 6) + ';' + str(gl) + '} = ' + _n(crit, 8)
           + r',\quad p = P(\chi^2_{' + str(gl) + r'} > ' + _n(chi, 8) + ') = ' + _n(pval, 8), 0)
    decision = 'Se rechaza H0' if rechaza else 'No se rechaza H0'
    r.paso('Decisión', ('χ² supera el valor crítico (p < α): los datos NO se ajustan a la distribución propuesta.' if rechaza else
                        'χ² no supera el valor crítico (p ≥ α): no hay evidencia contra el ajuste.'),
           r'\textbf{' + ('Se rechaza' if rechaza else 'No se rechaza') + r'}\ H_0', 0)
    r.valor = chi
    r.valores.update({'gl': gl, 'p': _n(pval, 12), 'critico': _n(crit, 12), 'decision': decision})
    r.resultado_latex = r'\chi^2 = ' + tex(chi) + r',\ gl = ' + str(gl) + r',\ p \approx ' + _n(pval, 6) + r'\ \Rightarrow\ \text{' \
        + ('Se rechaza' if rechaza else 'No se rechaza') + r'}\ H_0'
    r.comparar = ('numero', chi, {'tol': 1e-6})
    with mp.workdps(20):
        cdf, pdf, dens, (a, b), _, _ = _dist('chi2', (gl,))
        b = max(b, float(chi) * 1.1)
        r.grafico = _graf_densidad(dens, a, b, crit, b, [G.punto(float(chi), 0, f'χ² = {_n(chi, 5)}', '#dc2626')],
                                   ymax=_ymax_densidad(pdf, a, b))
    return r


def chi2_independencia(arg):
    partes = dividir(arg.replace(';', ','))
    if not partes or not partes[0].strip().startswith('[['):
        raise ErrorEntrada('Uso: chi2 independencia [[10,20],[30,40]][, α=0.05]   (tabla de contingencia por filas)')
    M = leer_matriz(partes[0])
    d, pos = _kv(','.join(partes[1:]))
    alpha = _alpha(d, pos)
    f_, c_ = M.shape
    if f_ < 2 or c_ < 2:
        raise ErrorEntrada('La tabla de contingencia debe ser al menos de 2×2.')
    O = [[sp.nsimplify(M[i, j]) for j in range(c_)] for i in range(f_)]
    tf = [sum(O[i]) for i in range(f_)]
    tc = [sum(O[i][j] for i in range(f_)) for j in range(c_)]
    N_ = sum(tf)
    E = [[tf[i] * tc[j] / N_ for j in range(c_)] for i in range(f_)]
    if any(E[i][j] == 0 for i in range(f_) for j in range(c_)):
        raise ErrorEntrada('Hay una fila o columna con total 0.')
    chi = sum((O[i][j] - E[i][j]) ** 2 / E[i][j] for i in range(f_) for j in range(c_))
    gl = (f_ - 1) * (c_ - 1)
    r = Resultado(TEMA_E, 'Prueba chi cuadrado de independencia')
    r.entrada_latex = tex(sp.Matrix(O))
    r.paso('Hipótesis', '', r'H_0:\ \text{las variables son independientes}\qquad H_1:\ \text{están asociadas}', 0)
    r.paso('Frecuencias esperadas', r'\(E_{ij} = \dfrac{(\text{total fila } i)(\text{total columna } j)}{N}\)', 'E = ' + tex(sp.Matrix(E)), 0)
    r.tabla = {'columnas': [''] + [f'Col {j + 1}' for j in range(c_)] + ['Total'],
               'filas': [[f'Fila {i + 1}'] + [f'{_n(O[i][j], 8)} ({_n(E[i][j], 6)})' for j in range(c_)] + [_n(tf[i], 8)] for i in range(f_)]
               + [['Total'] + [_n(v, 8) for v in tc] + [_n(N_, 8)]]}
    if any(E[i][j] < 5 for i in range(f_) for j in range(c_)):
        r.avisos.append('Hay frecuencias esperadas menores que 5: la aproximación chi cuadrado puede no ser buena.')
    r.paso('Estadístico', r'\(\chi^2 = \sum_{i,j}\frac{(O_{ij} - E_{ij})^2}{E_{ij}}\)', r'\chi^2 = ' + tex(chi) + r'\approx ' + _n(chi, 8), 0)
    with mp.workdps(DPS):
        pval = float(1 - _chi2_cdf(float(chi), gl))
        crit = float(_q_chi2(1 - alpha, gl))
    rechaza = float(chi) > crit
    r.paso('Grados de libertad y decisión', f'gl = (filas − 1)(columnas − 1) = {gl}; α = {_n(alpha, 6)}',
           r'\chi^2_{crit} = ' + _n(crit, 8) + r',\ p = ' + _n(pval, 8) + r'\Rightarrow \textbf{' + ('Se rechaza' if rechaza else 'No se rechaza')
           + r'}\ H_0', 0)
    decision = 'Se rechaza H0' if rechaza else 'No se rechaza H0'
    r.valor = chi
    r.valores.update({'gl': gl, 'p': _n(pval, 12), 'critico': _n(crit, 12), 'decision': decision})
    r.resultado_latex = r'\chi^2 \approx ' + _n(chi, 8) + r',\ gl = ' + str(gl) + r',\ p \approx ' + _n(pval, 6)
    r.comparar = ('numero', chi, {'tol': 1e-6})
    return r


def correlacion(arg):
    pts, resto = [], []
    for p in dividir(arg):
        if '=' in p or '%' in p:
            resto.append(p)
        else:
            pts.append(p)
    pts = _puntos(','.join(pts))
    d, pos = _kv(','.join(resto))
    alpha = _alpha(d, pos)
    n_ = len(pts)
    if n_ < 3:
        raise ErrorEntrada('Se necesitan al menos 3 puntos (x, y).')
    xs = [a for a, _ in pts]
    ys = [b for _, b in pts]
    mx, my = sum(xs) / sp.Integer(n_), sum(ys) / sp.Integer(n_)
    Sxy = sum((a - mx) * (b - my) for a, b in pts)
    Sxx = sum((a - mx) ** 2 for a in xs)
    Syy = sum((b - my) ** 2 for b in ys)
    if Sxx == 0 or Syy == 0:
        raise ErrorEntrada('Todas las x (o todas las y) son iguales: la correlación no está definida.')
    rr = Sxy / sp.sqrt(Sxx * Syy)
    rv = float(sp.N(rr, 20))
    r = Resultado(TEMA_E, 'Correlación lineal de Pearson')
    r.entrada_latex = ', '.join(f'({tex(a)}, {tex(b)})' for a, b in pts)
    r.tabla = {'columnas': ['xᵢ', 'yᵢ', 'xᵢ − x̄', 'yᵢ − ȳ', '(xᵢ − x̄)(yᵢ − ȳ)'],
               'filas': [[_n(a, 8), _n(b, 8), _n(a - mx, 8), _n(b - my, 8), _n((a - mx) * (b - my), 8)] for a, b in pts]
               + [['Σ', '', _n(Sxx, 8) + ' (Sxx)', _n(Syy, 8) + ' (Syy)', _n(Sxy, 8)]]}
    r.paso('Medias', '', r'\bar x = ' + tex(mx) + r',\ \bar y = ' + tex(my), 0)
    r.paso('Coeficiente de correlación', '', r'r = \frac{S_{xy}}{\sqrt{S_{xx}S_{yy}}} = \frac{' + _n(Sxy, 8) + r'}{\sqrt{' + _n(Sxx, 8) + r'\cdot '
           + _n(Syy, 8) + '}} = ' + _n(rv, 10), 0)
    fuerza = abs(rv)
    desc = ('muy fuerte' if fuerza >= 0.9 else 'fuerte' if fuerza >= 0.7 else 'moderada' if fuerza >= 0.4 else 'débil')
    r.paso('Interpretación', f'Correlación {desc} y ' + ('positiva' if rv > 0 else 'negativa') + f'; R² = {_n(rv * rv, 6)} '
           f'({_n(100 * rv * rv, 4)} % de la variación de y se explica linealmente por x).', '', 0)
    gl = n_ - 2
    if gl >= 1 and abs(rv) < 1:
        tt = rv * math.sqrt(gl) / math.sqrt(1 - rv * rv)
        with mp.workdps(DPS):
            pval = float(2 * (1 - _t_cdf(abs(tt), gl)))
            crit = float(_q_t(1 - alpha / 2, gl))
        rechaza = abs(tt) > crit
        r.paso('Prueba de significancia', r'\(H_0: \rho = 0\) contra \(H_1: \rho \ne 0\); t de Student con n − 2 gl.',
               r't = \frac{r\sqrt{n-2}}{\sqrt{1-r^2}} = ' + _n(tt, 8) + r',\ t_{crit} = ' + _n(crit, 6) + r',\ p = ' + _n(pval, 6), 0)
        r.paso('Decisión', ('La correlación es significativa' if rechaza else 'La correlación NO es significativa') + f' (α = {_n(alpha, 4)}).',
               r'\textbf{' + ('Se rechaza' if rechaza else 'No se rechaza') + r'}\ H_0', 0)
        r.valores.update({'t': _n(tt, 12), 'p': _n(pval, 12), 'decision': 'Se rechaza H0' if rechaza else 'No se rechaza H0'})
    b1 = Sxy / Sxx
    b0 = my - b1 * mx
    recta = sp.nsimplify(b1) * x + sp.nsimplify(b0)
    _extra(r, 'Recta de regresión', '', 'y = ' + tex(recta))
    r.valor = _F(rv, 12)
    r.valores.update({'r': _n(rv, 12), 'r2': _n(rv * rv, 12)})
    r.resultado_latex = r'r \approx ' + _n(rv, 8) + r',\quad R^2 \approx ' + _n(rv * rv, 6)
    r.comparar = ('numero', rv, {'tol': 1e-6})
    r.grafico = G.grafico_datos(pts, recta, 'y = ' + tex(recta))
    return r


# ============================================================================
# Cálculo numérico (complementos de numerico.py)
# ============================================================================
def _fn(expr, *vs):
    return sp.lambdify(vs, expr, modules=['math'])


def _rn(titulo):
    return Resultado(TEMA_N, titulo)


def _evaluar_f(F, v, nombre='f'):
    try:
        r_ = F(v)
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ErrorEntrada(f'No se puede evaluar {nombre} en {fmt_num(v, 8)} (fuera del dominio).') from exc
    if isinstance(r_, complex):
        raise ErrorEntrada(f'{nombre}({fmt_num(v, 8)}) no es real.')
    return float(r_)


def regula_falsi(arg):
    partes = dividir(arg)
    if len(partes) < 3:
        raise ErrorEntrada('Uso: regula falsi f, a, b[, tolerancia]')
    f = leer(partes[0])
    v = variable_principal(f)
    a, b = _fl(partes[1]), _fl(partes[2])
    tol = _fl(partes[3]) if len(partes) > 3 else 1e-10
    a0, b0 = a, b
    F0 = _fn(f, v)
    F = lambda q: _evaluar_f(F0, q)  # noqa
    fa, fb = F(a), F(b)
    if fa * fb > 0:
        raise ErrorEntrada(f'f(a) = {fmt_num(fa, 6)} y f(b) = {fmt_num(fb, 6)} tienen el mismo signo: elegí un intervalo donde f cambie de signo.')
    r = _rn('Método de regula falsi (falsa posición)')
    r.entrada_latex = 'f(' + tex(v) + ') = ' + tex(f) + r',\quad [' + fmt_num(a) + ', ' + fmt_num(b) + ']'
    r.paso('Teorema de Bolzano', 'f es continua y cambia de signo en [a, b]: hay al menos una raíz.',
           f'f({fmt_num(a)}) = {fmt_num(fa, 6)},\\ f({fmt_num(b)}) = {fmt_num(fb, 6)}', 0)
    r.paso('Fórmula', 'En lugar del punto medio (bisección) se toma el corte con el eje x de la recta que une (a, f(a)) y (b, f(b)); '
           'luego se conserva el subintervalo donde f cambia de signo.', r'c = \frac{a\,f(b) - b\,f(a)}{f(b) - f(a)}', 0)
    filas, c_ant, c = [], None, a
    for i in range(1000):
        if fb == fa:
            break
        c = (a * fb - b * fa) / (fb - fa)
        fc = F(c)
        err = abs(c - c_ant) if c_ant is not None else None
        if i < 60:
            filas.append([str(i + 1), fmt_num(a, 12), fmt_num(b, 12), fmt_num(c, 12), fmt_num(fc, 6), fmt_num(err, 3) if err is not None else '—'])
        if fc == 0 or (err is not None and err < tol):
            break
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        c_ant = c
    else:
        r.avisos.append('Se alcanzó el máximo de iteraciones.')
    if i >= 60:
        r.avisos.append(f'Se muestran 60 de {i + 1} iteraciones.')
    r.tabla = {'columnas': ['n', 'a', 'b', 'c', 'f(c)', '|cₙ − cₙ₋₁|'], 'filas': filas}
    _cierre(r, f, v, c, i + 1, tol, a0, b0)
    return r


def _fmtc(z, dig=10):
    z = complex(z)
    if abs(z.imag) <= 1e-13 * max(1.0, abs(z.real)):
        return fmt_num(z.real, dig)
    return f'{z.real:.{dig}g}{z.imag:+.{dig}g}i'


def muller(arg):
    partes = dividir(arg)
    if len(partes) < 2:
        raise ErrorEntrada('Uso: muller f, x0, x1, x2[, tolerancia]   (o muller f, x0)')
    f = leer(partes[0])
    v = variable_principal(f)
    nums = [_fl(p) for p in partes[1:]]
    tol = 1e-10
    if len(nums) >= 3:
        p0, p1, p2 = nums[:3]
        if len(nums) > 3:
            tol = nums[3]
    elif len(nums) == 2:
        p0, p1 = nums
        p2 = (p0 + p1) / 2
    else:
        p0, p1, p2 = nums[0] - 0.5, nums[0] + 0.5, nums[0]
    if len({p0, p1, p2}) < 3:
        raise ErrorEntrada('Los tres puntos iniciales deben ser distintos.')
    F = sp.lambdify(v, f, modules='mpmath')
    r = _rn('Método de Müller')
    r.entrada_latex = 'f(' + tex(v) + ') = ' + tex(f) + r',\quad x_0 = ' + fmt_num(p0) + r',\ x_1 = ' + fmt_num(p1) + r',\ x_2 = ' + fmt_num(p2)
    r.paso('Idea', 'Se pasa una parábola por los tres últimos puntos y se toma su raíz más cercana a x₂ (puede dar raíces complejas).',
           r'h_0 = x_1 - x_0,\ h_1 = x_2 - x_1,\ \delta_0 = \frac{f(x_1)-f(x_0)}{h_0},\ \delta_1 = \frac{f(x_2)-f(x_1)}{h_1}', 0)
    r.paso('Coeficientes y nueva aproximación', 'Se elige el signo que hace más grande el denominador.',
           r'a = \frac{\delta_1 - \delta_0}{h_1 + h_0},\ b = a h_1 + \delta_1,\ c = f(x_2),\quad x_3 = x_2 - \frac{2c}{b \pm\sqrt{b^2 - 4ac}}', 0)
    filas = []
    with mp.workdps(30):
        x0_, x1_, x2_ = mp.mpc(p0), mp.mpc(p1), mp.mpc(p2)
        for i in range(100):
            try:
                f0, f1, f2 = F(x0_), F(x1_), F(x2_)
            except Exception as exc:  # noqa
                raise ErrorEntrada('No se pudo evaluar f en los puntos de la iteración.') from exc
            h0, h1 = x1_ - x0_, x2_ - x1_
            if h0 == 0 or h1 == 0 or h0 + h1 == 0:
                break
            d0, d1 = (f1 - f0) / h0, (f2 - f1) / h1
            a_ = (d1 - d0) / (h1 + h0)
            b_ = a_ * h1 + d1
            c_ = f2
            raiz = mp.sqrt(b_ * b_ - 4 * a_ * c_)
            den = b_ + raiz if abs(b_ + raiz) >= abs(b_ - raiz) else b_ - raiz
            if den == 0:
                raise ErrorEntrada('El denominador se anuló: probá con otros puntos iniciales.')
            x3_ = x2_ - 2 * c_ / den
            err = abs(x3_ - x2_)
            filas.append([str(i + 1), _fmtc(x0_), _fmtc(x1_), _fmtc(x2_), _fmtc(x3_, 12), _fmtc(F(x3_), 6), fmt_num(float(err), 3)])
            x0_, x1_, x2_ = x1_, x2_, x3_
            if err < tol:
                break
        else:
            r.avisos.append('No convergió en 100 iteraciones.')
        raiz_f = complex(x2_)
    r.tabla = {'columnas': ['n', 'x₀', 'x₁', 'x₂', 'x₃', 'f(x₃)', '|x₃ − x₂|'], 'filas': filas}
    if abs(raiz_f.imag) <= 1e-10 * max(1.0, abs(raiz_f.real)):
        _cierre(r, f, v, raiz_f.real, len(filas), tol)
    else:
        z = sp.Float(round(raiz_f.real, 10), 15) + sp.I * sp.Float(round(raiz_f.imag, 10), 15)
        r.valor = z
        r.valores['iteraciones'] = len(filas)
        r.resultado_latex = tex(v) + r'\approx ' + _fmtc(raiz_f, 12)
        r.paso('Resultado', 'La raíz encontrada es compleja (también lo es su conjugada si f tiene coeficientes reales).', r.resultado_latex, 0)
        r.comparar = ('numero', z, {'tol': 1e-6})
    return r


def _leer_vector(texto):
    e = leer(texto)
    if isinstance(e, (list, tuple, sp.Tuple)):
        return [sp.nsimplify(q) for q in e]
    if isinstance(e, sp.MatrixBase):
        return [sp.nsimplify(q) for q in e]
    raise ErrorEntrada('Se esperaba un vector, por ejemplo [6, 8, 9].')


def _despeje(A, b, i, n_, gs):
    num = tex(b[i])
    for j in range(n_):
        if j == i or A[i, j] == 0:
            continue
        sup = '(k+1)' if (gs and j < i) else '(k)'
        coef = A[i, j]
        mag = abs(coef)
        num += (' - ' if coef > 0 else ' + ') + ('' if mag == 1 else tex(mag)) + f'x_{{{j + 1}}}^{{{sup}}}'
    return f'x_{{{i + 1}}}^{{(k+1)}} = \\frac{{{num}}}{{{tex(A[i, i])}}}'


def _iterativo(arg, gs):
    nombre = 'gauss seidel' if gs else 'jacobi'
    partes = dividir(arg)
    if len(partes) < 2:
        raise ErrorEntrada(f'Uso: {nombre} A, b[, x0][, tolerancia o cantidad de iteraciones]   ej.: {nombre} [[4,1,1],[1,5,2],[1,2,6]], [6,8,9]')
    A = sp.Matrix(leer_matriz(partes[0])).applyfunc(sp.nsimplify)
    b = _leer_vector(partes[1])
    n_ = A.rows
    if A.cols != n_:
        raise ErrorEntrada('La matriz A debe ser cuadrada.')
    if len(b) != n_:
        raise ErrorEntrada(f'b debe tener {n_} componentes.')
    x0 = [0.0] * n_
    tol, nfijo = 1e-10, None
    for p in partes[2:]:
        clave, txt = (_nk(p.split('=', 1)[0]), p.split('=', 1)[1]) if '=' in p else (None, p)
        if txt.strip().startswith('['):
            x0 = [float(q) for q in _leer_vector(txt)]
            if len(x0) != n_:
                raise ErrorEntrada(f'x0 debe tener {n_} componentes.')
        elif clave in ('iter', 'iteraciones', 'n', 'k', 'it'):
            nfijo = _entero(txt, 'la cantidad de iteraciones', 1)
        elif clave in ('tol', 'tolerancia', 'e', 'error', 'eps'):
            tol = _fl(txt)
        else:
            vv = _fl(txt)
            if vv >= 1 and float(vv).is_integer():
                nfijo = int(vv)
            else:
                tol = vv
    if any(A[i, i] == 0 for i in range(n_)):
        raise ErrorEntrada('Hay un cero en la diagonal: reordená las ecuaciones (idealmente para que A sea diagonalmente dominante).')
    Af = [[float(A[i, j]) for j in range(n_)] for i in range(n_)]
    bf = [float(q) for q in b]
    r = _rn('Método de Gauss-Seidel' if gs else 'Método de Jacobi')
    r.entrada_latex = tex(A) + r'\,\mathbf{x} = ' + tex(sp.Matrix(b))
    dom = [abs(Af[i][i]) > sum(abs(Af[i][j]) for j in range(n_) if j != i) for i in range(n_)]
    r.paso('Convergencia', ('A es diagonalmente dominante por filas: el método converge para cualquier x⁽⁰⁾.' if all(dom) else
                            'A NO es diagonalmente dominante por filas: el método puede no converger (probá reordenar las ecuaciones).'),
           r',\ '.join(f'|{tex(A[i, i])}| ' + ('>' if dom[i] else r'\le') + ' ' + tex(sum(abs(A[i, j]) for j in range(n_) if j != i))
                       for i in range(n_)), 0)
    r.paso('Despejes', ('Cada incógnita se despeja de su ecuación; Gauss-Seidel usa los valores nuevos apenas se calculan.' if gs else
                        'Cada incógnita se despeja de su ecuación usando solo los valores de la iteración anterior.'),
           r'\\ '.join(_despeje(A, b, i, n_, gs) for i in range(n_)), 0)
    xk = list(x0)
    filas = [['0'] + [fmt_num(q, 10) for q in xk] + ['—']]
    maxit = nfijo or 1000
    convergio = False
    for k_ in range(1, maxit + 1):
        nuevo = list(xk)
        for i in range(n_):
            fuente = nuevo if gs else xk
            s_ = sum(Af[i][j] * fuente[j] for j in range(n_) if j != i)
            nuevo[i] = (bf[i] - s_) / Af[i][i]
        err = max(abs(nuevo[i] - xk[i]) for i in range(n_))
        xk = nuevo
        if k_ <= 60:
            filas.append([str(k_)] + [fmt_num(q, 10) for q in xk] + [fmt_num(err, 3)])
        if not all(math.isfinite(q) for q in xk) or err > 1e12:
            raise ErrorEntrada('El método diverge: la matriz no es diagonalmente dominante. Reordená las ecuaciones.')
        if nfijo is None and err < tol:
            convergio = True
            break
    if nfijo is None and not convergio:
        r.avisos.append('No convergió en 1000 iteraciones.')
    if k_ > 60:
        filas.append([str(k_)] + [fmt_num(q, 10) for q in xk] + [fmt_num(err, 3)])
        r.avisos.append(f'Se muestran las primeras 60 de {k_} iteraciones y la última.')
    r.tabla = {'columnas': ['k'] + [f'x{chr(0x2080 + i + 1) if i < 9 else i + 1}' for i in range(n_)] + ['‖Δx‖∞'], 'filas': filas}
    r.paso('Iteraciones', (f'Se hicieron las {nfijo} iteraciones pedidas.' if nfijo else
                           f'Se itera hasta que ‖x⁽ᵏ⁺¹⁾ − x⁽ᵏ⁾‖∞ < {tol:g} ({k_} iteraciones).'), '', 0)
    try:
        if A.det() != 0:
            exacta = A.LUsolve(sp.Matrix(b))
            r.extra('Solución exacta (para comparar)', r'\mathbf{x} = ' + tex(exacta.T),
                    'Error máximo: ' + fmt_num(max(abs(float(exacta[i]) - xk[i]) for i in range(n_)), 3))
    except Exception:  # noqa
        pass
    r.valor = [sp.Float(round(q, 10), 15) for q in xk]
    r.valores['iteraciones'] = k_
    r.resultado_latex = r'\mathbf{x}^{(' + str(k_) + r')} \approx (' + r',\ '.join(fmt_num(q, 10) for q in xk) + ')'
    r.comparar = ('vector', [sp.Float(q, 15) for q in xk], {})
    return r


def jacobi(arg):
    return _iterativo(arg, False)


def gauss_seidel(arg):
    return _iterativo(arg, True)


def lu(arg):
    partes = dividir(arg)
    if not partes:
        raise ErrorEntrada('Uso: lu A[, b]   ej.: lu [[4,3],[6,3]]')
    A = sp.Matrix(leer_matriz(partes[0])).applyfunc(sp.nsimplify)
    n_ = A.rows
    if A.cols != n_:
        raise ErrorEntrada('La factorización LU se aplica a matrices cuadradas.')
    b = _leer_vector(partes[1]) if len(partes) > 1 else None
    if b is not None and len(b) != n_:
        raise ErrorEntrada(f'b debe tener {n_} componentes.')
    U = A.copy()
    L = sp.eye(n_)
    P = sp.eye(n_)
    permuto = False
    r = _rn('Factorización LU (Doolittle)')
    r.entrada_latex = 'A = ' + tex(A) + (r',\quad \mathbf{b} = ' + tex(sp.Matrix(b)) if b is not None else '')
    r.paso('Idea', 'Eliminación gaussiana guardando los multiplicadores: L es triangular inferior con 1 en la diagonal y U es la '
           'matriz escalonada.', r'm_{ik} = \frac{u_{ik}}{u_{kk}},\quad F_i \leftarrow F_i - m_{ik}F_k,\quad l_{ik} = m_{ik}', 0)
    for k_ in range(n_ - 1):
        if U[k_, k_] == 0:
            piv = next((i for i in range(k_ + 1, n_) if U[i, k_] != 0), None)
            if piv is None:
                continue
            U.row_swap(k_, piv)
            P.row_swap(k_, piv)
            for j in range(k_):
                L[k_, j], L[piv, j] = L[piv, j], L[k_, j]
            permuto = True
            r.paso(f'Pivote nulo en la columna {k_ + 1}', f'Se intercambian las filas {k_ + 1} y {piv + 1} (se registra en P).', 'U = ' + tex(U), 1)
        ops = []
        for i in range(k_ + 1, n_):
            m = U[i, k_] / U[k_, k_]
            if m == 0:
                continue
            L[i, k_] = m
            U[i, :] = U[i, :] - m * U[k_, :]
            ops.append(f'm_{{{i + 1}{k_ + 1}}} = {tex(m)}:\\ F_{{{i + 1}}} \\leftarrow F_{{{i + 1}}} - ({tex(m)})F_{{{k_ + 1}}}')
        if ops:
            r.paso(f'Columna {k_ + 1}', '', r'\\ '.join(ops) + r'\\ U = ' + tex(U), 1)
    r.paso('Resultado', 'PA = LU (P registra los intercambios de filas).' if permuto else 'A = LU',
           ('P = ' + tex(P) + r',\ ' if permuto else '') + 'L = ' + tex(L) + r',\quad U = ' + tex(U), 0)
    r.paso('Verificación', '', 'LU = ' + tex(L * U) + (' = PA' if permuto else ' = A'), 1)
    det = U.det() * (P.det())
    _extra(r, 'Determinante', 'det(A) = ±(producto de la diagonal de U)', r'\det A = ' + tex(det))
    r.valores.update({'L': L, 'U': U, 'det': tex(det)})
    if permuto:
        r.valores['P'] = P
    if b is not None:
        if any(U[i, i] == 0 for i in range(n_)):
            raise ErrorEntrada('U tiene un cero en la diagonal: A es singular y el sistema no tiene solución única.')
        pb = P * sp.Matrix(b)
        yv = sp.zeros(n_, 1)
        for i in range(n_):
            yv[i] = pb[i] - sum(L[i, j] * yv[j] for j in range(i))
        xv = sp.zeros(n_, 1)
        for i in reversed(range(n_)):
            xv[i] = (yv[i] - sum(U[i, j] * xv[j] for j in range(i + 1, n_))) / U[i, i]
        r.paso('Sustitución hacia adelante', r'\(L\mathbf{y} = ' + ('P' if permuto else '') + r'\mathbf{b}\)', r'\mathbf{y} = ' + tex(yv.T), 0)
        r.paso('Sustitución hacia atrás', r'\(U\mathbf{x} = \mathbf{y}\)', r'\mathbf{x} = ' + tex(xv.T), 0)
        r.valor = list(xv)
        r.resultado_latex = r'\mathbf{x} = ' + tex(xv.T)
        r.comparar = ('vector', list(xv), {})
    else:
        r.valor = [L, U]
        r.resultado_latex = ('P = ' + tex(P) + r',\ ' if permuto else '') + 'L = ' + tex(L) + r',\quad U = ' + tex(U)
        r.comparar = ('matriz', U, {})
        r.avisos.append('Para «¿Está bien mi respuesta?» escribí la matriz U.')
    return r


def spline(arg):
    pts_txt, xev = [], None
    for p in dividir(arg.replace(';', ',')):
        if p.strip().startswith('('):
            pts_txt.append(p)
        else:
            xev = _val(re.sub(r'^\s*x\s*=', '', p))
    pts = sorted(_puntos(','.join(pts_txt)), key=lambda q: float(q[0]))
    if len(pts) < 2:
        raise ErrorEntrada('Uso: spline (x0,y0), (x1,y1), (x2,y2), ...[; x a evaluar]   (spline cúbica natural)')
    xs = [q[0] for q in pts]
    ys = [q[1] for q in pts]
    if len(set(xs)) != len(xs):
        raise ErrorEntrada('Hay dos puntos con la misma x.')
    n_ = len(pts) - 1
    h = [xs[i + 1] - xs[i] for i in range(n_)]
    r = _rn('Spline cúbica natural')
    r.entrada_latex = ', '.join(f'({tex(a)}, {tex(b)})' for a, b in pts)
    r.paso('Condiciones', 'En cada tramo un polinomio cúbico; S, S\' y S\'\' son continuas en los nodos y en los extremos S\'\' = 0 (natural).',
           r'M_i = S\'\'(x_i),\quad M_0 = M_{' + str(n_) + '} = 0,\quad h_i = x_{i+1} - x_i', 0)
    M = [sp.Integer(0)] * (n_ + 1)
    if n_ >= 2:
        filas_s, rhs = [], []
        for i in range(1, n_):
            fila = [0] * (n_ - 1)
            if i - 1 >= 1:
                fila[i - 2] = h[i - 1]
            fila[i - 1] = 2 * (h[i - 1] + h[i])
            if i + 1 <= n_ - 1:
                fila[i] = h[i]
            filas_s.append(fila)
            rhs.append(6 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]))
        S_ = sp.Matrix(filas_s)
        R_ = sp.Matrix(rhs)
        sol = S_.LUsolve(R_)
        for i in range(1, n_):
            M[i] = sp.nsimplify(sol[i - 1])
        r.paso('Sistema tridiagonal', r'\(h_{i-1}M_{i-1} + 2(h_{i-1}+h_i)M_i + h_iM_{i+1} = 6\left(\frac{y_{i+1}-y_i}{h_i} - \frac{y_i-y_{i-1}}{h_{i-1}}\right)\)',
               tex(S_) + tex(sp.Matrix([sp.Symbol(f'M_{i}') for i in range(1, n_)])) + ' = ' + tex(R_), 0)
        r.paso('Segundas derivadas', '', r',\ '.join(f'M_{{{i}}} = {tex(M[i])}' for i in range(n_ + 1)), 0)
    polys, filas, tramos = [], [], []
    for i in range(n_):
        ai = ys[i]
        bi = (ys[i + 1] - ys[i]) / h[i] - h[i] * (2 * M[i] + M[i + 1]) / 6
        ci = M[i] / 2
        di = (M[i + 1] - M[i]) / (6 * h[i])
        Si = sp.expand(ai + bi * (x - xs[i]) + ci * (x - xs[i]) ** 2 + di * (x - xs[i]) ** 3)
        polys.append(Si)
        tramos.append((Si, sp.And(x >= xs[i], x <= xs[i + 1])))
        filas.append([str(i), f'[{fmt_num(xs[i])}, {fmt_num(xs[i + 1])}]', tex(ai), tex(bi), tex(ci), tex(di)])
        r.paso(f'Tramo {i}: [{fmt_num(xs[i])}, {fmt_num(xs[i + 1])}]',
               r'\(S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3\)',
               f'S_{{{i}}}(x) = ' + tex(ai + bi * (x - xs[i]) + ci * (x - xs[i]) ** 2 + di * (x - xs[i]) ** 3) + ' = ' + tex(Si), 1)
    r.tabla = {'columnas': ['i', 'tramo', 'aᵢ', 'bᵢ', 'cᵢ', 'dᵢ'], 'filas': filas}
    pw = sp.Piecewise(*tramos, (sp.nan, True))
    r.grafico = G.grafico_datos(pts, pw, 'S(x)')
    if xev is not None:
        idx = next((i for i in range(n_) if xs[i] <= xev <= xs[i + 1]), None)
        if idx is None:
            idx = 0 if xev < xs[0] else n_ - 1
            r.avisos.append('El punto está fuera de los nodos: se extrapola con el tramo más cercano.')
        val = sp.nsimplify(polys[idx].subs(x, xev))
        r.paso('Evaluación', f'{fmt_num(xev)} pertenece al tramo {idx}.', f'S({tex(xev)}) = S_{{{idx}}}({tex(xev)}) = {tex(val)}\\approx {_n(val, 10)}', 0)
        r.valor = val
        r.valores['decimal'] = _n(val, 12)
        r.valores['polinomios'] = [str(q) for q in polys]
        r.resultado_latex = f'S({tex(xev)}) = ' + tex(val) + (r'\approx ' + _n(val, 10) if not val.is_Integer else '')
        r.comparar = ('numero', val, {'tol': 1e-8})
        try:
            r.grafico['puntos'].append(G.punto(xev, val, f'S({fmt_num(xev)})', '#dc2626'))
        except Exception:  # noqa
            pass
    else:
        r.valor = polys
        r.resultado_latex = r'S(x) = \begin{cases} ' + r'\\ '.join(tex(q) + r', & ' + fmt_num(xs[i]) + r'\le x \le ' + fmt_num(xs[i + 1])
                                                                    for i, q in enumerate(polys)) + r'\end{cases}'
        r.comparar = ('vector', polys, {})
    return r


def hermite(arg):
    conds = []
    for p in dividir(arg.replace(';', ',')):
        m = re.match(r"^\s*[A-Za-z]\s*('*)\s*\(\s*(.+?)\s*\)\s*=\s*(.+)$", normalizar(p))
        if m:
            conds.append((_val(m.group(2)), len(m.group(1)), _val(m.group(3))))
            continue
        e = leer(p)
        if isinstance(e, (sp.Tuple, tuple, list)) and len(e) >= 2:
            for orden, vv in enumerate(e[1:]):
                conds.append((sp.nsimplify(e[0]), orden, sp.nsimplify(vv)))
        else:
            raise ErrorEntrada("Uso: hermite (x0, y0, y'0), (x1, y1, y'1), ...   o   hermite p(0)=1, p'(0)=0, p(1)=2, p'(1)=1")
    if len(conds) < 2:
        raise ErrorEntrada('Se necesitan al menos 2 condiciones.')
    claves = [(c[0], c[1]) for c in conds]
    if len(set(claves)) != len(claves):
        raise ErrorEntrada('Hay una condición repetida.')
    m_ = len(conds)
    cs = sp.symbols(f'c0:{m_}')
    P = sum(cs[k_] * x ** k_ for k_ in range(m_))
    ecs = [sp.Eq(sp.diff(P, x, o).subs(x, xi), vv) for xi, o, vv in conds]
    sol = sp.solve(ecs, cs, dict=True)
    if not sol or len(sol[0]) < m_:
        raise ErrorEntrada('Las condiciones no determinan un único polinomio.')
    Pf = sp.expand(P.subs(sol[0]))
    r = _rn('Interpolación de Hermite')
    nom = lambda o: "p" + "'" * o  # noqa
    r.entrada_latex = r',\ '.join(f"{nom(o)}({tex(xi)}) = {tex(vv)}" for xi, o, vv in conds)
    r.paso('Grado', f'{m_} condiciones determinan un polinomio de grado ≤ {m_ - 1}.', '', 0)
    # Diferencias divididas con nodos repetidos (si para cada nodo se dan las derivadas 0..k)
    nodos = {}
    for xi, o, vv in conds:
        nodos.setdefault(xi, {})[o] = vv
    confluente = all(sorted(d_) == list(range(len(d_))) for d_ in nodos.values())
    if confluente:
        z = []
        for xi in sorted(nodos, key=float):
            z += [xi] * len(nodos[xi])
        Q = [[None] * m_ for _ in range(m_)]
        for i in range(m_):
            Q[i][0] = nodos[z[i]][0]
        for j in range(1, m_):
            for i in range(j, m_):
                if z[i] == z[i - j]:
                    Q[i][j] = nodos[z[i]][j] / sp.factorial(j)
                else:
                    Q[i][j] = (Q[i][j - 1] - Q[i - 1][j - 1]) / (z[i] - z[i - j])
        r.tabla = {'columnas': ['zᵢ', 'f[zᵢ]'] + [f'orden {j}' for j in range(1, m_)],
                   'filas': [[tex(z[i])] + [tex(Q[i][j]) if Q[i][j] is not None else '' for j in range(m_)] for i in range(m_)]}
        r.paso('Diferencias divididas con nodos repetidos', 'Cada nodo se repite tantas veces como condiciones tiene; cuando los nodos '
               r'coinciden se usa la derivada: \(f[z_i, \dots, z_{i+j}] = f^{(j)}(z_i)/j!\).', '', 0)
        newton = sp.Integer(0)
        prod = sp.Integer(1)
        terminos = []
        for j in range(m_):
            newton += Q[j][j] * prod
            if Q[j][j] != 0:
                terminos.append(tex(Q[j][j]) + ''.join(f'(x - {tex(z[i])})' if z[i] != 0 else 'x' for i in range(j)))
            prod *= (x - z[j])
        r.paso('Forma de Newton', '', 'H(x) = ' + ' + '.join(terminos).replace('+ -', '- '), 0)
    else:
        r.paso('Sistema de ecuaciones', r'Se plantea \(p(x) = \sum c_k x^k\) y se imponen las condiciones.',
               r'\\ '.join(tex(e.lhs) + ' = ' + tex(e.rhs) for e in ecs), 0)
    r.paso('Polinomio', '', 'H(x) = ' + tex(Pf), 0)
    r.valor = Pf
    r.resultado_latex = 'H(x) = ' + tex(Pf)
    r.comparar = ('expr', Pf, {})
    r.grafico = G.grafico_datos([(xi, vv) for xi, o, vv in conds if o == 0], Pf, 'H(x)')
    return r


def minimos_cuadrados(arg):
    m = re.match(r'^\s*(\d+)\s*[,;]?\s*(.*)$', arg.strip())
    if not m:
        raise ErrorEntrada('Uso: minimos cuadrados grado m (x0,y0), (x1,y1), ...   ej.: minimos cuadrados grado 2 (0,1),(1,2),(2,5),(3,10)')
    g = int(m.group(1))
    pts = _puntos(m.group(2))
    n_ = len(pts)
    if g < 1:
        raise ErrorEntrada('El grado debe ser al menos 1.')
    if n_ <= g:
        raise ErrorEntrada(f'Para grado {g} se necesitan al menos {g + 1} puntos.')
    xs = [q[0] for q in pts]
    ys = [q[1] for q in pts]
    S = [sum(a ** k_ for a in xs) for k_ in range(2 * g + 1)]
    T = [sum(a ** k_ * b for a, b in pts) for k_ in range(g + 1)]
    N = sp.Matrix(g + 1, g + 1, lambda i, j: S[i + j])
    Tv = sp.Matrix(T)
    if N.det() == 0:
        raise ErrorEntrada('El sistema normal es singular (hay pocas x distintas para ese grado).')
    coef = N.LUsolve(Tv)
    P = sp.expand(sum(coef[k_] * x ** k_ for k_ in range(g + 1)))
    r = _rn(f'Mínimos cuadrados (polinomio de grado {g})')
    r.entrada_latex = ', '.join(f'({tex(a)}, {tex(b)})' for a, b in pts)
    r.paso('Modelo', 'Se minimiza la suma de los cuadrados de los residuos.', r'p(x) = ' + ' + '.join(f'a_{{{k_}}}x^{{{k_}}}' for k_ in range(g + 1))
           + r',\quad \min \sum (y_i - p(x_i))^2', 0)
    r.paso('Sumas', '', r',\ '.join(f'\\sum x^{{{k_}}} = {tex(S[k_])}' for k_ in range(1, 2 * g + 1)) + r',\ '
           + r',\ '.join(f'\\sum x^{{{k_}}}y = {tex(T[k_])}' for k_ in range(g + 1)), 1)
    a_sym = sp.Matrix([sp.Symbol(f'a_{k_}') for k_ in range(g + 1)])
    r.paso('Ecuaciones normales', '', tex(N) + tex(a_sym) + ' = ' + tex(Tv), 0)
    r.paso('Solución', '', r',\ '.join(f'a_{{{k_}}} = {tex(coef[k_])}' for k_ in range(g + 1)), 0)
    pred = [P.subs(x, a) for a in xs]
    ybar = sum(ys) / sp.Integer(n_)
    sse = sum((b - pq) ** 2 for b, pq in zip(ys, pred))
    sst = sum((b - ybar) ** 2 for b in ys)
    r2 = 1 - sse / sst if sst != 0 else sp.Integer(1)
    r.tabla = {'columnas': ['xᵢ', 'yᵢ', 'p(xᵢ)', 'residuo'], 'filas': [[fmt_num(a), fmt_num(b), _n(pq, 8), _n(b - pq, 6)]
                                                                     for a, b, pq in zip(xs, ys, pred)]}
    r.paso('Bondad del ajuste', '', r'R^2 = 1 - \frac{SSE}{SST} = ' + _n(r2, 8) + r',\quad SSE = ' + _n(sse, 8), 0)
    r.valor = P
    r.valores['r2'] = _n(r2, 12)
    r.resultado_latex = 'p(x) = ' + tex(P) + r',\quad R^2 \approx ' + _n(r2, 6)
    r.comparar = ('expr_num', P, {'tol': 1e-6})
    r.grafico = G.grafico_datos(pts, P, 'p(x)')
    return r


def ajuste_no_lineal(arg, tipo):
    pts = _puntos(arg)
    n_ = len(pts)
    if n_ < 2:
        raise ErrorEntrada('Se necesitan al menos 2 puntos.')
    if any(b <= 0 for _, b in pts):
        raise ErrorEntrada('Para linealizar con logaritmos todas las y deben ser positivas.')
    if tipo == 'potencial' and any(a <= 0 for a, _ in pts):
        raise ErrorEntrada('Para el ajuste potencial todas las x deben ser positivas.')
    X = [float(a) if tipo == 'exponencial' else math.log(float(a)) for a, _ in pts]
    Y = [math.log(float(b)) for _, b in pts]
    sx, sy = sum(X), sum(Y)
    sxx = sum(q * q for q in X)
    sxy = sum(p_ * q for p_, q in zip(X, Y))
    den = n_ * sxx - sx ** 2
    if den == 0:
        raise ErrorEntrada('Todas las x son iguales.')
    bb = (n_ * sxy - sx * sy) / den
    AA = (sy - bb * sx) / n_
    aa = math.exp(AA)
    if tipo == 'exponencial':
        modelo = _F(aa, 10) * sp.exp(_F(bb, 10) * x)
        r = _rn('Ajuste exponencial')
        r.paso('Linealización', r'\(y = a e^{bx} \Rightarrow \ln y = \ln a + b x\): regresión lineal de ln y contra x.',
               r'Y = \ln y,\quad Y = A + b x,\quad a = e^{A}', 0)
        cols = ['xᵢ', 'yᵢ', 'Yᵢ = ln yᵢ', 'xᵢYᵢ', 'xᵢ²']
    else:
        modelo = _F(aa, 10) * x ** _F(bb, 10)
        r = _rn('Ajuste potencial')
        r.paso('Linealización', r'\(y = a x^{b} \Rightarrow \ln y = \ln a + b \ln x\): regresión lineal de ln y contra ln x.',
               r'X = \ln x,\ Y = \ln y,\quad Y = A + b X,\quad a = e^{A}', 0)
        cols = ['xᵢ', 'yᵢ', 'Xᵢ = ln xᵢ', 'Yᵢ = ln yᵢ', 'XᵢYᵢ', 'Xᵢ²']
    r.entrada_latex = ', '.join(f'({tex(a)}, {tex(b)})' for a, b in pts)
    filas = []
    for (a, b), Xi, Yi in zip(pts, X, Y):
        fila = [fmt_num(a), fmt_num(b)] + ([] if tipo == 'exponencial' else [fmt_num(Xi, 8)]) + [fmt_num(Yi, 8), fmt_num(Xi * Yi, 8), fmt_num(Xi * Xi, 8)]
        filas.append(fila)
    filas.append(['Σ ' + fmt_num(sum(float(a) for a, _ in pts), 8), ''] + ([] if tipo == 'exponencial' else [fmt_num(sx, 8)])
                 + [fmt_num(sy, 8), fmt_num(sxy, 8), fmt_num(sxx, 8)])
    r.tabla = {'columnas': cols, 'filas': filas}
    r.paso('Ecuaciones normales', '', r'b = \frac{n\sum XY - \sum X\sum Y}{n\sum X^2 - (\sum X)^2} = ' + fmt_num(bb, 10)
           + r',\quad A = \frac{\sum Y - b\sum X}{n} = ' + fmt_num(AA, 10), 0)
    r.paso('Volver al modelo', '', r'a = e^{A} = ' + fmt_num(aa, 10) + r'\Rightarrow y = ' + tex(modelo), 0)
    ry = sum((q - sy / n_) ** 2 for q in Y)
    r2 = 1 - sum((q - (AA + bb * p_)) ** 2 for p_, q in zip(X, Y)) / ry if ry else 1.0
    _extra(r, 'R² (de la recta linealizada)', '', r'R^2 \approx ' + fmt_num(r2, 8))
    r.valor = modelo
    r.valores.update({'a': fmt_num(aa, 12), 'b': fmt_num(bb, 12), 'r2': fmt_num(r2, 12)})
    r.resultado_latex = 'y = ' + tex(modelo)
    r.comparar = ('expr_num', modelo, {'tol': 1e-4})
    r.grafico = G.grafico_datos(pts, modelo, 'y = ' + tex(modelo))
    return r


def simpson38(arg):
    partes = dividir(arg)
    if len(partes) < 3:
        raise ErrorEntrada('Uso: simpson 3/8 f, a, b[, n]   (n múltiplo de 3)')
    f = leer(partes[0])
    v = variable_principal(f)
    a, b = _fl(partes[1]), _fl(partes[2])
    nn = _entero(partes[3], 'n', 3) if len(partes) > 3 else 3
    if nn % 3:
        raise ErrorEntrada('Simpson 3/8 necesita una cantidad de subintervalos múltiplo de 3.')
    F0 = _fn(f, v)
    h = (b - a) / nn
    r = _rn('Regla de Simpson 3/8 compuesta')
    r.entrada_latex = r'\int_{' + fmt_num(a) + '}^{' + fmt_num(b) + '} ' + tex(f) + r'\,d' + tex(v) + r',\quad n = ' + str(nn)
    r.paso('Fórmula', 'Se aproxima f por cúbicas cada 3 subintervalos.',
           r'S = \frac{3h}{8}\left[f_0 + 3f_1 + 3f_2 + 2f_3 + 3f_4 + \dots + 3f_{n-1} + f_n\right],\quad h = \frac{b-a}{n} = ' + fmt_num(h, 10), 0)
    filas, total = [], 0.0
    for i in range(nn + 1):
        xi = a + i * h
        fi = _evaluar_f(F0, xi)
        c = 1 if i in (0, nn) else (2 if i % 3 == 0 else 3)
        total += c * fi
        if i <= 60:
            filas.append([str(i), fmt_num(xi, 10), fmt_num(fi, 10), str(c), fmt_num(c * fi, 10)])
    res = 3 * h / 8 * total
    r.tabla = {'columnas': ['i', 'xᵢ', 'f(xᵢ)', 'coef.', 'coef·f'], 'filas': filas}
    r.paso('Suma', '', r'S = \frac{3\cdot ' + fmt_num(h, 10) + r'}{8}\cdot ' + fmt_num(total, 12) + ' = ' + fmt_num(res, 12), 0)
    r.paso('Error', 'Es exacta para polinomios de grado ≤ 3.', r'E = -\frac{(b-a)h^4}{80}f^{(4)}(\xi)', 1)
    _cierre_integral(r, f, v, a, b, res)
    return r


def romberg(arg):
    partes = dividir(arg)
    if len(partes) < 3:
        raise ErrorEntrada('Uso: romberg f, a, b[, n]   (devuelve R(n,n); por defecto n = 4)')
    f = leer(partes[0])
    v = variable_principal(f)
    a, b = _fl(partes[1]), _fl(partes[2])
    nn = _entero(partes[3], 'n', 1) if len(partes) > 3 else 4
    if nn > 14:
        raise ErrorEntrada('n es demasiado grande (máximo 14).')
    F0 = _fn(f, v)
    F = lambda q: _evaluar_f(F0, q)  # noqa
    R = [[0.0] * (nn + 1) for _ in range(nn + 1)]
    R[0][0] = (b - a) / 2 * (F(a) + F(b))
    for i in range(1, nn + 1):
        h = (b - a) / 2 ** i
        R[i][0] = R[i - 1][0] / 2 + h * sum(F(a + (2 * k_ - 1) * h) for k_ in range(1, 2 ** (i - 1) + 1))
        for j in range(1, i + 1):
            R[i][j] = R[i][j - 1] + (R[i][j - 1] - R[i - 1][j - 1]) / (4 ** j - 1)
    r = _rn('Integración de Romberg')
    r.entrada_latex = r'\int_{' + fmt_num(a) + '}^{' + fmt_num(b) + '} ' + tex(f) + r'\,d' + tex(v) + r',\quad R(' + str(nn) + ',' + str(nn) + ')'
    r.paso('Primera columna: trapecios', r'R(i,0) es la regla del trapecio con \(2^i\) subintervalos (se reutilizan los puntos anteriores).',
           r'R(0,0) = \frac{b-a}{2}[f(a)+f(b)],\quad R(i,0) = \frac{R(i-1,0)}{2} + h_i\sum_{k=1}^{2^{i-1}} f(a + (2k-1)h_i)', 0)
    r.paso('Extrapolación de Richardson', 'Cada columna elimina un término del error.',
           r'R(i,j) = R(i,j-1) + \frac{R(i,j-1) - R(i-1,j-1)}{4^j - 1}', 0)
    r.tabla = {'columnas': ['i', 'h'] + [f'R(i,{j})' for j in range(nn + 1)],
               'filas': [[str(i), fmt_num((b - a) / 2 ** i, 8)] + [fmt_num(R[i][j], 12) if j <= i else '' for j in range(nn + 1)] for i in range(nn + 1)]}
    r.paso('Resultado', '', f'R({nn},{nn}) = ' + fmt_num(R[nn][nn], 12), 0)
    _cierre_integral(r, f, v, a, b, R[nn][nn])
    r.resultado_latex = f'R({nn},{nn}) \\approx ' + fmt_num(R[nn][nn], 12)
    return r


def _nodos_legendre(n_):
    """Nodos y pesos de Gauss-Legendre en [-1, 1] (Newton sobre P_n con la recurrencia de Bonnet)."""
    with mp.workdps(30):
        nodos, pesos = [], []
        for i in range(1, n_ + 1):
            z = mp.cos(mp.pi * (i - mp.mpf(1) / 4) / (n_ + mp.mpf(1) / 2))
            for _ in range(100):
                p0, p1 = mp.mpf(1), z
                for k_ in range(2, n_ + 1):
                    p0, p1 = p1, ((2 * k_ - 1) * z * p1 - (k_ - 1) * p0) / k_
                if n_ == 1:
                    p1, p0 = z, mp.mpf(1)
                dp = n_ * (z * p1 - p0) / (z * z - 1)
                dz = p1 / dp
                z -= dz
                if abs(dz) < mp.mpf(10) ** -28:
                    break
            p0, p1 = mp.mpf(1), z
            for k_ in range(2, n_ + 1):
                p0, p1 = p1, ((2 * k_ - 1) * z * p1 - (k_ - 1) * p0) / k_
            if n_ == 1:
                p1, p0 = z, mp.mpf(1)
            dp = n_ * (z * p1 - p0) / (z * z - 1)
            nodos.append(z)
            pesos.append(2 / ((1 - z * z) * dp * dp))
        orden = sorted(range(n_), key=lambda i: nodos[i])
        return [float(nodos[i]) for i in orden], [float(pesos[i]) for i in orden]


def gauss_legendre(arg):
    partes = dividir(arg)
    if len(partes) < 3:
        raise ErrorEntrada('Uso: gauss legendre f, a, b[, n puntos]   (n entre 1 y 20; por defecto 3)')
    f = leer(partes[0])
    v = variable_principal(f)
    a, b = _fl(partes[1]), _fl(partes[2])
    nn = _entero(partes[3], 'n', 1) if len(partes) > 3 else 3
    if nn > 20:
        raise ErrorEntrada('Usá como máximo 20 puntos.')
    F0 = _fn(f, v)
    ts, ws = _nodos_legendre(nn)
    c1, c0 = (b - a) / 2, (a + b) / 2
    r = _rn(f'Cuadratura de Gauss-Legendre ({nn} puntos)')
    r.entrada_latex = r'\int_{' + fmt_num(a) + '}^{' + fmt_num(b) + '} ' + tex(f) + r'\,d' + tex(v) + r',\quad n = ' + str(nn)
    r.paso('Cambio de variable a [−1, 1]', '', tex(v) + r' = \frac{b-a}{2}t + \frac{a+b}{2} = ' + fmt_num(c1, 10) + 't + ' + fmt_num(c0, 10)
           + r',\quad \int_a^b f\,d' + tex(v) + r' = \frac{b-a}{2}\int_{-1}^{1} f(' + tex(v) + r'(t))\,dt', 0)
    exactos = {1: r't = 0,\ w = 2', 2: r't = \pm\frac{1}{\sqrt3},\ w = 1',
               3: r't = 0,\ \pm\sqrt{\tfrac35};\quad w = \tfrac89,\ \tfrac59'}
    r.paso('Nodos y pesos', 'Los nodos son las raíces del polinomio de Legendre Pₙ; la fórmula es exacta hasta grado 2n − 1.',
           exactos.get(nn, r'w_i = \frac{2}{(1-t_i^2)[P_n\'(t_i)]^2}'), 0)
    filas, suma = [], 0.0
    for ti, wi in zip(ts, ws):
        xi = c1 * ti + c0
        fi = _evaluar_f(F0, xi)
        suma += wi * fi
        filas.append([fmt_num(ti, 12), fmt_num(wi, 12), fmt_num(xi, 10), fmt_num(fi, 10), fmt_num(wi * fi, 10)])
    res = c1 * suma
    r.tabla = {'columnas': ['tᵢ', 'wᵢ', 'xᵢ', 'f(xᵢ)', 'wᵢ·f(xᵢ)'], 'filas': filas}
    r.paso('Suma', '', r'I \approx \frac{b-a}{2}\sum w_i f(x_i) = ' + fmt_num(c1, 10) + r'\cdot ' + fmt_num(suma, 12) + ' = ' + fmt_num(res, 12), 0)
    _cierre_integral(r, f, v, a, b, res)
    return r


def derivada_numerica(arg):
    partes = dividir(arg)
    if len(partes) < 2:
        raise ErrorEntrada('Uso: derivada numerica f, x0[, h]   (por defecto h = 0.1)')
    f = leer(partes[0])
    v = variable_principal(f)
    x0 = _fl(partes[1])
    h = _fl(partes[2]) if len(partes) > 2 else 0.1
    if h <= 0:
        raise ErrorEntrada('h debe ser positivo.')
    F0 = _fn(f, v)
    F = lambda q: _evaluar_f(F0, q)  # noqa
    fm2, fm, f0, fp, fp2 = F(x0 - 2 * h), F(x0 - h), F(x0), F(x0 + h), F(x0 + 2 * h)
    prog = (fp - f0) / h
    reg = (f0 - fm) / h
    cen = (fp - fm) / (2 * h)
    cinco = (-fp2 + 8 * fp - 8 * fm + fm2) / (12 * h)
    seg = (fp - 2 * f0 + fm) / h ** 2
    r = _rn('Derivación numérica')
    r.entrada_latex = 'f(' + tex(v) + ') = ' + tex(f) + r',\quad x_0 = ' + fmt_num(x0) + r',\ h = ' + fmt_num(h)
    r.paso('Valores de f', '', r',\ '.join(f'f({fmt_num(q, 8)}) = {fmt_num(w, 10)}' for q, w in
                                            ((x0 - h, fm), (x0, f0), (x0 + h, fp))), 0)
    r.paso('Diferencia progresiva', 'Error O(h).', r"f'(x_0)\approx\frac{f(x_0+h) - f(x_0)}{h} = " + fmt_num(prog, 10), 0)
    r.paso('Diferencia regresiva', 'Error O(h).', r"f'(x_0)\approx\frac{f(x_0) - f(x_0-h)}{h} = " + fmt_num(reg, 10), 0)
    r.paso('Diferencia centrada', 'Error O(h²): la más precisa de las tres.', r"f'(x_0)\approx\frac{f(x_0+h) - f(x_0-h)}{2h} = " + fmt_num(cen, 10), 0)
    r.paso('Segunda derivada (centrada)', '', r"f''(x_0)\approx\frac{f(x_0+h) - 2f(x_0) + f(x_0-h)}{h^2} = " + fmt_num(seg, 10), 1)
    filas = [['Progresiva', fmt_num(prog, 10)], ['Regresiva', fmt_num(reg, 10)], ['Centrada', fmt_num(cen, 10)], ['5 puntos', fmt_num(cinco, 10)],
             ["f'' centrada", fmt_num(seg, 10)]]
    cols = ['Fórmula', 'Aproximación']
    exacta = None
    try:
        d1 = sp.diff(f, v)
        exacta = float(d1.subs(v, x0))
        ex2 = float(sp.diff(d1, v).subs(v, x0))
        for fila, vv in zip(filas, (exacta, exacta, exacta, exacta, ex2)):
            fila.append(fmt_num(abs(float(fila[1]) - vv), 4))
        cols.append('|error|')
        r.extra('Valor exacto', r"f'(" + tex(v) + ') = ' + tex(d1), f"f'({fmt_num(x0)}) = {fmt_num(exacta, 12)}")
    except Exception:  # noqa
        pass
    r.tabla = {'columnas': cols, 'filas': filas}
    r.valor = _F(cen, 12)
    r.valores.update({'progresiva': fmt_num(prog, 12), 'regresiva': fmt_num(reg, 12), 'centrada': fmt_num(cen, 12),
                      'cinco_puntos': fmt_num(cinco, 12), 'segunda': fmt_num(seg, 12)})
    r.resultado_latex = r"f'(" + fmt_num(x0) + r')\approx ' + fmt_num(cen, 10) + r'\ (\text{centrada})'
    r.comparar = ('numero', cen, {'tol': 1e-6})
    try:
        recta = f0 + cen * (v - x0)
        r.grafico = G.grafico_2d([f, recta], v, [G.punto(x0, f0, f'x₀ = {fmt_num(x0)}')], etiquetas=[tex(f), 'pendiente centrada'],
                                 xmin=x0 - max(1.0, 6 * h), xmax=x0 + max(1.0, 6 * h))
    except Exception:  # noqa
        pass
    return r


_AB = {1: [1], 2: [sp.Rational(3, 2), -sp.Rational(1, 2)], 3: [sp.Rational(23, 12), -sp.Rational(16, 12), sp.Rational(5, 12)],
       4: [sp.Rational(55, 24), -sp.Rational(59, 24), sp.Rational(37, 24), -sp.Rational(9, 24)]}


def adams(arg, corrector=False):
    nombre = 'adams moulton' if corrector else 'adams bashforth'
    partes = dividir(arg)
    if len(partes) < 5:
        raise ErrorEntrada(f"Uso: {nombre} y'=f(x,y), x0, y0, h, xf[, orden 2, 3 o 4]   (arranca con RK4)")
    f = _leer_edo1(partes[0])
    tv = sp.Symbol('t')
    v = tv if f.has(tv) and not f.has(x) else x
    x0, y0, h, xf = (_fl(p) for p in partes[1:5])
    orden = _entero(partes[5], 'el orden', 2) if len(partes) > 5 else 4
    if orden > 4:
        raise ErrorEntrada('El orden debe ser 2, 3 o 4.')
    if corrector:
        orden = 4
    if h == 0:
        raise ErrorEntrada('h no puede ser 0.')
    F = sp.lambdify((v, y), f, modules=['math'])
    pasos_n = int(round((xf - x0) / h))
    if pasos_n < 1:
        raise ErrorEntrada('xf debe estar al menos un paso h más allá de x0.')
    if pasos_n > 100000:
        raise ErrorEntrada('Demasiados pasos: aumentá h.')
    coefs = _AB[orden]
    r = _rn('Adams-Bashforth-Moulton (predictor-corrector de orden 4)' if corrector else f'Método de Adams-Bashforth de orden {orden}')
    r.entrada_latex = r"y' = " + tex(f) + r',\quad y(' + fmt_num(x0) + ') = ' + fmt_num(y0) + r',\quad h = ' + fmt_num(h) + r',\ ' + tex(v) + r'_f = ' + fmt_num(xf)
    formulas = {2: r'y_{n+1} = y_n + \frac h2\left(3f_n - f_{n-1}\right)', 3: r'y_{n+1} = y_n + \frac h{12}\left(23f_n - 16f_{n-1} + 5f_{n-2}\right)',
                4: r'y_{n+1} = y_n + \frac h{24}\left(55f_n - 59f_{n-1} + 37f_{n-2} - 9f_{n-3}\right)'}
    r.paso('Arranque', f'Es un método multipaso: los primeros {orden - 1} valores se calculan con Runge-Kutta 4.', '', 0)
    r.paso('Predictor (Adams-Bashforth)' if corrector else 'Fórmula', '', formulas[orden], 0)
    if corrector:
        r.paso('Corrector (Adams-Moulton)', 'Se corrige una vez usando el valor predicho.',
               r'y_{n+1} = y_n + \frac h{24}\left(9f^{*}_{n+1} + 19f_n - 5f_{n-1} + f_{n-2}\right)', 0)
    xs_, ys_, fs_ = [x0], [y0], [F(x0, y0)]
    filas = [['0', fmt_num(x0, 8), fmt_num(y0, 10), fmt_num(fs_[0], 10), 'condición inicial']]
    for i in range(pasos_n):
        xn, yn = xs_[-1], ys_[-1]
        if i < orden - 1:
            k1 = F(xn, yn)
            k2 = F(xn + h / 2, yn + h / 2 * k1)
            k3 = F(xn + h / 2, yn + h / 2 * k2)
            k4 = F(xn + h, yn + h * k3)
            y1 = yn + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            met = 'RK4'
        else:
            y1 = yn + h * sum(float(c) * fs_[-1 - j] for j, c in enumerate(coefs))
            met = f'AB{orden}'
            if corrector:
                fp = F(xn + h, y1)
                y1 = yn + h / 24 * (9 * fp + 19 * fs_[-1] - 5 * fs_[-2] + fs_[-3])
                met = 'AB4 + AM4'
        x1 = x0 + (i + 1) * h
        xs_.append(x1)
        ys_.append(y1)
        fs_.append(F(x1, y1))
        if not math.isfinite(y1):
            raise ErrorEntrada('La solución numérica diverge: probá con un h más chico.')
        if i < 100:
            filas.append([str(i + 1), fmt_num(x1, 8), fmt_num(y1, 10), fmt_num(fs_[-1], 10), met])
    r.tabla = {'columnas': ['n', 'xₙ', 'yₙ', 'fₙ = f(xₙ, yₙ)', 'cálculo'], 'filas': filas}
    yn = ys_[-1]
    r.valor = sp.Float(yn, 15)
    r.resultado_latex = 'y(' + fmt_num(xf) + r') \approx ' + fmt_num(yn, 12)
    r.paso('Resultado', f'{pasos_n} pasos de tamaño h = {fmt_num(h)}.', r.resultado_latex, 0)
    pts = list(zip(xs_, ys_))
    r.grafico = G.grafico_datos(pts[:400])
    try:
        if f.is_polynomial(y) and sp.degree(f, y) <= 1:
            Y = sp.Function('y')
            sol = sp.dsolve(sp.Eq(Y(v).diff(v), f.subs(y, Y(v))), Y(v), ics={Y(x0): y0})
            ex = float(sol.rhs.subs(v, xf))
            r.extra('Solución exacta', 'y = ' + tex(sol.rhs), f'y({fmt_num(xf)}) = {fmt_num(ex, 12)}; error = {fmt_num(abs(ex - yn), 4)}')
            r.grafico = G.grafico_datos(pts[:400], sol.rhs, 'exacta')
    except Exception:  # noqa
        pass
    r.comparar = ('numero', yn, {'tol': 1e-8})
    return r


def rk4_sistema(arg):
    partes = dividir(arg)
    ecs = []
    for p in partes:
        m = re.match(r"^\s*([A-Za-z]\w*)\s*'\s*=\s*(.+)$", normalizar(p))
        if not m:
            break
        ecs.append((m.group(1), m.group(2)))
    resto = partes[len(ecs):]
    uso = "Uso: rk4 sistema x'=f(t,x,y), y'=g(t,x,y), t0, (x0, y0), h, tf   ej.: rk4 sistema x'=y, y'=-x, 0, (1,0), 0.1, 1"
    if len(ecs) < 1 or len(resto) < 4:
        raise ErrorEntrada(uso)
    nombres = [nm for nm, _ in ecs]
    if len(set(nombres)) != len(nombres):
        raise ErrorEntrada('Hay dos ecuaciones para la misma variable.')
    syms = [sp.Symbol(nm) for nm in nombres]
    extra = {nm: s_ for nm, s_ in zip(nombres, syms)}
    indep = sp.Symbol('t')
    if 't' in nombres:
        indep = sp.Symbol('x') if 'x' not in nombres else sp.Symbol('s')
    extra[str(indep)] = indep
    rhs = [leer(txt, extra=extra) for _, txt in ecs]
    libres = set().union(*[e.free_symbols for e in rhs]) - set(syms) - {indep}
    if libres:
        raise ErrorEntrada('No sé qué es ' + ', '.join(str(q) for q in libres) + ': usá solo la variable independiente '
                           + str(indep) + ' y las incógnitas ' + ', '.join(nombres) + '.')
    t0 = _fl(resto[0])
    ci = leer(resto[1])
    if isinstance(ci, (sp.Tuple, tuple, list)):
        y0 = [float(q) for q in ci]
        cola = resto[2:]
    else:
        y0 = [_fl(q) for q in resto[1:1 + len(syms)]]
        cola = resto[1 + len(syms):]
    if len(y0) != len(syms):
        raise ErrorEntrada(f'Hacen falta {len(syms)} condiciones iniciales, una por incógnita.')
    if len(cola) < 2:
        raise ErrorEntrada(uso)
    h, tf = _fl(cola[0]), _fl(cola[1])
    if h == 0:
        raise ErrorEntrada('h no puede ser 0.')
    pasos_n = int(round((tf - t0) / h))
    if pasos_n < 1 or pasos_n > 100000:
        raise ErrorEntrada('Revisá h y tf: la cantidad de pasos debe estar entre 1 y 100000.')
    Fs = sp.lambdify([indep] + syms, rhs, modules=['math'])
    F = lambda tt, Y: [float(q) for q in Fs(tt, *Y)]  # noqa
    r = _rn('Runge-Kutta 4 para sistemas')
    r.entrada_latex = r'\begin{cases}' + r'\\ '.join(f"{nm}' = {tex(e)}" for nm, e in zip(nombres, rhs)) + r'\end{cases}\quad ' \
        + ', '.join(f'{nm}({fmt_num(t0)}) = {fmt_num(q)}' for nm, q in zip(nombres, y0)) + r',\ h = ' + fmt_num(h)
    r.paso('Forma vectorial', r'Se aplica RK4 al vector \(\mathbf{Y} = (' + ', '.join(nombres) + r')\) con \(\mathbf{Y}\' = \mathbf{F}(t, \mathbf{Y})\).',
           r'\mathbf{K}_1 = \mathbf{F}(t_n, \mathbf{Y}_n),\ \mathbf{K}_2 = \mathbf{F}(t_n + \tfrac h2, \mathbf{Y}_n + \tfrac h2\mathbf{K}_1),\ '
           r'\mathbf{K}_3 = \mathbf{F}(t_n + \tfrac h2, \mathbf{Y}_n + \tfrac h2\mathbf{K}_2),\ \mathbf{K}_4 = \mathbf{F}(t_n + h, \mathbf{Y}_n + h\mathbf{K}_3)', 0)
    r.paso('Actualización', '', r'\mathbf{Y}_{n+1} = \mathbf{Y}_n + \frac h6(\mathbf{K}_1 + 2\mathbf{K}_2 + 2\mathbf{K}_3 + \mathbf{K}_4)', 0)
    Y = list(y0)
    tt = t0
    filas = [['0', fmt_num(tt, 8)] + [fmt_num(q, 10) for q in Y]]
    tray = [[tt] + list(Y)]
    for i in range(pasos_n):
        K1 = F(tt, Y)
        K2 = F(tt + h / 2, [a + h / 2 * k_ for a, k_ in zip(Y, K1)])
        K3 = F(tt + h / 2, [a + h / 2 * k_ for a, k_ in zip(Y, K2)])
        K4 = F(tt + h, [a + h * k_ for a, k_ in zip(Y, K3)])
        Y = [a + h / 6 * (p1 + 2 * p2 + 2 * p3 + p4) for a, p1, p2, p3, p4 in zip(Y, K1, K2, K3, K4)]
        tt = t0 + (i + 1) * h
        if not all(math.isfinite(q) for q in Y):
            raise ErrorEntrada('La solución numérica diverge: probá con un h más chico.')
        if i < 100:
            filas.append([str(i + 1), fmt_num(tt, 8)] + [fmt_num(q, 10) for q in Y])
        if i % max(1, pasos_n // 400) == 0 or i == pasos_n - 1:
            tray.append([tt] + list(Y))
    if i == 0:
        r.paso('Primer paso', '', r',\ '.join(f'K_{{{j + 1}}} = ({", ".join(fmt_num(q, 8) for q in K)})' for j, K in enumerate((K1, K2, K3, K4))), 1)
    r.tabla = {'columnas': ['n', str(indep) + 'ₙ'] + nombres, 'filas': filas}
    if pasos_n > 100:
        r.avisos.append(f'Se muestran 100 de {pasos_n} pasos.')
    r.valor = [_F(q, 12) for q in Y]
    r.resultado_latex = ', '.join(f'{nm}({fmt_num(tf)}) \\approx {fmt_num(q, 10)}' for nm, q in zip(nombres, Y))
    r.paso('Resultado', f'{pasos_n} pasos de tamaño h = {fmt_num(h)}.', r.resultado_latex, 0)
    r.comparar = ('vector', [sp.Float(q, 15) for q in Y], {})
    curvas = [_curva_puntos([(p[0], p[j + 1]) for p in tray], nm + '(' + str(indep) + ')', G.COLORES[j % len(G.COLORES)])
              for j, nm in enumerate(nombres)]
    todos = [p[j + 1] for p in tray for j in range(len(nombres))]
    lo, hi = min(todos), max(todos)
    alto = (hi - lo) or 1.0
    r.grafico = {'tipo': '2d', 'curvas': curvas, 'puntos': [],
                 'ventana': {'xmin': min(t0, tf) - 0.05 * abs(tf - t0), 'xmax': max(t0, tf) + 0.05 * abs(tf - t0),
                             'ymin': lo - 0.15 * alto, 'ymax': hi + 0.15 * alto}}
    return r


def error_aprox(arg):
    d, pos = _kv(arg)
    va = _get(d, 'aproximado', 'aprox', 'valoraproximado', 'medido')
    ve = _get(d, 'exacto', 'real', 'valorexacto', 'verdadero')
    if va is None or ve is None:
        if len(pos) < 2:
            raise ErrorEntrada('Uso: error aproximado, exacto   ej.: error 3.14, pi')
        va, ve = pos[0], pos[1]
    a, e = _val(va), _val(ve)
    if e == 0:
        raise ErrorEntrada('El valor exacto es 0: el error relativo no está definido.')
    ea = sp.Abs(e - a)
    eaf = float(sp.N(ea, 20))
    er = eaf / abs(float(sp.N(e, 20)))
    r = _rn('Errores absoluto y relativo')
    r.entrada_latex = r'\tilde x = ' + tex(a) + r',\quad x = ' + tex(e)
    r.paso('Error absoluto', r'\(E_a = |x - \tilde x|\)', 'E_a = |' + tex(e) + ' - ' + tex(a) + '| = ' + (tex(ea) + r'\approx ' if not ea.is_Number else '')
           + fmt_num(eaf, 8), 0)
    r.paso('Error relativo', r'\(E_r = \frac{E_a}{|x|}\)', 'E_r = ' + fmt_num(er, 8), 0)
    r.paso('Error porcentual', '', r'E_\% = 100\,E_r = ' + fmt_num(100 * er, 8) + r'\,\%', 0)
    if er > 0:
        cifras = max(0, int(math.floor(math.log10(5 / er))))
        r.paso('Cifras significativas correctas', r'\(\tilde x\) aproxima a x con n cifras significativas si \(E_r \le 5\times10^{-n}\).',
               f'n = {cifras}', 1)
        r.valores['cifras'] = cifras
    r.valor = ea if ea.is_Rational else _F(eaf, 12)
    r.valores.update({'absoluto': fmt_num(eaf, 12), 'relativo': fmt_num(er, 12), 'porcentual': fmt_num(100 * er, 12)})
    r.resultado_latex = r'E_a \approx ' + fmt_num(eaf, 8) + r',\ E_r \approx ' + fmt_num(er, 6) + r',\ E_\% \approx ' + fmt_num(100 * er, 6) + r'\,\%'
    r.comparar = ('numero', ea, {'tol': 1e-6})
    return r


def propagacion_errores(arg):
    partes = dividir(arg)
    if len(partes) < 2:
        raise ErrorEntrada('Uso: propagacion de errores f, x=2±0.1, y=3±0.2   o   propagacion de errores f, [2, 3], [0.1, 0.2]')
    f = leer(partes[0])
    vs = variables_de(f)
    if not vs:
        raise ErrorEntrada('La función no tiene variables.')
    vals, errs = {}, {}
    resto = partes[1:]
    if resto[0].strip().startswith('['):
        if len(resto) < 2:
            raise ErrorEntrada('Faltan los errores: propagacion de errores f, [valores], [errores]')
        lv, le = _leer_vector(resto[0]), _leer_vector(resto[1])
        if len(lv) != len(vs) or len(le) != len(vs):
            raise ErrorEntrada(f'Se esperaban {len(vs)} valores y {len(vs)} errores (variables: {", ".join(map(str, vs))}).')
        vals, errs = dict(zip(vs, lv)), dict(zip(vs, le))
    else:
        for p in resto:
            if '=' not in p:
                raise ErrorEntrada(f'No entendí «{p}». Escribí cada variable como x=2±0.1')
            nm, txt = p.split('=', 1)
            trozos = re.split(r'±|\+/-|\+-', txt)
            if len(trozos) != 2:
                raise ErrorEntrada(f'Indicá el error de {nm.strip()} con ±, por ejemplo {nm.strip()}=2±0.1')
            s_ = sp.Symbol(nm.strip())
            cand = [q for q in vs if str(q) == nm.strip()]
            s_ = cand[0] if cand else s_
            vals[s_], errs[s_] = _val(trozos[0]), abs(_val(trozos[1]))
    faltan = [q for q in vs if q not in vals]
    if faltan:
        raise ErrorEntrada('Falta el valor de ' + ', '.join(map(str, faltan)) + '.')
    r = _rn('Propagación de errores')
    r.entrada_latex = 'f = ' + tex(f) + r',\quad ' + r',\ '.join(f'{tex(q)} = {tex(vals[q])} \\pm {tex(errs[q])}' for q in vs)
    f0 = f.subs(vals)
    r.paso('Valor de f', '', 'f = ' + tex(sp.nsimplify(f0)) + r'\approx ' + _n(f0, 10), 0)
    r.paso('Fórmula (cota lineal)', 'Se usa el diferencial total:', r'\Delta f \approx \sum_i \left|\frac{\partial f}{\partial x_i}\right|\Delta x_i', 0)
    terminos, filas = [], []
    for q in vs:
        dq = sp.diff(f, q)
        dv = dq.subs(vals)
        term = sp.Abs(dv) * errs[q]
        terminos.append(term)
        filas.append([str(q), _n(vals[q], 10), _n(errs[q], 10), tex(dq), _n(dv, 10), _n(term, 10)])
        r.paso(f'Derivada respecto de {q}', '', r'\frac{\partial f}{\partial ' + tex(q) + '} = ' + tex(dq) + ' = ' + _n(dv, 10)
               + r',\quad |\cdot|\,\Delta ' + tex(q) + ' = ' + _n(term, 10), 1)
    df = sum(terminos)
    dff = float(sp.N(df, 20))
    rel = dff / abs(float(sp.N(f0, 20))) if f0 != 0 else float('inf')
    cuad = math.sqrt(sum(float(sp.N(q_, 20)) ** 2 for q_ in terminos))
    r.tabla = {'columnas': ['variable', 'valor', 'Δ', '∂f/∂xᵢ', 'valor de ∂f/∂xᵢ', '|∂f/∂xᵢ|·Δxᵢ'], 'filas': filas}
    r.paso('Error absoluto de f', '', r'\Delta f \approx ' + _n(dff, 10) + r'\Rightarrow f = ' + _n(f0, 8) + r'\pm ' + _n(dff, 6), 0)
    if math.isfinite(rel):
        r.paso('Error relativo', '', r'\frac{\Delta f}{|f|} \approx ' + _n(rel, 8) + r'\ (' + _n(100 * rel, 6) + r'\,\%)', 0)
    _extra(r, 'Suma cuadrática (errores independientes)', '', r'\Delta f \approx \sqrt{\sum\left(\frac{\partial f}{\partial x_i}\Delta x_i\right)^2} = '
            + _n(cuad, 8))
    r.valor = _F(dff, 12)
    r.valores.update({'f': _n(f0, 12), 'error_absoluto': _n(dff, 12), 'error_relativo': _n(rel, 12) if math.isfinite(rel) else '',
                      'cuadratico': _n(cuad, 12)})
    r.resultado_latex = 'f = ' + _n(f0, 8) + r'\pm ' + _n(dff, 6)
    r.comparar = ('numero', dff, {'tol': 1e-6})
    return r


# ============================================================================
# Electrotecnia
# ============================================================================
def _complejo(texto):
    s_ = normalizar(str(texto)).replace('°', '').strip()
    if '@' in s_:
        mod, ang = s_.split('@', 1)
        return cmath.rect(_fl(mod), math.radians(_fl(ang)))
    v = leer(s_)
    try:
        return complex(sp.N(v))
    except Exception as exc:  # noqa
        raise ErrorEntrada(f'«{texto}» debería ser un fasor (módulo@ángulo en grados) o un complejo a+bj.') from exc


def _ang(z):
    return math.degrees(cmath.phase(z)) if z != 0 else 0.0


def _tex_polar(z, dig=8):
    return _n(abs(z), dig) + r'\angle ' + _n(_ang(z), 6) + '^\\circ'


def _tex_rect(z, dig=8):
    re_, im_ = z.real, z.imag
    if abs(im_) < 1e-12 * max(1.0, abs(re_)):
        return _n(re_, dig)
    return _n(re_, dig) + (' + ' if im_ >= 0 else ' - ') + 'j' + _n(abs(im_), dig)


def _triangulo_potencia(P, Q, etiqueta='S', color=None):
    return _curva_puntos([(0, 0), (P, 0), (P, Q), (0, 0)], etiqueta, color)


def potencia_compleja(arg):
    uso = 'potencia compleja V=220@0, I=10@-30   (valores eficaces; también V y Z, o I y Z)'
    d, pos = _kv(arg)
    V, I, Z = _get(d, 'v', 'u', 'tension'), _get(d, 'i', 'corriente'), _get(d, 'z', 'impedancia')
    if V is None and I is None and len(pos) >= 2:
        V, I = pos[0], pos[1]
    r = Resultado(TEMA_EL, 'Potencia compleja')
    if V is not None and I is not None:
        Vc, Ic = _complejo(V), _complejo(I)
    elif V is not None and Z is not None:
        Vc, Zc = _complejo(V), _complejo(Z)
        if Zc == 0:
            raise ErrorEntrada('Z no puede ser 0.')
        Ic = Vc / Zc
        r.paso('Corriente', r'\(I = V/Z\)', 'I = ' + _tex_polar(Vc) + '/' + _tex_polar(Zc) + ' = ' + _tex_polar(Ic) + r'\ \text{A}', 0)
    elif I is not None and Z is not None:
        Ic, Zc = _complejo(I), _complejo(Z)
        Vc = Ic * Zc
        r.paso('Tensión', r'\(V = I\,Z\)', 'V = ' + _tex_polar(Vc) + r'\ \text{V}', 0)
    else:
        raise ErrorEntrada('Faltan datos. Uso: ' + uso)
    r.entrada_latex = r'V = ' + _tex_polar(Vc) + r'\ \text{V},\quad I = ' + _tex_polar(Ic) + r'\ \text{A}'
    S = Vc * Ic.conjugate()
    P, Q, Sm = S.real, S.imag, abs(S)
    if Sm == 0:
        raise ErrorEntrada('La potencia aparente es 0.')
    fp = P / Sm
    tipo = 'inductivo (la corriente atrasa)' if Q > 1e-9 * Sm else 'capacitivo (la corriente adelanta)' if Q < -1e-9 * Sm else 'resistivo (en fase)'
    r.paso('Definición', 'Con valores eficaces, la potencia compleja usa el conjugado de la corriente.', r'S = V\,I^{*}', 0)
    r.paso('Conjugado de la corriente', '', r'I^{*} = ' + _tex_polar(Ic.conjugate()), 1)
    r.paso('Producto en forma polar', 'Se multiplican los módulos y se suman los ángulos.',
           r'S = ' + _n(abs(Vc), 8) + r'\cdot ' + _n(abs(Ic), 8) + r'\angle(' + _n(_ang(Vc), 6) + '^\\circ ' + ('+' if -_ang(Ic) >= 0 else '-') + ' '
           + _n(abs(_ang(Ic)), 6) + '^\\circ) = ' + _tex_polar(S) + r'\ \text{VA}', 0)
    r.paso('Forma binómica', r'\(S = P + jQ\)', 'S = ' + _tex_rect(S) + r'\ \text{VA}', 0)
    r.paso('Potencias', '', r'P = ' + _n(P, 8) + r'\ \text{W},\quad Q = ' + _n(Q, 8) + r'\ \text{var},\quad |S| = ' + _n(Sm, 8) + r'\ \text{VA}', 0)
    r.paso('Factor de potencia', r'\(fp = \cos\varphi = P/|S|\), con \(\varphi = \theta_V - \theta_I\).',
           r'fp = ' + _n(fp, 8) + r'\ (\varphi = ' + _n(_ang(S), 6) + r'^\circ)\ \text{' + tipo + '}', 0)
    r.valor = _F(P, 12) + sp.I * _F(Q, 12) if abs(Q) > 0 else _F(P, 12)
    r.valores.update({'P': _n(P, 12), 'Q': _n(Q, 12), 'S': _n(Sm, 12), 'fp': _n(fp, 12), 'phi': _n(_ang(S), 12),
                      'tipo': tipo.split(' ')[0], 'I': _n(abs(Ic), 12)})
    r.resultado_latex = 'S = ' + _tex_rect(S) + r'\ \text{VA},\quad |S| = ' + _n(Sm, 8) + r',\ fp = ' + _n(fp, 6) + r'\ \text{' + tipo.split(' ')[0] + '}'
    r.comparar = ('numero', sp.Float(P, 15) + sp.I * sp.Float(Q, 15), {'tol': 1e-6})
    m = max(abs(P), abs(Q), 1e-9)
    r.grafico = {'tipo': '2d', 'curvas': [_triangulo_potencia(P, Q, 'triángulo de potencias')],
                 'puntos': [G.punto(P, 0, f'P = {_n(P, 6)} W'), G.punto(P, Q, f'S = {_n(Sm, 6)} VA', '#dc2626')],
                 'ventana': {'xmin': min(0, P) - 0.2 * m, 'xmax': max(0, P) + 0.4 * m, 'ymin': min(0, Q) - 0.2 * m, 'ymax': max(0, Q) + 0.2 * m}}
    return r


def correccion_fp(arg):
    uso = 'correccion fp P=10000, fp1=0.7, fp2=0.95, V=220, f=50[, trifasico]'
    d, pos = _kv(arg)
    P = _get(d, 'p', 'potencia', 'potenciaactiva')
    fp1 = _get(d, 'fp1', 'fpi', 'fpinicial', 'cosphi1', 'cos1', 'fpactual', 'fp')
    fp2 = _get(d, 'fp2', 'fpf', 'fpfinal', 'fpdeseado', 'cosphi2', 'cos2', 'fpobjetivo')
    if P is None or fp1 is None or fp2 is None:
        if len(pos) >= 3 and not d:
            P, fp1, fp2 = pos[:3]
            if len(pos) > 3:
                d['v'] = pos[3]
            if len(pos) > 4:
                d['f'] = pos[4]
        else:
            raise ErrorEntrada('Faltan datos. Uso: ' + uso)
    Pv, f1, f2 = _fl(P), _fl(fp1), _fl(fp2)
    if not (0 < f1 <= 1 and 0 < f2 <= 1):
        raise ErrorEntrada('Los factores de potencia deben estar entre 0 y 1.')
    if f2 <= f1:
        raise ErrorEntrada('El factor de potencia deseado debe ser mayor que el actual.')
    Vt, ft = _get(d, 'v', 'u', 'tension'), _get(d, 'f', 'frecuencia') or '50'
    texto_pos = sin_tildes(' '.join(pos).lower())
    tri = 'trifas' in texto_pos or any('trifas' in k_ for k_ in d)
    estrella = 'estrella' in texto_pos
    p1, p2 = math.acos(f1), math.acos(f2)
    Q1, Q2 = Pv * math.tan(p1), Pv * math.tan(p2)
    Qc = Q1 - Q2
    r = Resultado(TEMA_EL, 'Corrección del factor de potencia')
    r.entrada_latex = r'P = ' + _n(Pv) + r'\ \text{W},\ fp_1 = ' + _n(f1) + r',\ fp_2 = ' + _n(f2) + \
        (r',\ V = ' + _n(_fl(Vt)) + r'\ \text{V},\ f = ' + _n(_fl(ft)) + r'\ \text{Hz}' if Vt else '') + (r'\ \text{(trifásico)}' if tri else '')
    r.paso('Ángulos', r'\(\varphi = \arccos(fp)\)', r'\varphi_1 = ' + _n(math.degrees(p1), 6) + r'^\circ,\quad \varphi_2 = ' + _n(math.degrees(p2), 6) + r'^\circ', 0)
    r.paso('Potencias reactivas', 'La potencia activa P no cambia al agregar capacitores; solo baja Q.',
           r'Q_1 = P\tan\varphi_1 = ' + _n(Q1, 8) + r'\ \text{var},\quad Q_2 = P\tan\varphi_2 = ' + _n(Q2, 8) + r'\ \text{var}', 0)
    r.paso('Potencia del capacitor', '', r'Q_C = Q_1 - Q_2 = P(\tan\varphi_1 - \tan\varphi_2) = ' + _n(Qc, 8) + r'\ \text{var}', 0)
    r.valores.update({'Q1': _n(Q1, 12), 'Q2': _n(Q2, 12), 'Qc': _n(Qc, 12)})
    r.comparar = ('numero', Qc, {'tol': 1e-4})
    r.valor = _F(Qc, 12)
    r.resultado_latex = r'Q_C \approx ' + _n(Qc, 8) + r'\ \text{var}'
    if Vt:
        V, fr = _fl(Vt), _fl(ft)
        w = 2 * math.pi * fr
        if tri:
            Cd = Qc / (3 * w * V ** 2)
            Cy = Qc / (w * V ** 2)
            r.paso('Capacidad (trifásico, V de línea)', 'Cada capacitor aporta Q_C/3.',
                   r'C_\Delta = \frac{Q_C}{3\,\omega V_L^2} = ' + _n(Cd * 1e6, 8) + r'\ \mu\text{F},\quad C_Y = \frac{Q_C}{\omega V_L^2} = '
                   + _n(Cy * 1e6, 8) + r'\ \mu\text{F}', 0)
            C = Cy if estrella else Cd
            r.paso('Resultado', 'Banco en ' + ('estrella' if estrella else 'triángulo (si es en estrella, agregá «estrella»)') + '; se compara en µF.',
                   r'C \approx ' + _n(C * 1e6, 8) + r'\ \mu\text{F por fase}', 0)
            I1, I2 = Pv / (math.sqrt(3) * V * f1), Pv / (math.sqrt(3) * V * f2)
        else:
            C = Qc / (w * V ** 2)
            r.paso('Capacidad', r'\(Q_C = \omega C V^2\Rightarrow C = \dfrac{Q_C}{2\pi f V^2}\)',
                   r'C = \frac{' + _n(Qc, 8) + r'}{2\pi\cdot ' + _n(fr) + r'\cdot ' + _n(V) + '^2} = ' + _n(C, 6) + r'\ \text{F} = ' + _n(C * 1e6, 8)
                   + r'\ \mu\text{F}', 0)
            I1, I2 = Pv / (V * f1), Pv / (V * f2)
        _extra(r, 'Corriente de línea', 'Baja al mejorar el factor de potencia:', r'I_1 = ' + _n(I1, 6) + r'\ \text{A} \to I_2 = ' + _n(I2, 6) + r'\ \text{A}')
        r.valor = _F(C * 1e6, 12)
        r.valores.update({'C_uF': _n(C * 1e6, 12), 'C_F': _n(C, 12), 'I1': _n(I1, 12), 'I2': _n(I2, 12)})
        r.resultado_latex = r'Q_C \approx ' + _n(Qc, 8) + r'\ \text{var},\quad C \approx ' + _n(C * 1e6, 8) + r'\ \mu\text{F}'
        r.comparar = ('numero', C * 1e6, {'tol': 1e-4})
    m = max(Pv, Q1)
    r.grafico = {'tipo': '2d', 'curvas': [_triangulo_potencia(Pv, Q1, 'antes', G.COLORES[1]), _triangulo_potencia(Pv, Q2, 'después', G.COLORES[2])],
                 'puntos': [G.punto(Pv, Q1, f'Q₁ = {_n(Q1, 5)}'), G.punto(Pv, Q2, f'Q₂ = {_n(Q2, 5)}', '#059669')],
                 'ventana': {'xmin': -0.1 * m, 'xmax': 1.4 * m, 'ymin': -0.1 * m, 'ymax': 1.2 * m}}
    return r


def trifasico(arg):
    uso = 'trifasico V=380, I=10, fp=0.8[, estrella | triangulo][, capacitivo]   (V e I de línea; o P en lugar de I)'
    d, pos = _kv(arg)
    V, I, fp, P = _get(d, 'v', 'vl', 'u', 'tension'), _get(d, 'i', 'il', 'corriente'), _get(d, 'fp', 'cosphi', 'cos'), _get(d, 'p', 'potencia')
    if V is None and len(pos) >= 3:
        V, I, fp = pos[:3]
    if V is None or fp is None or (I is None and P is None):
        raise ErrorEntrada('Faltan datos. Uso: ' + uso)
    Vv, fpv = _fl(V), _fl(fp)
    if not 0 < fpv <= 1:
        raise ErrorEntrada('El factor de potencia debe estar entre 0 y 1.')
    texto_pos = sin_tildes(' '.join(pos).lower())
    cap = 'capacit' in texto_pos
    conexion = 'estrella' if 'estrella' in texto_pos else 'triangulo' if 'triangulo' in texto_pos else None
    r = Resultado(TEMA_EL, 'Potencia trifásica (sistema equilibrado)')
    phi = math.acos(fpv) * (-1 if cap else 1)
    if I is not None:
        Iv = _fl(I)
    else:
        Pv_ = _fl(P)
        Iv = Pv_ / (math.sqrt(3) * Vv * fpv)
        r.paso('Corriente de línea', r'\(I_L = \dfrac{P}{\sqrt3\,V_L\cos\varphi}\)', r'I_L = ' + _n(Iv, 8) + r'\ \text{A}', 0)
    r.entrada_latex = r'V_L = ' + _n(Vv) + r'\ \text{V},\ I_L = ' + _n(Iv, 8) + r'\ \text{A},\ \cos\varphi = ' + _n(fpv)
    S = math.sqrt(3) * Vv * Iv
    Pp = S * fpv
    Q = S * math.sin(phi)
    r.paso('Potencia aparente', r'Con tensión y corriente de LÍNEA (vale para estrella y triángulo):',
           r'S = \sqrt3\,V_L I_L = \sqrt3\cdot ' + _n(Vv) + r'\cdot ' + _n(Iv, 8) + ' = ' + _n(S, 8) + r'\ \text{VA}', 0)
    r.paso('Potencia activa', '', r'P = \sqrt3\,V_L I_L\cos\varphi = ' + _n(Pp, 8) + r'\ \text{W}', 0)
    r.paso('Potencia reactiva', ('Capacitiva: Q < 0.' if cap else 'Inductiva (por defecto): Q > 0.'),
           r'Q = \sqrt3\,V_L I_L\sin\varphi = ' + _n(Q, 8) + r'\ \text{var}\quad (\varphi = ' + _n(math.degrees(phi), 6) + r'^\circ)', 0)
    if conexion == 'estrella':
        r.paso('Valores de fase (estrella)', '', r'V_f = \frac{V_L}{\sqrt3} = ' + _n(Vv / math.sqrt(3), 8) + r'\ \text{V},\quad I_f = I_L = ' + _n(Iv, 8) + r'\ \text{A}', 1)
        Zf = (Vv / math.sqrt(3)) / Iv
    elif conexion == 'triangulo':
        r.paso('Valores de fase (triángulo)', '', r'V_f = V_L = ' + _n(Vv) + r'\ \text{V},\quad I_f = \frac{I_L}{\sqrt3} = ' + _n(Iv / math.sqrt(3), 8) + r'\ \text{A}', 1)
        Zf = Vv / (Iv / math.sqrt(3))
    else:
        Zf = None
    if Zf is not None:
        _extra(r, 'Impedancia por fase', '', r'|Z_f| = \frac{V_f}{I_f} = ' + _n(Zf, 8) + r'\ \Omega\angle ' + _n(math.degrees(phi), 6) + r'^\circ')
    r.valor = _F(Pp, 12)
    r.valores.update({'P': _n(Pp, 12), 'Q': _n(Q, 12), 'S': _n(S, 12), 'I': _n(Iv, 12)})
    r.resultado_latex = r'P \approx ' + _n(Pp, 8) + r'\ \text{W},\ Q \approx ' + _n(Q, 8) + r'\ \text{var},\ S \approx ' + _n(S, 8) + r'\ \text{VA}'
    r.comparar = ('numero', Pp, {'tol': 1e-4})
    m = max(Pp, abs(Q))
    r.grafico = {'tipo': '2d', 'curvas': [_triangulo_potencia(Pp, Q, 'triángulo de potencias')],
                 'puntos': [G.punto(Pp, Q, f'S = {_n(S, 6)} VA', '#dc2626')],
                 'ventana': {'xmin': -0.2 * m, 'xmax': 1.4 * m, 'ymin': min(0, Q) - 0.2 * m, 'ymax': max(0, Q) + 0.2 * m}}
    return r


# ============================================================================
# Investigación operativa
# ============================================================================
class _Simplex:
    """Simplex de dos fases con aritmética exacta (racionales)."""

    def __init__(self, cmax, A, ops, b, nombres):
        self.m = len(A)
        self.nv = len(cmax)
        A = [list(f) for f in A]
        b = list(b)
        ops = list(ops)
        for i in range(self.m):
            if b[i] < 0:
                A[i] = [-a for a in A[i]]
                b[i] = -b[i]
                ops[i] = {'<=': '>=', '>=': '<=', '=': '='}[ops[i]]
        self.cols = list(nombres)
        self.T = [f[:] for f in A]
        self.base = [None] * self.m
        for i in range(self.m):
            if ops[i] == '<=':
                self.base[i] = self._col(f's_{{{i + 1}}}', i, 1)
            elif ops[i] == '>=':
                self._col(f'e_{{{i + 1}}}', i, -1)
        self.arts = []
        for i in range(self.m):
            if ops[i] in ('>=', '='):
                j = self._col(f'a_{{{i + 1}}}', i, 1)
                self.base[i] = j
                self.arts.append(j)
        for i in range(self.m):
            self.T[i].append(b[i])
        self.cmax = list(cmax)
        self.tablas = []

    def _col(self, nombre, fila, val):
        self.cols.append(nombre)
        for i in range(self.m):
            self.T[i].append(sp.Integer(val) if i == fila else sp.Integer(0))
        return len(self.cols) - 1

    def _z(self, c):
        z = [-q for q in c] + [sp.Integer(0)]
        for i in range(self.m):
            cb = c[self.base[i]]
            if cb != 0:
                z = [a + cb * q for a, q in zip(z, self.T[i])]
        return z

    def _pivot(self, i, j):
        p = self.T[i][j]
        self.T[i] = [v / p for v in self.T[i]]
        for k_ in range(self.m):
            if k_ != i and self.T[k_][j] != 0:
                f = self.T[k_][j]
                self.T[k_] = [a - f * q for a, q in zip(self.T[k_], self.T[i])]
        self.base[i] = j

    def tex_tabla(self, z, nombre_z):
        enc = r'\text{Base} & ' + ' & '.join(self.cols) + r' & b'
        filas = [self.cols[self.base[i]] + ' & ' + ' & '.join(tex(v) for v in self.T[i]) for i in range(self.m)]
        zf = nombre_z + ' & ' + ' & '.join(tex(v) for v in z)
        return (r'\begin{array}{c|' + 'c' * len(self.cols) + '|c} ' + enc + r'\\ \hline ' + r'\\ '.join(filas) + r'\\ \hline ' + zf
                + r'\end{array}')

    def _iterar(self, c, prohibidas, fase, nombre_z):
        for _ in range(60):
            z = self._z(c)
            cand = [j for j in range(len(self.cols)) if j not in prohibidas and z[j] < 0]
            if not cand:
                self.tablas.append((f'{fase}: tabla óptima', 'Ningún coeficiente de la fila ' + nombre_z.replace('\\', '') + ' es negativo.',
                                    self.tex_tabla(z, nombre_z)))
                return 'optimo', z
            j = min(cand, key=lambda q: (z[q], q))
            filas_ok = [(self.T[i][-1] / self.T[i][j], self.base[i], i) for i in range(self.m) if self.T[i][j] > 0]
            if not filas_ok:
                self.tablas.append((f'{fase}: no acotado', f'Entra {self.cols[j]} pero ninguna fila limita su crecimiento.',
                                    self.tex_tabla(z, nombre_z)))
                return 'no acotado', z
            _, _, i = min(filas_ok)
            self.tablas.append((f'{fase}: iteración {len(self.tablas) + 1}',
                                f'Entra {self.cols[j]} (coeficiente más negativo) y sale {self.cols[self.base[i]]} (menor cociente b/a); '
                                f'pivote = {self.T[i][j]}.', self.tex_tabla(z, nombre_z)))
            self._pivot(i, j)
        return 'ciclo', self._z(c)

    def resolver(self):
        if self.arts:
            c1 = [sp.Integer(-1) if j in self.arts else sp.Integer(0) for j in range(len(self.cols))]
            estado, z = self._iterar(c1, set(), 'Fase 1', 'W')
            if z[-1] < 0:
                return 'infactible'
            for i in range(self.m):
                if self.base[i] in self.arts:
                    j = next((q for q in range(len(self.cols)) if q not in self.arts and self.T[i][q] != 0), None)
                    if j is not None:
                        self._pivot(i, j)
            quedan = [i for i in range(self.m) if self.base[i] in self.arts]
            for i in reversed(quedan):
                del self.T[i]
                del self.base[i]
            self.m = len(self.T)
            keep = [j for j in range(len(self.cols)) if j not in self.arts]
            self.T = [[f[j] for j in keep] + [f[-1]] for f in self.T]
            self.base = [keep.index(q) for q in self.base]
            self.cols = [self.cols[j] for j in keep]
        c2 = self.cmax + [sp.Integer(0)] * (len(self.cols) - self.nv)
        estado, z = self._iterar(c2, set(), 'Fase 2' if self.arts else 'Simplex', 'Z')
        self.z = z
        if estado != 'optimo':
            return estado
        self.x = [sp.Integer(0)] * self.nv
        for i in range(self.m):
            if self.base[i] < self.nv:
                self.x[self.base[i]] = self.T[i][-1]
        self.valor = z[-1]
        self.alternativas = any(z[j] == 0 and j not in self.base for j in range(len(self.cols)))
        return 'optimo'


def _lineal(expr, vs, texto):
    coefs = [sp.nsimplify(sp.diff(expr, v_)) for v_ in vs]
    if any(not c.is_number for c in coefs):
        raise ErrorEntrada(f'«{texto.strip()}» no es lineal: la programación lineal solo admite expresiones lineales.')
    return coefs, sp.nsimplify(expr.subs({v_: 0 for v_ in vs}))


def _hull(pts):
    pts = sorted(set(pts), key=lambda p: (p[0], p[1]))
    if len(pts) <= 2:
        return pts, pts

    def cruz(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    inf, sup = [], []
    for p in pts:
        while len(inf) >= 2 and cruz(inf[-2], inf[-1], p) <= 0:
            inf.pop()
        inf.append(p)
    for p in reversed(pts):
        while len(sup) >= 2 and cruz(sup[-2], sup[-1], p) <= 0:
            sup.pop()
        sup.append(p)
    return inf, list(reversed(sup))


def _poligonal(cadena):
    tramos = []
    for (x1, y1), (x2, y2) in zip(cadena, cadena[1:]):
        if x2 > x1:
            tramos.append((y1 + (y2 - y1) / (x2 - x1) * (x - x1), x <= x2))
    if not tramos:
        return None
    tramos[-1] = (tramos[-1][0], True)
    return sp.Piecewise(*tramos)


def programacion_lineal(arg):
    uso = 'programacion lineal max 3x+5y; x<=4; 2y<=12; 3x+2y<=18   (x, y ≥ 0 se agregan solas)'
    s_ = normalizar(arg).replace('=<', '<=').replace('=>', '>=')
    s_ = re.sub(r'(?i)(sujeto\s+a|s\.\s*a\.?)\s*:?', ';', s_)
    partes = [p.strip() for p in (s_.split(';') if ';' in s_ else dividir(s_)) if p.strip()]
    if len(partes) < 2:
        raise ErrorEntrada('Uso: ' + uso)
    m = re.match(r'^\s*(max\w*|min\w*)\s*:?\s*(?:[zZ]\s*=\s*)?(.+)$', sin_tildes(partes[0]), re.I)
    if not m:
        raise ErrorEntrada('La primera parte debe ser la función objetivo: «max 3x+5y» o «min 2x+3y». Uso: ' + uso)
    maximizar = m.group(1).lower().startswith('max')
    obj = leer(m.group(2))
    crudas = []
    for p in partes[1:]:
        trozos = re.split(r'(<=|>=|<|>|=)', p)
        if len(trozos) != 3:
            raise ErrorEntrada(f'No entendí la restricción «{p}». Usá <=, >= o =.')
        izq, op, der = trozos
        op = {'<': '<=', '>': '>='}.get(op, op)
        if ',' in izq and op == '>=' and leer(der) == 0:
            continue  # "x, y >= 0"
        crudas.append((leer(izq) - leer(der), op, p))
    vs = variables_de(sp.Tuple(obj, *[e for e, _, _ in crudas]))
    if not vs:
        raise ErrorEntrada('No hay variables de decisión.')
    c, _ = _lineal(obj, vs, m.group(2))
    A, ops, b = [], [], []
    for e, op, p in crudas:
        coefs, cte = _lineal(e, vs, p)
        if all(q == 0 for q in coefs):
            continue
        nz = [i for i, q in enumerate(coefs) if q != 0]
        if len(nz) == 1 and -cte == 0 and ((op == '>=' and coefs[nz[0]] > 0) or (op == '<=' and coefs[nz[0]] < 0)):
            continue  # no negatividad (ya se supone)
        A.append(coefs)
        ops.append(op)
        b.append(-cte)
    r = Resultado(TEMA_IO, 'Programación lineal')
    nombres = [tex(v_) for v_ in vs]
    r.entrada_latex = (r'\max' if maximizar else r'\min') + ' Z = ' + tex(obj) + r'\quad\text{s.a.}\ ' + r',\ '.join(
        tex(sum(a * v_ for a, v_ in zip(fila, vs))) + {'<=': r'\le ', '>=': r'\ge ', '=': '='}[op] + tex(bb) for fila, op, bb in zip(A, ops, b)) \
        + r',\ ' + ', '.join(nombres) + r'\ge 0'
    if not A:
        raise ErrorEntrada('Faltan las restricciones.')
    cmax = c if maximizar else [-q for q in c]
    sx = _Simplex(cmax, A, ops, b, nombres)
    estado = sx.resolver()
    if len(vs) == 2:
        _pl_grafico(r, vs, c, A, ops, b, maximizar, estado, sx)
    else:
        r.paso('Forma estándar', 'Se agregan holguras s (≤), excesos e (≥) y artificiales a (≥, =); si hay artificiales se usa el método de dos fases.'
               + ('' if maximizar else ' Para minimizar Z se maximiza −Z.'), '', 0)
    for titulo, texto, latex in sx.tablas[:14]:
        r.paso(titulo, texto, latex, 1 if len(vs) == 2 else 0)
    if len(sx.tablas) > 14:
        r.avisos.append(f'Se muestran 14 de {len(sx.tablas)} tablas.')
    if estado == 'infactible':
        r.paso('Conclusión', 'La fase 1 termina con artificiales positivas: no hay ningún punto que cumpla todas las restricciones.',
               r'\text{Problema infactible}', 0)
        r.valor = 'infactible'
        r.resultado_latex = r'\text{No tiene solución factible}'
        return r
    if estado != 'optimo':
        r.paso('Conclusión', 'La función objetivo puede mejorar indefinidamente dentro de la región factible.', r'\text{Solución no acotada}', 0)
        r.valor = 'no acotado'
        r.resultado_latex = r'\text{Solución no acotada}'
        return r
    Zopt = sx.valor if maximizar else -sx.valor
    sol = {v_: q for v_, q in zip(vs, sx.x)}
    if len(vs) > 2:
        r.tabla = {'columnas': ['Base'] + [re.sub(r'[{}\\]', '', q) for q in sx.cols] + ['b'],
                   'filas': [[re.sub(r'[{}\\]', '', sx.cols[sx.base[i]])] + [str(q) for q in sx.T[i]] for i in range(sx.m)]
                   + [['Z'] + [str(q) for q in sx.z]]}
    r.paso('Solución óptima', '', r',\ '.join(f'{tex(v_)} = {tex(q)}' for v_, q in sol.items()) + r',\quad Z = ' + tex(Zopt), 0)
    if sx.alternativas:
        r.avisos.append('Hay soluciones óptimas alternativas (una variable no básica tiene costo reducido 0).')
    r.valor = sol
    r.valores['Z'] = _n(Zopt, 12)
    r.valores['Z_exacto'] = str(Zopt)
    for v_, q in sol.items():
        r.valores[str(v_)] = _n(q, 12)
    r.resultado_latex = r',\ '.join(f'{tex(v_)} = {tex(q)}' for v_, q in sol.items()) + r',\quad Z_{' + ('max' if maximizar else 'min') + '} = ' + tex(Zopt)
    r.comparar = ('dict', sol, {})
    return r


def _pl_grafico(r, vs, c, A, ops, b, maximizar, estado, sx):
    uno, cero = sp.Integer(1), sp.Integer(0)
    restr = list(zip(A, ops, b)) + [([uno, cero], '>=', cero), ([cero, uno], '>=', cero)]

    def factible(p, extra=()):
        for fila, op, bb in list(restr) + list(extra):
            v_ = fila[0] * p[0] + fila[1] * p[1]
            if (op == '<=' and v_ > bb) or (op == '>=' and v_ < bb) or (op == '=' and v_ != bb):
                return False
        return True

    def cortes(lineas):
        pts = []
        for i in range(len(lineas)):
            for j in range(i + 1, len(lineas)):
                (a1, a2), b1 = lineas[i][0], lineas[i][2]
                (c1, c2), b2 = lineas[j][0], lineas[j][2]
                det = a1 * c2 - a2 * c1
                if det != 0:
                    pts.append(((b1 * c2 - a2 * b2) / det, (a1 * b2 - b1 * c1) / det))
        return pts

    todos = cortes(restr)
    vert = sorted({p for p in todos if factible(p)}, key=lambda p: (p[0], p[1]))
    r.paso('Método gráfico', 'Cada restricción es un semiplano; la región factible es su intersección (con x, y ≥ 0). '
           'Si hay óptimo, está en un vértice: se evalúa Z en todos.', '', 0)
    filas = []
    mejor = None
    for p in vert:
        zv = c[0] * p[0] + c[1] * p[1]
        filas.append(['(' + str(p[0]) + ', ' + str(p[1]) + ')', _n(p[0], 8), _n(p[1], 8), str(zv)])
        if mejor is None or (zv > mejor[1] if maximizar else zv < mejor[1]):
            mejor = (p, zv)
    if estado == 'optimo' and mejor is not None:
        for f_ in filas:
            if f_[3] == str(mejor[1]):
                f_.append('← óptimo')
    r.tabla = {'columnas': ['vértice', tex(vs[0]), tex(vs[1]), 'Z'], 'filas': filas}
    if vert:
        r.paso('Vértices de la región factible', '', r',\ '.join(f'({tex(p[0])}, {tex(p[1])})' for p in vert), 0)
    else:
        r.paso('Región factible', 'No hay vértices factibles.', '', 0)
    # Gráfico
    coords = [abs(float(q)) for p in todos for q in p] + [1.0]
    B = sp.nsimplify(max(coords) * 1.6 + 1)
    caja = [([uno, cero], '<=', B), ([cero, uno], '<=', B)]
    lineas_caja = restr + caja
    poli = [p for p in cortes(lineas_caja) if factible(p, caja)]
    curvas, puntos, sombra = [], [], None
    xs_v = [float(p[0]) for p in vert] or [0.0, float(B)]
    ys_v = [float(p[1]) for p in vert] or [0.0, float(B)]
    acotada = estado == 'optimo' and all(float(max(p)) < float(B) * 0.6 for p in vert) and len(vert) >= 1
    xmax_v = max(xs_v + [1.0]) * 1.3 + 0.5
    ymax_v = max(ys_v + [1.0]) * 1.3 + 0.5
    if not acotada:
        xmax_v, ymax_v = max(xmax_v, float(B) * 0.7), max(ymax_v, float(B) * 0.7)
    for k_, (fila, op, bb) in enumerate(zip(A, ops, b)):
        et = tex(fila[0] * vs[0] + fila[1] * vs[1]) + {'<=': r'\le ', '>=': r'\ge ', '=': '='}[op] + tex(bb)
        col = G.COLORES[(k_ + 1) % len(G.COLORES)]
        if fila[1] != 0:
            curvas.append(G.curva((bb - fila[0] * x) / fila[1], x, et, col))
        else:
            x0 = float(bb / fila[0])
            curvas.append(_curva_puntos([(x0, -10 * ymax_v), (x0, 10 * ymax_v)], et, col))
    if len(poli) >= 3:
        inf, sup = _hull([(sp.nsimplify(p[0]), sp.nsimplify(p[1])) for p in poli])
        f_inf, f_sup = _poligonal(inf), _poligonal(sup)
        if f_inf is not None and f_sup is not None:
            xs_p = [float(p[0]) for p in poli]
            sombra = G.sombra_entre(f_sup, f_inf, min(xs_p), max(xs_p), x)
    for p in vert:
        puntos.append(G.punto(p[0], p[1], f'({_n(p[0], 5)}, {_n(p[1], 5)})'))
    if estado == 'optimo' and mejor is not None:
        p, zv = mejor
        puntos.append(G.punto(p[0], p[1], f'óptimo Z = {_n(zv, 6)}', '#dc2626'))
        if c[1] != 0:
            curvas.append(G.curva((zv - c[0] * x) / c[1], x, f'Z = {_n(zv, 6)}', '#dc2626', 'punteado'))
        elif c[0] != 0:
            curvas.append(_curva_puntos([(float(zv / c[0]), -10 * ymax_v), (float(zv / c[0]), 10 * ymax_v)], f'Z = {_n(zv, 6)}', '#dc2626',
                                        'punteado'))
        r.paso('Evaluar Z en los vértices', 'El ' + ('máximo' if maximizar else 'mínimo') + ' se alcanza en el vértice marcado.',
               f'Z({tex(p[0])}, {tex(p[1])}) = {tex(zv)}', 0)
        r.paso('Comprobación con el método simplex', 'Mismo resultado con tablas:', '', 0)
    g = {'tipo': '2d', 'curvas': curvas, 'puntos': [q for q in puntos if q],
         'ventana': {'xmin': -0.1 * xmax_v, 'xmax': xmax_v, 'ymin': -0.1 * ymax_v, 'ymax': ymax_v}}
    if sombra:
        g['sombra'] = sombra
    r.grafico = g


def _emparejar(Z):
    n_ = len(Z)
    mr, mc = [-1] * n_, [-1] * n_

    def probar(i, vistos):
        for j in range(n_):
            if Z[i][j] and not vistos[j]:
                vistos[j] = True
                if mc[j] == -1 or probar(mc[j], vistos):
                    mr[i], mc[j] = j, i
                    return True
        return False
    for i in range(n_):
        probar(i, [False] * n_)
    return mr, mc


def _cubrir(Z, mr, mc):
    """Mínima cantidad de líneas que cubren los ceros (teorema de König)."""
    n_ = len(Z)
    filas_vis = {i for i in range(n_) if mr[i] == -1}
    cols_vis = set()
    pila = list(filas_vis)
    while pila:
        i = pila.pop()
        for j in range(n_):
            if Z[i][j] and j not in cols_vis:
                cols_vis.add(j)
                k_ = mc[j]
                if k_ != -1 and k_ not in filas_vis:
                    filas_vis.add(k_)
                    pila.append(k_)
    return [i for i in range(n_) if i not in filas_vis], sorted(cols_vis)


def asignacion(arg):
    s_ = arg.strip()
    maximizar = bool(re.search(r'\bmax', sin_tildes(s_.lower())))
    s_ = re.sub(r'(?i)\b(max\w*|min\w*)\b', '', s_).strip(' ,;')
    if not s_.startswith('[['):
        raise ErrorEntrada('Uso: asignacion [[c11,c12,...],[c21,...],...][, max]   (método húngaro; filas = agentes, columnas = tareas)')
    M = leer_matriz(dividir(s_)[0]).applyfunc(sp.nsimplify)
    f_, c_ = M.shape
    n_ = max(f_, c_)
    C = [[M[i, j] if i < f_ and j < c_ else sp.Integer(0) for j in range(n_)] for i in range(n_)]
    r = Resultado(TEMA_IO, 'Problema de asignación (método húngaro)')
    r.entrada_latex = (r'\max\ ' if maximizar else '') + tex(M)
    if f_ != c_:
        r.paso('Matriz cuadrada', f'Se agregan {"filas" if f_ < c_ else "columnas"} ficticias de costo 0.', tex(sp.Matrix(C)), 0)
    if maximizar:
        mx = max(max(fila) for fila in C)
        C = [[mx - q for q in fila] for fila in C]
        r.paso('Maximización', f'Se convierte a minimización restando cada elemento del máximo ({tex(mx)}).', tex(sp.Matrix(C)), 0)
    W = [fila[:] for fila in C]
    for i in range(n_):
        mi = min(W[i])
        W[i] = [q - mi for q in W[i]]
    r.paso('Reducción por filas', 'A cada fila se le resta su mínimo.', tex(sp.Matrix(W)), 0)
    for j in range(n_):
        mj = min(W[i][j] for i in range(n_))
        for i in range(n_):
            W[i][j] -= mj
    r.paso('Reducción por columnas', 'A cada columna se le resta su mínimo.', tex(sp.Matrix(W)), 0)
    for it in range(4 * n_ * n_ + 4):
        Z = [[q == 0 for q in fila] for fila in W]
        mr, mc = _emparejar(Z)
        k_ = sum(1 for q in mr if q != -1)
        if k_ == n_:
            r.paso('Prueba de optimalidad', f'Se necesitan {n_} líneas para cubrir todos los ceros: hay una asignación óptima con costo reducido 0.', '', 0)
            break
        lf, lc = _cubrir(Z, mr, mc)
        theta = min(W[i][j] for i in range(n_) if i not in lf for j in range(n_) if j not in lc)
        r.paso(f'Iteración {it + 1}: cubrir ceros', f'Alcanzan {k_} líneas (< {n_}): filas {[i + 1 for i in lf]}, columnas {[j + 1 for j in lc]}. '
               f'El menor elemento no cubierto es θ = {theta}: se resta a los no cubiertos y se suma a las intersecciones.', '', 0)
        for i in range(n_):
            for j in range(n_):
                if i not in lf and j not in lc:
                    W[i][j] -= theta
                elif i in lf and j in lc:
                    W[i][j] += theta
        r.paso('Nueva matriz', '', tex(sp.Matrix(W)), 1)
    asign = [(i, mr[i]) for i in range(n_)]
    reales = [(i, j) for i, j in asign if i < f_ and j < c_]
    total = sum(M[i, j] for i, j in reales)
    filas = [[str(i + 1), str(j + 1), _n(M[i, j], 10)] for i, j in reales]
    for i, j in asign:
        if not (i < f_ and j < c_):
            filas.append([str(i + 1) + (' (ficticia)' if i >= f_ else ''), str(j + 1) + (' (ficticia)' if j >= c_ else ''), '0'])
    r.tabla = {'columnas': ['fila (agente)', 'columna (tarea)', 'costo' if not maximizar else 'beneficio'], 'filas': filas + [['', 'Total', _n(total, 10)]]}
    r.paso('Asignación', 'Se elige un cero por fila y por columna.', r',\ '.join(f'{i + 1}\\to {j + 1}' for i, j in reales), 0)
    r.paso('Valor ' + ('máximo' if maximizar else 'mínimo'), 'Se suman los valores de la matriz ORIGINAL.',
           ' + '.join(tex(M[i, j]) for i, j in reales) + ' = ' + tex(total), 0)
    r.valor = total
    r.valores['asignacion'] = ', '.join(f'{i + 1}→{j + 1}' for i, j in reales)
    r.resultado_latex = r',\ '.join(f'{i + 1}\\to {j + 1}' for i, j in reales) + r',\quad ' + ('Z_{max}' if maximizar else r'\text{costo}') + ' = ' + tex(total)
    r.comparar = ('numero', total, {})
    return r


def cola(arg, multiple=False):
    uso = 'cola mm1 λ=2, μ=3[, n=k]   ;   cola mmc λ=2, μ=3, c=2'
    d, pos = _kv(arg)
    lam = _get(d, 'lambda', 'lamda', 'l', 'tasadellegada', 'llegadas', 'tasallegada')
    mu = _get(d, 'mu', 'tasadeservicio', 'servicio', 'tasaservicio', 'm')
    cc = _get(d, 'c', 's', 'servidores', 'k')
    if lam is None and mu is None and len(pos) >= 2:
        lam, mu = pos[0], pos[1]
        if multiple and len(pos) >= 3:
            cc = pos[2]
    if lam is None or mu is None:
        raise ErrorEntrada('Faltan λ y μ. Uso: ' + uso)
    L_, M_ = _val(lam), _val(mu)
    if L_ <= 0 or M_ <= 0:
        raise ErrorEntrada('λ y μ deben ser positivos.')
    c = _entero(cc, 'c (servidores)', 1) if (multiple and cc is not None) else 1
    if multiple and cc is None:
        raise ErrorEntrada('Falta c (cantidad de servidores). Uso: ' + uso)
    rho = L_ / (c * M_)
    nq = _get(d, 'n')
    r = Resultado(TEMA_IO, f'Teoría de colas M/M/{c}')
    r.entrada_latex = r'\lambda = ' + tex(L_) + r',\ \mu = ' + tex(M_) + (r',\ c = ' + str(c) if c > 1 else '')
    r.paso('Modelo', 'Llegadas de Poisson (tasa λ), servicio exponencial (tasa μ), ' + (f'{c} servidores' if c > 1 else 'un servidor')
           + ', capacidad y población infinitas, disciplina FIFO.', '', 0)
    r.paso('Factor de utilización', 'El sistema es estable solo si ρ < 1.',
           r'\rho = \frac{\lambda}{' + ('c\\,' if c > 1 else '') + r'\mu} = ' + tex(rho) + (r'\approx ' + _n(rho, 8) if not rho.is_Integer else ''), 0)
    if rho >= 1:
        raise ErrorEntrada(f'ρ = {_n(rho, 6)} ≥ 1: la cola crece sin límite (el sistema no es estable). Hace falta μ' + (' o c' if c > 1 else '')
                           + ' mayor.')
    a = L_ / M_
    if c == 1:
        P0 = 1 - rho
        L = rho / (1 - rho)
        Lq = rho ** 2 / (1 - rho)
        W = 1 / (M_ - L_)
        Wq = L_ / (M_ * (M_ - L_))
        pn = lambda k_: (1 - rho) * rho ** k_  # noqa
        formulas = [('P₀', r'1-\rho', P0), ('L', r'\frac{\rho}{1-\rho}', L), ('L_q', r'\frac{\rho^2}{1-\rho}', Lq),
                    ('W', r'\frac{1}{\mu-\lambda}', W), ('W_q', r'\frac{\lambda}{\mu(\mu-\lambda)}', Wq)]
    else:
        suma = sum(a ** k_ / sp.factorial(k_) for k_ in range(c))
        cola_ = a ** c / (sp.factorial(c) * (1 - rho))
        P0 = 1 / (suma + cola_)
        Lq = P0 * a ** c * rho / (sp.factorial(c) * (1 - rho) ** 2)
        L = Lq + a
        Wq = Lq / L_
        W = Wq + 1 / M_
        pesp = P0 * cola_
        pn = lambda k_: P0 * a ** k_ / sp.factorial(k_) if k_ < c else P0 * a ** k_ / (sp.factorial(c) * c ** (k_ - c))  # noqa
        formulas = [('P₀', r'\left[\sum_{n=0}^{c-1}\frac{(\lambda/\mu)^n}{n!} + \frac{(\lambda/\mu)^c}{c!(1-\rho)}\right]^{-1}', P0),
                    ('L_q', r'\frac{P_0(\lambda/\mu)^c\rho}{c!(1-\rho)^2}', Lq), ('L', r'L_q + \frac{\lambda}{\mu}', L),
                    ('W_q', r'\frac{L_q}{\lambda}', Wq), ('W', r'W_q + \frac1\mu', W),
                    ('P(esperar)', r'\frac{(\lambda/\mu)^c}{c!(1-\rho)}P_0\ \text{(Erlang C)}', pesp)]
    for nom, fml, val in formulas:
        nt = nom.replace('₀', '_0')
        r.paso(nom, '', nt + ' = ' + fml + ' = ' + tex(sp.nsimplify(val)) + (r'\approx ' + _n(val, 8) if not sp.nsimplify(val).is_Integer else ''), 0)
    r.paso('Ley de Little', 'Comprobación:', r'L = \lambda W = ' + tex(L_) + r'\cdot ' + tex(sp.nsimplify(W)) + ' = ' + tex(sp.nsimplify(L_ * W)), 1)
    r.tabla = {'columnas': ['Medida', 'Significado', 'Valor'],
               'filas': [['ρ', 'utilización ' + ('de cada servidor' if c > 1 else 'del servidor'), _n(rho, 10)],
                         ['P₀', 'probabilidad de sistema vacío', _n(P0, 10)], ['L', 'clientes en el sistema (promedio)', _n(L, 10)],
                         ['Lq', 'clientes en la cola', _n(Lq, 10)], ['W', 'tiempo en el sistema', _n(W, 10)], ['Wq', 'tiempo en la cola', _n(Wq, 10)]]}
    if nq is not None:
        k_ = _entero(nq, 'n', 0)
        pk = pn(k_)
        r.paso(f'Probabilidad de {k_} clientes', '', f'P_{{{k_}}} = ' + tex(sp.nsimplify(pk)) + r'\approx ' + _n(pk, 8), 0)
        if c == 1:
            r.paso(f'Más de {k_} clientes', '', f'P(N > {k_}) = \\rho^{{{k_ + 1}}} = ' + tex(rho ** (k_ + 1)) + r'\approx ' + _n(rho ** (k_ + 1), 8), 1)
        r.valores['Pn'] = _n(pk, 12)
    r.valor = sp.nsimplify(L)
    r.valores.update({'rho': _n(rho, 12), 'P0': _n(P0, 12), 'L': _n(L, 12), 'Lq': _n(Lq, 12), 'W': _n(W, 12), 'Wq': _n(Wq, 12)})
    r.resultado_latex = r'\rho = ' + tex(rho) + r',\ L = ' + tex(sp.nsimplify(L)) + r',\ L_q = ' + tex(sp.nsimplify(Lq)) + r',\ W = ' + \
        tex(sp.nsimplify(W)) + r',\ W_q = ' + tex(sp.nsimplify(Wq))
    r.comparar = ('numero', sp.nsimplify(L), {'tol': 1e-6})
    tope = int(min(30, max(8, math.ceil(4 * float(L) + 4))))
    r.grafico = {'tipo': 'barras', 'barras': [{'x': k_, 'alto': float(pn(k_)), 'resaltar': nq is not None and k_ == int(_val(nq))}
                                              for k_ in range(tope + 1)]}
    return r


# ============================================================================
# Registro de comandos (se prueban ANTES que los de api.COMANDOS)
# ============================================================================
COMANDOS_EXTRA = [
    # Probabilidad
    (('cuantil normal', 'cuantil z', 'z critico', 'normal estandar inversa'), lambda a: cuantil(a, 'normal')),
    (('cuantil t', 'cuantil student', 't critico'), lambda a: cuantil(a, 't')),
    (('cuantil chi2', 'cuantil chi cuadrado', 'cuantil chi^2', 'cuantil χ2', 'cuantil χ²', 'chi2 critico'), lambda a: cuantil(a, 'chi2')),
    (('cuantil f', 'f critico'), lambda a: cuantil(a, 'f')),
    (('normal inversa', 'inversa normal', 'inversa de la normal'), normal_inversa),
    (('hipergeometrica',), hipergeometrica),
    (('geometrica',), geometrica),
    (('distribucion exponencial', 'exponencial'), exponencial),
    (('distribucion uniforme', 'uniforme'), uniforme),
    (('t student', 't de student', 'student'), lambda a: _prob_continua(a, 't')),
    (('chi2 bondad', 'chi cuadrado bondad', 'bondad de ajuste', 'χ2 bondad', 'χ² bondad'), chi2_bondad),
    (('chi2 independencia', 'chi cuadrado independencia', 'tabla de contingencia', 'χ2 independencia', 'χ² independencia'), chi2_independencia),
    (('chi2', 'chi cuadrado', 'χ2', 'χ²'), lambda a: _prob_continua(a, 'chi2')),
    (('f snedecor', 'f de fisher', 'fisher'), lambda a: _prob_continua(a, 'f')),
    (('probabilidad total',), probabilidad_total),
    (('teorema de bayes', 'bayes'), bayes),
    # Estadística descriptiva
    (('percentil',), percentil),
    (('cuartiles', 'boxplot', 'diagrama de caja', 'box plot'), cuartiles),
    (('coeficiente de asimetria', 'asimetria'), asimetria),
    (('coeficiente de curtosis', 'curtosis', 'kurtosis'), curtosis),
    (('tabla de frecuencias', 'tabla de frecuencia', 'frecuencias'), tabla_frecuencias),
    # Inferencia
    (('intervalo media', 'intervalo de confianza media', 'intervalo para la media', 'ic media'), intervalo_media),
    (('intervalo varianza', 'intervalo de confianza varianza', 'intervalo para la varianza', 'ic varianza'), intervalo_varianza),
    (('intervalo proporcion', 'intervalo de confianza proporcion', 'intervalo para la proporcion', 'ic proporcion'), intervalo_proporcion),
    (('tamano de muestra', 'tamano muestral', 'tamano de la muestra'), tamano_muestra),
    (('prueba z', 'test z', 'contraste z'), lambda a: prueba_media(a, 'z')),
    (('prueba t', 'test t', 'contraste t'), lambda a: prueba_media(a, 't')),
    (('prueba proporcion', 'test proporcion', 'prueba de proporcion'), prueba_proporcion),
    (('correlacion', 'coeficiente de correlacion', 'pearson'), correlacion),
    # Cálculo numérico
    (('regula falsi', 'falsa posicion', 'regla falsa'), regula_falsi),
    (('muller',), muller),
    (('gauss seidel', 'gauss-seidel'), gauss_seidel),
    (('jacobi',), jacobi),
    (('factorizacion lu', 'descomposicion lu', 'lu'), lu),
    (('spline natural', 'spline cubica', 'spline', 'trazador cubico', 'trazadores cubicos'), spline),
    (('interpolacion de hermite', 'hermite'), hermite),
    (('minimos cuadrados grado', 'ajuste polinomico grado', 'regresion polinomica grado', 'ajuste polinomico', 'regresion polinomica'),
     minimos_cuadrados),
    (('ajuste exponencial', 'regresion exponencial'), lambda a: ajuste_no_lineal(a, 'exponencial')),
    (('ajuste potencial', 'regresion potencial'), lambda a: ajuste_no_lineal(a, 'potencial')),
    (('simpson 3/8', 'simpson tres octavos', 'simpson38'), simpson38),
    (('romberg',), romberg),
    (('gauss legendre', 'gauss-legendre', 'cuadratura de gauss', 'cuadratura gaussiana'), gauss_legendre),
    (('derivada numerica', 'derivacion numerica', 'diferencias finitas'), derivada_numerica),
    (('adams bashforth moulton', 'adams moulton', 'predictor corrector'), lambda a: adams(a, True)),
    (('adams bashforth', 'adams'), adams),
    (('rk4 sistema', 'runge kutta sistema', 'runge-kutta sistema', 'sistema rk4', 'rk4 sistemas'), rk4_sistema),
    (('propagacion de errores', 'propagacion de error', 'propagacion'), propagacion_errores),
    (('error absoluto', 'error relativo', 'errores', 'error'), error_aprox),
    # Electrotecnia
    (('potencia compleja',), potencia_compleja),
    (('correccion fp', 'correccion del factor de potencia', 'correccion de factor de potencia', 'correccion factor de potencia',
      'compensacion reactiva'), correccion_fp),
    (('potencia trifasica', 'trifasico', 'trifasica'), trifasico),
    # Investigación operativa
    (('programacion lineal', 'simplex', 'metodo grafico', 'metodo simplex'), programacion_lineal),
    (('asignacion', 'metodo hungaro', 'hungaro'), asignacion),
    (('cola m/m/1', 'cola mm1', 'm/m/1', 'mm1'), lambda a: cola(a, False)),
    (('cola m/m/c', 'cola mmc', 'cola m/m/s', 'cola mms', 'm/m/c', 'mmc', 'm/m/s'), lambda a: cola(a, True)),
]
