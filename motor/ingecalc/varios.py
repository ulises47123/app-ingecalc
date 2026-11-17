"""Probabilidad y estadística; comando graficar."""
import re

import sympy as sp

from . import graficos as G
from .nucleo import (ErrorEntrada, Resultado, dividir, leer, leer_ecuacion, tex, variables_de, x, y, t, fmt_num,
                     normalizar, decimal)


def estadistica(arg):
    datos = [sp.nsimplify(leer(p)) for p in dividir(arg.replace(';', ','))]
    if len(datos) < 2:
        raise ErrorEntrada('Ingresá al menos 2 datos separados por comas.')
    n_ = len(datos)
    r = Resultado('Estadística', 'Estadística descriptiva')
    r.entrada_latex = r'\{' + ', '.join(tex(d) for d in datos) + r'\}'
    media = sp.Rational(sum(datos), n_) if all(d.is_Rational for d in datos) else sum(datos) / n_
    orden = sorted(datos, key=float)
    mediana = orden[n_ // 2] if n_ % 2 else (orden[n_ // 2 - 1] + orden[n_ // 2]) / 2
    frec = {}
    for d in datos:
        frec[d] = frec.get(d, 0) + 1
    fmax = max(frec.values())
    moda = [d for d, f in frec.items() if f == fmax] if fmax > 1 else []
    sc = sum((d - media) ** 2 for d in datos)
    var_m = sc / (n_ - 1)
    var_p = sc / n_
    r.paso('Media', r'\(\bar x = \frac{\sum x_i}{n}\)', r'\bar x = \frac{' + tex(sum(datos)) + '}{' + str(n_) + '} = ' + tex(media), 0)
    r.paso('Mediana', 'Datos ordenados; valor central (o promedio de los dos centrales).',
           tex(orden) + r'\Rightarrow Me = ' + tex(mediana), 0)
    r.paso('Moda', 'Valor(es) más frecuente(s).', 'Mo = ' + (tex(sorted(moda, key=float)) if moda else r'\text{no hay}'), 0)
    r.paso('Varianza muestral', r'\(s^2 = \frac{\sum (x_i - \bar x)^2}{n-1}\)',
           's^2 = \\frac{' + tex(sp.nsimplify(sc)) + '}{' + str(n_ - 1) + '} = ' + tex(sp.nsimplify(var_m)) + r'\approx ' + fmt_num(sp.N(var_m), 10), 0)
    r.paso('Desvío estándar muestral', '', r's = \sqrt{s^2} \approx ' + fmt_num(sp.N(sp.sqrt(var_m)), 10), 0)
    filas = [[fmt_num(d), str(f), fmt_num(sp.Rational(f, n_), 4)] for d, f in sorted(frec.items(), key=lambda kv: float(kv[0]))]
    r.tabla = {'columnas': ['xᵢ', 'frecuencia', 'f. relativa'], 'filas': filas}
    r.valor = media
    q1 = orden[: n_ // 2]
    q3 = orden[(n_ + 1) // 2:]
    med = lambda L: L[len(L) // 2] if len(L) % 2 else (L[len(L) // 2 - 1] + L[len(L) // 2]) / 2  # noqa
    r.valores.update({
        'media': fmt_num(sp.N(media), 12), 'mediana': fmt_num(sp.N(mediana), 12),
        'moda': ', '.join(fmt_num(m) for m in sorted(moda, key=float)) if moda else 'no hay',
        'varianza': fmt_num(sp.N(var_m), 12), 'desvio': fmt_num(sp.N(sp.sqrt(var_m)), 12),
        'varianza_pob': fmt_num(sp.N(var_p), 12), 'desvio_pob': fmt_num(sp.N(sp.sqrt(var_p)), 12),
        'n': n_, 'rango': fmt_num(orden[-1] - orden[0], 12),
        'q1': fmt_num(sp.N(med(q1)), 12) if q1 else '', 'q3': fmt_num(sp.N(med(q3)), 12) if q3 else '',
        'cv': fmt_num(sp.N(sp.sqrt(var_m) / media), 8) if media != 0 else '',
    })
    r.resultado_latex = (r'\bar x = ' + tex(media) + r',\ Me = ' + tex(mediana) + r',\ s = ' + fmt_num(sp.N(sp.sqrt(var_m)), 8)
                         + r',\ s^2 = ' + fmt_num(sp.N(var_m), 8))
    r.comparar = ('numero_o_expr', media, {})
    r.grafico = {'tipo': 'barras', 'barras': [{'x': float(d), 'alto': f} for d, f in frec.items()],
                 'media': float(media)}
    return r


def binomial(arg):
    partes = dividir(arg)
    if len(partes) < 3:
        raise ErrorEntrada('Uso: binomial n, p, k   → P(X = k)')
    n_, p, k_ = int(leer(partes[0])), sp.nsimplify(leer(partes[1])), int(leer(partes[2]))
    r = Resultado('Probabilidad', 'Distribución binomial')
    r.entrada_latex = r'X \sim B(' + str(n_) + ', ' + tex(p) + r'),\quad P(X = ' + str(k_) + ')'
    val = sp.binomial(n_, k_) * p ** k_ * (1 - p) ** (n_ - k_)
    r.paso('Fórmula', r'\(P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}\)',
           r'\binom{' + str(n_) + '}{' + str(k_) + '}(' + tex(p) + ')^{' + str(k_) + '}(' + tex(1 - p) + ')^{' + str(n_ - k_) + '} = '
           + tex(sp.binomial(n_, k_)) + r'\cdot' + tex(p ** k_) + r'\cdot' + tex((1 - p) ** (n_ - k_)), 0)
    acum = sum(sp.binomial(n_, i) * p ** i * (1 - p) ** (n_ - i) for i in range(k_ + 1))
    r.valor = val
    r.valores['decimal'] = fmt_num(sp.N(val), 12)
    r.valores['acumulada'] = fmt_num(sp.N(acum), 12)
    r.valores['media'] = fmt_num(sp.N(n_ * p), 12)
    r.valores['varianza'] = fmt_num(sp.N(n_ * p * (1 - p)), 12)
    r.resultado_latex = 'P(X = ' + str(k_) + ') = ' + tex(val) + r'\approx ' + fmt_num(sp.N(val), 8)
    r.extra('Acumulada', r'P(X \le ' + str(k_) + r') \approx ' + fmt_num(sp.N(acum), 8))
    r.extra('Media y varianza', r'E[X] = np = ' + tex(n_ * p) + r',\ V[X] = np(1-p) = ' + tex(n_ * p * (1 - p)))
    r.comparar = ('numero', val, {'tol': 1e-6})
    r.grafico = {'tipo': 'barras', 'barras': [{'x': i, 'alto': float(sp.binomial(n_, i) * p ** i * (1 - p) ** (n_ - i)),
                                               'resaltar': i == k_} for i in range(min(n_, 60) + 1)]}
    return r


def poisson(arg):
    partes = dividir(arg)
    if len(partes) < 2:
        raise ErrorEntrada('Uso: poisson λ, k   → P(X = k)')
    lam, k_ = sp.nsimplify(leer(partes[0])), int(leer(partes[1]))
    r = Resultado('Probabilidad', 'Distribución de Poisson')
    r.entrada_latex = r'X \sim P(' + tex(lam) + r'),\quad P(X = ' + str(k_) + ')'
    val = sp.exp(-lam) * lam ** k_ / sp.factorial(k_)
    r.paso('Fórmula', r'\(P(X = k) = \frac{e^{-\lambda}\lambda^k}{k!}\)', tex(val), 0)
    acum = sum(sp.exp(-lam) * lam ** i / sp.factorial(i) for i in range(k_ + 1))
    r.valor = val
    r.valores['decimal'] = fmt_num(sp.N(val), 12)
    r.valores['acumulada'] = fmt_num(sp.N(acum), 12)
    r.resultado_latex = 'P(X = ' + str(k_) + ') = ' + tex(val) + r'\approx ' + fmt_num(sp.N(val), 8)
    r.extra('Acumulada', r'P(X \le ' + str(k_) + r') \approx ' + fmt_num(sp.N(acum), 8))
    r.comparar = ('numero', val, {'tol': 1e-6})
    tope = int(max(float(lam) * 3, k_ + 3))
    r.grafico = {'tipo': 'barras', 'barras': [{'x': i, 'alto': float(sp.N(sp.exp(-lam) * lam ** i / sp.factorial(i))),
                                               'resaltar': i == k_} for i in range(min(tope, 60) + 1)]}
    return r


def normal(arg):
    partes = dividir(arg)
    if len(partes) < 4:
        raise ErrorEntrada('Uso: normal μ, σ, a, b   → P(a < X < b)   (usá -oo u oo para colas)')
    mu, sg, a, b = (leer(p) for p in partes[:4])
    r = Resultado('Probabilidad', 'Distribución normal')
    r.entrada_latex = r'X \sim N(' + tex(mu) + ', ' + tex(sg) + r'^2),\quad P(' + tex(a) + ' < X < ' + tex(b) + ')'
    za = (a - mu) / sg if a.is_finite else -sp.oo
    zb = (b - mu) / sg if b.is_finite else sp.oo
    r.paso('Estandarizar', r'\(Z = \frac{X - \mu}{\sigma} \sim N(0,1)\)',
           'z_a = ' + (fmt_num(sp.N(za), 6) if za.is_finite else r'-\infty') + r',\ z_b = ' + (fmt_num(sp.N(zb), 6) if zb.is_finite else r'\infty'), 0)
    Phi = lambda zz: (1 + sp.erf(zz / sp.sqrt(2))) / 2  # noqa
    val = sp.N(Phi(zb) - Phi(za), 15)
    r.paso('Tabla de la normal', r'\(P = \Phi(z_b) - \Phi(z_a)\)',
           r'\Phi(' + (fmt_num(sp.N(zb), 4) if zb.is_finite else r'\infty') + r') - \Phi(' + (fmt_num(sp.N(za), 4) if za.is_finite else r'-\infty')
           + ') = ' + fmt_num(sp.N(Phi(zb)), 6) + ' - ' + fmt_num(sp.N(Phi(za)), 6) + r' \approx ' + fmt_num(val, 8), 0)
    r.valor = val
    r.resultado_latex = 'P \\approx ' + fmt_num(val, 10)
    r.comparar = ('numero', val, {'tol': 1e-4})
    dens = sp.exp(-(x - mu) ** 2 / (2 * sg ** 2)) / (sg * sp.sqrt(2 * sp.pi))
    try:
        g = G.grafico_2d([dens], x, xmin=float(mu - 4 * sg), xmax=float(mu + 4 * sg))
        g['sombra'] = G.sombra_entre(dens, sp.Integer(0), max(float(sp.N(a)) if a.is_finite else -1e9, float(mu - 5 * sg)),
                                     min(float(sp.N(b)) if b.is_finite else 1e9, float(mu + 5 * sg)), x)
        r.grafico = g
    except Exception:  # noqa
        pass
    return r


# ----------------------------------------------------------------------------
# Graficar
# ----------------------------------------------------------------------------
def graficar(arg):
    a = arg.strip()
    low = a.lower()
    r = Resultado('Gráficos', 'Gráfico')
    if low.startswith('polar'):
        cuerpo = a[5:].strip()
        cuerpo = re.sub(r'^r\s*=\s*', '', cuerpo)
        rr = leer(cuerpo)
        th = sp.Symbol('theta')
        if rr.has(th):
            rr = rr.subs(th, t)
        if rr.has(x) and not rr.has(t):
            rr = rr.subs(x, t)
        r.titulo = 'Curva en polares'
        r.entrada_latex = r'r(\theta) = ' + tex(rr.subs(t, th))
        r.grafico = G.grafico_polar(rr, t, r.entrada_latex)
        r.paso('Coordenadas polares', r'Cada punto es \((r\cos\theta,\ r\sin\theta)\) con \(\theta\in[0, 2\pi]\).', r.entrada_latex, 0)
        try:
            area = sp.simplify(sp.integrate(rr ** 2, (t, 0, 2 * sp.pi)) / 2)
            r.extra('Área encerrada (una vuelta)', r'A = \frac12\int_0^{2\pi} r^2\,d\theta = ' + tex(area))
        except Exception:  # noqa
            pass
        r.valor = rr
        r.comparar = ('expr', rr, {})
        return r
    if low.startswith('parametrica') or low.startswith('param'):
        cuerpo = re.sub(r'^param\w*\s*', '', a, flags=re.I)
        cuerpo = cuerpo.strip()
        if cuerpo.startswith('(') and cuerpo.endswith(')') and dividir(cuerpo[1:-1]) and len(dividir(cuerpo[1:-1])) >= 2:
            cuerpo = cuerpo[1:-1]
        partes = dividir(cuerpo)
        if len(partes) < 2:
            raise ErrorEntrada('Uso: graficar parametrica x(t), y(t)')
        fx, fy = leer(partes[0]), leer(partes[1])
        desde = float(leer(partes[2])) if len(partes) > 3 else None
        hasta = float(leer(partes[3])) if len(partes) > 3 else None
        r.titulo = 'Curva paramétrica'
        r.entrada_latex = r'\begin{cases} x = ' + tex(fx) + r'\\ y = ' + tex(fy) + r'\end{cases}'
        r.grafico = G.grafico_parametrica(fx, fy, t, '', desde, hasta)
        r.paso('Paramétricas', 'Se recorre t y se marca (x(t), y(t)).', r.entrada_latex, 0)
        try:
            pend = sp.simplify(sp.diff(fy, t) / sp.diff(fx, t))
            r.extra('Pendiente', r'\frac{dy}{dx} = \frac{y\'(t)}{x\'(t)} = ' + tex(pend))
        except Exception:  # noqa
            pass
        r.valor = [fx, fy]
        r.comparar = ('vector', [fx, fy], {})
        return r
    if low.startswith('3d'):
        f = leer(a[2:])
        r.titulo = 'Superficie'
        r.entrada_latex = 'z = ' + tex(f)
        r.grafico = G.grafico_3d(f, (x, y), r.entrada_latex)
        r.paso('Superficie', 'Se dibuja z = f(x, y) sobre una grilla; podés girarla con el dedo.', r.entrada_latex, 0)
        r.valor = f
        r.comparar = ('expr', f, {})
        return r
    if low.startswith('datos'):
        from .numerico import _puntos
        pts = _puntos(a[5:])
        r.titulo = 'Datos'
        r.entrada_latex = ', '.join(f'({tex(p)}, {tex(q)})' for p, q in pts)
        r.grafico = G.grafico_datos(pts)
        r.valor = [list(p) for p in pts]
        return r
    # funciones o implícitas
    partes = dividir(a)
    exprs, implicitas = [], []
    for p in partes:
        if '=' in p:
            izq = p.split('=')[0].strip()
            if izq in ('y', 'f(x)') :
                exprs.append(leer(p.split('=', 1)[1]))
                continue
            eq = leer_ecuacion(p)
            implicitas.append((eq.lhs - eq.rhs, tex(eq.lhs) + ' = ' + tex(eq.rhs)))
        else:
            exprs.append(leer(p))
    if implicitas and not exprs:
        g = G.grafico_implicita(implicitas[0][0], (x, y), implicitas[0][1])
        for i, (F, et) in enumerate(implicitas[1:], 1):
            c = G.grafico_implicita(F, (x, y), et)['curvas'][0]
            c['color'] = G.COLORES[i % len(G.COLORES)]
            g['curvas'].append(c)
        r.titulo = 'Curva implícita'
        r.entrada_latex = r',\ '.join(et for _, et in implicitas)
        r.grafico = g
        _conica(r, implicitas[0][0])
        r.valor = implicitas[0][0]
        return r
    vs = set()
    for e in exprs:
        vs |= set(variables_de(e))
    if len(vs) == 2 and len(exprs) == 1:
        return graficar('3d ' + a)
    v = list(vs)[0] if vs else x
    r.titulo = 'Gráfico de funciones'
    r.entrada_latex = r',\ '.join('y = ' + tex(e) for e in exprs)
    puntos = []
    for e in exprs[:1]:
        _analisis_funcion(r, e, v, puntos)
    r.grafico = G.grafico_2d(exprs, v, puntos)
    r.valor = exprs[0] if len(exprs) == 1 else exprs
    return r


def _analisis_funcion(r, f, v, puntos):
    """Dominio, raíces, ordenada al origen, asíntotas: lo que suele pedir un ejercicio de 'graficar'."""
    vr = sp.Symbol(str(v), real=True)
    fr = f.subs(v, vr)
    try:
        from sympy.calculus.util import continuous_domain
        dom = continuous_domain(fr, vr, sp.S.Reals)
        r.paso('Dominio', '', r'\operatorname{Dom} f = ' + tex(dom), 0)
    except Exception:  # noqa
        dom = sp.S.Reals
    try:
        raices = sp.solveset(fr, vr, sp.S.Reals)
        if isinstance(raices, sp.FiniteSet) and len(raices) < 10:
            r.paso('Raíces (ceros)', '', 'f(x) = 0 \\Rightarrow x = ' + (tex(sorted(raices, key=float)) if raices else r'\text{no tiene}'), 0)
            for c in raices:
                p = G.punto(c, 0, f'({fmt_num(c, 4)}, 0)')
                if p:
                    puntos.append(p)
    except Exception:  # noqa
        pass
    try:
        if 0 in dom:
            f0 = sp.simplify(fr.subs(vr, 0))
            if f0.is_finite:
                r.paso('Ordenada al origen', '', 'f(0) = ' + tex(f0), 0)
                p = G.punto(0, f0, f'(0, {fmt_num(f0, 4)})')
                if p:
                    puntos.append(p)
    except Exception:  # noqa
        pass
    try:
        asint = []
        num, den = sp.fraction(sp.together(fr))
        if den.has(vr):
            for c in sp.solveset(den, vr, sp.S.Reals):
                lim = sp.limit(fr, vr, c, '+')
                if lim.is_infinite:
                    asint.append(f'x = {tex(c)}\\ (\\text{{vertical}})')
        for lado in (sp.oo, -sp.oo):
            L = sp.limit(fr, vr, lado)
            if L.is_finite:
                txt = f'y = {tex(L)}\\ (\\text{{horizontal}})'
                if txt not in asint:
                    asint.append(txt)
            elif L.is_infinite:
                m = sp.limit(fr / vr, vr, lado)
                if m.is_finite and m != 0:
                    b = sp.limit(fr - m * vr, vr, lado)
                    if b.is_finite:
                        txt = f'y = {tex(m * v + b)}\\ (\\text{{oblicua}})'
                        if txt not in asint:
                            asint.append(txt)
        if asint:
            r.paso('Asíntotas', '', r',\quad '.join(asint), 0)
    except Exception:  # noqa
        pass


def _conica(r, F):
    try:
        P = sp.Poly(sp.expand(F), x, y)
        if P.total_degree() != 2:
            return
        A = P.coeff_monomial(x ** 2)
        B = P.coeff_monomial(x * y)
        C = P.coeff_monomial(y ** 2)
        disc = B ** 2 - 4 * A * C
        if disc < 0:
            tipo = 'circunferencia' if A == C and B == 0 else 'elipse'
        elif disc == 0:
            tipo = 'parábola'
        else:
            tipo = 'hipérbola'
        r.paso('Cónica', r'Según el discriminante \(B^2 - 4AC = ' + tex(disc) + r'\).', r'\text{' + tipo + '}', 0)
        if B == 0 and tipo in ('circunferencia', 'elipse', 'hipérbola') and A != 0 and C != 0:
            D_ = P.coeff_monomial(x)
            E_ = P.coeff_monomial(y)
            h, k_ = -D_ / (2 * A), -E_ / (2 * C)
            r.extra('Centro', f'({tex(h)}, {tex(k_)})')
            if tipo == 'circunferencia':
                F0 = P.coeff_monomial(1)
                rad2 = (A * h ** 2 + C * k_ ** 2 - F0) / A
                r.extra('Radio', tex(sp.sqrt(rad2)))
        r.valores['conica'] = tipo
    except Exception:  # noqa
        pass
