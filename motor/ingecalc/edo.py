"""Ecuaciones diferenciales ordinarias y transformadas (Laplace, Fourier)."""
import re

import sympy as sp

from . import graficos as G
from .nucleo import (ErrorEntrada, Resultado, dividir, leer, simplificar_bonito, tex, variables_de, x, t, s, n,
                     normalizar, fmt_num)

yF = sp.Function('y')


def _var_indep(texto):
    sin_func = re.sub(r'[a-z]{2,}', ' ', texto.lower())
    return t if re.search(r'(?<![a-z])t(?![a-z])', sin_func) and not re.search(r'(?<![a-z])x(?![a-z])', sin_func) else x


def _separar_y(m):
    """En una palabra como 'xy' o '2xy', la y es la función: x*(y)."""
    from .nucleo import FUNCIONES, CONSTANTES, _PALABRAS_BASE
    palabra = m.group(0)
    conocidas = set(FUNCIONES) | set(CONSTANTES) | {a for a, _ in _PALABRAS_BASE} | {
        'theta', 'alpha', 'beta', 'lambda', 'omega', 'mu', 'sigma', 'phi', 'tau', 'rho', 'sqrt'}
    if palabra.lower() in conocidas or 'y' not in palabra.lower():
        return palabra
    return '*'.join(' (_y0) ' if ch.lower() == 'y' else ch for ch in palabra)


def leer_edo(texto, v=None):
    """'y''+3y'+2y=e^x' -> (Eq(...), v). Acepta y', y'', dy/dx, d2y/dx2."""
    s_ = normalizar(texto)
    v = v or _var_indep(s_)
    vn = str(v)
    s_ = re.sub(r'd\^?3y/d' + vn + r'\^?3', "y'''", s_)
    s_ = re.sub(r'd\^?2y/d' + vn + r'\^?2', "y''", s_)
    s_ = s_.replace('dy/d' + vn, "y'")
    s_ = re.sub(r"y\s*(''''|'''|''|')\s*\(\s*" + vn + r"\s*\)", r"y\1", s_)
    s_ = re.sub(r"(?<![a-z])y\s*\(\s*" + vn + r"\s*\)", 'y', s_)
    s_ = s_.replace("y''''", ' (_d4) ').replace("y'''", ' (_d3) ').replace("y''", ' (_d2) ').replace("y'", ' (_d1) ')
    s_ = re.sub(r'(?<![A-Za-z_])[A-Za-z]+', _separar_y, s_)
    Y = yF(v)
    extra = {'_y0': Y, '_d1': Y.diff(v), '_d2': Y.diff(v, 2), '_d3': Y.diff(v, 3), '_d4': Y.diff(v, 4)}
    if '=' in s_:
        a, b = s_.split('=', 1)
        eq = sp.Eq(leer(a, extra), leer(b, extra))
    else:
        eq = sp.Eq(leer(s_, extra), 0)
    if not eq.has(sp.Derivative):
        raise ErrorEntrada("No encontré derivadas en la ecuación (escribí y', y'' o dy/dx).")
    return eq, v


def leer_condiciones(conds, v):
    ics = {}
    for c in conds:
        m = re.match(r"\s*y\s*('*)\s*\(\s*([^)]+)\)\s*=\s*(.+)$", normalizar(c))
        if not m:
            raise ErrorEntrada(f'No entendí la condición «{c}». Usá por ejemplo y(0)=1 o y\'(0)=2.')
        orden = len(m.group(1))
        a, b = leer(m.group(2)), leer(m.group(3))
        clave = yF(a) if orden == 0 else yF(v).diff(v, orden).subs(v, a)
        ics[clave] = b
    return ics


def edo(arg):
    partes = dividir(arg, ';')
    eq, v = leer_edo(partes[0])
    ics = leer_condiciones(partes[1:], v) if len(partes) > 1 else {}
    Y = yF(v)
    orden = sp.ode_order(eq, Y)
    r = Resultado('Ecuaciones diferenciales', f'EDO de orden {orden}' + (' (problema de valor inicial)' if ics else ''))
    r.entrada_latex = tex(eq.lhs).replace(r'\frac{d}{d ' + str(v) + r'} y{\left(' + str(v) + r' \right)}', "y'") + ' = ' + tex(eq.rhs)
    r.entrada_latex = _bonito(tex(eq.lhs), v) + ' = ' + _bonito(tex(eq.rhs), v)
    if ics:
        r.entrada_latex += r',\quad ' + ', '.join(_bonito(tex(k_), v) + ' = ' + tex(b) for k_, b in ics.items())
    try:
        clases = sp.classify_ode(eq, Y)
    except Exception:  # noqa
        clases = ()
    _explicar_metodo(r, eq, Y, v, orden, clases)
    try:
        sol = sp.dsolve(eq, Y, ics=ics or None)
    except Exception as exc:  # noqa
        raise ErrorEntrada('No pude resolver esta EDO en forma cerrada.') from exc
    if isinstance(sol, list):
        sol = sol[0] if len(sol) == 1 else sol
    if isinstance(sol, list):
        rhs = [simplificar_bonito(sl.rhs) for sl in sol]
        r.valor = rhs
        r.resultado_latex = r'\\'.join('y = ' + tex(e) for e in rhs)
        r.comparar = ('edo', rhs, {'eq': eq, 'ics': ics, 'v': v, 'orden': orden})
        return r
    rhs = sol.rhs
    try:
        rhs = simplificar_bonito(rhs) if sp.count_ops(rhs) < 80 else rhs
    except Exception:  # noqa
        pass
    if not ics:
        r.paso('Solución general', f'Tiene {orden} constante(s) arbitraria(s).', 'y = ' + tex(rhs), 0)
    else:
        gen = sp.dsolve(eq, Y)
        if isinstance(gen, list):
            gen = gen[0]
        r.paso('Solución general', '', 'y = ' + tex(gen.rhs), 0)
        consts = sorted(gen.rhs.free_symbols - {v} - eq.free_symbols, key=str)
        ecs = []
        for k_, b in ics.items():
            if isinstance(k_, sp.Subs) or k_.has(sp.Derivative):
                pto = list(k_.point)[0] if isinstance(k_, sp.Subs) else k_.args[0]
                d_ord = k_.expr.derivative_count if hasattr(k_, 'expr') else 1
                ecs.append(sp.Eq(sp.diff(gen.rhs, v, d_ord).subs(v, pto), b))
            else:
                ecs.append(sp.Eq(gen.rhs.subs(v, k_.args[0]), b))
        try:
            cs = sp.solve(ecs, consts, dict=True)
            r.paso('Aplicar condiciones iniciales', 'Se reemplazan las condiciones y se despejan las constantes.',
                   r'\begin{cases}' + r'\\'.join(tex(e.lhs) + ' = ' + tex(e.rhs) for e in ecs) + r'\end{cases}'
                   + (r'\Rightarrow ' + ', '.join(f'{tex(c)} = {tex(val)}' for c, val in cs[0].items()) if cs else ''), 0)
        except Exception:  # noqa
            pass
        r.paso('Solución particular', '', 'y = ' + tex(rhs), 0)
    r.valor = rhs
    r.resultado_latex = 'y(' + tex(v) + ') = ' + tex(rhs)
    # comprobación
    try:
        ok = sp.checkodesol(eq, sp.Eq(Y, rhs))
        if ok[0]:
            r.paso('Verificación', 'Al reemplazar y y sus derivadas se cumple la ecuación.', r'\checkmark', 0)
    except Exception:  # noqa
        pass
    r.comparar = ('edo', rhs, {'eq': eq, 'ics': ics, 'v': v, 'orden': orden})
    if not rhs.free_symbols - {v}:
        try:
            r.grafico = G.grafico_2d([rhs], v, xmin=-1 if ics else -5, xmax=6 if ics else 5)
        except Exception:  # noqa
            pass
    elif len(rhs.free_symbols - {v}) <= 2:
        # familia de soluciones con constantes 1, 0, -1
        try:
            consts = sorted(rhs.free_symbols - {v}, key=str)
            fam = []
            for val in (-2, -1, 0, 1, 2):
                fam.append(rhs.subs({c: val if i == 0 else 1 for i, c in enumerate(consts)}))
            r.grafico = G.grafico_2d(fam, v, etiquetas=[f'C1={val}' for val in (-2, -1, 0, 1, 2)], xmin=-4, xmax=4)
        except Exception:  # noqa
            pass
    return r


def _bonito(latex, v):
    vn = str(v)
    latex = latex.replace(r'\frac{d^{3}}{d ' + vn + r'^{3}} y{\left(' + vn + r' \right)}', "y'''")
    latex = latex.replace(r'\frac{d^{2}}{d ' + vn + r'^{2}} y{\left(' + vn + r' \right)}', "y''")
    latex = latex.replace(r'\frac{d}{d ' + vn + r'} y{\left(' + vn + r' \right)}', "y'")
    latex = latex.replace(r'y{\left(' + vn + r' \right)}', 'y')
    return latex


def _explicar_metodo(r, eq, Y, v, orden, clases):
    expr = eq.lhs - eq.rhs
    if orden == 1 and 'separable' in clases:
        try:
            yp = sp.solve(expr, Y.diff(v))[0]
            ys = sp.Symbol('y')
            g = yp.subs(Y, ys)
            fx, gy = sp.separatevars(g, symbols=[v, ys], dict=True)[v], sp.separatevars(g, symbols=[v, ys], dict=True)[ys]
            coef = sp.separatevars(g, symbols=[v, ys], dict=True)['coeff']
            r.paso('Variables separables', r'La ecuación es \(y\' = f(x)\,g(y)\): se separan y se integran ambos lados.',
                   r'\int \frac{dy}{' + tex(gy) + r'} = \int ' + tex(coef * fx) + r'\,d' + tex(v), 0)
            iz = sp.integrate(1 / gy, ys)
            de = sp.integrate(coef * fx, v)
            r.paso('Integrar', '', tex(iz) + ' = ' + tex(de) + ' + C', 1)
            return
        except Exception:  # noqa
            pass
    if orden == 1 and '1st_linear' in clases:
        try:
            yp = sp.solve(expr, Y.diff(v))[0]
            ys = sp.Symbol('y')
            lin = sp.expand(yp.subs(Y, ys))
            P = -lin.coeff(ys, 1)
            Q = lin.coeff(ys, 0)
            mu = sp.simplify(sp.exp(sp.integrate(P, v)))
            r.paso('EDO lineal de primer orden', r'Forma \(y\' + P(x)\,y = Q(x)\).', 'P = ' + tex(P) + r',\quad Q = ' + tex(Q), 0)
            r.paso('Factor integrante', r'\(\mu = e^{\int P\,dx}\)', r'\mu = ' + tex(mu), 1)
            r.paso('Integrar', r'\((\mu\,y)\' = \mu\,Q \Rightarrow y = \frac{1}{\mu}\left(\int \mu Q\,dx + C\right)\)',
                   r'\int ' + tex(sp.simplify(mu * Q)) + r'\,d' + tex(v) + ' = ' + tex(sp.simplify(sp.integrate(sp.simplify(mu * Q), v))), 1)
            return
        except Exception:  # noqa
            pass
    if orden == 1 and 'Bernoulli' in clases:
        r.paso('Ecuación de Bernoulli', r'Forma \(y\' + P y = Q y^n\): con \(u = y^{1-n}\) se vuelve lineal.', '', 0)
        return
    if orden == 1 and '1st_exact' in clases:
        r.paso('Ecuación exacta', r'\(M\,dx + N\,dy = 0\) con \(M_y = N_x\): existe \(F(x,y)\) con \(F_x = M,\ F_y = N\); la solución es \(F = C\).', '', 0)
        return
    if orden == 1 and any(c.startswith('1st_homogeneous') for c in clases):
        r.paso('Ecuación homogénea', r'Se sustituye \(y = u\,x\) y queda separable.', '', 0)
        return
    lineal_cc = [c for c in clases if c.startswith('nth_linear_constant_coeff')]
    if lineal_cc:
        mvar = sp.Symbol('m')
        # coeficientes
        homog = expr
        coefs = []
        for k_ in range(orden, -1, -1):
            d = Y.diff(v, k_) if k_ > 0 else Y
            coefs.append(sp.expand(homog).coeff(d))
        car = sum(c * mvar ** (orden - i) for i, c in enumerate(coefs))
        raices = sp.roots(sp.Poly(car, mvar))
        r.paso('Ecuación característica', r'Se propone \(y = e^{m' + tex(v) + r'}\).', tex(car) + r' = 0 \Rightarrow m = '
               + ', '.join(tex(rr) + (f'\\ (\\times {mm})' if mm > 1 else '') for rr, mm in raices.items()), 0)
        casos = []
        for rr, mm in raices.items():
            if rr.is_real:
                casos.append('raíz real doble → e^{mx}, x e^{mx}' if mm > 1 else 'raíz real → e^{mx}')
            else:
                casos.append('raíces complejas α±βi → e^{αx}(C₁cos βx + C₂sin βx)')
        r.paso('Solución homogénea', '; '.join(dict.fromkeys(casos)) + '.', '', 1)
        resto = sp.expand(expr).subs({Y.diff(v, k_): 0 for k_ in range(1, orden + 1)}).subs(Y, 0)
        if resto != 0:
            if any('undetermined' in c for c in lineal_cc):
                r.paso('Solución particular (coeficientes indeterminados)', r'Se propone \(y_p\) con la forma del término independiente '
                       r'\(' + tex(-resto) + r'\) (multiplicando por \(x\) si coincide con la homogénea) y se calculan los coeficientes.', '', 0)
            else:
                r.paso('Solución particular (variación de parámetros)', r'\(y_p = u_1 y_1 + u_2 y_2\) con el wronskiano.', '', 0)
            r.paso('Solución general', r'\(y = y_h + y_p\)', '', 0)
        return
    if any('euler' in c for c in clases):
        r.paso('Ecuación de Cauchy-Euler', r'Se propone \(y = x^m\) y se obtiene una ecuación indicial en m.', '', 0)
        return
    if clases:
        r.paso('Método', 'Clasificación: ' + ', '.join(c.replace('_', ' ') for c in clases[:2]) + '.', '', 0)


# ----------------------------------------------------------------------------
# Laplace
# ----------------------------------------------------------------------------
def _regla_laplace(tm):
    if tm.has(sp.Heaviside) and any(a.args[0] != t for a in tm.atoms(sp.Heaviside)):
        return r'Segundo teorema de traslación: \(\mathcal{L}\{f(t-a)u(t-a)\} = e^{-as}F(s)\)'
    if tm.has(sp.DiracDelta):
        return r'\(\mathcal{L}\{\delta(t-a)\} = e^{-as}\)'
    exps = [a for a in tm.atoms(sp.exp) if a.has(t)]
    resto = tm
    for e_ in exps:
        resto = resto / e_
    if exps and resto.has(t):
        return r'Primer teorema de traslación: \(\mathcal{L}\{e^{at}f(t)\} = F(s-a)\)'
    facs = sp.Mul.make_args(tm)
    tiene_tn = any(f == t or (f.is_Pow and f.base == t) for f in facs)
    tiene_trig = any(f.has(sp.sin, sp.cos, sp.sinh, sp.cosh) for f in facs)
    if tiene_tn and tiene_trig:
        return r'Derivada de la transformada: \(\mathcal{L}\{t^n f(t)\} = (-1)^n F^{(n)}(s)\)'
    return 'Tabla de transformadas'


def laplace(arg):
    f = leer(arg)
    if f.has(x) and not f.has(t):
        f = f.subs(x, t)
    r = Resultado('Transformadas', 'Transformada de Laplace')
    r.entrada_latex = r'\mathcal{L}\left\{' + tex(f) + r'\right\}'
    terminos = sp.Add.make_args(sp.expand(f))
    if len(terminos) > 1:
        r.paso('Linealidad', r'\(\mathcal{L}\{a f + b g\} = a F + b G\)', ' + '.join(r'\mathcal{L}\left\{' + tex(tm) + r'\right\}' for tm in terminos), 0)
    total = 0
    for tm in terminos:
        Ft = sp.laplace_transform(tm, t, s, noconds=True)
        Ft = simplificar_bonito(Ft)
        r.paso(_regla_laplace(tm).split(':')[0] if ':' in _regla_laplace(tm) else 'Tabla', _regla_laplace(tm),
               r'\mathcal{L}\left\{' + tex(tm) + r'\right\} = ' + tex(Ft), 1 if len(terminos) > 1 else 0)
        total += Ft
    res = simplificar_bonito(sp.together(total)) if sp.count_ops(total) < 60 else total
    res2 = simplificar_bonito(total)
    res = min([res, res2, total], key=lambda e: (sp.count_ops(e), len(str(e))))
    r.valor = res
    r.resultado_latex = 'F(s) = ' + tex(res)
    r.comparar = ('expr', res, {})
    return r


def laplace_inversa(arg):
    F = leer(arg)
    if F.has(x) and not F.has(s):
        F = F.subs(x, s)
    r = Resultado('Transformadas', 'Transformada inversa de Laplace')
    r.entrada_latex = r'\mathcal{L}^{-1}\left\{' + tex(F) + r'\right\}'
    # agrupar por factores e^{-as}
    grupos = {}
    for tm in sp.Add.make_args(sp.expand(F)):
        exps = [a for a in sp.Mul.make_args(tm) if isinstance(a, sp.exp) or (a.is_Pow and a.base == sp.E)]
        clave = sp.Mul(*exps)
        grupos[clave] = grupos.get(clave, 0) + tm / clave
    total = 0
    for clave, G_ in grupos.items():
        G_ = sp.together(G_)
        retardo = 0
        if clave != 1:
            arg_exp = sp.log(clave).expand(force=True)
            retardo = sp.simplify(-arg_exp / s)
        ap = sp.apart(G_, s) if G_.is_rational_function(s) else G_
        if ap != G_:
            r.paso('Fracciones parciales', 'Se descompone para llevar cada término a la tabla.',
                   tex(G_) + ' = ' + tex(ap), 0)
        parcial = 0
        for tm in sp.Add.make_args(ap):
            num, den = sp.fraction(sp.together(tm))
            nota = 'Tabla de transformadas'
            if den.is_polynomial(s) and sp.degree(den, s) == 2 and sp.discriminant(den, s) < 0:
                a2, b2, c2 = sp.Poly(den, s).all_coeffs()
                h = -b2 / (2 * a2)
                kk = sp.sqrt(c2 / a2 - h ** 2)
                nota = r'Completar cuadrados: \(' + tex(den) + ' = ' + tex(a2 * ((s - h) ** 2 + kk ** 2)) + r'\); se usan seno/coseno trasladados.'
            ft = sp.inverse_laplace_transform(tm, s, t)
            ft = ft.subs(sp.Heaviside(t), 1)
            ft = simplificar_bonito(ft)
            r.paso('Término', nota, r'\mathcal{L}^{-1}\left\{' + tex(tm) + r'\right\} = ' + tex(ft), 1)
            parcial += ft
        if retardo != 0:
            desplaz = parcial.subs(t, t - retardo) * sp.Heaviside(t - retardo)
            r.paso('Segundo teorema de traslación', r'El factor \(e^{-as}\) desplaza la función y multiplica por el escalón \(u(t-a)\).',
                   r'\mathcal{L}^{-1}\left\{' + tex(clave * G_) + r'\right\} = ' + tex(desplaz), 0)
            parcial = desplaz
        total += parcial
    # forma de tabla: suma de términos; productos/potencias de seno y coseno pasan a sumas (sen²t -> (1-cos2t)/2)
    if not total.has(sp.Heaviside):
        from sympy.simplify.fu import TR8
        res = sp.expand(total)
        if any(isinstance(b.base, (sp.sin, sp.cos)) for b in res.atoms(sp.Pow)) or any(
                len([f for f in sp.Mul.make_args(tm) if isinstance(f, (sp.sin, sp.cos))]) > 1 for tm in sp.Add.make_args(res)):
            res = sp.expand(TR8(res))
    else:
        res = total
    r.valor = res
    r.resultado_latex = 'f(t) = ' + tex(res)
    r.comparar = ('expr', res, {})
    try:
        r.grafico = G.grafico_2d([res], t, xmin=-0.5, xmax=10)
    except Exception:  # noqa
        pass
    return r


def fourier(arg):
    partes = dividir(arg)
    f = leer(partes[0])
    v = leer(partes[1]) if len(partes) > 1 else x
    a = leer(partes[2]) if len(partes) > 2 else -sp.pi
    b = leer(partes[3]) if len(partes) > 3 else sp.pi
    N = int(leer(partes[4])) if len(partes) > 4 else 5
    T = b - a
    L = T / 2
    w = sp.pi / L
    r = Resultado('Transformadas', 'Serie de Fourier')
    r.entrada_latex = 'f(' + tex(v) + ') = ' + tex(f) + r',\quad ' + tex(v) + r'\in[' + tex(a) + ', ' + tex(b) + ']'
    r.paso('Fórmulas', r'Período \(T = ' + tex(T) + r'\), \(L = T/2 = ' + tex(L) + r'\).',
           r'a_0 = \frac{1}{L}\int f,\quad a_n = \frac{1}{L}\int f\cos\frac{n\pi x}{L},\quad b_n = \frac{1}{L}\int f\sin\frac{n\pi x}{L}', 0)
    nn = sp.Symbol('n', integer=True, positive=True)
    a0 = sp.simplify(sp.integrate(f, (v, a, b)) / L)
    an = sp.simplify(sp.integrate(f * sp.cos(nn * w * v), (v, a, b)) / L)
    bn = sp.simplify(sp.integrate(f * sp.sin(nn * w * v), (v, a, b)) / L)
    par = sp.simplify(f.subs(v, -v) - f) == 0 if a == -b else False
    impar = sp.simplify(f.subs(v, -v) + f) == 0 if a == -b else False
    if par:
        r.paso('Simetría', 'La función es par: todos los bₙ son 0.', '', 0)
    if impar:
        r.paso('Simetría', 'La función es impar: a₀ y todos los aₙ son 0.', '', 0)
    r.paso('Coeficientes', 'Se usa sin(nπ) = 0 y cos(nπ) = (−1)ⁿ.',
           'a_0 = ' + tex(a0) + r',\quad a_n = ' + tex(an) + r',\quad b_n = ' + tex(bn), 0)
    suma = a0 / 2
    for k_ in range(1, N + 1):
        ak = sp.simplify(sp.integrate(f * sp.cos(k_ * w * v), (v, a, b)) / L)
        bk = sp.simplify(sp.integrate(f * sp.sin(k_ * w * v), (v, a, b)) / L)
        suma += ak * sp.cos(k_ * w * v) + bk * sp.sin(k_ * w * v)
    r.paso(f'Suma parcial con {N} armónicos', '', 'S_{' + str(N) + '}(' + tex(v) + ') = ' + tex(suma), 0)
    r.valor = suma
    r.valores['a0'] = a0
    r.valores['an'] = an
    r.valores['bn'] = bn
    r.resultado_latex = 'f(' + tex(v) + r') \sim \frac{a_0}{2} + \sum_{n=1}^{\infty}\left(a_n\cos\frac{n\pi ' + tex(v) + '}{L} + b_n\sin\frac{n\pi '\
        + tex(v) + r'}{L}\right),\ ' + 'a_0 = ' + tex(a0) + r',\ a_n = ' + tex(an) + r',\ b_n = ' + tex(bn)
    r.extra('Suma parcial', 'S_{' + str(N) + '} = ' + tex(suma))
    r.comparar = ('expr', suma, {})
    try:
        r.grafico = G.grafico_2d([f, suma], v, etiquetas=['f', f'S_{N}'], xmin=float(a) - 0.2 * float(T), xmax=float(b) + 0.2 * float(T))
    except Exception:  # noqa
        pass
    return r
