"""Análisis Matemático I: derivadas, integrales, límites, Taylor, extremos, áreas."""
import re

import sympy as sp

from . import graficos as G
from .nucleo import (ErrorEntrada, Resultado, dividir, leer, leer_ecuacion, simplificar_bonito, tex, variables_de,
                     variable_principal, x, y, decimal, fmt_num)
from .pasos_derivada import derivar_con_pasos
from .pasos_integral import integrar_con_pasos


def con_abs_log(e, v):
    """ln(u) -> ln|u| para mostrar primitivas como en la cursada."""
    vr = sp.Symbol(str(v), real=True)

    def cambiar(a):
        u = a.args[0]
        try:
            pos = u.subs(v, vr).is_positive
        except Exception:  # noqa
            pos = None
        return sp.log(sp.Abs(u)) if pos is not True and not isinstance(u, sp.Abs) else a
    try:
        return e.replace(lambda a: isinstance(a, sp.log) and len(a.args) == 1, cambiar)
    except Exception:  # noqa
        return e


def _grafico_funcion(r, exprs, v, puntos=(), etiquetas=None, **kw):
    try:
        r.grafico = G.grafico_2d(exprs, v, puntos, etiquetas=etiquetas, **kw)
    except Exception:  # noqa
        pass


# ----------------------------------------------------------------------------
# Derivadas
# ----------------------------------------------------------------------------
def derivar(arg):
    partes = dividir(arg)
    if not partes:
        raise ErrorEntrada('Escribí la función a derivar, por ejemplo: derivar x^2*sen(x)')
    f = leer(partes[0])
    v = leer(partes[1]) if len(partes) > 1 and not partes[1].strip().isdigit() else variable_principal(f)
    orden = 1
    for p in partes[1:]:
        if p.strip().isdigit():
            orden = int(p)
    if len(partes) > 2 and partes[2].strip().isdigit():
        orden = int(partes[2])
    multiples = len(variables_de(f)) > 1
    r = Resultado('Cálculo', 'Derivada parcial' if multiples else ('Derivada' if orden == 1 else f'Derivada de orden {orden}'))
    simb = r'\frac{\partial' if multiples else r'\frac{d'
    sup = '' if orden == 1 else f'^{{{orden}}}'
    r.entrada_latex = simb + sup + '}{' + (r'\partial ' if multiples else 'd') + tex(v) + sup + r'}\left[' + tex(f) + r'\right]'
    actual = f
    for o in range(1, orden + 1):
        crudo, pasos = derivar_con_pasos(actual, v)
        if orden > 1:
            r.paso(f'Derivada {o}', f'Derivamos la expresión anterior (orden {o}).', '', 0)
        for p in pasos:
            p = dict(p)
            p['nivel'] += 1 if orden > 1 else 0
            r.pasos.append(p)
        bonito = simplificar_bonito(crudo)
        if bonito != crudo:
            r.paso('Simplificar', 'Se agrupan y simplifican los términos.', tex(crudo) + ' = ' + tex(bonito), 1 if orden > 1 else 0)
        actual = bonito
    r.valor = actual
    r.resultado_latex = (r.entrada_latex + ' = ' + tex(actual))
    r.comparar = ('expr', actual, {})
    if not multiples:
        _grafico_funcion(r, [f, actual], v, etiquetas=['f', "f'" if orden == 1 else f'f^{{({orden})}}'])
    return r


def derivar_implicita(arg):
    eq = leer_ecuacion(arg)
    if not isinstance(eq, sp.Equality):
        eq = sp.Eq(eq, 0)
    F = eq.lhs - eq.rhs
    yf = sp.Function('y')(x)
    r = Resultado('Cálculo', 'Derivación implícita')
    r.entrada_latex = tex(eq.lhs) + ' = ' + tex(eq.rhs)
    r.paso('Derivar ambos miembros respecto de x', r'Consideramos \(y = y(x)\); por la regla de la cadena, '
           r'cada término con \(y\) se multiplica por \(y\'\).', '', 0)
    Fx = sp.diff(F.subs(y, yf), x)
    yp = sp.Symbol("y'")
    Fx_s = Fx.subs(sp.Derivative(yf, x), yp).subs(yf, y)
    r.paso('Resultado de derivar', '', tex(Fx_s) + ' = 0', 0)
    sol = sp.solve(Fx_s, yp)
    if not sol:
        raise ErrorEntrada('No se pudo despejar dy/dx.')
    dydx = simplificar_bonito(sol[0])
    alt = -sp.diff(F, x) / sp.diff(F, y)
    r.paso('Despejar y\'', r'Se agrupan los términos con \(y\'\) y se despeja. (Equivale a \(y\' = -F_x/F_y\).)',
           r"y' = " + tex(dydx), 0)
    r.valor = dydx
    r.resultado_latex = r'\frac{dy}{dx} = ' + tex(dydx)
    r.comparar = ('expr', sp.simplify(alt), {})
    try:
        r.grafico = G.grafico_implicita(F, (x, y), tex(eq.lhs) + '=' + tex(eq.rhs))
    except Exception:  # noqa
        pass
    return r


def recta_tangente(arg):
    partes = dividir(arg)
    if len(partes) < 2:
        raise ErrorEntrada('Uso: recta tangente f, a   (por ejemplo: recta tangente x^2, 1)')
    f = leer(partes[0])
    v = variable_principal(f)
    a = leer(partes[1])
    r = Resultado('Cálculo', 'Recta tangente')
    r.entrada_latex = r'f(' + tex(v) + ') = ' + tex(f) + r',\quad ' + tex(v) + '_0 = ' + tex(a)
    fa = sp.simplify(f.subs(v, a))
    df, pasos = derivar_con_pasos(f, v)
    df = simplificar_bonito(df)
    r.paso('Punto de tangencia', '', f'f({tex(a)}) = ' + tex(fa), 0)
    r.paso('Derivada', 'La pendiente es la derivada evaluada en el punto.', "f'(" + tex(v) + ') = ' + tex(df), 0)
    for p in pasos:
        p = dict(p); p['nivel'] += 1
        r.pasos.append(p)
    m = sp.simplify(df.subs(v, a))
    r.paso('Pendiente', '', f"m = f'({tex(a)}) = " + tex(m), 0)
    recta = sp.expand(fa + m * (v - a))
    r.paso('Ecuación punto-pendiente', r'\(y - f(x_0) = m\,(x - x_0)\)',
           'y - ' + tex(fa) + ' = ' + tex(m) + r'\left(' + tex(v - a) + r'\right) \Rightarrow y = ' + tex(recta), 0)
    if m != 0:
        normal = sp.expand(fa - (v - a) / m)
        r.extra('Recta normal', 'y = ' + tex(normal))
    r.valor = recta
    r.resultado_latex = 'y = ' + tex(recta)
    r.comparar = ('expr', recta, {})
    _grafico_funcion(r, [f, recta], v, [G.punto(a, fa, f'({fmt_num(a, 4)}, {fmt_num(fa, 4)})')], ['f', 'tangente'])
    return r


# ----------------------------------------------------------------------------
# Integrales
# ----------------------------------------------------------------------------
def _separar_dx(texto):
    """'x^2 dx' -> ('x^2', 'x')"""
    m = re.search(r'\s*\*?\s*d([a-z])\s*$', texto)
    if m and not re.search(r'[a-z]d[a-z]$', texto.replace(' ', '')[-3:] if False else ''):
        antes = texto[:m.start()]
        if antes.strip():
            return antes, m.group(1)
    return texto, None


def integrar(arg):
    arg = arg.strip()
    # integrar f de a a b
    m = re.match(r'^(.*?)\s+(?:de|desde)\s+(\S+)\s+(?:a|hasta)\s+(\S+)\s*$', arg)
    if m:
        cuerpo, var_txt = _separar_dx(m.group(1))
        partes = [cuerpo] + ([var_txt] if var_txt else []) + [m.group(2), m.group(3)]
        if not var_txt:
            f = leer(cuerpo)
            partes = [cuerpo, str(variable_principal(f)), m.group(2), m.group(3)]
    else:
        partes = dividir(arg)
        cuerpo, var_txt = _separar_dx(partes[0])
        partes[0] = cuerpo
        if var_txt and (len(partes) == 1 or len(partes) == 3):
            partes.insert(1, var_txt)
    f = leer(partes[0])
    if len(partes) == 3:  # f, a, b
        partes.insert(1, str(variable_principal(f)))
    v = leer(partes[1]) if len(partes) > 1 else variable_principal(f)
    if len(partes) >= 4:
        return integral_definida(f, v, leer(partes[2]), leer(partes[3]))
    return integral_indefinida(f, v)


def integral_indefinida(f, v):
    r = Resultado('Cálculo', 'Integral indefinida')
    r.entrada_latex = r'\int ' + tex(f) + r'\,d' + tex(v)
    prim, pasos, metodo = integrar_con_pasos(f, v)
    r.pasos.extend(pasos)
    if metodo == 'sin_primitiva':
        r.avisos.append('Esta integral no tiene primitiva expresable con funciones elementales ni especiales conocidas.')
    bonito = simplificar_bonito(prim)
    if sp.simplify(sp.diff(bonito, v) - f) != 0:
        bonito = prim
    mostrar = con_abs_log(bonito, v)
    if mostrar != prim:
        r.paso('Resultado', 'Se simplifica y se escribe el valor absoluto dentro del logaritmo.', tex(mostrar) + ' + C', 0)
    r.paso('Verificación', 'Derivando el resultado se obtiene el integrando.',
           r'\frac{d}{d' + tex(v) + r'}\left[' + tex(mostrar) + r'\right] = ' + tex(simplificar_bonito(sp.diff(bonito, v))), 0)
    r.valor = bonito
    r.resultado_latex = r.entrada_latex + ' = ' + tex(mostrar) + ' + C'
    r.comparar = ('antiderivada', bonito, {'var': v, 'integrando': f})
    if len(variables_de(f)) <= 1:
        _grafico_funcion(r, [f, bonito], v, etiquetas=['f', 'F (C = 0)'])
    return r


def integral_definida(f, v, a, b):
    r = Resultado('Cálculo', 'Integral definida')
    r.entrada_latex = r'\int_{' + tex(a) + '}^{' + tex(b) + '} ' + tex(f) + r'\,d' + tex(v)
    impropia = a.is_infinite or b.is_infinite
    valor = sp.integrate(f, (v, a, b))
    if isinstance(valor, sp.Integral) or valor.has(sp.Integral):
        valor_num = sp.Integral(f, (v, a, b)).evalf(15)
        valor = valor_num
        r.avisos.append('No hay primitiva elemental: se calculó numéricamente.')
        r.paso('Integración numérica', 'Se aproxima el valor con cuadratura numérica.', r.entrada_latex + r' \approx ' + tex(sp.N(valor, 10)), 0)
    else:
        prim, pasos, metodo = integrar_con_pasos(f, v)
        if metodo != 'sin_primitiva':
            r.paso('1. Buscar una primitiva', '', r'\int ' + tex(f) + r'\,d' + tex(v) + ' = ' + tex(con_abs_log(prim, v)), 0)
            for p in pasos:
                p = dict(p); p['nivel'] += 1
                r.pasos.append(p)
            if impropia:
                r.paso('2. Integral impropia', 'Un límite de integración es infinito: se calcula como límite.',
                       r.entrada_latex + r' = \lim_{R\to\infty}\Big[F\Big] = ' + tex(valor), 0)
            else:
                Fa, Fb = sp.simplify(prim.subs(v, a)), sp.simplify(prim.subs(v, b))
                barrow = sp.simplify(Fb - Fa)
                r.paso('2. Regla de Barrow', r'\(\int_a^b f = F(b) - F(a)\)',
                       'F(' + tex(b) + ') - F(' + tex(a) + ') = ' + tex(Fb) + r' - \left(' + tex(Fa) + r'\right) = ' + tex(barrow), 0)
                if sp.simplify(barrow - valor) != 0 and not (barrow.is_number and valor.is_number and
                                                            abs(complex(sp.N(barrow - valor))) < 1e-9):
                    r.avisos.append('El integrando tiene una discontinuidad en el intervalo: es una integral impropia, '
                                    'Barrow directo no vale.')
                    r.paso('Discontinuidad en el intervalo', 'Se separa en el punto problemático y se calcula con límites.',
                           r.entrada_latex + ' = ' + tex(valor), 0)
    valor = sp.simplify(valor) if not valor.is_Float else valor
    r.valor = valor
    dec = decimal(valor)
    r.valores['decimal'] = fmt_num(dec, 12) if dec is not None else None
    r.resultado_latex = r.entrada_latex + ' = ' + tex(valor) + ((r' \approx ' + fmt_num(dec, 8)) if dec is not None and not valor.is_Rational and valor.is_finite else '')
    if valor.is_infinite:
        r.extra('Conclusión', '', 'La integral impropia diverge.')
    r.comparar = ('numero_o_expr', valor, {})
    if len(variables_de(f)) <= 1 and not impropia:
        try:
            g = G.grafico_2d([f], v, [], xmin=float(a) - 0.3 * float(b - a) - 0.5, xmax=float(b) + 0.3 * float(b - a) + 0.5)
            g['sombra'] = G.sombra_entre(f, sp.Integer(0), a, b, v)
            r.grafico = g
        except Exception:  # noqa
            pass
    return r


def area(arg):
    partes = dividir(arg)
    if len(partes) == 3:  # f, a, b  (área entre f y el eje x)
        partes = [partes[0], '0', partes[1], partes[2]]
    if len(partes) < 4:
        raise ErrorEntrada('Uso: area f, g, a, b   (área entre f y g de a a b)')
    f, g, a, b = (leer(p) for p in partes[:4])
    v = variable_principal(f - g)
    r = Resultado('Cálculo', 'Área entre curvas')
    r.entrada_latex = r'A = \int_{' + tex(a) + '}^{' + tex(b) + r'} \left|' + tex(f) + ' - ' + tex(g) + r'\right|\,d' + tex(v)
    h = sp.simplify(f - g)
    cortes = []
    try:
        for c in sp.solveset(h, v, sp.Interval.open(a, b)):
            if c.is_real:
                cortes.append(c)
    except Exception:  # noqa
        pass
    cortes = sorted(set(cortes), key=lambda c: float(c))
    bordes = [a] + cortes + [b]
    if cortes:
        r.paso('Intersecciones', 'Donde las curvas se cortan cambia cuál está arriba; se separa la integral.',
               tex(f) + ' = ' + tex(g) + r'\Rightarrow ' + tex(v) + ' = ' + tex(cortes), 0)
    total = 0
    for i in range(len(bordes) - 1):
        p, q = bordes[i], bordes[i + 1]
        medio = (p + q) / 2
        arriba = f if float(h.subs(v, medio)) >= 0 else g
        abajo = g if arriba is f else f
        parcial = sp.simplify(sp.integrate(arriba - abajo, (v, p, q)))
        r.paso(f'Tramo [{tex(p)}, {tex(q)}]', r'Arriba está \(' + tex(arriba) + r'\).',
               r'\int_{' + tex(p) + '}^{' + tex(q) + r'} \left(' + tex(arriba) + ' - ' + tex(abajo) + r'\right) d' + tex(v)
               + ' = ' + tex(parcial), 0)
        total += parcial
    total = sp.simplify(total)
    r.valor = total
    r.valores['decimal'] = fmt_num(decimal(total), 12)
    r.resultado_latex = 'A = ' + tex(total) + (r' \approx ' + fmt_num(decimal(total), 8) if not total.is_Rational else '')
    r.comparar = ('numero_o_expr', total, {})
    try:
        gr = G.grafico_2d([f, g], v, [G.punto(c, f.subs(v, c)) for c in cortes],
                          xmin=float(a) - 0.5, xmax=float(b) + 0.5)
        gr['sombra'] = {'arriba': G.js(f), 'abajo': G.js(g), 'var': str(v), 'desde': float(a), 'hasta': float(b),
                        'entre': True}
        r.grafico = gr
    except Exception:  # noqa
        pass
    return r


# ----------------------------------------------------------------------------
# Límites
# ----------------------------------------------------------------------------
def _leer_limite(arg):
    a = arg.strip()
    a = re.sub(r'cuando\s+([a-z])\s+tiende\s+a\s+', r', \1->', a)
    a = a.replace('tiende a', '->')
    # lim x->a f   |  lim_(x->a) f
    m = re.match(r'^_?\(?\s*([a-z])\s*->\s*([^\s,)]+)\s*\)?\s+(.+)$', a)
    if m:
        return m.group(3), m.group(1), m.group(2)
    partes = dividir(a)
    if len(partes) >= 2 and '->' in partes[-1]:
        v, punto_ = partes[-1].split('->')
        return ','.join(partes[:-1]), v.strip(), punto_.strip()
    if len(partes) == 3:
        return partes[0], partes[1], partes[2]
    raise ErrorEntrada('Uso: limite f, x->a   (por ejemplo: limite sen(x)/x, x->0)')


def _forma(valor_num, valor_den):
    def clase(v):
        if v == 0:
            return '0'
        if v.is_infinite or v in (sp.oo, -sp.oo, sp.zoo):
            return r'\infty'
        return None
    return clase(valor_num), clase(valor_den)


def limite(arg):
    # límite doble: (x,y)->(a,b)  o  x->a, y->b
    if re.search(r'\(\s*[a-z]\s*,\s*[a-z]\s*\)\s*(->|→)', arg) or len(re.findall(r'[a-z]\s*(->|→)', arg)) >= 2:
        from .calculo_extra import limite_doble
        return limite_doble(arg)
    cuerpo, v_txt, p_txt = _leer_limite(arg)
    f = leer(cuerpo)
    v = sp.Symbol(v_txt) if v_txt not in ('x', 'y', 't', 'n') else leer(v_txt)
    lado = '+-'
    p_txt = p_txt.strip()
    if p_txt.endswith('+') and len(p_txt) > 1:
        lado, p_txt = '+', p_txt[:-1]
    elif p_txt.endswith('-') and len(p_txt) > 1:
        lado, p_txt = '-', p_txt[:-1]
    a = leer(p_txt)
    r = Resultado('Cálculo', 'Límite')
    sup = {'+': '^+', '-': '^-'}.get(lado, '')
    r.entrada_latex = r'\lim_{' + tex(v) + r'\to ' + tex(a) + sup + '} ' + tex(f)

    def calc(expr):
        if lado == '+-' and not a.is_infinite:
            l1 = sp.limit(expr, v, a, '+')
            l2 = sp.limit(expr, v, a, '-')
            return l1 if l1 == l2 else sp.nan, (l1, l2)
        return sp.limit(expr, v, a, lado if lado != '+-' else '+'), None

    valor, laterales = calc(f)
    # Paso 1: sustitución directa
    try:
        directo = f.subs(v, a) if not a.is_infinite else None
    except Exception:  # noqa
        directo = None
    if directo is not None and directo.is_finite and not directo.has(sp.nan, sp.zoo) and sp.simplify(directo - valor) == 0:
        r.paso('Sustitución directa', 'La función es continua en el punto: se reemplaza.',
               r.entrada_latex + ' = ' + tex(directo), 0)
    else:
        num, den = sp.fraction(sp.together(f))
        explicado = False
        if den != 1 and den.has(v):
            ln_, ld_ = sp.limit(num, v, a, '+' if lado != '-' else '-'), sp.limit(den, v, a, '+' if lado != '-' else '-')
            fn, fd = _forma(ln_, ld_)
            if fn and fd and fn == fd:
                r.paso('Indeterminación', f'Al reemplazar queda la forma indeterminada.',
                       r'\frac{' + fn + '}{' + fd + '}', 0)
                explicado = True
                if fn == '0' and num.is_polynomial(v) and den.is_polynomial(v) and not a.is_infinite:
                    fact = sp.cancel(f)
                    r.paso('Factorizar y simplificar', 'Se factoriza y se cancela el factor que anula.',
                           r'\frac{' + tex(sp.factor(num)) + '}{' + tex(sp.factor(den)) + '} = ' + tex(fact), 0)
                    r.paso('Sustituir', '', r'\lim ' + tex(fact) + ' = ' + tex(valor), 0)
                elif a.is_infinite and num.is_polynomial(v) and den.is_polynomial(v):
                    gn, gd = sp.degree(num, v), sp.degree(den, v)
                    r.paso('Dividir por la mayor potencia', f'Grado del numerador {gn}, grado del denominador {gd}. '
                           + ('Mismo grado: el límite es el cociente de coeficientes principales.' if gn == gd else
                              ('El denominador crece más rápido: el límite es 0.' if gn < gd else
                               'El numerador crece más rápido: el límite es infinito.')),
                           r.entrada_latex + ' = ' + tex(valor), 0)
                else:
                    # L'Hôpital
                    nn, dd = num, den
                    for i in range(1, 5):
                        nn, dd = sp.diff(nn, v), sp.diff(dd, v)
                        r.paso(f"Regla de L'Hôpital ({i})", 'Se derivan numerador y denominador por separado.',
                               r'\lim \frac{' + tex(sp.simplify(nn)) + '}{' + tex(sp.simplify(dd)) + '}', 0)
                        try:
                            l1 = sp.limit(nn, v, a, '+' if lado != '-' else '-')
                            l2 = sp.limit(dd, v, a, '+' if lado != '-' else '-')
                        except Exception:  # noqa
                            break
                        f1, f2 = _forma(l1, l2)
                        if not (f1 and f2 and f1 == f2):
                            break
                    r.paso('Resultado', '', r.entrada_latex + ' = ' + tex(valor), 0)
            elif fd == '0' and fn is None:
                r.paso('División por cero', 'El numerador tiende a un valor no nulo y el denominador a 0: '
                       'el límite es infinito (se analizan los laterales).', '', 0)
                explicado = True
        if not explicado:
            # 1^oo, oo-oo, 0*oo ...
            base_exp = f.as_base_exp() if f.is_Pow else None
            if base_exp and base_exp[1].has(v) and base_exp[0].has(v):
                b_, e_ = base_exp
                lb = sp.limit(b_, v, a, '+' if lado != '-' else '-')
                le = sp.limit(e_, v, a, '+' if lado != '-' else '-')
                if lb == 1 and le.is_infinite:
                    r.paso('Indeterminación 1^∞', r'Se usa \(\lim f^g = e^{\lim g\,(f-1)}\).',
                           r'e^{\lim ' + tex(sp.simplify(e_ * (b_ - 1))) + '}', 0)
            r.paso('Cálculo del límite', '', r.entrada_latex + ' = ' + tex(valor), 0)
    if laterales and laterales[0] != laterales[1]:
        r.paso('Límites laterales', 'Los límites laterales son distintos.',
               r'\lim_{' + tex(v) + r'\to ' + tex(a) + '^+} = ' + tex(laterales[0]) + r',\quad \lim_{' + tex(v)
               + r'\to ' + tex(a) + '^-} = ' + tex(laterales[1]), 0)
        r.extra('Conclusión', '', 'El límite no existe (los laterales difieren).')
        r.valores['derecha'] = laterales[0]
        r.valores['izquierda'] = laterales[1]
        r.resultado_latex = r.entrada_latex + r'\ \text{no existe}'
        r.valor = 'no existe'
        r.comparar = ('texto', 'no existe', {})
    else:
        r.valor = valor
        r.resultado_latex = r.entrada_latex + ' = ' + tex(valor)
        r.comparar = ('numero_o_expr', valor, {})
    if not a.is_infinite and len(variables_de(f)) <= 1:
        pto = G.punto(a, valor, '') if valor.is_finite else None
        _grafico_funcion(r, [f], v, [pto] if pto else [], xmin=float(a) - 5, xmax=float(a) + 5)
    elif len(variables_de(f)) <= 1:
        _grafico_funcion(r, [f], v, xmin=-2 if a == sp.oo else -60, xmax=60 if a == sp.oo else 2)
    return r


# ----------------------------------------------------------------------------
# Taylor
# ----------------------------------------------------------------------------
def taylor(arg):
    partes = dividir(arg)
    f = leer(partes[0])
    v = leer(partes[1]) if len(partes) > 1 else variable_principal(f)
    a = leer(partes[2]) if len(partes) > 2 else sp.Integer(0)
    orden = int(leer(partes[3])) if len(partes) > 3 else 5
    r = Resultado('Cálculo', 'Polinomio de Taylor' if a != 0 else 'Polinomio de Mac Laurin')
    r.entrada_latex = 'P_{' + str(orden) + '}(' + tex(v) + r')\ \text{de}\ ' + tex(f) + r'\ \text{en}\ ' + tex(v) + '_0=' + tex(a)
    r.paso('Fórmula', '', r'P_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(a)}{k!}(x-a)^k', 0)
    filas = []
    pol = 0
    d = f
    for k_ in range(orden + 1):
        val = sp.simplify(d.subs(v, a)) if k_ == 0 or True else None
        if val.has(sp.nan, sp.zoo) or not val.is_finite:
            val = sp.limit(d, v, a)
        coef = sp.simplify(val / sp.factorial(k_))
        pol += coef * (v - a) ** k_
        filas.append([str(k_), '$' + tex(simplificar_bonito(d)) + '$', '$' + tex(val) + '$', '$' + tex(coef) + '$'])
        d = sp.diff(d, v)
    r.tabla = {'columnas': ['k', 'f^(k)(x)', 'f^(k)(a)', 'coeficiente'], 'filas': filas}
    pol = sp.expand(pol) if a == 0 else pol
    r.paso('Sumar los términos', '', 'P_{' + str(orden) + '}(' + tex(v) + ') = ' + tex(pol), 0)
    r.valor = sp.expand(pol)
    r.resultado_latex = 'P_{' + str(orden) + '}(' + tex(v) + ') = ' + tex(pol)
    r.comparar = ('expr', sp.expand(pol), {})
    _grafico_funcion(r, [f, pol], v, etiquetas=['f', f'P_{orden}'],
                     xmin=float(a) - 4, xmax=float(a) + 4)
    return r


# ----------------------------------------------------------------------------
# Extremos (una variable)
# ----------------------------------------------------------------------------
def extremos_1v(f, v):
    r = Resultado('Cálculo', 'Extremos y monotonía')
    r.entrada_latex = 'f(' + tex(v) + ') = ' + tex(f)
    df = simplificar_bonito(sp.diff(f, v))
    d2 = simplificar_bonito(sp.diff(f, v, 2))
    r.paso('Derivada primera', '', "f'(" + tex(v) + ') = ' + tex(df), 0)
    vr = sp.Symbol(str(v), real=True)
    try:
        crit = sp.solveset(df.subs(v, vr), vr, sp.S.Reals)
    except Exception:  # noqa
        crit = sp.ConditionSet(vr, sp.Eq(df, 0), sp.S.Reals)
    if not isinstance(crit, sp.FiniteSet):
        try:
            crit = sp.FiniteSet(*[c for c in sp.solve(df, v) if c.is_real])
        except Exception:  # noqa
            raise ErrorEntrada('No se pudieron encontrar los puntos críticos de forma exacta.')
    crit = sorted([sp.nsimplify(c) if c.is_Float else c for c in crit], key=lambda c: float(c))
    r.paso('Puntos críticos', "Se resuelve f'(x) = 0.", "f'(" + tex(v) + r') = 0 \Rightarrow ' + tex(v) + ' = '
           + (tex(crit) if crit else r'\text{ninguno}'), 0)
    r.paso('Derivada segunda', '', "f''(" + tex(v) + ') = ' + tex(d2), 0)
    tipos, puntos = [], []
    for c in crit:
        val2 = sp.simplify(d2.subs(v, c))
        fc = sp.simplify(f.subs(v, c))
        if val2.is_positive:
            tipo = 'mínimo'
        elif val2.is_negative:
            tipo = 'máximo'
        else:
            # criterio de la primera derivada
            eps = sp.Rational(1, 1000)
            izq, der = df.subs(v, c - eps), df.subs(v, c + eps)
            if izq < 0 < der:
                tipo = 'mínimo'
            elif izq > 0 > der:
                tipo = 'máximo'
            else:
                tipo = 'ninguno (punto de inflexión)'
        tipos.append(tipo)
        r.paso(f'Clasificar x = {tex(c)}', '', "f''(" + tex(c) + ') = ' + tex(val2) + r'\Rightarrow\ \text{' + tipo + '}'
               + r',\ f(' + tex(c) + ') = ' + tex(fc), 1)
        puntos.append(G.punto(c, fc, f'{tipo} ({fmt_num(c, 4)}, {fmt_num(fc, 4)})'))
    # monotonía
    if crit:
        bordes = [-sp.oo] + crit + [sp.oo]
        filas = []
        for i in range(len(bordes) - 1):
            p, q = bordes[i], bordes[i + 1]
            prueba = (p + q) / 2 if p.is_finite and q.is_finite else (q - 1 if p == -sp.oo else p + 1)
            try:
                signo = float(df.subs(v, prueba))
            except Exception:  # noqa
                continue
            filas.append([f'({fmt_num(p, 4) if p.is_finite else "-∞"}, {fmt_num(q, 4) if q.is_finite else "∞"})',
                          '+' if signo > 0 else '−', 'crece' if signo > 0 else 'decrece'])
        r.tabla = {'columnas': ['Intervalo', "signo f'", 'f'], 'filas': filas}
    # inflexión
    try:
        infl = [c for c in sp.solveset(d2.subs(v, vr), vr, sp.S.Reals)] if d2 != 0 else []
        if infl and len(infl) < 8:
            r.extra('Posibles puntos de inflexión', r"f''(x) = 0 \Rightarrow x = " + tex(infl))
    except Exception:  # noqa
        pass
    r.valor = crit
    r.valores['tipos'] = ', '.join(tipos)
    r.resultado_latex = (r'\text{Puntos críticos: } ' + ', '.join(f'{tex(c)}\\ (\\text{{{t_}}})' for c, t_ in zip(crit, tipos))
                         if crit else r'\text{Sin puntos críticos}')
    r.comparar = ('conjunto', crit, {})
    _grafico_funcion(r, [f], v, puntos)
    return r


# ----------------------------------------------------------------------------
# Sumatorias
# ----------------------------------------------------------------------------
def sumatoria(arg):
    partes = dividir(arg)
    if len(partes) < 4:
        raise ErrorEntrada('Uso: sumatoria término, k, desde, hasta   (por ejemplo: sumatoria 1/k^2, k, 1, oo)')
    f = leer(partes[0])
    k_ = leer(partes[1])
    a, b = leer(partes[2]), leer(partes[3])
    r = Resultado('Cálculo', 'Sumatoria' if b != sp.oo else 'Serie')
    r.entrada_latex = r'\sum_{' + tex(k_) + '=' + tex(a) + '}^{' + tex(b) + '} ' + tex(f)
    valor = sp.summation(f, (k_, a, b))
    valor = simplificar_bonito(valor) if not valor.has(sp.Sum) else valor
    if b == sp.oo:
        # criterio
        razon = None
        try:
            kk = sp.Symbol(str(k_), integer=True, positive=True)
            fk = f.subs(k_, kk)
            cociente = sp.simplify(sp.combsimp(sp.powsimp(fk.subs(kk, kk + 1) / fk, force=True)))
            razon = sp.limit(sp.Abs(cociente), kk, sp.oo)
            r.paso('Criterio del cociente (D\'Alembert)', r'\(L = \lim \left|\frac{a_{k+1}}{a_k}\right|\)',
                   'L = ' + tex(razon) + (r' < 1 \Rightarrow \text{converge}' if razon < 1 else
                                         (r' > 1 \Rightarrow \text{diverge}' if razon > 1 else r' = 1\ \text{(no decide)}')), 0)
        except Exception:  # noqa
            pass
        try:
            # la condición necesaria solo hace falta si el cociente no decidió
            kk = sp.Symbol(str(k_), integer=True, positive=True)
            lt = sp.limit(f.subs(k_, kk), kk, sp.oo) if razon is None or razon == 1 else 0
            if lt != 0:
                r.paso('Condición necesaria', 'El término general no tiende a 0: la serie diverge.', r'\lim a_k = ' + tex(lt), 0)
        except Exception:  # noqa
            pass
        conv = valor.is_finite if not valor.has(sp.Sum) else None
        r.extra('Conclusión', '', 'La serie converge.' if conv else ('La serie diverge.' if conv is False else 'No se pudo determinar la suma exacta.'))
    r.paso('Suma', 'Se usan las fórmulas de sumas conocidas (aritmética, geométrica, potencias, series notables).',
           r.entrada_latex + ' = ' + tex(valor), 0)
    if valor.has(sp.Sum):
        aprox = sp.N(sp.Sum(f, (k_, a, b)), 12) if b == sp.oo else None
        if aprox is not None:
            r.valores['decimal'] = fmt_num(aprox, 12)
    r.valor = valor
    r.resultado_latex = r.entrada_latex + ' = ' + tex(valor)
    r.comparar = ('numero_o_expr', valor, {})
    return r
