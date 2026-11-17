"""Álgebra: simplificar, factorizar, ecuaciones, sistemas, inecuaciones, complejos, circuitos."""
import re

import sympy as sp

from . import graficos as G
from .nucleo import (ErrorEntrada, coma_decimal, leer_sin_evaluar, pasos_jerarquia, tex_crudo, Resultado, dividir, leer, leer_ecuacion, simplificar_bonito, tex, variables_de,
                     variable_principal, x, decimal, fmt_num, normalizar)


def _grafico_1v(r, exprs, puntos=()):
    vs = set()
    for e in exprs:
        vs |= set(variables_de(e))
    if len(vs) == 1:
        try:
            r.grafico = G.grafico_2d(exprs, list(vs)[0], puntos)
        except Exception:  # noqa
            pass


def simplificar(arg, titulo='Simplificar'):
    e = leer(coma_decimal(arg))
    try:
        crudo = leer_sin_evaluar(coma_decimal(arg))
        tex_original = tex_crudo(crudo)
    except Exception:  # noqa
        crudo, tex_original = None, None
    if isinstance(e, sp.Tuple) or isinstance(e, list):
        raise ErrorEntrada('Para vectores y matrices usá los comandos de álgebra lineal (det, inversa, cruz...).')
    r = Resultado('Álgebra', titulo)
    r.entrada_latex = tex(e)
    if e.has(sp.I) and not variables_de(e):
        return complejo(e)
    if tex_original:
        r.entrada_latex = tex_original
    if e.is_number:
        exacto = sp.nsimplify(sp.simplify(e)) if e.has(sp.Float) else sp.simplify(e)
        r.titulo = 'Calcular'
        r.valor = exacto
        dec = decimal(exacto)
        r.valores['decimal'] = fmt_num(dec, 15) if dec is not None else None
        pasos = []
        try:
            pasos = pasos_jerarquia(crudo) if crudo is not None else []
        except Exception:  # noqa
            pasos = []
        if pasos:
            r.paso('Jerarquía de operaciones', 'Primero los paréntesis (de adentro hacia afuera), después potencias y '
                   'raíces, luego multiplicaciones y divisiones, y al final sumas y restas, de izquierda a derecha.',
                   tex_original, 0)
            for desc, lat in pasos:
                r.paso(desc, '', ('' if desc == 'Simplificar la raíz' else '= ') + lat, 1)
            def _firma(latex_):  # mismos símbolos sin importar el orden (3√6 + 2 = 2 + 3√6)
                return sorted(latex_.replace('\\cdot', '').replace(' ', ''))
            if _firma(pasos[-1][1].split('=')[-1]) != _firma(tex(exacto)):
                r.paso('Agrupar términos semejantes', 'Se suman los términos del mismo tipo (por ejemplo, radicales con '
                       'el mismo radicando).', '= ' + tex(exacto), 1)
            r.paso('Resultado exacto', 'Como fracción irreducible y en decimal.', tex(exacto) + (r' \approx ' + fmt_num(dec, 10) if dec is not None and not exacto.is_Integer and not isinstance(dec, complex) else ''), 0)
        else:
            r.paso('Operar', 'Se resuelven las operaciones respetando la jerarquía (paréntesis, potencias, '
                   'productos y cocientes, sumas y restas).', (tex_original or tex(e)) + ' = ' + tex(exacto), 0)
        r.resultado_latex = tex(exacto) + (r' \approx ' + fmt_num(dec, 10) if dec is not None and not exacto.is_Integer
                                           and not isinstance(dec, complex) else '')
        r.comparar = ('numero_o_expr', exacto, {})
        return r
    num, den = sp.fraction(sp.together(e))
    if den != 1 and den.has(*e.free_symbols):
        fn, fd = sp.factor(num), sp.factor(den)
        if fn != num or fd != den:
            r.paso('Factorizar numerador y denominador', '', r'\frac{' + tex(fn) + '}{' + tex(fd) + '}', 0)
        c = sp.cancel(e)
        if c != e:
            r.paso('Cancelar factores comunes', 'Se simplifican los factores repetidos (válido donde el factor no se anula).',
                   tex(sp.factor(c)), 0)
    if e.has(sp.sin, sp.cos, sp.tan, sp.sec, sp.csc, sp.cot):
        ts = sp.trigsimp(e)
        if ts != e:
            r.paso('Identidades trigonométricas', r'Se usan \(\sin^2 + \cos^2 = 1\), ángulo doble, etc.', tex(ts), 0)
    final = simplificar_bonito(e)
    if tex_original and tex_original != tex(e):
        r.paso('Operar los números', 'Se multiplican y simplifican los coeficientes numéricos.', tex_original + ' = ' + tex(e), 0)
    r.paso('Resultado', '', tex(e) + ' = ' + tex(final), 0)
    r.valor = final
    r.resultado_latex = (tex_original or tex(e)) + ' = ' + tex(final)
    r.comparar = ('expr', final, {})
    _grafico_1v(r, [final])
    return r


def evaluar(arg):
    partes = dividir(arg)
    e = leer(partes[0])
    subs = {}
    for p in partes[1:]:
        if '=' in p:
            a, b = p.split('=', 1)
            subs[leer(a)] = leer(b)
    r = Resultado('Álgebra', 'Evaluar')
    r.entrada_latex = tex(e) + (r'\Big|_{' + ', '.join(f'{tex(a)}={tex(b)}' for a, b in subs.items()) + '}' if subs else '')
    v = sp.simplify(e.subs(subs))
    if subs:
        r.paso('Reemplazar', '', tex(e.subs(subs, simultaneous=True).doit() if False else e) + r'\ \to\ ' + tex(v), 0)
    dec = decimal(v)
    r.valor = v
    r.valores['decimal'] = fmt_num(dec, 15) if dec is not None else None
    r.resultado_latex = r.entrada_latex + ' = ' + tex(v) + (r' \approx ' + fmt_num(dec, 12) if dec is not None and not v.is_Integer else '')
    r.paso('Valor', '', r.resultado_latex, 0)
    r.comparar = ('numero_o_expr', v, {})
    return r


def expandir(arg):
    e = leer(arg)
    r = Resultado('Álgebra', 'Expandir')
    r.entrada_latex = tex(e)
    res = sp.expand(e)
    if e.is_Pow and e.exp.is_Integer and e.base.is_Add and len(e.base.args) == 2 and e.exp <= 8:
        a, b = e.base.args
        nn = int(e.exp)
        r.paso('Binomio de Newton', r'\((a+b)^n = \sum \binom{n}{k} a^{n-k} b^k\)',
               ' + '.join(r'\binom{' + str(nn) + '}{' + str(k_) + '}' + '(' + tex(a) + ')^{' + str(nn - k_) + '}(' + tex(b) + ')^{' + str(k_) + '}' for k_ in range(nn + 1)), 0)
    else:
        r.paso('Propiedad distributiva', 'Se multiplica término a término y se agrupan los semejantes.', tex(e) + ' = ' + tex(res), 0)
    r.valor = res
    r.resultado_latex = tex(e) + ' = ' + tex(res)
    r.comparar = ('expr_forma', res, {'forma': 'expandida'})
    return r


def factorizar(arg):
    e = leer(arg)
    if e.is_Integer and abs(e) > 1:
        from .geo_lineal import factores_primos
        return factores_primos(arg)
    r = Resultado('Álgebra', 'Factorizar')
    r.entrada_latex = tex(e)
    v = variable_principal(e)
    res = sp.factor(e)
    if e.is_polynomial(*e.free_symbols) and len(e.free_symbols) == 1:
        pol = sp.Poly(e, v)
        cont, prim = pol.primitive()
        mon = sp.gcd_terms(e)
        comun = sp.factor_terms(e)
        if comun != e:
            r.paso('Factor común', '', tex(e) + ' = ' + tex(comun), 0)
        raices = sp.roots(pol)
        if raices:
            r.paso('Raíces', 'Cada raíz r da un factor (x − r). Se buscan con Ruffini/Gauss (divisores del término '
                   'independiente sobre divisores del coeficiente principal) o la resolvente.',
                   ', '.join(f'{tex(v)} = {tex(rr)}' + (f'\\ (\\text{{mult. }}{m})' if m > 1 else '') for rr, m in raices.items()), 0)
    r.paso('Factorización', '', tex(e) + ' = ' + tex(res), 0)
    r.valor = res
    r.resultado_latex = tex(e) + ' = ' + tex(res)
    r.comparar = ('expr_forma', res, {'forma': 'factorizada'})
    return r


def fracciones_parciales(arg):
    e = leer(arg)
    v = variable_principal(e)
    r = Resultado('Álgebra', 'Fracciones parciales')
    r.entrada_latex = tex(e)
    num, den = sp.fraction(sp.cancel(e))
    if sp.degree(num, v) >= sp.degree(den, v):
        q, rr = sp.div(num, den, v)
        r.paso('División de polinomios', 'El grado del numerador no es menor: primero se divide.',
               tex(e) + ' = ' + tex(q) + r' + \frac{' + tex(rr) + '}{' + tex(den) + '}', 0)
    fden = sp.factor(den)
    r.paso('Factorizar el denominador', '', tex(den) + ' = ' + tex(fden), 0)
    # Plantear con incógnitas A, B, C...
    ap = sp.apart(e, v)
    terminos = sp.Add.make_args(ap)
    letras = iter('ABCDEFGHIJKLMN')
    plantilla = []
    for tm in terminos:
        n_, d_ = sp.fraction(tm)
        if d_.has(v):
            if sp.degree(d_.as_base_exp()[0] if d_.is_Pow else d_, v) == 2:
                plantilla.append(r'\frac{' + next(letras) + tex(v) + '+' + next(letras) + '}{' + tex(d_) + '}')
            else:
                plantilla.append(r'\frac{' + next(letras) + '}{' + tex(d_) + '}')
    if plantilla:
        r.paso('Plantear la descomposición', 'Un término por cada factor (y por cada potencia de factores repetidos).',
               ' + '.join(plantilla), 0)
        r.paso('Calcular las constantes', 'Se multiplica por el denominador común y se igualan coeficientes '
               '(o se evalúa en las raíces).', '', 0)
    r.paso('Resultado', '', tex(e) + ' = ' + tex(ap), 0)
    r.valor = ap
    r.resultado_latex = tex(e) + ' = ' + tex(ap)
    r.comparar = ('expr', ap, {})
    return r


# ----------------------------------------------------------------------------
# Ecuaciones
# ----------------------------------------------------------------------------
def _ordenar_sol(sols):
    def clave(s_):
        try:
            c = complex(sp.N(s_))
            return (abs(c.imag) > 1e-12, c.real, c.imag)
        except Exception:  # noqa
            return (True, 0, 0)
    return sorted(sols, key=clave)


def resolver(arg):
    if '≡' in arg or re.search(r'(?<![a-z])mod(?![a-z])', arg.lower()):
        from .geo_lineal import congruencia
        return congruencia(arg)
    arg = coma_decimal(arg)
    partes = dividir(arg, ';')
    if len(partes) > 1:
        return sistema(partes)
    # "resolver eq, x" -> variable indicada
    sub = dividir(arg)
    var_ind = None
    if len(sub) == 2 and re.fullmatch(r'\s*[a-z]\s*', sub[1]):
        arg, var_ind = sub[0], leer(sub[1])
    elif len(sub) > 1:
        return sistema(sub)
    eq = leer_ecuacion(arg)
    if isinstance(eq, (sp.StrictLessThan, sp.LessThan, sp.StrictGreaterThan, sp.GreaterThan, sp.And)):
        return inecuacion(eq)
    if not isinstance(eq, sp.Equality):
        eq = sp.Eq(eq, 0)
    expr = sp.simplify(eq.lhs - eq.rhs) if sp.count_ops(eq.lhs - eq.rhs) < 60 else eq.lhs - eq.rhs
    vs = variables_de(expr)
    v = var_ind or (vs[0] if vs else x)
    r = Resultado('Álgebra', 'Ecuación')
    r.entrada_latex = tex(eq.lhs) + ' = ' + tex(eq.rhs)
    if eq.rhs != 0:
        r.paso('Pasar todo a un miembro', '', tex(sp.expand(eq.lhs - eq.rhs)) + ' = 0', 0)
    # dominio: denominadores
    num, den = sp.fraction(sp.together(expr))
    restric = []
    if den.has(v):
        try:
            restric = [c for c in sp.solve(den, v)]
            if restric:
                r.paso('Restricciones', 'Los valores que anulan un denominador no pueden ser solución.',
                       tex(v) + r' \neq ' + tex(restric), 0)
        except Exception:  # noqa
            pass
        expr_pol = num
    else:
        expr_pol = expr
    soluciones = None
    complejas = False
    if expr_pol.is_polynomial(v) and len(variables_de(expr_pol)) == 1:
        pol = sp.Poly(sp.expand(expr_pol), v)
        g = pol.degree()
        coefs = pol.all_coeffs()
        if g == 1:
            a, b = coefs
            sol = -b / a
            r.paso('Despejar', 'Ecuación lineal: se despeja la incógnita.',
                   tex(a) + tex(v) + ' = ' + tex(-b) + r'\Rightarrow ' + tex(v) + ' = ' + tex(sol), 0)
            soluciones = [sol]
        elif g == 2:
            a, b, c = coefs
            disc = sp.simplify(b ** 2 - 4 * a * c)
            r.paso('Fórmula resolvente', r'Para \(a x^2 + b x + c = 0\): \(x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}\)',
                   f'a = {tex(a)},\\ b = {tex(b)},\\ c = {tex(c)}', 0)
            r.paso('Discriminante', ('Δ > 0: dos raíces reales distintas.' if disc.is_positive else
                                     'Δ = 0: una raíz doble.' if disc == 0 else
                                     'Δ < 0: raíces complejas conjugadas.' if disc.is_negative else ''),
                   r'\Delta = b^2 - 4ac = ' + tex(disc), 0)
            x1 = sp.simplify((-b + sp.sqrt(disc)) / (2 * a))
            x2 = sp.simplify((-b - sp.sqrt(disc)) / (2 * a))
            r.paso('Raíces', '', tex(v) + r'_{1,2} = \frac{' + tex(-b) + r' \pm \sqrt{' + tex(disc) + '}}{' + tex(2 * a)
                   + '} \\Rightarrow ' + tex(v) + '_1 = ' + tex(x1) + r',\ ' + tex(v) + '_2 = ' + tex(x2), 0)
            soluciones = [x1] if disc == 0 else [x1, x2]
            complejas = disc.is_negative
        else:
            fac = sp.factor(expr_pol)
            if fac != sp.expand(expr_pol):
                r.paso('Factorizar', 'Se factoriza (factor común, Ruffini con divisores del término independiente, '
                       'productos notables).', tex(fac) + ' = 0', 0)
                r.paso('Producto igual a cero', 'Cada factor puede anularse.', '', 0)
            raices = sp.roots(pol)
            if sum(raices.values()) == g:
                soluciones = list(raices.keys())
            else:
                soluciones = [s_ for s_ in sp.solve(expr_pol, v)]
                if not soluciones or len(soluciones) < g:
                    soluciones = [sp.Float(s_, 12) for s_ in sp.Poly(expr_pol, v).nroots(n=15)]
                    r.avisos.append('Raíces calculadas numéricamente (no tienen expresión simple).')
    if soluciones is None:
        try:
            sset = sp.solveset(expr, v, sp.S.Complexes if expr.has(sp.I) else sp.S.Reals)
        except Exception:  # noqa
            sset = None
        if isinstance(sset, sp.FiniteSet):
            soluciones = list(sset)
            r.paso('Despejar', 'Se aplican las operaciones inversas (logaritmos, funciones inversas, potencias).',
                   tex(v) + ' = ' + tex(soluciones), 0)
        elif sset is not None and sset != sp.EmptySet and not isinstance(sset, sp.ConditionSet):
            # soluciones periódicas (trigonométricas)
            r.paso('Solución general', 'La ecuación tiene infinitas soluciones (periódicas), con n entero.',
                   tex(v) + r' \in ' + tex(sset), 0)
            r.valor = sset
            r.resultado_latex = tex(v) + r' \in ' + tex(sset)
            princ = []
            try:
                princ = sorted(sp.solveset(expr, v, sp.Interval.Ropen(0, 2 * sp.pi)), key=lambda c: float(c))
                if len(princ) < 12:
                    r.extra('Soluciones en [0, 2π)', tex(v) + ' = ' + tex(princ))
                    r.valores['principales'] = princ
            except Exception:  # noqa
                pass
            r.comparar = ('conjunto_periodico', sset, {'var': v, 'principales': princ})
            _grafico_1v(r, [eq.lhs, eq.rhs] if eq.rhs != 0 else [eq.lhs])
            return r
        elif sset == sp.EmptySet:
            soluciones = []
        else:
            # numérico
            soluciones = []
            for semilla in (-10, -3, -1, 0.5, 1, 3, 10):
                try:
                    s_ = sp.nsolve(expr, v, semilla)
                    if all(abs(s_ - o) > 1e-8 for o in soluciones):
                        soluciones.append(sp.Float(s_, 12))
                except Exception:  # noqa
                    pass
            r.avisos.append('Ecuación trascendente: soluciones aproximadas por método numérico (Newton).')
            r.paso('Método numérico', 'No se puede despejar en forma exacta; se buscan raíces con Newton-Raphson '
                   'desde varios puntos iniciales.', tex(v) + r' \approx ' + tex(soluciones), 0)
    soluciones = [sp.nsimplify(s_) if isinstance(s_, sp.Float) and abs(float(s_) - round(float(s_))) < 1e-12 else s_ for s_ in soluciones]
    soluciones = [s_ for s_ in soluciones if not any(sp.simplify(s_ - q) == 0 for q in restric)]
    # Dominio: logaritmos y raíces de índice par (en los reales)
    fuera = [s_ for s_ in soluciones if s_.is_real and not _en_dominio(sp.Add(eq.lhs, eq.rhs, evaluate=False), v, s_)]
    if fuera:
        r.paso('Verificar el dominio', 'Se descartan los valores donde un logaritmo tiene argumento ≤ 0 o una raíz par '
               'tiene radicando negativo.', tex(v) + ' = ' + tex(fuera) + r'\ 	ext{no pertenece(n) al dominio}', 0)
        soluciones = [s_ for s_ in soluciones if s_ not in fuera]
    soluciones = _ordenar_sol(list(dict.fromkeys(soluciones)))
    reales = [s_ for s_ in soluciones if s_.is_real is not False]
    r.valor = soluciones
    if not soluciones:
        r.resultado_latex = r'\text{No tiene solución}'
        r.paso('Conclusión', 'La ecuación no tiene solución.', '', 0)
    else:
        r.resultado_latex = ',\\ '.join(f'{tex(v)}_{{{i + 1}}} = {tex(s_)}' if len(soluciones) > 1 else f'{tex(v)} = {tex(s_)}'
                                         for i, s_ in enumerate(soluciones))
        if complejas or any(s_.is_real is False for s_ in soluciones):
            r.extra('Soluciones reales', ', '.join(tex(s_) for s_ in reales) if reales else r'\text{ninguna}')
        # verificación
        if len(soluciones) <= 6:
            for s_ in soluciones[:4]:
                try:
                    ver = sp.simplify(expr.subs(v, s_))
                    r.paso(f'Verificación {tex(v)} = {tex(s_)}', '', r'\text{reemplazando: } ' + tex(ver) + ' = 0', 1)
                except Exception:  # noqa
                    pass
    r.comparar = ('conjunto', soluciones, {})
    puntos = [G.punto(s_, 0, f'{fmt_num(s_, 5)}') for s_ in reales]
    if len(variables_de(expr)) == 1:
        try:
            r.grafico = G.grafico_2d([expr] if eq.rhs == 0 else [eq.lhs, eq.rhs], v,
                                     [G.punto(s_, eq.lhs.subs(v, s_)) for s_ in reales] if eq.rhs != 0 else puntos)
        except Exception:  # noqa
            pass
    return r


def _en_dominio(expr, v, valor):
    try:
        for a in expr.atoms(sp.log):
            arg = a.args[0].subs(v, valor)
            if arg.is_real is False or (arg.is_number and float(sp.N(arg)) <= 0):
                return False
        for p in expr.atoms(sp.Pow):
            if p.exp.is_Rational and p.exp.q % 2 == 0:
                b = p.base.subs(v, valor)
                if b.is_number and b.is_real and float(sp.N(b)) < 0:
                    return False
        for f in expr.atoms(sp.asin, sp.acos):
            a = f.args[0].subs(v, valor)
            if a.is_number and abs(float(sp.N(a))) > 1:
                return False
    except Exception:  # noqa
        return True
    return True


def sistema(partes):
    eqs = []
    for p in partes:
        e = leer_ecuacion(p)
        if not isinstance(e, sp.Equality):
            e = sp.Eq(e, 0)
        eqs.append(e)
    vs = []
    for e in eqs:
        for v in variables_de(e.lhs - e.rhs):
            if v not in vs:
                vs.append(v)
    vs = variables_de(sp.Tuple(*[e.lhs - e.rhs for e in eqs]))
    r = Resultado('Álgebra', 'Sistema de ecuaciones')
    r.entrada_latex = r'\begin{cases}' + r'\\'.join(tex(e.lhs) + ' = ' + tex(e.rhs) for e in eqs) + r'\end{cases}'
    lineal = all(sp.Poly(e.lhs - e.rhs, *vs).total_degree() <= 1 for e in eqs
                 if (e.lhs - e.rhs).is_polynomial(*vs))
    lineal = lineal and all((e.lhs - e.rhs).is_polynomial(*vs) for e in eqs)
    if lineal:
        A, b = sp.linear_eq_to_matrix([e.lhs - e.rhs for e in eqs], vs)
        from .lineal import gauss_jordan_pasos
        M = A.row_join(b)
        r.paso('Matriz ampliada', 'Se escribe el sistema como [A | b].', tex(M), 0)
        rref, pasos = gauss_jordan_pasos(M, aumentada=True)
        r.pasos.extend(pasos)
        rangoA, rangoM = A.rank(), M.rank()
        r.paso('Teorema de Rouché-Frobenius', f'rango(A) = {rangoA}, rango(A|b) = {rangoM}, incógnitas = {len(vs)}.',
               r'\text{' + ('Sistema compatible determinado (solución única)' if rangoA == rangoM == len(vs) else
                            'Sistema compatible indeterminado (infinitas soluciones)' if rangoA == rangoM else
                            'Sistema incompatible (sin solución)') + '}', 0)
        r.valores['clasificacion'] = ('SCD' if rangoA == rangoM == len(vs) else 'SCI' if rangoA == rangoM else 'SI')
    sol = sp.solve(eqs, vs, dict=True)
    if not sol:
        r.valor = []
        r.resultado_latex = r'\text{El sistema no tiene solución}'
        r.comparar = ('conjunto', [], {})
        return r
    if len(sol) == 1:
        s_ = {k_: sp.simplify(v) for k_, v in sol[0].items()}
        r.valor = s_
        r.resultado_latex = ',\\ '.join(f'{tex(k_)} = {tex(v)}' for k_, v in s_.items())
        libres = [v for v in vs if v not in s_]
        if libres:
            r.extra('Parámetros libres', ', '.join(tex(v) for v in libres))
        if not lineal:
            r.paso('Sustitución', 'Se despeja una incógnita y se reemplaza en las otras ecuaciones.', r.resultado_latex, 0)
        r.comparar = ('dict', s_, {})
    else:
        r.valor = sol
        r.resultado_latex = r'\\'.join('(' + ', '.join(f'{tex(k_)} = {tex(v)}' for k_, v in s_.items()) + ')' for s_ in sol)
        r.paso('Sustitución', 'Se despeja una incógnita y se reemplaza; aparecen varias soluciones.', r.resultado_latex, 0)
        r.comparar = ('lista_dict', sol, {})
    # gráfico: dos ecuaciones en x, y
    if len(vs) == 2 and len(eqs) == 2:
        try:
            g = G.grafico_implicita(eqs[0].lhs - eqs[0].rhs, tuple(vs), tex(eqs[0].lhs) + '=' + tex(eqs[0].rhs))
            g2 = G.grafico_implicita(eqs[1].lhs - eqs[1].rhs, tuple(vs), tex(eqs[1].lhs) + '=' + tex(eqs[1].rhs))
            g2['curvas'][0]['color'] = G.COLORES[1]
            g['curvas'] += g2['curvas']
            pts = []
            for s_ in sol:
                if all(v in s_ for v in vs):
                    p = G.punto(s_[vs[0]], s_[vs[1]], f'({fmt_num(s_[vs[0]], 4)}, {fmt_num(s_[vs[1]], 4)})')
                    if p:
                        pts.append(p)
            g['puntos'] = pts
            if pts:
                m = max(max(abs(p['x']), abs(p['y'])) for p in pts) * 1.6 + 2
                g['ventana'] = {'xmin': -m, 'xmax': m, 'ymin': -m, 'ymax': m}
            r.grafico = g
        except Exception:  # noqa
            pass
    return r


def inecuacion(rel):
    vs = variables_de(rel)
    v = vs[0] if vs else x
    vr = sp.Symbol(str(v), real=True)
    r = Resultado('Álgebra', 'Inecuación')
    r.entrada_latex = tex(rel)
    relr = rel.subs(v, vr)
    sol = sp.solveset(relr, vr, sp.S.Reals) if not isinstance(relr, sp.And) else \
        sp.Intersection(*[sp.solveset(a, vr, sp.S.Reals) for a in relr.args])
    sol = sol.subs(vr, v) if hasattr(sol, 'subs') else sol
    if not isinstance(rel, sp.And):
        expr = rel.lhs - rel.rhs
        r.paso('Pasar todo a un miembro', '', tex(rel.func(sp.simplify(expr), 0)), 0)
        num, den = sp.fraction(sp.together(sp.simplify(expr)))
        crit = set()
        for parte in (num, den):
            try:
                crit |= {c for c in sp.solveset(parte.subs(v, vr), vr, sp.S.Reals) if c.is_real}
            except Exception:  # noqa
                pass
        crit = sorted(crit, key=lambda c: float(c))
        if crit and len(crit) < 10:
            r.paso('Puntos críticos', 'Raíces del numerador y del denominador; dividen la recta en intervalos.',
                   tex(v) + ' = ' + tex(crit), 0)
            bordes = [-sp.oo] + crit + [sp.oo]
            filas = []
            for i in range(len(bordes) - 1):
                p, q = bordes[i], bordes[i + 1]
                prueba = (p + q) / 2 if p.is_finite and q.is_finite else (q - 1 if p == -sp.oo else p + 1)
                try:
                    val = float(expr.subs(v, prueba))
                except Exception:  # noqa
                    continue
                cumple = bool(rel.func(val, 0))
                filas.append([f'({fmt_num(p, 4) if p.is_finite else "-∞"}, {fmt_num(q, 4) if q.is_finite else "∞"})',
                              fmt_num(prueba, 4), '+' if val > 0 else '−', 'sí' if cumple else 'no'])
            r.tabla = {'columnas': ['Intervalo', 'valor de prueba', 'signo', '¿cumple?'], 'filas': filas}
    r.paso('Conjunto solución', '', tex(v) + r' \in ' + tex(sol), 0)
    r.valor = sol
    r.resultado_latex = tex(v) + r' \in ' + tex(sol)
    r.comparar = ('intervalo', sol, {'var': v})
    try:
        if not isinstance(rel, sp.And):
            r.grafico = G.grafico_2d([rel.lhs - rel.rhs], v)
    except Exception:  # noqa
        pass
    return r


def mcd_mcm(arg, cual):
    arg = arg.strip().replace(';', ',')
    if arg.startswith('(') and arg.endswith(')'):
        arg = arg[1:-1]
    nums = [leer(p) for p in dividir(arg)]
    r = Resultado('Álgebra', 'Máximo común divisor' if cual == 'mcd' else 'Mínimo común múltiplo')
    r.entrada_latex = ('mcd' if cual == 'mcd' else 'mcm') + '(' + ', '.join(tex(a) for a in nums) + ')'
    if all(a.is_Integer for a in nums):
        for a in nums:
            r.paso(f'Factorizar {a}', '', tex(a) + ' = ' + r' \cdot '.join(f'{p}^{{{e}}}' if e > 1 else str(p)
                                                                         for p, e in sp.factorint(abs(int(a))).items()), 0)
        r.paso('Regla', 'MCD: factores comunes con el menor exponente.' if cual == 'mcd' else
               'MCM: factores comunes y no comunes con el mayor exponente.', '', 0)
    res = sp.gcd_list(nums) if cual == 'mcd' else sp.lcm_list(nums)
    r.valor = res
    r.resultado_latex = r.entrada_latex + ' = ' + tex(res)
    r.comparar = ('numero_o_expr', res, {})
    return r


# ----------------------------------------------------------------------------
# Complejos y circuitos
# ----------------------------------------------------------------------------
def complejo(e):
    r = Resultado('Números complejos', 'Forma binómica')
    r.entrada_latex = tex(e)
    num, den = sp.fraction(e)
    if den.has(sp.I):
        conj = sp.conjugate(den)
        r.paso('Multiplicar por el conjugado', r'Se multiplica numerador y denominador por el conjugado del denominador.',
               r'\frac{' + tex(num) + '}{' + tex(den) + r'}\cdot\frac{' + tex(conj) + '}{' + tex(conj) + '} = '
               r'\frac{' + tex(sp.expand(num * conj)) + '}{' + tex(sp.expand(den * conj)) + '}', 0)
    res = sp.nsimplify(sp.expand_complex(sp.simplify(e)))
    res = sp.simplify(sp.re(res)) + sp.I * sp.simplify(sp.im(res))
    re_, im_ = sp.re(res), sp.im(res)
    res = sp.nsimplify(res) if res.has(sp.Float) else res
    r.paso('Forma binómica', r'\(i^2 = -1\)', tex(e) + ' = ' + tex(res), 0)
    mod = sp.simplify(sp.Abs(res))
    arg_ = sp.simplify(sp.arg(res))
    r.extra('Forma polar', tex(mod) + r'\,e^{i\,' + tex(arg_) + '}' + r' = ' + tex(mod) + r'\angle ' + fmt_num(sp.N(arg_ * 180 / sp.pi), 6) + r'^\circ')
    r.valores['modulo'] = mod
    r.valores['argumento'] = arg_
    r.valor = res
    r.resultado_latex = tex(e) + ' = ' + tex(res)
    r.comparar = ('numero_o_expr', res, {})
    try:
        r.grafico = G.grafico_vectores([((float(re_), float(im_)), tex(res))])
    except Exception:  # noqa
        pass
    return r


def polar(arg):
    z_ = sp.expand(sp.simplify(leer(arg)))
    r = Resultado('Números complejos', 'Forma polar')
    r.entrada_latex = 'z = ' + tex(z_)
    a, b = sp.re(z_), sp.im(z_)
    mod = sp.sqrt(a ** 2 + b ** 2)
    mod = sp.simplify(mod)
    arg_ = sp.simplify(sp.atan2(b, a))
    r.paso('Parte real e imaginaria', '', f'a = {tex(a)},\\ b = {tex(b)}', 0)
    r.paso('Módulo', '', r'|z| = \sqrt{a^2 + b^2} = ' + tex(mod), 0)
    r.paso('Argumento', 'Se corrige el cuadrante según los signos de a y b.', r'\theta = ' + tex(arg_) + r' \approx '
           + fmt_num(sp.N(arg_ * 180 / sp.pi), 6) + r'^\circ', 0)
    r.valor = [mod, arg_]
    r.valores['grados'] = fmt_num(sp.N(arg_ * 180 / sp.pi), 10)
    r.resultado_latex = 'z = ' + tex(mod) + r'\,e^{i\,' + tex(arg_) + '} = ' + tex(mod) + r'\left(\cos ' + tex(arg_) + r' + i\sin ' + tex(arg_) + r'\right)'
    r.comparar = ('vector', [mod, arg_], {})
    try:
        r.grafico = G.grafico_vectores([((float(a), float(b)), tex(z_))])
    except Exception:  # noqa
        pass
    return r


def fasor(arg):
    texto = normalizar(arg)
    # A@θ -> A*exp(i*θ°)
    texto = re.sub(r'([0-9.]+)\s*@\s*(-?[0-9.]+)', r'(\1*exp(i*(\2)*pi/180))', texto)
    e = leer(texto)
    r = Resultado('Circuitos', 'Operación con fasores')
    r.entrada_latex = tex(arg.replace('@', r'\angle '))
    r.entrada_latex = re.sub(r'([0-9.]+)\s*@\s*(-?[0-9.]+)', r'\1\\angle \2^\\circ', arg)
    terminos = re.findall(r'([0-9.]+)\s*@\s*(-?[0-9.]+)', arg)
    for A, th in terminos:
        c = sp.N(sp.Float(A) * sp.exp(sp.I * sp.Float(th) * sp.pi / 180))
        r.paso(f'{A}∠{th}° a binómica', r'\(A\angle\theta = A\cos\theta + i\,A\sin\theta\)',
               f'{A}\\angle {th}^\\circ = ' + fmt_num(sp.re(c), 6) + ' + ' + fmt_num(sp.im(c), 6) + 'i', 0)
    res = sp.N(sp.expand(e), 15)
    mod = sp.N(sp.Abs(res), 12)
    ang = sp.N(sp.arg(res) * 180 / sp.pi, 12)
    r.paso('Operar en forma binómica', 'Sumas/restas en binómica; productos y cocientes también se pueden hacer en polar.',
           '= ' + fmt_num(sp.re(res), 8) + ' + ' + fmt_num(sp.im(res), 8) + 'i', 0)
    r.paso('Volver a polar', '', '= ' + fmt_num(mod, 8) + r'\angle ' + fmt_num(ang, 8) + r'^\circ', 0)
    r.valor = res
    r.valores['modulo'] = fmt_num(mod, 12)
    r.valores['angulo'] = fmt_num(ang, 12)
    r.resultado_latex = fmt_num(sp.re(res), 8) + ('+' if sp.im(res) >= 0 else '') + fmt_num(sp.im(res), 8) + r'i = ' \
        + fmt_num(mod, 8) + r'\angle ' + fmt_num(ang, 8) + r'^\circ'
    r.comparar = ('numero', res, {'tol': 1e-4})
    try:
        r.grafico = G.grafico_vectores([((float(sp.re(res)), float(sp.im(res))), 'resultado')])
    except Exception:  # noqa
        pass
    return r


def resistencias(arg, modo):
    vals = [leer(p) for p in dividir(arg)]
    r = Resultado('Circuitos', 'Resistencias en paralelo' if modo == 'paralelo' else 'Resistencias en serie')
    r.entrada_latex = ', '.join(tex(v) for v in vals)
    if modo == 'paralelo':
        inv = sum(1 / v for v in vals)
        res = sp.simplify(1 / inv)
        r.paso('Fórmula', '', r'\frac{1}{R_{eq}} = ' + ' + '.join(r'\frac{1}{' + tex(v) + '}' for v in vals) + ' = ' + tex(sp.simplify(inv)), 0)
        if len(vals) == 2:
            r.extra('Para dos resistencias', r'R_{eq} = \frac{R_1 R_2}{R_1 + R_2}')
    else:
        res = sp.simplify(sum(vals))
        r.paso('Fórmula', 'En serie las resistencias (o impedancias) se suman.', r'R_{eq} = ' + ' + '.join(tex(v) for v in vals), 0)
    r.valor = res
    r.valores['decimal'] = fmt_num(decimal(res), 12)
    r.resultado_latex = r'R_{eq} = ' + tex(res) + (r' \approx ' + fmt_num(decimal(res), 8) if not res.is_Integer else '')
    r.comparar = ('numero_o_expr', res, {})
    return r
