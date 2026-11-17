"""Análisis Matemático II: varias variables y cálculo vectorial."""
import sympy as sp

from . import graficos as G
from .nucleo import (ErrorEntrada, Resultado, dividir, leer, leer_ecuacion, leer_lista, simplificar_bonito, tex,
                     variables_de, x, y, z, fmt_num, decimal)

XYZ = [x, y, z]


def _vars(expr_o_lista, minimo=2):
    vs = variables_de(sp.Tuple(*expr_o_lista) if isinstance(expr_o_lista, (list, tuple)) else expr_o_lista)
    if set(vs) <= set(XYZ):
        n_ = max([XYZ.index(v) + 1 for v in vs] + [minimo])
        return XYZ[:n_]
    return vs


def _r(titulo):
    return Resultado('Análisis II', titulo)


def gradiente(arg):
    f = leer(dividir(arg)[0])
    vs = _vars(f)
    r = _r('Gradiente')
    r.entrada_latex = r'\nabla\left(' + tex(f) + r'\right)'
    comps = []
    for v in vs:
        d = simplificar_bonito(sp.diff(f, v))
        comps.append(d)
        r.paso(f'Derivada parcial respecto de {v}', 'Las otras variables se tratan como constantes.',
               r'\frac{\partial f}{\partial ' + tex(v) + '} = ' + tex(d), 0)
    r.valor = comps
    r.resultado_latex = r'\nabla f = \left(' + ', '.join(tex(c) for c in comps) + r'\right)'
    r.extra('Uso', '', 'El gradiente apunta en la dirección de máximo crecimiento; su módulo es la derivada direccional máxima.')
    r.comparar = ('vector', comps, {})
    return r


def divergencia(arg):
    F = leer_lista(dividir(arg)[0]) if not arg.strip().startswith('[') else leer_lista(arg)
    vs = XYZ[:len(F)]
    r = _r('Divergencia')
    r.entrada_latex = r'\nabla\cdot\left(' + ', '.join(tex(c) for c in F) + r'\right)'
    partes = [sp.diff(c, v) for c, v in zip(F, vs)]
    val = simplificar_bonito(sum(partes))
    r.paso('Suma de derivadas parciales', r'\(\nabla\cdot F = \frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y} + \frac{\partial R}{\partial z}\)',
           ' + '.join(tex(p) for p in partes) + ' = ' + tex(val), 0)
    r.valor = val
    r.resultado_latex = r'\nabla\cdot F = ' + tex(val)
    if val == 0:
        r.extra('Conclusión', '', 'El campo es solenoidal (divergencia nula).')
    r.comparar = ('expr', val, {})
    return r


def rotacional(arg):
    F = leer_lista(arg)
    if len(F) == 2:
        F = F + [sp.Integer(0)]
    P, Q, R = F
    r = _r('Rotacional')
    r.entrada_latex = r'\nabla\times\left(' + ', '.join(tex(c) for c in F) + r'\right)'
    comps = [sp.diff(R, y) - sp.diff(Q, z), sp.diff(P, z) - sp.diff(R, x), sp.diff(Q, x) - sp.diff(P, y)]
    comps = [simplificar_bonito(c) for c in comps]
    r.paso('Determinante simbólico', '', r'\begin{vmatrix}\hat i & \hat j & \hat k\\ \partial_x & \partial_y & \partial_z \\ '
           + ' & '.join(tex(c) for c in F) + r'\end{vmatrix}', 0)
    r.paso('Componentes', '', r'\left(R_y - Q_z,\ P_z - R_x,\ Q_x - P_y\right) = \left(' + ', '.join(tex(c) for c in comps) + r'\right)', 0)
    r.valor = comps
    r.resultado_latex = r'\nabla\times F = \left(' + ', '.join(tex(c) for c in comps) + r'\right)'
    if all(c == 0 for c in comps):
        r.extra('Conclusión', '', 'El campo es irrotacional (conservativo en un dominio simplemente conexo).')
        try:
            pot = sp.integrate(P, x)
            pot += sp.integrate(sp.simplify(Q - sp.diff(pot, y)), y)
            pot += sp.integrate(sp.simplify(R - sp.diff(pot, z)), z)
            r.extra('Función potencial', r'\phi = ' + tex(sp.simplify(pot)) + ' + C')
        except Exception:  # noqa
            pass
    r.comparar = ('vector', comps, {})
    return r


def laplaciano(arg):
    f = leer(arg)
    vs = _vars(f)
    r = _r('Laplaciano')
    r.entrada_latex = r'\nabla^2\left(' + tex(f) + r'\right)'
    partes = [sp.diff(f, v, 2) for v in vs]
    val = simplificar_bonito(sum(partes))
    r.paso('Suma de segundas derivadas', '', ' + '.join(tex(p) for p in partes) + ' = ' + tex(val), 0)
    r.valor = val
    r.resultado_latex = r'\nabla^2 f = ' + tex(val)
    if val == 0:
        r.extra('Conclusión', '', 'La función es armónica.')
    r.comparar = ('expr', val, {})
    return r


def jacobiano(arg):
    partes = dividir(arg)
    F = leer_lista(partes[0])
    vs = leer_lista(partes[1]) if len(partes) > 1 else _vars(F, minimo=len(F) if len(F) <= 3 else 2)
    r = _r('Matriz jacobiana')
    r.entrada_latex = r'J\left(' + ', '.join(tex(c) for c in F) + r'\right)'
    J = sp.Matrix([[sp.diff(f, v) for v in vs] for f in F]).applyfunc(simplificar_bonito)
    r.paso('Derivadas parciales', 'Fila i: gradiente de la componente i.', 'J = ' + tex(J), 0)
    r.valor = J
    r.resultado_latex = 'J = ' + tex(J)
    if J.is_square:
        d = sp.simplify(J.det())
        r.extra('Determinante jacobiano', '|J| = ' + tex(d))
        r.valores['determinante'] = d
    r.comparar = ('matriz', J, {})
    return r


def hessiano(arg):
    f = leer(arg)
    vs = _vars(f)
    r = _r('Matriz hessiana')
    r.entrada_latex = r'H\left(' + tex(f) + r'\right)'
    H = sp.hessian(f, vs).applyfunc(simplificar_bonito)
    r.paso('Segundas derivadas', r'\(H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}\) (simétrica por Schwarz).', 'H = ' + tex(H), 0)
    r.valor = H
    r.resultado_latex = 'H = ' + tex(H)
    r.extra('Determinante', tex(sp.simplify(H.det())))
    r.comparar = ('matriz', H, {})
    return r


def integral_multiple(arg, dim):
    partes = dividir(arg)
    if len(partes) != 1 + 3 * dim:
        raise ErrorEntrada('Uso: integral doble f, x, a, b, y, c, d   (se integra primero en la primera variable)'
                           if dim == 2 else 'Uso: integral triple f, x, a, b, y, c, d, z, e, f')
    f = leer(partes[0])
    lims = []
    for i in range(dim):
        v = leer(partes[1 + 3 * i])
        lims.append((v, leer(partes[2 + 3 * i]), leer(partes[3 + 3 * i])))
    r = _r('Integral doble' if dim == 2 else 'Integral triple')
    ints = ''.join(r'\int_{' + tex(a) + '}^{' + tex(b) + '}' for v, a, b in reversed(lims))
    r.entrada_latex = ints + ' ' + tex(f) + r'\,' + ''.join('d' + tex(v) for v, a, b in lims)
    actual = f
    for i, (v, a, b) in enumerate(lims):
        prim = sp.integrate(actual, v)
        nuevo = sp.simplify(sp.integrate(actual, (v, a, b)))
        r.paso(f'Integrar respecto de {v}', 'Las demás variables quedan como constantes.' if i < dim - 1 else '',
               r'\int_{' + tex(a) + '}^{' + tex(b) + '} ' + tex(actual) + r'\,d' + tex(v) + r' = \Big[' + tex(sp.simplify(prim))
               + r'\Big]_{' + tex(a) + '}^{' + tex(b) + '} = ' + tex(nuevo), 0)
        actual = nuevo
    r.valor = actual
    dec = decimal(actual)
    r.valores['decimal'] = fmt_num(dec, 12) if dec is not None else None
    r.resultado_latex = r.entrada_latex + ' = ' + tex(actual)
    r.comparar = ('numero_o_expr', actual, {})
    if dim == 2 and all(a.is_number and b.is_number for v, a, b in lims):
        try:
            r.grafico = G.grafico_3d(f, (lims[0][0], lims[1][0]), tex(f), rango=float(max(abs(lims[0][2]), abs(lims[1][2]), 1)))
        except Exception:  # noqa
            pass
    return r


def extremos_2v(f, vs):
    r = _r('Extremos de dos variables')
    r.entrada_latex = 'f(' + ', '.join(tex(v) for v in vs) + ') = ' + tex(f)
    grad = [sp.diff(f, v) for v in vs]
    r.paso('Gradiente nulo', 'Puntos críticos: todas las derivadas parciales valen 0.',
           r'\begin{cases}' + r'\\'.join(tex(g) + ' = 0' for g in grad) + r'\end{cases}', 0)
    sols = sp.solve(grad, vs, dict=True)
    sols = [s_ for s_ in sols if all(s_.get(v, v).is_real is not False for v in vs)]
    H = sp.hessian(f, vs)
    r.paso('Hessiano', '', 'H = ' + tex(H), 0)
    puntos, tipos, filas = [], [], []
    for s_ in sols:
        p = tuple(sp.simplify(s_.get(v, 0)) for v in vs)
        Hp = H.subs(dict(zip(vs, p)))
        D = sp.simplify(Hp.det())
        fxx = sp.simplify(Hp[0, 0])
        if len(vs) == 2:
            if D > 0 and fxx > 0:
                tipo = 'mínimo local'
            elif D > 0 and fxx < 0:
                tipo = 'máximo local'
            elif D < 0:
                tipo = 'punto silla'
            else:
                tipo = 'no concluyente'
        else:
            ev = [complex(sp.N(e_)).real for e_ in Hp.eigenvals()]
            tipo = ('mínimo local' if all(e_ > 0 for e_ in ev) else 'máximo local' if all(e_ < 0 for e_ in ev)
                    else 'punto silla' if any(e_ > 0 for e_ in ev) and any(e_ < 0 for e_ in ev) else 'no concluyente')
        fp = sp.simplify(f.subs(dict(zip(vs, p))))
        puntos.append(p)
        tipos.append(tipo)
        filas.append(['(' + ', '.join(fmt_num(c, 5) for c in p) + ')', fmt_num(D, 6), fmt_num(fxx, 6), tipo, fmt_num(fp, 6)])
        r.paso('Punto (' + ', '.join(tex(c) for c in p) + ')', '',
               r'D = ' + tex(D) + r',\ f_{xx} = ' + tex(fxx) + r'\Rightarrow\ \text{' + tipo + '}', 1)
    r.tabla = {'columnas': ['Punto', 'D = det H', 'f_xx', 'Tipo', 'f'], 'filas': filas}
    r.valor = [list(p) if len(p) > 1 else p[0] for p in puntos]
    r.valores['tipos'] = ', '.join(tipos)
    r.resultado_latex = r'\\'.join('(' + ', '.join(tex(c) for c in p) + r')\ \text{' + t_ + '}' for p, t_ in zip(puntos, tipos)) \
        or r'\text{Sin puntos críticos}'
    r.comparar = ('conjunto', [sp.Tuple(*p) for p in puntos], {})
    if len(vs) == 2:
        try:
            r.grafico = G.grafico_3d(f, tuple(vs), tex(f))
        except Exception:  # noqa
            pass
    return r


def lagrange(arg):
    partes = dividir(arg, ';')
    if len(partes) < 2:
        raise ErrorEntrada('Uso: lagrange f; g = c   (por ejemplo: lagrange x*y; x+y=10)')
    f = leer(partes[0])
    restr = []
    for p in partes[1:]:
        e = leer_ecuacion(p)
        restr.append(e.lhs - e.rhs if isinstance(e, sp.Equality) else e)
    vs = _vars([f] + restr)
    lams = sp.symbols(f'lamda1:{len(restr) + 1}') if len(restr) > 1 else [sp.Symbol('lamda')]
    r = _r('Multiplicadores de Lagrange')
    r.entrada_latex = r'\text{optimizar } ' + tex(f) + r'\ \text{sujeto a } ' + ', '.join(tex(g) + ' = 0' for g in restr)
    L = f - sum(l_ * g for l_, g in zip(lams, restr))
    r.paso('Función de Lagrange', '', r'\mathcal{L} = ' + tex(L), 0)
    eqs = [sp.diff(L, v) for v in vs] + restr
    r.paso('Sistema', r'\(\nabla f = \lambda \nabla g\) y la restricción.', r'\begin{cases}' + r'\\'.join(tex(e) + ' = 0' for e in eqs) + r'\end{cases}', 0)
    sols = sp.solve(eqs, list(vs) + list(lams), dict=True)
    sols = [s_ for s_ in sols if all(s_.get(v, 0).is_real is not False for v in vs)]
    puntos, valores_f, filas = [], [], []
    for s_ in sols:
        p = tuple(sp.simplify(s_.get(v, 0)) for v in vs)
        fv = sp.simplify(f.subs(dict(zip(vs, p))))
        puntos.append(p)
        valores_f.append(fv)
        filas.append(['(' + ', '.join(fmt_num(c, 6) for c in p) + ')', fmt_num(fv, 8)])
    r.tabla = {'columnas': ['Punto', 'f'], 'filas': filas}
    if valores_f:
        try:
            nums = [float(sp.N(v)) for v in valores_f]
            imax, imin = nums.index(max(nums)), nums.index(min(nums))
            r.valores['maximo'] = valores_f[imax]
            r.valores['minimo'] = valores_f[imin]
            r.paso('Comparar valores', 'Se evalúa f en cada candidato.', r'f_{max} = ' + tex(valores_f[imax]) + r',\ f_{min} = ' + tex(valores_f[imin]), 0)
        except Exception:  # noqa
            pass
    r.valor = [list(p) for p in puntos]
    r.resultado_latex = r'\text{Candidatos: }' + ', '.join('(' + ', '.join(tex(c) for c in p) + ')' for p in puntos)
    r.comparar = ('conjunto', [sp.Tuple(*p) for p in puntos], {})
    if len(vs) == 2 and len(restr) == 1:
        try:
            g = G.grafico_implicita(restr[0], tuple(vs), 'restricción')
            g['puntos'] = [G.punto(p[0], p[1], f'f = {fmt_num(fv, 5)}') for p, fv in zip(puntos, valores_f)]
            g['puntos'] = [p for p in g['puntos'] if p]
            r.grafico = g
        except Exception:  # noqa
            pass
    return r
