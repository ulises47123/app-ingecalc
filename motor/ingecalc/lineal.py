"""Álgebra lineal y vectores (Álgebra y Geometría Analítica)."""
import sympy as sp

from . import graficos as G
from .nucleo import ErrorEntrada, Resultado, dividir, leer, leer_lista, leer_matriz, tex, fmt_num, decimal


def _fila(i):
    return f'F_{{{i + 1}}}'


def gauss_jordan_pasos(M, aumentada=False, max_pasos=40):
    """Gauss-Jordan registrando cada operación elemental de filas."""
    A = sp.Matrix(M)
    filas, cols = A.shape
    cols_piv = cols - 1 if aumentada else cols
    pasos = []
    fila_piv = 0
    for c in range(cols_piv):
        if fila_piv >= filas:
            break
        # elegir pivote no nulo
        p = None
        for i in range(fila_piv, filas):
            if sp.simplify(A[i, c]) != 0:
                p = i
                break
        if p is None:
            continue
        if p != fila_piv:
            A.row_swap(p, fila_piv)
            pasos.append((f'{_fila(fila_piv)} \\leftrightarrow {_fila(p)}', A.copy()))
        piv = A[fila_piv, c]
        if piv != 1:
            A[fila_piv, :] = (A[fila_piv, :] / piv).applyfunc(sp.simplify)
            pasos.append((f'{_fila(fila_piv)} \\to \\frac{{1}}{{{tex(piv)}}}{_fila(fila_piv)}', A.copy()))
        ops = []
        for i in range(filas):
            if i != fila_piv and sp.simplify(A[i, c]) != 0:
                factor = A[i, c]
                A[i, :] = (A[i, :] - factor * A[fila_piv, :]).applyfunc(sp.simplify)
                signo = '-' if not str(factor).startswith('-') else '+'
                fac = tex(factor) if signo == '-' else tex(-factor)
                ops.append(f'{_fila(i)} \\to {_fila(i)} {signo} {fac if fac != "1" else ""}{_fila(fila_piv)}')
        if ops:
            pasos.append((r',\ '.join(ops), A.copy()))
        fila_piv += 1
    lista = []
    for op, mat in pasos[:max_pasos]:
        lista.append({'titulo': 'Operación de filas', 'texto': '', 'latex': op + r'\quad ' + tex(mat), 'nivel': 1})
    if len(pasos) > max_pasos:
        lista.append({'titulo': '…', 'texto': f'{len(pasos) - max_pasos} operaciones más.', 'latex': '', 'nivel': 1})
    return A, lista


def _r(titulo):
    return Resultado('Álgebra lineal', titulo)


def det(arg):
    A = leer_matriz(arg)
    if not A.is_square:
        raise ErrorEntrada('El determinante solo existe para matrices cuadradas.')
    r = _r('Determinante')
    r.entrada_latex = r'\det ' + tex(A)
    n_ = A.shape[0]
    d = sp.simplify(A.det())
    if n_ == 2:
        a, b, c, e = A
        r.paso('Matriz 2×2', r'\(\det = a\,d - b\,c\)', f'({tex(a)})({tex(e)}) - ({tex(b)})({tex(c)}) = ' + tex(d), 0)
    elif n_ == 3:
        a = A
        pos = [a[0, 0] * a[1, 1] * a[2, 2], a[0, 1] * a[1, 2] * a[2, 0], a[0, 2] * a[1, 0] * a[2, 1]]
        neg = [a[0, 2] * a[1, 1] * a[2, 0], a[0, 0] * a[1, 2] * a[2, 1], a[0, 1] * a[1, 0] * a[2, 2]]
        r.paso('Regla de Sarrus', 'Suma de los productos de las diagonales principales menos las secundarias.',
               '(' + ' + '.join(tex(p) for p in pos) + ') - (' + ' + '.join(tex(p) for p in neg) + ') = ' + tex(d), 0)
        r.paso('Desarrollo por la primera fila (cofactores)', '',
               ' + '.join(f'({tex(a[0, j] * (-1) ** j)})' + r'\det' + tex(a.minor_submatrix(0, j)) for j in range(3)) + ' = ' + tex(d), 0)
    else:
        # cofactores por la fila con más ceros
        fila = max(range(n_), key=lambda i: sum(1 for j in range(n_) if A[i, j] == 0))
        terminos = []
        for j in range(n_):
            if A[fila, j] != 0:
                cof = (-1) ** (fila + j) * A.minor_submatrix(fila, j).det()
                terminos.append(f'({tex(A[fila, j])})({tex(sp.simplify(cof))})')
        r.paso(f'Desarrollo por cofactores (fila {fila + 1})', r'\(\det A = \sum_j a_{ij}\,C_{ij}\), eligiendo la fila con más ceros.',
               ' + '.join(terminos) + ' = ' + tex(d), 0)
    r.valor = d
    r.resultado_latex = r'\det A = ' + tex(d)
    r.extra('Interpretación', '', 'La matriz es invertible (det ≠ 0).' if d != 0 else 'La matriz es singular (det = 0): no tiene inversa.')
    r.comparar = ('numero_o_expr', d, {})
    return r


def inversa(arg):
    A = leer_matriz(arg)
    if not A.is_square:
        raise ErrorEntrada('Solo las matrices cuadradas pueden tener inversa.')
    r = _r('Matriz inversa')
    r.entrada_latex = tex(A) + '^{-1}'
    d = A.det()
    r.paso('Determinante', 'Si det = 0 no hay inversa.', r'\det A = ' + tex(sp.simplify(d)), 0)
    if sp.simplify(d) == 0:
        r.valor = 'no existe'
        r.resultado_latex = r'\text{No existe: } \det A = 0'
        r.comparar = ('texto', 'no existe', {})
        return r
    n_ = A.shape[0]
    if n_ == 2:
        a, b, c, e = A
        r.paso('Fórmula 2×2', '', r'A^{-1} = \frac{1}{\det A}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}', 0)
    aum = A.row_join(sp.eye(n_))
    r.paso('Gauss-Jordan sobre [A | I]', 'Se lleva A a la identidad; a la derecha queda la inversa.', tex(aum), 0)
    red, pasos = gauss_jordan_pasos(aum)
    r.pasos.extend(pasos)
    inv = A.inv().applyfunc(sp.simplify)
    r.paso('Comprobación', r'\(A\,A^{-1} = I\)', tex(A) + tex(inv) + ' = ' + tex((A * inv).applyfunc(sp.simplify)), 0)
    r.valor = inv
    r.resultado_latex = 'A^{-1} = ' + tex(inv)
    r.comparar = ('matriz', inv, {})
    return r


def transpuesta(arg):
    A = leer_matriz(arg)
    r = _r('Transpuesta')
    r.entrada_latex = tex(A) + '^T'
    r.paso('Filas por columnas', 'La fila i pasa a ser la columna i.', '', 0)
    r.valor = A.T
    r.resultado_latex = 'A^T = ' + tex(A.T)
    r.comparar = ('matriz', A.T, {})
    return r


def traza(arg):
    A = leer_matriz(arg)
    r = _r('Traza')
    r.entrada_latex = r'\operatorname{tr}' + tex(A)
    tr = A.trace()
    r.paso('Suma de la diagonal', '', ' + '.join(tex(A[i, i]) for i in range(min(A.shape))) + ' = ' + tex(tr), 0)
    r.valor = tr
    r.resultado_latex = r'\operatorname{tr}A = ' + tex(tr)
    r.comparar = ('numero_o_expr', tr, {})
    return r


def rango(arg):
    A = leer_matriz(arg)
    r = _r('Rango')
    r.entrada_latex = r'\operatorname{rg}' + tex(A)
    red, pasos = gauss_jordan_pasos(A)
    r.paso('Escalonar', 'El rango es la cantidad de filas no nulas de la forma escalonada.', '', 0)
    r.pasos.extend(pasos)
    rg = A.rank()
    r.valor = rg
    r.resultado_latex = r'\operatorname{rg}A = ' + str(rg)
    r.comparar = ('numero_o_expr', sp.Integer(rg), {})
    return r


def gauss(arg):
    M = leer_matriz(arg)
    r = _r('Eliminación de Gauss-Jordan')
    r.entrada_latex = tex(M)
    red, pasos = gauss_jordan_pasos(M, aumentada=True)
    r.pasos.extend(pasos)
    rref = M.rref()[0]
    r.valor = rref
    r.resultado_latex = r'\text{Forma escalonada reducida: } ' + tex(rref)
    # interpretar como sistema ampliado
    A, b = M[:, :-1], M[:, -1]
    ra, rm = A.rank(), M.rank()
    nvars = A.shape[1]
    if ra != rm:
        r.extra('Como sistema [A | b]', '', 'Incompatible: rango(A) ≠ rango(A|b).')
        r.valores['solucion'] = 'incompatible'
    else:
        incog = sp.symbols(f'x1:{nvars + 1}')
        sol = sp.linsolve((A, b), *incog)
        r.extra('Como sistema [A | b]', tex(sol), 'Compatible ' + ('determinado' if ra == nvars else 'indeterminado') + '.')
        r.valores['solucion'] = str(list(sol)[0]) if sol else ''
    r.comparar = ('matriz', rref, {})
    return r


def autovalores(arg, con_vectores=False):
    A = leer_matriz(arg)
    if not A.is_square:
        raise ErrorEntrada('Los autovalores se calculan para matrices cuadradas.')
    lam = sp.Symbol('lamda')
    r = _r('Autovalores y autovectores' if con_vectores else 'Autovalores')
    r.entrada_latex = 'A = ' + tex(A)
    pc = sp.factor((A - lam * sp.eye(A.shape[0])).det())
    r.paso('Polinomio característico', r'\(p(\lambda) = \det(A - \lambda I)\)',
           r'\det' + tex(A - lam * sp.eye(A.shape[0])) + ' = ' + tex(sp.expand(pc)) + ' = ' + tex(pc), 0)
    ev = A.eigenvals()
    lista = []
    for val, mult in sorted(ev.items(), key=lambda kv: (complex(sp.N(kv[0])).real, complex(sp.N(kv[0])).imag)):
        lista += [sp.simplify(val)] * mult
    r.paso('Raíces', r'\(p(\lambda) = 0\)', r'\lambda = ' + tex(lista), 0)
    vecs = []
    for val, mult, base in A.eigenvects():
        val = sp.simplify(val)
        base = [sp.simplify(b_ / max([abs(c) for c in b_ if c.is_number and c != 0] + [1]) if False else b_) for b_ in base]
        r.paso(f'Autovectores de λ = {tex(val)}', r'Se resuelve \((A - \lambda I)\,v = 0\).',
               r'(A - ' + tex(val) + r'I) = ' + tex(A - val * sp.eye(A.shape[0])) + r'\Rightarrow v \in \operatorname{gen}\left\{'
               + ', '.join(tex(b_.T) for b_ in base) + r'\right\}', 1)
        vecs.append([val, mult, [list(b_) for b_ in base]])
        r.extra(f'λ = {tex(val)}', ', '.join(tex(b_) for b_ in base), f'multiplicidad algebraica {mult}, geométrica {len(base)}')
    diag = all(len(v[2]) == v[1] for v in vecs)
    r.extra('Diagonalización', '', 'A es diagonalizable.' if diag else 'A no es diagonalizable (falta algún autovector).')
    if con_vectores:
        r.valor = vecs
        r.resultado_latex = r'\\'.join(r'\lambda = ' + tex(v[0]) + r':\ ' + ', '.join(tex(sp.Matrix(b_)) for b_ in v[2]) for v in vecs)
        r.comparar = ('autovectores', vecs, {'A': A})
    else:
        r.valor = lista
        r.resultado_latex = r'\lambda = ' + tex(lista)
        r.comparar = ('conjunto', lista, {'multiplicidad': True})
    r.valores['polinomio'] = sp.expand(pc)
    return r


def operar_matrices(e, texto):
    r = _r('Operación con matrices')
    r.entrada_latex = tex(e) if not isinstance(e, sp.MatrixBase) else texto
    res = e
    if isinstance(res, sp.MatrixBase):
        res = res.applyfunc(sp.simplify)
    r.paso('Resultado', 'Producto fila por columna; suma elemento a elemento.', tex(res), 0)
    r.valor = res
    r.resultado_latex = tex(res)
    r.comparar = ('matriz', res, {})
    return r


def resumen_matriz(A):
    r = _r('Matriz')
    r.entrada_latex = 'A = ' + tex(A)
    r.valor = A
    r.resultado_latex = 'A = ' + tex(A)
    r.extra('Dimensión', f'{A.shape[0]}\\times {A.shape[1]}')
    r.extra('Rango', str(A.rank()))
    if A.is_square:
        d = sp.simplify(A.det())
        r.extra('Determinante', tex(d))
        r.extra('Traza', tex(A.trace()))
        if d != 0 and A.shape[0] <= 4:
            r.extra('Inversa', tex(A.inv().applyfunc(sp.simplify)))
        if A.shape[0] <= 4:
            try:
                r.extra('Autovalores', tex(list(A.eigenvals().keys())))
            except Exception:  # noqa
                pass
    r.extra('Transpuesta', tex(A.T))
    r.comparar = ('matriz', A, {})
    return r


def _vectores(arg, cantidad):
    partes = dividir(arg)
    if len(partes) < cantidad:
        raise ErrorEntrada(f'Se esperaban {cantidad} vector(es), por ejemplo [1,2,3], [4,5,6].')
    return [sp.Matrix(leer_lista(p)) for p in partes[:cantidad]]


def producto_punto(arg):
    u, v = _vectores(arg, 2)
    r = Resultado('Vectores', 'Producto escalar')
    r.entrada_latex = tex(u.T) + r'\cdot ' + tex(v.T)
    val = u.dot(v)
    r.paso('Suma de productos componente a componente', '',
           ' + '.join(f'({tex(a)})({tex(b)})' for a, b in zip(u, v)) + ' = ' + tex(val), 0)
    r.valor = sp.simplify(val)
    r.resultado_latex = r'u\cdot v = ' + tex(val)
    if val == 0:
        r.extra('Conclusión', '', 'Los vectores son ortogonales.')
    r.comparar = ('numero_o_expr', val, {})
    return r


def producto_cruz(arg):
    u, v = _vectores(arg, 2)
    if len(u) == 2:
        u, v = sp.Matrix([*u, 0]), sp.Matrix([*v, 0])
    r = Resultado('Vectores', 'Producto vectorial')
    r.entrada_latex = tex(u.T) + r'\times ' + tex(v.T)
    w = u.cross(v)
    r.paso('Determinante simbólico', '', r'\begin{vmatrix} \hat i & \hat j & \hat k \\ ' + ' & '.join(tex(a) for a in u)
           + r' \\ ' + ' & '.join(tex(a) for a in v) + r'\end{vmatrix}', 0)
    r.paso('Componentes', '', r'\hat i(' + tex(u[1] * v[2] - u[2] * v[1]) + r') - \hat j(' + tex(u[0] * v[2] - u[2] * v[0])
           + r') + \hat k(' + tex(u[0] * v[1] - u[1] * v[0]) + ')', 0)
    r.valor = list(w)
    r.resultado_latex = r'u\times v = ' + tex(w.T)
    r.extra('Área del paralelogramo', r'|u\times v| = ' + tex(sp.sqrt(w.dot(w))))
    r.comparar = ('vector', list(w), {})
    return r


def norma(arg):
    u = sp.Matrix(leer_lista(dividir(arg)[0]))
    r = Resultado('Vectores', 'Norma')
    r.entrada_latex = r'\left\|' + tex(u.T) + r'\right\|'
    val = sp.sqrt(sum(a ** 2 for a in u))
    r.paso('Raíz de la suma de cuadrados', '', r'\sqrt{' + ' + '.join(f'({tex(a)})^2' for a in u) + '} = ' + tex(val), 0)
    r.valor = sp.simplify(val)
    r.valores['decimal'] = fmt_num(decimal(val), 12)
    r.resultado_latex = r.entrada_latex + ' = ' + tex(val)
    r.extra('Versor', tex((u / val).T))
    r.comparar = ('numero_o_expr', val, {})
    return r


def angulo(arg):
    u, v = _vectores(arg, 2)
    r = Resultado('Vectores', 'Ángulo entre vectores')
    r.entrada_latex = r'\theta(' + tex(u.T) + ', ' + tex(v.T) + ')'
    pu, nu, nv = u.dot(v), sp.sqrt(u.dot(u)), sp.sqrt(v.dot(v))
    c = sp.simplify(pu / (nu * nv))
    th = sp.simplify(sp.acos(c))
    r.paso('Fórmula', r'\(\cos\theta = \frac{u\cdot v}{\|u\|\,\|v\|}\)',
           r'\cos\theta = \frac{' + tex(pu) + '}{' + tex(nu) + r'\cdot ' + tex(nv) + '} = ' + tex(c), 0)
    r.paso('Ángulo', '', r'\theta = \arccos\left(' + tex(c) + r'\right) = ' + tex(th) + r'\approx '
           + fmt_num(sp.N(th * 180 / sp.pi), 8) + r'^\circ', 0)
    r.valor = th
    r.valores['grados'] = fmt_num(sp.N(th * 180 / sp.pi), 12)
    r.resultado_latex = r'\theta = ' + tex(th) + r'\approx ' + fmt_num(sp.N(th * 180 / sp.pi), 8) + r'^\circ'
    r.comparar = ('numero_o_expr', th, {})
    if len(u) == 2:
        r.grafico = G.grafico_vectores([(list(u), 'u'), (list(v), 'v')])
    return r


def proyeccion(arg):
    u, v = _vectores(arg, 2)
    r = Resultado('Vectores', 'Proyección')
    r.entrada_latex = r'\operatorname{proy}_{' + tex(v.T) + '}' + tex(u.T)
    esc = sp.simplify(u.dot(v) / v.dot(v))
    p = (esc * v).applyfunc(sp.simplify)
    r.paso('Fórmula', r'\(\operatorname{proy}_v u = \frac{u\cdot v}{v\cdot v}\,v\)',
           r'\frac{' + tex(u.dot(v)) + '}{' + tex(v.dot(v)) + '}' + tex(v.T) + ' = ' + tex(p.T), 0)
    r.valor = list(p)
    r.valores['escalar'] = sp.simplify(u.dot(v) / sp.sqrt(v.dot(v)))
    r.resultado_latex = r.entrada_latex + ' = ' + tex(p.T)
    r.comparar = ('vector', list(p), {})
    if len(u) == 2:
        r.grafico = G.grafico_vectores([(list(u), 'u'), (list(v), 'v'), (list(p), 'proy')])
    return r
