"""Derivación con explicación paso a paso (reglas de la cursada)."""
import sympy as sp

from .nucleo import tex

# Derivada de f(u) respecto de u, con el nombre de la regla
_TABLA = {
    sp.sin: (lambda u: sp.cos(u), r'\frac{d}{du}\sin u = \cos u'),
    sp.cos: (lambda u: -sp.sin(u), r'\frac{d}{du}\cos u = -\sin u'),
    sp.tan: (lambda u: sp.sec(u) ** 2, r'\frac{d}{du}\tan u = \sec^2 u'),
    sp.cot: (lambda u: -sp.csc(u) ** 2, r'\frac{d}{du}\cot u = -\csc^2 u'),
    sp.sec: (lambda u: sp.sec(u) * sp.tan(u), r'\frac{d}{du}\sec u = \sec u\tan u'),
    sp.csc: (lambda u: -sp.csc(u) * sp.cot(u), r'\frac{d}{du}\csc u = -\csc u\cot u'),
    sp.exp: (lambda u: sp.exp(u), r'\frac{d}{du}e^u = e^u'),
    sp.log: (lambda u: 1 / u, r'\frac{d}{du}\ln u = \frac{1}{u}'),
    sp.asin: (lambda u: 1 / sp.sqrt(1 - u ** 2), r'\frac{d}{du}\arcsin u = \frac{1}{\sqrt{1-u^2}}'),
    sp.acos: (lambda u: -1 / sp.sqrt(1 - u ** 2), r'\frac{d}{du}\arccos u = -\frac{1}{\sqrt{1-u^2}}'),
    sp.atan: (lambda u: 1 / (1 + u ** 2), r'\frac{d}{du}\arctan u = \frac{1}{1+u^2}'),
    sp.acot: (lambda u: -1 / (1 + u ** 2), r'\frac{d}{du}\operatorname{arccot} u = -\frac{1}{1+u^2}'),
    sp.sinh: (lambda u: sp.cosh(u), r'\frac{d}{du}\sinh u = \cosh u'),
    sp.cosh: (lambda u: sp.sinh(u), r'\frac{d}{du}\cosh u = \sinh u'),
    sp.tanh: (lambda u: 1 - sp.tanh(u) ** 2, r'\frac{d}{du}\tanh u = 1-\tanh^2 u'),
    sp.asinh: (lambda u: 1 / sp.sqrt(u ** 2 + 1), r'\frac{d}{du}\operatorname{arcsinh} u = \frac{1}{\sqrt{u^2+1}}'),
    sp.acosh: (lambda u: 1 / sp.sqrt(u ** 2 - 1), r'\frac{d}{du}\operatorname{arccosh} u = \frac{1}{\sqrt{u^2-1}}'),
    sp.atanh: (lambda u: 1 / (1 - u ** 2), r'\frac{d}{du}\operatorname{arctanh} u = \frac{1}{1-u^2}'),
    sp.Abs: (lambda u: sp.sign(u), r'\frac{d}{du}|u| = \frac{u}{|u|}'),
}


def _d(e, v):
    return r'\frac{d}{d' + tex(v) + r'}\left[' + tex(e) + r'\right]'


class Derivador:
    def __init__(self, v, max_pasos=60):
        self.v = v
        self.pasos = []
        self.max_pasos = max_pasos

    def _paso(self, idx, titulo, texto, latex, nivel):
        # El paso del término "padre" va antes que los de sus partes (lugar reservado)
        if idx < self.max_pasos:
            self.pasos[idx] = {'titulo': titulo, 'texto': texto, 'latex': latex, 'nivel': nivel}

    def derivar(self, e, nivel=0):
        v = self.v
        idx = len(self.pasos)
        self.pasos.append(None)
        if not e.has(v):
            if nivel == 0 or e.free_symbols:
                self._paso(idx, 'Constante', 'La derivada de una constante es 0.', _d(e, v) + ' = 0', nivel)
            return sp.Integer(0)
        if e == v:
            return sp.Integer(1)

        # c·f(x)
        if e.is_Mul:
            const, resto = e.as_independent(v)
            if const != 1:
                d_resto = self.derivar_explicado(resto, nivel + 1, mostrar=False)
                r = const * d_resto
                self._paso(idx, 'Constante por función', 'La constante sale fuera de la derivada.',
                           _d(e, v) + ' = ' + tex(const) + r'\cdot ' + _d(resto, v) + ' = ' + tex(r), nivel)
                return r

        # suma
        if e.is_Add:
            terminos = e.as_ordered_terms()
            self._paso(idx, 'Regla de la suma', 'Se deriva cada término por separado.',
                       _d(e, v) + ' = ' + ' + '.join(_d(tm, v) for tm in terminos), nivel)
            partes = [self.derivar_explicado(tm, nivel + 1) for tm in terminos]
            return sp.Add(*partes)

        # potencia
        if e.is_Pow:
            b, p = e.as_base_exp()
            if not p.has(v):  # u^n
                if b == v:
                    r = p * v ** (p - 1)
                    self._paso(idx, 'Regla de la potencia', r'\(\frac{d}{dx}x^n = n\,x^{n-1}\)',
                               _d(e, v) + ' = ' + tex(p) + r'\cdot ' + tex(v ** (p - 1)) + ' = ' + tex(r), nivel)
                    return r
                db = self.derivar_explicado(b, nivel + 1, mostrar=False)
                r = p * b ** (p - 1) * db
                nombre = 'Raíz y regla de la cadena' if p in (sp.Rational(1, 2), sp.Rational(-1, 2)) else 'Potencia y regla de la cadena'
                self._paso(idx, nombre, r'Con \(u = ' + tex(b) + r'\): \((u^n)\' = n\,u^{n-1}\,u\'\).',
                           _d(e, v) + ' = ' + tex(p) + r'\left(' + tex(b) + r'\right)^{' + tex(p - 1) + r'}\cdot '
                           + r'\left(' + tex(db) + r'\right) = ' + tex(sp.simplify(r) if sp.count_ops(r) > 25 else r), nivel)
                return r
            if not b.has(v):  # a^u
                dp = self.derivar_explicado(p, nivel + 1, mostrar=False)
                r = e * sp.log(b) * dp if b != sp.E else e * dp
                regla = r'\((e^u)\' = e^u\,u\'\)' if b == sp.E else r'\((a^u)\' = a^u \ln a\; u\'\)'
                self._paso(idx, 'Exponencial', regla + (' (regla de la cadena)' if p != v else ''),
                           _d(e, v) + ' = ' + tex(r), nivel)
                return r
            # u^w: derivación logarítmica
            lnf = p * sp.log(b)
            dl = self.derivar_explicado(lnf, nivel + 1, mostrar=False)
            r = e * dl
            self._paso(idx, 'Derivación logarítmica',
                       r'Como base y exponente dependen de la variable: \(\ln y = ' + tex(lnf)
                       + r'\Rightarrow \frac{y\'}{y} = ' + tex(dl) + r'\).',
                       _d(e, v) + ' = ' + tex(e) + r'\left(' + tex(dl) + r'\right)', nivel)
            return r

        # cociente (numerador/denominador)
        num, den = sp.fraction(sp.together(e)) if e.is_Mul else (e, sp.Integer(1))
        if e.is_Mul and den != 1 and den.has(v):
            du = self.derivar_explicado(num, nivel + 1, mostrar=False)
            dv = self.derivar_explicado(den, nivel + 1, mostrar=False)
            r = (du * den - num * dv) / den ** 2
            self._paso(idx, 'Regla del cociente', r'\(\left(\frac{u}{v}\right)\' = \frac{u\'v - u v\'}{v^2}\) con \(u = '
                       + tex(num) + r'\), \(v = ' + tex(den) + r'\), \(u\' = ' + tex(du) + r'\), \(v\' = ' + tex(dv) + r'\).',
                       _d(e, v) + r' = \frac{\left(' + tex(du) + r'\right)\left(' + tex(den) + r'\right) - \left('
                       + tex(num) + r'\right)\left(' + tex(dv) + r'\right)}{\left(' + tex(den) + r'\right)^2}', nivel)
            return r

        # producto
        if e.is_Mul:
            factores = [f for f in e.args]
            if len(factores) > 2:
                u = factores[0]
                w = sp.Mul(*factores[1:])
            else:
                u, w = factores
            du = self.derivar_explicado(u, nivel + 1, mostrar=False)
            dw = self.derivar_explicado(w, nivel + 1, mostrar=False)
            r = du * w + u * dw
            self._paso(idx, 'Regla del producto', r'\((u\,v)\' = u\'v + u\,v\'\) con \(u = ' + tex(u) + r'\), \(v = '
                       + tex(w) + r'\), \(u\' = ' + tex(du) + r'\), \(v\' = ' + tex(dw) + r'\).',
                       _d(e, v) + ' = ' + r'\left(' + tex(du) + r'\right)\left(' + tex(w) + r'\right) + \left('
                       + tex(u) + r'\right)\left(' + tex(dw) + r'\right)', nivel)
            return r

        # funciones elementales con regla de la cadena
        f = e.func
        if f in _TABLA and len(e.args) == 1:
            u = e.args[0]
            fprima, regla = _TABLA[f]
            if u == v:
                r = fprima(v)
                self._paso(idx, 'Derivada inmediata', 'Tabla de derivadas: \(' + regla + '\).',
                           _d(e, v) + ' = ' + tex(r), nivel)
                return r
            du = self.derivar_explicado(u, nivel + 1, mostrar=False)
            r = fprima(u) * du
            self._paso(idx, 'Regla de la cadena', r'Función exterior con \(u = ' + tex(u) + r'\): \(' + regla
                       + r'\), y se multiplica por \(u\' = ' + tex(du) + r'\).',
                       _d(e, v) + ' = ' + tex(fprima(u)) + r'\cdot ' + r'\left(' + tex(du) + r'\right) = ' + tex(r), nivel)
            return r
        if f == sp.log and len(e.args) == 2:
            return self.derivar(sp.log(e.args[0]) / sp.log(e.args[1]), nivel)

        # Lo que no tiene regla propia: SymPy directamente
        r = sp.diff(e, v)
        self._paso(idx, 'Derivada', 'Se aplica la definición/tabla correspondiente.', _d(e, v) + ' = ' + tex(r), nivel)
        return r

    def derivar_explicado(self, e, nivel, mostrar=True):
        """Deriva un subtérmino; si no es trivial lo explica en un subnivel."""
        if not e.has(self.v):
            return sp.Integer(0)
        if e == self.v:
            return sp.Integer(1)
        return self.derivar(e, nivel)


def derivar_con_pasos(expr, v):
    d = Derivador(v)
    crudo = d.derivar(expr)
    return crudo, [p for p in d.pasos if p]
