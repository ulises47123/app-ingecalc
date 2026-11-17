"""Matemática del ingreso y de taller: unidades (SIMELA y técnicas), regla de tres, geometría."""
import re

import sympy as sp

from .nucleo import ErrorEntrada, Resultado, coma_decimal, dividir, leer, normalizar, sin_tildes, tex, fmt_num, decimal

# ----------------------------------------------------------------------------
# Unidades: factor a la unidad base de cada magnitud
# ----------------------------------------------------------------------------
_R = sp.Rational
UNIDADES = {
    'longitud': {'km': 1000, 'hm': 100, 'dam': 10, 'm': 1, 'dm': _R(1, 10), 'cm': _R(1, 100), 'mm': _R(1, 1000),
                 'um': _R(1, 10 ** 6), 'nm': _R(1, 10 ** 9), 'in': _R(254, 10000), 'pulg': _R(254, 10000),
                 'ft': _R(3048, 10000), 'pie': _R(3048, 10000), 'yd': _R(9144, 10000), 'mi': _R(1609344, 1000),
                 'milla': _R(1609344, 1000)},
    'superficie': {'km2': 10 ** 6, 'hm2': 10 ** 4, 'ha': 10 ** 4, 'dam2': 100, 'a': 100, 'm2': 1, 'ca': 1,
                   'dm2': _R(1, 100), 'cm2': _R(1, 10 ** 4), 'mm2': _R(1, 10 ** 6)},
    'volumen': {'km3': 10 ** 9, 'hm3': 10 ** 6, 'dam3': 10 ** 3, 'm3': 1, 'dm3': _R(1, 10 ** 3), 'cm3': _R(1, 10 ** 6),
                'mm3': _R(1, 10 ** 9), 'kl': 1, 'hl': _R(1, 10), 'dal': _R(1, 100), 'l': _R(1, 1000),
                'dl': _R(1, 10 ** 4), 'cl': _R(1, 10 ** 5), 'ml': _R(1, 10 ** 6), 'cc': _R(1, 10 ** 6),
                'gal': _R(3785411784, 10 ** 12)},
    'masa': {'t': 1000, 'tn': 1000, 'q': 100, 'kg': 1, 'hg': _R(1, 10), 'dag': _R(1, 100), 'g': _R(1, 1000),
             'dg': _R(1, 10 ** 4), 'cg': _R(1, 10 ** 5), 'mg': _R(1, 10 ** 6), 'lb': _R(45359237, 10 ** 8),
             'oz': _R(28349523125, 10 ** 12)},
    'tiempo': {'semana': 604800, 'semanas': 604800, 'dia': 86400, 'dias': 86400, 'd': 86400, 'h': 3600, 'hora': 3600,
               'horas': 3600, 'min': 60, 's': 1, 'seg': 1, 'ms': _R(1, 1000)},
    'velocidad': {'km/h': _R(1000, 3600), 'm/s': 1, 'm/min': _R(1, 60), 'cm/s': _R(1, 100), 'mi/h': _R(1609344, 3600000),
                  'nudo': _R(1852, 3600)},
    'presion': {'pa': 1, 'kpa': 1000, 'mpa': 10 ** 6, 'bar': 10 ** 5, 'mbar': 100, 'atm': 101325, 'mmhg': _R(101325, 760),
                'psi': _R(689475729, 100000)},
    'energia': {'j': 1, 'kj': 1000, 'mj': 10 ** 6, 'cal': _R(4184, 1000), 'kcal': 4184, 'wh': 3600, 'kwh': 3600000},
    'potencia': {'w': 1, 'kw': 1000, 'mw': 10 ** 6, 'hp': _R(74569987, 100000), 'cv': _R(73549875, 100000)},
    'angulo': {'rad': 1, 'grados': sp.pi / 180, 'grado': sp.pi / 180, 'deg': sp.pi / 180, '°': sp.pi / 180,
               'rev': 2 * sp.pi, 'vuelta': 2 * sp.pi},
}
_ALIAS = {'m²': 'm2', 'cm²': 'cm2', 'mm²': 'mm2', 'km²': 'km2', 'dm²': 'dm2', 'm³': 'm3', 'cm³': 'cm3', 'dm³': 'dm3',
          'mm³': 'mm3', 'litro': 'l', 'litros': 'l', 'metro': 'm', 'metros': 'm', 'kilometro': 'km', 'kilometros': 'km',
          'centimetro': 'cm', 'centimetros': 'cm', 'milimetro': 'mm', 'milimetros': 'mm', 'gramo': 'g',
          'gramos': 'g', 'kilo': 'kg', 'kilos': 'kg', 'kilogramo': 'kg', 'kilogramos': 'kg', 'tonelada': 't',
          'toneladas': 't', 'segundo': 's', 'segundos': 's', 'minuto': 'min', 'minutos': 'min', 'hs': 'h',
          'hectarea': 'ha', 'hectareas': 'ha', 'area': 'a', 'areas': 'a', 'kmh': 'km/h', 'km/hs': 'km/h',
          'mts': 'm', 'lts': 'l', 'lt': 'l', 'grs': 'g', 'gr': 'g', 'radianes': 'rad'}
_TEMP = {'c', '°c', 'celsius', 'f', '°f', 'fahrenheit', 'k', 'kelvin'}


def _unidad(u):
    u = sin_tildes(u.strip().lower()).replace('^2', '2').replace('^3', '3').replace(' ', '')
    u = _ALIAS.get(u, u)
    for mag, tabla in UNIDADES.items():
        if u in tabla:
            return mag, u, sp.nsimplify(tabla[u])
    return None, u, None


def convertir(arg):
    a = coma_decimal(normalizar(arg)).strip()
    m = re.match(r'^(.+?)\s*([a-zA-Z°²³/µ]+[a-zA-Z0-9°²³/^]*)\s+(?:a|en|->|=>|to)\s+([a-zA-Z°²³/µ][a-zA-Z0-9°²³/^]*)\s*$', a)
    if not m:
        raise ErrorEntrada('Uso: convertir 9.1 hl a cl   (también km/h a m/s, °C a °F, m2 a ha, atm a kPa...)')
    valor = leer(m.group(1))
    u1, u2 = m.group(2), m.group(3)
    r = Resultado('Unidades', 'Conversión de unidades')
    k1, k2 = sin_tildes(u1.lower()), sin_tildes(u2.lower())
    if k1 in _TEMP and k2 in _TEMP:
        a_c = {'c': lambda v: v, 'f': lambda v: (v - 32) * sp.Rational(5, 9), 'k': lambda v: v - sp.Rational(27315, 100)}
        de_c = {'c': lambda v: v, 'f': lambda v: v * sp.Rational(9, 5) + 32, 'k': lambda v: v + sp.Rational(27315, 100)}
        c1, c2 = k1.strip('°')[0], k2.strip('°')[0]
        res = sp.nsimplify(de_c[c2](a_c[c1](valor)))
        formulas = {('c', 'f'): r'°F = °C\cdot\frac95 + 32', ('f', 'c'): r'°C = (°F - 32)\cdot\frac59',
                    ('c', 'k'): r'K = °C + 273.15', ('k', 'c'): r'°C = K - 273.15'}
        r.paso('Fórmula', '', formulas.get((c1, c2), r'\text{Se pasa por °C}'), 0)
    else:
        mag1, n1, f1 = _unidad(u1)
        mag2, n2, f2 = _unidad(u2)
        if f1 is None or f2 is None:
            raise ErrorEntrada(f'No conozco la unidad «{u1 if f1 is None else u2}».')
        if mag1 != mag2:
            raise ErrorEntrada(f'No se puede convertir {mag1} ({u1}) en {mag2} ({u2}).')
        factor = sp.nsimplify(f1 / f2)
        res = sp.nsimplify(valor * factor)
        if mag1 in ('longitud', 'superficie', 'volumen', 'masa') and factor.is_Rational and (factor.p == 1 or factor.q == 1):
            exp = sp.log(factor, 10)
            if exp.is_Integer:
                pasos_esc = {'longitud': 1, 'superficie': 2, 'volumen': 3, 'masa': 1}[mag1]
                r.paso('En el SIMELA', f'Cada escalón de unidad multiplica o divide por 10^{pasos_esc}. '
                       + ('Se pasa a una unidad menor: se multiplica.' if factor > 1 else 'Se pasa a una unidad mayor: se divide.'),
                       r'1\ \text{' + u1 + r'} = ' + tex(factor) + r'\ \text{' + u2 + '}', 0)
        r.paso('Factor de conversión', '', r'1\ \text{' + u1 + '} = ' + (tex(factor) if factor.is_Rational and len(str(factor)) < 14
                                                                       else fmt_num(sp.N(factor), 10)) + r'\ \text{' + u2 + '}', 0)
    r.paso('Multiplicar', '', tex(valor) + r'\ \text{' + u1 + r'} = ' + fmt_num(sp.N(res), 12) + r'\ \text{' + u2 + '}', 0)
    r.valor = res
    r.valores['decimal'] = fmt_num(sp.N(res), 12)
    r.entrada_latex = tex(valor) + r'\ \text{' + u1 + r'} \to \text{' + u2 + '}'
    r.resultado_latex = tex(valor) + r'\ \text{' + u1 + '} = ' + fmt_num(sp.N(res), 12) + r'\ \text{' + u2 + '}'
    r.comparar = ('numero_o_expr', res, {})
    return r


# ----------------------------------------------------------------------------
# Regla de tres
# ----------------------------------------------------------------------------
def regla_de_tres(arg, inversa=False):
    p = [leer(x_) for x_ in dividir(coma_decimal(arg))]
    if len(p) != 3:
        raise ErrorEntrada('Uso: regla de tres a, b, c   (si a → b, entonces c → ?)   ·   regla de tres inversa a, b, c')
    a, b, c = p
    r = Resultado('Proporcionalidad', 'Regla de tres ' + ('simple inversa' if inversa else 'simple directa'))
    r.entrada_latex = tex(a) + r'\ \to\ ' + tex(b) + r'\qquad ' + tex(c) + r'\ \to\ x'
    if inversa:
        res = sp.nsimplify(a * b / c)
        r.paso('Proporcionalidad inversa', 'Si una magnitud aumenta, la otra disminuye: el producto se mantiene.',
               tex(a) + r'\cdot ' + tex(b) + ' = ' + tex(c) + r'\cdot x \Rightarrow x = \frac{' + tex(a) + r'\cdot ' + tex(b) + '}{' + tex(c) + '}', 0)
    else:
        res = sp.nsimplify(b * c / a)
        r.paso('Proporcionalidad directa', 'Las magnitudes aumentan juntas: el cociente se mantiene.',
               r'\frac{' + tex(a) + '}{' + tex(b) + r'} = \frac{' + tex(c) + r'}{x} \Rightarrow x = \frac{' + tex(b) + r'\cdot ' + tex(c) + '}{' + tex(a) + '}', 0)
    r.paso('Resultado', '', 'x = ' + tex(res) + (r' \approx ' + fmt_num(sp.N(res), 10) if not res.is_Integer else ''), 0)
    r.valor = res
    r.valores['decimal'] = fmt_num(sp.N(res), 12)
    r.resultado_latex = 'x = ' + tex(res) + (r' \approx ' + fmt_num(sp.N(res), 10) if not res.is_Integer else '')
    r.comparar = ('numero_o_expr', res, {})
    return r


# ----------------------------------------------------------------------------
# Geometría
# ----------------------------------------------------------------------------
FIGURAS = {
    # nombre: (parámetros, área, perímetro, fórmula área, fórmula perímetro)
    'cuadrado': (['l'], lambda l: l ** 2, lambda l: 4 * l, 'A = l^2', 'P = 4l'),
    'rectangulo': (['b', 'h'], lambda b, h: b * h, lambda b, h: 2 * (b + h), r'A = b\cdot h', 'P = 2(b + h)'),
    'triangulo': (['b', 'h'], lambda b, h: b * h / 2, None, r'A = \frac{b\cdot h}{2}', ''),
    'circulo': (['r'], lambda r: sp.pi * r ** 2, lambda r: 2 * sp.pi * r, r'A = \pi r^2', r'P = 2\pi r'),
    'circunferencia': (['r'], lambda r: sp.pi * r ** 2, lambda r: 2 * sp.pi * r, r'A = \pi r^2', r'L = 2\pi r'),
    'trapecio': (['B', 'b', 'h'], lambda B, b, h: (B + b) * h / 2, None, r'A = \frac{(B + b)\,h}{2}', ''),
    'rombo': (['D', 'd'], lambda D, d: D * d / 2, lambda D, d: 4 * sp.sqrt((D / 2) ** 2 + (d / 2) ** 2),
              r'A = \frac{D\cdot d}{2}', r'P = 4\sqrt{(D/2)^2 + (d/2)^2}'),
    'paralelogramo': (['b', 'h'], lambda b, h: b * h, None, r'A = b\cdot h', ''),
    'sector': (['r', 'angulo'], lambda r, a: sp.pi * r ** 2 * a / 360, lambda r, a: 2 * r + 2 * sp.pi * r * a / 360,
               r'A = \frac{\pi r^2 \alpha}{360°}', r'P = 2r + \frac{2\pi r\alpha}{360°}'),
    'hexagono': (['l'], lambda l: 3 * sp.sqrt(3) * l ** 2 / 2, lambda l: 6 * l, r'A = \frac{3\sqrt3}{2} l^2', 'P = 6l'),
    'corona': (['R', 'r'], lambda R, r: sp.pi * (R ** 2 - r ** 2), lambda R, r: 2 * sp.pi * (R + r), r'A = \pi(R^2 - r^2)', r'P = 2\pi(R + r)'),
}
CUERPOS = {
    'cubo': (['a'], lambda a: a ** 3, lambda a: 6 * a ** 2, 'V = a^3', 'A_t = 6a^2'),
    'prisma': (['a', 'b', 'c'], lambda a, b, c: a * b * c, lambda a, b, c: 2 * (a * b + a * c + b * c), r'V = a\,b\,c', r'A_t = 2(ab + ac + bc)'),
    'ortoedro': (['a', 'b', 'c'], lambda a, b, c: a * b * c, lambda a, b, c: 2 * (a * b + a * c + b * c), r'V = a\,b\,c', r'A_t = 2(ab + ac + bc)'),
    'cilindro': (['r', 'h'], lambda r, h: sp.pi * r ** 2 * h, lambda r, h: 2 * sp.pi * r * (r + h), r'V = \pi r^2 h', r'A_t = 2\pi r(r + h)'),
    'cono': (['r', 'h'], lambda r, h: sp.pi * r ** 2 * h / 3, lambda r, h: sp.pi * r * (r + sp.sqrt(r ** 2 + h ** 2)),
             r'V = \frac{\pi r^2 h}{3}', r'A_t = \pi r\,(r + g),\ g = \sqrt{r^2 + h^2}'),
    'esfera': (['r'], lambda r: 4 * sp.pi * r ** 3 / 3, lambda r: 4 * sp.pi * r ** 2, r'V = \frac43\pi r^3', r'A = 4\pi r^2'),
    'piramide': (['Ab', 'h'], lambda Ab, h: Ab * h / 3, None, r'V = \frac{A_b\,h}{3}', ''),
}


def es_figura(arg):
    pal = sin_tildes(arg.strip().lower()).split(' ')[0] if arg.strip() else ''
    return pal in FIGURAS or pal in CUERPOS


def geometria(arg, que):
    a = sin_tildes(arg.strip().lower())
    nombre = a.split(' ')[0]
    resto = arg.strip()[len(nombre):]
    vals = [leer(re.sub(r'^[a-zA-Z]+\s*=\s*', '', v)) for v in dividir(coma_decimal(resto))] if resto.strip() else []
    tabla = FIGURAS if nombre in FIGURAS else CUERPOS
    params, f_a, f_p, form_a, form_p = tabla[nombre]
    if len(vals) != len(params):
        raise ErrorEntrada(f'Para {que} del {nombre} se necesitan: ' + ', '.join(params) + '.')
    r = Resultado('Geometría', f'{que.capitalize()} del {nombre}')
    r.entrada_latex = r'\text{' + nombre + '}:\ ' + ', '.join(f'{p} = {tex(v)}' for p, v in zip(params, vals))
    if que in ('area', 'superficie'):
        if nombre in CUERPOS:
            f, form = f_p, form_p
        else:
            f, form = f_a, form_a
        if nombre == 'triangulo' and False:
            pass
    elif que == 'volumen':
        if nombre not in CUERPOS:
            raise ErrorEntrada('El volumen se calcula para cuerpos: cubo, prisma, cilindro, cono, esfera, pirámide.')
        f, form = f_a, form_a
    else:  # perímetro
        f, form = f_p, form_p
        if f is None:
            raise ErrorEntrada(f'Para el perímetro del {nombre} sumá sus lados (no alcanza con base y altura).')
    res = sp.nsimplify(f(*vals))
    r.paso('Fórmula', '', form, 0)
    r.paso('Reemplazar', '', ', '.join(f'{p} = {tex(v)}' for p, v in zip(params, vals)) + r'\Rightarrow ' + tex(res)
           + (r' \approx ' + fmt_num(sp.N(res), 10) if not res.is_Rational else ''), 0)
    r.valor = res
    r.valores['decimal'] = fmt_num(sp.N(res), 12)
    r.resultado_latex = tex(res) + (r' \approx ' + fmt_num(sp.N(res), 10) if not res.is_Rational else '')
    r.extra('Con π = 3,14', fmt_num(sp.N(res.subs(sp.pi, sp.Rational(314, 100))), 10)) if res.has(sp.pi) else None
    r.comparar = ('numero_o_expr', res, {})
    return r
