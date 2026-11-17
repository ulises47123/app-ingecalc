# IngeCalc — sintaxis de entrada y contrato del motor

El motor recibe **una línea de texto** y devuelve un JSON. Esta es la especificación
que siguen la app, el motor (`motor/ingecalc`) y el banco de pruebas (`pruebas/banco_500.json`).

## Escritura de expresiones

| Escribir | Significa |
|---|---|
| `x^2`, `x**2` | potencia |
| `2x`, `3sen(x)`, `(x+1)(x-1)`, `2pi` | multiplicación implícita |
| `e`, `e^x`, `exp(x)` | número e |
| `pi`, `π` | pi |
| `i` (o `j`) | unidad imaginaria |
| `oo`, `inf`, `infinito`, `∞` | infinito |
| `sqrt(x)`, `raiz(x)`, `√x`, `cbrt(x)`, `root(x, n)` | raíces |
| `ln(x)` | logaritmo natural |
| `log(x)` | **logaritmo en base 10** (como en la cursada) ; `log(x, b)` base b |
| `sen`, `sin`, `cos`, `tg`, `tan`, `cot`, `sec`, `csc` | trigonométricas |
| `arcsen`, `asin`, `arccos`, `arctg`, `atan` | inversas |
| `senh`, `sinh`, `cosh`, `tgh`, `tanh` | hiperbólicas |
| `abs(x)`, `\|x\|` | valor absoluto |
| `x!`, `factorial(n)`, `binomial(n,k)` | combinatoria |
| `u(t)`, `heaviside(t)`, `delta(t)` | escalón y delta |
| `[[1,2],[3,4]]` | matriz (filas) |
| `[1,2,3]` | vector |
| `y'`, `y''`, `y'''` | derivadas de y(x) (o y(t) si aparece t) en una EDO |

Separadores: los argumentos van separados por **coma**; varias ecuaciones o condiciones por **`;`**.
Los comandos no distinguen mayúsculas ni tildes (`límite` = `limite`).

## Comandos

`valor` es el resultado principal (lo que se compara en las pruebas). `valores` son resultados con nombre.

### Álgebra
| Comando | Ejemplo | `valor` |
|---|---|---|
| (expresión sola) / `simplificar` | `(x^2-1)/(x-1)` | expresión simplificada |
| `evaluar` | `evaluar sqrt(2)*pi` | valor exacto; `valores.decimal` |
| `expandir` | `expandir (x+1)^3` | expresión |
| `factorizar` | `factorizar x^3-1` | expresión |
| `fracciones parciales` | `fracciones parciales (3x+5)/(x^2+3x+2)` | expresión |
| `resolver` | `resolver x^2-5x+6=0` | lista de soluciones (sin `=` se iguala a 0) |
| `resolver` (sistema) | `resolver x+y=3; x-y=1` | dict `{x: 2, y: 1}` (si hay varias soluciones: lista de dicts) |
| `resolver` (inecuación) | `resolver x^2<4` | conjunto de SymPy, p. ej. `Interval.open(-2, 2)` |
| `mcd`, `mcm` | `mcd 12, 18` | número |
| `sumatoria` | `sumatoria k^2, k, 1, n` | expresión o número (admite `oo`) |

### Cálculo (una variable)
| Comando | Ejemplo | `valor` |
|---|---|---|
| `derivar f[, var[, orden]]` | `derivar sen(x)^2` ; `derivar x^5, x, 3` | derivada |
| `derivar implicita F=G` | `derivar implicita x^2+y^2=25` | dy/dx |
| `integrar f[, var]` | `integrar x*e^x` | una primitiva (sin +C) |
| `integrar f, var, a, b` o `integrar f de a a b` | `integrar x^2, x, 0, 3` | número/expresión |
| `limite f, x->a` | `limite sen(x)/x, x->0` ; `limite 1/x, x->0+` ; `limite (1+1/x)^x, x->oo` | límite |
| `taylor f, x, a, n` | `taylor e^x, x, 0, 4` | polinomio hasta grado n (sin O) |
| `recta tangente f, a` | `recta tangente x^2, 1` | expresión de la recta (y = …), en x |
| `extremos f` (1 o 2 variables) | `extremos x^3-3x` | lista de puntos críticos (números o tuplas); `valores.tipos` |
| `area f, g, a, b` | `area x, x^2, 0, 1` | área entre curvas (positiva) |

### Cálculo varias variables y vectorial
| Comando | Ejemplo | `valor` |
|---|---|---|
| `derivar f, y` | `derivar x^2*y^3, y` | derivada parcial |
| `gradiente f` | `gradiente x^2*y+z` | vector (variables en orden x, y, z) |
| `divergencia [P,Q,R]` | `divergencia [x*y, y*z, z*x]` | expresión |
| `rotacional [P,Q,R]` | `rotacional [y, -x, 0]` | vector |
| `laplaciano f` | `laplaciano x^2+y^2` | expresión |
| `jacobiano [f1, f2, ...]` | `jacobiano [x^2*y, x+y]` | matriz (variables x, y, z en orden) |
| `hessiano f` | `hessiano x^3+x*y^2` | matriz |
| `integral doble f, x, a, b, y, c, d` | `integral doble x*y, x, 0, 1, y, 0, 2` | se integra primero en la primera variable |
| `integral triple f, x, a, b, y, c, d, z, e, f` | | número |
| `lagrange f; g` | `lagrange x*y; x+y=10` | lista de puntos candidatos (tuplas); `valores.maximo`, `valores.minimo` |
| `punto u, v` / `cruz u, v` | `cruz [1,0,0], [0,1,0]` | escalar / vector |
| `norma u` | `norma [3,4]` | número |
| `angulo u, v` | `angulo [1,0], [0,1]` | ángulo en radianes (exacto); `valores.grados` |
| `proyeccion u, v` | `proyeccion [1,2], [3,0]` | vector: proyección de u sobre v |

### Álgebra lineal
| Comando | Ejemplo | `valor` |
|---|---|---|
| `det M` | `det [[1,2],[3,4]]` | número |
| `inversa M` | | matriz |
| `transpuesta M`, `traza M`, `rango M` | | matriz / número / número |
| `autovalores M` | `autovalores [[2,0],[0,3]]` | lista con multiplicidad |
| `autovectores M` | | lista de `[autovalor, multiplicidad, [vectores]]` |
| `gauss M` | `gauss [[1,1,3],[1,-1,1]]` (matriz ampliada) | forma escalonada reducida; `valores.solucion` |
| operaciones | `[[1,2],[3,4]]*[[0,1],[1,0]]`, `[[1,2],[3,4]]^2` | matriz |

### Ecuaciones diferenciales y transformadas
| Comando | Ejemplo | `valor` |
|---|---|---|
| `edo ecuación[; condiciones]` | `edo y''+3y'+2y=0; y(0)=1; y'(0)=0` | expresión de y (constantes C1, C2, ... si no hay condiciones) |
| `laplace f` | `laplace t^2*e^(-3t)` | F(s) |
| `laplace inversa F` | `laplace inversa 1/(s^2+4)` | f(t) para t>0 (sin el factor escalón) |
| `fourier f, x, -L, L, n` | `fourier x, x, -pi, pi, 3` | suma parcial con n armónicos; `valores.a0`, `valores.an`, `valores.bn` |

### Métodos numéricos (Cálculo Numérico)
| Comando | Ejemplo | `valor` |
|---|---|---|
| `newton f, x0[, tol]` | `newton x^3-2x-5, 2` | raíz (decimal); tabla de iteraciones |
| `biseccion f, a, b[, tol]` | `biseccion x^2-2, 1, 2` | raíz |
| `secante f, x0, x1[, tol]` | | raíz |
| `punto fijo g, x0[, tol]` | `punto fijo cos(x), 1` | punto fijo de g |
| `trapecio f, a, b, n` | `trapecio x^2, 0, 1, 4` | aproximación compuesta con n subintervalos |
| `simpson f, a, b, n` | `simpson x^2, 0, 1, 4` | Simpson 1/3 compuesto (n par) |
| `euler y'=f, x0, y0, h, xf` | `euler y'=x+y, 0, 1, 0.1, 1` | aproximación de y(xf) |
| `rk4 y'=f, x0, y0, h, xf` | | aproximación de y(xf) por Runge-Kutta 4 |
| `interpolar (x0,y0), (x1,y1), ...` | `interpolar (0,1), (1,3), (2,2)` | polinomio de Lagrange (expandido) |
| `regresion (x0,y0), ...` | | recta de mínimos cuadrados `a*x+b`; `valores.r2` |

Tolerancia por defecto 1e-10; las raíces se devuelven con 10 decimales.

### Complejos, circuitos y estadística
| Comando | Ejemplo | `valor` |
|---|---|---|
| (expresión con i) | `(3+4i)/(1-2i)` | forma binómica a+bi |
| `polar z` | `polar 1+i` | `[modulo, argumento]` (argumento en radianes, exacto) |
| `fasor A@θ ± B@φ` (θ en grados) | `fasor 10@30 + 5@-45` | número complejo; `valores.modulo`, `valores.angulo` (grados) |
| `paralelo R1, R2, ...` | `paralelo 10, 20, 20` | resistencia equivalente |
| `serie R1, R2, ...` | `serie 10, 20` | suma |
| `estadistica d1, d2, ...` | `estadistica 2, 4, 4, 5` | media; `valores`: media, mediana, moda, varianza, desvio (muestrales, n-1), varianza_pob, desvio_pob, n |
| `binomial n, p, k` | `binomial 10, 0.5, 3` | P(X = k) |
| `poisson λ, k` | `poisson 2, 3` | P(X = k) |
| `normal μ, σ, a, b` | `normal 0, 1, -oo, 1.96` | P(a < X < b) |

### Gráficos
| Comando | Ejemplo | `grafico.tipo` |
|---|---|---|
| `graficar f[, g, ...]` | `graficar sen(x), cos(x)` | `2d` |
| `graficar F=G` | `graficar x^2+y^2=9` | `implicita` |
| `graficar polar r(t)` | `graficar polar 1+cos(t)` | `polar` |
| `graficar parametrica x(t), y(t)` | `graficar parametrica cos(t), sin(2t)` | `parametrica` |
| `graficar 3d f(x,y)` | `graficar 3d x^2-y^2` | `3d` |
| `graficar datos (x,y), ...` | | `datos` |

Además, la mayoría de los resultados de una variable incluyen un gráfico automático
(función, primitiva, área bajo la curva, recta tangente, raíces).

## Comprobación ("¿Está bien mi respuesta?")

`resolver(texto, respuesta)` compara la respuesta del alumno con la del motor:
- derivadas, simplificaciones, límites, transformadas: equivalencia simbólica (y numérica de respaldo)
- integrales indefinidas: se acepta cualquier primitiva (difiere en una constante)
- ecuaciones: mismo conjunto de soluciones (se aceptan listas en cualquier orden)
- EDO: se sustituye la respuesta en la ecuación y en las condiciones iniciales
- numéricos: tolerancia relativa 1e-4 (o la que se indique con `±`)

## Respuesta del motor (JSON)

```json
{
  "ok": true,
  "tema": "Cálculo",
  "titulo": "Derivada",
  "entrada_latex": "...",
  "resultado_latex": "...",
  "valor": "<texto parseable por sympy>",
  "valores": {"decimal": "0.5"},
  "pasos": [{"titulo": "Regla de la cadena", "texto": "...", "latex": "..."}],
  "tabla": {"columnas": ["n", "x_n", "f(x_n)"], "filas": [["0", "2", "-1"]]},
  "grafico": {"tipo": "2d", "...": "..."},
  "comprobacion": {"correcto": true, "mensaje": "..."},
  "error": null
}
```

## Ampliación (programas de la Facultad de Ingeniería – UNJu)

### Ingreso / notación escolar
`144 : 12` (división) · `0,726` (coma decimal) · `1 1/4` (número mixto) · `20% de 800` · `2/5 de 25` · `3 mas 4 por 2` ·
`convertir 9,1 hl a cl` (SIMELA, km/h, °C/°F/K, atm/kPa, kWh, HP...) · `regla de tres a, b, c` · `regla de tres inversa a, b, c` ·
`area circulo r` · `area trapecio B, b, h` · `perimetro rectangulo b, h` · `volumen cilindro r, h` (cuadrado, rectángulo, triángulo,
círculo, trapecio, rombo, sector, hexágono, corona; cubo, prisma, cilindro, cono, esfera, pirámide). Una palabra desconocida da error
(no se interpreta como producto de letras).

### Análisis I, II y III (`calculo_extra.py`)
| Comando | Ejemplo |
|---|---|
| `dominio f` · `asintotas f` · `inflexion f` · `estudio completo f` | `asintotas (x^2+1)/(x-1)` |
| `recta normal f, a` · `diferencial f, a, dx` · `aproximar expr` · `linealizar f, a` | `aproximar sqrt(4.1)` |
| `valor medio f, a, b` · `rolle f, a, b` · `funcion inversa f` | `valor medio x^3, 0, 2` |
| `area polar r, a, b` · `longitud de arco f, a, b` | `area polar 1+cos(t), 0, 2pi` |
| `volumen de revolucion f[, g], a, b[, eje y \| y=c \| x=c]` · `superficie de revolucion f, a, b` | `volumen de revolucion x^2, 0, 1, eje y` |
| `derivada parametrica x(t), y(t)` · `criterio serie a_n` · `radio de convergencia a_n(x)` | `criterio serie n!/n^n` |
| `limite doble f, (x,y)->(a,b)` (también `limite f, (x,y)->(a,b)`) | `limite doble x*y/(x^2+y^2), (x,y)->(0,0)` |
| `derivada direccional f, P, v` · `diferencial total f, P, dx, dy` | `derivada direccional x^2*y, (1,2), (3,4)` |
| `integral de linea F, r(t), a, b` · `potencial F` · `flujo F, superficie, ...` | `potencial (2x*y, x^2)` |
| `curvatura r(t)` · `funcion vectorial r(t)` · `sistema edo x'=..., y'=...[; x(0)=a]` · `trayectorias ortogonales familia` | `sistema edo x'=x+y, y'=4x+y` |

### Geometría analítica, álgebra lineal y discreta (`geo_lineal.py`)
`recta por P, Q` · `recta pendiente punto m, P` · `recta paralela/perpendicular L, P` · `plano por A, B, C` · `distancia X, Y` ·
`angulo entre X, Y` · `posicion relativa X, Y` · `interseccion de planos/rectas` · `conica ecuación` · `cuadrica ecuación` ·
`producto mixto u, v, w` · `nucleo M` · `imagen M` · `independencia v1, v2, ...` · `base y dimension ...` · `diagonalizar M` ·
`cramer ec1; ec2; ...` · `division de polinomios P, Q` · `ruffini P, a` · `coordenadas v en base ...` · `cambio de base B1; B2` ·
`factores primos n` (o `factorizar 360`) · `2^100 mod 7` · `3x≡4 (mod 7)` · `congruencia ...; ...` (resto chino) · `euclides a, b` ·
`diofantica ax+by=c` · `tabla de verdad p -> (q y r)` · `simplificar booleana ab+ab'` · `combinaciones n, k` · `variaciones n, k` ·
`permutaciones n` (con repetición, circulares, anagramas).

### Estadística, numérico, electrotecnia e investigación operativa (`estad_numerico.py`)
`cuantil normal p` · `cuantil t p, gl` · `cuantil chi2 p, gl` · `cuantil f p, gl1, gl2` · `normal inversa μ, σ, p` · `geometrica p, k` ·
`hipergeometrica N, K, n, k` · `exponencial λ, a, b` · `uniforme a, b, c, d` · `t student gl, a, b` · `bayes P(A)=…, P(B|A)=…, P(B|no A)=…` ·
`probabilidad total` · `percentil p, datos` · `asimetria datos` · `curtosis datos` · `tabla de frecuencias datos; clases k` · `cuartiles datos` ·
`intervalo media x̄=…, σ=… (o s=…), n=…, 95%` · `intervalo varianza` · `intervalo proporcion` · `tamano de muestra` · `prueba z` · `prueba t` ·
`prueba proporcion` · `chi2 bondad observados [..], esperados [..]` · `chi2 independencia [[..]]` · `correlacion (x,y), ...` ·
`regula falsi` · `muller` · `jacobi A, b[, iter]` · `gauss seidel A, b[, iter]` · `lu A[, b]` · `spline puntos; x` · `hermite` ·
`minimos cuadrados grado m puntos` · `ajuste exponencial` · `ajuste potencial` · `simpson 3/8` · `romberg` · `gauss legendre` ·
`derivada numerica f, x0, h` · `adams bashforth` · `adams moulton` · `rk4 sistema` · `error aprox, exacto` · `propagacion de errores` ·
`potencia compleja V=220@0, I=10@-30` · `correccion fp P=…, fp1=…, fp2=…, V=…, f=…` · `trifasico V=…, I=…, fp=…` ·
`programacion lineal max …; restricciones` · `asignacion [[..]]` · `cola mm1 λ=…, μ=…` · `cola mmc λ=…, μ=…, c=…`.
