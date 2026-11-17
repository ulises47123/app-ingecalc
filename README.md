# IngeCalc — matemática de ingeniería paso a paso

App Android que **resuelve, explica, grafica y corrige** ejercicios de matemática, sin internet: desde la cartilla de
ingreso hasta Análisis III, Cálculo Numérico y Probabilidad y Estadística de la Facultad de Ingeniería (UNJu).
Toma la idea de “sacar una foto o escribir el ejercicio y ver el desarrollo”, con diseño y motor propios.

Su función central es **revisar si una actividad está bien**: escribís el ejercicio y tu resultado
(“¿Está bien mi resultado?”) y la app compara matemáticamente (no por texto): acepta formas equivalentes, + C,
soluciones en cualquier orden, redondeos, otra parametrización de una recta, otra base de un subespacio… y si está mal
da una pista (signo cambiado, falta el factor de la regla de la cadena, difiere en una constante, no cumple la
condición inicial, raíz que sobra, extremo abierto/cerrado, etc.).

## Pantallas

1. **Cámara** (inicio): marco ajustable, disparador, galería, linterna. El recorte se lee con reconocimiento de texto
   en el celular (ML Kit, sin internet) y se puede corregir antes de resolver.
2. **Calculadora**: editor visual de fórmulas (MathLive: fracciones, raíces y potencias se dibujan mientras se escribe)
   con teclado matemático propio de 5 pestañas, y **resultado en vivo** (exacto y decimal) mientras se escribe.
   Botón `abc` = modo texto para comandos (`derivar`, `limite`, `edo`, `intervalo media`…).
3. **Soluciones**: tarjetas con cada forma de resolverlo (simplificar, factorizar, raíces, derivada, integral, estudio
   completo…), resultado resaltado y “Ver el desarrollo paso a paso”; abajo, “¿Está bien mi resultado?”.
4. **Paso a paso**: pasos numerados con “¿Por qué?”, gráfico interactivo (zoom, desplazamiento, lectura de
   coordenadas, 3D giratorio), tablas de iteraciones y más información.
5. **Temas**: 18 temas y 194 plantillas listas (Ingreso, Álgebra, Análisis I/II/III, Álgebra lineal, Espacios
   vectoriales, Geometría analítica, EDO y Laplace, Numérico, Numérico avanzado, Complejos y circuitos,
   Electrotecnia, Probabilidad, Inferencia, Discreta, Investigación operativa, Gráficos).

## Qué resuelve

| Área | Ejemplos |
|---|---|
| Ingreso | jerarquía de operaciones paso a paso (`48 : 6·2 + (9 + 5·7) : 11 − 2⁴ : √64`), coma decimal, números mixtos, `20% de 800`, `2/5 de 25`, regla de tres, SIMELA y unidades técnicas, áreas/perímetros/volúmenes |
| Álgebra | simplificar, factorizar, fracciones parciales, ecuaciones (resolvente, dominio), sistemas (Gauss + Rouché-Frobenius, Cramer), inecuaciones con tabla de signos |
| Análisis I | límites (L'Hôpital, 1^∞, laterales), derivadas con cada regla, implícita, tangente/normal, dominio, asíntotas, concavidad, estudio completo, valor medio, Rolle, función inversa, diferencial |
| Análisis II | integrales (sustitución, partes, fracciones parciales, trigonométrica), definidas e impropias, áreas (también en polares), longitud de arco, volúmenes y superficies de revolución, criterios de series, radio de convergencia, Taylor |
| Análisis III | límite doble (por caminos y polares), derivada direccional, diferencial total, gradiente/div/rot, jacobiano, hessiano, extremos, Lagrange, integrales dobles/triples, de línea, flujo, potencial, curvatura, sistemas de EDO |
| Álgebra lineal y geometría | determinantes, inversa, rango, autovalores, diagonalización, núcleo/imagen, independencia, bases, coordenadas, rectas y planos (distancias, ángulos, intersecciones, posición relativa), cónicas y cuádricas |
| EDO y transformadas | separables, lineales, Bernoulli, exactas, coeficientes constantes, Cauchy-Euler, PVI, Laplace directa/inversa con escalón, Fourier |
| Cálculo numérico | bisección, Newton, secante, punto fijo, regula falsi, Müller, Jacobi, Gauss-Seidel, LU, interpolación, spline, Hermite, mínimos cuadrados, trapecio, Simpson 1/3 y 3/8, Romberg, Gauss-Legendre, derivación numérica, Euler, Heun, RK4 (también sistemas), Adams, errores |
| Probabilidad y estadística | descriptiva, percentiles, asimetría, curtosis, tablas de frecuencia, boxplot, binomial, Poisson, normal, geométrica, hipergeométrica, exponencial, t, χ², F, Bayes, intervalos de confianza, pruebas z/t/proporción, χ² de bondad e independencia, correlación |
| Electrotecnia y discreta | fasores, potencia compleja, corrección del factor de potencia, trifásico; módulo, congruencias, resto chino, Euclides/Bézout, diofánticas, tablas de verdad, álgebra de Boole, combinatoria |
| Investigación operativa | programación lineal (gráfico y simplex), asignación (húngaro), colas M/M/1 y M/M/c |

La sintaxis completa está en [SINTAXIS.md](SINTAXIS.md) y en “Cómo se usa” dentro de la app.

## Estructura

```
motor/ingecalc/      Motor Python + SymPy + mpmath (el mismo código corre en la PC y en el celular)
  nucleo.py            lectura de la sintaxis amigable y escolar, LaTeX "tal como se escribió", jerarquía de operaciones
  pasos_derivada.py    derivación con cada regla explicada
  pasos_integral.py    integración paso a paso (traduce manualintegrate al español)
  calculo.py algebra.py lineal.py multivar.py edo.py numerico.py varios.py basico.py graficos.py
  calculo_extra.py     Análisis I–III ampliado        geo_lineal.py   geometría, lineal, discreta
  estad_numerico.py    inferencia, numérico avanzado, electrotecnia, investigación operativa
  comprobar.py         corrección de la respuesta del alumno
  api.py               resolver(), sugerencias() (tarjetas), previa(), llamar() (puente con Kotlin)
  autoprueba.py        corre el banco de ejercicios (PC y celular)
web/                 Interfaz: HTML/JS, KaTeX y MathLive sin conexión, gráficos en canvas propios
android/             Kotlin: WebView + Chaquopy (Python 3.11) + ML Kit; tools/cdp.py maneja la UI por depuración
pruebas/             banco_500.json (+ generador que verifica cada esperado aparte), pruebas por módulo, plantillas
```

## Compilar e instalar

```bash
cd android
bash tools/compilar.sh            # debug  -> app/build/outputs/apk/debug/app-debug.apk
bash tools/compilar.sh release    # liviana
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Necesita el JDK/SDK de `app-mando-facil/.tools` y Python 3.11 en la PC (Chaquopy lo usa para compilar).
Si el motor tarda en arrancar en el celular, la traza queda en `Android/data/ar.ingecalc.app/files/traza_arranque.txt`.

## Pruebas (resultados del 11/11/2026)

| Prueba | Resultado |
|---|---|
| Banco de 500 ejercicios, niveles 1→10, esperados verificados aparte con SymPy | **500/500 en la PC** y **500/500 dentro del celular** (Xiaomi, 124 s) |
| Corrección de respuestas sobre el banco | 442/442 correctas aceptadas, 430 alteradas rechazadas |
| Ejercicios difíciles de parcial | 97/97 |
| Todos los comandos (humo) | 94/94 |
| Módulos nuevos | Análisis I–III 96/96 · geometría/lineal/discreta 79/79 · estadística/numérico/IO 154/154 |
| Plantillas de la pantalla Temas | 194/194 |
| Interfaz en el celular (depuración remota) | cámara, teclado, editor visual, resultado en vivo, tarjetas, paso a paso |

```bash
python pruebas/correr_banco.py          # banco en la PC
python pruebas/humo.py                  # todos los comandos
python pruebas/plantillas.py            # plantillas de Temas
# en el celular:
adb push pruebas/banco_500.json /sdcard/Android/data/ar.ingecalc.app/files/
adb shell am start -n ar.ingecalc.app/.MainActivity --es autoprueba /sdcard/Android/data/ar.ingecalc.app/files/banco_500.json
adb pull /sdcard/Android/data/ar.ingecalc.app/files/resultado_autoprueba.json
```

## Bibliotecas de terceros incluidas

- [KaTeX](https://katex.org) (licencia MIT) en `web/katex/`
- [MathLive](https://cortexjs.io/mathlive/) 0.98.6 (licencia MIT) en `web/mathlive/`
- SymPy/mpmath (BSD) se instalan con Chaquopy al compilar; ML Kit se descarga con Gradle.

No se suben `build/`, `.gradle/`, `local.properties` ni APKs: `local.properties` lo genera Android Studio
o se crea a mano con `sdk.dir=<ruta al Android SDK>`.
