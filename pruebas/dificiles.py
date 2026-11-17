import sys, time
sys.path.insert(0, r'C:\Users\Usuario\Music\proyectos\app-ingecalc\motor')
from ingecalc.api import resolver
casos = [
 "integrar sec(x)^3", "integrar 1/(x^2*sqrt(x^2+4))", "integrar e^(2x)*cos(3x)", "integrar (x^3+1)/(x^2-4)", "integrar x^2*e^(-x), x, 0, oo",
 "integrar ln(x)/x^2, x, 1, oo", "integrar 1/sqrt(1-x^2), x, 0, 1", "integrar arctg(x)", "integrar sen(x)^3*cos(x)^2", "integrar 1/(x^4-1)",
 "limite (e^x-1-x)/x^2, x->0", "limite x*ln(x), x->0+", "limite (1+2/x)^(3x), x->oo", "limite sqrt(x^2+x)-x, x->oo", "limite (tg(x)-sen(x))/x^3, x->0",
 "limite x^x, x->0+", "limite (1/x - 1/sen(x)), x->0", "limite |x|/x, x->0",
 "derivar arcsen(x^2)", "derivar x^sen(x)", "derivar ln(sqrt((1+x)/(1-x)))", "derivar e^(x^2)*sen(3x)/x", "derivar implicita e^(xy)+y=x^2",
 "edo y''+y=tg(x)", "edo y''-4y'+4y=x*e^(2x)", "edo x^2y''-3xy'+4y=0", "edo y'+y/x=x^2*y^2", "edo (2xy+3)+(x^2-1)y'=0", "edo y'''-y=0",
 "edo y''+2y'+y=e^(-x); y(0)=0; y'(0)=1", "edo y'=(x+y)/x", "edo y' = 1 + y^2; y(0)=0",
 "laplace t*sen(2t)", "laplace e^(-t)*cos(3t)", "laplace u(t-2)*(t-2)^2", "laplace t^3-4t+cosh(2t)",
 "laplace inversa 5/(s*(s^2+4))", "laplace inversa (3s+2)/((s+1)^2*(s+3))", "laplace inversa 1/(s^2+2s+2)", "laplace inversa (s e^(-pi s))/(s^2+1)",
 "fourier abs(x), x, -pi, pi, 4", "fourier x^2, x, -1, 1, 3", "fourier heaviside(x), x, -pi, pi, 3",
 "extremos x^4-8x^2+3", "extremos x*e^(-x)", "extremos x^2+x*y+y^2-3x", "extremos x*y*(1-x-y)", "lagrange x^2+y^2+z^2; x+y+z=3",
 "lagrange x+y; x^2+y^2=1", "integral doble x^2+y^2, y, 0, sqrt(1-x^2), x, 0, 1", "integral triple 1, z, 0, 1-x-y, y, 0, 1-x, x, 0, 1",
 "autovalores [[1,2,0],[2,1,0],[0,0,3]]", "autovectores [[1,1],[0,1]]", "det [[1,2,3,4],[0,1,2,3],[1,0,1,2],[2,1,0,1]]",
 "inversa [[1,0,2],[2,-1,3],[4,1,8]]", "gauss [[1,1,1,6],[0,2,5,-4],[2,5,-1,27]]", "resolver x+y+z=6; 2y+5z=-4; 2x+5y-z=27",
 "resolver x^2+y^2=25; x+y=7", "resolver x^4-5x^2+4=0", "resolver x^3-6x^2+11x-6=0", "resolver ln(x)+ln(x-2)=ln(3)",
 "resolver |2x-1|<5", "resolver (x^2-1)/(x-3) <= 0", "resolver 2^x=3^(x-1)", "resolver cos(2x)=cos(x)", "resolver x^5=1",
 "taylor sen(x), x, pi, 3", "taylor 1/(1-x), x, 0, 6", "taylor ln(x), x, 1, 4", "sumatoria (-1)^(k+1)/k, k, 1, oo", "sumatoria 1/(k(k+1)), k, 1, oo",
 "newton e^x-3x, 0", "biseccion x^3-x-2, 1, 2, 0.001", "simpson sen(x)/x, 1, 2, 8", "rk4 y'=y-x^2+1, 0, 0.5, 0.2, 2", "euler y'=-2xy, 0, 1, 0.25, 1",
 "interpolar (1,1), (2,8), (3,27), (4,64)", "polar -1-i", "resolver z^3=8i", "(1+i)^10", "e^(i pi/3)",
 "estadistica 3.2, 4.5, 5.1, 2.8, 6.0, 4.4", "normal 50, 10, 45, oo", "binomial 20, 0.1, 0",
 "graficar x^3-3x, 3x^2-3", "graficar x*y=1", "graficar polar sen(3t)", "graficar 3d e^(-x^2-y^2)", "graficar ln(x)", "graficar tg(x)",
 "rotacional [x^2*y, y*z, x*z^2]", "divergencia [x^2, y^2, z^2]", "jacobiano [r*cos(theta), r*sen(theta)], [r, theta]",
 "integrar x/(x^2+1), x, 0, 1", "area sen(x), cos(x), 0, pi/2", "recta tangente e^x, 0", "derivar x^3*y^2+sen(x*y), x",
]
t0=time.time(); malos=0
for c in casos:
    d = resolver(c)
    if not d.get('ok'): malos+=1
    print(('OK ' if d.get('ok') else 'ERR'), f"{d['tiempo']:6.2f}s", c, '=>', (str(d.get('valor'))[:110] if d.get('ok') else d.get('error') + ' ' + str(d.get('detalle',''))[-300:].replace('\n',' | ')))
print('errores', malos, 'de', len(casos), round(time.time()-t0,1),'s')
