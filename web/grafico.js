/* IngeCalc — dibujo de gráficos en canvas (sin librerías).
   Tipos: 2d, implicita, polar, parametrica, 3d, datos, barras. Zoom con pellizco, arrastre, doble toque = reiniciar. */
'use strict';

const F = (() => {
  const erf = (x) => { // Abramowitz-Stegun 7.1.26
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  };
  const gamma = (z) => {
    if (z < 0.5) return Math.PI / (Math.sin(Math.PI * z) * gamma(1 - z));
    z -= 1;
    const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
      -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    let x = c[0];
    for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
    const t = z + g + 0.5;
    return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
  };
  return {
    sin: Math.sin, cos: Math.cos, tan: Math.tan, cot: (x) => 1 / Math.tan(x), sec: (x) => 1 / Math.cos(x), csc: (x) => 1 / Math.sin(x),
    asin: Math.asin, acos: Math.acos, atan: Math.atan, acot: (x) => Math.PI / 2 - Math.atan(x),
    asec: (x) => Math.acos(1 / x), acsc: (x) => Math.asin(1 / x),
    sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh, coth: (x) => 1 / Math.tanh(x), sech: (x) => 1 / Math.cosh(x), csch: (x) => 1 / Math.sinh(x),
    asinh: Math.asinh, acosh: Math.acosh, atanh: Math.atanh,
    exp: Math.exp, ln: Math.log, sqrt: Math.sqrt, pow: Math.pow, abs: Math.abs, sign: Math.sign,
    floor: Math.floor, ceil: Math.ceil, max: Math.max, min: Math.min, atan2: Math.atan2,
    rpow: (b, p, q) => { // b^(p/q) real con q impar
      const r = Math.pow(Math.abs(b), p / q);
      return (b < 0 && p % 2 !== 0) ? -r : r;
    },
    rroot: (b, n) => (b < 0 && n % 2 === 1) ? -Math.pow(-b, 1 / n) : Math.pow(b, 1 / n),
    H: (x) => (x > 0 ? 1 : (x < 0 ? 0 : 0.5)), zero: () => 0, erf, gamma,
    fact: (n) => gamma(n + 1), re_: (x) => x, im_: () => 0,
    E: Math.E, PI: Math.PI, Infinity: Infinity, NaN: NaN,
  };
})();

function compilar(codigo, vars) {
  if (!codigo) return null;
  try {
    const fn = new Function('F', ...vars, 'with (F) { return (' + codigo + '); }');
    return (...a) => { try { const v = fn(F, ...a); return typeof v === 'number' ? v : (typeof v === 'boolean' ? +v : NaN); } catch (e) { return NaN; } };
  } catch (e) { return null; }
}

function colorVar(nombre, defecto) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(nombre).trim();
  return v || defecto;
}

function pasoLindo(rango, objetivo) {
  const crudo = rango / objetivo;
  const p = Math.pow(10, Math.floor(Math.log10(crudo)));
  const m = crudo / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
}

function fmt(v) {
  if (Math.abs(v) < 1e-10) return '0';
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) return v.toExponential(1);
  return String(+v.toFixed(4));
}

class Grafico {
  constructor(contenedor, spec) {
    this.spec = spec;
    this.cont = contenedor;
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'lienzo';
    this.info = document.createElement('div');
    this.info.className = 'grafico-info';
    contenedor.appendChild(this.canvas);
    contenedor.appendChild(this.info);
    this.tipo = spec.tipo;
    this.ang = { a: -0.6, b: 0.9 }; // 3D
    this.cursor = null;
    this._prep();
    this._ventanaInicial();
    this._eventos();
    this.redimensionar();
    this.observador = new ResizeObserver(() => this.redimensionar());
    this.observador.observe(contenedor);
  }

  /* Libera memoria (canvas y observador) al mostrar otro resultado */
  destruir() {
    try { this.observador.disconnect(); } catch (e) { /* */ }
    this.canvas.width = 0; this.canvas.height = 0;
    this.ctx = null; this.curvas = []; this.sup = null;
  }

  _prep() {
    const s = this.spec;
    this.curvas = (s.curvas || []).map((c) => {
      const o = Object.assign({}, c);
      if (this.tipo === 'implicita') o.f = compilar(c.fn, c.vars || ['x', 'y']);
      else if (this.tipo === 'parametrica') { o.fx = compilar(c.fx, [c.var || 't']); o.fy = compilar(c.fy, [c.var || 't']); }
      else o.f = compilar(c.fn, [c.var || 'x']);
      return o;
    });
    if (s.sombra) {
      const v = s.sombra.var || 'x';
      this.sombra = Object.assign({}, s.sombra, { fa: compilar(s.sombra.arriba, [v]), fb: compilar(s.sombra.abajo, [v]) });
    }
    if (this.tipo === '3d' && s.superficie) this.sup = Object.assign({}, s.superficie, { f: compilar(s.superficie.fn, s.superficie.vars || ['x', 'y']) });
  }

  _ventanaInicial() {
    const s = this.spec;
    let v = s.ventana ? Object.assign({}, s.ventana) : null;
    if (this.tipo === 'polar' || this.tipo === 'parametrica') {
      let xs = [], ys = [];
      for (const c of this.curvas) {
        const a = c.desde ?? 0, b = c.hasta ?? 2 * Math.PI;
        for (let i = 0; i <= 600; i++) {
          const t = a + (b - a) * i / 600;
          let px, py;
          if (this.tipo === 'polar') { const r = c.f(t); px = r * Math.cos(t); py = r * Math.sin(t); } else { px = c.fx(t); py = c.fy(t); }
          if (isFinite(px) && isFinite(py)) { xs.push(px); ys.push(py); }
        }
      }
      if (xs.length) {
        const m = Math.max(...xs.map(Math.abs), ...ys.map(Math.abs), 0.5) * 1.25;
        v = { xmin: -m, xmax: m, ymin: -m, ymax: m };
      }
    }
    if (this.tipo === 'barras') {
      const xs = s.barras.map((b) => b.x), hs = s.barras.map((b) => b.alto);
      v = { xmin: Math.min(...xs) - 1, xmax: Math.max(...xs) + 1, ymin: -Math.max(...hs) * 0.08, ymax: Math.max(...hs) * 1.2 };
    }
    this.v0 = v || { xmin: -10, xmax: 10, ymin: -10, ymax: 10 };
    this.v = Object.assign({}, this.v0);
    this.igualarEscala = this.tipo === 'implicita' || this.tipo === 'polar' || this.tipo === 'parametrica' || !!s.vectores;
  }

  redimensionar() {
    const r = this.cont.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.w = Math.max(200, r.width);
    this.h = Math.round(this.w * (this.tipo === '3d' ? 0.9 : 0.72));
    this.canvas.style.height = this.h + 'px';
    this.canvas.width = Math.round(this.w * dpr);
    this.canvas.height = Math.round(this.h * dpr);
    this.ctx = this.canvas.getContext('2d');
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.igualarEscala && !this._igualado) { this._igualar(); this._igualado = true; }
    this.dibujar();
  }

  _igualar() {
    const v = this.v, cx = (v.xmin + v.xmax) / 2, cy = (v.ymin + v.ymax) / 2;
    const ex = (v.xmax - v.xmin) / this.w, ey = (v.ymax - v.ymin) / this.h;
    const e = Math.max(ex, ey);
    v.xmin = cx - e * this.w / 2; v.xmax = cx + e * this.w / 2; v.ymin = cy - e * this.h / 2; v.ymax = cy + e * this.h / 2;
    this.v0 = Object.assign({}, v);
  }

  X(x) { return (x - this.v.xmin) / (this.v.xmax - this.v.xmin) * this.w; }
  Y(y) { return this.h - (y - this.v.ymin) / (this.v.ymax - this.v.ymin) * this.h; }
  invX(px) { return this.v.xmin + px / this.w * (this.v.xmax - this.v.xmin); }
  invY(py) { return this.v.ymin + (this.h - py) / this.h * (this.v.ymax - this.v.ymin); }

  _eventos() {
    const c = this.canvas;
    const punteros = new Map();
    let inicio = null, ultimoToque = 0, movio = false;
    c.addEventListener('pointerdown', (e) => {
      c.setPointerCapture(e.pointerId);
      punteros.set(e.pointerId, { x: e.offsetX, y: e.offsetY });
      inicio = { v: Object.assign({}, this.v), p: [...punteros.values()].map((p) => Object.assign({}, p)), ang: Object.assign({}, this.ang) };
      movio = false;
      const ahora = Date.now();
      if (ahora - ultimoToque < 300 && punteros.size === 1) { this.reiniciar(); }
      ultimoToque = ahora;
    });
    c.addEventListener('pointermove', (e) => {
      if (!punteros.has(e.pointerId)) return;
      punteros.set(e.pointerId, { x: e.offsetX, y: e.offsetY });
      const ps = [...punteros.values()];
      if (!inicio) return;
      if (this.tipo === '3d') {
        if (ps.length === 1) {
          const dx = ps[0].x - inicio.p[0].x, dy = ps[0].y - inicio.p[0].y;
          if (Math.abs(dx) + Math.abs(dy) > 3) movio = true;
          this.ang.a = inicio.ang.a + dx * 0.01;
          this.ang.b = Math.max(0.05, Math.min(1.5, inicio.ang.b + dy * 0.01));
          this.dibujar();
        }
        return;
      }
      if (ps.length === 1 && inicio.p.length === 1) {
        const dx = ps[0].x - inicio.p[0].x, dy = ps[0].y - inicio.p[0].y;
        if (Math.abs(dx) + Math.abs(dy) > 4) movio = true;
        if (!movio) return;
        const sx = (inicio.v.xmax - inicio.v.xmin) / this.w, sy = (inicio.v.ymax - inicio.v.ymin) / this.h;
        this.v = { xmin: inicio.v.xmin - dx * sx, xmax: inicio.v.xmax - dx * sx, ymin: inicio.v.ymin + dy * sy, ymax: inicio.v.ymax + dy * sy };
        this.cursor = null;
        this.dibujar();
      } else if (ps.length >= 2 && inicio.p.length >= 2) {
        movio = true;
        const d0x = Math.abs(inicio.p[0].x - inicio.p[1].x) + 1, d0y = Math.abs(inicio.p[0].y - inicio.p[1].y) + 1;
        const d1x = Math.abs(ps[0].x - ps[1].x) + 1, d1y = Math.abs(ps[0].y - ps[1].y) + 1;
        let fx = d0x / d1x, fy = d0y / d1y;
        if (this.igualarEscala || Math.abs(Math.log(fx / fy)) < 0.6) { const f = Math.sqrt(fx * fy); fx = fy = f; }
        const mx0 = (inicio.p[0].x + inicio.p[1].x) / 2, my0 = (inicio.p[0].y + inicio.p[1].y) / 2;
        const mx1 = (ps[0].x + ps[1].x) / 2, my1 = (ps[0].y + ps[1].y) / 2;
        const v = inicio.v;
        const cx = v.xmin + mx0 / this.w * (v.xmax - v.xmin), cy = v.ymin + (this.h - my0) / this.h * (v.ymax - v.ymin);
        const W = (v.xmax - v.xmin) * fx, H = (v.ymax - v.ymin) * fy;
        const xmin = cx - mx1 / this.w * W, ymin = cy - (this.h - my1) / this.h * H;
        this.v = { xmin, xmax: xmin + W, ymin, ymax: ymin + H };
        this.dibujar();
      }
    });
    const fin = (e) => {
      if (punteros.size === 1 && !movio && this.tipo !== '3d' && this.tipo !== 'barras') this._tocar(e.offsetX, e.offsetY);
      punteros.delete(e.pointerId);
      const ps = [...punteros.values()];
      inicio = ps.length ? { v: Object.assign({}, this.v), p: ps.map((p) => Object.assign({}, p)), ang: Object.assign({}, this.ang) } : null;
    };
    c.addEventListener('pointerup', fin);
    c.addEventListener('pointercancel', fin);
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      const f = Math.exp(e.deltaY * 0.001);
      const cx = this.invX(e.offsetX), cy = this.invY(e.offsetY), v = this.v;
      this.v = { xmin: cx + (v.xmin - cx) * f, xmax: cx + (v.xmax - cx) * f, ymin: cy + (v.ymin - cy) * f, ymax: cy + (v.ymax - cy) * f };
      this.dibujar();
    }, { passive: false });
  }

  reiniciar() { this.v = Object.assign({}, this.v0); this.ang = { a: -0.6, b: 0.9 }; this.cursor = null; this.dibujar(); }
  zoom(f) {
    const v = this.v, cx = (v.xmin + v.xmax) / 2, cy = (v.ymin + v.ymax) / 2;
    this.v = { xmin: cx + (v.xmin - cx) * f, xmax: cx + (v.xmax - cx) * f, ymin: cy + (v.ymin - cy) * f, ymax: cy + (v.ymax - cy) * f };
    this.dibujar();
  }

  _tocar(px, py) {
    // Lectura de coordenadas: punto más cercano en alguna curva 2d
    const x = this.invX(px);
    let mejor = null;
    if (this.tipo === '2d' || this.tipo === 'datos') {
      for (const c of this.curvas) {
        if (!c.f) continue;
        const y = c.f(x);
        if (!isFinite(y)) continue;
        const d = Math.abs(this.Y(y) - py);
        if (!mejor || d < mejor.d) mejor = { d, x, y, color: c.color };
      }
    }
    for (const p of (this.spec.puntos || [])) {
      const d = Math.hypot(this.X(p.x) - px, this.Y(p.y) - py);
      if (d < 24 && (!mejor || d < mejor.d)) mejor = { d, x: p.x, y: p.y, color: p.color, et: p.etiqueta };
    }
    this.cursor = mejor && mejor.d < 60 ? mejor : { x, y: this.invY(py), libre: true };
    this.dibujar();
  }

  dibujar() {
    if (!this.ctx) return;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.w, this.h);
    ctx.fillStyle = colorVar('--lienzo', '#fff');
    ctx.fillRect(0, 0, this.w, this.h);
    if (this.tipo === '3d') { this._dibujar3d(); return; }
    this._ejes();
    if (this.sombra) this._sombra();
    if (this.tipo === '2d' || this.tipo === 'datos') this.curvas.forEach((c) => this._curva2d(c));
    if (this.tipo === 'implicita') this.curvas.forEach((c) => this._implicita(c));
    if (this.tipo === 'polar' || this.tipo === 'parametrica') this.curvas.forEach((c) => this._parametrica(c));
    if (this.tipo === 'barras') this._barras();
    (this.spec.verticales || []).forEach((v) => this._vertical(v));
    (this.spec.vectores || []).forEach((v) => this._flecha(v));
    (this.spec.puntos || []).forEach((p) => this._punto(p));
    if (this.cursor) this._dibujarCursor();
    this._leyenda();
  }

  _ejes() {
    const ctx = this.ctx, v = this.v;
    const grilla = colorVar('--grilla', '#e5e7eb'), eje = colorVar('--eje', '#6b7280'), texto = colorVar('--texto-suave', '#6b7280');
    const px = pasoLindo(v.xmax - v.xmin, this.w / 70), py = pasoLindo(v.ymax - v.ymin, this.h / 55);
    ctx.lineWidth = 1; ctx.strokeStyle = grilla; ctx.beginPath();
    for (let x = Math.ceil(v.xmin / px) * px; x <= v.xmax; x += px) { const X = Math.round(this.X(x)) + 0.5; ctx.moveTo(X, 0); ctx.lineTo(X, this.h); }
    for (let y = Math.ceil(v.ymin / py) * py; y <= v.ymax; y += py) { const Y = Math.round(this.Y(y)) + 0.5; ctx.moveTo(0, Y); ctx.lineTo(this.w, Y); }
    ctx.stroke();
    const X0 = Math.min(Math.max(this.X(0), 0), this.w), Y0 = Math.min(Math.max(this.Y(0), 0), this.h);
    ctx.strokeStyle = eje; ctx.lineWidth = 1.3; ctx.beginPath();
    ctx.moveTo(0, Y0); ctx.lineTo(this.w, Y0); ctx.moveTo(X0, 0); ctx.lineTo(X0, this.h); ctx.stroke();
    ctx.fillStyle = texto; ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let x = Math.ceil(v.xmin / px) * px; x <= v.xmax; x += px) {
      if (Math.abs(x) < px / 2) continue;
      const Yt = Y0 + 4 > this.h - 14 ? Y0 - 16 : Y0 + 4;
      ctx.fillText(fmt(x), this.X(x), Yt);
    }
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    for (let y = Math.ceil(v.ymin / py) * py; y <= v.ymax; y += py) {
      if (Math.abs(y) < py / 2) continue;
      const Xt = X0 + 5 > this.w - 40 ? X0 - 40 : X0 + 5;
      ctx.fillText(fmt(y), Xt, this.Y(y));
    }
  }

  _curva2d(c) {
    const ctx = this.ctx;
    ctx.strokeStyle = c.color || '#2563eb'; ctx.lineWidth = 2.4; ctx.lineJoin = 'round';
    if (c.estilo === 'punteado') ctx.setLineDash([6, 5]);
    ctx.beginPath();
    if (c.f) {
      const n = Math.round(this.w * 1.5);
      let antes = null;
      const alto = this.v.ymax - this.v.ymin;
      for (let i = 0; i <= n; i++) {
        const x = this.v.xmin + (this.v.xmax - this.v.xmin) * i / n;
        const y = c.f(x);
        if (!isFinite(y)) { antes = null; continue; }
        // corte en asíntotas: salto enorme con cambio de signo de la pendiente
        if (antes !== null && Math.abs(y - antes) > alto * 3) { antes = null; }
        const Y = Math.max(-1e4, Math.min(1e4, this.Y(y)));
        if (antes === null) ctx.moveTo(this.X(x), Y); else ctx.lineTo(this.X(x), Y);
        antes = y;
      }
    } else if (c.muestras) {
      let pen = false;
      for (const p of c.muestras) {
        if (!p) { pen = false; continue; }
        if (!pen) { ctx.moveTo(this.X(p[0]), this.Y(p[1])); pen = true; } else ctx.lineTo(this.X(p[0]), this.Y(p[1]));
      }
    }
    ctx.stroke(); ctx.setLineDash([]);
  }

  _sombra() {
    const s = this.sombra, ctx = this.ctx;
    if (!s.fa || !s.fb) return;
    const a = Math.max(s.desde, this.v.xmin - 1), b = Math.min(s.hasta, this.v.xmax + 1);
    if (b <= a) return;
    const n = 300;
    ctx.fillStyle = colorVar('--sombra', 'rgba(37,99,235,0.18)');
    ctx.beginPath();
    for (let i = 0; i <= n; i++) { const x = a + (b - a) * i / n; const y = s.fa(x); ctx.lineTo(this.X(x), this.Y(isFinite(y) ? y : 0)); }
    for (let i = n; i >= 0; i--) { const x = a + (b - a) * i / n; const y = s.fb(x); ctx.lineTo(this.X(x), this.Y(isFinite(y) ? y : 0)); }
    ctx.closePath(); ctx.fill();
  }

  _implicita(c) {
    // marching squares sobre F(x,y)
    if (!c.f) return;
    const ctx = this.ctx, N = Math.round(Math.min(220, this.w / 2)), M = Math.round(N * this.h / this.w);
    const v = this.v, dx = (v.xmax - v.xmin) / N, dy = (v.ymax - v.ymin) / M;
    const val = new Float64Array((N + 1) * (M + 1));
    for (let j = 0; j <= M; j++) for (let i = 0; i <= N; i++) val[j * (N + 1) + i] = c.f(v.xmin + i * dx, v.ymin + j * dy);
    ctx.strokeStyle = c.color || '#2563eb'; ctx.lineWidth = 2.4; ctx.beginPath();
    const interp = (x1, y1, v1, x2, y2, v2) => { const t = v1 / (v1 - v2); return [x1 + (x2 - x1) * t, y1 + (y2 - y1) * t]; };
    for (let j = 0; j < M; j++) {
      for (let i = 0; i < N; i++) {
        const a = val[j * (N + 1) + i], b = val[j * (N + 1) + i + 1], cc = val[(j + 1) * (N + 1) + i + 1], d = val[(j + 1) * (N + 1) + i];
        if (!isFinite(a) || !isFinite(b) || !isFinite(cc) || !isFinite(d)) continue;
        const x0 = v.xmin + i * dx, y0 = v.ymin + j * dy, x1 = x0 + dx, y1 = y0 + dy;
        const pts = [];
        if ((a > 0) !== (b > 0)) pts.push(interp(x0, y0, a, x1, y0, b));
        if ((b > 0) !== (cc > 0)) pts.push(interp(x1, y0, b, x1, y1, cc));
        if ((cc > 0) !== (d > 0)) pts.push(interp(x1, y1, cc, x0, y1, d));
        if ((d > 0) !== (a > 0)) pts.push(interp(x0, y1, d, x0, y0, a));
        // descartar polos (cambio de signo por infinito, ej. 1/x)
        const m = Math.max(Math.abs(a), Math.abs(b), Math.abs(cc), Math.abs(d));
        if (m > 1e6) continue;
        for (let k = 0; k + 1 < pts.length; k += 2) {
          ctx.moveTo(this.X(pts[k][0]), this.Y(pts[k][1])); ctx.lineTo(this.X(pts[k + 1][0]), this.Y(pts[k + 1][1]));
        }
      }
    }
    ctx.stroke();
  }

  _parametrica(c) {
    const ctx = this.ctx, a = c.desde ?? 0, b = c.hasta ?? 2 * Math.PI, n = 1500;
    ctx.strokeStyle = c.color || '#2563eb'; ctx.lineWidth = 2.4; ctx.beginPath();
    let pen = false, prev = null;
    for (let i = 0; i <= n; i++) {
      const t = a + (b - a) * i / n;
      let x, y;
      if (this.tipo === 'polar') { const r = c.f(t); x = r * Math.cos(t); y = r * Math.sin(t); } else { x = c.fx(t); y = c.fy(t); }
      if (!isFinite(x) || !isFinite(y)) { pen = false; continue; }
      const X = this.X(x), Y = this.Y(y);
      if (prev && Math.hypot(X - prev[0], Y - prev[1]) > this.h) pen = false;
      if (!pen) { ctx.moveTo(X, Y); pen = true; } else ctx.lineTo(X, Y);
      prev = [X, Y];
    }
    ctx.stroke();
  }

  _barras() {
    const ctx = this.ctx, bs = this.spec.barras;
    const xs = bs.map((b) => b.x).sort((p, q) => p - q);
    let ancho = 1;
    for (let i = 1; i < xs.length; i++) ancho = Math.min(ancho, xs[i] - xs[i - 1] || 1);
    for (const b of bs) {
      ctx.fillStyle = b.resaltar ? '#dc2626' : colorVar('--acento', '#2563eb');
      const x0 = this.X(b.x - ancho * 0.4), x1 = this.X(b.x + ancho * 0.4);
      ctx.fillRect(x0, this.Y(b.alto), x1 - x0, this.Y(0) - this.Y(b.alto));
    }
    if (this.spec.media !== undefined) {
      ctx.strokeStyle = '#059669'; ctx.setLineDash([5, 4]); ctx.lineWidth = 2; ctx.beginPath();
      ctx.moveTo(this.X(this.spec.media), 0); ctx.lineTo(this.X(this.spec.media), this.h); ctx.stroke(); ctx.setLineDash([]);
    }
  }

  /* recta vertical punteada (asíntota x = a, eje de giro x = c) */
  _vertical(v) {
    const ctx = this.ctx, X = this.X(v.x);
    if (X < 0 || X > this.w) return;
    ctx.strokeStyle = v.color || '#dc2626'; ctx.lineWidth = 1.6; ctx.setLineDash([6, 5]);
    ctx.beginPath(); ctx.moveTo(X, 0); ctx.lineTo(X, this.h); ctx.stroke(); ctx.setLineDash([]);
    if (v.etiqueta) {
      ctx.font = '11px system-ui'; ctx.fillStyle = v.color || '#dc2626'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText(v.etiqueta, X + 4, 6);
    }
  }

  _flecha(vc) {
    const ctx = this.ctx, x0 = this.X(0), y0 = this.Y(0), x1 = this.X(vc.x), y1 = this.Y(vc.y);
    ctx.strokeStyle = ctx.fillStyle = vc.color || '#2563eb'; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    const ang = Math.atan2(y1 - y0, x1 - x0);
    ctx.beginPath(); ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - 12 * Math.cos(ang - 0.4), y1 - 12 * Math.sin(ang - 0.4));
    ctx.lineTo(x1 - 12 * Math.cos(ang + 0.4), y1 - 12 * Math.sin(ang + 0.4)); ctx.fill();
    ctx.font = '13px system-ui'; ctx.fillText(vc.etiqueta || '', x1 + 6, y1 - 6);
  }

  _punto(p) {
    const ctx = this.ctx, X = this.X(p.x), Y = this.Y(p.y);
    if (X < -10 || X > this.w + 10 || Y < -10 || Y > this.h + 10) return;
    ctx.fillStyle = p.color && p.color !== '#111827' ? p.color : colorVar('--punto', '#111827');
    ctx.beginPath(); ctx.arc(X, Y, 4.5, 0, 2 * Math.PI); ctx.fill();
    if (p.etiqueta && this.tipo !== 'datos') {
      ctx.font = '11px system-ui'; ctx.fillStyle = colorVar('--texto', '#111');
      ctx.textAlign = X > this.w - 120 ? 'right' : 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText(p.etiqueta, X + (X > this.w - 120 ? -7 : 7), Y - 5);
    }
  }

  _dibujarCursor() {
    const ctx = this.ctx, c = this.cursor, X = this.X(c.x), Y = this.Y(c.y);
    ctx.strokeStyle = colorVar('--eje', '#6b7280'); ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X, 0); ctx.lineTo(X, this.h); ctx.moveTo(0, Y); ctx.lineTo(this.w, Y); ctx.stroke(); ctx.setLineDash([]);
    if (!c.libre) { ctx.fillStyle = c.color || '#2563eb'; ctx.beginPath(); ctx.arc(X, Y, 5.5, 0, 2 * Math.PI); ctx.fill(); }
    this.info.textContent = (c.et ? c.et + '  ' : '') + `x = ${fmt(c.x)}   y = ${fmt(c.y)}`;
  }

  _leyenda() {
    const etqs = this.curvas.filter((c) => c.etiqueta_corta || c.etiqueta);
    if (!this.cursor) this.info.textContent = this.tipo === '3d' ? 'Arrastrá para girar · doble toque: reiniciar' : 'Tocá la curva para ver coordenadas · pellizcá para zoom · doble toque: reiniciar';
    if (etqs.length < 2) return;
    const ctx = this.ctx;
    ctx.font = '12px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    let y = 14;
    for (const c of etqs) {
      const txt = (c.etiqueta || '').replace(/\\left|\\right|\\/g, '').replace(/[{}]/g, '').slice(0, 28);
      ctx.fillStyle = colorVar('--lienzo', '#fff'); ctx.globalAlpha = 0.85; ctx.fillRect(this.w - 150, y - 8, 146, 16); ctx.globalAlpha = 1;
      ctx.fillStyle = c.color; ctx.fillRect(this.w - 146, y - 2, 14, 4);
      ctx.fillStyle = colorVar('--texto', '#111'); ctx.fillText(txt, this.w - 128, y);
      y += 18;
    }
  }

  _dibujar3d() {
    const s = this.sup, ctx = this.ctx;
    if (!s || !s.f) return;
    const R = s.rango || 3, N = 36;
    const pts = [];
    let zmin = Infinity, zmax = -Infinity;
    for (let i = 0; i <= N; i++) {
      pts.push([]);
      for (let j = 0; j <= N; j++) {
        const x = -R + 2 * R * i / N, y = -R + 2 * R * j / N;
        let zz = s.f(x, y);
        if (!isFinite(zz)) zz = NaN;
        pts[i].push([x, y, zz]);
        if (isFinite(zz)) { zmin = Math.min(zmin, zz); zmax = Math.max(zmax, zz); }
      }
    }
    if (!isFinite(zmin)) return;
    const zr = (zmax - zmin) || 1;
    const esc = Math.min(this.w, this.h) / (R * 3.3);
    const zesc = (2 * R) / zr * 0.8;
    const ca = Math.cos(this.ang.a), sa = Math.sin(this.ang.a), cb = Math.cos(this.ang.b), sb = Math.sin(this.ang.b);
    const proy = (x, y, zz) => {
      const z2 = (zz - (zmin + zmax) / 2) * zesc;
      const xr = x * ca - y * sa, yr = x * sa + y * ca;
      const yv = yr * cb + z2 * sb, prof = yr * sb - z2 * cb;
      return [this.w / 2 + xr * esc, this.h / 2 - yv * esc * 0.9 + 0, prof];
    };
    const caras = [];
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const q = [pts[i][j], pts[i + 1][j], pts[i + 1][j + 1], pts[i][j + 1]];
      if (q.some((p) => !isFinite(p[2]))) continue;
      const pr = q.map((p) => proy(p[0], p[1], p[2]));
      const zm = (q[0][2] + q[1][2] + q[2][2] + q[3][2]) / 4;
      caras.push({ pr, prof: pr.reduce((a, p) => a + p[2], 0) / 4, t: (zm - zmin) / zr });
    }
    caras.sort((a, b) => b.prof - a.prof);
    for (const c of caras) {
      const h = 240 - 220 * c.t;
      ctx.fillStyle = `hsl(${h}, 70%, 55%)`; ctx.strokeStyle = 'rgba(0,0,0,0.18)'; ctx.lineWidth = 0.5;
      ctx.beginPath(); c.pr.forEach((p, k) => (k ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.closePath(); ctx.fill(); ctx.stroke();
    }
    // ejes
    const ejes = [[[0, 0, zmin], [R * 1.25, 0, zmin], 'x'], [[0, 0, zmin], [0, R * 1.25, zmin], 'y']];
    ctx.strokeStyle = colorVar('--eje', '#555'); ctx.fillStyle = colorVar('--texto', '#111'); ctx.font = '13px system-ui'; ctx.lineWidth = 1.5;
    for (const [a, b, nm] of ejes) {
      const pa = proy(...a), pb = proy(...b);
      ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke(); ctx.fillText(nm, pb[0] + 4, pb[1]);
    }
    ctx.fillText(`z ∈ [${fmt(zmin)}, ${fmt(zmax)}]`, 8, 16);
    this._leyenda();
  }
}

window.Grafico = Grafico;
