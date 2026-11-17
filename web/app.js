/* IngeCalc — interfaz: cámara → calculadora → soluciones (tarjetas) → paso a paso.
   El motor (Python/SymPy) corre en el celular vía Chaquopy (window.Motor). */
'use strict';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const entrada = $('#entrada');

// ---------------------------------------------------------------------------
// Puente con el motor
// ---------------------------------------------------------------------------
const pendientes = new Map();
let siguienteId = 1;
window.__respuesta = (id, json) => {
  const p = pendientes.get(id);
  if (!p) return;
  pendientes.delete(id);
  try { p(typeof json === 'string' ? JSON.parse(json) : json); } catch (e) { p({ ok: false, error: 'Respuesta inválida del motor.' }); }
};
let motorListo = !window.Motor;
const esperaMotor = [];
window.__motorListo = () => { motorListo = true; esperaMotor.splice(0).forEach((f) => f()); };

function llamar(funcion, ...args) {
  return new Promise((ok) => {
    const id = siguienteId++;
    pendientes.set(id, ok);
    const ir = () => {
      if (window.Motor) window.Motor.llamar(id, funcion, JSON.stringify(args));
      else fetch('/api', { method: 'POST', body: JSON.stringify({ funcion, args }) })
        .then((r) => r.text()).then((t) => window.__respuesta(id, t))
        .catch(() => window.__respuesta(id, { ok: false, error: 'Sin conexión con el motor.' }));
    };
    if (motorListo) ir(); else esperaMotor.push(ir);
  });
}

// ---------------------------------------------------------------------------
// LaTeX
// ---------------------------------------------------------------------------
function notacion(l) {
  return String(l).replace(/\\'/g, "'")
    .replace(/\\arcsin(?![a-z])/g, '\\operatorname{arcsen}').replace(/\\arctan(?![a-z])/g, '\\operatorname{arctg}')
    .replace(/\\sin(?![a-z])/g, '\\operatorname{sen}').replace(/\\tan(?![a-z])/g, '\\operatorname{tg}')
    .replace(/\\sinh(?![a-z])/g, '\\operatorname{senh}');
}
function tex(latex, display) {
  try { return katex.renderToString(notacion(latex), { displayMode: !!display, throwOnError: false, strict: false }); } catch (e) { return esc(latex); }
}
function esc(s) { return String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
function textoMat(t) {
  if (!t) return '';
  let out = '', i = 0;
  const s = String(t);
  while (i < s.length) {
    const a = s.indexOf('\\(', i), b = s.indexOf('$', i);
    if (a === -1 && b === -1) { out += esc(s.slice(i)); break; }
    if (a !== -1 && (b === -1 || a < b)) {
      const f = s.indexOf('\\)', a + 2);
      if (f === -1) { out += esc(s.slice(i)); break; }
      out += esc(s.slice(i, a)) + tex(s.slice(a + 2, f)); i = f + 2;
    } else {
      const f = s.indexOf('$', b + 1);
      if (f === -1) { out += esc(s.slice(i)); break; }
      out += esc(s.slice(i, b)) + tex(s.slice(b + 1, f)); i = f + 1;
    }
  }
  return out;
}
function toast(msg, ms = 2600) {
  const t = $('#toast');
  t.textContent = msg; t.classList.remove('oculto');
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add('oculto'), ms);
}

// ---------------------------------------------------------------------------
// Navegación entre pantallas (con pila para el botón Atrás de Android)
// ---------------------------------------------------------------------------
const PANTALLAS = { camara: '#p-camara', calc: '#p-calc', soluciones: '#p-soluciones', pasos: '#p-pasos',
  temas: '#p-temas', historial: '#p-historial', ayuda: '#p-ayuda' };
const pila = ['camara'];
function mostrarPantalla(nombre) {
  for (const [n, sel] of Object.entries(PANTALLAS)) $(sel).classList.toggle('oculto', n !== nombre);
  if (nombre === 'camara') iniciarCamara(); else detenerCamara();
  if (nombre === 'calc') setTimeout(() => enfocar(), 50);
  if (nombre === 'historial') pintarHistorial();
  if (nombre === 'ayuda') pintarAyuda();
}
function ir(nombre) {
  $('#menu').classList.add('oculto');
  if (pila[pila.length - 1] !== nombre) pila.push(nombre);
  mostrarPantalla(nombre);
}
function atras() {
  if (!$('#menu').classList.contains('oculto')) { $('#menu').classList.add('oculto'); return true; }
  if (pila.length <= 1) return false;
  pila.pop();
  mostrarPantalla(pila[pila.length - 1]);
  return true;
}
window.__atras = () => (atras() ? 1 : 0);
document.addEventListener('click', (e) => {
  const b = e.target.closest('[data-ir]');
  if (b) { ir(b.dataset.ir); return; }
  if (e.target.closest('[data-atras]')) atras();
});
$('#btn-menu').onclick = () => $('#menu').classList.remove('oculto');
$('#menu').onclick = (e) => { if (e.target.id === 'menu') $('#menu').classList.add('oculto'); };

// ---------------------------------------------------------------------------
// Cámara (getUserMedia en el WebView) + recorte del marco + reconocimiento de texto en Kotlin
// ---------------------------------------------------------------------------
const video = $('#video');
let flujo = null, linterna = false;
async function iniciarCamara() {
  if (flujo || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    if (!flujo) sinCamara('Este dispositivo no permite usar la cámara aquí.');
    return;
  }
  try {
    flujo = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false });
    video.srcObject = flujo;
    $('#sin-camara').classList.add('oculto');
  } catch (e) {
    sinCamara('Sin permiso para la cámara. Podés escribir el ejercicio con la calculadora o elegir una imagen.');
  }
}
function detenerCamara() {
  if (flujo) { flujo.getTracks().forEach((t) => t.stop()); flujo = null; linterna = false; }
}
function sinCamara(msg) { $('#sin-camara-msj').textContent = msg; $('#sin-camara').classList.remove('oculto'); }
document.addEventListener('visibilitychange', () => {
  if (document.hidden) detenerCamara(); else if (pila[pila.length - 1] === 'camara') iniciarCamara();
});

// Marco ajustable: arrastrar las esquinas
(function marcoAjustable() {
  const marco = $('#marco');
  let toque = null;
  marco.addEventListener('pointerdown', (e) => {
    const r = marco.getBoundingClientRect();
    const izq = e.clientX < r.left + r.width / 2, arr = e.clientY < r.top + r.height / 2;
    toque = { id: e.pointerId, izq, arr, r, x: e.clientX, y: e.clientY };
    marco.setPointerCapture(e.pointerId);
  });
  marco.addEventListener('pointermove', (e) => {
    if (!toque || e.pointerId !== toque.id) return;
    const W = window.innerWidth, H = window.innerHeight, dx = e.clientX - toque.x, dy = e.clientY - toque.y, r = toque.r;
    let l = r.left, t = r.top, w = r.width, h = r.height;
    if (toque.izq) { l = Math.min(r.right - 90, Math.max(8, r.left + dx)); w = r.right - l; } else { w = Math.max(90, Math.min(W - 8 - r.left, r.width + dx)); }
    if (toque.arr) { t = Math.min(r.bottom - 50, Math.max(60, r.top + dy)); h = r.bottom - t; } else { h = Math.max(50, Math.min(H - 260 - r.top, r.height + dy)); }
    Object.assign(marco.style, { left: l + 'px', top: t + 'px', width: w + 'px', height: h + 'px', right: 'auto' });
  });
  marco.addEventListener('pointerup', () => { toque = null; });
})();

$('#disparador').onclick = () => {
  if (!flujo || !video.videoWidth) { toast('La cámara no está lista.'); return; }
  const vr = video.getBoundingClientRect(), mr = $('#marco').getBoundingClientRect();
  // object-fit: cover -> escala y desplazamiento del video dentro del elemento
  const esc_ = Math.max(vr.width / video.videoWidth, vr.height / video.videoHeight);
  const ox = (vr.width - video.videoWidth * esc_) / 2, oy = (vr.height - video.videoHeight * esc_) / 2;
  const sx = (mr.left - vr.left - ox) / esc_, sy = (mr.top - vr.top - oy) / esc_;
  const sw = mr.width / esc_, sh = mr.height / esc_;
  const c = document.createElement('canvas');
  const k = Math.min(1, 1600 / sw);
  c.width = Math.round(sw * k); c.height = Math.round(sh * k);
  c.getContext('2d').drawImage(video, sx, sy, sw, sh, 0, 0, c.width, c.height);
  const datos = c.toDataURL('image/jpeg', 0.92).split(',')[1];
  $('#disparador').disabled = true;
  toast('Leyendo el ejercicio…', 8000);
  if (window.Motor && window.Motor.ocr) window.Motor.ocr(datos);
  else window.__escaneo('', 'El reconocimiento de texto funciona en la app del celular.');
};
$('#btn-galeria').onclick = () => {
  if (window.Motor && window.Motor.galeria) window.Motor.galeria(); else toast('Disponible en la app del celular.');
};
$('#btn-linterna').onclick = async () => {
  try {
    const pista = flujo && flujo.getVideoTracks()[0];
    if (!pista || !(pista.getCapabilities && pista.getCapabilities().torch)) { toast('Este celular no permite la linterna desde la app.'); return; }
    linterna = !linterna;
    await pista.applyConstraints({ advanced: [{ torch: linterna }] });
  } catch (e) { toast('No se pudo usar la linterna.'); }
};
window.__escaneo = (texto, error) => {
  $('#disparador').disabled = false;
  $('#toast').classList.add('oculto');
  if (error) { toast(error, 4000); return; }
  if (!texto) return;
  ir('calc');
  ponerTexto(texto);
  $('#aviso-entrada').textContent = 'Texto leído por la cámara: revisalo (la cámara puede confundir símbolos) y tocá ⏎.';
};

// ---------------------------------------------------------------------------
// Entrada: modo FÓRMULA (editor visual MathLive, como se escribe a mano) y modo TEXTO (comandos)
// ---------------------------------------------------------------------------
const campo = $('#campo');
let modoTexto = false;
if (window.MathfieldElement) {
  MathfieldElement.fontsDirectory = 'mathlive/fonts/';
  MathfieldElement.soundsDirectory = null;
}
try {
  campo.mathVirtualKeyboardPolicy = 'manual';
  campo.smartFence = true;
  campo.smartSuperscript = true;
  campo.addEventListener('focusin', () => { try { window.mathVirtualKeyboard.hide(); } catch (e) { /* */ } });
  campo.addEventListener('input', () => { $('#aviso-entrada').textContent = ''; programarVivo(); });
  campo.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); resolverEntrada(); } });
} catch (e) { /* sin MathLive: queda el modo texto */ }

/** Índice del paréntesis que cierra al que abre en s[i] */
function cierre(s, i) {
  let n = 0;
  for (let j = i; j < s.length; j++) {
    if (s[j] === '(') n++;
    else if (s[j] === ')') { n--; if (n === 0) return j; }
  }
  return -1;
}
/** ascii-math de MathLive -> sintaxis del motor */
function asciiAMotor(a) {
  let s = String(a || '')
    .replace(/⋅|·|×/g, '*').replace(/\bxx\b/g, '*').replace(/÷/g, '/').replace(/−/g, '-')
    .replace(/∞/g, 'oo').replace(/π/g, 'pi').replace(/≤/g, '<=').replace(/≥/g, '>=')
    .replace(/\{,\}/g, ',').replace(/→/g, '->')
    .replace(/\^@/g, '*pi/180')                       // grados: 30°
    .replace(/(\w)\s+\(/g, '$1(')                     // "sin (x)" -> "sin(x)"
    .replace(/\s+/g, ' ').trim();
  // root(n)(x) -> root(x, n)
  for (let k = s.indexOf('root('); k >= 0; k = s.indexOf('root(', k + 5)) {
    const c1 = cierre(s, k + 4);
    if (c1 < 0 || s[c1 + 1] !== '(') continue;
    const c2 = cierre(s, c1 + 1);
    if (c2 < 0) continue;
    s = s.slice(0, k) + 'root(' + s.slice(c1 + 2, c2) + ', ' + s.slice(k + 5, c1) + ')' + s.slice(c2 + 1);
  }
  // log _b(x) / log_(b)(x) -> log(x, b)
  s = s.replace(/log\s*_\s*(\([^()]*\)|[\w.]+)\s*\(/g, (m, b) => 'log_base[' + b.replace(/^\(|\)$/g, '') + '](');
  for (let k = s.indexOf('log_base['); k >= 0; k = s.indexOf('log_base[', k + 1)) {
    const fb = s.indexOf('](', k);
    const c = cierre(s, fb + 1);
    if (c < 0) break;
    s = s.slice(0, k) + 'log(' + s.slice(fb + 2, c) + ', ' + s.slice(k + 9, fb) + ')' + s.slice(c + 1);
  }
  // pi pegado a otra letra (2pir -> 2pi*r)
  s = s.replace(/pi(?=[a-z])/g, 'pi*');
  return s;
}
/** 9.348469228 -> "9,348469" (coma decimal, hasta 6 decimales) */
function decimalAR(v) {
  const n = Number(v);
  if (!isFinite(n)) return String(v);
  return n.toLocaleString('es-AR', { maximumFractionDigits: Math.abs(n) >= 1e6 ? 2 : 6, useGrouping: false });
}
function textoActual() {
  if (modoTexto || !window.MathfieldElement) return entrada.value.trim();
  try { return asciiAMotor(campo.getValue('ascii-math')); } catch (e) { return ''; }
}
function latexActual() {
  if (modoTexto || !window.MathfieldElement) return null;
  try { return campo.getValue('latex'); } catch (e) { return null; }
}
function mostrarModo() {
  campo.classList.toggle('oculto', modoTexto);
  $('#previa').classList.toggle('oculto', !modoTexto);
  entrada.classList.toggle('oculto', !modoTexto);
  $('.tec-barra [data-accion="abc"]').textContent = modoTexto ? 'f(x)' : 'abc';
}
/** Pone un texto (plantilla, historial, cámara) en modo texto */
function ponerTexto(t) {
  modoTexto = true;
  mostrarModo();
  entrada.value = t;
  entrada.dispatchEvent(new Event('input'));
  setTimeout(() => entrada.focus(), 40);
}
function enfocar() {
  if (modoTexto) entrada.focus(); else campo.focus();
}

// ---------------------------------------------------------------------------
// Teclado matemático
// ---------------------------------------------------------------------------
// [etiqueta (texto o $latex$), lo que inserta (| = posición del cursor), clase]
const TECLADOS = [
  { nombre: '+ −<br>× ÷', teclas: [
    ['( )', '(|)'], ['>', '>'], ['7', '7', 'num'], ['8', '8', 'num'], ['9', '9', 'num'], ['÷', '/'],
    ['$\\frac{\\square{}}{\\square{}}$', '(|)/()'], ['$\\sqrt{\\square{}}$', 'sqrt(|)'], ['4', '4', 'num'], ['5', '5', 'num'], ['6', '6', 'num'], ['×', '*'],
    ['$\\square{}^2$', '^2'], ['x', 'x'], ['1', '1', 'num'], ['2', '2', 'num'], ['3', '3', 'num'], ['−', '-'],
    ['π', 'pi'], ['%', '%'], ['0', '0', 'num'], [',', ','], ['=', '='], ['+', '+'],
  ] },
  { nombre: 'f(x) e<br>log ln', teclas: [
    ['x', 'x'], ['y', 'y'], ['z', 'z'], ['t', 't'], ['e', 'e'], ['$e^\\square{}$', 'e^(|)'],
    ['ln', 'ln(|)'], ['log', 'log(|)'], ['$\\log_\\square{}$', 'log(|, )'], ['$\\square{}^\\square{}$', '^(|)'], ['$\\sqrt[3]{\\square{}}$', 'cbrt(|)'], ['$\\sqrt[n]{\\square{}}$', 'root(|, )'],
    ['|x|', 'abs(|)'], ['n!', '!'], ['<', '<'], ['≤', '<='], ['≥', '>='], ['i', 'i'],
    ['(', '('], [')', ')'], ['[', '['], [']', ']'], [', ', ', ', 'chico'], [';', '; '],
  ] },
  { nombre: 'sen cos<br>tg cot', teclas: [
    ['sen', 'sen(|)', 'chico'], ['cos', 'cos(|)', 'chico'], ['tg', 'tg(|)', 'chico'], ['cot', 'cot(|)', 'chico'], ['sec', 'sec(|)', 'chico'], ['csc', 'csc(|)', 'chico'],
    ['arcsen', 'arcsen(|)', 'chico'], ['arccos', 'arccos(|)', 'chico'], ['arctg', 'arctg(|)', 'chico'], ['senh', 'senh(|)', 'chico'], ['cosh', 'cosh(|)', 'chico'], ['tgh', 'tgh(|)', 'chico'],
    ['π', 'pi'], ['θ', 'theta'], ['°', '*pi/180'], ['x', 'x'], ['(', '('], [')', ')'],
    ['$\\square{}^2$', '^2'], ['$\\frac{\\square{}}{\\square{}}$', '(|)/()'], ['$\\sqrt{\\square{}}$', 'sqrt(|)'], ['=', '='], [', ', ', ', 'chico'], ['t', 't'],
  ] },
  { nombre: 'lím dx<br>∫ Σ ∞', teclas: [
    ['$\\frac{d}{dx}$', 'derivar |'], ['$\\int$', 'integrar |'], ['$\\int_a^b$', 'integrar |, x, , '], ['lím', 'limite |, x->'], ['Σ', 'sumatoria |, k, 1, oo'], ['∞', 'oo'],
    ["y'", "y'"], ["y''", "y''"], ['EDO', 'edo |', 'chico'], ['Laplace', 'laplace |', 'chico'], ['$\\mathcal{L}^{-1}$', 'laplace inversa |'], ['Taylor', 'taylor |, x, 0, 5', 'chico'],
    ['→', '->'], ['0⁺', '0+'], ['área', 'area |, , , ', 'chico'], ['extr.', 'extremos |', 'chico'], ['∂', 'derivar |, y', 'chico'], ['∇', 'gradiente |'],
    ['∬', 'integral doble |, x, , , y, , '], ['tang.', 'recta tangente |, ', 'chico'], ['gráf.', 'graficar |', 'chico'], ['x', 'x'], [', ', ', ', 'chico'], ['=', '='],
  ] },
  { nombre: '[ ] ÁLG.<br>LINEAL', teclas: [
    ['2×2', '[[|, ], [, ]]', 'chico'], ['3×3', '[[|, , ], [, , ], [, , ]]', 'chico'], ['det', 'det |', 'chico'], ['A⁻¹', 'inversa |', 'chico'], ['λ', 'autovalores |', 'chico'], ['Gauss', 'gauss |', 'chico'],
    ['[', '['], [']', ']'], ['[ , ]', '[|, ]', 'chico'], ['·', 'punto |, ', 'chico'], ['×', 'cruz |, ', 'chico'], ['‖v‖', 'norma |', 'chico'],
    ['resolver', 'resolver |', 'chico'], [';', '; '], ['∠', '@'], ['i', 'i'], ['rango', 'rango |', 'chico'], ['=', '='],
    ['−', '-'], [', ', ', ', 'chico'], ['x', 'x'], ['y', 'y'], ['z', 'z'], ['^', '^'],
  ] },
];
// Equivalente en el editor visual (LaTeX de MathLive; #0 = cursor, #? = casilla, #@ = lo anterior)
const EN_FORMULA = {
  '(|)': '\\left(#0\\right)', '(|)/()': '\\frac{#0}{#?}', '/': '\\frac{#@}{#?}', 'sqrt(|)': '\\sqrt{#0}', '^2': '#@^{2}',
  '^(|)': '#@^{#0}', 'e^(|)': 'e^{#0}', 'cbrt(|)': '\\sqrt[3]{#0}', 'root(|, )': '\\sqrt[#?]{#0}', 'ln(|)': '\\ln\\left(#0\\right)',
  'log(|)': '\\log\\left(#0\\right)', 'log(|, )': '\\log_{#?}\\left(#0\\right)', 'abs(|)': '\\left|#0\\right|', 'pi': '\\pi',
  'oo': '\\infty', '*': '\\cdot', '<=': '\\le', '>=': '\\ge', '%': '\\%', ',': '{,}', 'theta': '\\theta', '*pi/180': '^{\\circ}',
  'sen(|)': '\\sin\\left(#0\\right)', 'cos(|)': '\\cos\\left(#0\\right)', 'tg(|)': '\\tan\\left(#0\\right)', 'cot(|)': '\\cot\\left(#0\\right)',
  'sec(|)': '\\sec\\left(#0\\right)', 'csc(|)': '\\csc\\left(#0\\right)', 'arcsen(|)': '\\arcsin\\left(#0\\right)',
  'arccos(|)': '\\arccos\\left(#0\\right)', 'arctg(|)': '\\arctan\\left(#0\\right)', 'senh(|)': '\\sinh\\left(#0\\right)',
  'cosh(|)': '\\cosh\\left(#0\\right)', 'tgh(|)': '\\tanh\\left(#0\\right)',
};
let tecladoActual = 0;
function pintarTeclado() {
  $('#tec-pestanas').innerHTML = TECLADOS.map((t, i) => `<button data-i="${i}" class="${i === tecladoActual ? 'activa' : ''}">${t.nombre}</button>`).join('');
  $$('#tec-pestanas button').forEach((b) => {
    b.addEventListener('pointerdown', (e) => e.preventDefault());
    b.onclick = () => { tecladoActual = +b.dataset.i; pintarTeclado(); };
  });
  const g = $('#tec-grilla');
  g.innerHTML = '';
  for (const [et, val, cls] of TECLADOS[tecladoActual].teclas) {
    const b = document.createElement('button');
    if (cls) b.className = cls;
    b.innerHTML = et.startsWith('$') ? tex(et.slice(1, -1)) : esc(et);
    b.addEventListener('pointerdown', (e) => e.preventDefault()); // no robar el foco
    b.onclick = () => insertar(val);
    g.appendChild(b);
  }
}
function esComando(p) {
  return /^[a-z][a-z ]+ \|/.test(p) || p.includes('[') || ['@', "y'", "y''", '->', '0+', '; ', ', '].includes(p);
}
function insertar(plantilla) {
  if (navigator.vibrate) navigator.vibrate(8);
  if (!modoTexto && window.MathfieldElement) {
    if (esComando(plantilla)) {
      // los comandos (derivar, límite, matrices...) se escriben en modo texto, con lo que ya había en la fórmula
      const actual = textoActual();
      ponerTexto('');
      insertarTexto(plantilla.includes('|') ? plantilla.replace('|', actual + '|') : actual + plantilla);
      return;
    }
    const lat = EN_FORMULA[plantilla];
    campo.focus();
    if (lat) campo.executeCommand(['insert', lat, { focus: true, feedback: false, mode: 'math', format: 'latex', selectionMode: 'placeholder' }]);
    else campo.executeCommand(['typedText', plantilla.replace('|', '')]);
    programarVivo();
    return;
  }
  insertarTexto(plantilla);
}
function insertarTexto(plantilla) {
  const el = entrada;
  const a = el.selectionStart ?? el.value.length, b = el.selectionEnd ?? el.value.length;
  const i = plantilla.indexOf('|');
  const txt = plantilla.replace('|', '');
  el.value = el.value.slice(0, a) + txt + el.value.slice(b);
  const pos = a + (i >= 0 ? i : txt.length);
  el.selectionStart = el.selectionEnd = pos;
  el.focus();
  el.dispatchEvent(new Event('input'));
}
$$('.tec-barra button').forEach((b) => {
  b.addEventListener('pointerdown', (e) => e.preventDefault());
  b.onclick = async () => {
    const accion = b.dataset.accion;
    if (accion === 'historial') { ir('historial'); return; }
    if (accion === 'resolver') { resolverEntrada(); return; }
    if (accion === 'abc') {
      if (!modoTexto) {
        ponerTexto(textoActual());
        entrada.inputMode = 'text'; entrada.blur(); setTimeout(() => entrada.focus(), 60);
      } else {
        // volver al modo fórmula: se convierte el texto si es una expresión sin comando
        const t = entrada.value.trim();
        const r = t ? await llamar('previa', t) : { latex: '' };
        if (r && r.comando) { toast('Los comandos (' + r.comando + ') se escriben en modo texto.'); return; }
        modoTexto = false; entrada.inputMode = 'none'; mostrarModo();
        try { campo.setValue(r && r.latex && !r.latex.startsWith('\\text') ? r.latex : ''); } catch (e) { /* */ }
        campo.focus(); programarVivo();
      }
      return;
    }
    if (!modoTexto && window.MathfieldElement) {
      campo.focus();
      if (accion === 'izq') campo.executeCommand('moveToPreviousChar');
      if (accion === 'der') campo.executeCommand('moveToNextChar');
      if (accion === 'borrar') { campo.executeCommand('deleteBackward'); programarVivo(); }
      return;
    }
    const el = entrada, a = el.selectionStart, z = el.selectionEnd;
    if (accion === 'izq') el.selectionStart = el.selectionEnd = Math.max(0, a - 1);
    if (accion === 'der') el.selectionStart = el.selectionEnd = Math.min(el.value.length, z + 1);
    if (accion === 'borrar') {
      if (a === z && a > 0) { el.value = el.value.slice(0, a - 1) + el.value.slice(z); el.selectionStart = el.selectionEnd = a - 1; }
      else { el.value = el.value.slice(0, a) + el.value.slice(z); el.selectionStart = el.selectionEnd = a; }
      el.dispatchEvent(new Event('input'));
    }
  };
});
// mantener apretado ⌫ borra todo
(function () {
  const b = $('.tec-barra [data-accion="borrar"]');
  let t = null;
  b.addEventListener('pointerdown', () => {
    t = setTimeout(() => {
      entrada.value = ''; entrada.dispatchEvent(new Event('input'));
      try { campo.setValue(''); } catch (e) { /* */ }
      programarVivo(); toast('Entrada borrada');
    }, 650);
  });
  ['pointerup', 'pointerleave', 'pointercancel'].forEach((ev) => b.addEventListener(ev, () => clearTimeout(t)));
})();
pintarTeclado();

// Vista previa (modo texto)
let temporizador = null, ultimaPrevia = '';
entrada.addEventListener('input', () => {
  entrada.style.height = 'auto'; entrada.style.height = Math.min(140, entrada.scrollHeight) + 'px';
  $('#aviso-entrada').textContent = '';
  clearTimeout(temporizador);
  programarVivo();
  const t = entrada.value.trim();
  if (!t) { $('#previa').innerHTML = '<span class="placeholder">Escribí un ejercicio…</span>'; ultimaPrevia = ''; return; }
  temporizador = setTimeout(async () => {
    if (t === ultimaPrevia) return;
    ultimaPrevia = t;
    const r = await llamar('previa', t);
    if (entrada.value.trim() !== t) return;
    $('#previa').innerHTML = r && r.latex ? (r.comando ? `<div class="sobre">${esc(r.comando)}</div>` : '') + tex(r.latex, true)
      : `<span class="placeholder">${esc(t)}</span>`;
  }, 300);
});
entrada.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); resolverEntrada(); } });

// Resultado en vivo mientras se escribe (primera solución, con límite de tiempo corto)
let vivoTimer = null, vivoOcupado = false, vivoPendiente = false;
function programarVivo() {
  clearTimeout(vivoTimer);
  vivoTimer = setTimeout(actualizarVivo, 550);
}
async function actualizarVivo() {
  const t = textoActual();
  const caja = $('#vivo');
  if (!t) { caja.innerHTML = ''; return; }
  if (vivoOcupado) { vivoPendiente = true; return; }
  vivoOcupado = true;
  try {
    const sug = await llamar('sugerencias', t);
    const s0 = Array.isArray(sug) && sug.length ? sug[0] : { entrada: t, titulo: '' };
    const r = await llamar('rapido', s0.entrada);
    if (textoActual() === t) {
      if (r && r.ok && r.resultado_latex) {
        // el resultado exacto arriba y su valor decimal abajo (con coma), como "forma alternativa"
        let principal = resumenResultado(r);
        let decimal = r.valores && r.valores.decimal;
        const k = principal.lastIndexOf('\\approx');
        if (k > 0) { if (!decimal) decimal = principal.slice(k + 7).trim(); principal = principal.slice(0, k).trim(); }
        const exacto = principal.replace(/\s/g, '');
        const dec = decimal && exacto !== String(decimal) && /^-?[\d.]+(e-?\d+)?$/.test(String(decimal))
          ? `<div class="vivo-dec">≈ ${esc(decimalAR(decimal))}</div>` : '';
        caja.innerHTML = `<div class="vivo-tit">${esc(s0.titulo || r.titulo)}</div><div class="formula vivo-res">${tex((/=|\\in|\\text/.test(principal) ? '' : '= ') + principal, true)}</div>${dec}`;
      } else caja.innerHTML = '';
    }
  } finally {
    vivoOcupado = false;
    if (vivoPendiente) { vivoPendiente = false; programarVivo(); }
  }
}

// ---------------------------------------------------------------------------
// Soluciones (tarjetas)
// ---------------------------------------------------------------------------
let resultados = [];         // resultados de cada tarjeta
let graficoActual = null;

function resolverEntrada() {
  const t = textoActual();
  if (!t) { enfocar(); return; }
  guardarHistorial(t);
  ir('soluciones');
  pintarSoluciones(t, latexActual());
}

async function pintarSoluciones(texto, latexOriginal) {
  const lista = $('#lista-sol');
  resultados = [];
  lista.innerHTML = `<div class="tarjeta"><div class="sobre">Tu ejercicio</div>
      <div class="ejercicio"><div class="formula" id="sol-ejercicio">${esc(texto)}</div>
      <button class="lapiz" aria-label="Editar" id="btn-editar">✎</button></div></div>
    <div id="tarjetas"></div>
    <div class="tarjeta revisar"><div class="sobre">Revisar mi trabajo</div><h2>¿Está bien mi resultado?</h2>
      <input id="respuesta" type="text" spellcheck="false" autocapitalize="off" autocomplete="off"
        placeholder="Tu resultado. Ej: 2x·cos(x)   ·   x = 2, x = 3">
      <div class="fila"><button class="boton-sec" id="btn-comprobar">Comprobar</button></div>
      <div id="veredicto"></div></div>`;
  $('#btn-editar').onclick = () => atras();
  $('#btn-comprobar').onclick = () => comprobar(texto);
  $('#respuesta').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); comprobar(texto); } });
  if (latexOriginal) $('#sol-ejercicio').innerHTML = tex(latexOriginal, true);
  else llamar('previa', texto).then((r) => { if (r && r.latex) $('#sol-ejercicio').innerHTML = tex(r.latex, true); });

  const tarjetas = $('#tarjetas');
  tarjetas.innerHTML = `<div class="tarjeta"><div class="cargando"><div class="girando"></div>
    ${motorListo ? 'Resolviendo…' : 'Preparando el motor matemático (solo la primera vez)…'}</div></div>`;
  const sug = await llamar('sugerencias', texto);
  const lista_ = Array.isArray(sug) && sug.length ? sug : [{ titulo: '', entrada: texto }];
  tarjetas.innerHTML = lista_.map((s, i) => `<div class="tarjeta" id="sol-${i}"><div class="sobre">Solución ${lista_.length > 1 ? i + 1 : ''}</div>
      <h2>${esc(s.titulo || '…')}</h2><div class="cargando"><div class="girando"></div>Resolviendo…</div></div>`).join('');
  for (let i = 0; i < lista_.length; i++) {
    const r = await llamar('resolver', lista_[i].entrada, '');
    resultados[i] = r;
    pintarTarjeta(i, lista_[i], r);
  }
}

function pintarTarjeta(i, sug, r) {
  const t = $('#sol-' + i);
  if (!t) return;
  if (!r.ok) {
    if (i > 0) { t.remove(); return; }   // una interpretación alternativa que no aplica: no se muestra
    t.innerHTML = `<div class="sobre">No se pudo resolver</div><p class="error-sol">${esc(r.error)}</p>
      <p class="tiempo">Revisá la escritura o elegí una plantilla en Temas.</p>`;
    return;
  }
  const metodo = r.pasos && r.pasos.length ? r.pasos[0].titulo : r.titulo;
  t.innerHTML = `<div class="sobre">${esc(r.tema)}</div><h2>${esc(sug.titulo || r.titulo)}</h2>
    <div class="formula">${tex(r.entrada_latex || '', true)}</div>
    <div class="metodo"><span class="linea"></span><span>${esc(metodo)}</span></div>
    <div class="formula final">${tex(resumenResultado(r), true)}</div>
    ${r.pasos && r.pasos.length || r.grafico || r.tabla ? `<button class="boton-pasos" data-i="${i}">Ver el desarrollo paso a paso <span>→</span></button>` : ''}`;
  const b = t.querySelector('.boton-pasos');
  if (b) b.onclick = () => abrirPasos(i);
}

function resumenResultado(r) {
  // Se resalta solo el resultado: si el resultado repite el ejercicio ("ejercicio = resultado"), se saca esa parte
  const l = r.resultado_latex || '', e = r.entrada_latex || '';
  if (!l) return '';
  if (e && l.startsWith(e)) {
    const resto = l.slice(e.length).trim();
    if (resto.startsWith('=')) return resto.slice(1).trim();
  }
  return l;
}

async function comprobar(texto) {
  const resp = $('#respuesta').value.trim();
  if (!resp) { $('#respuesta').focus(); return; }
  const v = $('#veredicto');
  v.innerHTML = '<div class="cargando"><div class="girando"></div>Comparando…</div>';
  // se compara contra la interpretación principal (la primera tarjeta)
  const sug = await llamar('sugerencias', texto);
  const principal = Array.isArray(sug) && sug.length ? sug[0].entrada : texto;
  const r = await llamar('resolver', principal, resp);
  const c = r.comprobacion;
  if (!r.ok || !c) { v.innerHTML = `<div class="veredicto nada"><b>No se pudo comparar</b><p>${esc(r.error || '')}</p></div>`; return; }
  const cls = c.correcto === true ? 'bien' : c.correcto === false ? 'mal' : 'nada';
  const titulo = c.correcto === true ? (c.parcial ? '✔ Es equivalente' : '✔ ¡Está bien!') : c.correcto === false ? '✘ Hay un error' : 'ℹ No se pudo comparar';
  let h = `<div class="veredicto ${cls}"><b>${titulo}</b><p>${textoMat(c.mensaje)}</p>`;
  if (c.pista) h += `<p><b>Pista:</b> ${textoMat(c.pista)}</p>`;
  if (c.esperado && c.correcto === false) h += `<p>Resultado correcto:</p><div class="formula">${tex(c.esperado, true)}</div>`;
  v.innerHTML = h + '</div>';
}

// ---------------------------------------------------------------------------
// Paso a paso
// ---------------------------------------------------------------------------
function abrirPasos(i) {
  const r = resultados[i];
  if (!r) return;
  if (graficoActual) { graficoActual.destruir(); graficoActual = null; }
  $('#pasos-titulo').textContent = r.titulo;
  const c = $('#pasos-contenido');
  let h = `<div class="tarjeta resumen"><div class="sobre">${esc(r.tema)}</div>
    <div class="formula">${tex(r.entrada_latex || '', true)}</div>
    <div class="formula final">${tex(r.resultado_latex || '', true)}</div>
    ${r.valores && r.valores.decimal && !/approx/.test(r.resultado_latex || '') ? `<div class="tiempo">≈ ${esc(r.valores.decimal)}</div>` : ''}
    <div class="tiempo">Resuelto en ${r.tiempo} s</div></div>`;
  (r.avisos || []).forEach((a) => { h += `<div class="aviso">${esc(a)}</div>`; });
  if (r.pasos && r.pasos.length) {
    h += '<div class="bloque"><h3>Desarrollo</h3>';
    let n = 0;
    r.pasos.forEach((s, k) => {
      const sub = (s.nivel || 0) > 0;
      if (!sub) n++;
      h += `<div class="paso ${sub ? 'sub' : ''}"><div class="num">${sub ? '·' : n}</div><div>
        <div class="pt">${esc(s.titulo)}</div>
        ${s.latex ? `<div class="formula">${tex(s.latex, true)}</div>` : ''}
        ${s.texto ? `<button class="por-que" data-k="${k}">¿Por qué? ▾</button><div class="px oculto" id="px-${k}">${textoMat(s.texto)}</div>` : ''}
      </div></div>`;
    });
    h += '</div>';
  }
  if (r.grafico) h += '<div class="bloque"><h3>Gráfico</h3><div class="grafico" id="g-pasos"></div></div>';
  if (r.tabla && r.tabla.filas && r.tabla.filas.length) {
    h += '<div class="bloque"><h3>Tabla</h3><div class="tabla-envoltura"><table><thead><tr>' + r.tabla.columnas.map((x) => `<th>${esc(x)}</th>`).join('')
      + '</tr></thead><tbody>' + r.tabla.filas.map((f) => '<tr>' + f.map((x) => `<td>${textoMat(x)}</td>`).join('') + '</tr>').join('') + '</tbody></table></div></div>';
  }
  if (r.extras && r.extras.length) {
    h += '<div class="bloque"><h3>Más información</h3>' + r.extras.map((e) => `<div class="extra"><div class="pt">${esc(e.titulo)}</div>
      ${e.latex ? `<div class="formula">${tex(e.latex, true)}</div>` : ''}${e.texto ? `<div>${textoMat(e.texto)}</div>` : ''}</div>`).join('') + '</div>';
  }
  c.innerHTML = h;
  c.scrollTop = 0;
  $$('.por-que', c).forEach((b) => { b.onclick = () => { $('#px-' + b.dataset.k).classList.toggle('oculto'); b.textContent = b.textContent.includes('▾') ? '¿Por qué? ▴' : '¿Por qué? ▾'; }; });
  ir('pasos');
  if (r.grafico) {
    const g = $('#g-pasos');
    try {
      graficoActual = new Grafico(g, r.grafico);
      if (r.grafico.tipo !== '3d') {
        const ctl = document.createElement('div');
        ctl.className = 'controles-grafico';
        ctl.innerHTML = '<button aria-label="Acercar">+</button><button aria-label="Alejar">−</button><button aria-label="Reiniciar vista">⟲</button>';
        const [mas, menos, rein] = ctl.querySelectorAll('button');
        mas.onclick = () => graficoActual.zoom(0.7); menos.onclick = () => graficoActual.zoom(1.4); rein.onclick = () => graficoActual.reiniciar();
        g.appendChild(ctl);
      }
    } catch (e) { g.textContent = 'No se pudo dibujar el gráfico.'; }
  }
}

// ---------------------------------------------------------------------------
// Temas, historial y ayuda
// ---------------------------------------------------------------------------
function elegirTema(id) {
  $$('#temas button').forEach((b) => b.classList.toggle('activo', b.dataset.id === id));
  const tema = window.TEMAS.find((t) => t.id === id);
  $('#plantillas').innerHTML = '';
  for (const [nombre, ej] of tema.items) {
    const b = document.createElement('button');
    b.innerHTML = `<span class="pt">${esc(nombre)}</span><span class="ej">${esc(ej)}</span>`;
    b.onclick = () => { pila.splice(1); ir('calc'); ponerTexto(ej); };
    $('#plantillas').appendChild(b);
  }
  try { localStorage.setItem('tema', id); } catch (e) { /* */ }
}
$('#temas').innerHTML = window.TEMAS.map((t) => `<button data-id="${t.id}">${esc(t.nombre)}</button>`).join('');
$$('#temas button').forEach((b) => { b.onclick = () => elegirTema(b.dataset.id); });
let temaInicial = 'calculo';
try { temaInicial = localStorage.getItem('tema') || temaInicial; } catch (e) { /* */ }
elegirTema(window.TEMAS.some((t) => t.id === temaInicial) ? temaInicial : 'calculo');

function leerHistorial() { try { return JSON.parse(localStorage.getItem('historial') || '[]'); } catch (e) { return []; } }
function guardarHistorial(t) {
  try { const h = leerHistorial().filter((x) => x !== t); h.unshift(t); localStorage.setItem('historial', JSON.stringify(h.slice(0, 80))); } catch (e) { /* */ }
}
function pintarHistorial() {
  const h = leerHistorial();
  $('#lista-historial').innerHTML = h.length ? h.map((t, i) => `<button data-i="${i}">${esc(t)}</button>`).join('') : '<p class="vacio">Todavía no resolviste ejercicios.</p>';
  $$('#lista-historial button').forEach((b) => {
    b.onclick = () => { pila.splice(1); pila.push('calc'); ponerTexto(h[+b.dataset.i]); resolverEntrada(); };
  });
}
$('#borrar-historial').onclick = () => { try { localStorage.removeItem('historial'); } catch (e) { /* */ } pintarHistorial(); };

function pintarAyuda() {
  if ($('#ayuda').dataset.listo) return;
  $('#ayuda').dataset.listo = '1';
  $('#ayuda').innerHTML = `
<p><b>1. Sacá una foto</b> del ejercicio (ajustá el marco arrastrando las esquinas) o <b>escribilo</b> con la calculadora.
La cámara funciona mejor con texto impreso o letra clara en una línea; siempre podés corregir lo que leyó.</p>
<p><b>2. Elegí una solución.</b> La app muestra las formas de resolverlo (simplificar, raíces, derivada, integral, gráfico…)
y cada una tiene su <b>desarrollo paso a paso</b>, con el “¿por qué?” de cada paso.</p>
<p><b>3. Revisá tu trabajo.</b> En “¿Está bien mi resultado?” escribí lo que te dio: la app lo compara matemáticamente
(acepta formas equivalentes, + C en integrales, soluciones en cualquier orden, redondeos) y si está mal te da una pista.</p>
<h3>Cómo escribir</h3>
<table>
<tr><td><code>x^2</code> <code>2x</code> <code>3sen(2x)</code></td><td>potencias y multiplicación implícita</td></tr>
<tr><td><code>144 : 12</code> <code>0,5</code> <code>1 1/4</code> <code>20% de 800</code></td><td>notación de la escuela (división, coma decimal, mixtos, porcentajes)</td></tr>
<tr><td><code>sqrt(x)</code> <code>cbrt(x)</code> <code>e^x</code> <code>ln(x)</code> <code>log(x)</code></td><td>log = base 10, ln = natural</td></tr>
<tr><td><code>sen tg arcsen arctg</code></td><td>trigonométricas en español o inglés</td></tr>
<tr><td><code>pi</code> <code>oo</code> <code>i</code></td><td>π, infinito, unidad imaginaria</td></tr>
<tr><td><code>[[1,2],[3,4]]</code> <code>[1,2,3]</code></td><td>matriz por filas, vector</td></tr>
</table>
<h3>Comandos</h3>
<p>Escribí la expresión sola y la app propone qué hacer, o usá un comando:</p>
<p><code>derivar</code> · <code>integrar f, x, a, b</code> · <code>limite f, x->a</code> · <code>resolver</code> (sistemas con <code>;</code>) · <code>taylor</code> ·
<code>extremos</code> · <code>area</code> · <code>recta tangente</code> · <code>sumatoria</code> · <code>gradiente</code> · <code>divergencia</code> · <code>rotacional</code> ·
<code>jacobiano</code> · <code>hessiano</code> · <code>integral doble</code> · <code>lagrange</code> · <code>det</code> · <code>inversa</code> · <code>gauss</code> ·
<code>autovalores</code> · <code>edo</code> · <code>laplace</code> · <code>laplace inversa</code> · <code>fourier</code> · <code>newton</code> · <code>biseccion</code> ·
<code>trapecio</code> · <code>simpson</code> · <code>euler</code> · <code>rk4</code> · <code>interpolar</code> · <code>regresion</code> · <code>polar</code> · <code>fasor</code> ·
<code>estadistica</code> · <code>binomial</code> · <code>normal</code> · <code>convertir 9,1 hl a cl</code> · <code>regla de tres</code> · <code>area circulo 6</code> ·
<code>graficar</code> (también <code>polar</code>, <code>parametrica</code>, <code>3d</code>, <code>datos</code>).</p>
<p>En <b>Temas</b> hay ejemplos listos de cada materia: tocá uno y se carga en la calculadora.</p>`;
}

// Arranque: pantalla de cámara
mostrarPantalla('camara');
