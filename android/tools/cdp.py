"""Maneja la interfaz de la app en el celular por Chrome DevTools Protocol (solo builds debug).
Uso: python tools/cdp.py "<serial adb>" ejercicios.txt [captura_dir]
Cada línea: ejercicio  ||  respuesta (opcional). Muestra título, resultado y errores de JS."""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

import websocket

ADB = 'adb'
PUERTO = 9333


def adb(serial, *args):
    return subprocess.run([ADB, '-s', serial, *args], capture_output=True, text=True).stdout


def conectar(serial):
    socks = re.findall(r'@(webview_devtools_remote_\d+)', adb(serial, 'shell', 'cat', '/proc/net/unix'))
    if not socks:
        raise SystemExit('No hay WebView depurable (¿app abierta y build debug?)')
    adb(serial, 'forward', f'tcp:{PUERTO}', f'localabstract:{socks[-1]}')
    paginas = json.load(urllib.request.urlopen(f'http://127.0.0.1:{PUERTO}/json'))
    pag = [p for p in paginas if 'index.html' in p.get('url', '')][0]
    return websocket.create_connection(pag['webSocketDebuggerUrl'], timeout=120, suppress_origin=True)


class CDP:
    def __init__(self, ws):
        self.ws, self.n, self.consola = ws, 0, []

    def cmd(self, metodo, **params):
        self.n += 1
        self.ws.send(json.dumps({'id': self.n, 'method': metodo, 'params': params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get('method') == 'Runtime.exceptionThrown':
                self.consola.append(m['params']['exceptionDetails'].get('exception', {}).get('description', '?'))
            if m.get('id') == self.n:
                return m.get('result', {})

    def js(self, codigo, esperar=True):
        r = self.cmd('Runtime.evaluate', expression=codigo, awaitPromise=esperar, returnByValue=True)
        return r.get('result', {}).get('value')


PROBAR = r"""
(async () => {
  const entrada = document.querySelector('#entrada');
  entrada.value = %s;
  const resp = %s;
  if (resp) { document.querySelector('#respuesta').value = resp; }
  document.querySelector('#resultado').innerHTML = '';
  const t0 = performance.now();
  await resolver(!!resp);
  const r = ultimo || {};
  const tarjetas = [...document.querySelectorAll('#resultado .tarjeta')].length;
  const err = document.querySelector('#estado.error');
  const katexMal = document.querySelectorAll('#resultado .katex-error').length;
  const lienzo = document.querySelector('#resultado canvas');
  let pintado = null;
  if (lienzo) {
    const c = lienzo.getContext('2d'), d = c.getImageData(0, 0, lienzo.width, lienzo.height).data;
    let colores = 0;
    for (let i = 0; i < d.length; i += 4 * 97) { const r_ = d[i], g = d[i+1], b = d[i+2]; if (Math.max(r_, g, b) - Math.min(r_, g, b) > 60) colores++; }
    pintado = colores;
  }
  const anchoMal = document.documentElement.scrollWidth > window.innerWidth + 2;
  return { ms: Math.round(performance.now() - t0), titulo: r.titulo, valor: r.valor, tarjetas,
           error: err ? err.innerText : null, katexMal, grafico: r.grafico ? r.grafico.tipo : null, pintado,
           comprobacion: r.comprobacion ? (r.comprobacion.correcto + ' ' + r.comprobacion.mensaje) : null, anchoMal };
})()
"""


def main():
    serial, archivo = sys.argv[1], sys.argv[2]
    capturas = sys.argv[3] if len(sys.argv) > 3 else None
    cdp = CDP(conectar(serial))
    cdp.cmd('Runtime.enable')
    lineas = [l.strip() for l in open(archivo, encoding='utf-8') if l.strip() and not l.startswith('#')]
    malos = 0
    for i, l in enumerate(lineas, 1):
        ej, _, resp = l.partition('||')
        r = cdp.js(PROBAR % (json.dumps(ej.strip()), json.dumps(resp.strip())))
        problema = (not r) or r.get('error') or r.get('katexMal') or r.get('anchoMal') or (r.get('grafico') and not r.get('pintado'))
        malos += bool(problema)
        print(('MAL ' if problema else 'ok  ') + f"{i:3d} {r.get('ms') if r else '?':>6}ms {ej.strip()[:55]:55} -> "
              + json.dumps({k: v for k, v in (r or {}).items() if k not in ('ms',) and v not in (None, 0, False)}, ensure_ascii=False)[:230],
              flush=True)
        if capturas and (problema or i % 5 == 0):
            png = cdp.cmd('Page.captureScreenshot', format='png').get('data')
            if png:
                import base64
                open(os.path.join(capturas, f'ui_{i:03d}.png'), 'wb').write(base64.b64decode(png))
    if cdp.consola:
        print('Excepciones JS:', *cdp.consola[:20], sep='\n  ')
    print(f'Problemas: {malos} de {len(lineas)}')


if __name__ == '__main__':
    main()
