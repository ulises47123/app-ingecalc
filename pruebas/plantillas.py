"""Verifica que todas las plantillas de la pantalla Temas se resuelvan (y sus tarjetas sugeridas)."""
import json, os, sys, time
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..', 'motor'))
from ingecalc.api import resolver, sugerencias  # noqa
items = json.load(open(os.path.join(AQUI, 'plantillas.json'), encoding='utf-8'))
mal = 0
for tema, nombre, entrada in items:
    for s in sugerencias(entrada)[:1]:
        t = time.time(); d = resolver(s['entrada'])
        ok = d.get('ok')
        mal += not ok
        print(('OK ' if ok else 'MAL'), f'{time.time()-t:5.1f}s', tema, '|', nombre, '|', s['entrada'], '=>', str(d.get('valor'))[:70] if ok else d.get('error'), flush=True)
print('PLANTILLAS con problema:', mal, 'de', len(items))
