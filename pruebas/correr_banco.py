"""Corre el banco de 500 ejercicios contra el motor en la PC.
Uso: python pruebas/correr_banco.py [desde] [hasta]"""
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..', 'motor'))
from ingecalc.autoprueba import correr  # noqa

desde = int(sys.argv[1]) if len(sys.argv) > 1 else 1
hasta = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
correr(os.path.join(AQUI, 'banco_500.json'), os.path.join(AQUI, 'resultado_pc.json'), desde=desde, hasta=hasta,
       eco=lambda s: print(s, flush=True))
