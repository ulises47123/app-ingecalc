#!/usr/bin/env bash
# Entorno: reutiliza el JDK/SDK ya descargados en app-mando-facil/.tools
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERRAMIENTAS="${HERRAMIENTAS:-/c/Users/Usuario/Music/proyectos/app-mando-facil/.tools}"
export JAVA_HOME="$HERRAMIENTAS/jdk"
export ANDROID_HOME="$HERRAMIENTAS/android-sdk"
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export PATH="$JAVA_HOME/bin:$ANDROID_HOME/platform-tools:$PATH"
export MSYS2_ARG_CONV_EXCL="/data;/sdcard;/dev;/system;/storage;/proc;/vendor"
PAQUETE="ar.ingecalc.app"
