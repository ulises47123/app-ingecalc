package ar.ingecalc.app

import android.webkit.JavascriptInterface

/** Lo que la página puede llamar como window.Motor.* */
class Puente(private val act: MainActivity) {
    @JavascriptInterface
    fun llamar(id: Int, funcion: String, args: String) = act.ejecutar(id, funcion, args)

    @JavascriptInterface
    fun escanear() = act.runOnUiThread { act.escanear() }

    /** Recorte del marco de la cámara (JPEG en base64) -> texto reconocido */
    @JavascriptInterface
    fun ocr(base64: String) = act.reconocerBase64(base64)

    @JavascriptInterface
    fun galeria() = act.runOnUiThread { act.abrirGaleria() }
}

/** Convierte el texto que reconoce la cámara a la sintaxis de la app (lo mejor posible). */
object Escaneo {
    private val superindices = mapOf('²' to "^2", '³' to "^3", '⁴' to "^4", '¹' to "^1", '⁰' to "^0", 'ⁿ' to "^n")

    fun aExpresion(lineas: List<String>): String {
        var t = lineas.joinToString(" ").trim()
        superindices.forEach { (k, v) -> t = t.replace(k.toString(), v) }
        t = t.replace('×', '*').replace('÷', '/').replace('−', '-').replace('—', '-').replace('–', '-')
            .replace("·", "*").replace('{', '(').replace('}', ')')
        // "dx" final de una integral
        t = t.replace(Regex("""\s+d([xyt])\s*$"""), " d$1")
        // O confundida con 0 entre números, l con 1
        t = t.replace(Regex("""(?<=\d)[Oo](?=\d)"""), "0")
        // ∫ al principio -> integrar
        if (t.startsWith("∫")) t = "integrar " + t.drop(1).trim()
        if (t.lowercase().startsWith("lim")) t = "limite " + t.drop(3).trim()
        return t
    }
}
