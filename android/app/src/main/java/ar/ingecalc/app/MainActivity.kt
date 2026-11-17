package ar.ingecalc.app

import android.annotation.SuppressLint
import android.app.AlertDialog
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.webkit.ConsoleMessage
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.chaquo.python.Python
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    lateinit var web: WebView
    private var paginaLista = false
    @Volatile private var motorListo = false
    private val hilos = Executors.newFixedThreadPool(2)
    private var fotoUri: Uri? = null

    private val tomarFoto = registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        val u = fotoUri
        if (ok && u != null) reconocer(u) else js("window.__escaneo('', null)")
    }
    private val elegirImagen = registerForActivityResult(ActivityResultContracts.GetContent()) { u ->
        if (u != null) reconocer(u) else js("window.__escaneo('', null)")
    }
    private var pedidoCamara: android.webkit.PermissionRequest? = null
    private val permisoCamara = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        val p = pedidoCamara ?: return@registerForActivityResult
        pedidoCamara = null
        if (ok) p.grant(arrayOf(android.webkit.PermissionRequest.RESOURCE_VIDEO_CAPTURE)) else p.deny()
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        crearWeb()

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                // cerrar el panel si está abierto; si no, salir
                // la página maneja su propia pila de pantallas; si ya está en el inicio, se sale
                web.evaluateJavascript("window.__atras ? window.__atras() : 0") { r ->
                    if (r != "1") finish()
                }
            }
        })

        // Arrancar Python y precargar SymPy en segundo plano
        hilos.execute {
            val t0 = System.currentTimeMillis()
            Log.i(TAG, "Iniciando motor…")
            val py = Python.getInstance()
            // Diagnóstico: si el arranque se traba más de 30 s, se guarda la traza de todos los hilos de Python
            val fh = try {
                val f = py.builtins.callAttr("open", File(getExternalFilesDir(null), "traza_arranque.txt").path, "w")
                py.getModule("faulthandler").also {
                    it.callAttr("dump_traceback_later", 30, false, f)
                }
            } catch (e: Throwable) { null }
            try {
                py.getModule("ingecalc.api").callAttr("calentar")
            } catch (e: Throwable) {
                Log.e(TAG, "Error al iniciar el motor", e)
            }
            try { fh?.callAttr("cancel_dump_traceback_later") } catch (e: Throwable) { }
            Log.i(TAG, "Motor listo en ${System.currentTimeMillis() - t0} ms")
            motorListo = true
            runOnUiThread { if (paginaLista) js("window.__motorListo && window.__motorListo()") }
            intent?.getStringExtra("autoprueba")?.let { ruta -> autoprueba(ruta) }
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun crearWeb() {
        paginaLista = false
        web = WebView(this)
        // El WebView ignora su propio padding: se lo envuelve en un contenedor que sí lo respeta
        val contenedor = android.widget.FrameLayout(this)
        contenedor.setBackgroundColor(android.graphics.Color.BLACK)
        contenedor.addView(web, android.widget.FrameLayout.LayoutParams(-1, -1))
        setContentView(contenedor)
        // Android 15 dibuja de borde a borde: dejar lugar a la barra de estado, la de navegación y el teclado
        ViewCompat.setOnApplyWindowInsetsListener(contenedor) { v, insets ->
            val b = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.ime())
            v.setPadding(b.left, b.top, b.right, b.bottom)
            WindowInsetsCompat.CONSUMED
        }
        if (applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE != 0) {
            WebView.setWebContentsDebuggingEnabled(true)
        }
        web.settings.javaScriptEnabled = true
        web.settings.domStorageEnabled = true
        web.settings.allowFileAccess = true
        web.settings.textZoom = 100
        web.addJavascriptInterface(Puente(this), "Motor")
        web.settings.mediaPlaybackRequiresUserGesture = false
        web.webChromeClient = object : WebChromeClient() {
            // La página pide la cámara (getUserMedia): se concede si el usuario dio el permiso de Android
            override fun onPermissionRequest(request: android.webkit.PermissionRequest) {
                runOnUiThread {
                    if (!request.resources.contains(android.webkit.PermissionRequest.RESOURCE_VIDEO_CAPTURE)) {
                        request.deny(); return@runOnUiThread
                    }
                    if (checkSelfPermission(android.Manifest.permission.CAMERA) == android.content.pm.PackageManager.PERMISSION_GRANTED) {
                        request.grant(arrayOf(android.webkit.PermissionRequest.RESOURCE_VIDEO_CAPTURE))
                    } else {
                        pedidoCamara = request
                        permisoCamara.launch(android.Manifest.permission.CAMERA)
                    }
                }
            }

            override fun onConsoleMessage(m: ConsoleMessage): Boolean {
                Log.i(TAG, "JS ${m.sourceId()}:${m.lineNumber()} ${m.message()}")
                return true
            }
        }
        web.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView, url: String) {
                paginaLista = true
                if (motorListo) js("window.__motorListo && window.__motorListo()")
            }

            // Si el sistema mata el proceso que dibuja la página (poca memoria), se recrea en vez de cerrar la app
            override fun onRenderProcessGone(view: WebView, detail: android.webkit.RenderProcessGoneDetail): Boolean {
                Log.w(TAG, "WebView caído (sin memoria=${!detail.didCrash()}); se recrea")
                ((view.parent as? android.view.ViewGroup)?.parent as? android.view.ViewGroup)?.removeAllViews()
                view.destroy()
                crearWeb()
                return true
            }
        }
        web.loadUrl("file:///android_asset/web/index.html")
    }

    fun ejecutar(id: Int, funcion: String, args: String) {
        hilos.execute {
            val r = try {
                Python.getInstance().getModule("ingecalc.api").callAttr("llamar", funcion, args).toString()
            } catch (e: Throwable) {
                Log.e(TAG, "Error en $funcion", e)
                JSONObject().put("ok", false).put("error", "Error interno: ${e.message?.take(200)}").toString()
            }
            runOnUiThread { js("window.__respuesta($id, ${JSONObject.quote(r)})") }
        }
    }

    fun js(codigo: String) {
        if (!isFinishing) web.evaluateJavascript(codigo, null)
    }

    fun escanear() {
        AlertDialog.Builder(this)
            .setTitle("Escanear ejercicio")
            .setItems(arrayOf("Sacar foto", "Elegir imagen de la galería")) { _, i ->
                if (i == 0) {
                    val dir = File(cacheDir, "fotos").apply { mkdirs() }
                    val f = File(dir, "ejercicio.jpg")
                    val u = FileProvider.getUriForFile(this, "ar.ingecalc.app.archivos", f)
                    fotoUri = u
                    tomarFoto.launch(u)
                } else elegirImagen.launch("image/*")
            }
            .show()
    }

    fun abrirGaleria() = elegirImagen.launch("image/*")

    fun reconocerBase64(b64: String) {
        hilos.execute {
            try {
                val bytes = android.util.Base64.decode(b64, android.util.Base64.DEFAULT)
                val bmp = android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                runOnUiThread { procesar(InputImage.fromBitmap(bmp, 0)) }
            } catch (e: Exception) {
                runOnUiThread { js("window.__escaneo('', ${JSONObject.quote("No se pudo leer la foto.")})") }
            }
        }
    }

    private fun reconocer(u: Uri) {
        try {
            procesar(InputImage.fromFilePath(this, u))
        } catch (e: Exception) {
            js("window.__escaneo('', ${JSONObject.quote("No se pudo abrir la imagen.")})")
        }
    }

    private fun procesar(img: InputImage) {
        try {
            TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS).process(img)
                .addOnSuccessListener { res ->
                    val texto = Escaneo.aExpresion(res.textBlocks.flatMap { b -> b.lines.map { it.text } })
                    js("window.__escaneo(${JSONObject.quote(texto)}, ${if (texto.isBlank()) JSONObject.quote("No se reconoció texto en la imagen.") else "null"})")
                }
                .addOnFailureListener { e -> js("window.__escaneo('', ${JSONObject.quote("No se pudo leer la imagen: ${e.message}")})") }
        } catch (e: Exception) {
            js("window.__escaneo('', ${JSONObject.quote("No se pudo procesar la imagen.")})")
        }
    }

    /** Corre el banco de pruebas dentro del celular: adb shell am start ... --es autoprueba <ruta.json> */
    private fun autoprueba(ruta: String) {
        val salida = File(getExternalFilesDir(null), "resultado_autoprueba.json").absolutePath
        Log.i(TAG, "AUTOPRUEBA inicio: $ruta")
        try {
            val res = Python.getInstance().getModule("ingecalc.autoprueba").callAttr("correr", ruta, salida)
            Log.i(TAG, "AUTOPRUEBA fin: $res -> $salida")
        } catch (e: Throwable) {
            Log.e(TAG, "AUTOPRUEBA error", e)
        }
    }

    override fun onDestroy() {
        hilos.shutdownNow()
        super.onDestroy()
    }

    companion object { const val TAG = "IngeCalc" }
}
