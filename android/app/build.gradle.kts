plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

android {
    namespace = "ar.ingecalc.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "ar.ingecalc.app"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0.0"
        resourceConfigurations += listOf("es")
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a") }
    }
    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    // La interfaz web vive en ../../web (se comparte con el servidor de desarrollo de la PC)
    sourceSets["main"].assets.srcDir(layout.buildDirectory.dir("web-assets"))
}

val copiarWeb by tasks.registering(Sync::class) {
    from("../../web")
    into(layout.buildDirectory.dir("web-assets/web"))
}
tasks.named("preBuild") { dependsOn(copiarWeb) }

chaquopy {
    defaultConfig {
        version = "3.11"
        buildPython("C:/Users/Usuario/AppData/Local/Programs/Python/Python311/python.exe")
        pip {
            install("mpmath==1.3.0")
            install("sympy==1.14.0")
        }
        // .pyc precompilados: arranque más rápido
        pyc { src = true; pip = true }
    }
    sourceSets {
        getByName("main") { srcDir("../../motor") }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("com.google.mlkit:text-recognition:16.0.1")
}
