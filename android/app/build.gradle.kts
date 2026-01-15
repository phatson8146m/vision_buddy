plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.ai_depth_object_app"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = "27.0.12077973"

    compileOptions {
        // ✅ ใช้ Java 17
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        // ✅ ใช้ JVM target 17
        jvmTarget = "17"
    }

    defaultConfig {
        applicationId = "com.example.ai_depth_object_app"
        // ✅ ยืนยัน minSdk 21 (เข้ากับ camera 0.9.x และ tflite 0.9.x)
        minSdk = maxOf(21, flutter.minSdkVersion)
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    buildTypes {
        release {
            // ใส่ signingConfig ของคุณถ้า build release จริง
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    // (ตัวเลือก) ลดขนาด APK โดยแบ่งตาม ABI
    // เปิดคอมเมนต์ถ้าต้องการ
    /*
    splits {
        abi {
            isEnable = true
            reset()
            include("armeabi-v7a", "arm64-v8a", "x86_64")
            isUniversalApk = false
        }
    }
    */
}

flutter {
    source = "../.."
}
