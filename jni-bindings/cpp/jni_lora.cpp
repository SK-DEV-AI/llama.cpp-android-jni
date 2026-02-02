#include "jni_common.h"

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_cortex_app_LlamaEngine_loadLora(JNIEnv* env, jobject, jlong contextPtr, jstring path, jfloat scale) {
    if (contextPtr == 0) return false;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;

    const char* path_str = env->GetStringUTFChars(path, nullptr);
    
    // Load adapter
    llama_adapter_lora* adapter = llama_adapter_lora_init(wrapper->model, path_str);
    env->ReleaseStringUTFChars(path, path_str);

    if (!adapter) {
        LOGE("Failed to load LoRA adapter");
        return false;
    }

    // Apply to context
    int32_t res = llama_set_adapter_lora(wrapper->ctx, adapter, scale);
    if (res != 0) {
        LOGE("Failed to set LoRA adapter: %d", res);
        return false;
    }

    // Track the loaded adapter
    wrapper->loaded_loras.push_back({path_str, adapter, scale});

    LOGD("LoRA adapter loaded and applied: %s (scale=%.2f)", path_str, scale);
    return true;
}

JNIEXPORT void JNICALL
Java_com_cortex_app_LlamaEngine_clearLoras(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    
    llama_clear_adapter_lora(wrapper->ctx);
    
    // Clear our tracking but don't free adapters - llama_clear_adapter_lora handles that
    wrapper->loaded_loras.clear();
    
    LOGD("All LoRA adapters cleared.");
}

JNIEXPORT jboolean JNICALL
Java_com_cortex_app_LlamaEngine_removeLora(JNIEnv* env, jobject, jlong contextPtr, jstring path) {
    if (contextPtr == 0 || path == nullptr) return false;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;

    const char* path_str = env->GetStringUTFChars(path, nullptr);
    if (!path_str) return false;

    // Find the adapter in our tracking
    auto it = std::find_if(wrapper->loaded_loras.begin(), wrapper->loaded_loras.end(),
        [path_str](const LoraAdapterInfo& info) {
            return info.path == path_str;
        });

    if (it == wrapper->loaded_loras.end()) {
        LOGW("LoRA adapter not found: %s", path_str);
        env->ReleaseStringUTFChars(path, path_str);
        return false;
    }

    // Remove from context
    int32_t res = llama_rm_adapter_lora(wrapper->ctx, it->adapter);
    
    env->ReleaseStringUTFChars(path, path_str);

    if (res < 0) {
        LOGE("Failed to remove LoRA adapter: %d", res);
        return false;
    }

    // Remove from tracking
    wrapper->loaded_loras.erase(it);

    LOGD("LoRA adapter removed: %s", path_str);
    return true;
}

JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_getLoraCount(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return 0;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    return static_cast<jint>(wrapper->loaded_loras.size());
}

JNIEXPORT jobjectArray JNICALL
Java_com_cortex_app_LlamaEngine_getLoadedLoras(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return nullptr;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;

    // Create String array
    jobjectArray result = env->NewObjectArray(
        wrapper->loaded_loras.size(),
        env->FindClass("java/lang/String"),
        nullptr
    );

    if (!result) return nullptr;

    // Fill array with paths
    for (size_t i = 0; i < wrapper->loaded_loras.size(); i++) {
        jstring path = env->NewStringUTF(wrapper->loaded_loras[i].path.c_str());
        env->SetObjectArrayElement(result, i, path);
        env->DeleteLocalRef(path);
    }

    return result;
}

/**
 * Apply a control vector (cvec) to the context.
 * Control vectors steer model behavior in specific directions.
 * 
 * Note: llama_apply_adapter_cvec requires raw float data, not a file path.
 * This simplified implementation returns false as control vectors need
 * to be loaded and parsed from files externally.
 *
 * JNI Signature: applyControlVector(JLjava/lang/String;F)Z
 */
JNIEXPORT jboolean JNICALL
Java_com_cortex_app_LlamaEngine_applyControlVector(
        JNIEnv* env,
        jobject,
        jlong contextPtr,
        jstring path,
        jfloat strength) {

    if (contextPtr == 0 || path == nullptr) return JNI_FALSE;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;

    const char* path_str = env->GetStringUTFChars(path, nullptr);
    if (!path_str) return JNI_FALSE;

    // Note: llama_apply_adapter_cvec takes raw float data, not a file path
    // The proper implementation would need to:
    // 1. Load the control vector file (GGUF format)
    // 2. Extract the float array data
    // 3. Call llama_apply_adapter_cvec with the data
    
    LOGW("Control vectors require raw float data, not file paths.");
    LOGW("To use control vectors, load the data externally and pass float[] to a custom implementation.");
    
    env->ReleaseStringUTFChars(path, path_str);
    
    // Return false as we can't process file paths directly
    return JNI_FALSE;
}

/**
 * Get LoRA adapter metadata.
 *
 * JNI Signature: getLoraMetadata(JLjava/lang/String;)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_com_cortex_app_LlamaEngine_getLoraMetadata(
        JNIEnv* env,
        jobject,
        jlong contextPtr,
        jstring path) {

    if (contextPtr == 0 || path == nullptr) return nullptr;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;

    const char* path_str = env->GetStringUTFChars(path, nullptr);
    if (!path_str) return nullptr;

    // Load adapter temporarily to get metadata
    llama_adapter_lora* adapter = llama_adapter_lora_init(wrapper->model, path_str);
    env->ReleaseStringUTFChars(path, path_str);

    if (!adapter) {
        return nullptr;
    }

    // Get adapter info
    const char* metadata = "{}"; // Simplified - actual metadata extraction would require GGUF parsing
    jstring result = env->NewStringUTF(metadata);

    // Note: We don't apply the adapter, just load it for metadata
    // In a real implementation, you'd extract metadata from the GGUF file
    llama_adapter_lora_free(adapter);

    return result;
}

} // extern "C"