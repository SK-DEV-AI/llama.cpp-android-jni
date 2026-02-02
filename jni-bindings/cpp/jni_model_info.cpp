#include "jni_common.h"

extern "C" {

JNIEXPORT jstring JNICALL
Java_com_cortex_app_LlamaEngine_getModelDescription(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return nullptr;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    
    char buf[256];
    int len = llama_model_desc(wrapper->model, buf, sizeof(buf));
    if (len > 0) {
        return env->NewStringUTF(buf);
    }
    return nullptr;
}

JNIEXPORT jlong JNICALL
Java_com_cortex_app_LlamaEngine_getModelParameterCount(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return 0;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    return (jlong)llama_model_n_params(wrapper->model);
}

JNIEXPORT jlong JNICALL
Java_com_cortex_app_LlamaEngine_getModelSize(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return 0;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    return (jlong)llama_model_size(wrapper->model);
}

JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_getModelEmbeddingSize(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return 0;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    return (jint)llama_model_n_embd(wrapper->model);
}

JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_getModelLayerCount(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return 0;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    return (jint)llama_model_n_layer(wrapper->model);
}

JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_getModelHeadCount(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return 0;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    return (jint)llama_model_n_head(wrapper->model);
}

JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_getModelHeadCountKV(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return 0;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    return (jint)llama_model_n_head_kv(wrapper->model);
}

JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_getModelContextSize(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return 0;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    return (jint)llama_model_n_ctx_train(wrapper->model);
}

JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_getModelVocabSize(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return 0;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
    return (jint)llama_vocab_n_tokens(vocab);
}

JNIEXPORT jboolean JNICALL
Java_com_cortex_app_LlamaEngine_modelHasEncoder(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return JNI_FALSE;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    return llama_model_has_encoder(wrapper->model) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_cortex_app_LlamaEngine_modelHasDecoder(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return JNI_FALSE;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    return llama_model_has_decoder(wrapper->model) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_cortex_app_LlamaEngine_modelIsRecurrent(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return JNI_FALSE;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    return llama_model_is_recurrent(wrapper->model) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jstring JNICALL
Java_com_cortex_app_LlamaEngine_getModelChatTemplate(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return nullptr;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    
    const char* tmpl = llama_model_chat_template(wrapper->model, nullptr);
    if (tmpl && tmpl[0] != '\0') {
        return env->NewStringUTF(tmpl);
    }
    return nullptr;
}

// Context Getters
JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_getContextSize(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return 0;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    return (jint)llama_n_ctx(wrapper->ctx);
}

JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_getBatchSize(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return 0;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    return (jint)llama_n_batch(wrapper->ctx);
}

// Performance Metrics Control
JNIEXPORT void JNICALL
Java_com_cortex_app_LlamaEngine_resetPerformanceMetrics(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    llama_perf_context_reset(wrapper->ctx);
}

JNIEXPORT jstring JNICALL
Java_com_cortex_app_LlamaEngine_printPerformanceMetrics(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return nullptr;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    
    // Get performance data
    llama_perf_context_data perf = llama_perf_context(wrapper->ctx);
    
    // Calculate tokens per second
    double tps_prompt = perf.t_p_eval_ms > 0 ? (1000.0 * perf.n_p_eval / perf.t_p_eval_ms) : 0;
    double tps_eval = perf.t_eval_ms > 0 ? (1000.0 * perf.n_eval / perf.t_eval_ms) : 0;
    
    // Format as a readable string
    char buf[512];
    snprintf(buf, sizeof(buf),
        "Performance Metrics:\n"
        "  Prompt: %d tokens @ %.2f T/s (%.2f ms)\n"
        "  Predict: %d tokens @ %.2f T/s (%.2f ms)\n"
        "  Total: %d tokens\n",
        perf.n_p_eval, tps_prompt, perf.t_p_eval_ms,
        perf.n_eval, tps_eval, perf.t_eval_ms,
        perf.n_p_eval + perf.n_eval
    );
    
    return env->NewStringUTF(buf);
}

// Model Metadata Functions

/**
 * Get the number of metadata key/value pairs in the model.
 */
JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_getModelMetadataCount(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return 0;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    return (jint)llama_model_meta_count(wrapper->model);
}

/**
 * Get a metadata value by key.
 * Returns null if the key doesn't exist.
 */
JNIEXPORT jstring JNICALL
Java_com_cortex_app_LlamaEngine_getModelMetadataValue(JNIEnv* env, jobject, jlong contextPtr, jstring key) {
    if (contextPtr == 0 || key == nullptr) return nullptr;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    
    const char* keyStr = env->GetStringUTFChars(key, nullptr);
    if (!keyStr) return nullptr;
    
    char buf[1024];
    int len = llama_model_meta_val_str(wrapper->model, keyStr, buf, sizeof(buf));
    
    env->ReleaseStringUTFChars(key, keyStr);
    
    if (len > 0) {
        return env->NewStringUTF(buf);
    }
    return nullptr;
}

/**
 * Get all metadata as a single JSON string.
 * Format: {"key1": "value1", "key2": "value2", ...}
 */
JNIEXPORT jstring JNICALL
Java_com_cortex_app_LlamaEngine_getAllModelMetadata(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return nullptr;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    
    int32_t count = llama_model_meta_count(wrapper->model);
    if (count <= 0) {
        return env->NewStringUTF("{}");
    }
    
    // Build JSON string
    std::string json = "{";
    bool first = true;
    
    for (int32_t i = 0; i < count; i++) {
        char keyBuf[256];
        char valBuf[1024];
        
        int keyLen = llama_model_meta_key_by_index(wrapper->model, i, keyBuf, sizeof(keyBuf));
        int valLen = llama_model_meta_val_str_by_index(wrapper->model, i, valBuf, sizeof(valBuf));
        
        if (keyLen > 0 && valLen > 0) {
            if (!first) {
                json += ",";
            }
            first = false;
            
            // Simple JSON escaping (handle quotes and backslashes)
            json += "\"";
            json += keyBuf;
            json += "\":\"";
            
            // Escape special characters in value
            for (int j = 0; j < valLen && valBuf[j] != '\0'; j++) {
                char c = valBuf[j];
                if (c == '"') {
                    json += "\\\"";
                } else if (c == '\\') {
                    json += "\\\\";
                } else if (c == '\n') {
                    json += "\\n";
                } else if (c == '\r') {
                    json += "\\r";
                } else if (c == '\t') {
                    json += "\\t";
                } else if (static_cast<unsigned char>(c) >= 0x20) {
                    json += c;
                }
                // Skip control characters
            }
            
            json += "\"";
        }
    }
    
    json += "}";
    
    return env->NewStringUTF(json.c_str());
}

// Chat Templates

/**
 * Get list of built-in chat templates supported by llama.cpp.
 * Returns a JSON array of template names.
 */
JNIEXPORT jstring JNICALL
Java_com_cortex_app_LlamaEngine_getBuiltinChatTemplates(JNIEnv* env, jobject) {
    // Get the count first
    int32_t count = llama_chat_builtin_templates(nullptr, 0);
    if (count <= 0) {
        return env->NewStringUTF("[]");
    }
    
    // Allocate array for templates
    const char** templates = new const char*[count];
    llama_chat_builtin_templates(templates, count);
    
    // Build JSON array
    std::string json = "[";
    for (int32_t i = 0; i < count; i++) {
        if (i > 0) json += ",";
        json += "\"";
        if (templates[i]) {
            json += templates[i];
        }
        json += "\"";
    }
    json += "]";
    
    delete[] templates;
    
    return env->NewStringUTF(json.c_str());
}

/**
 * Set causal attention mode.
 * If true, the model will only attend to past tokens.
 * 
 * JNI Signature: setCausalAttention(JZ)V
 * 
 * @param contextPtr The context pointer
 * @param causalAttn true for causal attention, false for bidirectional
 */
JNIEXPORT void JNICALL
Java_com_cortex_app_LlamaEngine_nativeSetCausalAttention(
        JNIEnv* env,
        jobject,
        jlong contextPtr,
        jboolean causalAttn) {
    
    if (contextPtr == 0) {
        LOGE("setCausalAttention: null context");
        return;
    }
    
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    llama_set_causal_attn(wrapper->ctx, causalAttn);
    
    LOGD("Causal attention set to %s", causalAttn ? "true" : "false");
}

}
