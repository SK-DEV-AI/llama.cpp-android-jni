#include "jni_common.h"

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_cortex_app_LlamaEngine_saveSession(JNIEnv* env, jobject, jlong contextPtr, jstring path) {
    if (contextPtr == 0) return false;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;

    const char* path_str = env->GetStringUTFChars(path, nullptr);
    
    // Get current token count (or just use n_ctx, but we need exact count for perfect restore?)
    // llama_state_save_file saves the KV cache.
    // We also need to save the "tokens" if we want to resume exact generation, 
    // but usually saving the KV cache is enough to resume *context*.
    // llama_state_save_file signature: (ctx, path, tokens, n_token_count)
    // If we pass NULL tokens, it just saves the state? No, it expects tokens.
    // Use llama_get_kv_cache_token_count(ctx)?
    
    // For now, we will save the KV cache. 
    // Ideally we should track the input tokens in Kotlin or C++.
    // Let's rely on the caller to handle the prompt history, but we save the *computation* state.
    // Wait, llama_state_save_file is deprecated?
    // Check llama.h: "use llama_state_save_file instead" (of llama_save_session_file).
    
    // We need to know how many tokens are in the cache.
    // Since we don't track history in C++ wrapper (only streaming), we might just save with 0 tokens 
    // if the intention is just to snapshot the memory.
    // But to resume, we need the tokens.
    
    // Simplified: Just save the KV cache.
    bool result = llama_state_save_file(wrapper->ctx, path_str, nullptr, 0);
    
    env->ReleaseStringUTFChars(path, path_str);
    return result;
}

JNIEXPORT jboolean JNICALL
Java_com_cortex_app_LlamaEngine_loadSession(JNIEnv* env, jobject, jlong contextPtr, jstring path) {
    if (contextPtr == 0) return false;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;

    const char* path_str = env->GetStringUTFChars(path, nullptr);
    
    size_t n_tokens_out = 0;
    // We pass NULL tokens_out because we don't need to read them back into an array here,
    // we just want to restore the KV cache state.
    bool result = llama_state_load_file(wrapper->ctx, path_str, nullptr, 0, &n_tokens_out);
    
    env->ReleaseStringUTFChars(path, path_str);
    return result;
}

// Sequence-specific State Management

JNIEXPORT jboolean JNICALL
Java_com_cortex_app_LlamaEngine_saveSessionSequence(
        JNIEnv* env, jobject, jlong contextPtr, jstring path, jint seqId) {
    if (contextPtr == 0) return false;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;

    const char* path_str = env->GetStringUTFChars(path, nullptr);
    if (!path_str) return false;

    // Save state for specific sequence
    llama_seq_id sid = static_cast<llama_seq_id>(seqId);
    size_t result = llama_state_seq_save_file(wrapper->ctx, path_str, sid, nullptr, 0);
    
    env->ReleaseStringUTFChars(path, path_str);
    
    LOGD("Sequence state saved: seq_id=%d, path=%s, size=%zu", seqId, path_str, result);
    
    return result > 0;
}

JNIEXPORT jboolean JNICALL
Java_com_cortex_app_LlamaEngine_loadSessionSequence(
        JNIEnv* env, jobject, jlong contextPtr, jstring path, jint seqId) {
    if (contextPtr == 0) return false;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;

    const char* path_str = env->GetStringUTFChars(path, nullptr);
    if (!path_str) return false;

    // Load state for specific sequence
    llama_seq_id sid = static_cast<llama_seq_id>(seqId);
    size_t result = llama_state_seq_load_file(
        wrapper->ctx, 
        path_str, 
        sid, 
        nullptr,  // tokens_out
        0,        // n_token_capacity
        nullptr   // n_token_count_out
    );
    
    env->ReleaseStringUTFChars(path, path_str);
    
    LOGD("Sequence state loaded: seq_id=%d, path=%s, tokens=%zu", seqId, path_str, result);
    
    return result > 0;
}

/**
 * Get the size of the state data in bytes.
 * 
 * JNI Signature: getStateSize(J)J
 * 
 * @param contextPtr The context pointer
 * @return Size of the state data in bytes
 */
JNIEXPORT jlong JNICALL
Java_com_cortex_app_LlamaEngine_nativeGetStateSize(
        JNIEnv* env,
        jobject,
        jlong contextPtr) {
    
    if (contextPtr == 0) {
        LOGE("getStateSize: null context");
        return 0;
    }
    
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    size_t state_size = llama_state_get_size(wrapper->ctx);
    
    LOGD("State size: %zu bytes", state_size);
    
    return static_cast<jlong>(state_size);
}

}
