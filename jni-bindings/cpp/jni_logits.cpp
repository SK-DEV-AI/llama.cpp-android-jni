#include "jni_common.h"
#include <vector>
#include <algorithm>

extern "C" {

/**
 * Get the raw logits for the last token in the context.
 * Returns a float array of size vocab_size containing log probabilities.
 * 
 * JNI Signature: getLogits(J)[F
 * 
 * @param contextPtr The context pointer
 * @return Float array of logits (size = vocab_size), or null on error
 */
JNIEXPORT jfloatArray JNICALL
Java_com_cortex_app_LlamaEngine_nativeGetLogits(
        JNIEnv* env,
        jobject,
        jlong contextPtr) {
    
    if (contextPtr == 0) {
        LOGE("getLogits: null context");
        return nullptr;
    }
    
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    
    // Get vocabulary size
    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    
    // Get logits pointer
    float* logits = llama_get_logits(wrapper->ctx);
    if (!logits) {
        LOGE("getLogits: failed to get logits");
        return nullptr;
    }
    
    // Create Java float array
    jfloatArray result = env->NewFloatArray(n_vocab);
    if (!result) {
        LOGE("getLogits: failed to allocate float array");
        return nullptr;
    }
    
    // Copy logits to Java array
    env->SetFloatArrayRegion(result, 0, n_vocab, logits);
    
    LOGD("getLogits: returned %d logits", n_vocab);
    
    return result;
}

/**
 * Get logits for a specific token index in the batch.
 * Negative indices access logits in reverse order (-1 = last).
 * 
 * JNI Signature: getLogitsIth(JI)[F
 * 
 * @param contextPtr The context pointer
 * @param index Token index (0 = first, -1 = last)
 * @return Float array of logits, or null on error
 */
JNIEXPORT jfloatArray JNICALL
Java_com_cortex_app_LlamaEngine_nativeGetLogitsIth(
        JNIEnv* env,
        jobject,
        jlong contextPtr,
        jint index) {
    
    if (contextPtr == 0) {
        LOGE("getLogitsIth: null context");
        return nullptr;
    }
    
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    
    // Get vocabulary size
    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    
    // Get logits for ith token
    float* logits = llama_get_logits_ith(wrapper->ctx, index);
    if (!logits) {
        LOGE("getLogitsIth: failed to get logits for index %d", index);
        return nullptr;
    }
    
    // Create Java float array
    jfloatArray result = env->NewFloatArray(n_vocab);
    if (!result) {
        LOGE("getLogitsIth: failed to allocate float array");
        return nullptr;
    }
    
    // Copy logits to Java array
    env->SetFloatArrayRegion(result, 0, n_vocab, logits);
    
    LOGD("getLogitsIth: returned logits for index %d", index);
    
    return result;
}

/**
 * Get the vocabulary size (needed to interpret logits array).
 * 
 * JNI Signature: getVocabSizeFromContext(J)I
 * 
 * @param contextPtr The context pointer
 * @return Vocabulary size
 */
JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_nativeGetVocabSizeFromContext(
        JNIEnv* env,
        jobject,
        jlong contextPtr) {
    
    if (contextPtr == 0) {
        LOGE("getVocabSizeFromContext: null context");
        return 0;
    }
    
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
    
    return static_cast<jint>(llama_vocab_n_tokens(vocab));
}

/**
 * Decode a batch of tokens without sampling.
 * This is useful when you want to manually handle logits and sampling.
 * 
 * JNI Signature: decodeTokens(J[I)Z
 * 
 * @param contextPtr The context pointer
 * @param tokens Array of token IDs to decode
 * @return true if successful
 */
JNIEXPORT jboolean JNICALL
Java_com_cortex_app_LlamaEngine_nativeDecodeTokens(
        JNIEnv* env,
        jobject,
        jlong contextPtr,
        jintArray tokens) {
    
    if (contextPtr == 0 || tokens == nullptr) {
        LOGE("decodeTokens: null context or tokens");
        return JNI_FALSE;
    }
    
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    
    // Get tokens from Java array
    jsize n_tokens = env->GetArrayLength(tokens);
    jint* token_array = env->GetIntArrayElements(tokens, nullptr);
    if (!token_array) {
        LOGE("decodeTokens: failed to get token array");
        return JNI_FALSE;
    }
    
    // Create batch
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    
    // Add tokens to batch
    for (int i = 0; i < n_tokens; i++) {
        batch_add(batch, static_cast<llama_token>(token_array[i]), i, {0}, false);
    }
    
    // Enable logits for last token
    if (n_tokens > 0) {
        batch.logits[n_tokens - 1] = true;
    }
    
    // Decode
    int ret = llama_decode(wrapper->ctx, batch);
    
    // Cleanup
    llama_batch_free(batch);
    env->ReleaseIntArrayElements(tokens, token_array, JNI_ABORT);
    
    if (ret != 0) {
        LOGE("decodeTokens: llama_decode failed with code %d", ret);
        return JNI_FALSE;
    }
    
    LOGD("decodeTokens: decoded %d tokens", n_tokens);
    
    return JNI_TRUE;
}

/**
 * Sample a single token from logits using provided temperature.
 * This is a convenience function for custom sampling.
 * 
 * JNI Signature: sampleTokenFromLogits(J[FF)I
 * 
 * @param contextPtr The context pointer
 * @param logits Float array of logits
 * @param temperature Sampling temperature (1.0 = default)
 * @return Sampled token ID, or -1 on error
 */
JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_nativeSampleTokenFromLogits(
        JNIEnv* env,
        jobject,
        jlong contextPtr,
        jfloatArray logits,
        jfloat temperature) {
    
    if (contextPtr == 0 || logits == nullptr) {
        LOGE("sampleTokenFromLogits: null context or logits");
        return -1;
    }
    
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    
    // Get vocabulary size
    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    
    // Get logits from Java
    jsize n_logits = env->GetArrayLength(logits);
    if (n_logits != n_vocab) {
        LOGE("sampleTokenFromLogits: logits size mismatch (%d vs %d)", n_logits, n_vocab);
        return -1;
    }
    
    // Copy logits to native array
    std::vector<float> logits_copy(n_vocab);
    env->GetFloatArrayRegion(logits, 0, n_vocab, logits_copy.data());
    
    // Create token data array for sampling
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    
    for (int32_t i = 0; i < n_vocab; i++) {
        candidates.push_back({i, logits_copy[i], 0.0f});
    }
    
    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
    
    // Apply temperature by scaling logits
    if (temperature != 1.0f && temperature > 0.0f) {
        for (size_t i = 0; i < candidates_p.size; i++) {
            candidates[i].logit /= temperature;
        }
    }
    
    // Sort candidates by logit (descending)
    // Find the token with highest logit as a simple argmax
    llama_token token = 0;
    float max_logit = candidates[0].logit;
    for (size_t i = 1; i < candidates_p.size; i++) {
        if (candidates[i].logit > max_logit) {
            max_logit = candidates[i].logit;
            token = candidates[i].id;
        }
    }
    
    LOGD("sampleTokenFromLogits: sampled token %d", token);
    
    return static_cast<jint>(token);
}

} // extern "C"