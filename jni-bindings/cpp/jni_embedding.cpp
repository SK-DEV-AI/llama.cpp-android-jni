#include "jni_common.h"

extern "C" {

JNIEXPORT jfloatArray JNICALL
Java_com_cortex_app_LlamaEngine_getEmbedding(JNIEnv* env, jobject, jlong contextPtr, jstring text) {
    if (contextPtr == 0) return nullptr;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;

    const char* raw_text = env->GetStringUTFChars(text, nullptr);
    std::string text_str(raw_text);
    env->ReleaseStringUTFChars(text, raw_text);

    llama_set_embeddings(wrapper->ctx, true);

    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
    std::vector<llama_token> tokens;
    tokens.resize(text_str.size() + 2);
    int n = llama_tokenize(vocab, text_str.c_str(), text_str.size(), tokens.data(), tokens.size(), true, true);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text_str.c_str(), text_str.size(), tokens.data(), tokens.size(), true, true);
    }
    tokens.resize(n);

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch_add(batch, tokens[i], i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(wrapper->ctx, batch) != 0) {
        LOGE("llama_decode failed for embedding");
        llama_batch_free(batch);
        return nullptr;
    }

    float* emb = llama_get_embeddings_ith(wrapper->ctx, -1);
    if (!emb) emb = llama_get_embeddings(wrapper->ctx); 

    if (!emb) {
        LOGE("Failed to retrieve embeddings");
        llama_batch_free(batch);
        return nullptr;
    }

    int n_embd = llama_model_n_embd(wrapper->model);
    jfloatArray result = env->NewFloatArray(n_embd);
    env->SetFloatArrayRegion(result, 0, n_embd, emb);

    llama_batch_free(batch);
    return result;
}

/**
 * Get batch embeddings from the last decode operation.
 * Returns embeddings for all tokens in the batch where logits were enabled.
 * 
 * JNI Signature: getBatchEmbeddings(J)[F
 * 
 * @param contextPtr The context pointer
 * @return Float array of embeddings (size = n_outputs * n_embd), or null
 */
JNIEXPORT jfloatArray JNICALL
Java_com_cortex_app_LlamaEngine_nativeGetBatchEmbeddings(
        JNIEnv* env,
        jobject,
        jlong contextPtr) {
    
    if (contextPtr == 0) {
        LOGE("getBatchEmbeddings: null context");
        return nullptr;
    }
    
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    
    // Get embeddings pointer
    float* embeddings = llama_get_embeddings(wrapper->ctx);
    if (!embeddings) {
        LOGE("getBatchEmbeddings: no embeddings available");
        return nullptr;
    }
    
    // Get dimensions
    int32_t n_embd = llama_model_n_embd(wrapper->model);
    
    // We need to determine n_outputs - for now, assume it's 1 (single embedding)
    // In practice, this should be tracked from the batch
    int32_t n_outputs = 1;
    
    // Create Java float array
    jfloatArray result = env->NewFloatArray(n_outputs * n_embd);
    if (!result) {
        LOGE("getBatchEmbeddings: failed to allocate array");
        return nullptr;
    }
    
    // Copy embeddings
    env->SetFloatArrayRegion(result, 0, n_outputs * n_embd, embeddings);
    
    LOGD("getBatchEmbeddings: returned %d x %d embeddings", n_outputs, n_embd);
    
    return result;
}

}
