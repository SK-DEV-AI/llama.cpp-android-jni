#include "jni_common.h"

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_cortex_app_LlamaEngine_loadModel(JNIEnv* env, jobject, jstring modelPath) {
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    
    static bool is_initialized = false;
    if (!is_initialized) {
        llama_log_set(llama_log_callback, nullptr);
        llama_backend_init();
        is_initialized = true;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    
    llama_model* model = llama_model_load_from_file(path, model_params);
    env->ReleaseStringUTFChars(modelPath, path);
    
    if (!model) {
        LOGE("Failed to load model from %s", path);
        return 0;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 4096;
    ctx_params.n_batch = 2048; 
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.no_perf = false; // Enable metrics

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOGE("Failed to create context");
        llama_model_free(model);
        return 0;
    }

    auto* wrapper = new LlamaContextWrapper();
    wrapper->model = model;
    wrapper->ctx = ctx;

    LOGD("Model loaded successfully. Ptr: %ld", (long)wrapper);
    return (jlong)wrapper;
}

JNIEXPORT jlong JNICALL
Java_com_cortex_app_LlamaEngine_loadReranker(JNIEnv* env, jobject, jstring modelPath) {
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    
    static bool is_initialized = false;
    if (!is_initialized) {
        llama_log_set(llama_log_callback, nullptr);
        llama_backend_init();
        is_initialized = true;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    
    llama_model* model = llama_model_load_from_file(path, model_params);
    env->ReleaseStringUTFChars(modelPath, path);
    
    if (!model) {
        LOGE("Failed to load reranker from %s", path);
        return 0;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 4096; // Rerankers usually need less context, but safe default
    ctx_params.n_batch = 2048; 
    ctx_params.n_ubatch = 2048; // Non-causal models require ubatch == batch
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_RANK; // FORCE RANK POOLING
    ctx_params.embeddings = true; // Essential for BERT/Rerankers
    ctx_params.no_perf = false;

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOGE("Failed to create reranker context");
        llama_model_free(model);
        return 0;
    }

    auto* wrapper = new LlamaContextWrapper();
    wrapper->model = model;
    wrapper->ctx = ctx;

    LOGD("Reranker loaded successfully. Ptr: %ld", (long)wrapper);
    return (jlong)wrapper;
}

JNIEXPORT void JNICALL
Java_com_cortex_app_LlamaEngine_freeModel(JNIEnv*, jobject, jlong contextPtr) {
    if (contextPtr == 0) return;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    
    if (wrapper->ctx) llama_free(wrapper->ctx);
    if (wrapper->model) llama_model_free(wrapper->model);
    
    delete wrapper;
    LOGD("Model memory freed.");
}

}