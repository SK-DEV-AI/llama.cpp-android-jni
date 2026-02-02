#include "jni_common.h"

extern "C" {

/**
 * Load a model from multiple split files.
 * Useful for loading very large models that are split into multiple GGUF files.
 * 
 * JNI Signature: loadModelFromSplits([Ljava/lang/String;)J
 * 
 * @param env JNIEnv pointer
 * @param paths Array of file paths to the split model files
 * @return Context pointer or 0 on error
 */
JNIEXPORT jlong JNICALL
Java_com_cortex_app_LlamaEngine_loadModelFromSplits(
        JNIEnv* env,
        jobject,
        jobjectArray paths) {
    
    if (paths == nullptr) {
        LOGE("loadModelFromSplits: null paths array");
        return 0;
    }
    
    jsize n_paths = env->GetArrayLength(paths);
    if (n_paths == 0) {
        LOGE("loadModelFromSplits: empty paths array");
        return 0;
    }
    
    // Allocate array of C strings
    const char** path_array = new const char*[n_paths];
    jstring* jstrings = new jstring[n_paths];
    
    // Convert Java strings to C strings
    for (jsize i = 0; i < n_paths; i++) {
        jstrings[i] = (jstring)env->GetObjectArrayElement(paths, i);
        if (jstrings[i] == nullptr) {
            LOGE("loadModelFromSplits: null path at index %d", i);
            // Cleanup already converted strings
            for (jsize j = 0; j < i; j++) {
                env->ReleaseStringUTFChars(jstrings[j], path_array[j]);
                env->DeleteLocalRef(jstrings[j]);
            }
            delete[] path_array;
            delete[] jstrings;
            return 0;
        }
        path_array[i] = env->GetStringUTFChars(jstrings[i], nullptr);
    }
    
    // Set up model parameters
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // CPU only for now
    
    // Load model from splits
    llama_model* model = llama_model_load_from_splits(path_array, n_paths, mparams);
    
    // Cleanup strings
    for (jsize i = 0; i < n_paths; i++) {
        env->ReleaseStringUTFChars(jstrings[i], path_array[i]);
        env->DeleteLocalRef(jstrings[i]);
    }
    delete[] path_array;
    delete[] jstrings;
    
    if (!model) {
        LOGE("loadModelFromSplits: failed to load model");
        return 0;
    }
    
    // Create context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 2048;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;
    
    llama_context* ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        LOGE("loadModelFromSplits: failed to create context");
        llama_model_free(model);
        return 0;
    }
    
    // Create wrapper
    auto* wrapper = new LlamaContextWrapper();
    wrapper->model = model;
    wrapper->ctx = ctx;
    
    LOGD("Model loaded from %d split files", n_paths);
    
    return reinterpret_cast<jlong>(wrapper);
}

/**
 * Save a model to a file.
 * Exports the current model to a GGUF file.
 * 
 * JNI Signature: saveModelToFile(JLjava/lang/String;)Z
 * 
 * @param contextPtr The context pointer
 * @param path The destination file path
 * @return true if successful
 */
JNIEXPORT jboolean JNICALL
Java_com_cortex_app_LlamaEngine_saveModelToFile(
        JNIEnv* env,
        jobject,
        jlong contextPtr,
        jstring path) {
    
    if (contextPtr == 0 || path == nullptr) {
        LOGE("saveModelToFile: null context or path");
        return JNI_FALSE;
    }
    
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    
    const char* path_str = env->GetStringUTFChars(path, nullptr);
    if (!path_str) return JNI_FALSE;
    
    llama_model_save_to_file(wrapper->model, path_str);
    
    env->ReleaseStringUTFChars(path, path_str);
    
    LOGD("Model saved to: %s", path_str);
    
    return JNI_TRUE;
}

/**
 * Get default quantization parameters.
 * Helper function for quantization.
 */
JNIEXPORT jobject JNICALL
Java_com_cortex_app_LlamaEngine_getQuantizeParams(
        JNIEnv* env,
        jobject) {
    
    llama_model_quantize_params qparams = llama_model_quantize_default_params();
    
    // Create a simple data class to return parameters
    jclass paramsClass = env->FindClass("com/cortex/app/QuantizeParams");
    if (!paramsClass) {
        LOGE("QuantizeParams class not found");
        return nullptr;
    }
    
    jmethodID constructor = env->GetMethodID(paramsClass, "<init>", "(II)V");
    if (!constructor) {
        LOGE("QuantizeParams constructor not found");
        return nullptr;
    }
    
    // Return default params object
    jobject params = env->NewObject(paramsClass, constructor, 
        qparams.nthread, 
        static_cast<jint>(qparams.ftype));
    
    return params;
}

/**
 * Quantize a model to a smaller precision.
 * Converts the model to a quantized format for reduced size and faster inference.
 * 
 * JNI Signature: quantizeModel(JLjava/lang/String;II)Z
 * 
 * @param contextPtr The context pointer
 * @param outputPath Destination path for quantized model
 * @param ftype Target file type (e.g., LLAMA_FTYPE_MOSTLY_Q4_K_M = 15)
 * @param nThreads Number of threads to use (0 = auto)
 * @return true if successful
 */
JNIEXPORT jboolean JNICALL
Java_com_cortex_app_LlamaEngine_quantizeModel(
        JNIEnv* env,
        jobject,
        jlong contextPtr,
        jstring outputPath,
        jint ftype,
        jint nThreads) {
    
    if (contextPtr == 0 || outputPath == nullptr) {
        LOGE("quantizeModel: null context or output path");
        return JNI_FALSE;
    }
    
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    
    const char* output_path = env->GetStringUTFChars(outputPath, nullptr);
    if (!output_path) return JNI_FALSE;
    
    // Set up quantization parameters
    llama_model_quantize_params qparams = llama_model_quantize_default_params();
    qparams.nthread = nThreads;
    qparams.ftype = static_cast<llama_ftype>(ftype);
    
    // Note: llama_model_quantize takes (source_path, output_path, params)
    // We need to track the source path, but for now use model pointer approach
    // Actually, llama_model_quantize works with file paths, not loaded models
    // This is a limitation - we can only quantize from files, not loaded models
    
    LOGW("quantizeModel: Model quantization requires file paths, not loaded models");
    LOGW("quantizeModel: Please use llama.cpp CLI tools for quantization");
    
    env->ReleaseStringUTFChars(outputPath, output_path);
    
    // Return false since we can't quantize a loaded model in memory
    return JNI_FALSE;
}

/**
 * Auto-fit model and context parameters to available memory.
 */
JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_fitModelParams(
        JNIEnv* env,
        jobject,
        jstring modelPath,
        jint targetContextSize,
        jint maxGpuLayers) {

    if (modelPath == nullptr) return 2;

    const char* path_str = env->GetStringUTFChars(modelPath, nullptr);
    if (!path_str) return 2;

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = maxGpuLayers;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = targetContextSize;

    // Allocate buffers required by llama_params_fit
    size_t max_devices = llama_max_devices();
    size_t max_overrides = llama_max_tensor_buft_overrides();
    
    float* tensor_split = new float[max_devices];
    struct llama_model_tensor_buft_override* tensor_buft_overrides = 
        new llama_model_tensor_buft_override[max_overrides];
    size_t* margins = new size_t[max_devices];
    
    // Initialize with zeros
    for (size_t i = 0; i < max_devices; i++) {
        tensor_split[i] = 0.0f;
        margins[i] = 0;
    }

    llama_params_fit_status status = llama_params_fit(
        path_str, 
        &mparams, 
        &cparams,
        tensor_split,
        tensor_buft_overrides,
        margins,
        2048,  // n_ctx_min - minimum context size when reducing memory
        GGML_LOG_LEVEL_INFO  // log_level
    );
    
    // Clean up allocated buffers
    delete[] tensor_split;
    delete[] tensor_buft_overrides;
    delete[] margins;

    env->ReleaseStringUTFChars(modelPath, path_str);

    return (status == LLAMA_PARAMS_FIT_STATUS_SUCCESS) ? 0 :
           (status == LLAMA_PARAMS_FIT_STATUS_FAILURE) ? 1 : 2;
}

} // extern "C"