#include "jni_common.h"

extern "C" {

/**
 * Set the number of threads for generation and batch processing.
 * 
 * JNI Signature: setThreadConfig(JIJII)V
 * 
 * @param nThreads Number of threads for generation (0 = auto)
 * @param nThreadsBatch Number of threads for batch processing (0 = auto)
 * @param cpuAffinity Whether to use CPU affinity (pin threads to cores)
 */
JNIEXPORT void JNICALL
Java_com_cortex_app_LlamaEngine_nativeSetThreadConfig(
        JNIEnv* env,
        jobject,
        jlong contextPtr,
        jint nThreads,
        jint nThreadsBatch,
        jboolean cpuAffinity) {
    
    if (contextPtr == 0) {
        LOGE("setThreadConfig: null context");
        return;
    }
    
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    
    // llama.cpp uses 0 for auto-detect
    int32_t threads = nThreads <= 0 ? 0 : static_cast<int32_t>(nThreads);
    int32_t threadsBatch = nThreadsBatch <= 0 ? 0 : static_cast<int32_t>(nThreadsBatch);
    
    llama_set_n_threads(wrapper->ctx, threads, threadsBatch);
    
    LOGD("Thread config updated: n_threads=%d, n_threads_batch=%d, affinity=%s",
         threads, threadsBatch, cpuAffinity ? "true" : "false");
    
    // Note: CPU affinity is not directly supported by llama.cpp public API
    // It would require GGML-level changes or platform-specific code
    if (cpuAffinity) {
        LOGW("CPU affinity requested but not yet implemented");
    }
}

/**
 * Get current thread configuration.
 * 
 * JNI Signature: getThreadConfig(J)[I
 * 
 * @return int array: [nThreads, nThreadsBatch, cpuAffinity (0/1)]
 */
JNIEXPORT jintArray JNICALL
Java_com_cortex_app_LlamaEngine_nativeGetThreadConfig(
        JNIEnv* env,
        jobject,
        jlong contextPtr) {
    
    if (contextPtr == 0) {
        LOGE("getThreadConfig: null context");
        jintArray result = env->NewIntArray(3);
        jint values[] = {0, 0, 0};
        env->SetIntArrayRegion(result, 0, 3, values);
        return result;
    }
    
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    
    int32_t nThreads = llama_n_threads(wrapper->ctx);
    int32_t nThreadsBatch = llama_n_threads_batch(wrapper->ctx);
    
    jintArray result = env->NewIntArray(3);
    jint values[] = {
        static_cast<jint>(nThreads),
        static_cast<jint>(nThreadsBatch),
        0  // cpuAffinity - not currently tracked
    };
    env->SetIntArrayRegion(result, 0, 3, values);
    
    LOGD("Thread config read: n_threads=%d, n_threads_batch=%d", nThreads, nThreadsBatch);
    
    return result;
}

/**
 * Get the number of available CPU cores on the device.
 * This is useful for determining optimal thread counts.
 * 
 * JNI Signature: getCpuCoreCount()I
 */
JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_nativeGetCpuCoreCount(JNIEnv*, jobject) {
    // Use sysconf to get the number of processors
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (cores <= 0) {
        cores = 4; // Fallback to 4 if we can't determine
    }
    
    LOGD("CPU core count: %ld", cores);
    return static_cast<jint>(cores);
}

/**
 * Apply a thread configuration preset.
 * 
 * JNI Signature: applyThreadPreset(JI)V
 * 
 * Preset IDs:
 *   0 = AUTO (use all cores)
 *   1 = SINGLE (1 thread)
 *   2 = PERFORMANCE (all cores)
 *   3 = BALANCED (half cores for gen, all for batch)
 *   4 = BATTERY (2 threads)
 */
JNIEXPORT void JNICALL
Java_com_cortex_app_LlamaEngine_nativeApplyThreadPreset(
        JNIEnv* env,
        jobject,
        jlong contextPtr,
        jint preset) {
    
    if (contextPtr == 0) {
        LOGE("applyThreadPreset: null context");
        return;
    }
    
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    
    long totalCores = sysconf(_SC_NPROCESSORS_ONLN);
    if (totalCores <= 0) totalCores = 4;
    
    int32_t nThreads = 0;
    int32_t nThreadsBatch = 0;
    const char* presetName = "unknown";
    
    switch (preset) {
        case 0: // AUTO
            nThreads = 0;
            nThreadsBatch = 0;
            presetName = "AUTO";
            break;
            
        case 1: // SINGLE
            nThreads = 1;
            nThreadsBatch = 1;
            presetName = "SINGLE";
            break;
            
        case 2: // PERFORMANCE
            nThreads = static_cast<int32_t>(totalCores);
            nThreadsBatch = static_cast<int32_t>(totalCores);
            presetName = "PERFORMANCE";
            break;
            
        case 3: // BALANCED
            nThreads = static_cast<int32_t>(totalCores / 2);
            if (nThreads < 1) nThreads = 1;
            nThreadsBatch = static_cast<int32_t>(totalCores);
            presetName = "BALANCED";
            break;
            
        case 4: // BATTERY
            nThreads = 2;
            nThreadsBatch = 2;
            presetName = "BATTERY";
            break;
            
        default:
            LOGE("Unknown thread preset: %d", preset);
            return;
    }
    
    llama_set_n_threads(wrapper->ctx, nThreads, nThreadsBatch);
    
    LOGD("Applied thread preset '%s': n_threads=%d, n_threads_batch=%d",
         presetName, nThreads, nThreadsBatch);
}

} // extern "C"