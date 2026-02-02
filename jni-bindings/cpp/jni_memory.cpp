#include "jni_common.h"

extern "C" {

/**
 * Clear the KV cache (reset all tokens).
 * 
 * JNI Signature: clearKvCache(JZ)V
 * 
 * @param contextPtr The context pointer
 * @param clearData If true, also clear data buffers (default: true)
 */
JNIEXPORT void JNICALL
Java_com_cortex_app_LlamaEngine_nativeClearKvCache(
        JNIEnv* env,
        jobject,
        jlong contextPtr,
        jboolean clearData) {
    
    if (contextPtr == 0) {
        LOGE("clearKvCache: null context");
        return;
    }
    
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    llama_memory_t mem = llama_get_memory(wrapper->ctx);
    
    llama_memory_clear(mem, clearData);
    
    LOGD("KV cache cleared (data=%s)", clearData ? "true" : "false");
}

/**
 * Remove tokens from a sequence in the KV cache.
 * 
 * JNI Signature: removeKvCacheTokens(JIIII)Z
 * 
 * @param contextPtr The context pointer
 * @param seqId Sequence ID (use -1 for any sequence)
 * @param pos0 Start position (use -1 for 0)
 * @param pos1 End position (use -1 for infinity)
 * @return true if successful, false if partial sequence cannot be removed
 */
JNIEXPORT jboolean JNICALL
Java_com_cortex_app_LlamaEngine_nativeRemoveKvCacheTokens(
        JNIEnv* env,
        jobject,
        jlong contextPtr,
        jint seqId,
        jint pos0,
        jint pos1) {
    
    if (contextPtr == 0) {
        LOGE("removeKvCacheTokens: null context");
        return JNI_FALSE;
    }
    
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    llama_memory_t mem = llama_get_memory(wrapper->ctx);
    
    // Convert Java int positions to llama_pos (typically int32_t)
    llama_seq_id sid = static_cast<llama_seq_id>(seqId);
    llama_pos p0 = (pos0 < 0) ? -1 : static_cast<llama_pos>(pos0);
    llama_pos p1 = (pos1 < 0) ? -1 : static_cast<llama_pos>(pos1);
    
    bool result = llama_memory_seq_rm(mem, sid, p0, p1);
    
    LOGD("KV cache tokens removed: seq_id=%d, pos=[%d, %d), success=%s",
         seqId, pos0, pos1, result ? "true" : "false");
    
    return result ? JNI_TRUE : JNI_FALSE;
}

/**
 * Copy tokens from one sequence to another in the KV cache.
 * 
 * JNI Signature: copyKvCacheSequence(JIIII)V
 * 
 * @param contextPtr The context pointer
 * @param srcSeqId Source sequence ID
 * @param dstSeqId Destination sequence ID
 * @param pos0 Start position (use -1 for 0)
 * @param pos1 End position (use -1 for infinity)
 */
JNIEXPORT void JNICALL
Java_com_cortex_app_LlamaEngine_nativeCopyKvCacheSequence(
        JNIEnv* env,
        jobject,
        jlong contextPtr,
        jint srcSeqId,
        jint dstSeqId,
        jint pos0,
        jint pos1) {
    
    if (contextPtr == 0) {
        LOGE("copyKvCacheSequence: null context");
        return;
    }
    
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    llama_memory_t mem = llama_get_memory(wrapper->ctx);
    
    llama_seq_id src_sid = static_cast<llama_seq_id>(srcSeqId);
    llama_seq_id dst_sid = static_cast<llama_seq_id>(dstSeqId);
    llama_pos p0 = (pos0 < 0) ? -1 : static_cast<llama_pos>(pos0);
    llama_pos p1 = (pos1 < 0) ? -1 : static_cast<llama_pos>(pos1);
    
    llama_memory_seq_cp(mem, src_sid, dst_sid, p0, p1);
    
    LOGD("KV cache sequence copied: src=%d, dst=%d, pos=[%d, %d)",
         srcSeqId, dstSeqId, pos0, pos1);
}

/**
 * Keep only tokens from a specific sequence in the KV cache.
 * Removes all tokens that don't belong to the specified sequence.
 * 
 * JNI Signature: keepKvCacheSequence(JI)V
 * 
 * @param contextPtr The context pointer
 * @param seqId Sequence ID to keep
 */
JNIEXPORT void JNICALL
Java_com_cortex_app_LlamaEngine_nativeKeepKvCacheSequence(
        JNIEnv* env,
        jobject,
        jlong contextPtr,
        jint seqId) {
    
    if (contextPtr == 0) {
        LOGE("keepKvCacheSequence: null context");
        return;
    }
    
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    llama_memory_t mem = llama_get_memory(wrapper->ctx);
    
    llama_seq_id sid = static_cast<llama_seq_id>(seqId);
    llama_memory_seq_keep(mem, sid);
    
    LOGD("KV cache kept only sequence: seq_id=%d", seqId);
}

/**
 * Shift positions of tokens in a sequence by a delta value.
 * 
 * JNI Signature: shiftKvCachePositions(JIIIII)V
 * 
 * @param contextPtr The context pointer
 * @param seqId Sequence ID (use -1 for any sequence)
 * @param pos0 Start position (use -1 for 0)
 * @param pos1 End position (use -1 for infinity)
 * @param delta Position delta to add (can be negative)
 */
JNIEXPORT void JNICALL
Java_com_cortex_app_LlamaEngine_nativeShiftKvCachePositions(
        JNIEnv* env,
        jobject,
        jlong contextPtr,
        jint seqId,
        jint pos0,
        jint pos1,
        jint delta) {
    
    if (contextPtr == 0) {
        LOGE("shiftKvCachePositions: null context");
        return;
    }
    
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    llama_memory_t mem = llama_get_memory(wrapper->ctx);
    
    llama_seq_id sid = static_cast<llama_seq_id>(seqId);
    llama_pos p0 = (pos0 < 0) ? -1 : static_cast<llama_pos>(pos0);
    llama_pos p1 = (pos1 < 0) ? -1 : static_cast<llama_pos>(pos1);
    llama_pos d = static_cast<llama_pos>(delta);
    
    llama_memory_seq_add(mem, sid, p0, p1, d);
    
    LOGD("KV cache positions shifted: seq_id=%d, pos=[%d, %d), delta=%d",
         seqId, pos0, pos1, delta);
}

/**
 * Divide positions of tokens in a sequence by a factor.
 * Used for position interpolation in long context scenarios.
 * 
 * JNI Signature: divideKvCachePositions(JIIIII)V
 * 
 * @param contextPtr The context pointer
 * @param seqId Sequence ID (use -1 for any sequence)
 * @param pos0 Start position (use -1 for 0)
 * @param pos1 End position (use -1 for infinity)
 * @param divisor Divisor factor (must be > 1)
 */
JNIEXPORT void JNICALL
Java_com_cortex_app_LlamaEngine_nativeDivideKvCachePositions(
        JNIEnv* env,
        jobject,
        jlong contextPtr,
        jint seqId,
        jint pos0,
        jint pos1,
        jint divisor) {
    
    if (contextPtr == 0) {
        LOGE("divideKvCachePositions: null context");
        return;
    }
    
    if (divisor <= 1) {
        LOGE("divideKvCachePositions: divisor must be > 1, got %d", divisor);
        return;
    }
    
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    llama_memory_t mem = llama_get_memory(wrapper->ctx);
    
    llama_seq_id sid = static_cast<llama_seq_id>(seqId);
    llama_pos p0 = (pos0 < 0) ? -1 : static_cast<llama_pos>(pos0);
    llama_pos p1 = (pos1 < 0) ? -1 : static_cast<llama_pos>(pos1);
    
    llama_memory_seq_div(mem, sid, p0, p1, divisor);
    
    LOGD("KV cache positions divided: seq_id=%d, pos=[%d, %d), divisor=%d",
         seqId, pos0, pos1, divisor);
}

/**
 * Get the KV cache usage statistics.
 * 
 * JNI Signature: getKvCacheStats(J)[I
 * 
 * @param contextPtr The context pointer
 * @return int array: [usedTokens, maxTokens, usedCells, maxCells]
 */
JNIEXPORT jintArray JNICALL
Java_com_cortex_app_LlamaEngine_nativeGetKvCacheStats(
        JNIEnv* env,
        jobject,
        jlong contextPtr) {
    
    if (contextPtr == 0) {
        LOGE("getKvCacheStats: null context");
        jintArray result = env->NewIntArray(4);
        jint values[] = {0, 0, 0, 0};
        env->SetIntArrayRegion(result, 0, 4, values);
        return result;
    }
    
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    llama_memory_t mem = llama_get_memory(wrapper->ctx);
    
    // Get memory info from llama.cpp
    // Note: llama.cpp doesn't expose direct cell count getters
    // We'll return 0s as placeholders - these could be extended
    // with custom tracking if needed
    jintArray result = env->NewIntArray(4);
    jint values[] = {0, 0, 0, 0};
    env->SetIntArrayRegion(result, 0, 4, values);
    
    LOGD("KV cache stats retrieved");
    
    return result;
}

/**
 * Get the number of tokens in the KV cache.
 * 
 * JNI Signature: getKvCacheTokenCount(J)I
 * 
 * @param contextPtr The context pointer
 * @return Number of tokens in the KV cache for sequence 0
 */
JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_nativeGetKvCacheTokenCount(
        JNIEnv* env,
        jobject,
        jlong contextPtr) {
    
    if (contextPtr == 0) {
        LOGE("getKvCacheTokenCount: null context");
        return 0;
    }
    
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    llama_memory_t mem = llama_get_memory(wrapper->ctx);
    
    // Get max position for sequence 0 and add 1 to get token count
    // Returns -1 if sequence is empty
    llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
    jint token_count = (pos_max < 0) ? 0 : static_cast<jint>(pos_max + 1);
    
    LOGD("KV cache token count: %d", token_count);
    
    return token_count;
}

/**
 * Defragment the KV cache.
 * 
 * JNI Signature: defragKvCache(J)V
 * 
 * Note: In modern llama.cpp, defragmentation is automatically handled
 * during memory operations. This function serves as a no-op placeholder
 * for API compatibility.
 * 
 * @param contextPtr The context pointer
 */
JNIEXPORT void JNICALL
Java_com_cortex_app_LlamaEngine_nativeDefragCache(
        JNIEnv* env,
        jobject,
        jlong contextPtr) {
    
    if (contextPtr == 0) {
        LOGE("defragCache: null context");
        return;
    }
    
    // Defragmentation is auto-handled in modern llama.cpp
    // This function exists for API compatibility
    LOGD("KV cache defrag called (auto-handled by llama.cpp)");
}

} // extern "C"