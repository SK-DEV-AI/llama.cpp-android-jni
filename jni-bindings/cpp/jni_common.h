#ifndef CORTEX_JNI_COMMON_H
#define CORTEX_JNI_COMMON_H

#include <jni.h>
#include <string>
#include <vector>
#include <unistd.h>
#include <android/log.h>
#include "llama.h" 

#define TAG "CortexEngine"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// LoRA Adapter Info
struct LoraAdapterInfo {
    std::string path;
    llama_adapter_lora* adapter;
    float scale;
};

// Shared Wrapper
struct LlamaContextWrapper {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    std::vector<LoraAdapterInfo> loaded_loras;
};

// Logging Callback
void llama_log_callback(ggml_log_level level, const char * text, void * user_data);

// Batch Helpers
void batch_add(struct llama_batch & batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool logits);
void batch_clear(struct llama_batch & batch);

#endif // CORTEX_JNI_COMMON_H
