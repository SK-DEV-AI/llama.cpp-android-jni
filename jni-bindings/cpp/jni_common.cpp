#include "jni_common.h"

void llama_log_callback(ggml_log_level level, const char * text, void * user_data) {
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: __android_log_print(ANDROID_LOG_ERROR, "CortexInternal", "%s", text); break;
        case GGML_LOG_LEVEL_INFO:  __android_log_print(ANDROID_LOG_INFO,  "CortexInternal", "%s", text); break;
        case GGML_LOG_LEVEL_WARN:  __android_log_print(ANDROID_LOG_WARN,  "CortexInternal", "%s", text); break;
        default:                   __android_log_print(ANDROID_LOG_DEBUG, "CortexInternal", "%s", text); break;
    }
}

void batch_add(struct llama_batch & batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;

    batch.n_tokens++;
}

void batch_clear(struct llama_batch & batch) {
    batch.n_tokens = 0;
}

// Timing utility
extern "C" JNIEXPORT jlong JNICALL
Java_com_cortex_app_LlamaEngine_nativeGetTimeMicros(JNIEnv*, jobject) {
    return static_cast<jlong>(llama_time_us());
}
