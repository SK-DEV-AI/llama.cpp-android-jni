#include "jni_common.h"
#include <cstdio>

extern "C" {

JNIEXPORT jintArray JNICALL
Java_com_cortex_app_LlamaEngine_tokenize(JNIEnv* env, jobject, jlong contextPtr, jstring text) {
    if (contextPtr == 0) return nullptr;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;

    const char* raw_text = env->GetStringUTFChars(text, nullptr);
    std::string text_str(raw_text);
    env->ReleaseStringUTFChars(text, raw_text);

    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
    std::vector<llama_token> tokens;
    tokens.resize(text_str.size() + 2);
    int n = llama_tokenize(vocab, text_str.c_str(), text_str.size(), tokens.data(), tokens.size(), true, true);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text_str.c_str(), text_str.size(), tokens.data(), tokens.size(), true, true);
    }
    tokens.resize(n);

    jintArray result = env->NewIntArray(n);
    env->SetIntArrayRegion(result, 0, n, tokens.data());
    return result;
}

JNIEXPORT jstring JNICALL
Java_com_cortex_app_LlamaEngine_detokenize(JNIEnv* env, jobject, jlong contextPtr, jintArray tokens) {
    if (contextPtr == 0) return nullptr;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;

    jsize len = env->GetArrayLength(tokens);
    jint* token_data = env->GetIntArrayElements(tokens, nullptr);
    
    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
    std::string result_str = "";
    char buf[256];

    for (int i = 0; i < len; ++i) {
        int n = llama_token_to_piece(vocab, token_data[i], buf, sizeof(buf), 0, true);
        if (n >= 0) {
            result_str += std::string(buf, n);
        }
    }

    env->ReleaseIntArrayElements(tokens, token_data, 0);
    return env->NewStringUTF(result_str.c_str());
}

// applyChatTemplate(roles[], contents[], customTemplate?) -> String
JNIEXPORT jstring JNICALL
Java_com_cortex_app_LlamaEngine_applyChatTemplate(JNIEnv* env, jobject, jlong contextPtr, jobjectArray roles, jobjectArray contents, jstring customTemplate) {
    if (contextPtr == 0) return nullptr;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;

    jsize n_msgs = env->GetArrayLength(roles);
    if (env->GetArrayLength(contents) != n_msgs) return nullptr;

    const char* tmpl = nullptr;
    if (customTemplate != nullptr) {
        tmpl = env->GetStringUTFChars(customTemplate, nullptr);
    }

    std::vector<llama_chat_message> messages(n_msgs);
    std::vector<std::string> role_strs(n_msgs);
    std::vector<std::string> content_strs(n_msgs);

    for (int i = 0; i < n_msgs; ++i) {
        jstring jRole = (jstring)env->GetObjectArrayElement(roles, i);
        jstring jContent = (jstring)env->GetObjectArrayElement(contents, i);

        const char* role_c = env->GetStringUTFChars(jRole, nullptr);
        const char* content_c = env->GetStringUTFChars(jContent, nullptr);

        role_strs[i] = std::string(role_c);
        content_strs[i] = std::string(content_c);

        messages[i].role = role_strs[i].c_str();
        messages[i].content = content_strs[i].c_str();

        env->ReleaseStringUTFChars(jRole, role_c);
        env->ReleaseStringUTFChars(jContent, content_c);
    }

    // Allocate buffer
    std::vector<char> buf(4096); 
    
    // Resolve template: Use custom if provided, otherwise fetch model's default
    const char* actual_tmpl = tmpl;
    if (actual_tmpl == nullptr) {
        actual_tmpl = llama_model_chat_template(wrapper->model, nullptr);
    }

    if (actual_tmpl == nullptr) {
        // Fallback or error if no template found
        // Some models might not have one. llama_chat_apply_template might accept NULL if it has internal fallbacks?
        // But the signature requires const char* tmpl.
        // Let's try passing nullptr if model template is missing, hoping for a fallback, or fail.
        // Actually, let's just pass actual_tmpl.
    }

    // Call llama_chat_apply_template (6 args: tmpl, chat, n_msg, add_ass, buf, len)
    int res = llama_chat_apply_template(actual_tmpl, messages.data(), messages.size(), true, buf.data(), buf.size());
    
    if (res < 0) {
        // Error
        if (tmpl) env->ReleaseStringUTFChars(customTemplate, tmpl);
        return nullptr;
    }

    if ((size_t)res > buf.size()) {
        buf.resize(res);
        res = llama_chat_apply_template(actual_tmpl, messages.data(), messages.size(), true, buf.data(), buf.size());
    }

    if (tmpl) env->ReleaseStringUTFChars(customTemplate, tmpl);

    if (res < 0) return nullptr;

    return env->NewStringUTF(std::string(buf.data(), res).c_str());
}

JNIEXPORT jstring JNICALL
Java_com_cortex_app_LlamaEngine_getMetrics(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return nullptr;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;

    llama_perf_context_data perf = llama_perf_context(wrapper->ctx); 
    
    char buf[512];
    snprintf(buf, sizeof(buf), 
        "{"
        "\"t_start\": %.2f, "
        "\"t_load\": %.2f, "
        "\"t_p_eval\": %.2f, "
        "\"t_eval\": %.2f, "
        "\"n_p_eval\": %d, "
        "\"n_eval\": %d"
        "}",
        perf.t_start_ms, perf.t_load_ms, perf.t_p_eval_ms, perf.t_eval_ms, perf.n_p_eval, perf.n_eval
    );

    return env->NewStringUTF(buf);
}

}
