#include "jni_common.h"

extern "C" {

JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_getVocabSize(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return 0;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
    return (jint)llama_vocab_n_tokens(vocab);
}

JNIEXPORT jstring JNICALL
Java_com_cortex_app_LlamaEngine_getTokenText(JNIEnv* env, jobject, jlong contextPtr, jint token) {
    if (contextPtr == 0) return nullptr;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
    
    char buf[256];
    int len = llama_token_to_piece(vocab, (llama_token)token, buf, sizeof(buf), 0, false);
    if (len < 0) len = 0;
    
    return env->NewStringUTF(buf);
}

JNIEXPORT jboolean JNICALL
Java_com_cortex_app_LlamaEngine_isEogToken(JNIEnv* env, jobject, jlong contextPtr, jint token) {
    if (contextPtr == 0) return JNI_FALSE;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
    return llama_vocab_is_eog(vocab, (llama_token)token) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_cortex_app_LlamaEngine_isControlToken(JNIEnv* env, jobject, jlong contextPtr, jint token) {
    if (contextPtr == 0) return JNI_FALSE;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
    return llama_vocab_is_control(vocab, (llama_token)token) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_getBosToken(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return -1;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
    return (jint)llama_vocab_bos(vocab);
}

JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_getEosToken(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return -1;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
    return (jint)llama_vocab_eos(vocab);
}

JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_getEotToken(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return -1;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
    return (jint)llama_vocab_eot(vocab);
}

JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_getSepToken(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return -1;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
    return (jint)llama_vocab_sep(vocab);
}

JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_getNlToken(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return -1;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
    return (jint)llama_vocab_nl(vocab);
}

JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_getPadToken(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return -1;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
    return (jint)llama_vocab_pad(vocab);
}

JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_getFimPreToken(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return -1;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
    return (jint)llama_vocab_fim_pre(vocab);
}

JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_getFimSufToken(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return -1;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
    return (jint)llama_vocab_fim_suf(vocab);
}

JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_getFimMidToken(JNIEnv* env, jobject, jlong contextPtr) {
    if (contextPtr == 0) return -1;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
    return (jint)llama_vocab_fim_mid(vocab);
}

}
