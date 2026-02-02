#include "jni_common.h"

extern "C" {

JNIEXPORT jint JNICALL
Java_com_cortex_app_BackendInfo_nativeGetMaxDevices(JNIEnv* env, jclass) {
    return (jint)llama_max_devices();
}

JNIEXPORT jboolean JNICALL
Java_com_cortex_app_BackendInfo_nativeSupportsMmap(JNIEnv* env, jclass) {
    return llama_supports_mmap() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_cortex_app_BackendInfo_nativeSupportsMlock(JNIEnv* env, jclass) {
    return llama_supports_mlock() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_cortex_app_BackendInfo_nativeSupportsGpuOffload(JNIEnv* env, jclass) {
    return llama_supports_gpu_offload() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_cortex_app_BackendInfo_nativeSupportsRpc(JNIEnv* env, jclass) {
    return llama_supports_rpc() ? JNI_TRUE : JNI_FALSE;
}

}
