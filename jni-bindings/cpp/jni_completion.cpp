#include "jni_common.h"
#include <thread>
#include <pthread.h>
#include <vector>

// Struct to pass data to the worker thread
struct GenerateParams {
    LlamaContextWrapper* wrapper;
    std::string prompt;
    std::vector<llama_token> prompt_tokens; // Optional: Pre-tokenized prompt
    std::string grammar;
    JavaVM* jvm;
    jobject callback;
    jmethodID onTokenMethod;
    // Sampling parameters
    float temperature;
    int topK;
    float topP;
    float minP;
    float repeatPenalty;
    int repeatLastN;
    float frequencyPenalty;
    float presencePenalty;
    int seed;
    int maxTokens;
    std::vector<std::string> stopSequences;
    // Advanced samplers
    float typicalP;
    float xtcProbability;
    float xtcThreshold;
    int mirostatMode;
    float mirostatTau;
    float mirostatEta;
    float dryMultiplier;
    float dryBase;
    int dryAllowedLength;
};

// Forward declaration for sampler chain builder from jni_sampling.cpp
llama_sampler* build_sampler_chain(const GenerateParams* params, const llama_vocab* vocab);

// The worker function that runs on the big-stack thread
void generate_worker(GenerateParams* params) {
    JNIEnv* env = nullptr;
    // Attach thread to JVM to call callback
    if (params->jvm->AttachCurrentThread(&env, nullptr) != JNI_OK) {
        LOGE("Failed to attach thread to JVM");
        return;
    }

    const llama_vocab* vocab = llama_model_get_vocab(params->wrapper->model);
    
    // Clear KV cache
    llama_memory_t mem = llama_get_memory(params->wrapper->ctx);
    llama_memory_clear(mem, true);

    // Build sampler chain using the helper function
    llama_sampler* smpl = build_sampler_chain(params, vocab);

    std::vector<llama_token> tokens_list;
    
    if (!params->prompt_tokens.empty()) {
        // Use pre-constructed tokens (e.g. for FIM)
        tokens_list = params->prompt_tokens;
    } else {
        // Tokenize text prompt
        tokens_list.resize(params->prompt.size() + 2);
        int n_tokens = llama_tokenize(vocab, params->prompt.c_str(), params->prompt.size(), tokens_list.data(), tokens_list.size(), true, true);
        if (n_tokens < 0) {
            tokens_list.resize(-n_tokens);
            n_tokens = llama_tokenize(vocab, params->prompt.c_str(), params->prompt.size(), tokens_list.data(), tokens_list.size(), true, true);
        }
        tokens_list.resize(n_tokens);
    }

    llama_batch batch = llama_batch_init(4096, 0, 1);

    for (size_t i = 0; i < tokens_list.size(); i++) {
        batch_add(batch, tokens_list[i], i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    LOGD("Decoding prompt with %d tokens", batch.n_tokens);

    int ret = llama_decode(params->wrapper->ctx, batch);
    if (ret != 0) {
        LOGE("llama_decode failed during prompt processing. Return code: %d", ret);
        llama_batch_free(batch);
        llama_sampler_free(smpl);
        params->jvm->DetachCurrentThread();
        return;
    }

    int n_cur = batch.n_tokens;
    int n_decode = 0;
    std::string accumulatedOutput;

    while (n_decode < params->maxTokens) {
        try {
            // Sample next token
            // The grammar sampler in the chain already filters logits during sampling
            // so we get a token that satisfies grammar constraints
            llama_token new_token_id = llama_sampler_sample(smpl, params->wrapper->ctx, -1);
            
            // DEBUG: Log the sampled token
            char token_buf[256];
            int token_len = llama_token_to_piece(vocab, new_token_id, token_buf, sizeof(token_buf), 0, true);
            if (token_len > 0) {
                std::string token_str(token_buf, token_len);
                LOGD("[DEBUG] Sampled token %d: '%s' (len=%d)", new_token_id, token_str.c_str(), token_len);
            } else {
                LOGD("[DEBUG] Sampled token %d: <special>", new_token_id);
            }
            
            // NOTE: We intentionally do NOT call llama_sampler_accept here.
            // The grammar sampler in the chain already enforced constraints during
            // llama_sampler_sample by filtering the logits. Calling accept would
            // re-validate and potentially throw on multi-character tokens.
            // This matches how llama-server handles grammar constraints.

            if (llama_vocab_is_eog(vocab, new_token_id)) {
                LOGD("EOS reached.");
                break;
            }

            char buf[256];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n >= 0) {
                // Append token to accumulated output for stop sequence checking
                accumulatedOutput.append(buf, n);
                
                // Check if any stop sequence is present at the end of accumulated output
                bool stopTriggered = false;
                for (const auto& stopSeq : params->stopSequences) {
                    if (accumulatedOutput.size() >= stopSeq.size()) {
                        if (accumulatedOutput.compare(accumulatedOutput.size() - stopSeq.size(), stopSeq.size(), stopSeq) == 0) {
                            LOGD("Stop sequence matched: '%s'", stopSeq.c_str());
                            stopTriggered = true;
                            break;
                        }
                    }
                }
                
                if (stopTriggered) {
                    break;
                }
                
                // Pass raw bytes to Java to avoid JNI UTF-8 validation crash
                jbyteArray jBytes = env->NewByteArray(n);
                env->SetByteArrayRegion(jBytes, 0, n, (jbyte*)buf);
                env->CallVoidMethod(params->callback, params->onTokenMethod, jBytes);
                env->DeleteLocalRef(jBytes);
            }

            batch_clear(batch);
            batch_add(batch, new_token_id, n_cur, { 0 }, true);

            n_cur++;
            n_decode++;

            ret = llama_decode(params->wrapper->ctx, batch);
            if (ret != 0) {
                LOGE("llama_decode failed during generation. Return code: %d", ret);
                break;
            }
        } catch (const std::exception& e) {
            LOGE("[DEBUG] EXCEPTION CAUGHT: %s", e.what());
            LOGE("[DEBUG] Exception type: %s", typeid(e).name());
            LOGE("[DEBUG] Token decode count: %d", n_decode);
            break;
        }
    }

    llama_batch_free(batch);
    llama_sampler_free(smpl);
    
    params->jvm->DetachCurrentThread();
}

extern "C" {

JNIEXPORT void JNICALL
Java_com_cortex_app_LlamaEngine_generateCompletion(
        JNIEnv* env,
        jobject,
        jlong contextPtr,
        jstring prompt,
        jstring grammarJson,
        jfloat temperature,
        jint topK,
        jfloat topP,
        jfloat minP,
        jfloat repeatPenalty,
        jint repeatLastN,
        jfloat frequencyPenalty,
        jfloat presencePenalty,
        jint seed,
        jint maxTokens,
        jobjectArray stopSequences,
        jfloat typicalP,
        jfloat xtcProbability,
        jfloat xtcThreshold,
        jint mirostatMode,
        jfloat mirostatTau,
        jfloat mirostatEta,
        jfloat dryMultiplier,
        jfloat dryBase,
        jint dryAllowedLength,
        jobject callback) {
            
    if (contextPtr == 0) return;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    
    const char* raw_prompt = env->GetStringUTFChars(prompt, nullptr);
    std::string prompt_str(raw_prompt);
    env->ReleaseStringUTFChars(prompt, raw_prompt);

    const char* raw_grammar = grammarJson ? env->GetStringUTFChars(grammarJson, nullptr) : nullptr;
    std::string grammar_str = raw_grammar ? raw_grammar : "";
    if (grammarJson) env->ReleaseStringUTFChars(grammarJson, raw_grammar);

    // Convert stop sequences jobjectArray to vector<string>
    std::vector<std::string> stop_sequences;
    if (stopSequences != nullptr) {
        jsize stopCount = env->GetArrayLength(stopSequences);
        for (jsize i = 0; i < stopCount; i++) {
            jstring stopStr = (jstring)env->GetObjectArrayElement(stopSequences, i);
            if (stopStr != nullptr) {
                const char* raw_stop = env->GetStringUTFChars(stopStr, nullptr);
                if (raw_stop != nullptr) {
                    stop_sequences.emplace_back(raw_stop);
                    env->ReleaseStringUTFChars(stopStr, raw_stop);
                }
                env->DeleteLocalRef(stopStr);
            }
        }
    }

    GenerateParams params;
    params.wrapper = wrapper;
    params.prompt = prompt_str;
    params.grammar = grammar_str;
    params.temperature = temperature;
    params.topK = topK;
    params.topP = topP;
    params.minP = minP;
    params.repeatPenalty = repeatPenalty;
    params.repeatLastN = repeatLastN;
    params.frequencyPenalty = frequencyPenalty;
    params.presencePenalty = presencePenalty;
    params.seed = seed;
    params.maxTokens = maxTokens;
    params.stopSequences = stop_sequences;
    params.typicalP = typicalP;
    params.xtcProbability = xtcProbability;
    params.xtcThreshold = xtcThreshold;
    params.mirostatMode = mirostatMode;
    params.mirostatTau = mirostatTau;
    params.mirostatEta = mirostatEta;
    params.dryMultiplier = dryMultiplier;
    params.dryBase = dryBase;
    params.dryAllowedLength = dryAllowedLength;
    env->GetJavaVM(&params.jvm);
    params.callback = env->NewGlobalRef(callback);
    jclass callbackClass = env->GetObjectClass(callback);
    params.onTokenMethod = env->GetMethodID(callbackClass, "onToken", "([B)V");

    pthread_t thread;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    // Huge stack to handle std::regex deep recursion in grammar parsing
    size_t stack_size = 32 * 1024 * 1024; // 32MB
    pthread_attr_setstacksize(&attr, stack_size);

    int rc = pthread_create(&thread, &attr, [](void* arg) -> void* {
        GenerateParams* p = (GenerateParams*)arg;
        generate_worker(p);
        return nullptr;
    }, &params);

    if (rc != 0) {
        LOGE("Failed to create worker thread");
    } else {
        pthread_join(thread, nullptr);
    }

    pthread_attr_destroy(&attr);
    env->DeleteGlobalRef(params.callback);
}

JNIEXPORT void JNICALL
Java_com_cortex_app_LlamaEngine_infill(
        JNIEnv* env,
        jobject,
        jlong contextPtr,
        jstring prefix,
        jstring suffix,
        jobject callback) {

    if (contextPtr == 0) return;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);

    // Get FIM tokens
    llama_token fim_pre = llama_vocab_fim_pre(vocab);
    llama_token fim_suf = llama_vocab_fim_suf(vocab);
    llama_token fim_mid = llama_vocab_fim_mid(vocab);

    if (fim_pre == LLAMA_TOKEN_NULL || fim_suf == LLAMA_TOKEN_NULL || fim_mid == LLAMA_TOKEN_NULL) {
        LOGE("Model does not support FIM (missing special tokens)");
        return;
    }

    const char* raw_prefix = env->GetStringUTFChars(prefix, nullptr);
    const char* raw_suffix = env->GetStringUTFChars(suffix, nullptr);
    
    // Tokenize Prefix
    std::vector<llama_token> prefix_tokens(strlen(raw_prefix) + 2);
    int n_pre = llama_tokenize(vocab, raw_prefix, strlen(raw_prefix), prefix_tokens.data(), prefix_tokens.size(), false, false);
    if (n_pre < 0) {
        prefix_tokens.resize(-n_pre);
        n_pre = llama_tokenize(vocab, raw_prefix, strlen(raw_prefix), prefix_tokens.data(), prefix_tokens.size(), false, false);
    }
    prefix_tokens.resize(n_pre);

    // Tokenize Suffix
    std::vector<llama_token> suffix_tokens(strlen(raw_suffix) + 2);
    int n_suf = llama_tokenize(vocab, raw_suffix, strlen(raw_suffix), suffix_tokens.data(), suffix_tokens.size(), false, false);
    if (n_suf < 0) {
        suffix_tokens.resize(-n_suf);
        n_suf = llama_tokenize(vocab, raw_suffix, strlen(raw_suffix), suffix_tokens.data(), suffix_tokens.size(), false, false);
    }
    suffix_tokens.resize(n_suf);

    env->ReleaseStringUTFChars(prefix, raw_prefix);
    env->ReleaseStringUTFChars(suffix, raw_suffix);

    // Construct FIM Prompt: [PRE] prefix [SUF] suffix [MID]
    std::vector<llama_token> prompt_tokens;
    // Add BOS if needed? Usually FIM models handle it or not. Let's assume BOS is implicitly added by tokenizer if 'add_special' was true, but we used false.
    // Ideally, check llama_vocab_get_add_bos(vocab)
    
    if (llama_vocab_get_add_bos(vocab)) {
        prompt_tokens.push_back(llama_vocab_bos(vocab));
    }

    prompt_tokens.push_back(fim_pre);
    prompt_tokens.insert(prompt_tokens.end(), prefix_tokens.begin(), prefix_tokens.end());
    prompt_tokens.push_back(fim_suf);
    prompt_tokens.insert(prompt_tokens.end(), suffix_tokens.begin(), suffix_tokens.end());
    prompt_tokens.push_back(fim_mid);

    // Launch worker
    GenerateParams params;
    params.wrapper = wrapper;
    params.prompt = ""; // Not used
    params.prompt_tokens = prompt_tokens;
    params.grammar = "";
    env->GetJavaVM(&params.jvm);
    params.callback = env->NewGlobalRef(callback);
    jclass callbackClass = env->GetObjectClass(callback);
    params.onTokenMethod = env->GetMethodID(callbackClass, "onToken", "([B)V"); // Byte array signature

    pthread_t thread;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    size_t stack_size = 8 * 1024 * 1024;
    pthread_attr_setstacksize(&attr, stack_size);

    int rc = pthread_create(&thread, &attr, [](void* arg) -> void* {
        GenerateParams* p = (GenerateParams*)arg;
        generate_worker(p);
        return nullptr;
    }, &params);

    if (rc != 0) {
        LOGE("Failed to create worker thread for infill");
    } else {
        pthread_join(thread, nullptr);
    }

    pthread_attr_destroy(&attr);
    env->DeleteGlobalRef(params.callback);
}

}