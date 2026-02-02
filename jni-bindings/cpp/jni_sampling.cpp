#include "jni_common.h"

// Sampling parameter struct (mirrors jni_completion.cpp)
struct GenerateParams {
    LlamaContextWrapper* wrapper;
    std::string prompt;
    std::vector<llama_token> prompt_tokens;
    std::string grammar;
    JavaVM* jvm;
    jobject callback;
    jmethodID onTokenMethod;
    // Basic sampling parameters
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

/**
 * Build a sampler chain based on generation parameters.
 * 
 * This function encapsulates all sampler initialization logic,
 * making it easy to modify sampling strategies without touching
 * the main generation loop.
 * 
 * @param params Generation parameters containing all sampling settings
 * @param vocab Vocabulary for grammar and DRY samplers
 * @return Configured sampler chain (must be freed by caller)
 */
llama_sampler* build_sampler_chain(const GenerateParams* params, const llama_vocab* vocab) {
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    
    // 1. Grammar sampler (if grammar provided)
    // Grammar sampler should be FIRST in the chain to constrain the token space
    if (!params->grammar.empty()) {
        LOGD("Initializing grammar sampler. Length: %zu", params->grammar.length());
        
        // Make a local copy to ensure thread locality
        std::string local_grammar = params->grammar; 
        LOGD("Grammar content: %s", local_grammar.c_str());

        llama_sampler* grammar_sampler = llama_sampler_init_grammar(vocab, local_grammar.c_str(), "root");
        if (grammar_sampler) {
            llama_sampler_chain_add(smpl, grammar_sampler);
            LOGD("Grammar sampler added.");
        } else {
            LOGE("Failed to initialize grammar sampler.");
        }
    }
    
    // 2. DRY sampler (Don't Repeat Yourself)
    // DRY prevents repetitive token sequences
    if (params->dryMultiplier > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_dry(
            vocab,
            0,  // n_ctx_train: 0 uses default
            params->dryMultiplier,
            params->dryBase,
            params->dryAllowedLength,
            params->repeatLastN,  // dry_penalty_last_n
            nullptr,  // seq_breakers: use defaults
            0  // num_breakers
        ));
    }
    
    // 3. Repetition penalties sampler
    // Penalizes recently seen tokens
    if (params->repeatPenalty != 1.0f || params->frequencyPenalty != 0.0f || params->presencePenalty != 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
            params->repeatLastN,
            params->repeatPenalty,
            params->frequencyPenalty,
            params->presencePenalty
        ));
    }
    
    // 4. Top-K sampling
    // Keeps only the top K highest probability tokens
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params->topK));
    
    // 5. Top-P (Nucleus) sampling
    // Keeps tokens until cumulative probability reaches threshold
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params->topP, 1));
    
    // 6. Min-P sampling
    // Filters tokens below minimum probability threshold
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(params->minP, 1));
    
    // 7. Typical-P sampling (if enabled)
    // Filters tokens by entropy/typicality
    if (params->typicalP < 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_typical(params->typicalP, 1));
    }
    
    // 8. XTC (Exclude Top Choices) sampling (if enabled)
    // Randomly excludes top probability tokens for diversity
    if (params->xtcProbability > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_xtc(
            params->xtcProbability, 
            params->xtcThreshold, 
            1,  // min_keep
            1   // seed (use main seed)
        ));
    }
    
    // 9. Mirostat sampling (if enabled)
    // Entropy-based automatic temperature control
    if (params->mirostatMode == 1) {
        llama_sampler_chain_add(smpl, llama_sampler_init_mirostat(
            llama_vocab_n_tokens(vocab),
            params->seed,
            params->mirostatTau,
            params->mirostatEta,
            100  // m: number of tokens considered
        ));
    } else if (params->mirostatMode == 2) {
        llama_sampler_chain_add(smpl, llama_sampler_init_mirostat_v2(
            params->seed, 
            params->mirostatTau, 
            params->mirostatEta
        ));
    }
    
    // 10. Temperature sampling (only if not using mirostat)
    // Controls randomness of token selection
    if (params->mirostatMode == 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(params->temperature));
    }
    
    // 11. Distribution sampling with seed
    // Final sampling step that actually picks a token
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params->seed));
    
    return smpl;
}

// Additional sampler utility functions

/**
 * Get the seed from a sampler chain.
 * This is useful when a random seed (-1) was used and you want to know the actual seed.
 * 
 * JNI Signature: getSamplerSeed(J)I
 * 
 * @param samplerPtr Pointer to sampler (from build_sampler_chain)
 * @return The seed used, or -1 if not applicable
 */
extern "C" JNIEXPORT jint JNICALL
Java_com_cortex_app_LlamaEngine_nativeGetSamplerSeed(
        JNIEnv* env,
        jobject,
        jlong samplerPtr) {
    
    if (samplerPtr == 0) {
        return -1;
    }
    
    auto* smpl = (llama_sampler*)samplerPtr;
    uint32_t seed = llama_sampler_get_seed(smpl);
    
    // LLAMA_DEFAULT_SEED is typically 0xFFFFFFFF or similar
    if (seed == UINT32_MAX || seed == 0xFFFFFFFF) {
        return -1; // Return -1 for default/random seed
    }
    
    return static_cast<jint>(seed);
}

/**
 * Reset sampler performance metrics.
 * 
 * JNI Signature: resetSamplerPerformance(J)V
 */
extern "C" JNIEXPORT void JNICALL
Java_com_cortex_app_LlamaEngine_nativeResetSamplerPerformance(
        JNIEnv* env,
        jobject,
        jlong samplerPtr) {
    
    if (samplerPtr == 0) return;
    
    auto* smpl = (llama_sampler*)samplerPtr;
    llama_perf_sampler_reset(smpl);
    
    LOGD("Sampler performance metrics reset");
}