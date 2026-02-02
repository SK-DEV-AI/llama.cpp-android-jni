package com.cortex.app

/**
 * Sampling parameters for text generation.
 * Controls how tokens are selected during generation.
 */
data class SamplingParams(
    /** Temperature (0.0 - 2.0+). Higher = more random, Lower = more deterministic. Default: 0.8 */
    val temperature: Float = 0.8f,
    
    /** Top-K sampling (1 - vocab_size). Keep only top K tokens. Default: 40 */
    val topK: Int = 40,
    
    /** Top-P / Nucleus sampling (0.0 - 1.0). Cumulative probability cutoff. Default: 0.95 */
    val topP: Float = 0.95f,
    
    /** Min-P sampling (0.0 - 1.0). Minimum probability threshold. Default: 0.0 (disabled) */
    val minP: Float = 0.0f,
    
    /** Repetition penalty (1.0+). Higher = less repetition. Default: 1.0 (disabled) */
    val repeatPenalty: Float = 1.0f,
    
    /** Number of tokens to consider for repetition penalty. Default: 64 */
    val repeatLastN: Int = 64,
    
    /** Frequency penalty (-2.0 to 2.0). Penalize frequent tokens. Default: 0.0 */
    val frequencyPenalty: Float = 0.0f,
    
    /** Presence penalty (-2.0 to 2.0). Penalize present tokens. Default: 0.0 */
    val presencePenalty: Float = 0.0f,
    
    /** Random seed for reproducibility. Default: -1 (random) */
    val seed: Int = -1,
    
    /** Maximum tokens to generate. Default: 512 */
    val maxTokens: Int = 512,
    
    /** Stop sequences to end generation. Default: empty */
    val stopSequences: List<String> = emptyList(),
    
    /** Typical-P sampling (0.0 - 1.0). Filters by entropy. Default: 1.0 (disabled) */
    val typicalP: Float = 1.0f,
    
    /** XTC (Exclude Top Choices) probability (0.0 - 1.0). Default: 0.0 (disabled) */
    val xtcProbability: Float = 0.0f,
    
    /** XTC threshold (0.0 - 1.0). Tokens above this prob are excluded. Default: 0.1 */
    val xtcThreshold: Float = 0.1f,
    
    /** Mirostat mode (0=disabled, 1=v1, 2=v2). Entropy-based auto temp. Default: 0 */
    val mirostatMode: Int = 0,
    
    /** Mirostat target entropy (tau). Default: 5.0 */
    val mirostatTau: Float = 5.0f,
    
    /** Mirostat learning rate (eta). Default: 0.1 */
    val mirostatEta: Float = 0.1f,
    
    /** DRY (Don't Repeat Yourself) multiplier. Default: 0.0 (disabled) */
    val dryMultiplier: Float = 0.0f,
    
    /** DRY base penalty. Default: 1.75 */
    val dryBase: Float = 1.75f,
    
    /** DRY allowed length before penalty. Default: 2 */
    val dryAllowedLength: Int = 2
) {
    companion object {
        /** Conservative settings for deterministic output */
        val CONSERVATIVE = SamplingParams(
            temperature = 0.2f,
            topK = 10,
            topP = 0.5f,
            minP = 0.0f,
            repeatPenalty = 1.2f,
            typicalP = 1.0f,
            xtcProbability = 0.0f,
            mirostatMode = 0
        )
        
        /** Creative settings for diverse output */
        val CREATIVE = SamplingParams(
            temperature = 1.0f,
            topK = 100,
            topP = 0.98f,
            minP = 0.05f,
            repeatPenalty = 1.0f,
            typicalP = 0.95f,
            xtcProbability = 0.5f,
            mirostatMode = 0
        )
        
        /** Mirostat v2 settings for consistent entropy */
        val MIROSTAT = SamplingParams(
            temperature = 1.0f,
            mirostatMode = 2,
            mirostatTau = 5.0f,
            mirostatEta = 0.1f
        )
        
        /** Balanced settings (default) */
        val BALANCED = SamplingParams()
    }
}
