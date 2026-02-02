package com.cortex.app

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/**
 * Advanced logits access and custom sampling for power users.
 * 
 * This class provides low-level access to model logits for implementing
 * custom sampling strategies, token filtering, and advanced generation techniques.
 */
class LogitsSampler(private val engine: LlamaEngine) {
    
    /**
     * Get the raw logits for the last decoded token.
     * 
     * @return Float array of size vocab_size containing log probabilities
     */
    fun getLogits(): FloatArray? {
        return engine.nativeGetLogits(engine.getContextPointer())
    }
    
    /**
     * Get logits for a specific token position.
     * Use negative indices to access from the end (-1 = last token).
     * 
     * @param index Token position index
     * @return Float array of logits
     */
    fun getLogitsAt(index: Int): FloatArray? {
        return engine.nativeGetLogitsIth(engine.getContextPointer(), index)
    }
    
    /**
     * Get the vocabulary size.
     * This is the size of the logits array.
     */
    fun vocabSize(): Int {
        return engine.nativeGetVocabSizeFromContext(engine.getContextPointer())
    }
    
    /**
     * Decode tokens and return logits without sampling.
     * After calling this, use getLogits() to access the raw logits.
     * 
     * @param tokens Token IDs to decode
     * @return true if successful
     */
    fun decode(tokens: IntArray): Boolean {
        return engine.nativeDecodeTokens(engine.getContextPointer(), tokens)
    }
    
    /**
     * Sample a token from provided logits.
     * 
     * @param logits Logits array (must be vocab_size)
     * @param temperature Sampling temperature
     * @return Sampled token ID
     */
    fun sampleFromLogits(logits: FloatArray, temperature: Float = 1.0f): Int {
        return engine.nativeSampleTokenFromLogits(engine.getContextPointer(), logits, temperature)
    }
    
    /**
     * Generate text with custom sampling using a callback.
     * The callback receives logits and returns the selected token.
     * 
     * @param prompt Input prompt
     * @param maxTokens Maximum tokens to generate
     * @param sampler Callback that receives logits and returns token ID
     * @return Flow of generated tokens
     */
    fun generateWithCustomSampler(
        prompt: String,
        maxTokens: Int = 512,
        sampler: (logits: FloatArray, position: Int) -> Int
    ): Flow<String> = flow {
        // Tokenize prompt
        val promptTokens = engine.tokenize(prompt)
        
        // Decode prompt
        decode(promptTokens)
        
        var position = 0
        
        while (position < maxTokens) {
            // Get logits
            val logits = getLogits()
            if (logits == null) break
            
            // Call custom sampler
            val tokenId = sampler(logits, position)
            
            // Check for EOS
            if (tokenId < 0) break
            
            // Convert to text
            val tokenText = engine.detokenize(intArrayOf(tokenId))
            emit(tokenText)
            
            // Decode this token for next iteration
            decode(intArrayOf(tokenId))
            
            position++
        }
    }
    
    /**
     * Find the top K tokens from logits.
     * 
     * @param logits Logits array
     * @param k Number of top tokens to return
     * @return List of (tokenId, probability) pairs sorted by probability
     */
    fun topK(logits: FloatArray, k: Int = 10): List<Pair<Int, Float>> {
        // Convert logits to probabilities using softmax
        val maxLogit = logits.maxOrNull() ?: 0f
        val expLogits = logits.map { kotlin.math.exp(it - maxLogit) }
        val sumExp = expLogits.sum()
        val probs = expLogits.map { it / sumExp }
        
        // Get top K
        return probs
            .mapIndexed { index, prob -> index to prob }
            .sortedByDescending { it.second }
            .take(k)
    }
    
    /**
     * Apply temperature scaling to logits.
     * 
     * @param logits Original logits
     * @param temperature Temperature (1.0 = no change, <1.0 = more focused, >1.0 = more random)
     * @return Scaled logits
     */
    fun applyTemperature(logits: FloatArray, temperature: Float): FloatArray {
        return logits.map { it / temperature }.toFloatArray()
    }
    
    /**
     * Apply top-p (nucleus) filtering to logits.
     * 
     * @param logits Original logits
     * @param p Cumulative probability threshold (0.0-1.0)
     * @return Masked logits (filtered tokens set to -Infinity)
     */
    fun applyTopP(logits: FloatArray, p: Float): FloatArray {
        // Convert to probabilities
        val maxLogit = logits.maxOrNull() ?: 0f
        val expLogits = logits.map { kotlin.math.exp(it - maxLogit) }
        val sumExp = expLogits.sum()
        val probs = expLogits.map { it / sumExp }
        
        // Sort by probability
        val sorted = probs
            .mapIndexed { index, prob -> index to prob }
            .sortedByDescending { it.second }
        
        // Find cutoff
        var cumsum = 0f
        val cutoffIndex = sorted.indexOfFirst { 
            cumsum += it.second
            cumsum > p 
        }.coerceAtLeast(0)
        
        // Create mask
        val allowed = sorted.take(cutoffIndex + 1).map { it.first }.toSet()
        
        // Apply mask
        return logits.mapIndexed { index, logit ->
            if (index in allowed) logit else Float.NEGATIVE_INFINITY
        }.toFloatArray()
    }
    
    /**
     * Apply repetition penalty to logits.
     * 
     * @param logits Original logits
     * @param previousTokens Tokens to penalize
     * @param penalty Penalty factor (1.0 = no penalty, >1.0 = penalize)
     * @return Penalized logits
     */
    fun applyRepetitionPenalty(
        logits: FloatArray,
        previousTokens: List<Int>,
        penalty: Float
    ): FloatArray {
        if (penalty <= 1.0f) return logits
        
        val result = logits.copyOf()
        previousTokens.toSet().forEach { tokenId ->
            if (tokenId in result.indices) {
                result[tokenId] = if (result[tokenId] > 0) {
                    result[tokenId] / penalty
                } else {
                    result[tokenId] * penalty
                }
            }
        }
        return result
    }
    
    /**
     * Sample using argmax (greedy decoding).
     * 
     * @param logits Logits array
     * @return Token with highest logit
     */
    fun argmax(logits: FloatArray): Int {
        return logits.indices.maxByOrNull { logits[it] } ?: 0
    }
    
    /**
     * Sample using softmax with temperature.
     * 
     * @param logits Logits array
     * @param temperature Temperature
     * @return Sampled token
     */
    fun sampleSoftmax(logits: FloatArray, temperature: Float = 1.0f): Int {
        val scaled = if (temperature != 1.0f) applyTemperature(logits, temperature) else logits
        
        // Softmax
        val maxLogit = scaled.maxOrNull() ?: 0f
        val expLogits = scaled.map { kotlin.math.exp(it - maxLogit) }
        val sumExp = expLogits.sum()
        val probs = expLogits.map { it / sumExp }
        
        // Sample
        val rand = kotlin.random.Random.nextFloat()
        var cumsum = 0f
        probs.forEachIndexed { index, prob ->
            cumsum += prob
            if (rand <= cumsum) return index
        }
        
        return probs.size - 1
    }
}

/**
 * Extension function to create LogitsSampler from LlamaEngine.
 */
fun LlamaEngine.logits(): LogitsSampler = LogitsSampler(this)