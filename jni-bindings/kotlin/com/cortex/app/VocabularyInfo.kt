package com.cortex.app

/**
 * Vocabulary information and token operations.
 * Provides access to the model's token vocabulary.
 */
class VocabularyInfo private constructor(
    private val engine: LlamaEngine,
    private val contextPtr: Long
) {
    /**
     * Get the size of the vocabulary (number of tokens)
     */
    val vocabSize: Int
        get() = engine.getVocabSize(contextPtr)

    /**
     * Get the text representation of a token
     */
    fun getTokenText(token: Int): String {
        return engine.getTokenText(contextPtr, token) ?: ""
    }

    /**
     * Check if a token is an end-of-generation token
     */
    fun isEogToken(token: Int): Boolean {
        return engine.isEogToken(contextPtr, token)
    }

    /**
     * Check if a token is a control token
     */
    fun isControlToken(token: Int): Boolean {
        return engine.isControlToken(contextPtr, token)
    }

    /**
     * Get the beginning-of-sequence token ID
     */
    val bosToken: Int
        get() = engine.getBosToken(contextPtr)

    /**
     * Get the end-of-sequence token ID
     */
    val eosToken: Int
        get() = engine.getEosToken(contextPtr)

    /**
     * Get the end-of-turn token ID
     */
    val eotToken: Int
        get() = engine.getEotToken(contextPtr)

    /**
     * Get the separator token ID
     */
    val sepToken: Int
        get() = engine.getSepToken(contextPtr)

    /**
     * Get the newline token ID
     */
    val nlToken: Int
        get() = engine.getNlToken(contextPtr)

    /**
     * Get the padding token ID
     */
    val padToken: Int
        get() = engine.getPadToken(contextPtr)

    /**
     * Get the FIM (Fill-In-Middle) prefix token ID
     */
    val fimPreToken: Int
        get() = engine.getFimPreToken(contextPtr)

    /**
     * Get the FIM suffix token ID
     */
    val fimSufToken: Int
        get() = engine.getFimSufToken(contextPtr)

    /**
     * Get the FIM middle token ID
     */
    val fimMidToken: Int
        get() = engine.getFimMidToken(contextPtr)

    /**
     * Get all special tokens as a map
     */
    val specialTokens: Map<String, Int>
        get() = mapOf(
            "BOS" to bosToken,
            "EOS" to eosToken,
            "EOT" to eotToken,
            "SEP" to sepToken,
            "NL" to nlToken,
            "PAD" to padToken,
            "FIM_PRE" to fimPreToken,
            "FIM_SUF" to fimSufToken,
            "FIM_MID" to fimMidToken
        )

    companion object {
        /**
         * Create a VocabularyInfo instance for the given engine
         */
        fun from(engine: LlamaEngine): VocabularyInfo {
            return VocabularyInfo(engine, engine.getContextPointer())
        }
    }
}
