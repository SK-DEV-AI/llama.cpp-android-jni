package com.cortex.app

/**
 * Thread configuration for llama.cpp inference.
 * Controls CPU threading behavior for optimal performance on different devices.
 */
data class ThreadConfig(
    /** Number of threads to use for generation. 0 = auto (all cores). Default: 0 */
    val nThreads: Int = 0,
    
    /** Number of threads to use for batch processing. 0 = auto. Default: 0 */
    val nThreadsBatch: Int = 0,
    
    /** Whether to use CPU affinity (pin threads to cores). Default: false */
    val cpuAffinity: Boolean = false,
    
    /** Priority hint for threads (0=normal, 1=high). Default: 0 */
    val threadPriority: Int = 0
) {
    companion object {
        /** Auto-detect optimal thread count based on device */
        val AUTO = ThreadConfig(nThreads = 0, nThreadsBatch = 0)
        
        /** Single-threaded mode (useful for debugging or battery saving) */
        val SINGLE_THREAD = ThreadConfig(nThreads = 1, nThreadsBatch = 1)
        
        /** Performance mode (use all available cores) */
        val PERFORMANCE = ThreadConfig(
            nThreads = Runtime.getRuntime().availableProcessors(),
            nThreadsBatch = Runtime.getRuntime().availableProcessors(),
            cpuAffinity = true
        )
        
        /** Balanced mode (half cores for generation, all for batch) */
        val BALANCED: ThreadConfig
            get() {
                val totalCores = Runtime.getRuntime().availableProcessors()
                return ThreadConfig(
                    nThreads = (totalCores / 2).coerceAtLeast(1),
                    nThreadsBatch = totalCores
                )
            }
        
        /** Battery saver mode (minimal threads) */
        val BATTERY_SAVER = ThreadConfig(
            nThreads = 2,
            nThreadsBatch = 2
        )
    }
    
    /**
     * Get the effective thread count (handles 0 = auto)
     */
    fun effectiveThreads(): Int = if (nThreads <= 0) {
        Runtime.getRuntime().availableProcessors()
    } else {
        nThreads
    }
    
    /**
     * Get the effective batch thread count
     */
    fun effectiveBatchThreads(): Int = if (nThreadsBatch <= 0) {
        Runtime.getRuntime().availableProcessors()
    } else {
        nThreadsBatch
    }
}