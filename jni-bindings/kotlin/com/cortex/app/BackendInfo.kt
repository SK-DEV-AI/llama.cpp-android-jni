package com.cortex.app

/**
 * Backend and GPU information.
 * Provides information about device capabilities and hardware support.
 */
class BackendInfo private constructor() {

    /**
     * Maximum number of devices available for computation
     */
    val maxDevices: Int
        get() = nativeGetMaxDevices()

    /**
     * Check if memory-mapped file support is available
     */
    val supportsMmap: Boolean
        get() = nativeSupportsMmap()

    /**
     * Check if memory locking support is available
     */
    val supportsMlock: Boolean
        get() = nativeSupportsMlock()

    /**
     * Check if GPU offload is supported
     */
    val supportsGpuOffload: Boolean
        get() = nativeSupportsGpuOffload()

    /**
     * Check if RPC (Remote Procedure Call) backend is supported
     */
    val supportsRpc: Boolean
        get() = nativeSupportsRpc()

    /**
     * Get all backend capabilities as a map
     */
    val capabilities: Map<String, Boolean>
        get() = mapOf(
            "mmap" to supportsMmap,
            "mlock" to supportsMlock,
            "gpuOffload" to supportsGpuOffload,
            "rpc" to supportsRpc
        )

    /**
     * Get a human-readable summary of backend capabilities
     */
    val summary: String
        get() = buildString {
            appendLine("Backend Capabilities:")
            appendLine("  Max Devices: $maxDevices")
            appendLine("  MMAP: ${if (supportsMmap) "✓" else "✗"}")
            appendLine("  MLOCK: ${if (supportsMlock) "✓" else "✗"}")
            appendLine("  GPU Offload: ${if (supportsGpuOffload) "✓" else "✗"}")
            appendLine("  RPC: ${if (supportsRpc) "✓" else "✗"}")
        }

    companion object {
        init {
            System.loadLibrary("cortex-engine")
        }

        /**
         * Get backend information singleton
         */
        fun getInstance(): BackendInfo {
            return BackendInfo()
        }

        // Native methods - no context needed, these are global backend functions
        @JvmStatic
        private external fun nativeGetMaxDevices(): Int

        @JvmStatic
        private external fun nativeSupportsMmap(): Boolean

        @JvmStatic
        private external fun nativeSupportsMlock(): Boolean

        @JvmStatic
        private external fun nativeSupportsGpuOffload(): Boolean

        @JvmStatic
        private external fun nativeSupportsRpc(): Boolean
    }
}
