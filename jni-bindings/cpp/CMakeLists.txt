cmake_minimum_required(VERSION 3.22.1)

project("cortex-engine")

# -------------------------------------------------------------------------
# Flags & Optimization
# -------------------------------------------------------------------------
# Aggressive optimization for modern ARM64 (Pixel 6+, S21+, etc.)
# +fp16: Native half-precision
# +dotprod: Acceleration for quantized dot products
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=armv8.2-a+fp16+dotprod -flto")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=armv8.2-a+fp16+dotprod -flto")

# Enable Vulkan for GPU acceleration
set(GGML_VULKAN OFF CACHE BOOL "Enable Vulkan backend" FORCE)

# Disable OpenMP to prevent SIGSEGV/Stack issues on Android
set(GGML_OPENMP OFF CACHE BOOL "Disable OpenMP" FORCE)

# Disable unnecessary build targets to speed up compilation
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

# -------------------------------------------------------------------------
# PCRE2 Configuration (MUST be before llama.cpp)
# -------------------------------------------------------------------------
# PCRE2 uses non-recursive NFA matching, avoiding stack overflow on JNI threads

# Check if we're building for Android
if(ANDROID)
    message(STATUS "Building for Android - will download and build PCRE2 from source")
    
    # Download PCRE2 10.42 source
    set(PCRE2_VERSION "10.42")
    set(PCRE2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/pcre2-${PCRE2_VERSION}")
    set(PCRE2_URL "https://github.com/PCRE2Project/pcre2/releases/download/pcre2-${PCRE2_VERSION}/pcre2-${PCRE2_VERSION}.tar.gz")
    
    if(NOT EXISTS ${PCRE2_DIR})
        message(STATUS "Downloading PCRE2 ${PCRE2_VERSION}...")
        file(DOWNLOAD ${PCRE2_URL} ${CMAKE_CURRENT_BINARY_DIR}/pcre2.tar.gz SHOW_PROGRESS)
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar -xzf ${CMAKE_CURRENT_BINARY_DIR}/pcre2.tar.gz
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/vendor
        )
        file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/pcre2.tar.gz)
    endif()
    
    # Configure PCRE2 build options
    set(PCRE2_BUILD_PCRE2_8 ON CACHE BOOL "" FORCE)
    set(PCRE2_BUILD_PCRE2_16 OFF CACHE BOOL "" FORCE)
    set(PCRE2_BUILD_PCRE2_32 OFF CACHE BOOL "" FORCE)
    set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "" FORCE)
    set(PCRE2_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(PCRE2_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(PCRE2_SUPPORT_UNICODE ON CACHE BOOL "" FORCE)
    set(PCRE2_SUPPORT_JIT OFF CACHE BOOL "" FORCE)  # JIT doesn't work on all Android devices
    
    # Add PCRE2 before llama.cpp so headers are available
    add_subdirectory(${PCRE2_DIR} ${CMAKE_CURRENT_BINARY_DIR}/pcre2-build)
    
    # PCRE2 headers are generated in the build directory, not the source directory
    set(PCRE2_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/pcre2-build)
    set(PCRE2_LIBRARIES pcre2-8)
    
    include_directories(${PCRE2_INCLUDE_DIR})
    include_directories(${PCRE2_DIR}/src)
    message(STATUS "PCRE2 will be built from source for Android")
else()
    # Desktop build - use system PCRE2
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(PCRE2 QUIET libpcre2-8)
    endif()

    if(PCRE2_FOUND)
        message(STATUS "PCRE2 found via pkg-config")
        include_directories(${PCRE2_INCLUDE_DIRS})
    else()
        find_path(PCRE2_INCLUDE_DIR pcre2.h)
        find_library(PCRE2_LIBRARY pcre2-8)
        
        if(PCRE2_INCLUDE_DIR AND PCRE2_LIBRARY)
            message(STATUS "PCRE2 found: ${PCRE2_LIBRARY}")
            include_directories(${PCRE2_INCLUDE_DIR})
            set(PCRE2_LIBRARIES ${PCRE2_LIBRARY})
        else()
            message(FATAL_ERROR "PCRE2 not found! Please install libpcre2-8-dev or pcre2-devel")
        endif()
    endif()
endif()

# -------------------------------------------------------------------------
# Dependencies
# -------------------------------------------------------------------------
# Import llama.cpp as a subdirectory
add_subdirectory(llama.cpp)

# Link PCRE2 to llama library for regex support (required on Android)
if(ANDROID AND PCRE2_LIBRARIES)
    target_link_libraries(llama PRIVATE ${PCRE2_LIBRARIES})
endif()

# -------------------------------------------------------------------------
# Main Library (JNI)
# -------------------------------------------------------------------------
# Include the JNI directory for headers
include_directories(jni)
include_directories(llama.cpp/vendor)
include_directories(.)

add_library(cortex-engine SHARED
        jni/jni_common.cpp
        jni/jni_lifecycle.cpp
        jni/jni_completion.cpp
        jni/jni_embedding.cpp
        jni/jni_utils.cpp
        jni/jni_grammar.cpp
        jni/jni_session.cpp
        jni/jni_lora.cpp
        jni/jni_rerank.cpp
        jni/jni_model_info.cpp
        jni/jni_vocabulary.cpp
        jni/jni_backend.cpp
        jni/jni_threading.cpp
        jni/jni_sampling.cpp
        jni/jni_memory.cpp
        jni/jni_logits.cpp
        jni/jni_model_io.cpp
        
        # Llama.cpp components
        llama.cpp/common/json-schema-to-grammar.cpp
        llama.cpp/common/unicode.cpp
        llama.cpp/common/common.cpp
        llama.cpp/common/log.cpp
        llama.cpp/common/sampling.cpp
)

# Find Android system libraries
find_library(log-lib log)
find_library(android-lib android)

# Link dependencies
target_link_libraries(cortex-engine
    PRIVATE
    llama
    ggml
    ${log-lib}
    ${android-lib}
    ${PCRE2_LIBRARIES}
)

# If Vulkan is enabled, ensure it's linked
if (GGML_VULKAN)
    find_library(vulkan-lib vulkan)
    target_link_libraries(cortex-engine PRIVATE ${vulkan-lib})
endif()
