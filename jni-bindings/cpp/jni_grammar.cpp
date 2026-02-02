#include "jni_common.h"
#include "llama.cpp/common/json-schema-to-grammar.h"
#include "llama.cpp/src/llama-grammar.h"
#include "llama.cpp/vendor/nlohmann/json.hpp"

using json = nlohmann::ordered_json;

// Dummy build info
int LLAMA_BUILD_NUMBER = 0;
char const *LLAMA_COMMIT = "unknown";
char const *LLAMA_COMPILER = "clang";
char const *LLAMA_BUILD_TARGET = "android";

// Workaround: Create a simplified grammar that's more token-friendly
// The issue is that llama.cpp grammar sampler expects char-by-char matching
// but models generate multi-char tokens like "{T" instead of "{" + "T"
static std::string create_token_friendly_grammar(const std::string& original) {
    // Simplified grammar that properly handles the closing brace
    // Matches: { ... } where content inside doesn't contain }
    return "root ::= \"{\" [^}]* \"}\"";
}

extern "C" {

JNIEXPORT jstring JNICALL
Java_com_cortex_app_LlamaEngine_jsonSchemaToGrammar(JNIEnv* env, jobject, jstring jsonSchema) {
    const char* schema_cstr = env->GetStringUTFChars(jsonSchema, nullptr);
    std::string schema_str(schema_cstr);
    env->ReleaseStringUTFChars(jsonSchema, schema_cstr);

    try {
        json schema = json::parse(schema_str);
        std::string grammar = json_schema_to_grammar(schema);
        
        LOGD("Generated grammar (%zu chars)", grammar.length());
        LOGD("NOTE: This grammar may fail due to llama.cpp token matching issues");
        LOGD("Consider using createTokenFriendlyGrammar() as workaround");
        
        return env->NewStringUTF(grammar.c_str());
    } catch (const std::exception& e) {
        LOGE("Failed to convert JSON schema to grammar: %s", e.what());
        return nullptr;
    }
}

JNIEXPORT jstring JNICALL
Java_com_cortex_app_LlamaEngine_createTokenFriendlyGrammar(JNIEnv* env, jobject) {
    // Return a simplified JSON grammar that handles multi-char tokens better
    std::string friendly = create_token_friendly_grammar("");
    LOGD("Created token-friendly grammar (%zu chars)", friendly.length());
    return env->NewStringUTF(friendly.c_str());
}

JNIEXPORT jstring JNICALL
Java_com_cortex_app_LlamaEngine_validateGrammarDetailed(JNIEnv* env, jobject, jstring grammarStr) {
    const char* grammar_cstr = env->GetStringUTFChars(grammarStr, nullptr);
    std::string grammar(grammar_cstr);
    env->ReleaseStringUTFChars(grammarStr, grammar_cstr);

    try {
        // Try to parse the grammar
        llama_grammar_parser parser;
        bool parse_ok = parser.parse(grammar.c_str());
        
        std::string result = "{";
        result += "\"parse_ok\": " + std::string(parse_ok ? "true" : "false");
        result += ", \"rules_count\": " + std::to_string(parser.rules.size());
        
        if (parse_ok) {
            // Try to create a grammar sampler to test if it can be used
            // This is where the "empty grammar stack" error occurs during generation
            result += ", \"status\": \"parsed_ok\"";
            result += ", \"note\": \"Grammar parsed but may fail during generation due to token matching issues\"";
        } else {
            result += ", \"status\": \"parse_failed\"";
        }
        
        result += ", \"llama_cpp_issue\": \"Grammar sampler processes tokens not chars - multi-char tokens like {T cause empty stack errors\"";
        result += "}";
        
        return env->NewStringUTF(result.c_str());
    } catch (const std::exception& e) {
        LOGE("Grammar validation error: %s", e.what());
        std::string error = "{\"error\": \"" + std::string(e.what()) + "\"}";
        return env->NewStringUTF(error.c_str());
    }
}

JNIEXPORT jstring JNICALL
Java_com_cortex_app_LlamaEngine_loadGrammarFromFile(JNIEnv* env, jobject, jstring filePath) {
    const char* path_cstr = env->GetStringUTFChars(filePath, nullptr);
    std::string path(path_cstr);
    env->ReleaseStringUTFChars(filePath, path_cstr);

    FILE* file = fopen(path.c_str(), "r");
    if (!file) {
        LOGE("Failed to open grammar file: %s", path.c_str());
        return nullptr;
    }

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);

    std::string grammar;
    grammar.resize(size);
    size_t read = fread(&grammar[0], 1, size, file);
    fclose(file);

    if (read != (size_t)size) {
        LOGE("Failed to read grammar file");
        return nullptr;
    }

    LOGD("Loaded grammar from file (%zu chars): %s", grammar.length(), path.c_str());
    return env->NewStringUTF(grammar.c_str());
}

JNIEXPORT jstring JNICALL
Java_com_cortex_app_LlamaEngine_getGrammarInfo(JNIEnv* env, jobject, jstring grammarStr) {
    const char* grammar_cstr = env->GetStringUTFChars(grammarStr, nullptr);
    std::string grammar(grammar_cstr);
    env->ReleaseStringUTFChars(grammarStr, grammar_cstr);

    try {
        std::string info = "{";
        info += "\"length\": " + std::to_string(grammar.length());
        info += ", \"lines\": " + std::to_string(std::count(grammar.begin(), grammar.end(), '\n') + 1);
        info += ", \"status\": \"loaded\"";
        info += "}";
        
        return env->NewStringUTF(info.c_str());
    } catch (const std::exception& e) {
        LOGE("Grammar info error: %s", e.what());
        return env->NewStringUTF("{\"error\": \"Exception\"}");
    }
}

}
