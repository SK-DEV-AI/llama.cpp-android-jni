#include "jni_common.h"
#include <vector>
#include <string>
#include <algorithm>
#include <pthread.h>

static void string_replace_all(std::string & s, const std::string & search, const std::string & replace) {
    for (size_t pos = 0; ; pos += replace.length()) {
        pos = s.find(search, pos);
        if (pos == std::string::npos) break;
        s.erase(pos, search.length());
        s.insert(pos, replace);
    }
}

struct RerankParams {
    LlamaContextWrapper* wrapper;
    std::string query;
    std::vector<std::string> documents;
    std::vector<float> scores;
    bool success = false;
};

void rerank_worker(RerankParams* params) {
    const llama_vocab* vocab = llama_model_get_vocab(params->wrapper->model);
    const char* tmpl = llama_model_chat_template(params->wrapper->model, "rerank");
    
    int n_docs = params->documents.size();
    params->scores.resize(n_docs);
    
    // Process one document at a time to avoid batching issues
    llama_batch batch = llama_batch_init(4096, 0, 1); 

    for (int i = 0; i < n_docs; ++i) {
        std::vector<llama_token> tokens;
        
        if (tmpl) {
            std::string prompt = tmpl;
            string_replace_all(prompt, "{query}", params->query);
            string_replace_all(prompt, "{document}", params->documents[i]);
            
            std::vector<llama_token> prompt_tokens(prompt.length() + 2);
            int n = llama_tokenize(vocab, prompt.c_str(), prompt.length(), prompt_tokens.data(), prompt_tokens.size(), true, true);
            if (n < 0) {
                prompt_tokens.resize(-n);
                n = llama_tokenize(vocab, prompt.c_str(), prompt.length(), prompt_tokens.data(), prompt_tokens.size(), true, true);
            }
            prompt_tokens.resize(n);
            tokens = prompt_tokens;
        } else {
            // Manual format fallback
            std::vector<llama_token> q_toks(params->query.length() + 2);
            int nq = llama_tokenize(vocab, params->query.c_str(), params->query.length(), q_toks.data(), q_toks.size(), false, false);
            if (nq < 0) { q_toks.resize(-nq); nq = llama_tokenize(vocab, params->query.c_str(), params->query.length(), q_toks.data(), q_toks.size(), false, false); }
            q_toks.resize(nq);

            std::vector<llama_token> d_toks(params->documents[i].length() + 2);
            int nd = llama_tokenize(vocab, params->documents[i].c_str(), params->documents[i].length(), d_toks.data(), d_toks.size(), false, false);
            if (nd < 0) { d_toks.resize(-nd); nd = llama_tokenize(vocab, params->documents[i].c_str(), params->documents[i].length(), d_toks.data(), d_toks.size(), false, false); }
            d_toks.resize(nd);

            if (llama_vocab_get_add_bos(vocab)) tokens.push_back(llama_vocab_bos(vocab));
            tokens.insert(tokens.end(), q_toks.begin(), q_toks.end());
            
            llama_token sep = llama_vocab_sep(vocab);
            llama_token eos = llama_vocab_eos(vocab);
            if (sep == LLAMA_TOKEN_NULL) sep = eos;

            if (sep != LLAMA_TOKEN_NULL) tokens.push_back(sep);
            tokens.insert(tokens.end(), d_toks.begin(), d_toks.end());
            if (llama_vocab_get_add_eos(vocab) && eos != LLAMA_TOKEN_NULL) tokens.push_back(eos);
        }

        // Clear previous batch state
        batch_clear(batch);

        // Add tokens to batch (Sequence ID 0 for all single-doc batches)
        for (size_t t = 0; t < tokens.size(); ++t) {
            batch_add(batch, tokens[t], t, { 0 }, (t == tokens.size() - 1)); 
        }

        // Run inference
        int ret = 0;
        if (llama_model_has_encoder(params->wrapper->model)) {
            ret = llama_encode(params->wrapper->ctx, batch);
        } else {
            ret = llama_decode(params->wrapper->ctx, batch);
        }

        if (ret < 0) {
            LOGE("Inference failed for doc %d. Return code: %d", i, ret);
            llama_batch_free(batch);
            params->success = false;
            return;
        }

        // Get score
        float* embd = llama_get_embeddings_seq(params->wrapper->ctx, 0);
        if (!embd) {
            // Fallback to ith
            // For RANK pooling, it might be ith?
            // Try to find the token that had logits=true (last token)
            // batch_add sets logits=true for last token.
            // llama_get_embeddings_ith(ctx, i) -> gets embedding for i-th token in batch
            // The last token is at index batch.n_tokens - 1
            embd = llama_get_embeddings_ith(params->wrapper->ctx, batch.n_tokens - 1);
        }

        if (!embd) {
            LOGE("Failed to get embeddings/score for doc %d", i);
            params->scores[i] = -999.0f;
        } else {
            params->scores[i] = embd[0];
        }
    }
    
    llama_batch_free(batch);
    params->success = true;
}

extern "C" {

JNIEXPORT jfloatArray JNICALL
Java_com_cortex_app_LlamaEngine_rerank(JNIEnv* env, jobject, jlong contextPtr, jstring jQuery, jobjectArray jDocuments) {
    if (contextPtr == 0) return nullptr;
    auto* wrapper = (LlamaContextWrapper*)contextPtr;
    
    const char* query_c = env->GetStringUTFChars(jQuery, nullptr);
    std::string query(query_c);
    env->ReleaseStringUTFChars(jQuery, query_c);

    jsize n_docs = env->GetArrayLength(jDocuments);
    std::vector<std::string> documents(n_docs);
    for (int i = 0; i < n_docs; ++i) {
        jstring jDoc = (jstring)env->GetObjectArrayElement(jDocuments, i);
        const char* doc_c = env->GetStringUTFChars(jDoc, nullptr);
        documents[i] = std::string(doc_c);
        env->ReleaseStringUTFChars(jDoc, doc_c);
    }

    RerankParams params;
    params.wrapper = wrapper;
    params.query = query;
    params.documents = documents;

    pthread_t thread;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    size_t stack_size = 32 * 1024 * 1024; 
    pthread_attr_setstacksize(&attr, stack_size);

    int rc = pthread_create(&thread, &attr, [](void* arg) -> void* {
        RerankParams* p = (RerankParams*)arg;
        rerank_worker(p);
        return nullptr;
    }, &params);

    if (rc != 0) {
        LOGE("Failed to create worker thread for rerank");
        pthread_attr_destroy(&attr);
        return nullptr;
    }

    pthread_join(thread, nullptr);
    pthread_attr_destroy(&attr);

    if (!params.success) return nullptr;

    jfloatArray result = env->NewFloatArray(n_docs);
    env->SetFloatArrayRegion(result, 0, n_docs, params.scores.data());
    return result;
}

}
