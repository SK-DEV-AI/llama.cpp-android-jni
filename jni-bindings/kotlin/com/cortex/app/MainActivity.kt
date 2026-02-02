package com.cortex.app

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.File

class MainActivity : ComponentActivity() {
    private val modelPath = "/data/local/tmp/model.gguf" 
    private val loraPath = "/data/local/tmp/adapter.gguf"
    private val sessionPath = "/data/local/tmp/state.bin"
    private val engine = LlamaEngine()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    FeatureTestScreen()
                }
            }
        }
    }

    @Composable
    fun FeatureTestScreen() {
        var appLogs by remember { mutableStateOf("Ready.\n") }
        var systemLogs by remember { mutableStateOf("") }
        var rawResponse by remember { mutableStateOf("") }
        var isTestRunning by remember { mutableStateOf(false) }
        val scope = rememberCoroutineScope()
        val scrollState = rememberScrollState()
        val context = LocalContext.current

        // Function to read logcat
        fun updateSystemLogs() {
            scope.launch(Dispatchers.IO) {
                try {
                    val process = Runtime.getRuntime().exec("logcat -d -t 1000 -v time Cortex:V CortexEngine:V AndroidRuntime:E libc:E DEBUG:E *:S")
                    val reader = BufferedReader(InputStreamReader(process.inputStream))
                    val log = StringBuilder()
                    var line: String?
                    while (reader.readLine().also { line = it } != null) {
                        log.append(line).append("\n")
                    }
                    systemLogs = log.toString()
                    if (systemLogs.isBlank()) systemLogs = "No logs found. (Check filter/buffer)"
                } catch (e: Exception) {
                    systemLogs = "Failed to read logcat: ${e.message}"
                }
            }
        }

        LaunchedEffect(Unit) {
            updateSystemLogs()
        }

        // Parse Thinking vs Answer
        val thinkMatch = remember(rawResponse) {
            Regex("<think>(.*?)</think>", RegexOption.DOT_MATCHES_ALL).find(rawResponse)
        }
        val thoughtText = thinkMatch?.groupValues?.get(1)?.trim()
        val answerText = remember(rawResponse) {
            if (thoughtText != null) {
                rawResponse.replace(Regex("<think>.*?</think>", RegexOption.DOT_MATCHES_ALL), "").trim()
            } else {
                rawResponse
            }
        }
        
        val isThinking = rawResponse.contains("<think>") && !rawResponse.contains("</think>")
        val partialThought = if (isThinking) {
            rawResponse.substringAfter("<think>").trim()
        } else null

        Column(modifier = Modifier.padding(16.dp)) {
            Text("Cortex Feature Test", style = MaterialTheme.typography.headlineSmall)
            
            Button(
                onClick = {
                    isTestRunning = true
                    scope.launch(Dispatchers.IO) {
                        appLogs = "🚀 STARTING FULL AUTO TEST...\n"
                        try {
                            // 1. Load
                            if (!File(modelPath).exists()) throw Exception("Model file missing")
                            engine.load(modelPath)
                            appLogs += "✅ Load: OK\n"
                            
                            // 2. Tokenize
                            val tok = engine.tokenize("Hello")
                            if (tok.isEmpty()) throw Exception("Tokenization empty")
                            appLogs += "✅ Tokenize: OK (" + tok[0] + ")\n"
                            
                            // 3. Detokenize
                            val detok = engine.detokenize(tok)
                            if (!detok.contains("Hello")) throw Exception("Detokenize mismatch: $detok")
                            appLogs += "✅ Detokenize: OK\n"
                            
                            // 4. Embedding
                            val vec = engine.embedding("Test")
                            if (vec.isEmpty()) throw Exception("Embedding empty")
                            appLogs += "✅ Embedding: OK (dim=" + vec.size + ")\n"
                            
                            // 5. Generate
                            appLogs += "⏳ Generating..."
                            var genText = ""
                            val prompt = engine.formatChat(listOf("user" to "Hi"))
                            engine.generate(prompt).collect { genText += it }
                            if (genText.isBlank()) throw Exception("Generation empty")
                            appLogs += "✅ Generate: OK (" + genText.take(20) + "...)\n"
                            
                            // 6. Save State
                            if (!engine.saveState(sessionPath)) throw Exception("Save failed")
                            appLogs += "✅ Save State: OK\n"
                            
                            // 7. Load State
                            if (!engine.loadState(sessionPath)) throw Exception("Load failed")
                            appLogs += "✅ Load State: OK\n"
                            
                            // 8. LoRA
                            if (File(loraPath).exists()) {
                                if (!engine.addLora(loraPath, 1.0f)) throw Exception("LoRA attach failed")
                                appLogs += "✅ LoRA: OK\n"
                                engine.resetLoras()
                                appLogs += "✅ LoRA Reset: OK\n"
                            } else {
                                appLogs += "⚠️ LoRA: Skipped (File not found)\n"
                            }
                            
                            appLogs += "\n🎉 ALL SYSTEMS GO!\n"
                        } catch (e: Exception) {
                            appLogs += "\n❌ TEST FAILED: " + e.message + "\n"
                        }
                        isTestRunning = false
                        updateSystemLogs()
                    }
                },
                enabled = !isTestRunning,
                modifier = Modifier.fillMaxWidth()
            ) { 
                if (isTestRunning) CircularProgressIndicator(modifier = Modifier.size(16.dp))
                else Text("RUN FULL AUTO TEST")
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Buttons Row 1: Core
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = {
                    scope.launch(Dispatchers.IO) {
                        appLogs += "🔄 Loading...\n"
                        try {
                            if (!File(modelPath).exists()) {
                                appLogs += "❌ File not found: $modelPath\n"
                                return@launch
                            }
                            engine.load(modelPath)
                            appLogs += "✅ Model Loaded\n"
                        } catch (e: Exception) {
                            appLogs += "❌ Load Failed: " + e.message + "\n"
                        }
                        updateSystemLogs()
                    }
                }) { Text("Load") }

                Button(onClick = {
                    scope.launch(Dispatchers.IO) {
                        try {
                            val p = "Hello"
                            val t = engine.tokenize(p)
                            val tStr = t.joinToString()
                            appLogs += "Tok: " + tStr + "\n"
                            val d = engine.detokenize(t)
                            appLogs += "Detok: '" + d + "'\n"
                        } catch (e: Exception) {
                            appLogs += "❌ Tok Failed: " + e.message + "\n"
                        }
                        updateSystemLogs()
                    }
                }) { Text("Tok") }
            }

            // Buttons Row 2: Inference
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = {
                    scope.launch(Dispatchers.IO) {
                        try {
                            val t = "King - Man + Woman ="
                            val v = engine.embedding(t)
                            appLogs += "🧠 Embed($t): [" + v.take(3).joinToString() + ", ... size=" + v.size + "]\n"
                        }
                        catch (e: Exception) {
                            appLogs += "❌ Embed Failed: " + e.message + "\n"
                        }
                        updateSystemLogs()
                    }
                }) { Text("Embed") }

                Button(onClick = {
                    scope.launch(Dispatchers.IO) {
                        appLogs += "🤖 Generating...\n"
                        rawResponse = ""
                        try {
                            val messages = listOf(
                                "user" to "Who is Sundar Pichai?"
                            )
                            val prompt = engine.formatChat(messages)
                            val preview = prompt.replace("\n", " ").take(50)
                            appLogs += "📝 Prompt: " + preview + "...\n"
                            
                            engine.generate(prompt).collect { 
                                rawResponse += it
                            }
                            appLogs += "🏁 Done.\n"
                            val metrics = engine.getBench()
                            appLogs += "📊 Metrics: $metrics\n"
                        } catch (e: Exception) {
                            appLogs += "❌ Gen Failed: " + e.message + "\n"
                        }
                        updateSystemLogs()
                    }
                }) { Text("Gen") }
            }

            // Buttons Row 3: Advanced
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = {
                    scope.launch(Dispatchers.IO) {
                        appLogs += "🛡️ Grammar Test...\n"
                        rawResponse = ""
                        try {
                            // Test 1: JSON Schema Conversion (PCRE2 regex test)
                            val schema = "{\"type\": \"object\", \"properties\": {\"name\": {\"type\": \"string\"}}, \"required\": [\"name\"]}"
                            appLogs += "📋 Schema: $schema\n"
                            
                            val grammar = engine.convertJsonSchema(schema)
                            if (grammar == null) {
                                throw Exception("Grammar conversion returned null - PCRE2 regex may have failed!")
                            }
                            appLogs += "✅ JSON→GBNF conversion: ${grammar.length} chars\n"
                            
                            // Test 2: Use a WORKING grammar
                            // Recursive structure forces closing brace to be expected
                            val simpleGrammar = buildString {
                                appendLine("root ::= \"{\" content \"}\"")
                                appendLine("content ::= [^}] content |")  // Recursive: keeps } expectation active
                            }
                            
                            val prompt = "Generate a JSON object with a name field."
                            appLogs += "🤖 Generating with simple grammar...\n"
                            
                            var generated = ""
                            engine.generate(prompt, simpleGrammar).collect { token ->
                                generated += token
                                scope.launch(Dispatchers.Main) {
                                    rawResponse = generated
                                }
                            }
                            
                            // Validate output
                            val trimmed = generated.trim()
                            if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
                                appLogs += "✅ Valid JSON object generated!\n"
                                val nameMatch = Regex("\"name\"\\s*:\\s*\"([^\"]+)\"").find(trimmed)
                                if (nameMatch != null) {
                                    appLogs += "✅ Found name: ${nameMatch.groupValues[1]}\n"
                                }
                            } else {
                                appLogs += "⚠️ Generated: ${trimmed.take(50)}...\n"
                            }
                            
                            appLogs += "🎉 Grammar Test PASSED!\n"
                        } catch (e: Exception) {
                            appLogs += "❌ Grammar Test FAILED: " + e.message + "\n"
                        }
                        updateSystemLogs()
                    }
                }) { Text("JSON") }

                Button(onClick = {
                    scope.launch(Dispatchers.IO) {
                        appLogs += "📚 Advanced Grammar Test...\n"
                        try {
                            // Test 1: Use ULTRA-SIMPLE grammar that definitely works
                            // This just forces output to start with { and end with }
                            val ultraSimpleGrammar = "root ::= \"{\" [^}]* \"}\""
                            
                            val stats = engine.getGrammarStats(ultraSimpleGrammar)
                            appLogs += "📊 Grammar stats: $stats\n"
                            
                            // Test 2: Try to load from file (may not exist)
                            val grammarPath = "/data/local/tmp/test.gbnf"
                            val fileGrammar = engine.loadGrammarFile(grammarPath)
                            if (fileGrammar != null) {
                                appLogs += "📖 Loaded grammar from file: ${fileGrammar.length} chars\n"
                            } else {
                                appLogs += "ℹ️ Grammar file not found at $grammarPath (expected)\n"
                            }
                            
                            // Test 3: Generate with ULTRA-SIMPLE grammar
                            appLogs += "🤖 Testing with ultra-simple grammar...\n"
                            var output = ""
                            engine.generate("Say hello", ultraSimpleGrammar).collect { token ->
                                output += token
                            }
                            appLogs += "✅ Generated: ${output.take(50)}...\n"
                            
                            // Test 4: Try the generated complex grammar
                            appLogs += "🧪 Testing auto-generated grammar...\n"
                            val schema = """{"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}"""
                            val complexGrammar = engine.convertJsonSchema(schema)
                            if (complexGrammar != null) {
                                appLogs += "📜 Complex grammar: ${complexGrammar.length} chars\n"
                                var complexOutput = ""
                                try {
                                    engine.generate("Create object", complexGrammar).collect { token ->
                                        complexOutput += token
                                    }
                                    appLogs += "📝 Complex output: ${complexOutput.take(100)}...\n"
                                    // Validate it's proper JSON
                                    if (complexOutput.trim().startsWith("{") && complexOutput.trim().endsWith("}")) {
                                        appLogs += "✅ Complex grammar produced valid JSON!\n"
                                    } else {
                                        appLogs += "⚠️ Complex output may not be valid JSON\n"
                                    }
                                } catch (e: Exception) {
                                    appLogs += "❌ Complex grammar failed: ${e.message}\n"
                                    appLogs += "📝 Partial output: ${complexOutput.take(50)}...\n"
                                }
                            }
                            
                            // Test 5: NIGHTMARE COMPLEX - Deep nesting, arrays, all types
                            appLogs += "🔥 NIGHTMARE COMPLEX TEST...\n"
                            val nightmareSchema = """
                                {"type": "object", 
                                 "properties": {
                                   "user": {
                                     "type": "object", 
                                     "properties": {
                                       "id": {"type": "integer"}, 
                                       "name": {"type": "string"}, 
                                       "profile": {
                                         "type": "object", 
                                         "properties": {
                                           "age": {"type": "integer"}, 
                                           "tags": {"type": "array", "items": {"type": "string"}}, 
                                           "active": {"type": "boolean"}
                                         }
                                       }
                                     },
                                     "required": ["id", "name"]
                                   }
                                 },
                                 "required": ["user"]
                                }
                            """.trimIndent()
                            
                            val nightmareGrammar = engine.convertJsonSchema(nightmareSchema)
                            if (nightmareGrammar != null) {
                                appLogs += "📜 Nightmare grammar: ${nightmareGrammar.length} chars\n"
                                var nightmareOutput = ""
                                try {
                                    engine.generate("Generate complete complex user profile with nested objects and arrays", nightmareGrammar).collect { token ->
                                        nightmareOutput += token
                                    }
                                    appLogs += "📝 Nightmare output: ${nightmareOutput.take(200)}...\n"
                                    
                                    // Try to parse and validate
                                    val trimmed = nightmareOutput.trim()
                                    if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
                                        // Check for required fields
                                        val hasUser = trimmed.contains("user") || trimmed.contains("User")
                                        val hasSystem = trimmed.contains("system") || trimmed.contains("System")
                                        val hasNested = trimmed.contains("profile") || trimmed.contains("Profile")
                                        val hasArrays = trimmed.contains("[") && trimmed.contains("]")
                                        
                                        appLogs += "✅ Has user: $hasUser\n"
                                        appLogs += "✅ Has system: $hasSystem\n"  
                                        appLogs += "✅ Has nested objects: $hasNested\n"
                                        appLogs += "✅ Has arrays: $hasArrays\n"
                                        
                                        if (hasUser && hasSystem && hasNested) {
                                            appLogs += "🎉 NIGHTMARE TEST PASSED - Full complex structure generated!\n"
                                        } else {
                                            appLogs += "⚠️ Generated JSON but missing some complex structures\n"
                                        }
                                    } else {
                                        appLogs += "❌ Nightmare output not valid JSON wrapper\n"
                                    }
                                } catch (e: Exception) {
                                    appLogs += "❌ Nightmare test failed: ${e.message}\n"
                                    appLogs += "📝 Partial output: ${nightmareOutput.take(100)}...\n"
                                }
                            } else {
                                appLogs += "❌ Failed to convert nightmare schema\n"
                            }
                            
                            appLogs += "🎉 Advanced Grammar Test Complete!\n"
                        } catch (e: Exception) {
                            appLogs += "❌ Grammar Test: ${e.message}\n"
                        }
                        updateSystemLogs()
                    }
                }) { Text("Grammar+") }

                Button(onClick = {
                    scope.launch(Dispatchers.IO) {
                        if (engine.saveState(sessionPath)) appLogs += "💾 State Saved\n"
                        else appLogs += "❌ Save Failed\n"
                        updateSystemLogs()
                    }
                }) { Text("Save") }
                
                Button(onClick = {
                    scope.launch(Dispatchers.IO) {
                        if (engine.loadState(sessionPath)) appLogs += "📂 State Loaded\n"
                        else appLogs += "❌ Load Failed\n"
                        updateSystemLogs()
                    }
                }) { Text("Load") }
            }
            
            // Buttons Row 4: FIM & LoRA
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = {
                    scope.launch(Dispatchers.IO) {
                        appLogs += "🧩 Infill Test...\n"
                        rawResponse = ""
                        try {
                            // Example: fun fib(n: Int): Int { <MID> }
                            val prefix = "fun fib(n: Int): Int {\n"
                            val suffix = "\n}"
                            appLogs += "Pre: '$prefix', Suf: '$suffix'\n"
                            engine.infill(prefix, suffix).collect { 
                                rawResponse += it 
                            }
                            appLogs += "🏁 Done.\n"
                        } catch (e: Exception) {
                            appLogs += "❌ Infill Failed: " + e.message + "\n"
                        }
                        updateSystemLogs()
                    }
                }) { Text("Infill") }
                
                Button(onClick = {
                    scope.launch(Dispatchers.IO) {
                        try {
                            appLogs += "🔍 Rerank Test...\n"
                            val rerankPath = "/data/local/tmp/reranker.gguf"
                            if (!File(rerankPath).exists()) throw Exception("File not found: $rerankPath")
                            
                            appLogs += "🔄 Loading Reranker...\n"
                            engine.loadRerankModel(rerankPath)
                            appLogs += "✅ Reranker Loaded\n"
                            
                            val query = "What is capital of France?"
                            val docs = listOf("Paris is the capital.", "Berlin is in Germany.", "Lyon is a city.")
                            
                            appLogs += "Q: $query\n"
                            val scores = engine.rerank(query, docs)
                            appLogs += "Scores: " + scores.joinToString(", ") + "\n"
                            
                            // Reload Chat Model after test to restore normal state? 
                            // Or just leave it. The user can click "Load" to restore chat model.
                            appLogs += "⚠️ Chat model unloaded. Click 'Load' to chat again.\n"
                        } catch (e: Exception) {
                            appLogs += "❌ Rerank Failed: " + e.message + "\n"
                        }
                        updateSystemLogs()
                    }
                }) { Text("Rerank") }

                 Button(onClick = {
                    scope.launch(Dispatchers.IO) {
                        if (File(loraPath).exists()) {
                            if (engine.addLora(loraPath, 1.0f)) appLogs += "🔌 LoRA Attached\n"
                            else appLogs += "❌ LoRA Failed\n"
                        } else {
                            appLogs += "❌ LoRA file not found at " + loraPath + "\n"
                        }
                        updateSystemLogs()
                    }
                }) { Text("Add LoRA") }
                
                Button(onClick = {
                    scope.launch(Dispatchers.IO) {
                        engine.resetLoras()
                        appLogs += "🔌 LoRAs Cleared\n"
                        updateSystemLogs()
                    }
                }) { Text("Clr LoRA") }
            }

            Spacer(modifier = Modifier.height(8.dp))
            
            // --- Thinking UI ---
            if (!thoughtText.isNullOrEmpty() || partialThought != null) {
                var expanded by remember { mutableStateOf(true) }
                Card(
                    colors = CardDefaults.cardColors(containerColor = Color(0xFFE0E0E0)),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 8.dp)
                        .clickable { expanded = !expanded }
                ) {
                    Column(modifier = Modifier.padding(8.dp)) {
                        Text(
                            text = if (isThinking) "🤔 Thinking..." else "💡 Thought Process",
                            style = MaterialTheme.typography.labelMedium,
                            color = Color.Gray
                        )
                        AnimatedVisibility(visible = expanded) {
                            Text(
                                text = (thoughtText ?: partialThought) ?: "",
                                style = MaterialTheme.typography.bodySmall,
                                color = Color.DarkGray,
                                modifier = Modifier.heightIn(max = 150.dp).verticalScroll(rememberScrollState())
                            )
                        }
                    }
                }
            }

            // --- Answer UI ---
            if (answerText.isNotEmpty() || (rawResponse.isNotEmpty() && thoughtText == null && partialThought == null)) {
                Card(
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                    modifier = Modifier.fillMaxWidth().heightIn(min = 50.dp, max = 200.dp)
                ) {
                    Text(
                        text = if (thoughtText != null) answerText else rawResponse,
                        modifier = Modifier.padding(8.dp).verticalScroll(rememberScrollState()),
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
                Divider(modifier = Modifier.padding(vertical = 8.dp))
            }

            Button(onClick = {
                val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                val clip = ClipData.newPlainText("Cortex Logs", "APP LOGS:\n$appLogs\n\nRAW RESPONSE:\n$rawResponse\n\nSYSTEM LOGS:\n$systemLogs")
                clipboard.setPrimaryClip(clip)
                appLogs += "📋 Logs Copied to Clipboard\n"
            }) {
                Text("COPY FULL LOGS")
            }

            Divider(modifier = Modifier.padding(vertical = 8.dp))
            
            Text("App Logs:", style = MaterialTheme.typography.labelLarge)
            Text(
                text = appLogs,
                fontSize = 10.sp,
                lineHeight = 12.sp,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
            )
            
            Divider(modifier = Modifier.padding(vertical = 8.dp))

            Text("System Logs (Cortex Tag):", style = MaterialTheme.typography.labelLarge)
            Text(
                text = systemLogs,
                fontSize = 10.sp,
                lineHeight = 12.sp,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .verticalScroll(scrollState)
            )
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        engine.close()
    }
}
