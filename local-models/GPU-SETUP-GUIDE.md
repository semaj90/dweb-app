# Ollama GPU Configuration Guide for Gemma 3

This guide provides the steps to configure and run local, Unsloth-trained Gemma 3 models with full GPU acceleration using Ollama on Windows.

## 🚀 Quick Start

1. **Export Your Model from Unsloth:**
   After training with Unsloth, save your model to GGUF format:
   ```python
   # Example from your training script
   model.save_pretrained_gguf("gemma3-legal", tokenizer, quantization_method="q5_k_m")
   ```
   This creates a file like `gemma3-legal-q5_k_m.gguf`

2. **Launch Ollama:**
   ```batch
   .\launch-ollama-gpu.bat
   ```

3. **Create the Model in Ollama:**
   In a **new terminal**:
   ```bash
   ollama create gemma3-legal -f Modelfile.gemma3-legal
   ollama create gemma3-quick -f Modelfile.gemma3-quick
   ```

4. **Test GPU Acceleration:**
   ```bash
   ollama run gemma3-legal "Explain the elements of a valid contract."
   ```

---

## 📁 Directory Structure

```
C:\Users\james\Desktop\deeds-web\deeds-web-app\local-models\
├── Modelfile.gemma3-legal       # Modelfile for detailed legal analysis
├── Modelfile.gemma3-quick       # Modelfile for quick responses
├── launch-ollama-gpu.bat        # GPU-enabled launcher
├── setup-ollama-models.ps1      # PowerShell setup script
├── test-gpu-acceleration.ps1    # GPU testing script
├── gemma3-legal-q5_k_m.gguf     # Model weights from Unsloth (you need to add)
└── gemma3-quick-q5_k_m.gguf     # Quick model weights (you need to add)
```

## 🎮 GPU Acceleration Setup

GPU acceleration is controlled through the **Modelfile**. This is the modern and recommended approach.

### Modelfile Parameters

The key parameters in your Modelfile:

```modelfile
# Reference your GGUF file
FROM ./gemma3-legal-q5_k_m.gguf

# GPU parameters
PARAMETER num_gpu -1       # Use -1 to offload ALL layers to GPU
PARAMETER main_gpu 0       # Primary GPU (0 for single GPU)
PARAMETER num_ctx 8192     # Context window (VRAM dependent)
PARAMETER num_thread 8     # CPU threads for non-GPU ops
```

### Environment Variables

Only one environment variable is needed:
- `CUDA_VISIBLE_DEVICES=0` - Isolates GPU 0 for Ollama

## 🧪 Verify GPU Usage

1. **Run a Model:**
   ```bash
   ollama run gemma3-legal "What is promissory estoppel?"
   ```

2. **Monitor GPU (in another terminal):**
   ```bash
   nvidia-smi -l 1
   ```
   
   You should see:
   - GPU-Util: 20-90%
   - Memory Usage: 5-9GB for a 9B model

### Expected Performance

| Configuration | Speed | GPU Memory |
|--------------|-------|------------|
| With GPU | 30-100+ tokens/sec | 5-9GB |
| CPU Only | 1-10 tokens/sec | 0GB |

## 🔧 Troubleshooting

### Model Not Using GPU

1. **Check Modelfile:**
   Ensure `PARAMETER num_gpu -1` is present

2. **Recreate Model:**
   ```bash
   ollama rm gemma3-legal
   ollama create gemma3-legal -f Modelfile.gemma3-legal
   ```

3. **Check Drivers:**
   ```bash
   nvidia-smi
   ```

4. **Check Ollama Logs:**
   ```powershell
   Get-Content $env:LOCALAPPDATA\Ollama\logs\server.log -Tail 50
   ```
   Look for "cuda" or "gpu" mentions

### Low GPU Usage

- First inference is slower (model loading)
- Larger prompts = better GPU utilization
- Ensure you're using q4/q5/q6 quantization (not q2/q3)

## 🌐 Web App Integration

Your web app backend can query the model via HTTP:

```javascript
// Example API call
const response = await fetch('http://127.0.0.1:11434/api/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'gemma3-legal',
    prompt: 'Explain strict liability in tort law.',
    stream: false
  })
});

const data = await response.json();
console.log(data.response);
```

## 📝 Creating GGUF Files from Unsloth

If you haven't exported your models yet:

```python
from unsloth import FastLanguageModel

# Load your fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="your_model_path",
    max_seq_length=8192,
    dtype=None,
    load_in_4bit=True,
)

# Export to GGUF with q5_K_M quantization (recommended)
model.save_pretrained_gguf(
    "gemma3-legal",
    tokenizer,
    quantization_method="q5_K_M"  # Good balance of quality/size
)
```

This creates `gemma3-legal-q5_K_M.gguf` in your current directory.

## ✅ Complete Setup Checklist

- [ ] NVIDIA GPU with updated drivers
- [ ] `nvidia-smi` command works
- [ ] GGUF files exported from Unsloth
- [ ] Modelfiles reference correct GGUF paths
- [ ] Ollama server running via launcher
- [ ] Models created with `ollama create`
- [ ] GPU usage visible during inference

---

**Note**: Always use `PARAMETER num_gpu -1` in your Modelfile for full GPU acceleration. The old environment variable approach (`OLLAMA_GPU_LAYERS`) is deprecated.
