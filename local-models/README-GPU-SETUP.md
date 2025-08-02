# 🚀 OLLAMA GPU ACCELERATION - COMPLETE SETUP

## ✅ QUICK FIX - RUN THIS NOW:

```batch
RUN-GPU-SETUP.bat
```

This will:
1. Check your GPU
2. Configure GPU acceleration 
3. Start Ollama with GPU support
4. Create/update your models
5. Test GPU acceleration

## 📁 Files Created:

- **RUN-GPU-SETUP.bat** - One-click setup (RUN THIS FIRST!)
- **setup-and-test-gpu.ps1** - Complete PowerShell setup script
- **fix-ollama-gpu.ps1** - GPU configuration fix
- **test-gpu.ps1** - GPU acceleration tester
- **launch-ollama-gpu.bat** - GPU-enabled launcher
- **Modelfile.gemma3-quick** - Quick response model config
- **GPU-SETUP-GUIDE.md** - Detailed documentation

## 🎮 Verify GPU is Working:

1. **During inference, open another terminal:**
   ```bash
   nvidia-smi -l 1
   ```
   You should see GPU usage spike to 20-90%

2. **Check response speed:**
   - With GPU: 20-100+ tokens/second
   - Without GPU: 1-10 tokens/second

## 🧪 Test Your Models:

```bash
# Legal analysis (detailed)
ollama run gemma3-legal "Explain the elements of a valid contract"

# Quick legal lookup
ollama run gemma3-quick "Define negligence"

# Check model list
ollama list
```

## 🔧 Troubleshooting:

### If GPU not working:
1. Check NVIDIA drivers: `nvidia-smi`
2. Restart Ollama: Run `RUN-GPU-SETUP.bat` again
3. Check logs: `%LOCALAPPDATA%\Ollama\logs\server.log`

### If models missing:
```bash
ollama create gemma3-legal -f Modelfile.gemma3-legal
ollama create gemma3-quick -f Modelfile.gemma3-quick
```

### Environment Variables Set:
- OLLAMA_GPU_DRIVER=cuda
- CUDA_VISIBLE_DEVICES=0
- OLLAMA_NUM_GPU=1  
- OLLAMA_GPU_LAYERS=999

## ✨ Success Indicators:

- ✅ nvidia-smi shows GPU usage during queries
- ✅ Models respond in < 5 seconds
- ✅ GPU memory usage increases (4-8GB)
- ✅ test-gpu.ps1 shows > 20% GPU usage

---
**IMPORTANT**: First run may be slower as models load into GPU memory. Subsequent runs will be much faster!
