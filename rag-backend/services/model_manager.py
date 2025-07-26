"""
Local Model Manager for GGUF models
Supports Unsloth-trained models and integrates with Ollama
"""

import os
import asyncio
import json
import httpx
from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger
import torch
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

class LocalModelManager:
    """Manages local GGUF models and embeddings"""
    
    def __init__(self):
        self.models_dir = Path("/app/models")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.loaded_models = {}
        self.embedding_model = None
        self.llama_model = None
        
    async def initialize(self):
        """Initialize the model manager"""
        try:
            # Create models directory if it doesn't exist
            self.models_dir.mkdir(exist_ok=True)
            
            # Initialize embedding model
            await self._initialize_embedding_model()
            
            # Scan for local GGUF models
            await self._scan_local_models()
            
            # Initialize Ollama connection
            await self._initialize_ollama()
            
            logger.info("✅ Model manager initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize model manager: {e}")
            raise
    
    async def _initialize_embedding_model(self):
        """Initialize sentence transformer for embeddings"""
        try:
            # Use a legal-domain optimized embedding model if available
            model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight and fast
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"✅ Embedding model loaded: {model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    async def _scan_local_models(self):
        """Scan for local GGUF models"""
        try:
            gguf_files = list(self.models_dir.glob("*.gguf"))
            
            for gguf_file in gguf_files:
                model_info = {
                    "path": str(gguf_file),
                    "name": gguf_file.stem,
                    "size": gguf_file.stat().st_size,
                    "type": "gguf",
                    "loaded": False
                }
                self.loaded_models[gguf_file.stem] = model_info
                logger.info(f"📁 Found GGUF model: {gguf_file.name}")
            
            if not gguf_files:
                logger.warning("⚠️  No GGUF models found in models directory")
                
        except Exception as e:
            logger.error(f"❌ Failed to scan local models: {e}")
    
    async def _initialize_ollama(self):
        """Initialize connection to Ollama"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.ollama_url}/api/version")
                if response.status_code == 200:
                    version_info = response.json()
                    logger.info(f"✅ Ollama connected: {version_info}")
                else:
                    logger.warning("⚠️  Ollama not available, using local GGUF only")
        except Exception as e:
            logger.warning(f"⚠️  Ollama connection failed: {e}")
    
    async def load_local_model(self, model_path: str) -> Dict[str, Any]:
        """Load a local GGUF model"""
        try:
            model_file = Path(model_path)
            
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            if not model_file.suffix == ".gguf":
                raise ValueError("Only GGUF models are supported")
            
            # Load model with llama-cpp-python
            logger.info(f"🔄 Loading GGUF model: {model_file.name}")
            
            # Configure model parameters for legal use case
            self.llama_model = Llama(
                model_path=str(model_file),
                n_ctx=4096,  # Context length
                n_batch=512,  # Batch size
                n_gpu_layers=35 if torch.cuda.is_available() else 0,  # GPU acceleration
                verbose=False
            )
            
            # Update model info
            model_name = model_file.stem
            if model_name in self.loaded_models:
                self.loaded_models[model_name]["loaded"] = True
                self.loaded_models[model_name]["context_length"] = 4096
            
            logger.info(f"✅ GGUF model loaded successfully: {model_name}")
            
            return {
                "model_name": model_name,
                "model_path": str(model_file),
                "context_length": 4096,
                "gpu_layers": 35 if torch.cuda.is_available() else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to load GGUF model: {e}")
            raise
    
    async def generate_text(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate text using loaded GGUF model"""
        try:
            if not self.llama_model:
                # Try to use Ollama as fallback
                return await self._generate_with_ollama(prompt, max_tokens, temperature)
            
            # Create legal-optimized prompt template
            legal_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a specialized legal AI assistant. Analyze legal documents, contracts, and case law with precision. Focus on:
- Legal compliance and regulatory requirements
- Risk assessment and mitigation
- Contract clause analysis
- Legal precedent identification
- Accurate legal terminology

<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            
            # Generate response
            response = self.llama_model(
                legal_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|eot_id|>", "<|end_of_text|>"],
                echo=False
            )
            
            generated_text = response["choices"][0]["text"].strip()
            return generated_text
            
        except Exception as e:
            logger.error(f"❌ Text generation failed: {e}")
            raise
    
    async def _generate_with_ollama(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Fallback to Ollama for text generation"""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {
                    "model": "gemma2:9b",  # Fallback model
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    }
                }
                
                response = await client.post(f"{self.ollama_url}/api/generate", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "")
                else:
                    raise Exception(f"Ollama API error: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"❌ Ollama generation failed: {e}")
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks"""
        try:
            if not self.embedding_model:
                raise Exception("Embedding model not initialized")
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            raise
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        try:
            status = {
                "embedding_model_loaded": self.embedding_model is not None,
                "llama_model_loaded": self.llama_model is not None,
                "available_models": len(self.loaded_models),
                "models": self.loaded_models,
                "gpu_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
            
            # Check Ollama status
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{self.ollama_url}/api/version")
                    status["ollama_available"] = response.status_code == 200
            except:
                status["ollama_available"] = False
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Failed to get model status: {e}")
            return {"error": str(e)}
    
    async def analyze_legal_document(self, text: str) -> Dict[str, Any]:
        """Analyze legal document with specialized prompts"""
        try:
            # Legal analysis prompt
            analysis_prompt = f"""
Analyze this legal document and provide:

1. Document Type Classification
2. Key Legal Entities (parties, dates, amounts, references)
3. Risk Assessment (potential issues, compliance concerns)
4. Key Clauses and Terms
5. Recommendations

Document Text:
{text[:2000]}...

Provide structured analysis:
"""
            
            response = await self.generate_text(
                analysis_prompt, 
                max_tokens=800, 
                temperature=0.3  # Lower temperature for more consistent legal analysis
            )
            
            return {
                "analysis": response,
                "document_length": len(text),
                "model_used": "local_gguf" if self.llama_model else "ollama_fallback"
            }
            
        except Exception as e:
            logger.error(f"❌ Legal document analysis failed: {e}")
            raise
    
    async def extract_legal_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract legal entities from text"""
        try:
            entity_prompt = f"""
Extract legal entities from this text. Return a JSON list with:
- entity_type (CASE_NUMBER, COURT, JUDGE, ATTORNEY, DATE, AMOUNT, STATUTE, PARTY)
- entity_value (the actual text)
- confidence (0.0 to 1.0)

Text: {text[:1500]}

JSON format only:
"""
            
            response = await self.generate_text(entity_prompt, max_tokens=400, temperature=0.1)
            
            # Try to parse JSON response
            try:
                import json
                entities = json.loads(response)
                return entities if isinstance(entities, list) else []
            except:
                # Fallback to simple parsing if JSON fails
                return [{"entity_type": "ANALYSIS", "entity_value": response, "confidence": 0.8}]
                
        except Exception as e:
            logger.error(f"❌ Entity extraction failed: {e}")
            return []
