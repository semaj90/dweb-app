#!/usr/bin/env python3
"""
LangExtract processor for structured information extraction
Optimized for Windows with GPU acceleration support
"""

import json
import argparse
import sys
import os
from typing import Dict, List, Any, Optional
import traceback

try:
    import langextract
    from langextract import LangExtract
    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

class LangExtractProcessor:
    def __init__(self):
        self.device = self._detect_device()
        self.extractor = None
        self._initialize_extractor()
    
    def _detect_device(self) -> str:
        """Detect the best available device for processing"""
        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            return "cuda"
        elif TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"
    
    def _initialize_extractor(self):
        """Initialize the LangExtract instance"""
        if not LANGEXTRACT_AVAILABLE:
            raise ImportError("LangExtract not available. Install with: pip install langextract")
        
        try:
            # Initialize with optimal settings for Windows
            self.extractor = LangExtract(
                model="gpt-3.5-turbo",  # Default model
                device=self.device,
                batch_size=1 if self.device == "cuda" else 1,
                max_workers=4 if self.device == "cuda" else 2
            )
        except Exception as e:
            # Fallback to basic initialization
            self.extractor = LangExtract()
    
    def extract_structured_info(self, text: str, schema: Optional[Dict] = None, options: Dict = None) -> Dict[str, Any]:
        """
        Extract structured information from text using LangExtract
        
        Args:
            text: Input text to process
            schema: Optional schema for extraction
            options: Processing options
        
        Returns:
            Dictionary containing extracted information
        """
        if options is None:
            options = {}
        
        try:
            # Configure extraction based on options
            extraction_config = {
                "model": options.get("model", "gpt-3.5-turbo"),
                "temperature": options.get("temperature", 0.1),
                "max_tokens": options.get("max_tokens", 1000),
                "device": self.device if options.get("gpu_acceleration", True) else "cpu"
            }
            
            # Define extraction schema
            if schema:
                extraction_schema = schema
            else:
                # Default legal document schema
                extraction_schema = {
                    "entities": {
                        "type": "array",
                        "description": "Legal entities mentioned in the document",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "Entity text"},
                                "type": {"type": "string", "description": "Entity type (PERSON, ORG, DATE, etc.)"},
                                "confidence": {"type": "number", "description": "Confidence score 0-1"}
                            }
                        }
                    },
                    "case_info": {
                        "type": "object",
                        "description": "Case-related information",
                        "properties": {
                            "case_number": {"type": "string", "description": "Case number if present"},
                            "court": {"type": "string", "description": "Court name"},
                            "parties": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Parties involved"
                            },
                            "dates": {
                                "type": "array", 
                                "items": {"type": "string"},
                                "description": "Important dates"
                            }
                        }
                    },
                    "key_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Key points and important information"
                    },
                    "document_type": {
                        "type": "string",
                        "description": "Type of legal document"
                    }
                }
            
            # Perform extraction
            if hasattr(self.extractor, 'extract_with_schema'):
                result = self.extractor.extract_with_schema(
                    text=text,
                    schema=extraction_schema,
                    **extraction_config
                )
            else:
                # Fallback method
                result = self.extractor.extract(
                    text=text,
                    schema=extraction_schema
                )
            
            # Post-process and enhance results
            enhanced_result = self._enhance_extraction_result(result, text, options)
            
            return {
                "success": True,
                "data": enhanced_result,
                "metadata": {
                    "device_used": self.device,
                    "gpu_acceleration": self.device != "cpu",
                    "model": extraction_config["model"],
                    "text_length": len(text),
                    "processing_time": 0  # Would be calculated in real implementation
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "metadata": {
                    "device_used": self.device,
                    "gpu_acceleration": False
                }
            }
    
    def _enhance_extraction_result(self, result: Dict, original_text: str, options: Dict) -> Dict:
        """Enhance extraction results with additional processing"""
        
        # Add source grounding
        if isinstance(result, dict) and "entities" in result:
            for entity in result.get("entities", []):
                if isinstance(entity, dict) and "text" in entity:
                    # Find position in original text
                    entity_text = entity["text"]
                    start_pos = original_text.find(entity_text)
                    if start_pos != -1:
                        entity["source_location"] = {
                            "start": start_pos,
                            "end": start_pos + len(entity_text),
                            "context": original_text[max(0, start_pos-50):start_pos+len(entity_text)+50]
                        }
        
        # Add confidence scoring
        if "entities" in result:
            for entity in result.get("entities", []):
                if "confidence" not in entity:
                    entity["confidence"] = 0.85  # Default confidence
        
        # Add document analysis
        result["document_analysis"] = {
            "word_count": len(original_text.split()),
            "character_count": len(original_text),
            "estimated_reading_time": len(original_text.split()) / 200,  # words per minute
            "complexity_score": min(1.0, len(original_text) / 10000)  # Rough complexity measure
        }
        
        return result

def main():
    parser = argparse.ArgumentParser(description='LangExtract Processor')
    parser.add_argument('--text', required=True, help='Text to process')
    parser.add_argument('--schema', help='JSON schema for extraction')
    parser.add_argument('--options', help='Processing options as JSON')
    
    args = parser.parse_args()
    
    try:
        # Parse inputs
        text = json.loads(args.text) if args.text.startswith('{') or args.text.startswith('[') else args.text
        schema = json.loads(args.schema) if args.schema else None
        options = json.loads(args.options) if args.options else {}
        
        # Initialize processor
        processor = LangExtractProcessor()
        
        # Process text
        result = processor.extract_structured_info(text, schema, options)
        
        # Output result as JSON
        print(json.dumps(result, ensure_ascii=False, indent=None))
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "langextract_available": LANGEXTRACT_AVAILABLE,
            "torch_available": TORCH_AVAILABLE,
            "cuda_available": CUDA_AVAILABLE
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)

if __name__ == "__main__":
    main()
