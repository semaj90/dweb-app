"""
RAG Service for Legal Document Retrieval and Generation
Integrates local GGUF models with pgvector semantic search
"""

import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger
import time
import re
from .model_manager import LocalModelManager
from .vector_store import VectorStoreService

class RAGService:
    """Advanced RAG system for legal document analysis"""
    
    def __init__(self, model_manager: LocalModelManager, vector_store: VectorStoreService):
        self.model_manager = model_manager
        self.vector_store = vector_store
        self.legal_context_templates = {
            "contract_analysis": """You are analyzing a legal contract. Focus on:
- Liability clauses and risk allocation
- Termination conditions and notice requirements
- Payment terms and dispute resolution
- Compliance with applicable laws""",
            
            "case_research": """You are researching legal cases. Focus on:
- Relevant legal precedents and citations
- Key holdings and legal principles
- Jurisdictional considerations
- Applicable statutes and regulations""",
            
            "compliance_review": """You are reviewing for compliance. Focus on:
- Regulatory requirements and standards
- Potential violations and remediation
- Risk mitigation strategies
- Documentation and reporting obligations"""
        }
    
    async def query(
        self, 
        query: str, 
        max_results: int = 5,
        confidence_threshold: float = 0.7,
        case_id: Optional[str] = None,
        document_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform RAG query with semantic search and legal analysis
        """
        try:
            start_time = time.time()
            
            # Step 1: Generate query embedding
            logger.info(f"🔍 Processing RAG query: {query[:100]}...")
            query_embedding = await self.model_manager.generate_embeddings([query])
            
            # Step 2: Semantic search in vector store
            search_results = await self.vector_store.semantic_search(
                query_embedding=query_embedding[0],
                limit=max_results * 2,  # Get more results for better context
                confidence_threshold=confidence_threshold,
                case_id=case_id,
                document_types=document_types
            )
            
            # Step 3: Rerank and filter results
            relevant_documents = await self._rerank_documents(query, search_results)
            relevant_documents = relevant_documents[:max_results]
            
            # Step 4: Determine query type and context
            query_context = await self._determine_query_context(query)
            
            # Step 5: Generate RAG response
            rag_response = await self._generate_rag_response(
                query=query,
                documents=relevant_documents,
                context_type=query_context
            )
            
            # Step 6: Calculate confidence score
            confidence_score = await self._calculate_confidence(
                query, 
                relevant_documents, 
                rag_response
            )
            
            processing_time = time.time() - start_time
            
            result = {
                "response": rag_response,
                "sources": [
                    {
                        "document_id": doc["document_id"],
                        "title": doc["title"],
                        "document_type": doc["document_type"],
                        "similarity_score": doc["similarity_score"],
                        "excerpt": doc["content"][:300] + "..." if len(doc["content"]) > 300 else doc["content"]
                    }
                    for doc in relevant_documents
                ],
                "confidence": confidence_score,
                "processing_time": processing_time,
                "query_context": query_context,
                "documents_searched": len(search_results)
            }
            
            logger.info(f"✅ RAG query completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ RAG query failed: {e}")
            raise
    
    async def _rerank_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Rerank documents based on legal relevance"""
        try:
            # Legal keywords that boost relevance
            legal_keywords = [
                "contract", "agreement", "clause", "liability", "damages",
                "court", "judgment", "statute", "regulation", "compliance",
                "defendant", "plaintiff", "attorney", "counsel", "precedent"
            ]
            
            query_lower = query.lower()
            
            for doc in documents:
                content_lower = doc["content"].lower()
                
                # Base similarity score
                relevance_score = doc["similarity_score"]
                
                # Boost for legal keyword matches
                keyword_matches = sum(1 for keyword in legal_keywords if keyword in content_lower)
                relevance_score += keyword_matches * 0.05
                
                # Boost for exact phrase matches
                if query_lower in content_lower:
                    relevance_score += 0.15
                
                # Boost for document type relevance
                if doc.get("document_type") in ["contract", "case_law", "statute", "regulation"]:
                    relevance_score += 0.1
                
                # Update the score
                doc["relevance_score"] = min(relevance_score, 1.0)
            
            # Sort by relevance score
            documents.sort(key=lambda x: x.get("relevance_score", x["similarity_score"]), reverse=True)
            
            return documents
            
        except Exception as e:
            logger.error(f"❌ Document reranking failed: {e}")
            return documents
    
    async def _determine_query_context(self, query: str) -> str:
        """Determine the type of legal query"""
        try:
            query_lower = query.lower()
            
            # Contract-related queries
            if any(word in query_lower for word in ["contract", "agreement", "clause", "terms", "liability"]):
                return "contract_analysis"
            
            # Case research queries
            elif any(word in query_lower for word in ["case", "precedent", "court", "judgment", "ruling"]):
                return "case_research"
            
            # Compliance queries
            elif any(word in query_lower for word in ["compliance", "regulation", "law", "requirement", "violation"]):
                return "compliance_review"
            
            # Default to general legal analysis
            else:
                return "general_legal"
                
        except Exception as e:
            logger.error(f"❌ Query context determination failed: {e}")
            return "general_legal"
    
    async def _generate_rag_response(
        self, 
        query: str, 
        documents: List[Dict], 
        context_type: str
    ) -> str:
        """Generate RAG response using retrieved documents"""
        try:
            if not documents:
                return "I couldn't find relevant legal documents to answer your query. Please try rephrasing your question or check if the relevant documents have been uploaded to the system."
            
            # Get context template
            context_instruction = self.legal_context_templates.get(
                context_type, 
                "You are a legal AI assistant. Provide accurate, well-reasoned legal analysis."
            )
            
            # Prepare document context
            document_context = ""
            for i, doc in enumerate(documents, 1):
                document_context += f"\n--- Document {i}: {doc['title']} ({doc['document_type']}) ---\n"
                document_context += doc["content"][:1000]  # Limit context length
                if len(doc["content"]) > 1000:
                    document_context += "...\n"
                document_context += "\n"
            
            # Create comprehensive prompt
            rag_prompt = f"""{context_instruction}

Based on the following legal documents, please answer this question: {query}

Relevant Documents:
{document_context}

Instructions:
1. Provide a comprehensive answer based on the retrieved documents
2. Cite specific documents when referencing information
3. If information is insufficient, clearly state limitations
4. Focus on legal accuracy and practical implications
5. Use proper legal terminology and structure

Answer:"""
            
            # Generate response using local GGUF model
            response = await self.model_manager.generate_text(
                rag_prompt,
                max_tokens=800,
                temperature=0.4  # Balanced temperature for legal accuracy
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"❌ RAG response generation failed: {e}")
            return f"I encountered an error while analyzing the legal documents: {str(e)}"
    
    async def _calculate_confidence(
        self, 
        query: str, 
        documents: List[Dict], 
        response: str
    ) -> float:
        """Calculate confidence score for the RAG response"""
        try:
            confidence = 0.0
            
            # Base confidence from document relevance
            if documents:
                avg_similarity = sum(doc["similarity_score"] for doc in documents) / len(documents)
                confidence += avg_similarity * 0.6
            
            # Boost for multiple supporting documents
            doc_count_factor = min(len(documents) / 5.0, 1.0)
            confidence += doc_count_factor * 0.2
            
            # Boost for legal-specific content
            legal_indicators = ["pursuant to", "in accordance with", "subject to", "whereas", "therefore"]
            legal_score = sum(1 for indicator in legal_indicators if indicator.lower() in response.lower())
            confidence += min(legal_score * 0.05, 0.15)
            
            # Penalize if response is too short or generic
            if len(response) < 100:
                confidence *= 0.7
            
            # Ensure confidence is between 0 and 1
            confidence = max(0.1, min(confidence, 1.0))
            
            return round(confidence, 2)
            
        except Exception as e:
            logger.error(f"❌ Confidence calculation failed: {e}")
            return 0.5  # Default moderate confidence
    
    async def analyze_document_relevance(self, query: str, document_id: str) -> Dict[str, Any]:
        """Analyze how relevant a specific document is to a query"""
        try:
            # Get document content
            document = await self.vector_store.get_document(document_id)
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            # Generate embeddings
            query_embedding = await self.model_manager.generate_embeddings([query])
            doc_embedding = document.get("embedding")
            
            if not doc_embedding:
                # Generate embedding if not exists
                doc_embedding = await self.model_manager.generate_embeddings([document["content"]])
                doc_embedding = doc_embedding[0]
            
            # Calculate similarity
            similarity = await self.vector_store.calculate_similarity(
                query_embedding[0], 
                doc_embedding
            )
            
            # Extract key sections relevant to query
            key_sections = await self._extract_relevant_sections(
                query, 
                document["content"]
            )
            
            return {
                "document_id": document_id,
                "similarity_score": similarity,
                "relevance_level": self._categorize_relevance(similarity),
                "key_sections": key_sections,
                "document_title": document.get("title", "Unknown"),
                "document_type": document.get("document_type", "Unknown")
            }
            
        except Exception as e:
            logger.error(f"❌ Document relevance analysis failed: {e}")
            raise
    
    def _categorize_relevance(self, similarity_score: float) -> str:
        """Categorize relevance based on similarity score"""
        if similarity_score >= 0.85:
            return "very_high"
        elif similarity_score >= 0.75:
            return "high"
        elif similarity_score >= 0.65:
            return "medium"
        elif similarity_score >= 0.5:
            return "low"
        else:
            return "very_low"
    
    async def _extract_relevant_sections(self, query: str, content: str) -> List[Dict[str, Any]]:
        """Extract sections of document most relevant to query"""
        try:
            # Split content into sections (paragraphs)
            sections = [s.strip() for s in content.split('\n\n') if len(s.strip()) > 50]
            
            if not sections:
                return []
            
            # Generate embeddings for sections
            section_embeddings = await self.model_manager.generate_embeddings(sections)
            query_embedding = await self.model_manager.generate_embeddings([query])
            
            # Calculate relevance for each section
            relevant_sections = []
            for i, (section, embedding) in enumerate(zip(sections, section_embeddings)):
                similarity = await self.vector_store.calculate_similarity(
                    query_embedding[0], 
                    embedding
                )
                
                if similarity > 0.6:  # Only include reasonably relevant sections
                    relevant_sections.append({
                        "section_index": i,
                        "content": section[:500] + "..." if len(section) > 500 else section,
                        "similarity_score": similarity
                    })
            
            # Sort by relevance and return top 3
            relevant_sections.sort(key=lambda x: x["similarity_score"], reverse=True)
            return relevant_sections[:3]
            
        except Exception as e:
            logger.error(f"❌ Section extraction failed: {e}")
            return []
    
    async def generate_legal_summary(self, document_ids: List[str]) -> Dict[str, Any]:
        """Generate a comprehensive legal summary from multiple documents"""
        try:
            # Retrieve documents
            documents = []
            for doc_id in document_ids:
                doc = await self.vector_store.get_document(doc_id)
                if doc:
                    documents.append(doc)
            
            if not documents:
                raise ValueError("No documents found for summary generation")
            
            # Create summary prompt
            document_summaries = []
            for doc in documents:
                doc_summary = f"Document: {doc.get('title', 'Unknown')}\n"
                doc_summary += f"Type: {doc.get('document_type', 'Unknown')}\n"
                doc_summary += f"Content: {doc['content'][:800]}...\n"
                document_summaries.append(doc_summary)
            
            summary_prompt = f"""
Generate a comprehensive legal summary of the following documents:

{chr(10).join(document_summaries)}

Provide:
1. Executive Summary
2. Key Legal Issues Identified
3. Risk Assessment
4. Recommendations
5. Next Steps

Format as a structured legal memorandum:
"""
            
            summary = await self.model_manager.generate_text(
                summary_prompt,
                max_tokens=1200,
                temperature=0.3
            )
            
            return {
                "summary": summary,
                "documents_analyzed": len(documents),
                "document_types": list(set(doc.get("document_type", "Unknown") for doc in documents)),
                "generated_at": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Legal summary generation failed: {e}")
            raise
