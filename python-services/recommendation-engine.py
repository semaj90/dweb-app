# recommendation-engine.py
# Production recommendation engine with RL self-prompting for legal AI platform
# Install: pip install fastapi uvicorn redis psycopg2-binary numpy scikit-learn sentence-transformers requests

import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import time
import uuid
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://legal_admin:123456@localhost:5432/legal_ai_db")
EMBED_SERVICE_URL = os.getenv("EMBED_SERVICE_URL", "http://localhost:9001")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# RL Parameters
EXPLORATION_RATE = float(os.getenv("EXPLORATION_RATE", "0.2"))  # Epsilon for epsilon-greedy
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.1"))
REWARD_DECAY = float(os.getenv("REWARD_DECAY", "0.95"))
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "10"))  # Recent interactions to consider

# Recommendation Parameters
MAX_RECOMMENDATIONS = int(os.getenv("MAX_RECOMMENDATIONS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
TYPING_DELAY_MS = int(os.getenv("TYPING_DELAY_MS", "2000"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models
class RecommendationRequest(BaseModel):
    user_id: str
    case_id: Optional[str] = None
    current_text: str = Field(default="")
    context: Dict[str, Any] = Field(default_factory=dict)
    is_typing: bool = Field(default=False)
    interaction_type: str = Field(default="query")  # query, draft, analysis, search

class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    user_intent: Dict[str, Any]
    self_prompts: List[str]
    confidence_scores: List[float]
    processing_time_ms: int

class FeedbackRequest(BaseModel):
    user_id: str
    recommendation_id: str
    action: str  # "accepted", "rejected", "clicked", "ignored"
    context: Dict[str, Any] = Field(default_factory=dict)

class UserAnalytics(BaseModel):
    user_id: str
    total_interactions: int
    accepted_recommendations: int
    acceptance_rate: float
    common_intents: List[str]
    productivity_score: float

# Recommendation Types
RECOMMENDATION_TYPES = {
    "did_you_mean": "Suggest text corrections or clarifications",
    "auto_complete": "Complete the current text based on legal context",
    "related_cases": "Find similar cases or precedents",
    "suggested_actions": "Propose next steps in the workflow",
    "evidence_tags": "Suggest tags for evidence items",
    "legal_research": "Recommend legal research directions",
    "document_analysis": "Suggest document analysis approaches"
}

class RecommendationEngine:
    def __init__(self):
        self.redis_client = None
        self.embedding_model = None
        self.user_profiles = {}  # Simple in-memory user profiles
        self.action_values = defaultdict(lambda: defaultdict(float))  # Q-learning values
        self.user_interactions = defaultdict(lambda: deque(maxlen=CONTEXT_WINDOW))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.app = self.create_app()
        
    def create_app(self) -> FastAPI:
        app = FastAPI(
            title="Legal AI Recommendation Engine",
            description="AI-powered recommendation engine with reinforcement learning",
            version="1.0.0"
        )
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @app.on_event("startup")
        async def startup_event():
            await self.initialize()
            
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "redis": self.redis_client is not None,
                    "embedding_model": self.embedding_model is not None
                }
            }
            
        @app.post("/recommend", response_model=RecommendationResponse)
        async def get_recommendations(request: RecommendationRequest, background_tasks: BackgroundTasks):
            return await self.process_recommendation_request(request, background_tasks)
            
        @app.post("/feedback")
        async def submit_feedback(request: FeedbackRequest):
            return await self.process_user_feedback(request)
            
        @app.get("/analytics/{user_id}", response_model=UserAnalytics)
        async def get_user_analytics(user_id: str):
            return await self.get_user_analytics(user_id)
            
        @app.post("/self-prompt")
        async def generate_self_prompt(query: str, context: Dict[str, Any] = None):
            return await self.generate_self_prompts(query, context or {})
            
        @app.get("/user-intent/{user_id}")
        async def analyze_user_intent(user_id: str):
            return await self.analyze_user_intent(user_id)
            
        return app
    
    async def initialize(self):
        """Initialize the recommendation engine"""
        try:
            # Initialize Redis
            self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            await asyncio.get_event_loop().run_in_executor(None, self.redis_client.ping)
            logger.info("✅ Connected to Redis")
            
            # Initialize embedding model (lightweight for fast inference)
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Loaded embedding model")
            
            # Load user profiles and Q-values from Redis
            await self.load_user_data()
            
            logger.info("🚀 Recommendation Engine initialized")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
            
    async def load_user_data(self):
        """Load user profiles and learning data from Redis"""
        try:
            # Load user profiles
            profile_keys = self.redis_client.keys("user:profile:*")
            for key in profile_keys:
                user_id = key.split(":")[-1]
                profile_data = self.redis_client.hgetall(key)
                self.user_profiles[user_id] = profile_data
                
            # Load Q-learning values
            q_keys = self.redis_client.keys("rl:qvalues:*")
            for key in q_keys:
                user_id = key.split(":")[-1]
                q_data = self.redis_client.hgetall(key)
                for state_action, value in q_data.items():
                    state, action = state_action.split("|")
                    self.action_values[user_id][f"{state}:{action}"] = float(value)
                    
            logger.info(f"Loaded data for {len(self.user_profiles)} users")
            
        except Exception as e:
            logger.error(f"Failed to load user data: {e}")
            
    async def process_recommendation_request(self, request: RecommendationRequest, 
                                           background_tasks: BackgroundTasks) -> RecommendationResponse:
        """Process recommendation request with RL-based selection"""
        start_time = time.time()
        
        try:
            # Skip recommendations if user is actively typing
            if request.is_typing:
                return RecommendationResponse(
                    recommendations=[],
                    user_intent={"status": "typing", "confidence": 0.0},
                    self_prompts=[],
                    confidence_scores=[],
                    processing_time_ms=int((time.time() - start_time) * 1000)
                )
            
            # Analyze user intent
            user_intent = await self.analyze_user_intent_from_text(
                request.user_id, request.current_text, request.context
            )
            
            # Generate candidate recommendations
            candidates = await self.generate_candidate_recommendations(request, user_intent)
            
            # Apply RL-based selection
            selected_recs = await self.select_recommendations_with_rl(
                request.user_id, candidates, user_intent
            )
            
            # Generate self-prompts
            self_prompts = await self.generate_self_prompts(request.current_text, request.context)
            
            # Store interaction for future learning
            background_tasks.add_task(
                self.store_interaction,
                request.user_id,
                request.current_text,
                user_intent,
                selected_recs
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return RecommendationResponse(
                recommendations=selected_recs,
                user_intent=user_intent,
                self_prompts=self_prompts,
                confidence_scores=[rec.get("confidence", 0.0) for rec in selected_recs],
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Recommendation processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    async def analyze_user_intent_from_text(self, user_id: str, text: str, 
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user intent from current text and context"""
        if not text.strip():
            return {"type": "idle", "confidence": 1.0, "details": {}}
            
        # Get text embedding
        embedding = self.embedding_model.encode([text])[0]
        
        # Simple intent classification based on keywords and patterns
        intent_patterns = {
            "question": ["what", "how", "why", "when", "where", "?"],
            "search": ["find", "search", "look for", "locate"],
            "analysis": ["analyze", "review", "examine", "assess"],
            "draft": ["write", "draft", "compose", "create"],
            "research": ["research", "investigate", "study", "precedent"],
            "comparison": ["compare", "versus", "vs", "difference", "similar"],
            "summary": ["summarize", "summary", "overview", "brief"]
        }
        
        text_lower = text.lower()
        intent_scores = {}
        
        for intent_type, keywords in intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                intent_scores[intent_type] = score / len(keywords)
                
        # Determine primary intent
        if intent_scores:
            primary_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[primary_intent]
        else:
            primary_intent = "general"
            confidence = 0.5
            
        return {
            "type": primary_intent,
            "confidence": confidence,
            "details": intent_scores,
            "text_length": len(text),
            "context": context.get("current_page", "unknown")
        }
        
    async def generate_candidate_recommendations(self, request: RecommendationRequest, 
                                               user_intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate candidate recommendations based on user intent and context"""
        candidates = []
        
        try:
            # Text completion/correction recommendations
            if len(request.current_text) > 10:
                completion_candidates = await self.generate_text_completions(
                    request.current_text, request.context
                )
                candidates.extend(completion_candidates)
                
            # Semantic search recommendations
            if user_intent["type"] in ["search", "research", "question"]:
                search_candidates = await self.generate_search_recommendations(
                    request.current_text, request.case_id
                )
                candidates.extend(search_candidates)
                
            # Action-based recommendations
            workflow_candidates = await self.generate_workflow_recommendations(
                request.user_id, request.context, user_intent
            )
            candidates.extend(workflow_candidates)
            
            # Similar case recommendations
            if request.case_id:
                case_candidates = await self.generate_case_recommendations(
                    request.case_id, request.current_text
                )
                candidates.extend(case_candidates)
                
            # Legal research suggestions
            research_candidates = await self.generate_research_suggestions(
                request.current_text, user_intent
            )
            candidates.extend(research_candidates)
            
        except Exception as e:
            logger.error(f"Error generating candidates: {e}")
            
        return candidates[:20]  # Limit candidates for performance
        
    async def generate_text_completions(self, text: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate text completion suggestions"""
        candidates = []
        
        # Simple completion based on common legal phrases
        legal_completions = {
            "pursuant to": "pursuant to the applicable law",
            "in accordance with": "in accordance with the terms and conditions",
            "subject to": "subject to the provisions herein",
            "whereas": "whereas the parties agree",
            "therefore": "therefore, it is hereby resolved",
        }
        
        text_lower = text.lower()
        for trigger, completion in legal_completions.items():
            if trigger in text_lower and not completion.lower() in text_lower:
                candidates.append({
                    "id": str(uuid.uuid4()),
                    "type": "auto_complete",
                    "title": f"Complete: {trigger}",
                    "content": completion,
                    "confidence": 0.8,
                    "context": {"trigger": trigger}
                })
                
        return candidates[:3]
        
    async def generate_search_recommendations(self, query: str, case_id: Optional[str]) -> List[Dict[str, Any]]:
        """Generate semantic search recommendations"""
        candidates = []
        
        if len(query) < 5:
            return candidates
            
        try:
            # Call embedding service for semantic search
            embed_response = requests.post(
                f"{EMBED_SERVICE_URL}/embed",
                json={"texts": [query]},
                timeout=5
            )
            
            if embed_response.status_code == 200:
                # Use embedding for Qdrant search
                qdrant_response = requests.post(
                    f"{QDRANT_URL}/collections/legal_documents/points/search",
                    json={
                        "vector": embed_response.json()["vectors"][0],
                        "limit": 5,
                        "with_payload": True
                    },
                    timeout=5
                )
                
                if qdrant_response.status_code == 200:
                    for i, result in enumerate(qdrant_response.json()["result"]):
                        candidates.append({
                            "id": str(uuid.uuid4()),
                            "type": "related_cases",
                            "title": f"Similar Document {i+1}",
                            "content": result["payload"].get("content_preview", ""),
                            "confidence": result["score"],
                            "context": {"qdrant_id": result["id"]}
                        })
                        
        except Exception as e:
            logger.error(f"Search recommendation error: {e}")
            
        return candidates
        
    async def generate_workflow_recommendations(self, user_id: str, context: Dict[str, Any], 
                                             user_intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate workflow-based action recommendations"""
        candidates = []
        
        # Common legal workflow actions
        workflow_actions = {
            "analysis": [
                "Review evidence chronologically",
                "Identify key legal issues",
                "Analyze precedent cases",
                "Assess strengths and weaknesses"
            ],
            "research": [
                "Search case law database",
                "Review recent precedents",
                "Analyze statutory requirements",
                "Check jurisdiction-specific rules"
            ],
            "draft": [
                "Create document outline",
                "Draft opening statement",
                "Prepare legal argument",
                "Review and revise content"
            ]
        }
        
        intent_type = user_intent.get("type", "general")
        if intent_type in workflow_actions:
            for i, action in enumerate(workflow_actions[intent_type][:3]):
                candidates.append({
                    "id": str(uuid.uuid4()),
                    "type": "suggested_actions",
                    "title": action,
                    "content": f"Click to {action.lower()}",
                    "confidence": 0.7 - (i * 0.1),
                    "context": {"action_type": intent_type}
                })
                
        return candidates
        
    async def generate_case_recommendations(self, case_id: str, query: str) -> List[Dict[str, Any]]:
        """Generate case-specific recommendations"""
        candidates = []
        
        try:
            # Query database for case context
            conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT title, case_number, status, meta
                FROM cases 
                WHERE id = %s
            """, (case_id,))
            
            case_data = cursor.fetchone()
            if case_data:
                candidates.append({
                    "id": str(uuid.uuid4()),
                    "type": "case_context",
                    "title": f"Current Case: {case_data['case_number']}",
                    "content": case_data['title'][:100],
                    "confidence": 0.9,
                    "context": {"case_status": case_data['status']}
                })
                
            conn.close()
            
        except Exception as e:
            logger.error(f"Case recommendation error: {e}")
            
        return candidates
        
    async def generate_research_suggestions(self, text: str, user_intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate legal research suggestions"""
        candidates = []
        
        research_templates = [
            "Research precedent cases for similar facts",
            "Review statutory authority for this issue",
            "Check recent court decisions in this jurisdiction",
            "Analyze comparative law approaches"
        ]
        
        for i, template in enumerate(research_templates[:2]):
            candidates.append({
                "id": str(uuid.uuid4()),
                "type": "legal_research",
                "title": template,
                "content": f"Start research: {template.lower()}",
                "confidence": 0.6 - (i * 0.1),
                "context": {"research_type": "precedent"}
            })
            
        return candidates
        
    async def select_recommendations_with_rl(self, user_id: str, candidates: List[Dict[str, Any]], 
                                           user_intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select recommendations using reinforcement learning (epsilon-greedy)"""
        if not candidates:
            return []
            
        state = f"{user_intent['type']}:{user_intent.get('context', 'unknown')}"
        
        # Epsilon-greedy selection
        if np.random.random() < EXPLORATION_RATE:
            # Exploration: random selection
            selected = np.random.choice(len(candidates), size=min(MAX_RECOMMENDATIONS, len(candidates)), replace=False)
        else:
            # Exploitation: select based on Q-values
            q_scores = []
            for candidate in candidates:
                action = f"{candidate['type']}:{candidate.get('confidence', 0.5)}"
                q_key = f"{state}:{action}"
                q_value = self.action_values[user_id].get(q_key, 0.0)
                q_scores.append(q_value + candidate.get('confidence', 0.5))  # Combine Q-value with base confidence
                
            # Select top recommendations
            selected = np.argsort(q_scores)[-MAX_RECOMMENDATIONS:][::-1]
            
        # Return selected recommendations with metadata
        selected_recs = []
        for idx in selected:
            if idx < len(candidates):
                rec = candidates[idx].copy()
                rec["selection_method"] = "exploration" if np.random.random() < EXPLORATION_RATE else "exploitation"
                rec["q_value"] = self.action_values[user_id].get(f"{state}:{rec['type']}:{rec.get('confidence', 0.5)}", 0.0)
                selected_recs.append(rec)
                
        return selected_recs
        
    async def generate_self_prompts(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Generate self-prompts to improve user productivity"""
        if len(query) < 10:
            return []
            
        prompts = []
        
        # Context-aware self-prompts
        if "draft" in query.lower():
            prompts.append("Consider: What are the key legal arguments you want to make?")
            prompts.append("Tip: Start with your strongest point and support with precedent")
            
        elif "research" in query.lower():
            prompts.append("Focus: What specific legal question are you trying to answer?")
            prompts.append("Strategy: Search recent cases first, then expand to broader precedents")
            
        elif "analyze" in query.lower():
            prompts.append("Framework: What legal standard or test applies here?")
            prompts.append("Method: Break down the analysis element by element")
            
        # Add general productivity prompts
        if context.get("current_page") == "case_details":
            prompts.append("Quick action: Review evidence timeline for gaps")
            
        return prompts[:3]  # Limit to 3 self-prompts
        
    async def process_user_feedback(self, request: FeedbackRequest) -> Dict[str, Any]:
        """Process user feedback and update RL model"""
        try:
            # Calculate reward based on action
            reward = self.calculate_reward(request.action, request.context)
            
            # Update Q-values (simplified Q-learning update)
            await self.update_q_values(request.user_id, request.recommendation_id, reward, request.context)
            
            # Update user profile
            await self.update_user_profile(request.user_id, request.action, reward)
            
            # Store feedback for analytics
            await self.store_feedback(request)
            
            return {
                "status": "success",
                "reward": reward,
                "learning_updated": True
            }
            
        except Exception as e:
            logger.error(f"Feedback processing error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    def calculate_reward(self, action: str, context: Dict[str, Any]) -> float:
        """Calculate reward for user action"""
        reward_map = {
            "accepted": 1.0,
            "clicked": 0.5,
            "rejected": -0.3,
            "ignored": -0.1
        }
        
        base_reward = reward_map.get(action, 0.0)
        
        # Context-based reward adjustments
        if context.get("completion_time") and context["completion_time"] < 30:
            base_reward += 0.2  # Quick acceptance bonus
            
        if context.get("user_satisfaction"):
            base_reward += context["user_satisfaction"] * 0.3
            
        return base_reward
        
    async def update_q_values(self, user_id: str, recommendation_id: str, reward: float, context: Dict[str, Any]):
        """Update Q-values using simple Q-learning"""
        # This is a simplified implementation
        # In production, you'd want more sophisticated state representation
        
        state = context.get("state", "unknown")
        action = context.get("action_type", "unknown")
        q_key = f"{state}:{action}"
        
        current_q = self.action_values[user_id].get(q_key, 0.0)
        
        # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        # Simplified version without next state max Q
        updated_q = current_q + LEARNING_RATE * (reward - current_q)
        
        self.action_values[user_id][q_key] = updated_q
        
        # Persist to Redis
        self.redis_client.hset(f"rl:qvalues:{user_id}", q_key, updated_q)
        
    async def update_user_profile(self, user_id: str, action: str, reward: float):
        """Update user profile with feedback"""
        profile_key = f"user:profile:{user_id}"
        
        # Update interaction counts
        self.redis_client.hincrby(profile_key, "total_interactions", 1)
        
        if action == "accepted":
            self.redis_client.hincrby(profile_key, "accepted_recommendations", 1)
            
        # Update average reward
        current_avg = float(self.redis_client.hget(profile_key, "avg_reward") or 0.0)
        total_interactions = int(self.redis_client.hget(profile_key, "total_interactions") or 1)
        
        new_avg = (current_avg * (total_interactions - 1) + reward) / total_interactions
        self.redis_client.hset(profile_key, "avg_reward", new_avg)
        
    async def store_feedback(self, request: FeedbackRequest):
        """Store detailed feedback for analytics"""
        feedback_data = {
            "user_id": request.user_id,
            "recommendation_id": request.recommendation_id,
            "action": request.action,
            "context": request.context,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in Redis list for recent feedback
        self.redis_client.lpush(f"feedback:{request.user_id}", json.dumps(feedback_data))
        self.redis_client.ltrim(f"feedback:{request.user_id}", 0, 100)  # Keep last 100 feedback items
        
    async def store_interaction(self, user_id: str, text: str, user_intent: Dict[str, Any], recommendations: List[Dict[str, Any]]):
        """Store user interaction for future learning"""
        interaction = {
            "user_id": user_id,
            "text": text,
            "intent": user_intent,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
        self.user_interactions[user_id].append(interaction)
        
        # Also store in Redis for persistence
        self.redis_client.lpush(f"interactions:{user_id}", json.dumps(interaction))
        self.redis_client.ltrim(f"interactions:{user_id}", 0, CONTEXT_WINDOW)
        
    async def analyze_user_intent(self, user_id: str) -> Dict[str, Any]:
        """Analyze user intent based on interaction history"""
        interactions = list(self.user_interactions.get(user_id, []))
        
        if not interactions:
            return {"status": "no_data", "patterns": []}
            
        # Analyze patterns in recent interactions
        intent_types = [i["intent"]["type"] for i in interactions if "intent" in i]
        
        if intent_types:
            intent_counts = defaultdict(int)
            for intent in intent_types:
                intent_counts[intent] += 1
                
            most_common = max(intent_counts, key=intent_counts.get)
            
            return {
                "status": "analyzed",
                "primary_intent": most_common,
                "intent_distribution": dict(intent_counts),
                "interaction_count": len(interactions),
                "patterns": self.extract_patterns(interactions)
            }
            
        return {"status": "insufficient_data", "patterns": []}
        
    def extract_patterns(self, interactions: List[Dict[str, Any]]) -> List[str]:
        """Extract behavioral patterns from user interactions"""
        patterns = []
        
        if len(interactions) >= 3:
            # Check for repetitive behavior
            recent_intents = [i["intent"]["type"] for i in interactions[-3:] if "intent" in i]
            if len(set(recent_intents)) == 1:
                patterns.append(f"Focused on {recent_intents[0]} tasks")
                
        # Add more pattern detection logic here
        
        return patterns
        
    async def get_user_analytics(self, user_id: str) -> UserAnalytics:
        """Get comprehensive user analytics"""
        profile_key = f"user:profile:{user_id}"
        
        total_interactions = int(self.redis_client.hget(profile_key, "total_interactions") or 0)
        accepted_recommendations = int(self.redis_client.hget(profile_key, "accepted_recommendations") or 0)
        
        acceptance_rate = accepted_recommendations / max(total_interactions, 1)
        
        # Get common intents from recent interactions
        intent_analysis = await self.analyze_user_intent(user_id)
        common_intents = list(intent_analysis.get("intent_distribution", {}).keys())[:5]
        
        # Calculate productivity score (simplified)
        avg_reward = float(self.redis_client.hget(profile_key, "avg_reward") or 0.0)
        productivity_score = max(0.0, min(1.0, (avg_reward + 1.0) / 2.0))  # Normalize to 0-1
        
        return UserAnalytics(
            user_id=user_id,
            total_interactions=total_interactions,
            accepted_recommendations=accepted_recommendations,
            acceptance_rate=acceptance_rate,
            common_intents=common_intents,
            productivity_score=productivity_score
        )

# Create service instance
recommendation_engine = RecommendationEngine()
app = recommendation_engine.app

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Legal AI Recommendation Engine")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=9002, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"""
    🧠 Legal AI Recommendation Engine with RL Self-Prompting
    
    Features:
    - Reinforcement Learning (ε-greedy with Q-learning)
    - User Intent Analysis
    - Self-Prompting for Productivity
    - Real-time Recommendations
    
    Endpoints:
    - Health: http://{args.host}:{args.port}/health
    - Recommend: http://{args.host}:{args.port}/recommend
    - Feedback: http://{args.host}:{args.port}/feedback
    - Analytics: http://{args.host}:{args.port}/analytics/{{user_id}}
    - Docs: http://{args.host}:{args.port}/docs
    """)
    
    uvicorn.run(
        "recommendation-engine:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )