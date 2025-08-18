-- Enhanced RAG V2 PostgreSQL Schema with pgvector
-- Complete schema for User Intent, Recommendations, Todo Solver, and Analytics

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ==========================================
-- USER INTENTS TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS user_intents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    intent VARCHAR(100) NOT NULL,
    keywords TEXT[] DEFAULT '{}',
    confidence FLOAT DEFAULT 0.0,
    context JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for user_intents
CREATE INDEX idx_user_intents_user_id ON user_intents(user_id);
CREATE INDEX idx_user_intents_intent ON user_intents(intent);
CREATE INDEX idx_user_intents_created_at ON user_intents(created_at DESC);
CREATE INDEX idx_user_intents_keywords ON user_intents USING GIN(keywords);
CREATE INDEX idx_user_intents_context ON user_intents USING GIN(context);

-- ==========================================
-- RECOMMENDATIONS TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    type VARCHAR(50) NOT NULL, -- legal, precedent, compliance, etc.
    confidence FLOAT DEFAULT 0.0,
    context JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'active', -- active, dismissed, applied
    embedding vector(384), -- For semantic search
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for recommendations
CREATE INDEX idx_recommendations_user_id ON recommendations(user_id);
CREATE INDEX idx_recommendations_type ON recommendations(type);
CREATE INDEX idx_recommendations_status ON recommendations(status);
CREATE INDEX idx_recommendations_created_at ON recommendations(created_at DESC);
CREATE INDEX idx_recommendations_context ON recommendations USING GIN(context);
CREATE INDEX idx_recommendations_embedding ON recommendations USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ==========================================
-- TODO ITEMS TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS todo_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    priority INTEGER DEFAULT 0, -- 0-10 scale
    status VARCHAR(50) DEFAULT 'pending', -- pending, in_progress, solved, cancelled
    solution TEXT,
    due_date TIMESTAMP WITH TIME ZONE,
    solved_at TIMESTAMP WITH TIME ZONE,
    ai_confidence FLOAT,
    solution_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for todo_items
CREATE INDEX idx_todo_items_user_id ON todo_items(user_id);
CREATE INDEX idx_todo_items_status ON todo_items(status);
CREATE INDEX idx_todo_items_priority ON todo_items(priority DESC);
CREATE INDEX idx_todo_items_due_date ON todo_items(due_date);
CREATE INDEX idx_todo_items_created_at ON todo_items(created_at DESC);

-- ==========================================
-- USER SESSIONS TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) UNIQUE NOT NULL,
    state VARCHAR(50) NOT NULL, -- idle, active, processing
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    idle_start_time TIMESTAMP WITH TIME ZONE,
    context JSONB DEFAULT '{}',
    behavior_pattern VARCHAR(100),
    activity_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for user_sessions
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_state ON user_sessions(state);
CREATE INDEX idx_user_sessions_last_activity ON user_sessions(last_activity DESC);

-- ==========================================
-- ANALYTICS EVENTS TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS analytics_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    session_id UUID REFERENCES user_sessions(id)
);

-- Indexes for analytics_events
CREATE INDEX idx_analytics_events_user_id ON analytics_events(user_id);
CREATE INDEX idx_analytics_events_type ON analytics_events(event_type);
CREATE INDEX idx_analytics_events_timestamp ON analytics_events(timestamp DESC);
CREATE INDEX idx_analytics_events_data ON analytics_events USING GIN(event_data);

-- ==========================================
-- SOM CLUSTERS TABLE (Self-Organizing Maps)
-- ==========================================
CREATE TABLE IF NOT EXISTS som_clusters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cluster_name VARCHAR(255) NOT NULL,
    centroid vector(384), -- Cluster center
    documents TEXT[] DEFAULT '{}',
    document_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    quality_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for som_clusters
CREATE INDEX idx_som_clusters_name ON som_clusters(cluster_name);
CREATE INDEX idx_som_clusters_centroid ON som_clusters USING ivfflat (centroid vector_cosine_ops) WITH (lists = 50);
CREATE INDEX idx_som_clusters_metadata ON som_clusters USING GIN(metadata);

-- ==========================================
-- BEHAVIOR PATTERNS TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS behavior_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    pattern_type VARCHAR(100) NOT NULL,
    pattern_data JSONB DEFAULT '{}',
    frequency INTEGER DEFAULT 0,
    last_observed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    confidence FLOAT DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for behavior_patterns
CREATE INDEX idx_behavior_patterns_user_id ON behavior_patterns(user_id);
CREATE INDEX idx_behavior_patterns_type ON behavior_patterns(pattern_type);
CREATE INDEX idx_behavior_patterns_frequency ON behavior_patterns(frequency DESC);

-- ==========================================
-- IDLE ACTIONS TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS idle_actions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    action_type VARCHAR(100) NOT NULL, -- fetch_recommendations, solve_todo, update_analytics
    action_data JSONB DEFAULT '{}',
    triggered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'pending', -- pending, processing, completed, failed
    result JSONB DEFAULT '{}'
);

-- Indexes for idle_actions
CREATE INDEX idx_idle_actions_user_id ON idle_actions(user_id);
CREATE INDEX idx_idle_actions_type ON idle_actions(action_type);
CREATE INDEX idx_idle_actions_status ON idle_actions(status);
CREATE INDEX idx_idle_actions_triggered ON idle_actions(triggered_at DESC);

-- ==========================================
-- XSTATE TRANSITIONS TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS xstate_transitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES user_sessions(id),
    from_state VARCHAR(50),
    to_state VARCHAR(50),
    event VARCHAR(100),
    context JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for xstate_transitions
CREATE INDEX idx_xstate_transitions_session ON xstate_transitions(session_id);
CREATE INDEX idx_xstate_transitions_states ON xstate_transitions(from_state, to_state);
CREATE INDEX idx_xstate_transitions_timestamp ON xstate_transitions(timestamp DESC);

-- ==========================================
-- RABBITMQ MESSAGE QUEUE TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS message_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    queue_name VARCHAR(100) NOT NULL,
    message_type VARCHAR(100) NOT NULL,
    payload JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'pending', -- pending, processing, completed, failed
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT
);

-- Indexes for message_queue
CREATE INDEX idx_message_queue_name ON message_queue(queue_name);
CREATE INDEX idx_message_queue_status ON message_queue(status);
CREATE INDEX idx_message_queue_created ON message_queue(created_at);

-- ==========================================
-- CONFLICT RESOLUTION TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS conflict_resolutions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_type VARCHAR(100) NOT NULL,
    resource_id VARCHAR(255) NOT NULL,
    conflict_type VARCHAR(100) NOT NULL,
    resolution_strategy VARCHAR(100),
    original_data JSONB DEFAULT '{}',
    resolved_data JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'pending',
    resolved_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for conflict_resolutions
CREATE INDEX idx_conflict_resolutions_type ON conflict_resolutions(resource_type);
CREATE INDEX idx_conflict_resolutions_status ON conflict_resolutions(status);
CREATE INDEX idx_conflict_resolutions_created ON conflict_resolutions(created_at DESC);

-- ==========================================
-- TRIGGERS FOR UPDATED_AT
-- ==========================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to all tables with updated_at
CREATE TRIGGER update_user_intents_updated_at BEFORE UPDATE ON user_intents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_recommendations_updated_at BEFORE UPDATE ON recommendations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_todo_items_updated_at BEFORE UPDATE ON todo_items
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_sessions_updated_at BEFORE UPDATE ON user_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_som_clusters_updated_at BEFORE UPDATE ON som_clusters
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_behavior_patterns_updated_at BEFORE UPDATE ON behavior_patterns
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ==========================================
-- FUNCTIONS FOR ANALYTICS
-- ==========================================

-- Function to get user activity summary
CREATE OR REPLACE FUNCTION get_user_activity_summary(p_user_id VARCHAR)
RETURNS TABLE (
    total_intents BIGINT,
    total_recommendations BIGINT,
    pending_todos BIGINT,
    solved_todos BIGINT,
    avg_confidence FLOAT,
    last_activity TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        (SELECT COUNT(*) FROM user_intents WHERE user_id = p_user_id) as total_intents,
        (SELECT COUNT(*) FROM recommendations WHERE user_id = p_user_id) as total_recommendations,
        (SELECT COUNT(*) FROM todo_items WHERE user_id = p_user_id AND status = 'pending') as pending_todos,
        (SELECT COUNT(*) FROM todo_items WHERE user_id = p_user_id AND status = 'solved') as solved_todos,
        (SELECT AVG(confidence) FROM user_intents WHERE user_id = p_user_id) as avg_confidence,
        (SELECT MAX(last_activity) FROM user_sessions WHERE user_id = p_user_id) as last_activity;
END;
$$ LANGUAGE plpgsql;

-- Function to find similar recommendations using vector similarity
CREATE OR REPLACE FUNCTION find_similar_recommendations(
    p_embedding vector(384),
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    title TEXT,
    description TEXT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        r.id,
        r.title,
        r.description,
        1 - (r.embedding <=> p_embedding) as similarity
    FROM recommendations r
    WHERE r.embedding IS NOT NULL
    ORDER BY r.embedding <=> p_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- ==========================================
-- SAMPLE DATA FOR TESTING
-- ==========================================

-- Insert sample user session
INSERT INTO user_sessions (user_id, state, behavior_pattern, context)
VALUES 
    ('test-user-1', 'active', 'legal_research', '{"current_case": "Contract Review 2025"}'),
    ('test-user-2', 'idle', 'document_drafting', '{"last_document": "NDA Template"}')
ON CONFLICT (user_id) DO NOTHING;

-- Insert sample intents
INSERT INTO user_intents (user_id, intent, keywords, confidence, context)
VALUES 
    ('test-user-1', 'contract_review', ARRAY['liability', 'indemnification', 'warranty'], 0.92, '{"document_type": "SaaS Agreement"}'),
    ('test-user-1', 'case_research', ARRAY['precedent', 'breach', 'damages'], 0.87, '{"jurisdiction": "California"}')
ON CONFLICT DO NOTHING;

-- Insert sample todos
INSERT INTO todo_items (user_id, title, description, priority, status)
VALUES 
    ('test-user-1', 'Review Section 3.2 of Contract', 'Check liability limitations', 8, 'pending'),
    ('test-user-1', 'Research similar cases', 'Find precedents for breach of contract', 7, 'pending'),
    ('test-user-2', 'Draft response to motion', 'Prepare response by Friday', 9, 'pending')
ON CONFLICT DO NOTHING;

-- Insert sample recommendations
INSERT INTO recommendations (user_id, title, description, type, confidence, status)
VALUES 
    ('test-user-1', 'Similar Case: ABC Corp v. XYZ Inc', 'Relevant precedent for contract breach', 'precedent', 0.89, 'active'),
    ('test-user-1', 'Compliance Alert: GDPR Update', 'New regulations effective next month', 'compliance', 0.95, 'active')
ON CONFLICT DO NOTHING;

-- ==========================================
-- VIEWS FOR MONITORING
-- ==========================================

-- View for active user sessions
CREATE OR REPLACE VIEW v_active_sessions AS
SELECT 
    s.user_id,
    s.state,
    s.last_activity,
    s.behavior_pattern,
    COUNT(DISTINCT i.id) as recent_intents,
    COUNT(DISTINCT t.id) as pending_todos
FROM user_sessions s
LEFT JOIN user_intents i ON s.user_id = i.user_id 
    AND i.created_at > NOW() - INTERVAL '1 hour'
LEFT JOIN todo_items t ON s.user_id = t.user_id 
    AND t.status = 'pending'
WHERE s.last_activity > NOW() - INTERVAL '30 minutes'
GROUP BY s.user_id, s.state, s.last_activity, s.behavior_pattern;

-- View for recommendation effectiveness
CREATE OR REPLACE VIEW v_recommendation_effectiveness AS
SELECT 
    r.type,
    COUNT(*) as total_count,
    AVG(r.confidence) as avg_confidence,
    SUM(CASE WHEN r.status = 'applied' THEN 1 ELSE 0 END) as applied_count,
    SUM(CASE WHEN r.status = 'dismissed' THEN 1 ELSE 0 END) as dismissed_count
FROM recommendations r
GROUP BY r.type;

-- ==========================================
-- PERMISSIONS
-- ==========================================

-- Grant permissions to application user (adjust as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_app_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO your_app_user;

-- ==========================================
-- INDEXES FOR PERFORMANCE
-- ==========================================

-- Composite indexes for common queries
CREATE INDEX idx_user_activity ON analytics_events(user_id, timestamp DESC);
CREATE INDEX idx_todo_user_status ON todo_items(user_id, status, priority DESC);
CREATE INDEX idx_recommendations_user_status ON recommendations(user_id, status, created_at DESC);

-- Partial indexes for optimization
CREATE INDEX idx_pending_todos ON todo_items(user_id) WHERE status = 'pending';
CREATE INDEX idx_active_recommendations ON recommendations(user_id) WHERE status = 'active';
CREATE INDEX idx_idle_sessions ON user_sessions(user_id) WHERE state = 'idle';

-- ==========================================
-- COMPLETION MESSAGE
-- ==========================================
DO $$
BEGIN
    RAISE NOTICE 'Enhanced RAG V2 Schema created successfully!';
    RAISE NOTICE 'Tables created: user_intents, recommendations, todo_items, user_sessions, analytics_events, etc.';
    RAISE NOTICE 'Run the application with: go run cmd/enhanced-rag-v2-local/main.go';
END $$;
