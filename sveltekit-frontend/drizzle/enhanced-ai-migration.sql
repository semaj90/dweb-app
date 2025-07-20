-- Enhanced Legal AI Database Schema Migration
-- Adds tables for thinking style analysis and document processing

-- Analysis results table for storing AI analysis results
CREATE TABLE IF NOT EXISTS analysis_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id TEXT,
  evidence_id UUID,
  case_id UUID,
  analysis_type TEXT NOT NULL,
  thinking_process TEXT,
  analysis_result TEXT NOT NULL,
  model_used TEXT NOT NULL,
  processing_time INTEGER,
  confidence_score REAL,
  thinking_enabled BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (evidence_id) REFERENCES evidence(id) ON DELETE CASCADE,
  FOREIGN KEY (case_id) REFERENCES cases(id) ON DELETE CASCADE
);

-- AI models configuration table
CREATE TABLE IF NOT EXISTS ai_models (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  model_type TEXT NOT NULL CHECK (model_type IN ('thinking', 'quick')),
  ollama_name TEXT NOT NULL,
  capabilities JSONB DEFAULT '[]'::jsonb,
  is_active BOOLEAN DEFAULT TRUE,
  thinking_enabled BOOLEAN DEFAULT FALSE,
  configuration JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User AI preferences table
CREATE TABLE IF NOT EXISTS user_ai_preferences (
  user_id UUID PRIMARY KEY,
  thinking_style_enabled BOOLEAN DEFAULT FALSE,
  preferred_model UUID,
  thinking_depth TEXT DEFAULT 'detailed' CHECK (thinking_depth IN ('basic', 'detailed', 'comprehensive')),
  focus_areas JSONB DEFAULT '[]'::jsonb,
  analysis_settings JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY (preferred_model) REFERENCES ai_models(id) ON DELETE SET NULL
);

-- Document processing jobs table
CREATE TABLE IF NOT EXISTS document_processing_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id TEXT,
  evidence_id UUID,
  job_type TEXT NOT NULL CHECK (job_type IN ('ocr', 'analysis', 'classification', 'reasoning')),
  status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
  thinking_enabled BOOLEAN DEFAULT FALSE,
  model_used TEXT,
  result JSONB,
  error_message TEXT,
  processing_time INTEGER,
  confidence_score REAL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  FOREIGN KEY (evidence_id) REFERENCES evidence(id) ON DELETE CASCADE
);

-- Document analysis cache table for performance
CREATE TABLE IF NOT EXISTS document_analysis_cache (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  content_hash TEXT NOT NULL UNIQUE,
  analysis_type TEXT NOT NULL,
  thinking_enabled BOOLEAN NOT NULL,
  model_used TEXT NOT NULL,
  analysis_result JSONB NOT NULL,
  confidence_score REAL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  expires_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP + INTERVAL '7 days'),
  access_count INTEGER DEFAULT 0,
  last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_analysis_results_evidence_id ON analysis_results(evidence_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_case_id ON analysis_results(case_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_created_at ON analysis_results(created_at);
CREATE INDEX IF NOT EXISTS idx_analysis_results_model_used ON analysis_results(model_used);

CREATE INDEX IF NOT EXISTS idx_ai_models_type ON ai_models(model_type);
CREATE INDEX IF NOT EXISTS idx_ai_models_active ON ai_models(is_active);

CREATE INDEX IF NOT EXISTS idx_document_processing_jobs_status ON document_processing_jobs(status);
CREATE INDEX IF NOT EXISTS idx_document_processing_jobs_evidence_id ON document_processing_jobs(evidence_id);
CREATE INDEX IF NOT EXISTS idx_document_processing_jobs_created_at ON document_processing_jobs(created_at);

CREATE INDEX IF NOT EXISTS idx_document_analysis_cache_hash ON document_analysis_cache(content_hash);
CREATE INDEX IF NOT EXISTS idx_document_analysis_cache_expires ON document_analysis_cache(expires_at);

-- Insert default AI models
INSERT INTO ai_models (id, name, model_type, ollama_name, capabilities, thinking_enabled, configuration) VALUES
(
  'quick-gemma3-7b'::uuid, 
  'Gemma 3 Quick', 
  'quick', 
  'gemma3:7b', 
  '["classification", "extraction", "summarization", "quick_analysis"]'::jsonb, 
  FALSE,
  '{"temperature": 0.3, "max_tokens": 512, "response_time": "fast"}'::jsonb
),
(
  'thinking-gemma3-legal'::uuid, 
  'Gemma 3 Legal Thinking', 
  'thinking', 
  'legal-gemma3-thinking', 
  '["reasoning", "detailed_analysis", "chain_of_custody", "legal_research", "case_assessment"]'::jsonb, 
  TRUE,
  '{"temperature": 0.7, "max_tokens": 2048, "thinking_depth": "detailed"}'::jsonb
)
ON CONFLICT (id) DO UPDATE SET
  name = EXCLUDED.name,
  model_type = EXCLUDED.model_type,
  ollama_name = EXCLUDED.ollama_name,
  capabilities = EXCLUDED.capabilities,
  thinking_enabled = EXCLUDED.thinking_enabled,
  configuration = EXCLUDED.configuration,
  updated_at = CURRENT_TIMESTAMP;

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_analysis_results_updated_at 
  BEFORE UPDATE ON analysis_results 
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ai_models_updated_at 
  BEFORE UPDATE ON ai_models 
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_ai_preferences_updated_at 
  BEFORE UPDATE ON user_ai_preferences 
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function to clean up expired cache entries
CREATE OR REPLACE FUNCTION cleanup_expired_analysis_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM document_analysis_cache 
    WHERE expires_at < CURRENT_TIMESTAMP;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to get analysis statistics
CREATE OR REPLACE FUNCTION get_analysis_statistics(
    start_date TIMESTAMP DEFAULT (CURRENT_TIMESTAMP - INTERVAL '30 days'),
    end_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
RETURNS TABLE(
    total_analyses BIGINT,
    thinking_analyses BIGINT,
    quick_analyses BIGINT,
    avg_confidence NUMERIC,
    avg_processing_time NUMERIC,
    most_used_model TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_analyses,
        COUNT(*) FILTER (WHERE thinking_enabled = TRUE) as thinking_analyses,
        COUNT(*) FILTER (WHERE thinking_enabled = FALSE) as quick_analyses,
        ROUND(AVG(confidence_score), 3) as avg_confidence,
        ROUND(AVG(processing_time), 2) as avg_processing_time,
        MODE() WITHIN GROUP (ORDER BY model_used) as most_used_model
    FROM analysis_results 
    WHERE created_at BETWEEN start_date AND end_date;
END;
$$ LANGUAGE plpgsql;

-- Add comments to tables for documentation
COMMENT ON TABLE analysis_results IS 'Stores results from AI document analysis with thinking process details';
COMMENT ON TABLE ai_models IS 'Configuration for available AI models including thinking vs quick modes';
COMMENT ON TABLE user_ai_preferences IS 'User preferences for AI analysis including thinking style settings';
COMMENT ON TABLE document_processing_jobs IS 'Queue and status tracking for document processing jobs';
COMMENT ON TABLE document_analysis_cache IS 'Cache for analysis results to improve performance';

COMMENT ON COLUMN analysis_results.thinking_process IS 'Step-by-step reasoning process when thinking style is enabled';
COMMENT ON COLUMN analysis_results.analysis_result IS 'JSON structure containing the analysis findings and recommendations';
COMMENT ON COLUMN ai_models.capabilities IS 'JSON array of analysis types this model can perform';
COMMENT ON COLUMN user_ai_preferences.thinking_depth IS 'Preferred depth of reasoning: basic, detailed, or comprehensive';

-- Insert sample data for testing (optional - remove in production)
-- INSERT INTO user_ai_preferences (user_id, thinking_style_enabled, preferred_model, thinking_depth) 
-- SELECT id, FALSE, 'quick-gemma3-7b'::uuid, 'detailed' 
-- FROM users 
-- WHERE email LIKE '%@example.com'
-- ON CONFLICT (user_id) DO NOTHING;

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON analysis_results TO your_app_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ai_models TO your_app_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON user_ai_preferences TO your_app_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON document_processing_jobs TO your_app_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON document_analysis_cache TO your_app_user;

-- Verify the migration was successful
DO $$
BEGIN
    -- Check if all tables were created
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_name IN (
            'analysis_results', 
            'ai_models', 
            'user_ai_preferences', 
            'document_processing_jobs',
            'document_analysis_cache'
        )
        GROUP BY table_schema
        HAVING COUNT(*) = 5
    ) THEN
        RAISE NOTICE 'SUCCESS: All enhanced legal AI tables created successfully';
    ELSE
        RAISE EXCEPTION 'FAILED: Some tables were not created properly';
    END IF;
END $$;
