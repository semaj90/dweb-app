-- Migration 003: Search Terms + RAG Feedback + pg_trgm extension
-- Supports "Did You Mean" lexical suggestions & self-prompting feedback loops
-- Prerequisites: vector extension already enabled in 001; passages & graph_edges from 002.

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Table: search_terms
-- Purpose: Track distinct normalized search terms and aggregate usage for
--          lexical suggestion prioritization & analytics.
CREATE TABLE IF NOT EXISTS search_terms (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    term TEXT NOT NULL UNIQUE,              -- normalized (lowercase, trimmed)
    raw_variants JSONB DEFAULT '[]',        -- array of observed raw user inputs mapping to this normalized term
    usage_count BIGINT DEFAULT 1 CHECK (usage_count >= 0),
    last_used_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- GIN index with pg_trgm for fuzzy matching (lexical suggestions)
CREATE INDEX IF NOT EXISTS idx_search_terms_trgm ON search_terms USING gin (term gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_search_terms_usage ON search_terms (usage_count DESC);

-- Increment usage helper function (idempotent upsert style)
CREATE OR REPLACE FUNCTION increment_search_term(p_term TEXT, p_raw TEXT)
RETURNS UUID AS $$
DECLARE
    v_id UUID;
BEGIN
    p_term := lower(trim(p_term));
    LOOP
        UPDATE search_terms
        SET usage_count = usage_count + 1,
            last_used_at = NOW(),
            raw_variants = CASE WHEN p_raw IS NOT NULL THEN (
                 CASE WHEN NOT raw_variants ? p_raw THEN raw_variants || to_jsonb(p_raw) ELSE raw_variants END
            ) ELSE raw_variants END
        WHERE term = p_term
        RETURNING id INTO v_id;
        EXIT WHEN FOUND;
        BEGIN
            INSERT INTO search_terms(term, raw_variants)
            VALUES (p_term, CASE WHEN p_raw IS NOT NULL THEN to_jsonb(ARRAY[p_raw]) ELSE '[]'::jsonb END)
            RETURNING id INTO v_id;
            EXIT; -- success
        EXCEPTION WHEN unique_violation THEN
            -- retry loop
        END;
    END LOOP;
    RETURN v_id;
END;$$ LANGUAGE plpgsql;

-- Table: rag_feedback
-- Purpose: Log user interactions & quality signals for RAG answer + suggestion evaluation
CREATE TABLE IF NOT EXISTS rag_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID,                        -- nullable (anonymous)
    session_id TEXT,
    query TEXT NOT NULL,
    expanded_query TEXT,                 -- after self-prompting expansion
    top_passage_ids UUID[] ,             -- passage ids presented (ordered)
    clicked_passage_id UUID,             -- which one user engaged with
    relevance_label SMALLINT,            -- optional explicit rating (0-3 / 0-5 future)
    answer_quality SMALLINT,             -- user rating of final answer
    rerank_policy TEXT,                  -- e.g. BASELINE | LINUCB_v1
    prompt_tokens INT,                   -- LLM prompt token count (for cost tracking)
    answer_tokens INT,                   -- completion token count
    latency_ms INT,
    model_version TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rag_feedback_created ON rag_feedback(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rag_feedback_query_trgm ON rag_feedback USING gin (query gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_rag_feedback_user ON rag_feedback(user_id);

-- View: v_rag_feedback_stats (aggregated for quick analytics)
CREATE OR REPLACE VIEW v_rag_feedback_stats AS
SELECT
    date_trunc('hour', created_at) AS hour_bucket,
    COUNT(*) AS events,
    AVG(latency_ms) AS avg_latency_ms,
    AVG(answer_quality) AS avg_answer_quality,
    AVG(relevance_label) AS avg_relevance,
    AVG(answer_tokens) AS avg_answer_tokens
FROM rag_feedback
GROUP BY 1
ORDER BY 1 DESC;

-- Future extension note: consider partitioning rag_feedback by month if volume justifies.
