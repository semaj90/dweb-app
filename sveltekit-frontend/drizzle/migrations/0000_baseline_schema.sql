--
-- PostgreSQL database dump
--

-- Dumped from database version 17.5
-- Dumped by pg_dump version 17.5

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: btree_gin; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS btree_gin WITH SCHEMA public;


--
-- Name: EXTENSION btree_gin; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION btree_gin IS 'support for indexing common datatypes in GIN';


--
-- Name: pg_trgm; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public;


--
-- Name: EXTENSION pg_trgm; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION pg_trgm IS 'text similarity measurement and index searching based on trigrams';


--
-- Name: pgcrypto; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public;


--
-- Name: EXTENSION pgcrypto; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION pgcrypto IS 'cryptographic functions';


--
-- Name: uuid-ossp; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;


--
-- Name: EXTENSION "uuid-ossp"; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION "uuid-ossp" IS 'generate universally unique identifiers (UUIDs)';


--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


--
-- Name: cosine_similarity(public.vector, public.vector); Type: FUNCTION; Schema: public; Owner: legal_admin
--

CREATE FUNCTION public.cosine_similarity(a public.vector, b public.vector) RETURNS double precision
    LANGUAGE sql IMMUTABLE STRICT
    AS $$
      SELECT 1 - (a <=> b)
      $$;


ALTER FUNCTION public.cosine_similarity(a public.vector, b public.vector) OWNER TO legal_admin;

--
-- Name: create_vector_for_evidence(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.create_vector_for_evidence() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
  INSERT INTO vectors (owner_type, owner_id, embedding, payload)
  VALUES (
    'evidence',
    NEW.id,
    (SELECT ARRAY(SELECT 0.0 FROM generate_series(1, 768))::vector),
    jsonb_build_object('filename', NEW.filename, 'caseId', NEW.case_id)
  );
  PERFORM pg_notify('evidence_inserted', json_build_object('id', NEW.id, 'caseId', NEW.case_id)::text);
  RETURN NEW;
END;
$$;


ALTER FUNCTION public.create_vector_for_evidence() OWNER TO postgres;

--
-- Name: create_vector_for_report(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.create_vector_for_report() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
  INSERT INTO vectors (owner_type, owner_id, embedding, payload)
  VALUES (
    'report',
    NEW.id,
    (SELECT ARRAY(SELECT 0.0 FROM generate_series(1, 768))::vector),
    jsonb_build_object('title', NEW.title, 'caseId', NEW.case_id)
  );
  PERFORM pg_notify('report_inserted', json_build_object('id', NEW.id, 'caseId', NEW.case_id)::text);
  RETURN NEW;
END;
$$;


ALTER FUNCTION public.create_vector_for_report() OWNER TO postgres;

--
-- Name: get_similar_documents(public.vector, real, integer); Type: FUNCTION; Schema: public; Owner: legal_admin
--

CREATE FUNCTION public.get_similar_documents(query_embedding public.vector, similarity_threshold real DEFAULT 0.7, result_limit integer DEFAULT 10) RETURNS TABLE(id uuid, title character varying, similarity real, document_type character varying, practice_area character varying)
    LANGUAGE plpgsql
    AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.title,
        cosine_similarity(d.content_embedding, query_embedding) as similarity,
        d.document_type,
        d.practice_area
    FROM legal_documents d
    WHERE d.content_embedding IS NOT NULL
        AND d.deleted_at IS NULL
        AND d.status = 'active'
        AND cosine_similarity(d.content_embedding, query_embedding) >= similarity_threshold
    ORDER BY d.content_embedding <=> query_embedding
    LIMIT result_limit;
END;
$$;


ALTER FUNCTION public.get_similar_documents(query_embedding public.vector, similarity_threshold real, result_limit integer) OWNER TO legal_admin;

--
-- Name: notify_case_changes(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.notify_case_changes() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    PERFORM pg_notify('cases_changed', json_build_object('action', 'insert', 'id', NEW.id, 'userId', NEW.user_id)::text);
    RETURN NEW;
  ELSIF TG_OP = 'UPDATE' THEN
    PERFORM pg_notify('cases_changed', json_build_object('action', 'update', 'id', NEW.id, 'userId', NEW.user_id)::text);
    RETURN NEW;
  ELSIF TG_OP = 'DELETE' THEN
    PERFORM pg_notify('cases_changed', json_build_object('action', 'delete', 'id', OLD.id, 'userId', OLD.user_id)::text);
    RETURN OLD;
  END IF;
  RETURN NULL;
END;
$$;


ALTER FUNCTION public.notify_case_changes() OWNER TO postgres;

--
-- Name: notify_report_changes(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.notify_report_changes() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    PERFORM pg_notify('reports_changed', json_build_object('action', 'insert', 'id', NEW.id, 'caseId', NEW.case_id)::text);
    RETURN NEW;
  ELSIF TG_OP = 'UPDATE' THEN
    PERFORM pg_notify('reports_changed', json_build_object('action', 'update', 'id', NEW.id, 'caseId', NEW.case_id)::text);
    RETURN NEW;
  ELSIF TG_OP = 'DELETE' THEN
    PERFORM pg_notify('reports_changed', json_build_object('action', 'delete', 'id', OLD.id, 'caseId', OLD.case_id)::text);
    RETURN OLD;
  END IF;
  RETURN NULL;
END;
$$;


ALTER FUNCTION public.notify_report_changes() OWNER TO postgres;

--
-- Name: search_cases(text); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.search_cases(search_query text) RETURNS TABLE(id uuid, title character varying, case_number character varying, court character varying, case_type character varying, description text, rank real)
    LANGUAGE plpgsql
    AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        c.title,
        c.case_number,
        c.court,
        c.case_type,
        c.description,
        ts_rank(to_tsvector('english', c.title || ' ' || COALESCE(c.description, '')), plainto_tsquery('english', search_query)) AS rank
    FROM cases c
    WHERE to_tsvector('english', c.title || ' ' || COALESCE(c.description, '')) @@ plainto_tsquery('english', search_query)
    ORDER BY rank DESC;
END;
$$;


ALTER FUNCTION public.search_cases(search_query text) OWNER TO postgres;

--
-- Name: update_processing_stats(); Type: FUNCTION; Schema: public; Owner: legal_admin
--

CREATE FUNCTION public.update_processing_stats() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.indexed_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.update_processing_stats() OWNER TO legal_admin;

--
-- Name: update_updated_at_column(); Type: FUNCTION; Schema: public; Owner: legal_admin
--

CREATE FUNCTION public.update_updated_at_column() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.update_updated_at_column() OWNER TO legal_admin;

--
-- Name: update_vectors_updated_at(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.update_vectors_updated_at() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.update_vectors_updated_at() OWNER TO postgres;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: __drizzle_migrations__; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.__drizzle_migrations__ (
    id integer NOT NULL,
    hash text NOT NULL,
    created_at bigint
);


ALTER TABLE public.__drizzle_migrations__ OWNER TO postgres;

--
-- Name: __drizzle_migrations___id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.__drizzle_migrations___id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.__drizzle_migrations___id_seq OWNER TO postgres;

--
-- Name: __drizzle_migrations___id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.__drizzle_migrations___id_seq OWNED BY public.__drizzle_migrations__.id;


--
-- Name: activity_logs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.activity_logs (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    user_id uuid,
    action character varying(100) NOT NULL,
    entity_type character varying(50),
    entity_id uuid,
    details jsonb DEFAULT '{}'::jsonb,
    ip_address inet,
    user_agent text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.activity_logs OWNER TO postgres;

--
-- Name: auto_tags; Type: TABLE; Schema: public; Owner: legal_admin
--

CREATE TABLE public.auto_tags (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    entity_id uuid NOT NULL,
    entity_type character varying(50) NOT NULL,
    tag character varying(100) NOT NULL,
    confidence numeric(3,2) NOT NULL,
    source character varying(50) DEFAULT 'ai_analysis'::character varying NOT NULL,
    model character varying(100),
    extracted_at timestamp without time zone DEFAULT now() NOT NULL,
    is_confirmed boolean DEFAULT false NOT NULL,
    confirmed_by uuid,
    confirmed_at timestamp without time zone
);


ALTER TABLE public.auto_tags OWNER TO legal_admin;

--
-- Name: canvas_states; Type: TABLE; Schema: public; Owner: legal_admin
--

CREATE TABLE public.canvas_states (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    case_id uuid,
    name character varying(255) NOT NULL,
    canvas_data jsonb NOT NULL,
    version integer DEFAULT 1,
    is_default boolean DEFAULT false,
    created_by uuid,
    created_at timestamp without time zone DEFAULT now() NOT NULL,
    updated_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.canvas_states OWNER TO legal_admin;

--
-- Name: case_scores; Type: TABLE; Schema: public; Owner: legal_admin
--

CREATE TABLE public.case_scores (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    case_id uuid NOT NULL,
    score numeric(5,2) NOT NULL,
    risk_level character varying(20) NOT NULL,
    breakdown jsonb DEFAULT '{}'::jsonb NOT NULL,
    criteria jsonb DEFAULT '{}'::jsonb NOT NULL,
    recommendations jsonb DEFAULT '[]'::jsonb NOT NULL,
    calculated_by uuid,
    calculated_at timestamp without time zone DEFAULT now() NOT NULL,
    updated_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.case_scores OWNER TO legal_admin;

--
-- Name: cases; Type: TABLE; Schema: public; Owner: legal_admin
--

CREATE TABLE public.cases (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    title character varying(500) NOT NULL,
    description text,
    case_number character varying(100),
    status character varying(50) DEFAULT 'active'::character varying NOT NULL,
    priority character varying(20) DEFAULT 'medium'::character varying NOT NULL,
    practice_area character varying(100),
    jurisdiction character varying(100),
    court character varying(200),
    client_name character varying(200),
    opposing_party character varying(200),
    assigned_attorney uuid,
    filing_date timestamp with time zone,
    due_date timestamp with time zone,
    closed_date timestamp with time zone,
    case_embedding public.vector(384),
    qdrant_id uuid,
    qdrant_collection character varying(100) DEFAULT 'cases'::character varying,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.cases OWNER TO legal_admin;

--
-- Name: citations; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.citations (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    case_id uuid,
    citation_text text NOT NULL,
    citation_type character varying(100),
    source character varying(500),
    page_number integer,
    relevance_score numeric(3,2),
    context text,
    verified boolean DEFAULT false,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    metadata jsonb DEFAULT '{}'::jsonb
);


ALTER TABLE public.citations OWNER TO postgres;

--
-- Name: criminals; Type: TABLE; Schema: public; Owner: legal_admin
--

CREATE TABLE public.criminals (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    first_name character varying(100) NOT NULL,
    last_name character varying(100) NOT NULL,
    middle_name character varying(100),
    aliases jsonb DEFAULT '[]'::jsonb NOT NULL,
    date_of_birth timestamp without time zone,
    place_of_birth character varying(200),
    address text,
    phone character varying(20),
    email character varying(255),
    ssn character varying(11),
    drivers_license character varying(50),
    height integer,
    weight integer,
    eye_color character varying(20),
    hair_color character varying(20),
    distinguishing_marks text,
    photo_url text,
    fingerprints jsonb DEFAULT '{}'::jsonb,
    threat_level character varying(20) DEFAULT 'low'::character varying NOT NULL,
    status character varying(20) DEFAULT 'active'::character varying NOT NULL,
    notes text,
    ai_summary text,
    ai_tags jsonb DEFAULT '[]'::jsonb NOT NULL,
    created_by uuid,
    created_at timestamp without time zone DEFAULT now() NOT NULL,
    updated_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.criminals OWNER TO legal_admin;

--
-- Name: document_chunks; Type: TABLE; Schema: public; Owner: legal_admin
--

CREATE TABLE public.document_chunks (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    document_id uuid NOT NULL,
    document_type character varying(50) NOT NULL,
    chunk_index integer NOT NULL,
    content text NOT NULL,
    embedding public.vector(768) NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.document_chunks OWNER TO legal_admin;

--
-- Name: document_vectors; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.document_vectors (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    document_id uuid NOT NULL,
    document_type character varying(50) NOT NULL,
    chunk_index integer DEFAULT 0,
    content text NOT NULL,
    embedding public.vector(768),
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.document_vectors OWNER TO postgres;

--
-- Name: embedding_cache; Type: TABLE; Schema: public; Owner: legal_admin
--

CREATE TABLE public.embedding_cache (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    text_hash text NOT NULL,
    embedding public.vector(768) NOT NULL,
    model character varying(100) NOT NULL,
    created_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.embedding_cache OWNER TO legal_admin;

--
-- Name: evidence; Type: TABLE; Schema: public; Owner: legal_admin
--

CREATE TABLE public.evidence (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    case_id uuid,
    title character varying(255) NOT NULL,
    description text,
    evidence_type character varying(50) NOT NULL,
    file_url text,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    user_id uuid,
    title_embedding public.vector(384),
    content_embedding public.vector(384),
    sub_type character varying(50),
    file_name character varying(255),
    file_size integer,
    mime_type character varying(100),
    hash character varying(128),
    collected_at timestamp without time zone,
    collected_by character varying(255),
    location character varying(255),
    chain_of_custody jsonb DEFAULT '[]'::jsonb,
    tags jsonb DEFAULT '[]'::jsonb NOT NULL,
    is_admissible boolean DEFAULT true,
    confidentiality_level character varying(50) DEFAULT 'internal'::character varying,
    ai_analysis jsonb DEFAULT '{}'::jsonb,
    ai_tags jsonb DEFAULT '[]'::jsonb,
    ai_summary text,
    summary text,
    summary_type character varying(50),
    board_position jsonb DEFAULT '{}'::jsonb
);


ALTER TABLE public.evidence OWNER TO legal_admin;

--
-- Name: keys; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.keys (
    id character varying(255) NOT NULL,
    user_id uuid NOT NULL,
    hashed_password character varying(255),
    provider_id character varying(255),
    provider_user_id character varying(255),
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.keys OWNER TO postgres;

--
-- Name: legal_analysis_sessions; Type: TABLE; Schema: public; Owner: legal_admin
--

CREATE TABLE public.legal_analysis_sessions (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    case_id uuid,
    user_id uuid,
    session_type character varying(50) DEFAULT 'case_analysis'::character varying,
    analysis_prompt text,
    analysis_result text,
    confidence_level numeric(3,2),
    sources_used jsonb DEFAULT '[]'::jsonb NOT NULL,
    model character varying(100) DEFAULT 'gemma3-legal'::character varying,
    processing_time integer,
    is_active boolean DEFAULT true,
    created_at timestamp without time zone DEFAULT now() NOT NULL,
    updated_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.legal_analysis_sessions OWNER TO legal_admin;

--
-- Name: legal_documents; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.legal_documents (
    id integer NOT NULL,
    filename character varying(255) NOT NULL,
    original_path text,
    s3_bucket character varying(100),
    s3_key text,
    file_size bigint,
    mime_type character varying(100),
    upload_date timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    document_type character varying(50),
    title text,
    content_preview text,
    full_text text,
    metadata jsonb,
    processing_status character varying(20) DEFAULT 'uploaded'::character varying,
    error_message text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.legal_documents OWNER TO postgres;

--
-- Name: legal_documents_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.legal_documents_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.legal_documents_id_seq OWNER TO postgres;

--
-- Name: legal_documents_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.legal_documents_id_seq OWNED BY public.legal_documents.id;


--
-- Name: persons_of_interest; Type: TABLE; Schema: public; Owner: legal_admin
--

CREATE TABLE public.persons_of_interest (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    case_id uuid,
    name character varying(255) NOT NULL,
    aliases jsonb DEFAULT '[]'::jsonb NOT NULL,
    relationship character varying(100),
    threat_level character varying(20) DEFAULT 'low'::character varying,
    status character varying(20) DEFAULT 'active'::character varying,
    profile_data jsonb DEFAULT '{}'::jsonb NOT NULL,
    tags jsonb DEFAULT '[]'::jsonb NOT NULL,
    "position" jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_by uuid,
    created_at timestamp without time zone DEFAULT now() NOT NULL,
    updated_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.persons_of_interest OWNER TO legal_admin;

--
-- Name: rag_messages; Type: TABLE; Schema: public; Owner: legal_admin
--

CREATE TABLE public.rag_messages (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    session_id character varying(255) NOT NULL,
    message_index integer NOT NULL,
    role character varying(20) NOT NULL,
    content text NOT NULL,
    retrieved_sources jsonb DEFAULT '[]'::jsonb NOT NULL,
    source_count integer DEFAULT 0 NOT NULL,
    retrieval_score character varying(10),
    processing_time integer,
    model character varying(100),
    created_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.rag_messages OWNER TO legal_admin;

--
-- Name: rag_sessions; Type: TABLE; Schema: public; Owner: legal_admin
--

CREATE TABLE public.rag_sessions (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    session_id character varying(255) NOT NULL,
    user_id uuid,
    title character varying(255),
    model character varying(100),
    is_active boolean DEFAULT true NOT NULL,
    created_at timestamp without time zone DEFAULT now() NOT NULL,
    updated_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.rag_sessions OWNER TO legal_admin;

--
-- Name: reports; Type: TABLE; Schema: public; Owner: legal_admin
--

CREATE TABLE public.reports (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    case_id uuid,
    title character varying(255) NOT NULL,
    content text,
    report_type character varying(50) DEFAULT 'case_summary'::character varying,
    status character varying(50) DEFAULT 'draft'::character varying NOT NULL,
    is_public boolean DEFAULT false,
    tags jsonb DEFAULT '[]'::jsonb NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_by uuid,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.reports OWNER TO legal_admin;

--
-- Name: search_cache; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.search_cache (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    query_hash character varying(64) NOT NULL,
    query_text text NOT NULL,
    results jsonb NOT NULL,
    expires_at timestamp without time zone NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.search_cache OWNER TO postgres;

--
-- Name: sessions; Type: TABLE; Schema: public; Owner: legal_admin
--

CREATE TABLE public.sessions (
    id character varying(255) NOT NULL,
    user_id uuid NOT NULL,
    expires_at timestamp with time zone NOT NULL,
    ip_address character varying(45),
    user_agent text,
    session_context jsonb DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.sessions OWNER TO legal_admin;

--
-- Name: statutes; Type: TABLE; Schema: public; Owner: legal_admin
--

CREATE TABLE public.statutes (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    title character varying(255) NOT NULL,
    code character varying(100) NOT NULL,
    description text,
    category character varying(100),
    jurisdiction character varying(100),
    is_active boolean DEFAULT true,
    penalties jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp without time zone DEFAULT now() NOT NULL,
    updated_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.statutes OWNER TO legal_admin;

--
-- Name: user_ai_queries; Type: TABLE; Schema: public; Owner: legal_admin
--

CREATE TABLE public.user_ai_queries (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid NOT NULL,
    case_id uuid,
    query text NOT NULL,
    response text NOT NULL,
    model character varying(100) DEFAULT 'gemma3-legal'::character varying NOT NULL,
    query_type character varying(50) DEFAULT 'general'::character varying,
    confidence numeric(3,2),
    tokens_used integer,
    processing_time integer,
    context_used jsonb DEFAULT '[]'::jsonb NOT NULL,
    embedding public.vector(768),
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    is_successful boolean DEFAULT true NOT NULL,
    error_message text,
    created_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.user_ai_queries OWNER TO legal_admin;

--
-- Name: user_profiles; Type: TABLE; Schema: public; Owner: legal_admin
--

CREATE TABLE public.user_profiles (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid NOT NULL,
    bio text,
    phone character varying(20),
    address text,
    preferences jsonb DEFAULT '{}'::jsonb NOT NULL,
    permissions jsonb DEFAULT '[]'::jsonb NOT NULL,
    specializations jsonb DEFAULT '[]'::jsonb NOT NULL,
    certifications jsonb DEFAULT '[]'::jsonb NOT NULL,
    experience_level character varying(20) DEFAULT 'junior'::character varying,
    work_patterns jsonb DEFAULT '{}'::jsonb NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp without time zone DEFAULT now() NOT NULL,
    updated_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.user_profiles OWNER TO legal_admin;

--
-- Name: user_sessions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.user_sessions (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    user_id uuid,
    session_token character varying(255) NOT NULL,
    expires_at timestamp without time zone NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    ip_address inet,
    user_agent text
);


ALTER TABLE public.user_sessions OWNER TO postgres;

--
-- Name: users; Type: TABLE; Schema: public; Owner: legal_admin
--

CREATE TABLE public.users (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    email character varying(255) NOT NULL,
    hashed_password character varying(255),
    username character varying(100),
    first_name character varying(100),
    last_name character varying(100),
    role character varying(50) DEFAULT 'user'::character varying NOT NULL,
    department character varying(100),
    jurisdiction character varying(100),
    permissions jsonb DEFAULT '[]'::jsonb NOT NULL,
    is_active boolean DEFAULT true NOT NULL,
    email_verified boolean DEFAULT false NOT NULL,
    avatar_url character varying(500),
    last_login_at timestamp with time zone,
    practice_areas jsonb DEFAULT '[]'::jsonb,
    bar_number character varying(50),
    firm_name character varying(200),
    profile_embedding public.vector(384),
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    deleted_at timestamp with time zone
);


ALTER TABLE public.users OWNER TO legal_admin;

--
-- Name: vector_metadata; Type: TABLE; Schema: public; Owner: legal_admin
--

CREATE TABLE public.vector_metadata (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    document_id uuid NOT NULL,
    collection_name character varying(100) NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    content_hash text NOT NULL,
    created_at timestamp without time zone DEFAULT now() NOT NULL,
    updated_at timestamp without time zone DEFAULT now()
);


ALTER TABLE public.vector_metadata OWNER TO legal_admin;

--
-- Name: __drizzle_migrations__ id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.__drizzle_migrations__ ALTER COLUMN id SET DEFAULT nextval('public.__drizzle_migrations___id_seq'::regclass);


--
-- Name: legal_documents id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.legal_documents ALTER COLUMN id SET DEFAULT nextval('public.legal_documents_id_seq'::regclass);


--
-- Name: __drizzle_migrations__ __drizzle_migrations___pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.__drizzle_migrations__
    ADD CONSTRAINT __drizzle_migrations___pkey PRIMARY KEY (id);


--
-- Name: activity_logs activity_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.activity_logs
    ADD CONSTRAINT activity_logs_pkey PRIMARY KEY (id);


--
-- Name: auto_tags auto_tags_pkey; Type: CONSTRAINT; Schema: public; Owner: legal_admin
--

ALTER TABLE ONLY public.auto_tags
    ADD CONSTRAINT auto_tags_pkey PRIMARY KEY (id);


--
-- Name: canvas_states canvas_states_pkey; Type: CONSTRAINT; Schema: public; Owner: legal_admin
--

ALTER TABLE ONLY public.canvas_states
    ADD CONSTRAINT canvas_states_pkey PRIMARY KEY (id);


--
-- Name: case_scores case_scores_pkey; Type: CONSTRAINT; Schema: public; Owner: legal_admin
--

ALTER TABLE ONLY public.case_scores
    ADD CONSTRAINT case_scores_pkey PRIMARY KEY (id);


--
-- Name: cases cases_pkey; Type: CONSTRAINT; Schema: public; Owner: legal_admin
--

ALTER TABLE ONLY public.cases
    ADD CONSTRAINT cases_pkey PRIMARY KEY (id);


--
-- Name: citations citations_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.citations
    ADD CONSTRAINT citations_pkey PRIMARY KEY (id);


--
-- Name: criminals criminals_pkey; Type: CONSTRAINT; Schema: public; Owner: legal_admin
--

ALTER TABLE ONLY public.criminals
    ADD CONSTRAINT criminals_pkey PRIMARY KEY (id);


--
-- Name: document_chunks document_chunks_pkey; Type: CONSTRAINT; Schema: public; Owner: legal_admin
--

ALTER TABLE ONLY public.document_chunks
    ADD CONSTRAINT document_chunks_pkey PRIMARY KEY (id);


--
-- Name: document_vectors document_vectors_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.document_vectors
    ADD CONSTRAINT document_vectors_pkey PRIMARY KEY (id);


--
-- Name: embedding_cache embedding_cache_pkey; Type: CONSTRAINT; Schema: public; Owner: legal_admin
--

ALTER TABLE ONLY public.embedding_cache
    ADD CONSTRAINT embedding_cache_pkey PRIMARY KEY (id);


--
-- Name: evidence evidence_pkey; Type: CONSTRAINT; Schema: public; Owner: legal_admin
--

ALTER TABLE ONLY public.evidence
    ADD CONSTRAINT evidence_pkey PRIMARY KEY (id);


--
-- Name: keys keys_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.keys
    ADD CONSTRAINT keys_pkey PRIMARY KEY (id);


--
-- Name: legal_analysis_sessions legal_analysis_sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: legal_admin
--

ALTER TABLE ONLY public.legal_analysis_sessions
    ADD CONSTRAINT legal_analysis_sessions_pkey PRIMARY KEY (id);


--
-- Name: legal_documents legal_documents_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.legal_documents
    ADD CONSTRAINT legal_documents_pkey PRIMARY KEY (id);


--
-- Name: persons_of_interest persons_of_interest_pkey; Type: CONSTRAINT; Schema: public; Owner: legal_admin
--

ALTER TABLE ONLY public.persons_of_interest
    ADD CONSTRAINT persons_of_interest_pkey PRIMARY KEY (id);


--
-- Name: rag_messages rag_messages_pkey; Type: CONSTRAINT; Schema: public; Owner: legal_admin
--

ALTER TABLE ONLY public.rag_messages
    ADD CONSTRAINT rag_messages_pkey PRIMARY KEY (id);


--
-- Name: rag_sessions rag_sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: legal_admin
--

ALTER TABLE ONLY public.rag_sessions
    ADD CONSTRAINT rag_sessions_pkey PRIMARY KEY (id);


--
-- Name: reports reports_pkey; Type: CONSTRAINT; Schema: public; Owner: legal_admin
--

ALTER TABLE ONLY public.reports
    ADD CONSTRAINT reports_pkey PRIMARY KEY (id);


--
-- Name: search_cache search_cache_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.search_cache
    ADD CONSTRAINT search_cache_pkey PRIMARY KEY (id);


--
-- Name: search_cache search_cache_query_hash_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.search_cache
    ADD CONSTRAINT search_cache_query_hash_key UNIQUE (query_hash);


--
-- Name: sessions sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: legal_admin
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_pkey PRIMARY KEY (id);


--
-- Name: statutes statutes_pkey; Type: CONSTRAINT; Schema: public; Owner: legal_admin
--

ALTER TABLE ONLY public.statutes
    ADD CONSTRAINT statutes_pkey PRIMARY KEY (id);


--
-- Name: user_ai_queries user_ai_queries_pkey; Type: CONSTRAINT; Schema: public; Owner: legal_admin
--

ALTER TABLE ONLY public.user_ai_queries
    ADD CONSTRAINT user_ai_queries_pkey PRIMARY KEY (id);


--
-- Name: user_profiles user_profiles_pkey; Type: CONSTRAINT; Schema: public; Owner: legal_admin
--

ALTER TABLE ONLY public.user_profiles
    ADD CONSTRAINT user_profiles_pkey PRIMARY KEY (id);


--
-- Name: user_sessions user_sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_sessions
    ADD CONSTRAINT user_sessions_pkey PRIMARY KEY (id);


--
-- Name: user_sessions user_sessions_session_token_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_sessions
    ADD CONSTRAINT user_sessions_session_token_key UNIQUE (session_token);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: legal_admin
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: vector_metadata vector_metadata_pkey; Type: CONSTRAINT; Schema: public; Owner: legal_admin
--

ALTER TABLE ONLY public.vector_metadata
    ADD CONSTRAINT vector_metadata_pkey PRIMARY KEY (id);


--
-- Name: idx_activity_logs_action; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_activity_logs_action ON public.activity_logs USING btree (action);


--
-- Name: idx_activity_logs_created_at; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_activity_logs_created_at ON public.activity_logs USING btree (created_at);


--
-- Name: idx_activity_logs_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_activity_logs_user_id ON public.activity_logs USING btree (user_id);


--
-- Name: idx_cases_created_at; Type: INDEX; Schema: public; Owner: legal_admin
--

CREATE INDEX idx_cases_created_at ON public.cases USING btree (created_at);


--
-- Name: idx_cases_description_fts; Type: INDEX; Schema: public; Owner: legal_admin
--

CREATE INDEX idx_cases_description_fts ON public.cases USING gin (to_tsvector('english'::regconfig, description));


--
-- Name: idx_cases_metadata; Type: INDEX; Schema: public; Owner: legal_admin
--

CREATE INDEX idx_cases_metadata ON public.cases USING gin (metadata);


--
-- Name: idx_cases_status; Type: INDEX; Schema: public; Owner: legal_admin
--

CREATE INDEX idx_cases_status ON public.cases USING btree (status);


--
-- Name: idx_cases_title_fts; Type: INDEX; Schema: public; Owner: legal_admin
--

CREATE INDEX idx_cases_title_fts ON public.cases USING gin (to_tsvector('english'::regconfig, (title)::text));


--
-- Name: idx_citations_case_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_citations_case_id ON public.citations USING btree (case_id);


--
-- Name: idx_citations_relevance; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_citations_relevance ON public.citations USING btree (relevance_score);


--
-- Name: idx_citations_text_fts; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_citations_text_fts ON public.citations USING gin (to_tsvector('english'::regconfig, citation_text));


--
-- Name: idx_citations_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_citations_type ON public.citations USING btree (citation_type);


--
-- Name: idx_document_vectors_document_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_document_vectors_document_id ON public.document_vectors USING btree (document_id);


--
-- Name: idx_document_vectors_embedding; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_document_vectors_embedding ON public.document_vectors USING ivfflat (embedding public.vector_cosine_ops);


--
-- Name: idx_document_vectors_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_document_vectors_type ON public.document_vectors USING btree (document_type);


--
-- Name: idx_evidence_case_id; Type: INDEX; Schema: public; Owner: legal_admin
--

CREATE INDEX idx_evidence_case_id ON public.evidence USING btree (case_id);


--
-- Name: idx_evidence_tags; Type: INDEX; Schema: public; Owner: legal_admin
--

CREATE INDEX idx_evidence_tags ON public.evidence USING gin (tags);


--
-- Name: idx_evidence_title_fts; Type: INDEX; Schema: public; Owner: legal_admin
--

CREATE INDEX idx_evidence_title_fts ON public.evidence USING gin (to_tsvector('english'::regconfig, (title)::text));


--
-- Name: idx_evidence_type; Type: INDEX; Schema: public; Owner: legal_admin
--

CREATE INDEX idx_evidence_type ON public.evidence USING btree (evidence_type);


--
-- Name: idx_reports_case_id; Type: INDEX; Schema: public; Owner: legal_admin
--

CREATE INDEX idx_reports_case_id ON public.reports USING btree (case_id);


--
-- Name: idx_reports_type; Type: INDEX; Schema: public; Owner: legal_admin
--

CREATE INDEX idx_reports_type ON public.reports USING btree (report_type);


--
-- Name: idx_user_sessions_expires; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_user_sessions_expires ON public.user_sessions USING btree (expires_at);


--
-- Name: idx_user_sessions_token; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_user_sessions_token ON public.user_sessions USING btree (session_token);


--
-- Name: idx_user_sessions_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_user_sessions_user_id ON public.user_sessions USING btree (user_id);


--
-- Name: sessions_expires_at_idx; Type: INDEX; Schema: public; Owner: legal_admin
--

CREATE INDEX sessions_expires_at_idx ON public.sessions USING btree (expires_at);


--
-- Name: sessions_user_id_idx; Type: INDEX; Schema: public; Owner: legal_admin
--

CREATE INDEX sessions_user_id_idx ON public.sessions USING btree (user_id);


--
-- Name: users_active_idx; Type: INDEX; Schema: public; Owner: legal_admin
--

CREATE INDEX users_active_idx ON public.users USING btree (is_active);


--
-- Name: users_email_idx; Type: INDEX; Schema: public; Owner: legal_admin
--

CREATE INDEX users_email_idx ON public.users USING btree (email);


--
-- Name: users_role_idx; Type: INDEX; Schema: public; Owner: legal_admin
--

CREATE INDEX users_role_idx ON public.users USING btree (role);


--
-- Name: users_username_idx; Type: INDEX; Schema: public; Owner: legal_admin
--

CREATE INDEX users_username_idx ON public.users USING btree (username);


--
-- Name: vector_metadata_document_id_idx; Type: INDEX; Schema: public; Owner: legal_admin
--

CREATE INDEX vector_metadata_document_id_idx ON public.vector_metadata USING btree (document_id);


--
-- Name: evidence evidence_vector_insert; Type: TRIGGER; Schema: public; Owner: legal_admin
--

CREATE TRIGGER evidence_vector_insert AFTER INSERT ON public.evidence FOR EACH ROW EXECUTE FUNCTION public.create_vector_for_evidence();


--
-- Name: reports report_vector_insert; Type: TRIGGER; Schema: public; Owner: legal_admin
--

CREATE TRIGGER report_vector_insert AFTER INSERT ON public.reports FOR EACH ROW EXECUTE FUNCTION public.create_vector_for_report();


--
-- Name: reports reports_notify_changes; Type: TRIGGER; Schema: public; Owner: legal_admin
--

CREATE TRIGGER reports_notify_changes AFTER INSERT OR DELETE OR UPDATE ON public.reports FOR EACH ROW EXECUTE FUNCTION public.notify_report_changes();


--
-- Name: cases update_cases_updated_at; Type: TRIGGER; Schema: public; Owner: legal_admin
--

CREATE TRIGGER update_cases_updated_at BEFORE UPDATE ON public.cases FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: citations update_citations_updated_at; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER update_citations_updated_at BEFORE UPDATE ON public.citations FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: evidence update_evidence_updated_at; Type: TRIGGER; Schema: public; Owner: legal_admin
--

CREATE TRIGGER update_evidence_updated_at BEFORE UPDATE ON public.evidence FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: reports update_reports_updated_at; Type: TRIGGER; Schema: public; Owner: legal_admin
--

CREATE TRIGGER update_reports_updated_at BEFORE UPDATE ON public.reports FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: users update_users_updated_at; Type: TRIGGER; Schema: public; Owner: legal_admin
--

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON public.users FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: activity_logs activity_logs_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.activity_logs
    ADD CONSTRAINT activity_logs_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: citations citations_case_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.citations
    ADD CONSTRAINT citations_case_id_fkey FOREIGN KEY (case_id) REFERENCES public.cases(id) ON DELETE CASCADE;


--
-- Name: keys keys_user_id_users_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.keys
    ADD CONSTRAINT keys_user_id_users_id_fk FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: user_sessions user_sessions_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_sessions
    ADD CONSTRAINT user_sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: pg_database_owner
--

GRANT ALL ON SCHEMA public TO legal_admin;


--
-- Name: FUNCTION cosine_similarity(a public.vector, b public.vector); Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON FUNCTION public.cosine_similarity(a public.vector, b public.vector) TO postgres;


--
-- Name: FUNCTION get_similar_documents(query_embedding public.vector, similarity_threshold real, result_limit integer); Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON FUNCTION public.get_similar_documents(query_embedding public.vector, similarity_threshold real, result_limit integer) TO postgres;


--
-- Name: FUNCTION update_processing_stats(); Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON FUNCTION public.update_processing_stats() TO postgres;


--
-- Name: FUNCTION update_updated_at_column(); Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON FUNCTION public.update_updated_at_column() TO postgres;


--
-- Name: TABLE __drizzle_migrations__; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.__drizzle_migrations__ TO legal_admin;


--
-- Name: SEQUENCE __drizzle_migrations___id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.__drizzle_migrations___id_seq TO legal_admin;


--
-- Name: TABLE activity_logs; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.activity_logs TO legal_admin;


--
-- Name: TABLE auto_tags; Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON TABLE public.auto_tags TO postgres;


--
-- Name: TABLE canvas_states; Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON TABLE public.canvas_states TO postgres;


--
-- Name: TABLE case_scores; Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON TABLE public.case_scores TO postgres;


--
-- Name: TABLE cases; Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON TABLE public.cases TO postgres;


--
-- Name: TABLE citations; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.citations TO legal_admin;


--
-- Name: TABLE criminals; Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON TABLE public.criminals TO postgres;


--
-- Name: TABLE document_chunks; Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON TABLE public.document_chunks TO postgres;


--
-- Name: TABLE document_vectors; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.document_vectors TO legal_admin;


--
-- Name: TABLE embedding_cache; Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON TABLE public.embedding_cache TO postgres;


--
-- Name: TABLE evidence; Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON TABLE public.evidence TO postgres;


--
-- Name: TABLE keys; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.keys TO legal_admin;


--
-- Name: TABLE legal_analysis_sessions; Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON TABLE public.legal_analysis_sessions TO postgres;


--
-- Name: TABLE legal_documents; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.legal_documents TO legal_admin;


--
-- Name: SEQUENCE legal_documents_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.legal_documents_id_seq TO legal_admin;


--
-- Name: TABLE persons_of_interest; Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON TABLE public.persons_of_interest TO postgres;


--
-- Name: TABLE rag_messages; Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON TABLE public.rag_messages TO postgres;


--
-- Name: TABLE rag_sessions; Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON TABLE public.rag_sessions TO postgres;


--
-- Name: TABLE reports; Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON TABLE public.reports TO postgres;


--
-- Name: TABLE search_cache; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.search_cache TO legal_admin;


--
-- Name: TABLE sessions; Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON TABLE public.sessions TO postgres;


--
-- Name: TABLE statutes; Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON TABLE public.statutes TO postgres;


--
-- Name: TABLE user_ai_queries; Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON TABLE public.user_ai_queries TO postgres;


--
-- Name: TABLE user_profiles; Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON TABLE public.user_profiles TO postgres;


--
-- Name: TABLE user_sessions; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.user_sessions TO legal_admin;


--
-- Name: TABLE users; Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON TABLE public.users TO postgres;


--
-- Name: TABLE vector_metadata; Type: ACL; Schema: public; Owner: legal_admin
--

GRANT ALL ON TABLE public.vector_metadata TO postgres;


--
-- Name: DEFAULT PRIVILEGES FOR SEQUENCES; Type: DEFAULT ACL; Schema: public; Owner: postgres
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON SEQUENCES TO legal_admin;


--
-- Name: DEFAULT PRIVILEGES FOR TABLES; Type: DEFAULT ACL; Schema: public; Owner: postgres
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT SELECT,INSERT,DELETE,UPDATE ON TABLES TO legal_admin;


--
-- PostgreSQL database dump complete
--

