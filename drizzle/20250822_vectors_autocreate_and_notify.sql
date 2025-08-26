-- 20250822_vectors_autocreate_and_notify.sql
-- Migration: Auto-create vectors on insert (zeros) + NOTIFY triggers

CREATE EXTENSION IF NOT EXISTS vector;

-- Function to create vector for evidence
CREATE OR REPLACE FUNCTION create_vector_for_evidence()
RETURNS TRIGGER AS $$
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
$$ LANGUAGE plpgsql;

-- Function to create vector for report
CREATE OR REPLACE FUNCTION create_vector_for_report()
RETURNS TRIGGER AS $$
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
$$ LANGUAGE plpgsql;

-- Function to notify on case changes
CREATE OR REPLACE FUNCTION notify_case_changes()
RETURNS TRIGGER AS $$
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
$$ LANGUAGE plpgsql;

-- Function to notify on report changes
CREATE OR REPLACE FUNCTION notify_report_changes()
RETURNS TRIGGER AS $$
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
$$ LANGUAGE plpgsql;

-- Drop existing triggers if they exist
DROP TRIGGER IF EXISTS evidence_vector_insert ON evidence;
DROP TRIGGER IF EXISTS report_vector_insert ON reports;
DROP TRIGGER IF EXISTS cases_notify_changes ON cases;
DROP TRIGGER IF EXISTS reports_notify_changes ON reports;

-- Create triggers for auto-vector creation
CREATE TRIGGER evidence_vector_insert
AFTER INSERT ON evidence
FOR EACH ROW EXECUTE FUNCTION create_vector_for_evidence();

CREATE TRIGGER report_vector_insert
AFTER INSERT ON reports
FOR EACH ROW EXECUTE FUNCTION create_vector_for_report();

-- Create triggers for NOTIFY
CREATE TRIGGER cases_notify_changes
AFTER INSERT OR UPDATE OR DELETE ON cases
FOR EACH ROW EXECUTE FUNCTION notify_case_changes();

CREATE TRIGGER reports_notify_changes
AFTER INSERT OR UPDATE OR DELETE ON reports
FOR EACH ROW EXECUTE FUNCTION notify_report_changes();