-- Add Report Builder tables for Case Books functionality

-- Reports table for the core report/document management
CREATE TABLE "reports" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"title" varchar(255) NOT NULL,
	"content" text,
	"summary" text,
	"case_id" uuid NOT NULL,
	"report_type" varchar(50) DEFAULT 'prosecution_memo' NOT NULL,
	"status" varchar(20) DEFAULT 'draft' NOT NULL,
	"confidentiality_level" varchar(20) DEFAULT 'restricted' NOT NULL,
	"jurisdiction" varchar(100),
	"tags" jsonb DEFAULT '[]' NOT NULL,
	"metadata" jsonb DEFAULT '{}' NOT NULL,
	"sections" jsonb DEFAULT '[]' NOT NULL,
	"ai_summary" text,
	"ai_tags" jsonb DEFAULT '[]' NOT NULL,
	"word_count" integer DEFAULT 0 NOT NULL,
	"estimated_read_time" integer DEFAULT 0 NOT NULL,
	"template_id" uuid,
	"created_by" uuid NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL,
	"last_edited_by" uuid,
	"published_at" timestamp,
	"archived_at" timestamp
);

-- Citation Points table for managing legal references and citations
CREATE TABLE "citation_points" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"text" text NOT NULL,
	"source" varchar(500) NOT NULL,
	"page" integer,
	"context" text,
	"type" varchar(50) DEFAULT 'statute' NOT NULL,
	"jurisdiction" varchar(100),
	"tags" jsonb DEFAULT '[]' NOT NULL,
	"case_id" uuid,
	"report_id" uuid,
	"evidence_id" uuid,
	"statute_id" uuid,
	"ai_summary" text,
	"relevance_score" numeric(4, 3) DEFAULT '0.0',
	"metadata" jsonb DEFAULT '{}' NOT NULL,
	"is_bookmarked" boolean DEFAULT false NOT NULL,
	"usage_count" integer DEFAULT 0 NOT NULL,
	"created_by" uuid NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);

-- Canvas States table for storing Fabric.js interactive canvas data
CREATE TABLE "canvas_states" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"title" varchar(255),
	"report_id" uuid NOT NULL,
	"canvas_data" jsonb NOT NULL,
	"thumbnail_url" text,
	"dimensions" jsonb DEFAULT '{"width": 800, "height": 600}' NOT NULL,
	"background_color" varchar(20) DEFAULT '#ffffff',
	"metadata" jsonb DEFAULT '{}' NOT NULL,
	"version" integer DEFAULT 1 NOT NULL,
	"is_template" boolean DEFAULT false NOT NULL,
	"created_by" uuid NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);

-- AI Analyses table for storing AI-generated insights and suggestions
CREATE TABLE IF NOT EXISTS "ai_analyses" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"report_id" uuid,
	"citation_id" uuid,
	"case_id" uuid,
	"analysis_type" varchar(50) NOT NULL,
	"input_data" jsonb NOT NULL,
	"result" jsonb NOT NULL,
	"model" varchar(100),
	"tokens" integer DEFAULT 0,
	"processing_time" integer DEFAULT 0,
	"confidence" numeric(4, 3) DEFAULT '0.0',
	"status" varchar(20) DEFAULT 'completed' NOT NULL,
	"error_message" text,
	"created_by" uuid NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL
);

-- Report Templates table for standardized report formats
CREATE TABLE "report_templates" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"name" varchar(255) NOT NULL,
	"description" text,
	"template_content" text NOT NULL,
	"report_type" varchar(50) NOT NULL,
	"jurisdiction" varchar(100),
	"is_default" boolean DEFAULT false NOT NULL,
	"is_active" boolean DEFAULT true NOT NULL,
	"metadata" jsonb DEFAULT '{}' NOT NULL,
	"usage_count" integer DEFAULT 0 NOT NULL,
	"created_by" uuid NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);

-- Export History table for tracking document exports
CREATE TABLE "export_history" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"report_id" uuid NOT NULL,
	"export_format" varchar(20) NOT NULL,
	"file_name" varchar(500) NOT NULL,
	"file_size" integer,
	"download_url" text,
	"status" varchar(20) DEFAULT 'pending' NOT NULL,
	"options" jsonb DEFAULT '{}' NOT NULL,
	"error_message" text,
	"expires_at" timestamp,
	"download_count" integer DEFAULT 0 NOT NULL,
	"created_by" uuid NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL
);

-- Add foreign key constraints
ALTER TABLE "reports" ADD CONSTRAINT "reports_case_id_cases_id_fk" FOREIGN KEY ("case_id") REFERENCES "cases"("id") ON DELETE cascade;
ALTER TABLE "reports" ADD CONSTRAINT "reports_created_by_users_id_fk" FOREIGN KEY ("created_by") REFERENCES "users"("id");
ALTER TABLE "reports" ADD CONSTRAINT "reports_last_edited_by_users_id_fk" FOREIGN KEY ("last_edited_by") REFERENCES "users"("id");

ALTER TABLE "citation_points" ADD CONSTRAINT "citation_points_case_id_cases_id_fk" FOREIGN KEY ("case_id") REFERENCES "cases"("id") ON DELETE cascade;
ALTER TABLE "citation_points" ADD CONSTRAINT "citation_points_report_id_reports_id_fk" FOREIGN KEY ("report_id") REFERENCES "reports"("id") ON DELETE cascade;
ALTER TABLE "citation_points" ADD CONSTRAINT "citation_points_evidence_id_evidence_id_fk" FOREIGN KEY ("evidence_id") REFERENCES "evidence"("id") ON DELETE set null;
ALTER TABLE "citation_points" ADD CONSTRAINT "citation_points_statute_id_statutes_id_fk" FOREIGN KEY ("statute_id") REFERENCES "statutes"("id") ON DELETE set null;
ALTER TABLE "citation_points" ADD CONSTRAINT "citation_points_created_by_users_id_fk" FOREIGN KEY ("created_by") REFERENCES "users"("id");

ALTER TABLE "canvas_states" ADD CONSTRAINT "canvas_states_report_id_reports_id_fk" FOREIGN KEY ("report_id") REFERENCES "reports"("id") ON DELETE cascade;
ALTER TABLE "canvas_states" ADD CONSTRAINT "canvas_states_created_by_users_id_fk" FOREIGN KEY ("created_by") REFERENCES "users"("id");

ALTER TABLE "ai_analyses" ADD CONSTRAINT "ai_analyses_report_id_reports_id_fk" FOREIGN KEY ("report_id") REFERENCES "reports"("id") ON DELETE cascade;
ALTER TABLE "ai_analyses" ADD CONSTRAINT "ai_analyses_citation_id_citation_points_id_fk" FOREIGN KEY ("citation_id") REFERENCES "citation_points"("id") ON DELETE cascade;
ALTER TABLE "ai_analyses" ADD CONSTRAINT "ai_analyses_case_id_cases_id_fk" FOREIGN KEY ("case_id") REFERENCES "cases"("id") ON DELETE cascade;
ALTER TABLE "ai_analyses" ADD CONSTRAINT "ai_analyses_created_by_users_id_fk" FOREIGN KEY ("created_by") REFERENCES "users"("id");

ALTER TABLE "report_templates" ADD CONSTRAINT "report_templates_created_by_users_id_fk" FOREIGN KEY ("created_by") REFERENCES "users"("id");

ALTER TABLE "export_history" ADD CONSTRAINT "export_history_report_id_reports_id_fk" FOREIGN KEY ("report_id") REFERENCES "reports"("id") ON DELETE cascade;
ALTER TABLE "export_history" ADD CONSTRAINT "export_history_created_by_users_id_fk" FOREIGN KEY ("created_by") REFERENCES "users"("id");

-- Create indexes for performance
CREATE INDEX "idx_reports_case_id" ON "reports"("case_id");
CREATE INDEX "idx_reports_created_by" ON "reports"("created_by");
CREATE INDEX "idx_reports_status" ON "reports"("status");
CREATE INDEX "idx_reports_report_type" ON "reports"("report_type");
CREATE INDEX "idx_reports_created_at" ON "reports"("created_at");

CREATE INDEX "idx_citation_points_case_id" ON "citation_points"("case_id");
CREATE INDEX "idx_citation_points_report_id" ON "citation_points"("report_id");
CREATE INDEX "idx_citation_points_type" ON "citation_points"("type");
CREATE INDEX "idx_citation_points_relevance_score" ON "citation_points"("relevance_score");
CREATE INDEX "idx_citation_points_created_by" ON "citation_points"("created_by");
CREATE INDEX "idx_citation_points_is_bookmarked" ON "citation_points"("is_bookmarked");

CREATE INDEX "idx_canvas_states_report_id" ON "canvas_states"("report_id");
CREATE INDEX "idx_canvas_states_created_by" ON "canvas_states"("created_by");

CREATE INDEX "idx_ai_analyses_report_id" ON "ai_analyses"("report_id");
CREATE INDEX "idx_ai_analyses_case_id" ON "ai_analyses"("case_id");
CREATE INDEX "idx_ai_analyses_analysis_type" ON "ai_analyses"("analysis_type");
CREATE INDEX "idx_ai_analyses_created_at" ON "ai_analyses"("created_at");

CREATE INDEX "idx_export_history_report_id" ON "export_history"("report_id");
CREATE INDEX "idx_export_history_created_by" ON "export_history"("created_by");
CREATE INDEX "idx_export_history_status" ON "export_history"("status");

-- Insert sample data for testing
INSERT INTO "report_templates" ("name", "description", "template_content", "report_type", "is_default", "created_by") VALUES
('Standard Prosecution Memo', 'Standard template for prosecution memorandums', '<h1>Prosecution Memorandum</h1><h2>Case Summary</h2><p>[CASE_SUMMARY]</p><h2>Legal Analysis</h2><p>[LEGAL_ANALYSIS]</p><h2>Evidence Review</h2><p>[EVIDENCE_REVIEW]</p><h2>Recommendation</h2><p>[RECOMMENDATION]</p>', 'prosecution_memo', true, (SELECT id FROM users LIMIT 1)),
('Case Brief Template', 'Template for legal case briefs', '<h1>Case Brief</h1><h2>Facts</h2><p>[FACTS]</p><h2>Issues</h2><p>[ISSUES]</p><h2>Analysis</h2><p>[ANALYSIS]</p><h2>Conclusion</h2><p>[CONCLUSION]</p>', 'case_brief', true, (SELECT id FROM users LIMIT 1)),
('Evidence Summary', 'Template for evidence summaries', '<h1>Evidence Summary</h1><h2>Physical Evidence</h2><p>[PHYSICAL_EVIDENCE]</p><h2>Witness Testimony</h2><p>[WITNESS_TESTIMONY]</p><h2>Documentary Evidence</h2><p>[DOCUMENTARY_EVIDENCE]</p>', 'evidence_summary', true, (SELECT id FROM users LIMIT 1));

-- Add sample citation points for testing
INSERT INTO "citation_points" ("text", "source", "type", "context", "created_by") VALUES
('Criminal acts that constitute felonies under state law', 'Penal Code Section 17(a)', 'statute', 'Defines the classification of criminal offenses', (SELECT id FROM users LIMIT 1)),
('The prosecution must prove guilt beyond a reasonable doubt', 'People v. Doe (2020) 123 Cal.App.4th 456', 'case_law', 'Standard of proof in criminal cases', (SELECT id FROM users LIMIT 1)),
('Physical evidence must be properly documented and preserved', 'Evidence Code Section 1400', 'statute', 'Chain of custody requirements', (SELECT id FROM users LIMIT 1));
