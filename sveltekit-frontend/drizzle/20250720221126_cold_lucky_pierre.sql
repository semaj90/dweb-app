CREATE TABLE "ai_history" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"case_id" uuid,
	"user_id" uuid NOT NULL,
	"prompt" text NOT NULL,
	"response" text NOT NULL,
	"model" text NOT NULL,
	"tokens_used" integer,
	"cost" integer,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"metadata" jsonb
);
--> statement-breakpoint
CREATE TABLE "collaboration_sessions" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"case_id" uuid NOT NULL,
	"user_id" uuid NOT NULL,
	"session_id" text NOT NULL,
	"is_active" boolean DEFAULT true,
	"last_activity" timestamp DEFAULT now() NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "documents" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"case_id" uuid,
	"evidence_id" uuid,
	"filename" text NOT NULL,
	"file_path" text NOT NULL,
	"extracted_text" text,
	"embeddings" vector(384),
	"analysis" jsonb,
	"created_by" uuid NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "notes" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"case_id" uuid NOT NULL,
	"evidence_id" uuid,
	"content" text NOT NULL,
	"is_private" boolean DEFAULT false,
	"created_by" uuid NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
ALTER TABLE "account" DISABLE ROW LEVEL SECURITY;--> statement-breakpoint
ALTER TABLE "ai_analyses" DISABLE ROW LEVEL SECURITY;--> statement-breakpoint
ALTER TABLE "canvas_states" DISABLE ROW LEVEL SECURITY;--> statement-breakpoint
ALTER TABLE "case_activities" DISABLE ROW LEVEL SECURITY;--> statement-breakpoint
ALTER TABLE "case_law_links" DISABLE ROW LEVEL SECURITY;--> statement-breakpoint
ALTER TABLE "citation_points" DISABLE ROW LEVEL SECURITY;--> statement-breakpoint
ALTER TABLE "content_embeddings" DISABLE ROW LEVEL SECURITY;--> statement-breakpoint
ALTER TABLE "crimes" DISABLE ROW LEVEL SECURITY;--> statement-breakpoint
ALTER TABLE "criminals" DISABLE ROW LEVEL SECURITY;--> statement-breakpoint
ALTER TABLE "export_history" DISABLE ROW LEVEL SECURITY;--> statement-breakpoint
ALTER TABLE "law_paragraphs" DISABLE ROW LEVEL SECURITY;--> statement-breakpoint
ALTER TABLE "report_templates" DISABLE ROW LEVEL SECURITY;--> statement-breakpoint
ALTER TABLE "reports" DISABLE ROW LEVEL SECURITY;--> statement-breakpoint
ALTER TABLE "sessions" DISABLE ROW LEVEL SECURITY;--> statement-breakpoint
ALTER TABLE "statutes" DISABLE ROW LEVEL SECURITY;--> statement-breakpoint
ALTER TABLE "verificationToken" DISABLE ROW LEVEL SECURITY;--> statement-breakpoint
DROP TABLE "account" CASCADE;--> statement-breakpoint
DROP TABLE "ai_analyses" CASCADE;--> statement-breakpoint
DROP TABLE "canvas_states" CASCADE;--> statement-breakpoint
DROP TABLE "case_activities" CASCADE;--> statement-breakpoint
DROP TABLE "case_law_links" CASCADE;--> statement-breakpoint
DROP TABLE "citation_points" CASCADE;--> statement-breakpoint
DROP TABLE "content_embeddings" CASCADE;--> statement-breakpoint
DROP TABLE "crimes" CASCADE;--> statement-breakpoint
DROP TABLE "criminals" CASCADE;--> statement-breakpoint
DROP TABLE "export_history" CASCADE;--> statement-breakpoint
DROP TABLE "law_paragraphs" CASCADE;--> statement-breakpoint
DROP TABLE "report_templates" CASCADE;--> statement-breakpoint
DROP TABLE "reports" CASCADE;--> statement-breakpoint
DROP TABLE "sessions" CASCADE;--> statement-breakpoint
DROP TABLE "statutes" CASCADE;--> statement-breakpoint
DROP TABLE "verificationToken" CASCADE;--> statement-breakpoint
ALTER TABLE "cases" DROP CONSTRAINT "cases_lead_prosecutor_users_id_fk";
--> statement-breakpoint
ALTER TABLE "evidence" DROP CONSTRAINT "evidence_criminal_id_criminals_id_fk";
--> statement-breakpoint
ALTER TABLE "evidence" DROP CONSTRAINT "evidence_uploaded_by_users_id_fk";
--> statement-breakpoint
ALTER TABLE "evidence" DROP CONSTRAINT "evidence_case_id_cases_id_fk";
--> statement-breakpoint
ALTER TABLE "cases" ALTER COLUMN "title" SET DATA TYPE text;--> statement-breakpoint
ALTER TABLE "cases" ALTER COLUMN "priority" SET DATA TYPE text;--> statement-breakpoint
ALTER TABLE "cases" ALTER COLUMN "priority" SET DEFAULT 'medium';--> statement-breakpoint
ALTER TABLE "cases" ALTER COLUMN "priority" DROP NOT NULL;--> statement-breakpoint
ALTER TABLE "cases" ALTER COLUMN "status" SET DATA TYPE text;--> statement-breakpoint
ALTER TABLE "cases" ALTER COLUMN "status" SET DEFAULT 'active';--> statement-breakpoint
ALTER TABLE "cases" ALTER COLUMN "metadata" DROP DEFAULT;--> statement-breakpoint
ALTER TABLE "cases" ALTER COLUMN "metadata" DROP NOT NULL;--> statement-breakpoint
ALTER TABLE "cases" ALTER COLUMN "created_by" SET NOT NULL;--> statement-breakpoint
ALTER TABLE "evidence" ALTER COLUMN "case_id" SET NOT NULL;--> statement-breakpoint
ALTER TABLE "evidence" ALTER COLUMN "title" SET DATA TYPE text;--> statement-breakpoint
ALTER TABLE "users" ALTER COLUMN "email" SET DATA TYPE text;--> statement-breakpoint
ALTER TABLE "users" ALTER COLUMN "role" SET DATA TYPE text;--> statement-breakpoint
ALTER TABLE "users" ALTER COLUMN "role" SET DEFAULT 'user';--> statement-breakpoint
ALTER TABLE "users" ALTER COLUMN "name" SET NOT NULL;--> statement-breakpoint
ALTER TABLE "cases" ADD COLUMN "assigned_to" uuid;--> statement-breakpoint
ALTER TABLE "evidence" ADD COLUMN "type" text NOT NULL;--> statement-breakpoint
ALTER TABLE "evidence" ADD COLUMN "content" text;--> statement-breakpoint
ALTER TABLE "evidence" ADD COLUMN "file_path" text;--> statement-breakpoint
ALTER TABLE "evidence" ADD COLUMN "mime_type" text;--> statement-breakpoint
ALTER TABLE "evidence" ADD COLUMN "hash" text;--> statement-breakpoint
ALTER TABLE "evidence" ADD COLUMN "created_by" uuid NOT NULL;--> statement-breakpoint
ALTER TABLE "evidence" ADD COLUMN "created_at" timestamp DEFAULT now() NOT NULL;--> statement-breakpoint
ALTER TABLE "evidence" ADD COLUMN "metadata" jsonb;--> statement-breakpoint
ALTER TABLE "users" ADD COLUMN "password_hash" text NOT NULL;--> statement-breakpoint
ALTER TABLE "users" ADD COLUMN "last_login" timestamp;--> statement-breakpoint
ALTER TABLE "ai_history" ADD CONSTRAINT "ai_history_case_id_cases_id_fk" FOREIGN KEY ("case_id") REFERENCES "public"."cases"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "ai_history" ADD CONSTRAINT "ai_history_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "collaboration_sessions" ADD CONSTRAINT "collaboration_sessions_case_id_cases_id_fk" FOREIGN KEY ("case_id") REFERENCES "public"."cases"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "collaboration_sessions" ADD CONSTRAINT "collaboration_sessions_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "documents" ADD CONSTRAINT "documents_case_id_cases_id_fk" FOREIGN KEY ("case_id") REFERENCES "public"."cases"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "documents" ADD CONSTRAINT "documents_evidence_id_evidence_id_fk" FOREIGN KEY ("evidence_id") REFERENCES "public"."evidence"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "documents" ADD CONSTRAINT "documents_created_by_users_id_fk" FOREIGN KEY ("created_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "notes" ADD CONSTRAINT "notes_case_id_cases_id_fk" FOREIGN KEY ("case_id") REFERENCES "public"."cases"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "notes" ADD CONSTRAINT "notes_evidence_id_evidence_id_fk" FOREIGN KEY ("evidence_id") REFERENCES "public"."evidence"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "notes" ADD CONSTRAINT "notes_created_by_users_id_fk" FOREIGN KEY ("created_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "cases" ADD CONSTRAINT "cases_assigned_to_users_id_fk" FOREIGN KEY ("assigned_to") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "evidence" ADD CONSTRAINT "evidence_created_by_users_id_fk" FOREIGN KEY ("created_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "evidence" ADD CONSTRAINT "evidence_case_id_cases_id_fk" FOREIGN KEY ("case_id") REFERENCES "public"."cases"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "cases" DROP COLUMN "case_number";--> statement-breakpoint
ALTER TABLE "cases" DROP COLUMN "incident_date";--> statement-breakpoint
ALTER TABLE "cases" DROP COLUMN "location";--> statement-breakpoint
ALTER TABLE "cases" DROP COLUMN "category";--> statement-breakpoint
ALTER TABLE "cases" DROP COLUMN "danger_score";--> statement-breakpoint
ALTER TABLE "cases" DROP COLUMN "estimated_value";--> statement-breakpoint
ALTER TABLE "cases" DROP COLUMN "jurisdiction";--> statement-breakpoint
ALTER TABLE "cases" DROP COLUMN "lead_prosecutor";--> statement-breakpoint
ALTER TABLE "cases" DROP COLUMN "assigned_team";--> statement-breakpoint
ALTER TABLE "cases" DROP COLUMN "ai_summary";--> statement-breakpoint
ALTER TABLE "cases" DROP COLUMN "ai_tags";--> statement-breakpoint
ALTER TABLE "cases" DROP COLUMN "closed_at";--> statement-breakpoint
ALTER TABLE "cases" DROP COLUMN "name";--> statement-breakpoint
ALTER TABLE "cases" DROP COLUMN "summary";--> statement-breakpoint
ALTER TABLE "cases" DROP COLUMN "date_opened";--> statement-breakpoint
ALTER TABLE "cases" DROP COLUMN "verdict";--> statement-breakpoint
ALTER TABLE "cases" DROP COLUMN "court_dates";--> statement-breakpoint
ALTER TABLE "cases" DROP COLUMN "linked_criminals";--> statement-breakpoint
ALTER TABLE "cases" DROP COLUMN "linked_crimes";--> statement-breakpoint
ALTER TABLE "cases" DROP COLUMN "notes";--> statement-breakpoint
ALTER TABLE "cases" DROP COLUMN "tags";--> statement-breakpoint
ALTER TABLE "evidence" DROP COLUMN "criminal_id";--> statement-breakpoint
ALTER TABLE "evidence" DROP COLUMN "file_url";--> statement-breakpoint
ALTER TABLE "evidence" DROP COLUMN "file_type";--> statement-breakpoint
ALTER TABLE "evidence" DROP COLUMN "tags";--> statement-breakpoint
ALTER TABLE "evidence" DROP COLUMN "uploaded_by";--> statement-breakpoint
ALTER TABLE "evidence" DROP COLUMN "uploaded_at";--> statement-breakpoint
ALTER TABLE "evidence" DROP COLUMN "updated_at";--> statement-breakpoint
ALTER TABLE "evidence" DROP COLUMN "file_name";--> statement-breakpoint
ALTER TABLE "evidence" DROP COLUMN "summary";--> statement-breakpoint
ALTER TABLE "evidence" DROP COLUMN "ai_summary";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "email_verified";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "hashed_password";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "is_active";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "first_name";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "last_name";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "title";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "department";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "phone";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "office_address";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "avatar";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "bio";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "specializations";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "username";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "image";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "avatar_url";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "provider";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "profile";