CREATE TABLE "user_profiles" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" uuid NOT NULL,
	"bio" text,
	"phone" varchar(20),
	"address" text,
	"preferences" jsonb DEFAULT '{}'::jsonb NOT NULL,
	"permissions" jsonb DEFAULT '[]'::jsonb NOT NULL,
	"specializations" jsonb DEFAULT '[]'::jsonb NOT NULL,
	"certifications" jsonb DEFAULT '[]'::jsonb NOT NULL,
	"experience_level" varchar(20) DEFAULT 'junior',
	"work_patterns" jsonb DEFAULT '{}'::jsonb NOT NULL,
	"metadata" jsonb DEFAULT '{}'::jsonb NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "user_profiles_user_id_unique" UNIQUE("user_id")
);
--> statement-breakpoint
ALTER TABLE "document_chunks" ALTER COLUMN "embedding" SET DATA TYPE text;--> statement-breakpoint
ALTER TABLE "embedding_cache" ALTER COLUMN "embedding" SET DATA TYPE text;--> statement-breakpoint
ALTER TABLE "legal_documents" ALTER COLUMN "embedding" SET DATA TYPE text;--> statement-breakpoint
ALTER TABLE "user_ai_queries" ALTER COLUMN "embedding" SET DATA TYPE text;--> statement-breakpoint
ALTER TABLE "users" ALTER COLUMN "hashed_password" SET DATA TYPE varchar(255);--> statement-breakpoint
ALTER TABLE "users" ALTER COLUMN "name" SET DATA TYPE varchar(255);--> statement-breakpoint
ALTER TABLE "embedding_cache" ADD COLUMN "packed_embedding" text;--> statement-breakpoint
ALTER TABLE "embedding_cache" ADD COLUMN "embedding_scale" numeric(10, 6);--> statement-breakpoint
ALTER TABLE "user_profiles" ADD CONSTRAINT "user_profiles_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "email_verified";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "avatar_url";