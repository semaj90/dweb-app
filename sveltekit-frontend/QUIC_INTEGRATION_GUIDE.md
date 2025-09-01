syntax = "proto3";
package legalai.v1;

option go_package = "github.com/deeds-web/legal-ai/proto/legalai/v1";

// Legal AI Service Definition
service LegalAIService {
  // Document analysis and processing
  rpc AnalyzeDocument(AnalyzeDocumentRequest) returns (AnalyzeDocumentResponse);
  rpc ExtractEntities(ExtractEntitiesRequest) returns (ExtractEntitiesResponse);
  rpc GenerateSummary(GenerateSummaryRequest) returns (GenerateSummaryResponse);
  
  // Vector operations and search
  rpc VectorSearch(VectorSearchRequest) returns (VectorSearchResponse);
  rpc EmbedText(EmbedTextRequest) returns (EmbedTextResponse);
  rpc SimilaritySearch(SimilaritySearchRequest) returns (SimilaritySearchResponse);
  
  // Evidence and case management
  rpc ProcessEvidence(ProcessEvidenceRequest) returns (ProcessEvidenceResponse);
  rpc ValidateEvidence(ValidateEvidenceRequest) returns (ValidateEvidenceResponse);
  rpc CreateCase(CreateCaseRequest) returns (CreateCaseResponse);
}

// Document Analysis Messages
message AnalyzeDocumentRequest {
  string document_id = 1;
  string content = 2;
  string document_type = 3; // "contract", "evidence", "brief", "citation"
  int32 priority = 4; // 1=low, 2=medium, 3=high, 4=urgent
  map<string, string> metadata = 5;
}

message AnalyzeDocumentResponse {
  string analysis_id = 1;
  repeated string key_points = 2;
  double confidence_score = 3;
  string executive_summary = 4;
  repeated LegalEntity entities = 5;
  repeated string tags = 6;
  DocumentClassification classification = 7;
  repeated SimilarDocument similar_documents = 8;
  int64 processing_time_ms = 9;
}

// Vector Search Messages
message VectorSearchRequest {
  string query = 1;
  int32 limit = 2;
  double similarity_threshold = 3;
  repeated string document_types = 4;
  string case_id = 5;
}

message VectorSearchResponse {
  repeated VectorResult results = 1;
  string query_embedding_id = 2;
  int64 search_time_ms = 3;
  int32 total_candidates = 4;
}

// Evidence Processing Messages
message ProcessEvidenceRequest {
  string evidence_id = 1;
  string case_id = 2;
  bytes file_data = 3;
  string file_type = 4; // "pdf", "image", "text", "video"
  string chain_of_custody = 5;
  map<string, string> metadata = 6;
}

message ProcessEvidenceResponse {
  string processing_id = 1;
  EvidenceAnalysis analysis = 2;
  repeated string extracted_text_chunks = 3;
  string ocr_confidence = 4;
  repeated string detected_entities = 5;
  bool is_admissible = 6;
  string processing_status = 7;
}

// Supporting Types
message LegalEntity {
  string entity_type = 1; // "person", "organization", "date", "amount", "location"
  string entity_value = 2;
  double confidence = 3;
  int32 start_position = 4;
  int32 end_position = 5;
}

message DocumentClassification {
  string primary_type = 1;
  repeated string secondary_types = 2;
  double confidence = 3;
  string jurisdiction = 4;
  string practice_area = 5;
}

message SimilarDocument {
  string document_id = 1;
  string title = 2;
  double similarity_score = 3;
  string document_type = 4;
  repeated string matching_sections = 5;
}

message VectorResult {
  string document_id = 1;
  string content_preview = 2;
  double similarity_score = 3;
  map<string, string> metadata = 4;
  repeated string highlight_snippets = 5;
}

message EvidenceAnalysis {
  string analysis_type = 1;
  repeated string key_findings = 2;
  double authenticity_score = 3;
  string quality_assessment = 4;
  repeated string potential_issues = 5;
}

// Request/Response for other services
message ExtractEntitiesRequest {
  string text = 1;
  repeated string entity_types = 2;
}

message ExtractEntitiesResponse {
  repeated LegalEntity entities = 1;
  int64 processing_time_ms = 2;
}

message GenerateSummaryRequest {
  string text = 1;
  int32 max_length = 2;
  string summary_type = 3; // "executive", "technical", "bullet_points"
}

message GenerateSummaryResponse {
  string summary = 1;
  double confidence = 2;
  repeated string key_topics = 3;
}

message EmbedTextRequest {
  repeated string texts = 1;
  string embedding_model = 2;
}

message EmbedTextResponse {
  repeated EmbeddingVector embeddings = 1;
  string model_used = 2;
  int32 dimension = 3;
}

message EmbeddingVector {
  repeated float values = 1;
  string text_id = 2;
}

message SimilaritySearchRequest {
  repeated float query_embedding = 1;
  int32 limit = 2;
  double threshold = 3;
  string collection_name = 4;
}

message SimilaritySearchResponse {
  repeated VectorResult results = 1;
  int64 search_time_ms = 2;
}

message CreateCaseRequest {
  string title = 1;
  string description = 2;
  string priority = 3;
  string assigned_attorney = 4;
  map<string, string> metadata = 5;
}

message CreateCaseResponse {
  string case_id = 1;
  string status = 2;
  int64 created_at = 3;
}