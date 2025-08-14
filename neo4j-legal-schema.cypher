# Neo4j Legal AI Integration Scripts
# These scripts sync data between PostgreSQL, Neo4j, and Qdrant

# 1. Create Legal Knowledge Graph Schema
CREATE CONSTRAINT legal_case_id IF NOT EXISTS FOR (c:LegalCase) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT legal_document_id IF NOT EXISTS FOR (d:LegalDocument) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT legal_entity_id IF NOT EXISTS FOR (e:LegalEntity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT legal_precedent_id IF NOT EXISTS FOR (p:LegalPrecedent) REQUIRE p.id IS UNIQUE;

# 2. Create Legal Relationship Types
// CASE_DOCUMENT: Links cases to their documents
// DOCUMENT_ENTITY: Links documents to mentioned entities
// ENTITY_PRECEDENT: Links entities to legal precedents
// CASE_PRECEDENT: Links cases to applicable precedents
// SIMILAR_CASE: Links similar legal cases

# 3. Vector Similarity Index for Semantic Search
CREATE VECTOR INDEX document_embeddings IF NOT EXISTS
FOR (d:LegalDocument) ON (d.embedding)
OPTIONS {indexConfig: {ector.dimensions: 384, ector.similarity_function: 'cosine'}};
