import neo4j, { type Driver, type Session } from 'neo4j-driver';
import { NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD } from '$env/static/private';

/**
 * Neo4j Service (lazy singleton)
 * Provides minimal typed helpers for creating sessions and basic document relationship queries.
 * Postgres integration intentionally decoupled to avoid coupling to Drizzle schema here.
 */

let driver: Driver | null = null;

export function getNeo4jDriver(): Driver {
    if (!driver) {
        if (!NEO4J_URI || !NEO4J_USER || !NEO4J_PASSWORD) {
            throw new Error('Neo4j environment variables not set');
        }
        driver = neo4j.driver(NEO4J_URI, neo4j.auth.basic(NEO4J_USER, NEO4J_PASSWORD));
    }
    return driver;
}

export function getSession(mode: 'read' | 'write' = 'write'): Session {
    const access = mode === 'read' ? neo4j.session.READ : neo4j.session.WRITE;
    return getNeo4jDriver().session({ defaultAccessMode: access });
}

export async function upsertLegalDocumentNode(doc: {
    id: string; title?: string; caseId?: string; uploadedAt?: string | Date; type?: string; status?: string;
}): Promise<void> {
    const session = getSession('write');
    try {
        await session.run(
            'MERGE (d:LegalDocument {id: $id}) SET d += $props',
            { id: doc.id, props: { ...doc, uploadedAt: doc.uploadedAt ? new Date(doc.uploadedAt).toISOString() : undefined } }
        );
    } finally {
        await session.close();
    }
}

export async function createRelatedDocuments(sourceId: string, targetId: string, relation: string = 'RELATED_TO'): Promise<void> {
    const session = getSession('write');
    try {
        await session.run(
            `MATCH (a:LegalDocument {id: $sourceId}), (b:LegalDocument {id: $targetId})\nMERGE (a)-[r:${relation}]->(b) RETURN r`,
            { sourceId, targetId }
        );
    } finally { await session.close(); }
}

export async function getRelatedDocuments(documentId: string): Promise<string[]> {
    const session = getSession('read');
    try {
        const result = await session.run(
            'MATCH (d:LegalDocument {id: $id})-[:RELATED_TO]->(r:LegalDocument) RETURN r.id AS id',
            { id: documentId }
        );
        return result.records.map(r => r.get('id'));
    } finally { await session.close(); }
}

export async function shutdownNeo4j(): Promise<void> {
    if (driver) {
        await driver.close();
        driver = null;
    }
}

process.on('exit', () => { void shutdownNeo4j(); });