// Neo4j Type Definitions

export interface Neo4jNode {
    id: string | number;
    labels: string[];
    properties: Record<string, any>;
}

export interface Neo4jRelationship {
    id: string | number;
    type: string;
    startNodeId: string | number;
    endNodeId: string | number;
    properties: Record<string, any>;
}

export interface Neo4jPath {
    nodes: Neo4jNode[];
    relationships: Neo4jRelationship[];
}

export interface Neo4jQueryResult<T = any> {
    records: T[];
    summary?: any;
    paths?: Neo4jPath[];
}

// Example: Legal graph entities
export interface CaseNode extends Neo4jNode {
    properties: {
        caseId: string;
        title: string;
        embedding?: number[];
        [key: string]: any;
    };
}

export interface EvidenceNode extends Neo4jNode {
    properties: {
        evidenceId: string;
        description: string;
        [key: string]: any;
    };
}

export interface PrecedentNode extends Neo4jNode {
    properties: {
        precedentId: string;
        citation: string;
        [key: string]: any;
    };
}
