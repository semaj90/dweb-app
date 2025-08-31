// Flatbuffer Legal Schema Mock
// This is a mock implementation to resolve import errors

export interface LegalDocument {
  id: string;
  title: string;
  content: string;
  type: string;
  created: Date;
  modified: Date;
  metadata: Record<string, any>;
}

export interface LegalCase {
  id: string;
  title: string;
  documents: LegalDocument[];
  status: string;
  created: Date;
  modified: Date;
}

export class FlatBufferLegalSchema {
  static encode(data: LegalDocument | LegalCase): Uint8Array {
    // Mock implementation - in production, this would use actual FlatBuffers
    return new TextEncoder().encode(JSON.stringify(data));
  }

  static decode(buffer: Uint8Array): LegalDocument | LegalCase {
    // Mock implementation - in production, this would use actual FlatBuffers
    const jsonString = new TextDecoder().decode(buffer);
    return JSON.parse(jsonString);
  }

  static createLegalDocument(
    id: string,
    title: string,
    content: string,
    type: string,
    metadata: Record<string, any> = {}
  ): LegalDocument {
    return {
      id,
      title,
      content,
      type,
      created: new Date(),
      modified: new Date(),
      metadata
    };
  }

  static createLegalCase(
    id: string,
    title: string,
    documents: LegalDocument[] = []
  ): LegalCase {
    return {
      id,
      title,
      documents,
      status: 'active',
      created: new Date(),
      modified: new Date()
    };
  }
}

// Export both named and default
export const LegalDocumentBinarySerializer = FlatBufferLegalSchema;
export default FlatBufferLegalSchema;