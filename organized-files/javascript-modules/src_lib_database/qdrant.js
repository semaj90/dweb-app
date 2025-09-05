// Qdrant Vector Database Service
// Manages vector search and storage

class QdrantManager {
  constructor() {
    this.baseUrl = 'http://localhost:6333';
    this.collection = 'legal_documents';
    this.connected = false;
  }

  /**
   * Initialize connection to Qdrant
   */
  async connect() {
    try {
      const response = await fetch(`${this.baseUrl}/collections`);
      if (response.ok) {
        this.connected = true;
        console.log('Connected to Qdrant');
        await this.ensureCollection();
        return true;
      }
    } catch (error) {
      console.error('Failed to connect to Qdrant:', error);
      this.connected = false;
      return false;
    }
  }

  /**
   * Ensure collection exists
   */
  async ensureCollection() {
    try {
      const response = await fetch(`${this.baseUrl}/collections/${this.collection}`);
      if (!response.ok) {
        // Create collection if it doesn't exist
        await this.createCollection();
      }
    } catch (error) {
      console.error('Failed to check collection:', error);
    }
  }

  /**
   * Create a new collection
   */
  async createCollection() {
    try {
      const response = await fetch(`${this.baseUrl}/collections/${this.collection}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          vectors: {
            size: 384,
            distance: 'Cosine'
          }
        })
      });
      
      if (response.ok) {
        console.log(`Created collection: ${this.collection}`);
        return true;
      }
    } catch (error) {
      console.error('Failed to create collection:', error);
      return false;
    }
  }

  /**
   * Add vectors to collection
   * @param {Array} points - Array of points with id, vector, and payload
   */
  async upsert(points) {
    try {
      const response = await fetch(
        `${this.baseUrl}/collections/${this.collection}/points`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ points })
        }
      );

      if (response.ok) {
        console.log(`Upserted ${points.length} points`);
        return true;
      }
    } catch (error) {
      console.error('Failed to upsert points:', error);
      return false;
    }
  }

  /**
   * Search for similar vectors
   * @param {Array} vector - Query vector
   * @param {number} limit - Number of results
   */
  async search(vector, limit = 5) {
    try {
      const response = await fetch(
        `${this.baseUrl}/collections/${this.collection}/points/search`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            vector,
            limit,
            with_payload: true
          })
        }
      );

      if (response.ok) {
        const data = await response.json();
        return data.result || [];
      }
    } catch (error) {
      console.error('Search failed:', error);
      return [];
    }
  }

  /**
   * Get point by ID
   * @param {string} id - Point ID
   */
  async getPoint(id) {
    try {
      const response = await fetch(
        `${this.baseUrl}/collections/${this.collection}/points/${id}`
      );

      if (response.ok) {
        const data = await response.json();
        return data.result;
      }
    } catch (error) {
      console.error(`Failed to get point ${id}:`, error);
      return null;
    }
  }

  /**
   * Delete point by ID
   * @param {string} id - Point ID
   */
  async deletePoint(id) {
    try {
      const response = await fetch(
        `${this.baseUrl}/collections/${this.collection}/points/delete`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            points: [id]
          })
        }
      );

      return response.ok;
    } catch (error) {
      console.error(`Failed to delete point ${id}:`, error);
      return false;
    }
  }

  /**
   * Get collection info
   */
  async getCollectionInfo() {
    try {
      const response = await fetch(
        `${this.baseUrl}/collections/${this.collection}`
      );

      if (response.ok) {
        const data = await response.json();
        return data.result;
      }
    } catch (error) {
      console.error('Failed to get collection info:', error);
      return null;
    }
  }

  /**
   * Clear entire collection
   */
  async clearCollection() {
    try {
      const response = await fetch(
        `${this.baseUrl}/collections/${this.collection}`,
        {
          method: 'DELETE'
        }
      );

      if (response.ok) {
        console.log('Collection cleared');
        await this.createCollection();
        return true;
      }
    } catch (error) {
      console.error('Failed to clear collection:', error);
      return false;
    }
  }

  /**
   * Get service status
   */
  getStatus() {
    return {
      connected: this.connected,
      baseUrl: this.baseUrl,
      collection: this.collection
    };
  }
}

// Export singleton instance
export const qdrantManager = new QdrantManager();

// Also export class
export { QdrantManager };