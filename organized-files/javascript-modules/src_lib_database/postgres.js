// PostgreSQL Database Service
// Manages PostgreSQL connections and operations

import pg from 'pg';
const { Pool } = pg;

class PostgresDB {
  constructor() {
    this.pool = null;
    this.connected = false;
  }

  /**
   * Initialize database connection pool
   */
  async connect() {
    try {
      this.pool = new Pool({
        host: process.env.DB_HOST || 'localhost',
        port: process.env.DB_PORT || 5432,
        database: process.env.DB_NAME || 'legal_ai',
        user: process.env.DB_USER || 'postgres',
        password: process.env.DB_PASSWORD || 'postgres',
        max: 20,
        idleTimeoutMillis: 30000,
        connectionTimeoutMillis: 2000,
      });

      // Test connection
      const client = await this.pool.connect();
      await client.query('SELECT NOW()');
      client.release();

      this.connected = true;
      console.log('Connected to PostgreSQL');
      return true;
    } catch (error) {
      console.error('Failed to connect to PostgreSQL:', error);
      this.connected = false;
      return false;
    }
  }

  /**
   * Execute a query
   * @param {string} text - SQL query
   * @param {Array} params - Query parameters
   */
  async query(text, params = []) {
    if (!this.connected) await this.connect();
    
    try {
      const result = await this.pool.query(text, params);
      return result;
    } catch (error) {
      console.error('Query error:', error);
      throw error;
    }
  }

  /**
   * Get a client for transactions
   */
  async getClient() {
    if (!this.connected) await this.connect();
    return await this.pool.connect();
  }

  /**
   * Execute a transaction
   * @param {Function} callback - Transaction callback
   */
  async transaction(callback) {
    const client = await this.getClient();
    
    try {
      await client.query('BEGIN');
      const result = await callback(client);
      await client.query('COMMIT');
      return result;
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }

  /**
   * Insert data into table
   * @param {string} table - Table name
   * @param {Object} data - Data to insert
   */
  async insert(table, data) {
    const keys = Object.keys(data);
    const values = Object.values(data);
    const placeholders = keys.map((_, i) => `$${i + 1}`).join(', ');
    
    const query = `
      INSERT INTO ${table} (${keys.join(', ')})
      VALUES (${placeholders})
      RETURNING *
    `;
    
    const result = await this.query(query, values);
    return result.rows[0];
  }

  /**
   * Update data in table
   * @param {string} table - Table name
   * @param {Object} data - Data to update
   * @param {Object} where - Where conditions
   */
  async update(table, data, where) {
    const dataKeys = Object.keys(data);
    const dataValues = Object.values(data);
    const whereKeys = Object.keys(where);
    const whereValues = Object.values(where);
    
    const setClause = dataKeys
      .map((key, i) => `${key} = $${i + 1}`)
      .join(', ');
    
    const whereClause = whereKeys
      .map((key, i) => `${key} = $${dataValues.length + i + 1}`)
      .join(' AND ');
    
    const query = `
      UPDATE ${table}
      SET ${setClause}
      WHERE ${whereClause}
      RETURNING *
    `;
    
    const result = await this.query(query, [...dataValues, ...whereValues]);
    return result.rows;
  }

  /**
   * Delete data from table
   * @param {string} table - Table name
   * @param {Object} where - Where conditions
   */
  async delete(table, where) {
    const keys = Object.keys(where);
    const values = Object.values(where);
    
    const whereClause = keys
      .map((key, i) => `${key} = $${i + 1}`)
      .join(' AND ');
    
    const query = `
      DELETE FROM ${table}
      WHERE ${whereClause}
      RETURNING *
    `;
    
    const result = await this.query(query, values);
    return result.rows;
  }

  /**
   * Select data from table
   * @param {string} table - Table name
   * @param {Object} options - Query options
   */
  async select(table, options = {}) {
    const { where = {}, orderBy = null, limit = null, offset = null } = options;
    
    let query = `SELECT * FROM ${table}`;
    const values = [];
    
    // Add WHERE clause
    const whereKeys = Object.keys(where);
    if (whereKeys.length > 0) {
      const whereClause = whereKeys
        .map((key, i) => `${key} = $${i + 1}`)
        .join(' AND ');
      query += ` WHERE ${whereClause}`;
      values.push(...Object.values(where));
    }
    
    // Add ORDER BY
    if (orderBy) {
      query += ` ORDER BY ${orderBy}`;
    }
    
    // Add LIMIT
    if (limit) {
      query += ` LIMIT ${limit}`;
    }
    
    // Add OFFSET
    if (offset) {
      query += ` OFFSET ${offset}`;
    }
    
    const result = await this.query(query, values);
    return result.rows;
  }

  /**
   * Create tables if they don't exist
   */
  async initializeTables() {
    try {
      // Legal documents table
      await this.query(`
        CREATE TABLE IF NOT EXISTS legal_documents (
          id SERIAL PRIMARY KEY,
          title VARCHAR(255) NOT NULL,
          content TEXT,
          document_type VARCHAR(100),
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          metadata JSONB,
          embedding VECTOR(384)
        )
      `);

      // Cases table
      await this.query(`
        CREATE TABLE IF NOT EXISTS cases (
          id SERIAL PRIMARY KEY,
          case_number VARCHAR(100) UNIQUE,
          title VARCHAR(255),
          status VARCHAR(50),
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          metadata JSONB
        )
      `);

      // Chat sessions table
      await this.query(`
        CREATE TABLE IF NOT EXISTS chat_sessions (
          id SERIAL PRIMARY KEY,
          session_id VARCHAR(100) UNIQUE,
          user_id VARCHAR(100),
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          messages JSONB
        )
      `);

      console.log('Database tables initialized');
      return true;
    } catch (error) {
      console.error('Failed to initialize tables:', error);
      return false;
    }
  }

  /**
   * Close database connection
   */
  async disconnect() {
    if (this.pool) {
      await this.pool.end();
      this.connected = false;
      console.log('Disconnected from PostgreSQL');
    }
  }

  /**
   * Get connection status
   */
  getStatus() {
    return {
      connected: this.connected,
      totalCount: this.pool?.totalCount || 0,
      idleCount: this.pool?.idleCount || 0,
      waitingCount: this.pool?.waitingCount || 0
    };
  }
}

// Export singleton instance
export const db = new PostgresDB();

// Also export for drizzle compatibility
export const query = {
  legalDocuments: {
    findFirst: async (options) => {
      // Stub implementation for compatibility
      return null;
    }
  }
};

// Export class
export { PostgresDB };