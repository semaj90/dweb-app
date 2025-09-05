I see that your code already uses PostgreSQL exclusively. The setup script is correctly configured to use PostgreSQL with the pgvector extension for vector embeddings. There's no SQLite usage in the current code.

The script:
- Sets up PostgreSQL with the correct extensions (vector, uuid-ossp)
- Creates proper tables for legal documents, cases, and audit logs
- Configures the database connection for `legal_admin` user and `prosecutor_db` database
- Uses Docker to run PostgreSQL

The code is already following your requirement to never use SQLite and always use PostgreSQL. Is there a specific aspect of the PostgreSQL setup you'd like me to modify or enhance?
