# Backup & Recovery Strategy
## PostgreSQL + pgvector + Vector Data Protection

### 🎯 Backup Strategy Overview

This comprehensive backup strategy covers:
- **PostgreSQL Database** - Traditional ACID data
- **pgvector Extensions** - Vector embeddings and indexes
- **Document Storage** - Legal document files
- **Configuration** - Application and environment configs
- **AI Models** - Ollama model data

### 📋 Backup Requirements

#### Recovery Time Objectives (RTO)
- **Critical Systems**: 15 minutes
- **Vector Database**: 30 minutes
- **Document Storage**: 1 hour
- **AI Models**: 2 hours

#### Recovery Point Objectives (RPO)
- **Database**: 5 minutes (WAL streaming)
- **Vector Data**: 15 minutes
- **Documents**: 30 minutes
- **Configuration**: 1 hour

#### Data Classification
- **Tier 1**: Legal documents, case data, vector embeddings
- **Tier 2**: User accounts, session data
- **Tier 3**: Logs, temporary files, cache

### 🗄️ PostgreSQL Backup Strategy

#### 1. Continuous WAL Archiving
```bash
#!/bin/bash
# PostgreSQL WAL archiving configuration

# postgresql.conf settings
echo "
# WAL Configuration
wal_level = replica
archive_mode = on
archive_command = 'rsync %p backup-server:/var/lib/postgresql/wal_archive/%f'
archive_timeout = 300  # 5 minutes

# Replication settings
max_wal_senders = 3
max_replication_slots = 3
hot_standby = on
hot_standby_feedback = on

# Backup settings
checkpoint_timeout = 15min
checkpoint_completion_target = 0.7
" >> /etc/postgresql/15/main/postgresql.conf
```

#### 2. Physical Backups (pg_basebackup)
```bash
#!/bin/bash
# Physical backup script for PostgreSQL + pgvector

BACKUP_DIR="/var/backups/postgresql"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_NAME="legal_ai"
RETENTION_DAYS=30

# Create backup directory
mkdir -p "${BACKUP_DIR}/${TIMESTAMP}"

# Full physical backup
pg_basebackup \
  --host=localhost \
  --port=5432 \
  --username=backup_user \
  --pgdata="${BACKUP_DIR}/${TIMESTAMP}/base" \
  --format=tar \
  --wal-method=stream \
  --checkpoint=spread \
  --progress \
  --verbose

# Backup vector-specific data
echo "Creating vector data backup..."
pg_dump \
  --host=localhost \
  --port=5432 \
  --username=backup_user \
  --dbname=${DB_NAME} \
  --schema-only \
  --table=legal_documents \
  --table=content_embeddings \
  --file="${BACKUP_DIR}/${TIMESTAMP}/vector_schema.sql"

# Backup vector data with compression
pg_dump \
  --host=localhost \
  --port=5432 \
  --username=backup_user \
  --dbname=${DB_NAME} \
  --data-only \
  --table=legal_documents \
  --table=content_embeddings \
  --compress=9 \
  --file="${BACKUP_DIR}/${TIMESTAMP}/vector_data.sql.gz"

# Create manifest
cat > "${BACKUP_DIR}/${TIMESTAMP}/manifest.json" << EOF
{
  "timestamp": "${TIMESTAMP}",
  "database": "${DB_NAME}",
  "backup_type": "full",
  "vector_extension_version": "$(psql -h localhost -p 5432 -U backup_user -d ${DB_NAME} -t -c 'SELECT extversion FROM pg_extension WHERE extname = \"vector\";' | xargs)",
  "postgresql_version": "$(psql -h localhost -p 5432 -U backup_user -d ${DB_NAME} -t -c 'SELECT version();' | xargs)",
  "total_documents": "$(psql -h localhost -p 5432 -U backup_user -d ${DB_NAME} -t -c 'SELECT COUNT(*) FROM legal_documents;' | xargs)",
  "total_embeddings": "$(psql -h localhost -p 5432 -U backup_user -d ${DB_NAME} -t -c 'SELECT COUNT(*) FROM content_embeddings;' | xargs)",
  "backup_size": "$(du -sh ${BACKUP_DIR}/${TIMESTAMP} | cut -f1)",
  "created_at": "$(date -Iseconds)"
}
EOF

# Verify backup integrity
echo "Verifying backup integrity..."
tar -tf "${BACKUP_DIR}/${TIMESTAMP}/base.tar" > /dev/null
if [ $? -eq 0 ]; then
  echo "✅ Backup integrity verified"
else
  echo "❌ Backup integrity check failed"
  exit 1
fi

# Cleanup old backups
find "${BACKUP_DIR}" -type d -mtime +${RETENTION_DAYS} -exec rm -rf {} \;

echo "🎉 Backup completed: ${BACKUP_DIR}/${TIMESTAMP}"
```

#### 3. Logical Backups (pg_dump)
```bash
#!/bin/bash
# Logical backup script with vector-specific handling

BACKUP_DIR="/var/backups/logical"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_NAME="legal_ai"

mkdir -p "${BACKUP_DIR}/${TIMESTAMP}"

# Full logical backup
pg_dump \
  --host=localhost \
  --port=5432 \
  --username=backup_user \
  --dbname=${DB_NAME} \
  --format=custom \
  --compress=9 \
  --verbose \
  --file="${BACKUP_DIR}/${TIMESTAMP}/full_backup.dump"

# Schema-only backup for quick restoration testing
pg_dump \
  --host=localhost \
  --port=5432 \
  --username=backup_user \
  --dbname=${DB_NAME} \
  --schema-only \
  --file="${BACKUP_DIR}/${TIMESTAMP}/schema_only.sql"

# Vector-specific exports
echo "Exporting vector indexes..."
psql -h localhost -p 5432 -U backup_user -d ${DB_NAME} -c "
COPY (
  SELECT 
    id,
    title,
    document_type,
    jurisdiction,
    practice_area,
    content_embedding::text as embedding_vector,
    array_length(content_embedding, 1) as vector_dimensions,
    created_at
  FROM legal_documents 
  WHERE content_embedding IS NOT NULL
) TO '${BACKUP_DIR}/${TIMESTAMP}/vector_export.csv' 
WITH (FORMAT CSV, HEADER true);
"

# Export embedding metadata
psql -h localhost -p 5432 -U backup_user -d ${DB_NAME} -c "
COPY (
  SELECT 
    content_id,
    content_type,
    model,
    metadata,
    created_at
  FROM content_embeddings
) TO '${BACKUP_DIR}/${TIMESTAMP}/embeddings_metadata.csv' 
WITH (FORMAT CSV, HEADER true);
"

echo "✅ Logical backup completed"
```

### 🔄 Vector Data Backup Strategies

#### 1. Vector Index Backup
```sql
-- Create vector index backup procedures
CREATE OR REPLACE FUNCTION backup_vector_indexes()
RETURNS TABLE(
  index_name text,
  table_name text,
  index_size text,
  backup_path text
) AS $$
DECLARE
  rec RECORD;
  backup_dir text := '/var/backups/vector_indexes/' || to_char(now(), 'YYYYMMDD_HH24MISS');
BEGIN
  -- Create backup directory
  PERFORM pg_catalog.pg_stat_file(backup_dir, true);
  
  -- Export vector indexes information
  FOR rec IN 
    SELECT 
      i.indexrelid::regclass AS index_name,
      i.indrelid::regclass AS table_name,
      pg_size_pretty(pg_relation_size(i.indexrelid)) AS size
    FROM pg_index i
    JOIN pg_class c ON c.oid = i.indexrelid
    JOIN pg_am am ON am.oid = c.relam
    WHERE am.amname = 'ivfflat' -- pgvector indexes
  LOOP
    index_name := rec.index_name;
    table_name := rec.table_name;
    index_size := rec.size;
    backup_path := backup_dir || '/' || rec.index_name || '_backup.sql';
    
    -- Export index creation commands
    EXECUTE format('
      COPY (
        SELECT pg_get_indexdef(indexrelid) as index_definition
        FROM pg_index 
        WHERE indexrelid = %L::regclass
      ) TO %L
    ', rec.index_name, backup_path);
    
    RETURN NEXT;
  END LOOP;
END;
$$ LANGUAGE plpgsql;
```

#### 2. Incremental Vector Backup
```bash
#!/bin/bash
# Incremental vector backup script

BACKUP_DIR="/var/backups/vector_incremental"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_NAME="legal_ai"
LAST_BACKUP_FILE="/var/lib/postgresql/last_vector_backup"

mkdir -p "${BACKUP_DIR}/${TIMESTAMP}"

# Get last backup timestamp
if [ -f "$LAST_BACKUP_FILE" ]; then
  LAST_BACKUP=$(cat "$LAST_BACKUP_FILE")
else
  LAST_BACKUP="1970-01-01 00:00:00"
fi

echo "Performing incremental backup since: $LAST_BACKUP"

# Export only changed/new vectors
psql -h localhost -p 5432 -U backup_user -d ${DB_NAME} -c "
COPY (
  SELECT 
    id,
    title,
    content_embedding::text,
    updated_at,
    'INSERT' as operation_type
  FROM legal_documents 
  WHERE updated_at > '${LAST_BACKUP}'
    AND content_embedding IS NOT NULL
  
  UNION ALL
  
  SELECT 
    id,
    title,
    NULL as content_embedding,
    updated_at,
    'DELETE' as operation_type
  FROM deleted_legal_documents 
  WHERE deleted_at > '${LAST_BACKUP}'
) TO '${BACKUP_DIR}/${TIMESTAMP}/incremental_vectors.csv' 
WITH (FORMAT CSV, HEADER true);
"

# Update last backup timestamp
echo "$TIMESTAMP" > "$LAST_BACKUP_FILE"

echo "✅ Incremental vector backup completed"
```

### 💾 Document Storage Backup

#### 1. File System Backup
```bash
#!/bin/bash
# Document storage backup with rsync

DOCUMENT_DIR="/var/uploads/legal_docs"
BACKUP_DIR="/var/backups/documents"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REMOTE_SERVER="backup-server.example.com"

# Local backup with hardlinks for space efficiency
rsync -av \
  --link-dest="${BACKUP_DIR}/latest" \
  "${DOCUMENT_DIR}/" \
  "${BACKUP_DIR}/${TIMESTAMP}/"

# Update latest symlink
rm -f "${BACKUP_DIR}/latest"
ln -s "${TIMESTAMP}" "${BACKUP_DIR}/latest"

# Remote backup
rsync -avz \
  --delete \
  "${BACKUP_DIR}/${TIMESTAMP}/" \
  "${REMOTE_SERVER}:/backups/legal_ai/documents/${TIMESTAMP}/"

# Verify backup
if [ $? -eq 0 ]; then
  echo "✅ Document backup completed successfully"
else
  echo "❌ Document backup failed"
  exit 1
fi
```

#### 2. S3/Object Storage Backup
```typescript
// Document backup to cloud storage
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { createReadStream } from 'fs';
import { readdir, stat } from 'fs/promises';
import path from 'path';

class DocumentBackupService {
  private s3Client: S3Client;
  private bucketName: string;

  constructor() {
    this.s3Client = new S3Client({
      region: process.env.AWS_REGION || 'us-east-1',
      credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
      },
    });
    this.bucketName = process.env.S3_BACKUP_BUCKET || 'legal-ai-backups';
  }

  async backupDocuments(sourceDir: string): Promise<void> {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const backupPrefix = `documents/${timestamp}/`;

    try {
      const files = await this.getAllFiles(sourceDir);
      
      for (const filePath of files) {
        const relativePath = path.relative(sourceDir, filePath);
        const s3Key = backupPrefix + relativePath;
        
        const fileStream = createReadStream(filePath);
        const stats = await stat(filePath);
        
        await this.s3Client.send(new PutObjectCommand({
          Bucket: this.bucketName,
          Key: s3Key,
          Body: fileStream,
          ContentLength: stats.size,
          Metadata: {
            'original-path': filePath,
            'backup-timestamp': timestamp,
            'file-size': stats.size.toString(),
          },
        }));
        
        console.log(`Backed up: ${relativePath}`);
      }
      
      console.log(`✅ Document backup completed: ${backupPrefix}`);
      
    } catch (error) {
      console.error('❌ Document backup failed:', error);
      throw error;
    }
  }

  private async getAllFiles(dir: string): Promise<string[]> {
    const files: string[] = [];
    const entries = await readdir(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        files.push(...await this.getAllFiles(fullPath));
      } else {
        files.push(fullPath);
      }
    }
    
    return files;
  }
}
```

### 🤖 AI Model Backup Strategy

#### 1. Ollama Model Backup
```bash
#!/bin/bash
# Ollama model backup script

OLLAMA_DIR="/var/lib/ollama"
BACKUP_DIR="/var/backups/ollama_models"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "${BACKUP_DIR}/${TIMESTAMP}"

# Get list of installed models
MODELS=$(ollama list | tail -n +2 | awk '{print $1}')

echo "Backing up Ollama models..."

# Export each model
for model in $MODELS; do
  echo "Backing up model: $model"
  
  # Save model info
  ollama show "$model" --modelfile > "${BACKUP_DIR}/${TIMESTAMP}/${model}_modelfile.txt"
  
  # Export model weights (if possible)
  # Note: Ollama doesn't have direct export, so we backup the model directory
  model_dir=$(echo "$model" | tr ':' '_')
  if [ -d "${OLLAMA_DIR}/models/manifests/registry.ollama.ai/library/${model_dir}" ]; then
    cp -r "${OLLAMA_DIR}/models/manifests/registry.ollama.ai/library/${model_dir}" \
       "${BACKUP_DIR}/${TIMESTAMP}/"
  fi
done

# Backup Ollama configuration
cp -r "${OLLAMA_DIR}/models" "${BACKUP_DIR}/${TIMESTAMP}/"

# Create model inventory
cat > "${BACKUP_DIR}/${TIMESTAMP}/model_inventory.json" << EOF
{
  "timestamp": "${TIMESTAMP}",
  "models": [$(ollama list --format json | jq -r '.[] | @json' | paste -sd,)],
  "ollama_version": "$(ollama --version)",
  "backup_size": "$(du -sh ${BACKUP_DIR}/${TIMESTAMP} | cut -f1)"
}
EOF

echo "✅ Ollama model backup completed"
```

### 🔄 Automated Backup Scheduling

#### 1. Systemd Timer Configuration
```ini
# /etc/systemd/system/legal-ai-backup.timer
[Unit]
Description=Legal AI Backup Timer
Requires=legal-ai-backup.service

[Timer]
# Full backup daily at 2 AM
OnCalendar=*-*-* 02:00:00
# Incremental backup every 4 hours
OnCalendar=*-*-* 02,06,10,14,18,22:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

```ini
# /etc/systemd/system/legal-ai-backup.service
[Unit]
Description=Legal AI Backup Service
After=postgresql.service

[Service]
Type=oneshot
User=postgres
ExecStart=/usr/local/bin/legal-ai-backup.sh
Environment=PGPASSWORD=your_backup_password
```

#### 2. Cron-based Scheduling
```bash
# /etc/cron.d/legal-ai-backup

# Full backup daily at 2 AM
0 2 * * * postgres /usr/local/bin/legal-ai-full-backup.sh

# Incremental backup every 4 hours
0 */4 * * * postgres /usr/local/bin/legal-ai-incremental-backup.sh

# Vector data backup every hour
0 * * * * postgres /usr/local/bin/legal-ai-vector-backup.sh

# Document backup twice daily
0 6,18 * * * root /usr/local/bin/legal-ai-document-backup.sh

# Model backup weekly
0 3 * * 0 ollama /usr/local/bin/legal-ai-model-backup.sh

# Cleanup old backups daily
0 4 * * * root /usr/local/bin/legal-ai-backup-cleanup.sh
```

### 🔧 Recovery Procedures

#### 1. PostgreSQL Point-in-Time Recovery
```bash
#!/bin/bash
# PostgreSQL PITR recovery script

BACKUP_DIR="/var/backups/postgresql"
RECOVERY_TARGET_TIME="2024-01-15 14:30:00"
DATA_DIR="/var/lib/postgresql/15/main"
WAL_ARCHIVE="/var/lib/postgresql/wal_archive"

echo "Starting Point-in-Time Recovery to: $RECOVERY_TARGET_TIME"

# Stop PostgreSQL
systemctl stop postgresql

# Backup current data directory
mv "$DATA_DIR" "${DATA_DIR}.backup.$(date +%s)"

# Restore base backup
LATEST_BACKUP=$(ls -t "$BACKUP_DIR" | head -n1)
echo "Restoring from backup: $LATEST_BACKUP"

tar -xf "${BACKUP_DIR}/${LATEST_BACKUP}/base.tar" -C "$DATA_DIR"

# Configure recovery
cat > "${DATA_DIR}/recovery.conf" << EOF
restore_command = 'cp ${WAL_ARCHIVE}/%f %p'
recovery_target_time = '${RECOVERY_TARGET_TIME}'
recovery_target_action = 'promote'
EOF

# Start PostgreSQL in recovery mode
systemctl start postgresql

echo "✅ Point-in-Time Recovery initiated"
```

#### 2. Vector Data Recovery
```bash
#!/bin/bash
# Vector data recovery script

DB_NAME="legal_ai"
VECTOR_BACKUP_DIR="/var/backups/vector_incremental"
RECOVERY_TIMESTAMP="20240115_143000"

echo "Recovering vector data from: $RECOVERY_TIMESTAMP"

# Restore vector schema
psql -h localhost -p 5432 -U postgres -d $DB_NAME \
  -f "${VECTOR_BACKUP_DIR}/${RECOVERY_TIMESTAMP}/vector_schema.sql"

# Restore vector data
zcat "${VECTOR_BACKUP_DIR}/${RECOVERY_TIMESTAMP}/vector_data.sql.gz" | \
  psql -h localhost -p 5432 -U postgres -d $DB_NAME

# Rebuild vector indexes
psql -h localhost -p 5432 -U postgres -d $DB_NAME -c "
REINDEX INDEX CONCURRENTLY legal_documents_content_embedding_idx;
REINDEX INDEX CONCURRENTLY legal_documents_title_embedding_idx;
"

# Verify vector data integrity
psql -h localhost -p 5432 -U postgres -d $DB_NAME -c "
SELECT 
  COUNT(*) as total_documents,
  COUNT(content_embedding) as documents_with_embeddings,
  AVG(array_length(content_embedding, 1)) as avg_vector_dimensions
FROM legal_documents;
"

echo "✅ Vector data recovery completed"
```

### 📊 Backup Monitoring & Alerting

#### 1. Backup Health Check
```typescript
// Backup monitoring service
interface BackupStatus {
  timestamp: Date;
  type: 'full' | 'incremental' | 'vector' | 'document' | 'model';
  status: 'success' | 'failed' | 'partial';
  size: string;
  duration: number;
  errors?: string[];
}

class BackupMonitor {
  private statuses: BackupStatus[] = [];

  async checkBackupHealth(): Promise<{
    overall: 'healthy' | 'warning' | 'critical';
    lastFullBackup: Date | null;
    lastIncrementalBackup: Date | null;
    missedBackups: string[];
    storageUsage: { used: string; available: string; percentage: number };
  }> {
    const now = new Date();
    const oneDayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);
    const oneWeekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);

    // Check for recent backups
    const recentFull = this.statuses
      .filter(s => s.type === 'full' && s.timestamp > oneWeekAgo)
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())[0];

    const recentIncremental = this.statuses
      .filter(s => s.type === 'incremental' && s.timestamp > oneDayAgo)
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())[0];

    // Check for missed backups
    const missedBackups: string[] = [];
    if (!recentFull) missedBackups.push('No full backup in the last week');
    if (!recentIncremental) missedBackups.push('No incremental backup in the last day');

    // Check storage usage
    const storageUsage = await this.getStorageUsage();

    let overall: 'healthy' | 'warning' | 'critical' = 'healthy';
    if (missedBackups.length > 0 || storageUsage.percentage > 90) {
      overall = 'critical';
    } else if (storageUsage.percentage > 80) {
      overall = 'warning';
    }

    return {
      overall,
      lastFullBackup: recentFull?.timestamp || null,
      lastIncrementalBackup: recentIncremental?.timestamp || null,
      missedBackups,
      storageUsage
    };
  }

  private async getStorageUsage(): Promise<{ used: string; available: string; percentage: number }> {
    // Implementation would check actual disk usage
    return {
      used: '850 GB',
      available: '150 GB',
      percentage: 85
    };
  }
}
```

#### 2. Backup Alerting
```bash
#!/bin/bash
# Backup alerting script

WEBHOOK_URL="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
EMAIL_ALERT="admin@yourdomain.com"

send_alert() {
  local severity=$1
  local message=$2
  
  # Slack notification
  curl -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"🚨 [$severity] Legal AI Backup Alert: $message\"}" \
    "$WEBHOOK_URL"
  
  # Email notification
  echo "$message" | mail -s "[$severity] Legal AI Backup Alert" "$EMAIL_ALERT"
}

# Check backup status
check_backup_status() {
  local backup_dir="/var/backups/postgresql"
  local latest_backup=$(find "$backup_dir" -type d -name "2*" | sort | tail -1)
  
  if [ -z "$latest_backup" ]; then
    send_alert "CRITICAL" "No backups found in $backup_dir"
    return 1
  fi
  
  local backup_age=$(( ($(date +%s) - $(stat -c %Y "$latest_backup")) / 3600 ))
  
  if [ $backup_age -gt 48 ]; then
    send_alert "CRITICAL" "Latest backup is $backup_age hours old (older than 48 hours)"
    return 1
  elif [ $backup_age -gt 24 ]; then
    send_alert "WARNING" "Latest backup is $backup_age hours old (older than 24 hours)"
  fi
  
  return 0
}

# Check disk space
check_disk_space() {
  local usage=$(df /var/backups | tail -1 | awk '{print $5}' | sed 's/%//')
  
  if [ $usage -gt 90 ]; then
    send_alert "CRITICAL" "Backup disk usage is at ${usage}% (critical threshold: 90%)"
    return 1
  elif [ $usage -gt 80 ]; then
    send_alert "WARNING" "Backup disk usage is at ${usage}% (warning threshold: 80%)"
  fi
  
  return 0
}

# Run checks
check_backup_status
check_disk_space
```

### 📈 Backup Performance Optimization

#### 1. Parallel Backup Processing
```bash
#!/bin/bash
# Parallel backup script for large datasets

BACKUP_DIR="/var/backups/parallel"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_NAME="legal_ai"
JOBS=4  # Number of parallel jobs

mkdir -p "${BACKUP_DIR}/${TIMESTAMP}"

# Get list of large tables
LARGE_TABLES=$(psql -h localhost -p 5432 -U backup_user -d $DB_NAME -t -c "
SELECT tablename FROM pg_tables 
WHERE schemaname = 'public' 
AND tablename IN ('legal_documents', 'content_embeddings', 'search_sessions')
ORDER BY pg_total_relation_size(quote_ident(tablename)) DESC;
" | xargs)

# Backup tables in parallel
echo "Starting parallel backup of large tables..."
for table in $LARGE_TABLES; do
  {
    echo "Backing up table: $table"
    pg_dump \
      --host=localhost \
      --port=5432 \
      --username=backup_user \
      --dbname=$DB_NAME \
      --table=$table \
      --format=custom \
      --compress=9 \
      --file="${BACKUP_DIR}/${TIMESTAMP}/${table}.dump"
    echo "Completed backup of table: $table"
  } &
  
  # Limit concurrent jobs
  while [ $(jobs -r | wc -l) -ge $JOBS ]; do
    sleep 1
  done
done

# Wait for all background jobs to complete
wait

echo "✅ Parallel backup completed"
```

#### 2. Compression Optimization
```bash
#!/bin/bash
# Advanced compression for vector data

BACKUP_DIR="/var/backups/compressed"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_NAME="legal_ai"

mkdir -p "${BACKUP_DIR}/${TIMESTAMP}"

# Vector data with specialized compression
echo "Compressing vector data with LZMA..."
pg_dump \
  --host=localhost \
  --port=5432 \
  --username=backup_user \
  --dbname=$DB_NAME \
  --table=legal_documents \
  --data-only \
  --no-owner \
  --no-privileges | \
  xz -9 -T 0 > "${BACKUP_DIR}/${TIMESTAMP}/vector_data.sql.xz"

# Regular data with gzip
echo "Compressing regular data with gzip..."
pg_dump \
  --host=localhost \
  --port=5432 \
  --username=backup_user \
  --dbname=$DB_NAME \
  --exclude-table=legal_documents \
  --exclude-table=content_embeddings \
  --format=custom \
  --compress=9 \
  --file="${BACKUP_DIR}/${TIMESTAMP}/regular_data.dump"

# Compare compression ratios
original_size=$(psql -h localhost -p 5432 -U backup_user -d $DB_NAME -t -c "
SELECT pg_size_pretty(pg_database_size('$DB_NAME'));
" | xargs)

compressed_size=$(du -sh "${BACKUP_DIR}/${TIMESTAMP}" | cut -f1)

echo "Original size: $original_size"
echo "Compressed size: $compressed_size"
echo "✅ Compression optimization completed"
```

### 🧪 Recovery Testing

#### 1. Automated Recovery Testing
```bash
#!/bin/bash
# Automated backup recovery testing

TEST_DB="legal_ai_recovery_test"
BACKUP_DIR="/var/backups/postgresql"
LATEST_BACKUP=$(ls -t "$BACKUP_DIR" | head -n1)

echo "Testing recovery with backup: $LATEST_BACKUP"

# Create test database
createdb -h localhost -p 5432 -U postgres "$TEST_DB"

# Restore backup to test database
pg_restore \
  --host=localhost \
  --port=5432 \
  --username=postgres \
  --dbname="$TEST_DB" \
  --verbose \
  "${BACKUP_DIR}/${LATEST_BACKUP}/full_backup.dump"

# Verify data integrity
echo "Verifying data integrity..."
psql -h localhost -p 5432 -U postgres -d "$TEST_DB" -c "
SELECT 
  'legal_documents' as table_name,
  COUNT(*) as record_count,
  COUNT(content_embedding) as vectors_count
FROM legal_documents
UNION ALL
SELECT 
  'content_embeddings' as table_name,
  COUNT(*) as record_count,
  COUNT(embedding) as vectors_count
FROM content_embeddings;
"

# Test vector operations
psql -h localhost -p 5432 -U postgres -d "$TEST_DB" -c "
SELECT 
  title,
  1 - (content_embedding <=> (SELECT content_embedding FROM legal_documents LIMIT 1)) as similarity
FROM legal_documents 
WHERE content_embedding IS NOT NULL
ORDER BY similarity DESC
LIMIT 5;
"

# Cleanup test database
dropdb -h localhost -p 5432 -U postgres "$TEST_DB"

echo "✅ Recovery test completed successfully"
```

### 📋 Disaster Recovery Plan

#### Recovery Priority Matrix
| Component | Priority | RTO | RPO | Recovery Method |
|-----------|----------|-----|-----|-----------------|
| PostgreSQL Core | P1 | 15 min | 5 min | WAL streaming + PITR |
| Vector Data | P1 | 30 min | 15 min | Specialized vector backup |
| Document Files | P2 | 1 hour | 30 min | rsync + cloud storage |
| AI Models | P3 | 2 hours | 4 hours | Model repository backup |
| Configuration | P2 | 30 min | 1 hour | Git repository + secrets |

#### Recovery Runbook
1. **Assess Damage** - Determine scope of data loss
2. **Isolate Systems** - Prevent further damage
3. **Restore Core Database** - PostgreSQL with PITR
4. **Restore Vector Data** - pgvector embeddings and indexes
5. **Restore Documents** - File system and cloud backups
6. **Restore AI Models** - Ollama model re-download/restore
7. **Verify Integrity** - Run comprehensive tests
8. **Resume Operations** - Switch DNS/load balancer

---

**Document Version:** 2.0  
**Last Updated:** 2024-12-XX  
**Next Review:** 2024-XX-XX  
**Owner:** Infrastructure Team
