// Simple test to verify MinIO connection without debugger
import { S3Client, ListBucketsCommand } from '@aws-sdk/client-s3';

const s3Client = new S3Client({
  endpoint: 'http://localhost:9000',
  region: 'us-east-1',
  credentials: {
    accessKeyId: 'minioadmin',
    secretAccessKey: 'minioadmin',
  },
  forcePathStyle: true,
});

try {
  console.log('🔄 Testing MinIO connection...');
  const result = await s3Client.send(new ListBucketsCommand({}));
  console.log('✅ MinIO connection successful!');
  console.log('📁 Available buckets:', result.Buckets?.map(b => b.Name) || 'None');
} catch (error) {
  console.error('❌ MinIO connection failed:', error.message);
}
