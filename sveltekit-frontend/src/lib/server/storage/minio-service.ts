import { Client } from 'minio';

// Minimal MinIO wrapper; expects env vars set (fallbacks applied).
export class MinioService {
  private client: Client;
  private bucket: string;

  constructor() {
    const endpoint = process.env.MINIO_ENDPOINT || 'localhost:9000';
    const [endPoint, portStr] = endpoint.split(':');
    this.client = new Client({
      endPoint,
      port: Number(portStr || 9000),
      useSSL: false,
      accessKey: process.env.MINIO_ACCESS_KEY || 'minioadmin',
      secretKey: process.env.MINIO_SECRET_KEY || 'minioadmin'
    });
    this.bucket = process.env.MINIO_BUCKET || 'legal-documents';
  }

  async ensureBucket() {
    const exists = await this.client.bucketExists(this.bucket).catch(() => false);
    if (!exists) {
      await this.client.makeBucket(this.bucket, 'us-east-1');
    }
  }

  async upload(name: string, data: Buffer | Uint8Array, mimeType: string) {
    await this.ensureBucket();
    await this.client.putObject(this.bucket, name, data, { 'Content-Type': mimeType });
    return { url: `${this.bucket}/${name}` };
  }
}

export const minioService = new MinioService();
export default minioService;
