import { Client as MinioClient } from 'minio';
import { env } from '$env/dynamic/private';

const MINIO_ENDPOINT = env.MINIO_ENDPOINT || 'localhost';
const MINIO_PORT = Number(env.MINIO_PORT || 9000);
const MINIO_USE_SSL = (env.MINIO_USE_SSL || 'false') === 'true';
const MINIO_ACCESS_KEY = env.MINIO_ACCESS_KEY || 'minioadmin';
const MINIO_SECRET_KEY = env.MINIO_SECRET_KEY || 'minioadmin';

export const minio = new MinioClient({
  endPoint: MINIO_ENDPOINT,
  port: MINIO_PORT,
  useSSL: MINIO_USE_SSL,
  accessKey: MINIO_ACCESS_KEY,
  secretKey: MINIO_SECRET_KEY,
});

export async function ensureBucket(bucketName: string): Promise<any> {
  try {
    const exists = await minio.bucketExists(bucketName);
    if (!exists) {
      await minio.makeBucket(bucketName);
    }
    return true;
  } catch (err: any) {
    console.error('MinIO ensureBucket error', err);
    throw err;
  }
}

export async function putObject(bucketName: string, objectName: string, buffer: Buffer, meta?: Record<string,string>): Promise<any> {
  await ensureBucket(bucketName);
  return minio.putObject(bucketName, objectName, buffer, meta || {});
}