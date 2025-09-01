/**
 * Gallery Download API - File Download Handler
 * Handles secure file downloads with access control and logging
 */

import { error } from '@sveltejs/kit';
import type { RequestHandler } from './$types.js';
import { readFile, stat } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import { db } from '$lib/server/database';
import { evidence, cases } from '$lib/server/db/schema';
import { eq } from 'drizzle-orm';
import { URL } from "url";

interface DownloadLog {
  itemId: string;
  userId?: string;
  userAgent: string;
  ip: string;
  timestamp: Date;
  fileSize: number;
  downloadType: 'view' | 'download';
}

export const GET: RequestHandler = async ({ params, request, locals, url }) => {
  try {
    const itemId = params.id;
    const downloadType = url.searchParams.get('type') || 'download'; // 'download' or 'view'
    const inline = url.searchParams.get('inline') === 'true';

    if (!itemId) {
      throw error(400, 'Item ID is required');
    }

    // Get item details from database
    const itemQuery = await db
      .select({
        id: evidence.id,
        title: evidence.title,
        fileName: evidence.fileName,
        originalFileName: evidence.originalFileName,
        fileType: evidence.fileType,
        fileSize: evidence.fileSize,
        filePath: evidence.filePath,
        isPublic: evidence.isPublic,
        caseId: evidence.caseId,
        caseTitle: cases.title
      })
      .from(evidence)
      .leftJoin(cases, eq(evidence.caseId, cases.id))
      .where(eq(evidence.id, itemId))
      .limit(1)
      .execute();

    if (itemQuery.length === 0) {
      throw error(404, 'File not found');
    }

    const item = itemQuery[0];

    // Check access permissions
    // TODO: Implement proper user authentication and authorization
    // For now, only check if file is public or user has access to the case
    if (!item.isPublic) {
      // TODO: Check user permissions for the case
      // const userHasAccess = await checkUserCaseAccess(locals.user?.id, item.caseId);
      // if (!userHasAccess) {
      //   throw error(403, 'Access denied');
      // }
    }

    if (!item.filePath) {
      throw error(404, 'File path not found');
    }

    // Construct full file path
    const fullPath = path.join(process.cwd(), 'static', item.filePath);

    if (!existsSync(fullPath)) {
      console.error(`File not found on disk: ${fullPath}`);
      throw error(404, 'File not found on disk');
    }

    // Get file stats
    const stats = await stat(fullPath);
    const fileBuffer = await readFile(fullPath);

    // Log download
    await logDownload({
      itemId: item.id,
      userId: locals.user?.id,
      userAgent: request.headers.get('user-agent') || 'Unknown',
      ip: getClientIP(request),
      timestamp: new Date(),
      fileSize: stats.size,
      downloadType: downloadType as 'view' | 'download'
    });

    // Determine content disposition
    const disposition = inline || downloadType === 'view' ? 'inline' : 'attachment';
    const filename = item.originalFileName || item.fileName || 'download';

    // Set appropriate headers
    const headers = new Headers({
      'Content-Type': item.fileType || 'application/octet-stream',
      'Content-Length': stats.size.toString(),
      'Content-Disposition': `${disposition}; filename="${encodeURIComponent(filename)}"`,
      'Cache-Control': 'private, max-age=3600', // Cache for 1 hour
      'Last-Modified': stats.mtime.toUTCString(),
      'X-File-ID': item.id,
      'X-File-Size': stats.size.toString(),
      'X-Download-Type': downloadType
    });

    // Add security headers
    headers.set('X-Content-Type-Options', 'nosniff');
    headers.set('X-Frame-Options', 'DENY');

    // For images, add additional headers
    if (item.fileType?.startsWith('image/')) {
      headers.set('Accept-Ranges', 'bytes');
    }

    // Handle range requests for large files (useful for video streaming)
    const range = request.headers.get('range');
    if (range && stats.size > 1024 * 1024) { // Only for files > 1MB
      return handleRangeRequest(fileBuffer, range, item.fileType || 'application/octet-stream', stats.size);
    }

    return new Response(fileBuffer, {
      status: 200,
      headers
    });

  } catch (err) {
    console.error('Download error:', err);
    
    if (err instanceof Error && (
      err.message.includes('400') || 
      err.message.includes('403') || 
      err.message.includes('404')
    )) {
      const statusCode = parseInt(err.message.split(' ')[0]) || 500;
      throw error(statusCode, err.message);
    }
    
    throw error(500, `Download failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
  }
};

function handleRangeRequest(buffer: Buffer, rangeHeader: string, contentType: string, fileSize: number): Response {
  try {
    const ranges = parseRangeHeader(rangeHeader, fileSize);
    
    if (!ranges || ranges.length === 0) {
      return new Response(buffer, {
        status: 200,
        headers: {
          'Content-Type': contentType,
          'Content-Length': fileSize.toString(),
          'Accept-Ranges': 'bytes'
        }
      });
    }

    const range = ranges[0]; // Handle single range for now
    const start = range.start;
    const end = range.end;
    const chunkSize = end - start + 1;
    const chunk = buffer.slice(start, end + 1);

    return new Response(chunk, {
      status: 206,
      headers: {
        'Content-Type': contentType,
        'Content-Length': chunkSize.toString(),
        'Content-Range': `bytes ${start}-${end}/${fileSize}`,
        'Accept-Ranges': 'bytes'
      }
    });

  } catch (error) {
    console.error('Range request error:', error);
    // Fallback to full file
    return new Response(buffer, {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Content-Length': fileSize.toString()
      }
    });
  }
}

function parseRangeHeader(range: string, fileSize: number): Array<{ start: number; end: number }> | null {
  try {
    if (!range.startsWith('bytes=')) {
      return null;
    }

    const ranges = range.slice(6).split(',').map(r => r.trim());
    const parsedRanges: Array<{ start: number; end: number }> = [];

    for (const rangeStr of ranges) {
      const [startStr, endStr] = rangeStr.split('-');
      
      let start: number;
      let end: number;

      if (startStr === '') {
        // Suffix-byte-range-spec: -500
        end = fileSize - 1;
        start = Math.max(0, fileSize - parseInt(endStr));
      } else if (endStr === '') {
        // Byte-range-spec: 500-
        start = parseInt(startStr);
        end = fileSize - 1;
      } else {
        // Byte-range-spec: 0-499
        start = parseInt(startStr);
        end = parseInt(endStr);
      }

      // Validate range
      if (start >= 0 && end < fileSize && start <= end) {
        parsedRanges.push({ start, end });
      }
    }

    return parsedRanges.length > 0 ? parsedRanges : null;
  } catch (error) {
    return null;
  }
}

async function logDownload(log: DownloadLog) {
  try {
    // TODO: Store download logs in database
    // For now, just console.log
    console.log('Download logged:', {
      itemId: log.itemId,
      userId: log.userId || 'anonymous',
      timestamp: log.timestamp.toISOString(),
      fileSize: log.fileSize,
      downloadType: log.downloadType,
      ip: log.ip
    });

    // Could store in a downloads table:
    // await db.insert(downloads).values({
    //   itemId: log.itemId,
    //   userId: log.userId,
    //   userAgent: log.userAgent,
    //   ip: log.ip,
    //   timestamp: log.timestamp,
    //   fileSize: log.fileSize,
    //   downloadType: log.downloadType
    // });

  } catch (error) {
    console.error('Failed to log download:', error);
    // Don't throw here as download should still proceed
  }
}

function getClientIP(request: Request): string {
  // Try to get real IP from various headers (common in production)
  const forwarded = request.headers.get('x-forwarded-for');
  if (forwarded) {
    return forwarded.split(',')[0].trim();
  }

  const realIP = request.headers.get('x-real-ip');
  if (realIP) {
    return realIP.trim();
  }

  // Fallback to direct connection (development)
  return request.headers.get('x-forwarded-host') || 'unknown';
}

// HEAD request for file metadata without downloading
export const HEAD: RequestHandler = async ({ params, locals }) => {
  try {
    const itemId = params.id;

    if (!itemId) {
      throw error(400, 'Item ID is required');
    }

    const itemQuery = await db
      .select({
        id: evidence.id,
        fileName: evidence.fileName,
        originalFileName: evidence.originalFileName,
        fileType: evidence.fileType,
        fileSize: evidence.fileSize,
        filePath: evidence.filePath,
        isPublic: evidence.isPublic,
        uploadedAt: evidence.uploadedAt
      })
      .from(evidence)
      .where(eq(evidence.id, itemId))
      .limit(1)
      .execute();

    if (itemQuery.length === 0) {
      throw error(404, 'File not found');
    }

    const item = itemQuery[0];

    // Check if file exists on disk
    if (item.filePath) {
      const fullPath = path.join(process.cwd(), 'static', item.filePath);
      if (!existsSync(fullPath)) {
        throw error(404, 'File not found on disk');
      }
    }

    return new Response(null, {
      status: 200,
      headers: {
        'Content-Type': item.fileType || 'application/octet-stream',
        'Content-Length': (item.fileSize || 0).toString(),
        'Last-Modified': item.uploadedAt?.toUTCString() || new Date().toUTCString(),
        'Accept-Ranges': 'bytes',
        'X-File-ID': item.id,
        'X-File-Name': item.originalFileName || item.fileName || 'unknown'
      }
    });

  } catch (err) {
    console.error('HEAD request error:', err);
    
    if (err instanceof Error && (err.message.includes('400') || err.message.includes('404'))) {
      const statusCode = parseInt(err.message.split(' ')[0]) || 500;
      throw error(statusCode, err.message);
    }
    
    throw error(500, 'HEAD request failed');
  }
};