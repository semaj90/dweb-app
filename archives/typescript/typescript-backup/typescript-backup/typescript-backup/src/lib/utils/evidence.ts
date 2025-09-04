import type { EvidenceType, EvidenceTypeKey } from '$lib/types/evidence';

function inferFormatFromMime(mime?: string | null, fileName?: string | null) {
  if (!mime && !fileName) return 'unknown';
  const m = String(mime || '').toLowerCase();
  if (m.includes('pdf')) return 'pdf';
  if (m.includes('image') || (fileName || '').toLowerCase().match(/\.(png|jpg|jpeg|bmp|tiff)$/)) return 'image';
  if ((fileName || '').toLowerCase().endsWith('.gif') || m.includes('gif')) return 'gif';
  if (m.includes('video') || (fileName || '').toLowerCase().match(/\.(mp4|mov|mkv|webm)$/)) return 'movie';
  if (m.includes('audio') || (fileName || '').toLowerCase().match(/\.(mp3|wav|m4a)$/)) return 'audio';
  return 'unknown';
}

export function getEvidenceKind(record: any): EvidenceTypeKey | string {
  if (!record) return 'document';
  return (
    record.evidenceType ??
    record.evidenceTypeDetails?.kind ??
    record.type ??
    'document'
  ) as EvidenceTypeKey | string;
}

export function buildEvidenceTypeDetails(record: any, preferredKind?: string): EvidenceType {
  const kind = preferredKind ?? (getEvidenceKind(record) as string);
  const metadata = {
    format: inferFormatFromMime(record?.mimeType ?? record?.metadata?.mimeType, record?.fileName ?? record?.metadata?.fileName)
  } as Record<string, unknown>;

  return {
    kind,
    metadata,
  } as EvidenceType;
}

export default { getEvidenceKind, buildEvidenceTypeDetails }
