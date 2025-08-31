export function ensureError(payload: any): Error {
  if (payload instanceof Error) return payload;
  if (typeof payload === 'string') return new Error(payload);
  try {
    if (payload && typeof payload.message === 'string' && payload.message.length > 0) {
      return new Error(payload.message);
    }
    // Fallback: stringify object for message
    const msg = typeof payload === 'object' ? JSON.stringify(payload) : String(payload);
    return new Error(msg);
  } catch (e) {
    return new Error(String(payload));
  }
}
