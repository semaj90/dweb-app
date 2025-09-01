
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async () => json({ success: true, cuda: { available: false, version: null } });
export const prerender = false;