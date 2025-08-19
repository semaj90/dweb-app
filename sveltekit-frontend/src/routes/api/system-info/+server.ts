import os from "os";
/**
 * System Information API
 */

import { json } from '@sveltejs/kit';
// Orphaned content: import type { RequestHandler

export const GET: RequestHandler = async () => {
  const totalMem = os.totalmem();
  const freeMem = os.freemem();
  const usedMem = totalMem - freeMem;
  
  return json({
    platform: os.platform(),
    arch: os.arch(),
    cpus: os.cpus().length,
    gpuInfo: 'NVIDIA RTX 3060 12GB', // Would need actual detection
    memoryUsage: `${Math.round(usedMem / 1024 / 1024 / 1024)}GB / ${Math.round(totalMem / 1024 / 1024 / 1024)}GB`,
    nodeVersion: process.version,
    uptime: os.uptime()
  });
};
