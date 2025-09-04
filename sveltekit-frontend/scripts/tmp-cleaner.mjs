#!/usr/bin/env node
// Periodic cleanup of orphaned temp upload files older than a threshold
import { readdir, stat, unlink } from 'fs/promises';
import { join } from 'path';

const TMP_DIR = join('uploads', 'tmp');
const MAX_AGE_MIN = Number(process.env.TMP_MAX_AGE_MIN || 60); // default 60 minutes
const now = Date.now();
let removed = 0;

async function run() {
  try {
    const entries = await readdir(TMP_DIR, { withFileTypes: true });
    for (const ent of entries) {
      if (!ent.isFile()) continue;
      const full = join(TMP_DIR, ent.name);
      try {
        const s = await stat(full);
        const ageMin = (now - s.mtimeMs) / 60000;
        if (ageMin > MAX_AGE_MIN) {
          await unlink(full);
          removed++;
        }
      } catch {}
    }
    console.log(JSON.stringify({ success: true, removed, dir: TMP_DIR, maxAgeMin: MAX_AGE_MIN }));
  } catch (err) {
    console.error(JSON.stringify({ success: false, error: err.message }));
    process.exitCode = 1;
  }
}

run();
