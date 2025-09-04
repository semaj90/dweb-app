import fs from 'fs';
import path from 'path';

const root = process.argv[2] || process.cwd();
const results = [];

async function walk(dir) {
  let entries;
  try {
    entries = await fs.promises.readdir(dir, { withFileTypes: true });
  } catch (err) {
    return; // skip unreadable dirs
  }

  for (const ent of entries) {
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) {
      await walk(full);
    } else if (ent.isFile() && path.extname(ent.name).toLowerCase() === '.bat') {
      try {
        const st = await fs.promises.stat(full);
        const m = st.mtime;
        if (m.getMonth() === 6) { // July -> month index 6
          results.push({ fullPath: full, mtime: m.toISOString(), size: st.size });
        }
      } catch (e) {
        // ignore file stat errors
      }
    }
  }
}

(async () => {
  await walk(root);
  console.log(JSON.stringify(results, null, 2));
})();
