import fs from 'fs';
import path from 'path';

const MAX_FILES = 60;
const root = process.argv[2] || process.cwd();
const listPath = path.join(root, 'scripts', 'july-bats.json');

async function readFirstLines(file, n=20) {
  try {
    const data = await fs.promises.readFile(file, 'utf8');
    return data.split(/\r?\n/).slice(0, n).join('\n');
  } catch (e) {
    return '';
  }
}

(async () => {
  const raw = await fs.promises.readFile(listPath, 'utf8');
  const arr = JSON.parse(raw);
  const out = [];
  for (let i=0;i<Math.min(arr.length, MAX_FILES);i++) {
    const it = arr[i];
    const preview = await readFirstLines(it.fullPath, 20);
    let action = 'archive';
    if (/start|launch|dev|dev:|run|start-server|start-dev|start-legal|LAUNCH|START/i.test(preview) || /start|launch|dev|start/i.test(path.basename(it.fullPath))) action = 'keep';
    if (it.size > 15000) action = 'review';
    out.push({ path: it.fullPath, mtime: it.mtime, size: it.size, preview: preview.split(/\r?\n/)[0]||'', recommended: action });
  }
  await fs.promises.writeFile(path.join(root,'scripts','july-bats-summary.json'), JSON.stringify(out, null, 2));
  console.log('wrote', out.length, 'summaries to scripts/july-bats-summary.json');
})();
