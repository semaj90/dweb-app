const fs = require('fs');
const glob = require('glob');

glob('**/+server.ts', { ignore: ['node_modules/**', '.svelte-kit/**'] }, (err, files) => {
  if (err) throw err;
  files.forEach(f => {
    const s = fs.readFileSync(f, 'utf8');
    const n = (s.match(/RequestHandler/g) || []).length;
    if (n > 1) console.log(`${f} -> ${n}`);
  });
});
