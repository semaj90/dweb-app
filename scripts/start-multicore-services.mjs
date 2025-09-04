#!/usr/bin/env zx
/**
 * start-multicore-services.mjs
 * Parallel launcher for Enhanced Multicore Go Service + Context7 MCP Multi-Core Server
 * Cross-platform (requires Go + Node). Uses zx for concise async orchestration.
 */
import 'zx/globals';

const args = minimist(process.argv.slice(2), {
  string: ['goPort','mcpPort','workers','logDir'],
  boolean: ['gpu','watch','debug'],
  default: {
    goPort: process.env.MULTICORE_PORT || '8098',
    mcpPort: process.env.MCP_PORT || '4100',
    workers: '0',
    gpu: true,
    watch: false,
    debug: false,
    logDir: 'logs'
  }
});

function log(tag, msg, color='white') {
  const ts = new Date().toISOString().split('T')[1].replace('Z','');
  const c = chalk[color] || (x=>x);
  console.log(c(`[${ts}][${tag}] ${msg}`));
}

async function portFree(port){
  try { await fetch(`http://localhost:${port}/__unlikely__`); return false; } catch { return true; }
}

async function findFree(start, max=20){
  for(let i=0;i<max;i++){ const p = start + i; if(await portFree(p)) return p; }
  throw new Error(`No free port near ${start}`);
}

(async () => {
  const repoRoot = path.resolve(path.dirname(process.argv[1]), '..');
  const goSrc = path.join(repoRoot, 'go-microservice', 'enhanced-multicore-service.go');
  const goBinDir = path.join(repoRoot, 'go-microservice', 'bin');
  const goExe = path.join(goBinDir, process.platform === 'win32' ? 'multicore-service.exe' : 'multicore-service');
  const mcpJs = path.join(repoRoot, 'context7-mcp-server-multicore.js');
  const logsDir = path.join(repoRoot, args.logDir);
  await fs.ensureDir(goBinDir); await fs.ensureDir(logsDir);

  if(!fs.existsSync(goSrc)) throw new Error(`Missing Go source: ${goSrc}`);
  if(!fs.existsSync(mcpJs)) throw new Error(`Missing MCP server script: ${mcpJs}`);

  // Resolve ports
  let goPort = parseInt(args.goPort,10); if(!(await portFree(goPort))) goPort = await findFree(goPort);
  let mcpPort = parseInt(args.mcpPort,10); if(!(await portFree(mcpPort))) mcpPort = await findFree(mcpPort);

  // Workers
  const cpuCount = os.cpus().length; let workers = parseInt(args.workers,10); if(!workers) workers = Math.min(cpuCount,8);

  log('BUILD','Building Go multicore service','cyan');
  try {
    await $`go build -o ${goExe} ${goSrc}`;
    log('BUILD','Go build complete','green');
  } catch (e) {
    log('BUILD',`Go build failed: ${e.stderr || e.message}`,'red'); process.exit(1);
  }

  // Environment
  const commonEnv = {
    ...process.env,
    MULTICORE_PORT: String(goPort),
    ENABLE_GPU: String(!!args.gpu),
    MCP_PORT: String(mcpPort),
    MCP_MULTICORE: 'true',
    MCP_DEBUG: String(!!args.debug)
  };

  const goLog = path.join(logsDir,'multicore-go.log');
  const mcpLog = path.join(logsDir,'mcp-multicore.log');

  // Spawn processes
  log('PROC',`Starting Go service on :${goPort}`,'green');
  const goProc = $.spawn(goExe, [], { stdio: ['ignore','pipe','pipe'], env: commonEnv });
  goProc.stdout.pipe(fs.createWriteStream(goLog));
  goProc.stderr.pipe(fs.createWriteStream(goLog,{flags:'a'}));

  log('PROC',`Starting MCP server on :${mcpPort}`,'green');
  const nodePath = process.execPath;
  const mcpProc = $.spawn(nodePath, [mcpJs], { stdio:['ignore','pipe','pipe'], env: commonEnv });
  mcpProc.stdout.pipe(fs.createWriteStream(mcpLog));
  mcpProc.stderr.pipe(fs.createWriteStream(mcpLog,{flags:'a'}));

  // Health poll helper
  async function waitHealth(name,url,retries=40,delay=500){
    for(let i=0;i<retries;i++){
      try { const r = await fetch(url,{method:'GET'}); if(r.ok){ log('HEALTH',`${name} OK (${r.status})`,'green'); return true; } } catch {}
      await new Promise(r=>setTimeout(r,delay));
    }
    log('HEALTH',`${name} FAILED`,'red'); return false;
  }

  const goHealthy = await waitHealth('GO',`http://localhost:${goPort}/api/health`);
  const mcpHealthy = await waitHealth('MCP',`http://localhost:${mcpPort}/health`);

  if(!(goHealthy && mcpHealthy)){
    log('SYS','One or more services failed health checks; displaying tails','yellow');
    const tail = async (p,label)=>{ if(fs.existsSync(p)){ const data = (await fs.readFile(p,'utf8')).trim().split(/\r?\n/).slice(-50).join('\n'); log('TAIL',`---- ${label} ----\n${data}`,'gray'); } };
    await tail(goLog,'Go'); await tail(mcpLog,'MCP');
    goProc.kill('SIGTERM'); mcpProc.kill('SIGTERM');
    process.exit(1);
  }

  log('READY',`Both services ready (Go:${goPort}, MCP:${mcpPort})`,'cyan');
  log('INFO','Tailing logs (Ctrl+C to exit)','cyan');

  const tailFile = (file,label)=>{
    fs.watchFile(file,{interval:1000}, async ()=>{
      const data = (await fs.readFile(file,'utf8')).trim().split(/\r?\n/).slice(-10).join('\n');
      log(label,data,'gray');
    });
  };
  tailFile(goLog,'GOLOG');
  tailFile(mcpLog,'MCPLOG');

  function shutdown(){
    log('SHUTDOWN','Stopping services...','yellow');
    goProc.kill('SIGTERM');
    mcpProc.kill('SIGTERM');
    setTimeout(()=>process.exit(0),1000);
  }

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);

  if(args.watch){
    log('WATCH','Watch enabled for Go sources','yellow');
    const chokidar = (await import('chokidar')).default;
    const watcher = chokidar.watch(path.dirname(goSrc), { ignoreInitial:true, persistent:true });
    let rebuildScheduled = false;
    watcher.on('all', () => {
      if(rebuildScheduled) return; rebuildScheduled = true;
      setTimeout(async ()=>{
        log('WATCH','Rebuilding Go service...','yellow');
        try { await $`go build -o ${goExe} ${goSrc}`; log('WATCH','Rebuild OK - restarting','green'); goProc.kill('SIGTERM'); } catch(e){ log('WATCH',`Rebuild failed: ${e.stderr||e.message}`,'red'); }
        rebuildScheduled = false;
      },300);
    });
  }
})();
