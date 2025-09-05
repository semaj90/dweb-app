#!/usr/bin/env zx
/**
 * Cluster Orchestrator using zx
 * - Launch SvelteKit cluster (frontend)
 * - Launch API gateway cluster
 * - Monitor processes; restart on unexpected exit (basic supervisor)
 * - Optional concurrency flags via env: CLUSTER_WORKERS, GATEWAY_WORKERS
 */
import 'zx/globals';

$.verbose = false;

const ROOT = path.resolve(process.cwd());
const FRONTEND_DIR = path.join(ROOT,'sveltekit-frontend');
const GATEWAY_CLUSTER_FILE = path.join(ROOT,'api-gateway','src','cluster.js');
const FRONTEND_CLUSTER_FILE = path.join(FRONTEND_DIR,'cluster.js');

if(!fs.existsSync(FRONTEND_CLUSTER_FILE)){
  console.error('Missing frontend cluster.js at', FRONTEND_CLUSTER_FILE); // do not exit yet (optional)
}
if(!fs.existsSync(GATEWAY_CLUSTER_FILE)){
  console.error('Missing gateway cluster.js at', GATEWAY_CLUSTER_FILE); process.exit(1);
}

const FRONTEND_CMD = `node ${FRONTEND_CLUSTER_FILE}`;
const GATEWAY_CMD = `node ${GATEWAY_CLUSTER_FILE}`;

const procs = new Map();

function log(msg){ console.log(new Date().toISOString(),'[orchestrator]',msg); }

async function spawnProc(name, cmd, cwd){
  log(`Starting ${name}: ${cmd} (cwd=${cwd})`);
  const parts = cmd.split(/\s+/);
  const p = $.spawn(parts[0], parts.slice(1), {cwd, stdio: 'inherit', windowsHide: true});
  procs.set(name,p);
  p.on('exit', code => {
    log(`${name} exited with code ${code}`);
    if(code !== 0){
      setTimeout(()=>spawnProc(name, cmd, cwd), 2000);
    }
  });
}

async function ensureGatewayDeps(){
  const depPath = path.join(ROOT,'api-gateway','node_modules','http-proxy-middleware');
  if(!fs.existsSync(depPath)){
    log('api-gateway dependencies missing. Attempting install (workspaces aware)...');
    try {
      await $`npm install --workspaces=false --prefix ${path.join(ROOT,'api-gateway')}`;
      log('api-gateway local install complete');
    } catch(err){
      log('Install attempt failed, proceeding (gateway may crash): '+ err.message);
    }
  }
}

async function verifyFrontend(){
  if(!fs.existsSync(FRONTEND_CLUSTER_FILE)) return false;
  const archFile = path.join(FRONTEND_DIR,'src','lib','services','nodejs-cluster-architecture.js');
  if(!fs.existsSync(archFile)){
    log('Frontend cluster architecture JS missing; will skip frontend cluster (TS version exists but not compiled).');
    return false;
  }
  return true;
}

async function main(){
  await ensureGatewayDeps();
  const frontendOk = await verifyFrontend();

  if(process.env.CLUSTER_WORKERS) process.env.CLUSTER_WORKERS = process.env.CLUSTER_WORKERS;
  if(process.env.GATEWAY_WORKERS) process.env.WORKERS = process.env.GATEWAY_WORKERS; // gateway cluster.js expects WORKERS

  const tasks = [spawnProc('api-gateway', GATEWAY_CMD, ROOT)];
  if(frontendOk){
    tasks.push(spawnProc('frontend-cluster', FRONTEND_CMD, FRONTEND_DIR));
  } else {
    log('Skipping frontend cluster start (compile missing). Run build or tsc to generate JS.');
  }
  await Promise.all(tasks);

  setInterval(()=>{
    for(const [name,p] of procs.entries()){
      if(p.killed) log(`WARN: ${name} process object marked killed`);
    }
  }, 15000);
}

process.on('SIGINT', ()=>{ log('SIGINT received, terminating children...'); for(const p of procs.values()) p.kill(); process.exit(0); });
process.on('SIGTERM', ()=>{ log('SIGTERM received, terminating children...'); for(const p of procs.values()) p.kill(); process.exit(0); });

main().catch(e=>{ console.error('Orchestrator failed:', e); process.exit(1); });