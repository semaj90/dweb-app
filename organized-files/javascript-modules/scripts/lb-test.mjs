#!/usr/bin/env node
import http from 'node:http';

const base = process.env.LB_URL || 'http://localhost:8099';

function get(path){
  return new Promise((resolve,reject)=>{
    const req = http.get(base+path,res=>{
      let data='';
      res.on('data',d=>data+=d);
      res.on('end',()=> resolve({ status: res.statusCode, data }));
    });
    req.on('error',reject);
  });
}

(async () => {
  try {
    const status = await get('/status');
    if(status.status!==200) throw new Error('/status non-200');
    const prom = await get('/prometheus');
    console.log('✅ LB status OK, metrics length:', prom.data.length);
  } catch (e) {
    console.error('❌ LB test failed:', e.message);
    process.exit(1);
  }
})();
