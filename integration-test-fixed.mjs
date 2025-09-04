// INTEGRATION TEST FIX - Replace line 50-55 in integration-test.mjs

import { createConnection } from 'net';

async function checkPort(port, host = 'localhost', timeout = 5000) {
  return new Promise((resolve) => {
    const timeoutId = setTimeout(() => resolve(false), timeout);
    
    try {
      const socket = createConnection({ port, host });
      
      socket.setTimeout(timeout);
      socket.on('connect', () => {
        clearTimeout(timeoutId);
        socket.destroy();
        resolve(true);
      });
      
      socket.on('timeout', () => {
        clearTimeout(timeoutId);
        socket.destroy();
        resolve(false);
      });
      
      socket.on('error', () => {
        clearTimeout(timeoutId);
        socket.destroy();
        resolve(false);
      });
    } catch (e) {
      clearTimeout(timeoutId);
      resolve(false);
    }
  });
}
