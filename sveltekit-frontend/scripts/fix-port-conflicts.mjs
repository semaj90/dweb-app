#!/usr/bin/env node
/**
 * Port Conflict Resolver
 * Kills processes using conflicted ports and cleans up for dev:full
 */

import { exec } from 'child_process';
import { promisify } from 'util';
import chalk from 'chalk';

const execAsync = promisify(exec);

class PortConflictResolver {
  constructor() {
    this.conflictedPorts = [3005, 8080, 8094, 8095, 8099, 8101, 8102, 8103];
  }

  async resolveConflicts() {
    console.log(chalk.bold.yellow('🔧 RESOLVING PORT CONFLICTS\n'));

    try {
      // Kill Node.js processes that might be holding ports
      await this.killNodeProcesses();
      
      // Kill specific Go binaries
      await this.killGoProcesses();
      
      // Clean up any orphaned processes
      await this.cleanupOrphanedProcesses();
      
      console.log(chalk.bold.green('\n✅ Port conflicts resolved! Ready for npm run dev:full'));
      
    } catch (error) {
      console.error(chalk.red(`❌ Error resolving conflicts: ${error.message}`));
    }
  }

  async killNodeProcesses() {
    console.log(chalk.yellow('Stopping Node.js processes...'));
    
    try {
      // Kill node processes with common names
      const nodeCommands = [
        'taskkill /F /IM node.exe /T',
        'taskkill /F /IM "node.exe*" /T', 
        'taskkill /F /IM pnpm.exe /T'
      ];

      for (const cmd of nodeCommands) {
        try {
          await execAsync(cmd);
          console.log(chalk.green(`✅ ${cmd.split(' ')[3]} processes stopped`));
        } catch {
          // Process not found, that's ok
        }
      }
    } catch (error) {
      console.log(chalk.dim('No Node.js processes to stop'));
    }
  }

  async killGoProcesses() {
    console.log(chalk.yellow('Stopping Go service processes...'));
    
    const goServices = [
      'enhanced-rag.exe',
      'upload-service.exe', 
      'simple-upload-fixed.exe',
      'load-balancer.exe',
      'grpc-server.exe',
      'quic-gateway.exe'
    ];

    for (const service of goServices) {
      try {
        await execAsync(`taskkill /F /IM "${service}" /T`);
        console.log(chalk.green(`✅ ${service} stopped`));
      } catch {
        // Process not found, that's ok
      }
    }
  }

  async cleanupOrphanedProcesses() {
    console.log(chalk.yellow('Cleaning up orphaned processes...'));
    
    // Kill any processes using our specific ports
    for (const port of this.conflictedPorts) {
      try {
        const { stdout } = await execAsync(`netstat -ano | findstr ":${port}"`);
        if (stdout.trim()) {
          console.log(chalk.dim(`Port ${port} still in use, attempting cleanup...`));
          
          // Extract PIDs and kill them
          const lines = stdout.split('\n');
          for (const line of lines) {
            const parts = line.trim().split(/\s+/);
            const pid = parts[parts.length - 1];
            if (pid && /^\d+$/.test(pid)) {
              try {
                await execAsync(`taskkill /F /PID ${pid}`);
                console.log(chalk.green(`✅ Killed PID ${pid} on port ${port}`));
              } catch {
                // PID already gone
              }
            }
          }
        }
      } catch {
        // Port not in use, that's good
      }
    }
  }
}

// Run the resolver
const resolver = new PortConflictResolver();
resolver.resolveConflicts();