// workers/services/native-windows-setup.js
import { execSync, spawn } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class WindowsNativeServices {
  constructor() {
    this.servicesDir = path.join(__dirname, '..', '..', 'services');
    this.services = {
      redis: null,
      qdrant: null,
      neo4j: null,
      minio: null,
      ollama: null
    };
  }

  async ensureServicesDirectory() {
    try {
      await fs.mkdir(this.servicesDir, { recursive: true });
      console.log('✅ Services directory created');
    } catch (error) {
      console.log('📁 Services directory already exists');
    }
  }

  async downloadFile(url, outputPath) {
    console.log(`📥 Downloading ${url}...`);
    
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const buffer = await response.arrayBuffer();
      await fs.writeFile(outputPath, Buffer.from(buffer));
      console.log(`✅ Downloaded to ${outputPath}`);
    } catch (error) {
      console.error(`❌ Failed to download ${url}:`, error.message);
      throw error;
    }
  }

  async setupRedis() {
    const redisPath = path.join(this.servicesDir, 'redis-server.exe');
    
    if (await this.fileExists(redisPath)) {
      console.log('✅ Redis already installed');
      return;
    }

    console.log('🔴 Setting up Redis for Windows...');
    
    try {
      // Download Redis for Windows (from Microsoft Archive)
      await this.downloadFile(
        'https://github.com/microsoftarchive/redis/releases/download/win-3.0.504/Redis-x64-3.0.504.zip',
        path.join(this.servicesDir, 'redis.zip')
      );
      
      // Extract using PowerShell
      execSync(`powershell -command "Expand-Archive -Path '${path.join(this.servicesDir, 'redis.zip')}' -DestinationPath '${this.servicesDir}'"`, {
        stdio: 'inherit'
      });
      
      // Clean up zip
      await fs.unlink(path.join(this.servicesDir, 'redis.zip'));
      
      console.log('✅ Redis setup complete');
    } catch (error) {
      console.error('❌ Redis setup failed:', error.message);
    }
  }

  async setupQdrant() {
    const qdrantPath = path.join(this.servicesDir, 'qdrant.exe');
    
    if (await this.fileExists(qdrantPath)) {
      console.log('✅ Qdrant already installed');
      return;
    }

    console.log('🔍 Setting up Qdrant for Windows...');
    
    try {
      await this.downloadFile(
        'https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-pc-windows-msvc.zip',
        path.join(this.servicesDir, 'qdrant.zip')
      );
      
      execSync(`powershell -command "Expand-Archive -Path '${path.join(this.servicesDir, 'qdrant.zip')}' -DestinationPath '${this.servicesDir}'"`, {
        stdio: 'inherit'
      });
      
      await fs.unlink(path.join(this.servicesDir, 'qdrant.zip'));
      
      // Create Qdrant config
      const qdrantConfig = {
        log_level: 'INFO',
        storage: {
          storage_path: './qdrant_storage'
        },
        service: {
          http_port: 6333,
          grpc_port: 6334
        }
      };
      
      await fs.writeFile(
        path.join(this.servicesDir, 'qdrant-config.yaml'),
        JSON.stringify(qdrantConfig, null, 2)
      );
      
      console.log('✅ Qdrant setup complete');
    } catch (error) {
      console.error('❌ Qdrant setup failed:', error.message);
    }
  }

  async setupNeo4j() {
    const neo4jDir = path.join(this.servicesDir, 'neo4j');
    
    if (await this.fileExists(neo4jDir)) {
      console.log('✅ Neo4j already installed');
      return;
    }

    console.log('🕸️ Setting up Neo4j Community Edition...');
    
    try {
      await this.downloadFile(
        'https://dist.neo4j.org/neo4j-community-5.15.0-windows.zip',
        path.join(this.servicesDir, 'neo4j.zip')
      );
      
      execSync(`powershell -command "Expand-Archive -Path '${path.join(this.servicesDir, 'neo4j.zip')}' -DestinationPath '${this.servicesDir}'"`, {
        stdio: 'inherit'
      });
      
      // Rename to simpler name
      const extractedDir = path.join(this.servicesDir, 'neo4j-community-5.15.0');
      await fs.rename(extractedDir, neo4jDir);
      
      await fs.unlink(path.join(this.servicesDir, 'neo4j.zip'));
      
      // Configure Neo4j
      const neo4jConf = `
# Neo4j Evidence Processing Configuration
server.default_listen_address=0.0.0.0
server.http.listen_address=:7474
server.bolt.listen_address=:7687
server.http.advertised_address=:7474
server.bolt.advertised_address=:7687
dbms.security.auth_enabled=true
dbms.default_database=neo4j
server.memory.heap.initial_size=512m
server.memory.heap.max_size=1G
server.memory.pagecache.size=256m
`;
      
      await fs.writeFile(
        path.join(neo4jDir, 'conf', 'neo4j.conf'),
        neo4jConf.trim()
      );
      
      console.log('✅ Neo4j setup complete');
    } catch (error) {
      console.error('❌ Neo4j setup failed:', error.message);
    }
  }

  async setupMinIO() {
    const minioPath = path.join(this.servicesDir, 'minio.exe');
    
    if (await this.fileExists(minioPath)) {
      console.log('✅ MinIO already installed');
      return;
    }

    console.log('📦 Setting up MinIO for Windows...');
    
    try {
      await this.downloadFile(
        'https://dl.min.io/server/minio/release/windows-amd64/minio.exe',
        minioPath
      );
      
      // Create data directory
      await fs.mkdir(path.join(this.servicesDir, 'minio-data'), { recursive: true });
      
      console.log('✅ MinIO setup complete');
    } catch (error) {
      console.error('❌ MinIO setup failed:', error.message);
    }
  }

  async setupOllama() {
    const ollamaPath = path.join(this.servicesDir, 'ollama.exe');
    
    if (await this.fileExists(ollamaPath)) {
      console.log('✅ Ollama already installed');
      return;
    }

    console.log('🦙 Setting up Ollama for Windows...');
    
    try {
      await this.downloadFile(
        'https://ollama.com/download/ollama-windows-amd64.exe',
        ollamaPath
      );
      
      console.log('✅ Ollama setup complete');
      console.log('📝 To install models, run: ollama pull nomic-embed-text && ollama pull llama3.1');
    } catch (error) {
      console.error('❌ Ollama setup failed:', error.message);
      console.log('💡 You can download Ollama manually from: https://ollama.com/download');
    }
  }

  async fileExists(filePath) {
    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }

  async startService(serviceName) {
    const serviceConfig = {
      redis: {
        command: path.join(this.servicesDir, 'redis-server.exe'),
        args: [],
        name: 'Redis'
      },
      qdrant: {
        command: path.join(this.servicesDir, 'qdrant.exe'),
        args: ['--config-path', path.join(this.servicesDir, 'qdrant-config.yaml')],
        name: 'Qdrant'
      },
      neo4j: {
        command: path.join(this.servicesDir, 'neo4j', 'bin', 'neo4j.bat'),
        args: ['console'],
        name: 'Neo4j'
      },
      minio: {
        command: path.join(this.servicesDir, 'minio.exe'),
        args: ['server', path.join(this.servicesDir, 'minio-data'), '--console-address', ':9001'],
        name: 'MinIO',
        env: {
          MINIO_ROOT_USER: 'evidence',
          MINIO_ROOT_PASSWORD: 'evidence123'
        }
      },
      ollama: {
        command: path.join(this.servicesDir, 'ollama.exe'),
        args: ['serve'],
        name: 'Ollama'
      }
    };

    const config = serviceConfig[serviceName];
    if (!config) {
      throw new Error(`Unknown service: ${serviceName}`);
    }

    if (!(await this.fileExists(config.command))) {
      throw new Error(`Service executable not found: ${config.command}`);
    }

    console.log(`🚀 Starting ${config.name}...`);

    const process = spawn(config.command, config.args, {
      cwd: this.servicesDir,
      env: { ...process.env, ...config.env },
      stdio: 'pipe'
    });

    process.stdout.on('data', (data) => {
      console.log(`[${config.name}] ${data.toString().trim()}`);
    });

    process.stderr.on('data', (data) => {
      console.error(`[${config.name}] ${data.toString().trim()}`);
    });

    process.on('close', (code) => {
      console.log(`[${config.name}] Process exited with code ${code}`);
      this.services[serviceName] = null;
    });

    this.services[serviceName] = process;
    return process;
  }

  async stopService(serviceName) {
    const process = this.services[serviceName];
    if (process) {
      console.log(`🛑 Stopping ${serviceName}...`);
      process.kill('SIGTERM');
      this.services[serviceName] = null;
    }
  }

  async stopAllServices() {
    console.log('🛑 Stopping all services...');
    
    for (const serviceName of Object.keys(this.services)) {
      await this.stopService(serviceName);
    }
    
    // Also try to kill by process name
    const processNames = [
      'redis-server.exe',
      'qdrant.exe',
      'neo4j.exe',
      'minio.exe',
      'ollama.exe'
    ];
    
    for (const processName of processNames) {
      try {
        execSync(`taskkill /F /IM ${processName} /T`, { stdio: 'ignore' });
      } catch (error) {
        // Ignore errors - process might not be running
      }
    }
    
    console.log('✅ All services stopped');
  }

  async startAllServices() {
    console.log('🚀 Starting all Evidence Processing services...');
    
    try {
      // Start services in order (some depend on others)
      await this.startService('redis');
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      await this.startService('qdrant');
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      await this.startService('neo4j');
      await new Promise(resolve => setTimeout(resolve, 5000));
      
      await this.startService('minio');
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // Ollama is optional
      try {
        await this.startService('ollama');
      } catch (error) {
        console.log('⚠️ Ollama not available (optional service)');
      }
      
      console.log('✅ All services started successfully!');
      console.log('\n📋 Service URLs:');
      console.log('  • Qdrant: http://localhost:6333/dashboard');
      console.log('  • Neo4j: http://localhost:7474 (neo4j/neo4j)');
      console.log('  • MinIO: http://localhost:9001 (evidence/evidence123)');
      console.log('  • Ollama: http://localhost:11434 (if available)');
      
    } catch (error) {
      console.error('❌ Failed to start services:', error.message);
      await this.stopAllServices();
      throw error;
    }
  }

  async checkHealth() {
    const health = {
      redis: false,
      qdrant: false,
      neo4j: false,
      minio: false,
      ollama: false
    };

    // Check if processes are running
    for (const [serviceName, process] of Object.entries(this.services)) {
      health[serviceName] = process !== null && !process.killed;
    }

    // Additional network checks could be added here
    // For now, just check if processes are alive

    return health;
  }
}

// CLI interface
if (import.meta.url === `file://${process.argv[1]}`) {
  const services = new WindowsNativeServices();
  const command = process.argv[2];

  switch (command) {
    case 'setup':
      console.log('🔧 Setting up Windows native services...');
      await services.ensureServicesDirectory();
      await services.setupRedis();
      await services.setupQdrant();
      await services.setupNeo4j();
      await services.setupMinIO();
      await services.setupOllama();
      console.log('✅ Setup complete!');
      break;

    case 'start':
      await services.startAllServices();
      // Keep process alive
      process.on('SIGINT', async () => {
        await services.stopAllServices();
        process.exit(0);
      });
      break;

    case 'stop':
      await services.stopAllServices();
      break;

    case 'health':
      const health = await services.checkHealth();
      console.log('🏥 Service Health Check:');
      for (const [service, isHealthy] of Object.entries(health)) {
        console.log(`  ${isHealthy ? '✅' : '❌'} ${service}`);
      }
      break;

    default:
      console.log('Usage: node native-windows-setup.js [setup|start|stop|health]');
      break;
  }
}

export default WindowsNativeServices;
