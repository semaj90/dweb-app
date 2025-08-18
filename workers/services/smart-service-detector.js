// workers/services/smart-service-detector.js
import { execSync, spawn } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';

class SmartServiceDetector {
  constructor() {
    this.detectedServices = {
      postgresql: { installed: false, running: false, version: null, port: 5432 },
      redis: { installed: false, running: false, version: null, port: 6379 },
      rabbitmq: { installed: false, running: false, version: null, port: 5672 },
      qdrant: { installed: false, running: false, version: null, port: 6333, portable: true },
      neo4j: { installed: false, running: false, version: null, port: 7474, portable: true },
      minio: { installed: false, running: false, version: null, port: 9000, portable: true },
      ollama: { installed: false, running: false, version: null, port: 11434, portable: true }
    };
    
    this.portMapping = {
      5432: 'postgresql',
      6379: 'redis', 
      5672: 'rabbitmq',
      6333: 'qdrant',
      7474: 'neo4j',
      9000: 'minio',
      11434: 'ollama'
    };
  }

  async detectAllServices() {
    console.log('🔍 Smart Service Detection Starting...\n');
    
    await this.detectPostgreSQL();
    await this.detectRedis();
    await this.detectRabbitMQ();
    await this.detectPortableServices();
    await this.checkRunningPorts();
    
    return this.detectedServices;
  }

  async detectPostgreSQL() {
    console.log('🗄️ Detecting PostgreSQL...');
    
    try {
      // Check if pg_config exists (PostgreSQL installed)
      const version = execSync('pg_config --version', { encoding: 'utf8', stdio: 'pipe' });
      this.detectedServices.postgresql.installed = true;
      this.detectedServices.postgresql.version = version.trim();
      console.log(`  ✅ Found: ${version.trim()}`);
      
      // Test connection with password 123456
      try {
        process.env.PGPASSWORD = '123456';
        execSync('psql -U postgres -c "SELECT 1"', { encoding: 'utf8', stdio: 'pipe' });
        this.detectedServices.postgresql.running = true;
        console.log('  ✅ Connection successful with password 123456');
      } catch (error) {
        console.log('  ⚠️ PostgreSQL installed but cannot connect with password 123456');
        console.log('     Please ensure PostgreSQL is running and password is correct');
      }
      
    } catch (error) {
      console.log('  ❌ PostgreSQL not found or not in PATH');
      console.log('     Install from: https://www.postgresql.org/download/windows/');
    }
  }

  async detectRedis() {
    console.log('🔴 Detecting Redis...');
    
    try {
      // Check for Redis CLI
      const version = execSync('redis-cli --version', { encoding: 'utf8', stdio: 'pipe' });
      this.detectedServices.redis.installed = true;
      this.detectedServices.redis.version = version.trim();
      console.log(`  ✅ Found: ${version.trim()}`);
      
      // Test Redis connection
      try {
        const result = execSync('redis-cli ping', { encoding: 'utf8', stdio: 'pipe', timeout: 3000 });
        if (result.trim() === 'PONG') {
          this.detectedServices.redis.running = true;
          console.log('  ✅ Redis server responding');
        }
      } catch (error) {
        console.log('  ⚠️ Redis installed but server not responding');
        
        // Try to start Redis service
        try {
          execSync('sc start Redis', { stdio: 'pipe' });
          console.log('  🔄 Attempted to start Redis service');
          
          // Wait and test again
          await new Promise(resolve => setTimeout(resolve, 3000));
          const retryResult = execSync('redis-cli ping', { encoding: 'utf8', stdio: 'pipe' });
          if (retryResult.trim() === 'PONG') {
            this.detectedServices.redis.running = true;
            console.log('  ✅ Redis service started successfully');
          }
        } catch (startError) {
          console.log('  ❌ Could not start Redis service');
        }
      }
      
    } catch (error) {
      console.log('  ❌ Redis not found');
      console.log('     Will use portable version');
    }
  }

  async detectRabbitMQ() {
    console.log('🐰 Detecting RabbitMQ...');
    
    try {
      // Check for RabbitMQ
      const version = execSync('rabbitmqctl version', { encoding: 'utf8', stdio: 'pipe' });
      this.detectedServices.rabbitmq.installed = true;
      this.detectedServices.rabbitmq.version = version.trim();
      console.log(`  ✅ Found: ${version.trim()}`);
      
      // Check if service is running
      try {
        const serviceStatus = execSync('sc query RabbitMQ', { encoding: 'utf8', stdio: 'pipe' });
        if (serviceStatus.includes('RUNNING')) {
          this.detectedServices.rabbitmq.running = true;
          console.log('  ✅ RabbitMQ service is running');
        } else {
          console.log('  ⚠️ RabbitMQ service not running');
          
          // Try to start the service
          try {
            execSync('sc start RabbitMQ', { stdio: 'pipe' });
            console.log('  🔄 Started RabbitMQ service');
            this.detectedServices.rabbitmq.running = true;
          } catch (startError) {
            console.log('  ❌ Could not start RabbitMQ service');
          }
        }
      } catch (error) {
        console.log('  ⚠️ RabbitMQ installed but service not accessible');
      }
      
    } catch (error) {
      console.log('  ❌ RabbitMQ not found');
      console.log('     Install from: https://www.rabbitmq.com/download.html');
    }
  }

  async detectPortableServices() {
    console.log('📦 Detecting portable services...');
    
    const portableServices = ['qdrant', 'neo4j', 'minio', 'ollama'];
    const servicesDir = path.join(process.cwd(), 'services');
    
    for (const service of portableServices) {
      try {
        let executablePath;
        
        switch (service) {
          case 'qdrant':
            executablePath = path.join(servicesDir, 'qdrant.exe');
            break;
          case 'neo4j':
            executablePath = path.join(servicesDir, 'neo4j', 'bin', 'neo4j.bat');
            break;
          case 'minio':
            executablePath = path.join(servicesDir, 'minio.exe');
            break;
          case 'ollama':
            executablePath = path.join(servicesDir, 'ollama.exe');
            break;
        }
        
        const exists = await this.fileExists(executablePath);
        if (exists) {
          this.detectedServices[service].installed = true;
          console.log(`  ✅ ${service}: Portable installation found`);
          
          // Try to get version info
          if (service === 'minio') {
            try {
              const version = execSync(`"${executablePath}" --version`, { encoding: 'utf8', stdio: 'pipe' });
              this.detectedServices[service].version = version.trim();
            } catch (error) {
              this.detectedServices[service].version = 'Unknown';
            }
          }
        } else {
          console.log(`  ❌ ${service}: Not installed`);
        }
      } catch (error) {
        console.log(`  ❌ ${service}: Detection failed`);
      }
    }
  }

  async checkRunningPorts() {
    console.log('🌐 Checking running services on ports...');
    
    try {
      const netstatOutput = execSync('netstat -an', { encoding: 'utf8', stdio: 'pipe' });
      const lines = netstatOutput.split('\n');
      
      for (const line of lines) {
        if (line.includes('LISTENING')) {
          for (const [port, serviceName] of Object.entries(this.portMapping)) {
            if (line.includes(`:${port} `)) {
              this.detectedServices[serviceName].running = true;
              console.log(`  ✅ ${serviceName}: Running on port ${port}`);
            }
          }
        }
      }
    } catch (error) {
      console.log('  ⚠️ Could not check ports via netstat');
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

  generateServiceReport() {
    console.log('\n📋 SERVICE DETECTION REPORT');
    console.log('===========================\n');
    
    for (const [serviceName, info] of Object.entries(this.detectedServices)) {
      const status = info.running ? '🟢 RUNNING' : 
                    info.installed ? '🟡 INSTALLED' : '🔴 NOT FOUND';
      
      console.log(`${serviceName.toUpperCase()}: ${status}`);
      if (info.version) {
        console.log(`  Version: ${info.version}`);
      }
      if (info.portable) {
        console.log(`  Type: Portable installation`);
      } else {
        console.log(`  Type: System installation`);
      }
      console.log(`  Port: ${info.port}`);
      console.log('');
    }
  }

  generateStartupStrategy() {
    console.log('🚀 STARTUP STRATEGY');
    console.log('==================\n');
    
    const systemServices = [];
    const portableServices = [];
    const missingServices = [];
    
    for (const [serviceName, info] of Object.entries(this.detectedServices)) {
      if (info.running) {
        console.log(`✅ ${serviceName}: Already running`);
      } else if (info.installed && !info.portable) {
        systemServices.push(serviceName);
      } else if (info.installed && info.portable) {
        portableServices.push(serviceName);
      } else {
        missingServices.push(serviceName);
      }
    }
    
    if (systemServices.length > 0) {
      console.log(`\n🔧 System services to start: ${systemServices.join(', ')}`);
    }
    
    if (portableServices.length > 0) {
      console.log(`📦 Portable services to start: ${portableServices.join(', ')}`);
    }
    
    if (missingServices.length > 0) {
      console.log(`❌ Missing services: ${missingServices.join(', ')}`);
    }
  }

  getConnectionStrings() {
    const connections = {};
    
    if (this.detectedServices.postgresql.running) {
      connections.database = 'postgresql://postgres:123456@localhost:5432/evidence_processing';
    }
    
    if (this.detectedServices.redis.running) {
      connections.redis = 'redis://localhost:6379';
    }
    
    if (this.detectedServices.rabbitmq.running) {
      connections.rabbitmq = 'amqp://guest:guest@localhost:5672';
    }
    
    connections.qdrant = 'http://localhost:6333';
    connections.neo4j = 'bolt://localhost:7687';
    connections.minio = 'http://localhost:9000';
    connections.ollama = 'http://localhost:11434';
    
    return connections;
  }
}

// CLI usage
if (import.meta.url === `file://${process.argv[1]}`) {
  const detector = new SmartServiceDetector();
  
  try {
    await detector.detectAllServices();
    detector.generateServiceReport();
    detector.generateStartupStrategy();
    
    const connections = detector.getConnectionStrings();
    console.log('\n🔗 CONNECTION STRINGS');
    console.log('=====================');
    for (const [service, url] of Object.entries(connections)) {
      console.log(`${service}: ${url}`);
    }
    
  } catch (error) {
    console.error('❌ Detection failed:', error.message);
    process.exit(1);
  }
}

export default SmartServiceDetector;
