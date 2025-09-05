#!/usr/bin/env node

/**
 * Auto-Switch Ollama Configuration
 * Automatically detects and switches between native and Docker Ollama
 */

class OllamaAutoSwitch {
    constructor() {
        this.configurations = [
            { name: 'Native', url: 'http://localhost:11434', priority: 1 },
            { name: 'Docker-Alt', url: 'http://localhost:11435', priority: 2 },
            { name: 'Docker-Std', url: 'http://localhost:8080', priority: 3 }
        ];
        this.activeConfig = null;
    }

    async detectActiveConfiguration() {
        console.log('🔍 Auto-detecting Ollama configuration...');

        for (const config of this.configurations) {
            try {
                const fetch = require('node-fetch');
                const response = await fetch(`${config.url}/api/tags`, { timeout: 3000 });
                if (response.ok) {
                    this.activeConfig = config;
                    console.log(`✅ Found active Ollama: ${config.name} on ${config.url}`);
                    return config;
                }
            } catch {
                console.log(`⚠️ ${config.name} not available on ${config.url}`);
            }
        }

        console.log('❌ No active Ollama configuration found');
        return null;
    }

    async updateMCPConfiguration() {
        if (!this.activeConfig) {
            console.log('❌ No active configuration to update');
            return;
        }

        console.log(`🔧 Updating MCP configuration to use ${this.activeConfig.name}...`);

        // Update environment variables or config files
        const envContent = `OLLAMA_BASE_URL=${this.activeConfig.url}
OLLAMA_HOST=${this.activeConfig.url}
MCP_OLLAMA_URL=${this.activeConfig.url}
`;

        const fs = require('fs').promises;
        await fs.writeFile('.env.ollama', envContent);
        console.log('✅ Updated .env.ollama with active configuration');
    }

    async startRecommendedSetup() {
        console.log('🚀 Starting recommended Ollama setup...');

        if (process.platform === 'win32') {
            const { spawn } = require('child_process');

            console.log('Starting Docker Ollama with alternative port...');
            const startScript = spawn('cmd', ['/c', 'START-RAG-ALT-PORT.bat'], {
                stdio: 'inherit',
                detached: true
            });

            console.log('✅ Started Docker Ollama setup');
        } else {
            console.log('Manual setup required for non-Windows platforms');
        }
    }
}

// Auto-run detection
const autoSwitch = new OllamaAutoSwitch();
autoSwitch.detectActiveConfiguration()
    .then(config => {
        if (config) {
            return autoSwitch.updateMCPConfiguration();
        } else {
            console.log('🚀 No active Ollama found, starting recommended setup...');
            return autoSwitch.startRecommendedSetup();
        }
    })
    .catch(console.error);
