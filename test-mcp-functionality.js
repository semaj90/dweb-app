#!/usr/bin/env node

/**
 * Test MCP Server Functionality
 */

import { spawn } from 'child_process';

const testCommands = [
  {
    name: 'List Tools',
    command: { jsonrpc: '2.0', id: 1, method: 'tools/list', params: {} }
  },
  {
    name: 'List Resources', 
    command: { jsonrpc: '2.0', id: 2, method: 'resources/list', params: {} }
  },
  {
    name: 'Check Services',
    command: { 
      jsonrpc: '2.0', 
      id: 3, 
      method: 'tools/call', 
      params: { 
        name: 'check_services',
        arguments: {}
      }
    }
  },
  {
    name: 'Get Context7 Status',
    command: { 
      jsonrpc: '2.0', 
      id: 4, 
      method: 'tools/call', 
      params: { 
        name: 'get_context7_status',
        arguments: {}
      }
    }
  }
];

function testMCPServer() {
  console.log('🧪 Testing MCP Server Functionality\n');

  const serverProcess = spawn('node', ['mcp-servers/mcp-context7-wrapper.js'], {
    cwd: process.cwd(),
    env: {
      ...process.env,
      PROJECT_ROOT: 'C:\\Users\\james\\Desktop\\deeds-web\\deeds-web-app',
      OLLAMA_ENDPOINT: 'http://localhost:11434',
      DATABASE_URL: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db'
    }
  });

  let currentTest = 0;

  serverProcess.stdout.on('data', (data) => {
    try {
      const response = JSON.parse(data.toString());
      const test = testCommands[currentTest - 1];
      
      console.log(`✅ ${test.name}: Success`);
      if (response.result) {
        if (response.result.tools) {
          console.log(`   Found ${response.result.tools.length} tools`);
        }
        if (response.result.resources) {
          console.log(`   Found ${response.result.resources.length} resources`);
        }
        if (response.result.content) {
          console.log(`   Returned content: ${response.result.content[0].text.substring(0, 100)}...`);
        }
      }
      console.log('');
      
      // Send next test
      if (currentTest < testCommands.length) {
        sendNextTest();
      } else {
        console.log('🎉 All MCP tests completed successfully!');
        serverProcess.kill();
      }
    } catch (error) {
      console.log(`📝 Server output: ${data.toString().trim()}`);
    }
  });

  serverProcess.stderr.on('data', (data) => {
    const message = data.toString().trim();
    if (message.includes('Context7 MCP Server ready')) {
      console.log('🚀 MCP Server ready, starting tests...\n');
      sendNextTest();
    } else if (!message.includes('Debugger')) {
      console.log(`ℹ️ Server: ${message}`);
    }
  });

  serverProcess.on('exit', (code) => {
    console.log(`\nMCP Server process exited with code ${code}`);
  });

  function sendNextTest() {
    if (currentTest >= testCommands.length) return;
    
    const test = testCommands[currentTest];
    console.log(`🔍 Testing: ${test.name}`);
    
    serverProcess.stdin.write(JSON.stringify(test.command) + '\n');
    currentTest++;
  }
}

testMCPServer();