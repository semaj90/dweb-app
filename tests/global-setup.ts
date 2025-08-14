import { chromium, FullConfig } from '@playwright/test';

async function globalSetup(config: FullConfig) {
  console.log('🚀 Legal AI System - Global Test Setup');
  
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();
  
  try {
    const services = [
      { name: 'SvelteKit Frontend', url: 'http://localhost:5173', timeout: 30000 },
      { name: 'Go Upload Service', url: 'http://localhost:8093/health', timeout: 10000 },
      { name: 'Ollama LLM', url: 'http://localhost:11434/api/version', timeout: 10000 }
    ];
    
    for (const service of services) {
      console.log(`⏳ Checking ${service.name}...`);
      try {
        const response = await page.goto(service.url, { 
          waitUntil: 'networkidle',
          timeout: service.timeout 
        });
        
        if (response?.status() === 200 || response?.status() === 403) {
          console.log(`✅ ${service.name} is available`);
        } else {
          console.log(`⚠️  ${service.name} returned status ${response?.status()}`);
        }
      } catch (error) {
        console.log(`❌ ${service.name} is not available`);
      }
    }
    
    console.log('🎯 Global setup completed');
    
  } catch (error) {
    console.error('❌ Global setup failed:', error);
    throw error;
  } finally {
    await browser.close();
  }
}

export default globalSetup;