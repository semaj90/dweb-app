#!/usr/bin/env node
const testConnection = async () => {
  try {
    console.log('Testing connection...');
    const { testDatabaseConnection } = await import('../src/lib/database/connection.js');
    const result = await testDatabaseConnection();
    console.log('DB:', result.success ? '✅' : '❌');
  } catch (e) {
    console.log('DB: ❌', e.message);
  }
};

testConnection();
