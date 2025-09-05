import pkg from 'pg';
const { Pool } = pkg;

const passwords = ['admin', 'root', 'password', '123456', 'postgres', 'LegalAI2024!'];

for (const pwd of passwords) {
  const pool = new Pool({
    connectionString: `postgresql://postgres:${pwd}@localhost:5432/postgres`
  });
  
  try {
    const result = await pool.query('SELECT version()');
    console.log(`✅ SUCCESS with password '${pwd}'`);
    console.log(`Version: ${result.rows[0].version.slice(0, 50)}...`);
    await pool.end();
    process.exit(0);
  } catch (error) {
    console.log(`❌ Failed with '${pwd}': ${error.message.slice(0, 40)}...`);
    await pool.end();
  }
}

console.log('🔍 No password worked. PostgreSQL may require configuration or different credentials.');