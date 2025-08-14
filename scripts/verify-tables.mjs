import pg from 'pg';

const { Client } = pg;

async function main() {
  const conn = process.env.DATABASE_URL;
  if (!conn) {
    console.error('DATABASE_URL not set');
    process.exit(1);
  }
  const client = new Client({ connectionString: conn });
  try {
    await client.connect();
    const reg = await client.query(
      "SELECT to_regclass('public.document_metadata') AS dm, to_regclass('public.document_embeddings') AS de"
    );
    const dmReg = reg.rows?.[0]?.dm;
    const deReg = reg.rows?.[0]?.de;

    let dmCount = null;
    let deCount = null;
    try {
      const c1 = await client.query('SELECT count(*)::int AS n FROM public.document_metadata');
      dmCount = c1.rows?.[0]?.n ?? null;
    } catch (e) {
      dmCount = `error: ${e.message}`;
    }
    try {
      const c2 = await client.query('SELECT count(*)::int AS n FROM public.document_embeddings');
      deCount = c2.rows?.[0]?.n ?? null;
    } catch (e) {
      deCount = `error: ${e.message}`;
    }

    console.log(JSON.stringify({ tables: { document_metadata: dmReg, document_embeddings: deReg }, counts: { document_metadata: dmCount, document_embeddings: deCount } }, null, 2));
  } finally {
    await client.end().catch(() => {});
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
