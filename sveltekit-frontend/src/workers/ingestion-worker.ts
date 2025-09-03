import { Client } from 'pg';

async function splitIntoSections(text: string, maxChars = 2000) {
	if (!text) return [];
	const paragraphs = text.split(/\n\s*\n/).map(p => p.trim()).filter(Boolean);
	const sections: string[] = [];
	for (const p of paragraphs) {
		if (p.length <= maxChars) {
			sections.push(p);
		} else {
			for (let i = 0; i < p.length; i += maxChars) {
				sections.push(p.slice(i, i + maxChars));
			}
		}
	}
	if (sections.length === 0 && text.trim()) sections.push(text.trim());
	return sections;
}

async function main() {
	const databaseUrl = process.env.DATABASE_URL || process.env.DATABASE || 'postgresql://postgres:123456@localhost:5432/legal_ai_db';
	console.log('Starting ingestion worker using', databaseUrl.replace(/:(\\\w+?)@/, ':****@'));

	const client = new Client({ connectionString: databaseUrl });
	await client.connect();

	try {
		const res = await client.query(`SELECT id, title, full_text FROM public.legal_documents ORDER BY created_at DESC LIMIT 5`);
		console.log(`Found ${res.rowCount} documents to process`);

		for (const row of res.rows) {
			const docId = row.id;
			const title = row.title || '';
			const text = row.full_text || '';
			const sections = await splitIntoSections(text, 2000);
			console.log(`Processing document ${docId} (${title}) -> ${sections.length} sections`);

			let sectionIndex = 0;
			for (const content of sections) {
				await client.query(
					`INSERT INTO public.document_sections (document_id, section_index, title, content, content_tokens, created_at) VALUES ($1, $2, $3, $4, $5, NOW())`,
					[docId, sectionIndex, title, content, null]
				);
				sectionIndex += 1;
			}

			// Emit a NOTIFY so other workers can react (payload is simple id JSON)
			try {
				const payload = JSON.stringify({ document_id: docId, sections: sections.length });
				await client.query(`NOTIFY ingest_completed, $1`, [payload]);
			} catch (e) {
				// Some Postgres clients don't allow parameterized NOTIFY; fallback
				try { await client.query(`NOTIFY ingest_completed, '${docId}'`); } catch (_) { /* ignore */ }
			}
		}

		console.log('Ingestion worker finished successfully');
	} catch (err) {
		console.error('Ingestion worker error:', err?.message ?? err);
		process.exitCode = 1;
	} finally {
		await client.end();
	}
}

main().catch(e => {
	console.error('Fatal error in worker:', e?.message ?? e);
	process.exit(1);
});

