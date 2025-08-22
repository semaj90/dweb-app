
// src/routes/api/documents/+server.ts
import { type RequestHandler,  json } from '@sveltejs/kit';
import { URL } from "url";

const GO_SERVICE = 'http://localhost:8080';

export const POST: RequestHandler = async ({ request }) => {
	const body = await request.json();
	
	const response = await fetch(`${GO_SERVICE}/process-document`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body)
	});
	
	return json(await response.json());
};

export const GET: RequestHandler = async ({ url }) => {
	const query = url.searchParams.get('q') || '';
	const limit = parseInt(url.searchParams.get('limit') || '10');
	
	const response = await fetch(`${GO_SERVICE}/search-similar`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ query, limit })
	});
	
	return json(await response.json());
};
