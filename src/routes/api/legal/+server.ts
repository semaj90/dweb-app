import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const GO_BACKEND = 'http://localhost:8080';

export const POST: RequestHandler = async ({ request }) => {
    const { endpoint, ...payload } = await request.json();
    
    const response = await fetch(`${GO_BACKEND}/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });

    if (!response.ok) {
        const error = await response.text();
        throw new Error(`Backend error: ${response.status} ${error}`);
    }

    return json(await response.json());
};

export const GET: RequestHandler = async ({ url }) => {
    const endpoint = url.searchParams.get('endpoint') || 'health';
    
    const response = await fetch(`${GO_BACKEND}/${endpoint}`);
    return json(await response.json());
};