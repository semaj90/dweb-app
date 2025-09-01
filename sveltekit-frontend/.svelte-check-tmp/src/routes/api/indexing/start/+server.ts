import { json } from "@sveltejs/kit";

import fg from 'fast-glob'; // You might need to 'npm install fast-glob'
import axios from 'axios'; // You might need to 'npm install axios'
import type { RequestHandler } from './$types';


const GO_FILE_PROCESSOR_URL = 'http://localhost:8081/process-files'; // Assuming Go service runs on 8081

export async function POST({ request }): Promise<any> {
    try {
        const { directoryPath } = await request.json();

        if (!directoryPath) {
            return json({ status: 'error', message: 'directoryPath is required' }, { status: 400 });
        }

        console.log(`Starting indexing for directory: ${directoryPath}`);

        // 1. Scan files using fast-glob
        const files = await fg(`${directoryPath}/**/*`, {
            dot: true, // Include dotfiles
            onlyFiles: true, // Only return files, not directories
            ignore: ['**/node_modules/**', '**/.git/**', '**/dist/**', '**/build/**'] // Common ignores
        });

        console.log(`Found ${files.length} files. Delegating to Go microservice...`);

        // 2. Delegate to Go microservice
        const goResponse = await axios.post(GO_FILE_PROCESSOR_URL, { files });

        if (goResponse.status === 200) {
            console.log('Go microservice successfully processed files.');
            return json({ status: 'success', message: 'Indexing initiated and delegated to Go microservice', details: goResponse.data }, { status: 200 });
        } else {
            console.error('Go microservice returned an error:', goResponse.data);
            return json({ status: 'error', message: 'Go microservice failed to process files', details: goResponse.data }, { status: goResponse.status });
        }

    } catch (error: any) {
        console.error('Error in indexing start endpoint:', error);
        return json({ status: 'error', message: 'Failed to initiate indexing', error: error.message }, { status: 500 });
    }
}