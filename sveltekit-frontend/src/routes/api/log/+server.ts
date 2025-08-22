
import { json } from '@sveltejs/kit';

// Simple stub for logQueue
const logQueue = {
  add: async (data: any) => {
    console.log('Queue operation (stub):', data);
    return Promise.resolve();
  }
};

export async function POST({ request }) {
  try {
    const logData = await request.json();
    console.log('Received log data:', logData);

    // Add the log data to the BullMQ queue
    await logQueue.add('processLog', logData);

    return json({ status: 'success', message: 'Log received and queued' }, { status: 200 });
  } catch (error) {
    console.error('Error processing log:', error);
    return json({ status: 'error', message: 'Failed to process log' }, { status: 500 });
  }
}