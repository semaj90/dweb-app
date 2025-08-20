import { json, type RequestHandler } from '@sveltejs/kit';

// Placeholder orchestrator route (repaired). Full multi-agent logic will be reinstated later.
interface WorkflowStatus { id: string; state: string; createdAt: string }

// In-memory mock store
const workflows: Record<string, WorkflowStatus> = {};

export const GET: RequestHandler = async ({ url }) => {
    const id = url.searchParams.get('workflowId');
    if (id) {
        const wf = workflows[id];
        if (!wf) return json({ success: false, error: 'Workflow not found' }, { status: 404 });
        return json({ success: true, workflow: wf });
    }
    return json({ success: true, workflows: Object.values(workflows) });
};

export const POST: RequestHandler = async ({ request }) => {
    const body = await request.json().catch(() => ({}));
    const action = body.action || 'create';
    switch (action) {
        case 'create': {
            const id = crypto.randomUUID();
            workflows[id] = { id, state: 'created', createdAt: new Date().toISOString() };
            return json({ success: true, workflowId: id });
        }
        case 'execute': {
            const id = body.workflowId;
            if (!id || !workflows[id]) return json({ success: false, error: 'Workflow ID invalid' }, { status: 400 });
            workflows[id].state = 'executed';
            return json({ success: true, workflowId: id, result: { status: 'ok' } });
        }
        default:
            return json({ success: false, error: 'Invalid action' }, { status: 400 });
    }
};

export const DELETE: RequestHandler = async ({ url }) => {
    const id = url.searchParams.get('workflowId');
    if (!id || !workflows[id]) return json({ success: false, error: 'Workflow ID required' }, { status: 400 });
    delete workflows[id];
    return json({ success: true, message: 'Workflow cancelled' });
};

export const prerender = false;