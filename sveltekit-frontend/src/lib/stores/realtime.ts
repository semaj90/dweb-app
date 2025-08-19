// Lightweight realtime pipeline store subscribing to ws-fanout events
import { writable, get, derived } from 'svelte/store';

export const connectionStatus = writable('disconnected');
export const stages = writable({}); // traceId -> { gpu, wasm, embedding, retrieval, llm, final }
export const finalResults = writable([]); // list of { id, llmResult, context, ts }
export const recentEvents = writable([]); // rolling window

let ws: WebSocket | null = null;

export function connectRealtime(url = 'ws://localhost:8080') {
	if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
	connectionStatus.set('connecting');
	try {
		ws = new WebSocket(url);
		ws.onopen = () => connectionStatus.set('connected');
		ws.onclose = () => { connectionStatus.set('disconnected'); setTimeout(()=>connectRealtime(url), 3000); };
		ws.onerror = () => connectionStatus.set('error');
		ws.onmessage = (ev) => {
			try {
				const data = JSON.parse(ev.data);
				handleEvent(data);
			} catch {}
		};
	} catch {
		connectionStatus.set('error');
	}
}

export function disconnectRealtime(){ if (ws){ ws.close(); ws = null; } }

function pushRecent(evt){
	recentEvents.update(list => { const next = [evt, ...list]; return next.slice(0,100); });
}

function handleEvent(wrapper){
	// wrapper shape { type, msg }
	const type = wrapper?.type;
	const msg = wrapper?.msg || {};
	pushRecent({ type, msg, at: Date.now() });
	if (type === 'ai.response') {
		const { id, stage, final } = msg;
		if (!id) return;
		stages.update(map => {
			const curr = map[id] || { id };
			if (stage) curr[stage] = true;
			if (final){ curr.final = true; curr.completedAt = Date.now(); }
			map[id] = curr; return { ...map };
		});
		if (final){
			finalResults.update(arr => [{ id, llmResult: msg.llmResult, context: msg.context, ts: Date.now() }, ...arr].slice(0,50));
		}
	} else if (type === 'evidence.upload'){ // seed initial trace
		const id = msg?.traceId;
		if (id){ stages.update(map => { if(!map[id]) map[id] = { id, receivedAt: Date.now() }; return { ...map }; }); }
	}
}

export const activePipelines = derived(stages, ($s) => Object.values($s).filter(v => !v.final));
export const completedPipelines = derived(stages, ($s) => Object.values($s).filter(v => v.final).sort((a,b)=> (b.completedAt||0)-(a.completedAt||0)).slice(0,20));

// Convenience start on import (optional). Comment out if you prefer manual control.
if (typeof window !== 'undefined') {
	connectRealtime();
}

export default {
	connect: connectRealtime,
	disconnect: disconnectRealtime,
	connectionStatus,
	stages,
	finalResults,
	recentEvents,
	activePipelines,
	completedPipelines
};
