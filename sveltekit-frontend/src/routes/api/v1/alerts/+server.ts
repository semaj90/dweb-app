import type { RequestHandler } from './$types.js';
import { URL } from "url";

// Local placeholder implementations (replace with real module imports when available)
interface Alert {
  message: string;
  timestamp: string;
}
interface Baseline {
  generatedAt: string;
}

const alertHistory: Alert[] = [];

function getAlertHistory(): Alert[] {
  return alertHistory;
}

function buildBaseline(): Baseline {
  return { generatedAt: new Date().toISOString() };
}

function diffBaselines(prev: Baseline, curr: Baseline) {
  return {
    changed: prev.generatedAt !== curr.generatedAt,
    previous: prev,
    current: curr
  };
}

let lastBaseline: Baseline | null = null;

export const GET: RequestHandler = async ({ url }) => {
  const mode = url.searchParams.get('mode');
  if (mode === 'baseline') {
    const current = buildBaseline();
    const diff = lastBaseline ? diffBaselines(lastBaseline, current) : null;
    lastBaseline = current;
    return new Response(JSON.stringify({ baseline: current, diff }), {
      status: 200,
      headers: { 'content-type': 'application/json' }
    });
  }
  return new Response(JSON.stringify({ alerts: getAlertHistory() }), {
    status: 200,
    headers: { 'content-type': 'application/json' }
  });
};
