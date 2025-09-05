import type { RequestHandler } from '@sveltejs/kit';
import { URL } from "url";
import type { RequestHandler } from "@sveltejs/kit";
import { getAlertHistory, buildBaseline, diffBaselines } from "drizzle-orm";
let lastBaseline: any = null;

export const GET: RequestHandler = async ({ url }) => {
  const mode = url.searchParams.get('mode');
  if(mode === 'baseline'){
    const current = buildBaseline();
    const diff = lastBaseline ? diffBaselines(lastBaseline, current) : null;
    lastBaseline = current;
    return new Response(JSON.stringify({ baseline: current, diff }), { status:200, headers:{'content-type':'application/json'} });
  }
  return new Response(JSON.stringify({ alerts: getAlertHistory() }), { status:200, headers:{'content-type':'application/json'} });
};
