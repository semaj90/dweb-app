import { ContextService } from "$lib/services/context-service";
import { json } from "@sveltejs/kit";

export async function GET() {
  const context = await ContextService.getCurrentContext();
  return json({ context });
}
