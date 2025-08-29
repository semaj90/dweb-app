import { redirect } from "@sveltejs/kit";

export async function load(): Promise<any> {
	redirect(303, "/docs/introduction");
}
