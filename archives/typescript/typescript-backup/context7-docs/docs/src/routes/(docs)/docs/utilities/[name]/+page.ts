import { utilityDocs } from "$content/index.js";
import { getUtilityDoc } from "$lib/utils/docs.js";
import type { EntryGenerator } from './$types';

export const prerender = true;

export const entries: EntryGenerator = async () => {
	return utilityDocs.map((doc) => ({
		name: doc.slug,
	}));
};

export async function load(event): Promise<any> {
	return await getUtilityDoc(`utilities/${event.params.name}`);
}
