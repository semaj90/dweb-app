export async function handle({ event, resolve }): Promise<any> {
	return await resolve(event);
}
