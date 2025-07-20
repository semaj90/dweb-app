declare global {
	namespace App {
		interface Error {
			message: string;
			code?: string;
		}
		interface Locals {
			user?: {
				id: string;
				email: string;
			};
		}
		interface PageData {}
		interface Platform {}
	}
}

export {};
