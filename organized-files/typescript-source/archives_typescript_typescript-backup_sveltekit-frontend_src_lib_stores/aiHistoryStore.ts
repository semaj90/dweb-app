import { writable } from "svelte/store";

export const aiHistory = writable<
  Array<{
    prompt: string;
    response: string;
    embedding?: number[];
    timestamp: string;
    userId?: string;
  }>
>([]);
