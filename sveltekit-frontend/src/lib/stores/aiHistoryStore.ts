import { writable } from "svelte/store";
// @ts-nocheck
export const aiHistory = writable<
  Array<{
    prompt: string;
    response: string;
    embedding?: number[];
    timestamp: string;
    userId?: string;
  }>
>([]);
