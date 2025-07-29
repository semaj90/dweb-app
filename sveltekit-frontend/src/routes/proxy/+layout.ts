import type { LayoutLoad } from "./$types";

export const ssr = false; // Client-side only for proxy routes

export const load: LayoutLoad = async () => {
  return {};
};
