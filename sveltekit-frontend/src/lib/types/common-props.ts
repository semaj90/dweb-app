// Runtime helpers related to CommonProps
export function mergeClass(base?: string, extra?: string): string | undefined {
  if (base && extra) return base + ' ' + extra;
  return base || extra || undefined;
}
