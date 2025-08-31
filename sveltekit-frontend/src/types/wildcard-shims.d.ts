// Wildcard shims for common pattern imports to reduce 'has no exported member' noise.
declare module '*-service' {
  const whatever: any;
  export default whatever;
  export const named: any;
}

declare module '$lib/*' {
  const whatever: any;
  export default whatever;
}
