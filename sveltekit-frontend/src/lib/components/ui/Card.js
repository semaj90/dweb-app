export * from './card/index.ts';
// Redirect all legacy uppercase imports to the canonical lowercase barrel.
export { Card as default, Card as Root, CardHeader as Header, CardContent as Content, CardHeader, CardContent, CardTitle, CardDescription, CardFooter } from './card/index.ts';
