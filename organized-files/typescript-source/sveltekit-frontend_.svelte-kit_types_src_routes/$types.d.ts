import type * as Kit from '@sveltejs/kit';

type Expand<T> = T extends infer O ? { [K in keyof O]: O[K] } : never;
// @ts-ignore
type MatcherParam<M> = M extends (param : string) => param is infer U ? U extends string ? U : string : string;
type RouteParams = {  };
type RouteId = '/';
type MaybeWithVoid<T> = {} extends T ? T | void : T;
export type RequiredKeys<T> = { [K in keyof T]-?: {} extends { [P in K]: T[K] } ? never : K; }[keyof T];
type OutputDataShape<T> = MaybeWithVoid<Omit<App.PageData, RequiredKeys<T>> & Partial<Pick<App.PageData, keyof T & keyof App.PageData>> & Record<string, any>>
type EnsureDefined<T> = T extends null | undefined ? {} : T;
type OptionalUnion<U extends Record<string, any>, A extends keyof U = U extends U ? keyof U : never> = U extends unknown ? { [P in Exclude<A, keyof U>]?: never } & U : never;
export type Snapshot<T = any> = Kit.Snapshot<T>;
type PageServerParentData = EnsureDefined<LayoutServerData>;
type PageParentData = EnsureDefined<LayoutData>;
type LayoutRouteId = RouteId | "/" | "/admin/cluster" | "/admin/gpu-demo" | "/admin/users" | "/ai" | "/ai/enhanced-mcp" | "/ai/orchestrator" | "/ai-assistant" | "/ai-demo" | "/ai-summary" | "/ai-test" | "/aiassistant" | "/all-routes" | "/bits-uno-demo" | "/cases" | "/cases/create" | "/cases/new" | "/cases/[caseId]/rag" | "/cases/[id]" | "/cases/[id]/canvas" | "/cases/[id]/enhanced" | "/chat" | "/compiler-ai-demo" | "/context7-demo" | "/context7-demo.disabled" | "/copilot/autonomous" | "/crud-dashboard" | "/dashboard" | "/demo" | "/demo/ai-assistant" | "/demo/ai-complete-test" | "/demo/ai-dashboard" | "/demo/ai-integration" | "/demo/ai-pipeline" | "/demo/ai-summary" | "/demo/ai-test" | "/demo/bvector-test" | "/demo/component-gallery" | "/demo/cyber-elephant" | "/demo/document-ai" | "/demo/full-stack-integration" | "/demo/gpu-legal-ai" | "/demo/gpu-legal-ai/lawpdfs" | "/demo/hybrid-cache-architecture" | "/demo/inline-suggestions" | "/demo/legal-ai-complete" | "/demo/live-agents" | "/demo/nes-yorha-hybrid" | "/demo/neural-sprite-engine" | "/demo/notes" | "/demo/phase5" | "/demo/professional-editor" | "/demo/quic-tensor" | "/demo/rag-integration" | "/demo/simple-test" | "/demo/system-summary" | "/demo/unified-architecture" | "/demo/unified-inference" | "/demo/unocss-svelte5" | "/demo/vector-intelligence" | "/demo/vector-pipeline" | "/demo/vector-search" | "/demo/webgpu-ranking-cache" | "/demo/yorha-tables" | "/detective" | "/detective/canvas" | "/dev/ai-setup" | "/dev/cache-demo" | "/dev/context7-test" | "/dev/copilot-optimizer" | "/dev/enhanced-processor" | "/dev/mcp-tools" | "/dev/self-prompting-demo" | "/dev/suggestions" | "/dev/vector-search-demo" | "/dev/vite-error-demo" | "/document-editor-demo" | "/editor" | "/enhanced" | "/enhanced-ai-demo" | "/evidence" | "/evidence/analyze" | "/evidence/files" | "/evidence/hash" | "/evidence/manage" | "/evidence/realtime" | "/evidence/upload" | "/evidence-editor" | "/evidenceboard" | "/export" | "/frameworks-demo" | "/gaming-demo" | "/golden-ratio-demo" | "/help" | "/import" | "/interactive-canvas" | "/law" | "/laws" | "/legal/documents" | "/legal-ai-suite" | "/local-ai-demo" | "/login" | "/logout" | "/memory-dashboard" | "/modern-demo" | "/nier-showcase" | "/optimization-dashboard" | "/original-home" | "/perf" | "/persons" | "/phase13-demo" | "/profile" | "/proxy" | "/rag-demo" | "/register" | "/report-builder" | "/reports" | "/saved-citations" | "/search" | "/security" | "/semantic-search-demo" | "/settings" | "/showcase" | "/studio" | "/test" | "/test-ai-ask" | "/test-components" | "/test-gemma3" | "/test-integration" | "/test-simple" | "/test-upload" | "/ui-demo" | "/upload" | "/wasm-gpu-demo" | "/windows-gguf-demo" | null
type LayoutParams = RouteParams & { caseId?: string; id?: string }
type LayoutServerParentData = EnsureDefined<{}>;
type LayoutParentData = EnsureDefined<{}>;

export type PageServerLoad<OutputData extends OutputDataShape<PageServerParentData> = OutputDataShape<PageServerParentData>> = Kit.ServerLoad<RouteParams, PageServerParentData, OutputData, RouteId>;
export type PageServerLoadEvent = Parameters<PageServerLoad>[0];
export type ActionData = unknown;
export type PageServerData = null;
export type PageData = Expand<PageParentData>;
export type Action<OutputData extends Record<string, any> | void = Record<string, any> | void> = Kit.Action<RouteParams, OutputData, RouteId>
export type Actions<OutputData extends Record<string, any> | void = Record<string, any> | void> = Kit.Actions<RouteParams, OutputData, RouteId>
export type PageProps = { params: RouteParams; data: PageData; form: ActionData }
export type LayoutServerLoad<OutputData extends Partial<App.PageData> & Record<string, any> | void = Partial<App.PageData> & Record<string, any> | void> = Kit.ServerLoad<LayoutParams, LayoutServerParentData, OutputData, LayoutRouteId>;
export type LayoutServerLoadEvent = Parameters<LayoutServerLoad>[0];
export type LayoutServerData = Expand<OptionalUnion<EnsureDefined<Kit.LoadProperties<Awaited<ReturnType<typeof import('../../../../src/routes/+layout.server.js').load>>>>>>;
export type LayoutLoad<OutputData extends OutputDataShape<LayoutParentData> = OutputDataShape<LayoutParentData>> = Kit.Load<LayoutParams, LayoutServerData, LayoutParentData, OutputData, LayoutRouteId>;
export type LayoutLoadEvent = Parameters<LayoutLoad>[0];
export type LayoutData = Expand<Omit<LayoutParentData, keyof Kit.LoadProperties<Awaited<ReturnType<typeof import('./proxy+layout.js').load>>>> & OptionalUnion<EnsureDefined<Kit.LoadProperties<Awaited<ReturnType<typeof import('./proxy+layout.js').load>>>>>>;
export type LayoutProps = { params: LayoutParams; data: LayoutData; children: import("svelte").Snippet }
export type RequestEvent = Kit.RequestEvent<RouteParams, RouteId>;