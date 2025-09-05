
import root from '../root.js';
import { set_building, set_prerendering } from '__sveltekit/environment';
import { set_assets } from '__sveltekit/paths';
import { set_manifest, set_read_implementation } from '__sveltekit/server';
import { set_private_env, set_public_env, set_safe_public_env } from '../../../node_modules/@sveltejs/kit/src/runtime/shared-server.js';

export const options = {
	app_template_contains_nonce: false,
	csp: {"mode":"auto","directives":{"upgrade-insecure-requests":false,"block-all-mixed-content":false},"reportOnly":{"upgrade-insecure-requests":false,"block-all-mixed-content":false}},
	csrf_check_origin: true,
	embedded: false,
	env_public_prefix: 'PUBLIC_',
	env_private_prefix: '',
	hash_routing: false,
	hooks: null, // added lazily, via `get_hooks`
	preload_strategy: "modulepreload",
	root,
	service_worker: true,
	templates: {
		app: ({ head, body, assets, nonce, env }) => "<!doctype html>\r\n<html lang=\"en\" %sveltekit.theme%>\r\n  <head>\r\n    <meta charset=\"utf-8\" />\r\n    <link rel=\"icon\" href=\"" + assets + "/favicon.png\" />\r\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\r\n    <title>YoRHa Legal AI - Evidence Management System</title>\r\n    <meta\r\n      name=\"description\"\r\n      content=\"Advanced legal AI evidence management with vector intelligence and semantic analysis\"\r\n    />\r\n\r\n    <!-- UnoCSS Core Styles -->\r\n    <link\r\n      rel=\"stylesheet\"\r\n      href=\"https://cdn.jsdelivr.net/npm/@unocss/reset/tailwind.min.css\"\r\n    />\r\n\r\n    <!-- Preconnect for performance -->\r\n    <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\" />\r\n    <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin />\r\n    <link\r\n      href=\"https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap\"\r\n      rel=\"stylesheet\"\r\n    />\r\n\r\n    <!-- YoRHa Interface CSS Variables & Styles -->\r\n    <style>\r\n      :root {\r\n        /* YoRHa Terminal Color Scheme */\r\n        --yorha-primary: #e0e0e0;\r\n        --yorha-primary-dark: #b0b0b0;\r\n        --yorha-secondary: #ffd700;\r\n        --yorha-secondary-dark: #cc8800;\r\n        --yorha-accent: #00ff41;\r\n        --yorha-accent-dark: #00cc33;\r\n        --yorha-danger: #ff0041;\r\n        --yorha-danger-dark: #cc0033;\r\n        \r\n        /* YoRHa Background Colors */\r\n        --yorha-bg-primary: #0a0a0a;\r\n        --yorha-bg-secondary: #1a1a1a;\r\n        --yorha-bg-tertiary: #2a2a2a;\r\n        --yorha-bg-overlay: rgba(0, 0, 0, 0.9);\r\n        \r\n        /* YoRHa Text Colors */\r\n        --yorha-text-primary: #e0e0e0;\r\n        --yorha-text-secondary: #b0b0b0;\r\n        --yorha-text-muted: #808080;\r\n        --yorha-text-accent: #ffd700;\r\n        \r\n        /* YoRHa Terminal Fonts */\r\n        --yorha-font-primary: 'JetBrains Mono', 'Courier New', monospace;\r\n        --yorha-font-secondary: 'Orbitron', 'Arial', sans-serif;\r\n        --yorha-font-ui: 'Rajdhani', 'Arial', sans-serif;\r\n        \r\n        /* Shadcn-compatible theme variables (YoRHa themed) */\r\n        --background: 0 0% 4%;\r\n        --foreground: 0 0% 88%;\r\n        --card: 0 0% 10%;\r\n        --card-foreground: 0 0% 88%;\r\n        --popover: 0 0% 6%;\r\n        --popover-foreground: 0 0% 88%;\r\n        --primary: 51 100% 50%;\r\n        --primary-foreground: 0 0% 4%;\r\n        --secondary: 0 0% 69%;\r\n        --secondary-foreground: 0 0% 4%;\r\n        --muted: 0 0% 16%;\r\n        --muted-foreground: 0 0% 50%;\r\n        --accent: 126 100% 50%;\r\n        --accent-foreground: 0 0% 4%;\r\n        --destructive: 348 100% 50%;\r\n        --destructive-foreground: 0 0% 88%;\r\n        --border: 0 0% 38%;\r\n        --input: 0 0% 16%;\r\n        --ring: 51 100% 50%;\r\n        --radius: 0.25rem;\r\n      }\r\n\r\n      * {\r\n        border-color: hsl(var(--border));\r\n      }\r\n\r\n      html {\r\n        background: linear-gradient(135deg, var(--yorha-bg-primary) 0%, var(--yorha-bg-secondary) 50%, var(--yorha-bg-tertiary) 100%);\r\n        color: var(--yorha-text-primary);\r\n        font-family: var(--yorha-font-ui);\r\n        overflow-x: hidden;\r\n      }\r\n\r\n      body {\r\n        margin: 0;\r\n        padding: 0;\r\n        min-height: 100vh;\r\n        background: transparent;\r\n        font-family: var(--yorha-font-ui);\r\n        line-height: 1.6;\r\n      }\r\n      \r\n      /* YoRHa Terminal Grid Background */\r\n      body::before {\r\n        content: '';\r\n        position: fixed;\r\n        top: 0;\r\n        left: 0;\r\n        right: 0;\r\n        bottom: 0;\r\n        background-image: \r\n          linear-gradient(rgba(255, 215, 0, 0.05) 1px, transparent 1px),\r\n          linear-gradient(90deg, rgba(255, 215, 0, 0.05) 1px, transparent 1px);\r\n        background-size: 24px 24px;\r\n        pointer-events: none;\r\n        z-index: -1;\r\n      }\r\n\r\n      /* YoRHa Terminal Scrollbar */\r\n      ::-webkit-scrollbar {\r\n        width: 12px;\r\n        height: 12px;\r\n      }\r\n      \r\n      ::-webkit-scrollbar-track {\r\n        background: var(--yorha-bg-primary);\r\n        border: 2px solid var(--yorha-bg-tertiary);\r\n      }\r\n      \r\n      ::-webkit-scrollbar-thumb {\r\n        background: linear-gradient(180deg, var(--yorha-secondary) 0%, var(--yorha-secondary-dark) 100%);\r\n        border-radius: 0;\r\n        border: 1px solid var(--yorha-text-muted);\r\n      }\r\n      \r\n      ::-webkit-scrollbar-thumb:hover {\r\n        background: linear-gradient(180deg, var(--yorha-accent) 0%, var(--yorha-accent-dark) 100%);\r\n      }\r\n\r\n      /* YoRHa Terminal Typography */\r\n      h1, h2, h3, h4, h5, h6 {\r\n        font-family: var(--yorha-font-secondary);\r\n        color: var(--yorha-text-primary);\r\n        text-transform: uppercase;\r\n        letter-spacing: 2px;\r\n        font-weight: 400;\r\n      }\r\n      \r\n      h1 {\r\n        font-size: 2.5rem;\r\n        font-weight: 700;\r\n        text-shadow: 0 0 10px var(--yorha-secondary);\r\n        border-bottom: 2px solid var(--yorha-secondary);\r\n        padding-bottom: 0.5rem;\r\n      }\r\n\r\n      /* YoRHa Terminal Input Styles */\r\n      input, textarea, select {\r\n        background: var(--yorha-bg-secondary);\r\n        border: 2px solid var(--yorha-text-muted);\r\n        color: var(--yorha-text-primary);\r\n        padding: 12px 16px;\r\n        border-radius: 0;\r\n        font-family: var(--yorha-font-primary);\r\n        transition: all 0.2s ease;\r\n      }\r\n      \r\n      input:focus, textarea:focus, select:focus {\r\n        border-color: var(--yorha-secondary);\r\n        box-shadow: 0 0 0 2px rgba(255, 215, 0, 0.3);\r\n        background: var(--yorha-bg-tertiary);\r\n        outline: none;\r\n      }\r\n\r\n      /* YoRHa Terminal Link Styles */\r\n      a {\r\n        color: var(--yorha-secondary);\r\n        text-decoration: none;\r\n        transition: all 0.2s ease;\r\n        border-bottom: 1px solid transparent;\r\n      }\r\n      \r\n      a:hover {\r\n        color: var(--yorha-accent);\r\n        text-shadow: 0 0 4px currentColor;\r\n        border-bottom: 1px solid currentColor;\r\n      }\r\n    </style>\r\n\r\n    " + head + "\r\n  </head>\r\n  <body data-sveltekit-preload-data=\"hover\" class=\"min-h-screen antialiased\">\r\n    <div style=\"display: contents\">" + body + "</div>\r\n  </body>\r\n</html>\r\n",
		error: ({ status, message }) => "<!doctype html>\n<html lang=\"en\">\n\t<head>\n\t\t<meta charset=\"utf-8\" />\n\t\t<title>" + message + "</title>\n\n\t\t<style>\n\t\t\tbody {\n\t\t\t\t--bg: white;\n\t\t\t\t--fg: #222;\n\t\t\t\t--divider: #ccc;\n\t\t\t\tbackground: var(--bg);\n\t\t\t\tcolor: var(--fg);\n\t\t\t\tfont-family:\n\t\t\t\t\tsystem-ui,\n\t\t\t\t\t-apple-system,\n\t\t\t\t\tBlinkMacSystemFont,\n\t\t\t\t\t'Segoe UI',\n\t\t\t\t\tRoboto,\n\t\t\t\t\tOxygen,\n\t\t\t\t\tUbuntu,\n\t\t\t\t\tCantarell,\n\t\t\t\t\t'Open Sans',\n\t\t\t\t\t'Helvetica Neue',\n\t\t\t\t\tsans-serif;\n\t\t\t\tdisplay: flex;\n\t\t\t\talign-items: center;\n\t\t\t\tjustify-content: center;\n\t\t\t\theight: 100vh;\n\t\t\t\tmargin: 0;\n\t\t\t}\n\n\t\t\t.error {\n\t\t\t\tdisplay: flex;\n\t\t\t\talign-items: center;\n\t\t\t\tmax-width: 32rem;\n\t\t\t\tmargin: 0 1rem;\n\t\t\t}\n\n\t\t\t.status {\n\t\t\t\tfont-weight: 200;\n\t\t\t\tfont-size: 3rem;\n\t\t\t\tline-height: 1;\n\t\t\t\tposition: relative;\n\t\t\t\ttop: -0.05rem;\n\t\t\t}\n\n\t\t\t.message {\n\t\t\t\tborder-left: 1px solid var(--divider);\n\t\t\t\tpadding: 0 0 0 1rem;\n\t\t\t\tmargin: 0 0 0 1rem;\n\t\t\t\tmin-height: 2.5rem;\n\t\t\t\tdisplay: flex;\n\t\t\t\talign-items: center;\n\t\t\t}\n\n\t\t\t.message h1 {\n\t\t\t\tfont-weight: 400;\n\t\t\t\tfont-size: 1em;\n\t\t\t\tmargin: 0;\n\t\t\t}\n\n\t\t\t@media (prefers-color-scheme: dark) {\n\t\t\t\tbody {\n\t\t\t\t\t--bg: #222;\n\t\t\t\t\t--fg: #ddd;\n\t\t\t\t\t--divider: #666;\n\t\t\t\t}\n\t\t\t}\n\t\t</style>\n\t</head>\n\t<body>\n\t\t<div class=\"error\">\n\t\t\t<span class=\"status\">" + status + "</span>\n\t\t\t<div class=\"message\">\n\t\t\t\t<h1>" + message + "</h1>\n\t\t\t</div>\n\t\t</div>\n\t</body>\n</html>\n"
	},
	version_hash: "n1xx1v"
};

export async function get_hooks() {
	let handle;
	let handleFetch;
	let handleError;
	let handleValidationError;
	let init;
	({ handle, handleFetch, handleError, handleValidationError, init } = await import("../../../src/hooks.server.ts"));

	let reroute;
	let transport;
	

	return {
		handle,
		handleFetch,
		handleError,
		handleValidationError,
		init,
		reroute,
		transport
	};
}

export { set_assets, set_building, set_manifest, set_prerendering, set_private_env, set_public_env, set_read_implementation, set_safe_public_env };
