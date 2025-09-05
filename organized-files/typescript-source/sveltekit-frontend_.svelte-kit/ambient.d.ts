
// this file is generated — do not edit it


/// <reference types="@sveltejs/kit" />

/**
 * Environment variables [loaded by Vite](https://vitejs.dev/guide/env-and-mode.html#env-files) from `.env` files and `process.env`. Like [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), this module cannot be imported into client-side code. This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured).
 * 
 * _Unlike_ [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), the values exported from this module are statically injected into your bundle at build time, enabling optimisations like dead code elimination.
 * 
 * ```ts
 * import { API_KEY } from '$env/static/private';
 * ```
 * 
 * Note that all environment variables referenced in your code should be declared (for example in an `.env` file), even if they don't have a value until the app is deployed:
 * 
 * ```
 * MY_FEATURE_FLAG=""
 * ```
 * 
 * You can override `.env` values from the command line like so:
 * 
 * ```sh
 * MY_FEATURE_FLAG="enabled" npm run dev
 * ```
 */
declare module '$env/static/private' {
	export const DATABASE_URL: string;
	export const VITE_DATABASE_URL: string;
	export const DEV_DATABASE_URL: string;
	export const POSTGRES_HOST: string;
	export const POSTGRES_PORT: string;
	export const POSTGRES_DB: string;
	export const POSTGRES_USER: string;
	export const POSTGRES_PASSWORD: string;
	export const OLLAMA_URL: string;
	export const OLLAMA_API_URL: string;
	export const OLLAMA_MODEL: string;
	export const EMBEDDING_MODEL: string;
	export const REDIS_URL: string;
	export const QDRANT_URL: string;
	export const MINIO_URL: string;
	export const MINIO_HOST: string;
	export const MINIO_PORT: string;
	export const MINIO_ACCESS_KEY: string;
	export const MINIO_SECRET_KEY: string;
	export const MINIO_USE_SSL: string;
	export const MINIO_REGION: string;
	export const JWT_SECRET: string;
	export const SESSION_SECRET: string;
	export const ENCRYPTION_KEY: string;
	export const RATE_LIMIT_MAX: string;
	export const RATE_LIMIT_WINDOW: string;
	export const VITE_GPU_ENABLED: string;
	export const VITE_DEMO_MODE: string;
	export const NODE_ENV: string;
	export const MCP_SERVER_HOST: string;
	export const MCP_WORKER_THREADS: string;
	export const LOG_LEVEL: string;
	export const DEBUG_MODE: string;
	export const CORS_ORIGIN: string;
	export const ALLOWED_ORIGINS: string;
	export const TORCH_HOME: string;
	export const HF_HOME: string;
	export const TRANSFORMERS_CACHE: string;
	export const VITE_GO_GPU_URL: string;
	export const GO_GPU_SERVICE_URL: string;
	export const DB_HOST: string;
	export const DB_PORT: string;
	export const DB_NAME: string;
	export const DB_USER: string;
	export const DB_PASSWORD: string;
	export const GO_MICROSERVICE_URL: string;
	export const GO_SERVER_URL: string;
	export const VITE_UPLOAD_MAX_SIZE: string;
	export const PGPASSWORD: string;
	export const SKIP_RAG_INITIALIZATION: string;
	export const SKIP_QDRANT_HEALTH_CHECK: string;
	export const USE_POSTGRESQL_ONLY: string;
	export const VITE_MAX_MEMORY: string;
	export const NODE_OPTIONS: string;
	export const HOST: string;
	export const PORT: string;
	export const VITE_DEV_PORT: string;
	export const HMR_PORT: string;
	export const VITE_WORKERS: string;
	export const VITE_MEMORY_LIMIT: string;
	export const CONCURRENT_BUILDS: string;
	export const ENABLE_HOT_RELOAD: string;
	export const ENABLE_SOURCE_MAPS: string;
	export const MAX_WORKERS: string;
	export const WORKER_MEMORY_LIMIT: string;
	export const CLUSTER_WORKERS: string;
	export const ENABLE_CLUSTERING: string;
	export const DEBUG: string;
	export const ENABLE_DEVTOOLS: string;
	export const ENABLE_INSPECTOR: string;
	export const INSPECTOR_PORT: string;
	export const OPENAI_API_KEY: string;
	export const ANTHROPIC_API_KEY: string;
	export const MINIO_ENDPOINT: string;
	export const MINIO_BUCKET: string;
	export const ENABLE_PERFORMANCE_MONITORING: string;
	export const METRICS_INTERVAL: string;
	export const ENABLE_MEMORY_MONITORING: string;
	export const VITE_BUILD_PARALLEL: string;
	export const VITE_BUILD_TARGET: string;
	export const ENABLE_CSS_OPTIMIZATION: string;
	export const ENABLE_BUNDLE_ANALYZER: string;
	export const ENABLE_WEBGPU: string;
	export const ENABLE_GPU_ACCELERATION: string;
	export const GPU_MEMORY_LIMIT: string;
	export const LOKI_ADAPTER: string;
	export const LOKI_AUTO_SAVE: string;
	export const LOKI_AUTO_SAVE_INTERVAL: string;
	export const POSTCSS_PARALLEL: string;
	export const POSTCSS_WORKERS: string;
	export const ENABLE_CSS_MODULES: string;
	export const ENABLE_HTTPS: string;
	export const CSRF_PROTECTION: string;
	export const ENABLE_EXPERIMENTAL_FEATURES: string;
	export const ENABLE_NEURAL_SPRITE_ENGINE: string;
	export const ENABLE_CYBER_ELEPHANT_DEMO: string;
	export const ACLOCAL_PATH: string;
	export const ALLUSERSPROFILE: string;
	export const APPDATA: string;
	export const BULLMQ_REDIS_URL: string;
	export const CC: string;
	export const CGO_ENABLED: string;
	export const ChocolateyInstall: string;
	export const ChocolateyLastPathUpdate: string;
	export const CHROME_CRASHPAD_PIPE_NAME: string;
	export const CLAUDECODE: string;
	export const CLAUDE_CODE_ENTRYPOINT: string;
	export const CLAUDE_CODE_OAUTH_TOKEN: string;
	export const CLAUDE_CODE_SSE_PORT: string;
	export const COLOR: string;
	export const COLORTERM: string;
	export const COMMONPROGRAMFILES: string;
	export const CommonProgramW6432: string;
	export const COMPUTERNAME: string;
	export const COMSPEC: string;
	export const CONFIG_SITE: string;
	export const COREPACK_ENABLE_AUTO_PIN: string;
	export const CUDA_PATH: string;
	export const CUDA_PATH_V12_8: string;
	export const CUDA_PATH_V13_0: string;
	export const CUDA_VISIBLE_DEVICES: string;
	export const CXX: string;
	export const DISPLAY: string;
	export const DriverData: string;
	export const EDITOR: string;
	export const ENABLE_IDE_INTEGRATION: string;
	export const ERLANG_HOME: string;
	export const EXEPATH: string;
	export const GIT_ASKPASS: string;
	export const GIT_EDITOR: string;
	export const GOPATH: string;
	export const HOME: string;
	export const HOMEDRIVE: string;
	export const HOMEPATH: string;
	export const HOSTNAME: string;
	export const INFOPATH: string;
	export const INIT_CWD: string;
	export const LANG: string;
	export const LANGCHAIN_CACHE_TTL: string;
	export const LANGCHAIN_CACHE_TYPE: string;
	export const LANGCHAIN_CONCURRENT_REQUESTS: string;
	export const LOCALAPPDATA: string;
	export const LOGONSERVER: string;
	export const MANPATH: string;
	export const MINGW_CHOST: string;
	export const MINGW_PACKAGE_PREFIX: string;
	export const MINGW_PREFIX: string;
	export const MSMPI_BENCHMARKS: string;
	export const MSMPI_BIN: string;
	export const MSYSTEM: string;
	export const MSYSTEM_CARCH: string;
	export const MSYSTEM_CHOST: string;
	export const MSYSTEM_PREFIX: string;
	export const NODE: string;
	export const npm_command: string;
	export const npm_config_cache: string;
	export const npm_config_globalconfig: string;
	export const npm_config_global_prefix: string;
	export const npm_config_init_module: string;
	export const npm_config_legacy_peer_deps: string;
	export const npm_config_local_prefix: string;
	export const npm_config_node_gyp: string;
	export const npm_config_noproxy: string;
	export const npm_config_npm_version: string;
	export const npm_config_prefix: string;
	export const npm_config_userconfig: string;
	export const npm_config_user_agent: string;
	export const npm_execpath: string;
	export const npm_lifecycle_event: string;
	export const npm_lifecycle_script: string;
	export const npm_node_execpath: string;
	export const npm_package_json: string;
	export const npm_package_name: string;
	export const npm_package_version: string;
	export const NUMBER_OF_PROCESSORS: string;
	export const OLDPWD: string;
	export const OneDrive: string;
	export const OneDriveConsumer: string;
	export const ORIGINAL_PATH: string;
	export const ORIGINAL_TEMP: string;
	export const ORIGINAL_TMP: string;
	export const ORIGINAL_XDG_CURRENT_DESKTOP: string;
	export const OS: string;
	export const OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE: string;
	export const PATH: string;
	export const PATHEXT: string;
	export const PKG_CONFIG_PATH: string;
	export const PKG_CONFIG_SYSTEM_INCLUDE_PATH: string;
	export const PKG_CONFIG_SYSTEM_LIBRARY_PATH: string;
	export const PLINK_PROTOCOL: string;
	export const POWERSHELL_DISTRIBUTION_CHANNEL: string;
	export const POWERSHELL_TELEMETRY_OPTOUT: string;
	export const PROCESSOR_ARCHITECTURE: string;
	export const PROCESSOR_IDENTIFIER: string;
	export const PROCESSOR_LEVEL: string;
	export const PROCESSOR_REVISION: string;
	export const ProgramData: string;
	export const PROGRAMFILES: string;
	export const ProgramW6432: string;
	export const PROMPT: string;
	export const PSModulePath: string;
	export const PUBLIC: string;
	export const PWD: string;
	export const RABBITMQ_URL: string;
	export const SENTENCE_SPLITTER_BATCH_SIZE: string;
	export const SENTENCE_SPLITTER_WORKERS: string;
	export const SESSIONNAME: string;
	export const SHELL: string;
	export const SHLVL: string;
	export const SSH_ASKPASS: string;
	export const SYSTEMDRIVE: string;
	export const SYSTEMROOT: string;
	export const TEMP: string;
	export const TENSOR_CACHE_SIZE: string;
	export const TERM: string;
	export const TERM_PROGRAM: string;
	export const TERM_PROGRAM_VERSION: string;
	export const TF_FORCE_GPU_ALLOW_GROWTH: string;
	export const TMP: string;
	export const TMPDIR: string;
	export const USERDOMAIN: string;
	export const USERDOMAIN_ROAMINGPROFILE: string;
	export const USERNAME: string;
	export const USERPROFILE: string;
	export const VITE_USER_NODE_ENV: string;
	export const VSCODE_GIT_ASKPASS_EXTRA_ARGS: string;
	export const VSCODE_GIT_ASKPASS_MAIN: string;
	export const VSCODE_GIT_ASKPASS_NODE: string;
	export const VSCODE_GIT_IPC_HANDLE: string;
	export const VSCODE_INJECTION: string;
	export const VSCODE_INSPECTOR_OPTIONS: string;
	export const WINDIR: string;
}

/**
 * Similar to [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private), except that it only includes environment variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`), and can therefore safely be exposed to client-side code.
 * 
 * Values are replaced statically at build time.
 * 
 * ```ts
 * import { PUBLIC_BASE_URL } from '$env/static/public';
 * ```
 */
declare module '$env/static/public' {
	export const PUBLIC_API_URL: string;
	export const PUBLIC_RAG_SERVICE_URL: string;
	export const PUBLIC_UPLOAD_SERVICE_URL: string;
	export const PUBLIC_VECTOR_SERVICE_URL: string;
	export const PUBLIC_CUDA_SERVICE_URL: string;
	export const PUBLIC_LOAD_BALANCER_URL: string;
	export const PUBLIC_ENHANCED_API_URL: string;
	export const PUBLIC_MAX_FILE_SIZE: string;
	export const PUBLIC_MAX_FILES_PER_UPLOAD: string;
	export const PUBLIC_REQUEST_TIMEOUT: string;
}

/**
 * This module provides access to runtime environment variables, as defined by the platform you're running on. For example if you're using [`adapter-node`](https://github.com/sveltejs/kit/tree/main/packages/adapter-node) (or running [`vite preview`](https://svelte.dev/docs/kit/cli)), this is equivalent to `process.env`. This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured).
 * 
 * This module cannot be imported into client-side code.
 * 
 * Dynamic environment variables cannot be used during prerendering.
 * 
 * ```ts
 * import { env } from '$env/dynamic/private';
 * console.log(env.DEPLOYMENT_SPECIFIC_VARIABLE);
 * ```
 * 
 * > In `dev`, `$env/dynamic` always includes environment variables from `.env`. In `prod`, this behavior will depend on your adapter.
 */
declare module '$env/dynamic/private' {
	export const env: {
		DATABASE_URL: string;
		VITE_DATABASE_URL: string;
		DEV_DATABASE_URL: string;
		POSTGRES_HOST: string;
		POSTGRES_PORT: string;
		POSTGRES_DB: string;
		POSTGRES_USER: string;
		POSTGRES_PASSWORD: string;
		OLLAMA_URL: string;
		OLLAMA_API_URL: string;
		OLLAMA_MODEL: string;
		EMBEDDING_MODEL: string;
		REDIS_URL: string;
		QDRANT_URL: string;
		MINIO_URL: string;
		MINIO_HOST: string;
		MINIO_PORT: string;
		MINIO_ACCESS_KEY: string;
		MINIO_SECRET_KEY: string;
		MINIO_USE_SSL: string;
		MINIO_REGION: string;
		JWT_SECRET: string;
		SESSION_SECRET: string;
		ENCRYPTION_KEY: string;
		RATE_LIMIT_MAX: string;
		RATE_LIMIT_WINDOW: string;
		VITE_GPU_ENABLED: string;
		VITE_DEMO_MODE: string;
		NODE_ENV: string;
		MCP_SERVER_HOST: string;
		MCP_WORKER_THREADS: string;
		LOG_LEVEL: string;
		DEBUG_MODE: string;
		CORS_ORIGIN: string;
		ALLOWED_ORIGINS: string;
		TORCH_HOME: string;
		HF_HOME: string;
		TRANSFORMERS_CACHE: string;
		VITE_GO_GPU_URL: string;
		GO_GPU_SERVICE_URL: string;
		DB_HOST: string;
		DB_PORT: string;
		DB_NAME: string;
		DB_USER: string;
		DB_PASSWORD: string;
		GO_MICROSERVICE_URL: string;
		GO_SERVER_URL: string;
		VITE_UPLOAD_MAX_SIZE: string;
		PGPASSWORD: string;
		SKIP_RAG_INITIALIZATION: string;
		SKIP_QDRANT_HEALTH_CHECK: string;
		USE_POSTGRESQL_ONLY: string;
		VITE_MAX_MEMORY: string;
		NODE_OPTIONS: string;
		HOST: string;
		PORT: string;
		VITE_DEV_PORT: string;
		HMR_PORT: string;
		VITE_WORKERS: string;
		VITE_MEMORY_LIMIT: string;
		CONCURRENT_BUILDS: string;
		ENABLE_HOT_RELOAD: string;
		ENABLE_SOURCE_MAPS: string;
		MAX_WORKERS: string;
		WORKER_MEMORY_LIMIT: string;
		CLUSTER_WORKERS: string;
		ENABLE_CLUSTERING: string;
		DEBUG: string;
		ENABLE_DEVTOOLS: string;
		ENABLE_INSPECTOR: string;
		INSPECTOR_PORT: string;
		OPENAI_API_KEY: string;
		ANTHROPIC_API_KEY: string;
		MINIO_ENDPOINT: string;
		MINIO_BUCKET: string;
		ENABLE_PERFORMANCE_MONITORING: string;
		METRICS_INTERVAL: string;
		ENABLE_MEMORY_MONITORING: string;
		VITE_BUILD_PARALLEL: string;
		VITE_BUILD_TARGET: string;
		ENABLE_CSS_OPTIMIZATION: string;
		ENABLE_BUNDLE_ANALYZER: string;
		ENABLE_WEBGPU: string;
		ENABLE_GPU_ACCELERATION: string;
		GPU_MEMORY_LIMIT: string;
		LOKI_ADAPTER: string;
		LOKI_AUTO_SAVE: string;
		LOKI_AUTO_SAVE_INTERVAL: string;
		POSTCSS_PARALLEL: string;
		POSTCSS_WORKERS: string;
		ENABLE_CSS_MODULES: string;
		ENABLE_HTTPS: string;
		CSRF_PROTECTION: string;
		ENABLE_EXPERIMENTAL_FEATURES: string;
		ENABLE_NEURAL_SPRITE_ENGINE: string;
		ENABLE_CYBER_ELEPHANT_DEMO: string;
		ACLOCAL_PATH: string;
		ALLUSERSPROFILE: string;
		APPDATA: string;
		BULLMQ_REDIS_URL: string;
		CC: string;
		CGO_ENABLED: string;
		ChocolateyInstall: string;
		ChocolateyLastPathUpdate: string;
		CHROME_CRASHPAD_PIPE_NAME: string;
		CLAUDECODE: string;
		CLAUDE_CODE_ENTRYPOINT: string;
		CLAUDE_CODE_OAUTH_TOKEN: string;
		CLAUDE_CODE_SSE_PORT: string;
		COLOR: string;
		COLORTERM: string;
		COMMONPROGRAMFILES: string;
		CommonProgramW6432: string;
		COMPUTERNAME: string;
		COMSPEC: string;
		CONFIG_SITE: string;
		COREPACK_ENABLE_AUTO_PIN: string;
		CUDA_PATH: string;
		CUDA_PATH_V12_8: string;
		CUDA_PATH_V13_0: string;
		CUDA_VISIBLE_DEVICES: string;
		CXX: string;
		DISPLAY: string;
		DriverData: string;
		EDITOR: string;
		ENABLE_IDE_INTEGRATION: string;
		ERLANG_HOME: string;
		EXEPATH: string;
		GIT_ASKPASS: string;
		GIT_EDITOR: string;
		GOPATH: string;
		HOME: string;
		HOMEDRIVE: string;
		HOMEPATH: string;
		HOSTNAME: string;
		INFOPATH: string;
		INIT_CWD: string;
		LANG: string;
		LANGCHAIN_CACHE_TTL: string;
		LANGCHAIN_CACHE_TYPE: string;
		LANGCHAIN_CONCURRENT_REQUESTS: string;
		LOCALAPPDATA: string;
		LOGONSERVER: string;
		MANPATH: string;
		MINGW_CHOST: string;
		MINGW_PACKAGE_PREFIX: string;
		MINGW_PREFIX: string;
		MSMPI_BENCHMARKS: string;
		MSMPI_BIN: string;
		MSYSTEM: string;
		MSYSTEM_CARCH: string;
		MSYSTEM_CHOST: string;
		MSYSTEM_PREFIX: string;
		NODE: string;
		npm_command: string;
		npm_config_cache: string;
		npm_config_globalconfig: string;
		npm_config_global_prefix: string;
		npm_config_init_module: string;
		npm_config_legacy_peer_deps: string;
		npm_config_local_prefix: string;
		npm_config_node_gyp: string;
		npm_config_noproxy: string;
		npm_config_npm_version: string;
		npm_config_prefix: string;
		npm_config_userconfig: string;
		npm_config_user_agent: string;
		npm_execpath: string;
		npm_lifecycle_event: string;
		npm_lifecycle_script: string;
		npm_node_execpath: string;
		npm_package_json: string;
		npm_package_name: string;
		npm_package_version: string;
		NUMBER_OF_PROCESSORS: string;
		OLDPWD: string;
		OneDrive: string;
		OneDriveConsumer: string;
		ORIGINAL_PATH: string;
		ORIGINAL_TEMP: string;
		ORIGINAL_TMP: string;
		ORIGINAL_XDG_CURRENT_DESKTOP: string;
		OS: string;
		OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE: string;
		PATH: string;
		PATHEXT: string;
		PKG_CONFIG_PATH: string;
		PKG_CONFIG_SYSTEM_INCLUDE_PATH: string;
		PKG_CONFIG_SYSTEM_LIBRARY_PATH: string;
		PLINK_PROTOCOL: string;
		POWERSHELL_DISTRIBUTION_CHANNEL: string;
		POWERSHELL_TELEMETRY_OPTOUT: string;
		PROCESSOR_ARCHITECTURE: string;
		PROCESSOR_IDENTIFIER: string;
		PROCESSOR_LEVEL: string;
		PROCESSOR_REVISION: string;
		ProgramData: string;
		PROGRAMFILES: string;
		ProgramW6432: string;
		PROMPT: string;
		PSModulePath: string;
		PUBLIC: string;
		PWD: string;
		RABBITMQ_URL: string;
		SENTENCE_SPLITTER_BATCH_SIZE: string;
		SENTENCE_SPLITTER_WORKERS: string;
		SESSIONNAME: string;
		SHELL: string;
		SHLVL: string;
		SSH_ASKPASS: string;
		SYSTEMDRIVE: string;
		SYSTEMROOT: string;
		TEMP: string;
		TENSOR_CACHE_SIZE: string;
		TERM: string;
		TERM_PROGRAM: string;
		TERM_PROGRAM_VERSION: string;
		TF_FORCE_GPU_ALLOW_GROWTH: string;
		TMP: string;
		TMPDIR: string;
		USERDOMAIN: string;
		USERDOMAIN_ROAMINGPROFILE: string;
		USERNAME: string;
		USERPROFILE: string;
		VITE_USER_NODE_ENV: string;
		VSCODE_GIT_ASKPASS_EXTRA_ARGS: string;
		VSCODE_GIT_ASKPASS_MAIN: string;
		VSCODE_GIT_ASKPASS_NODE: string;
		VSCODE_GIT_IPC_HANDLE: string;
		VSCODE_INJECTION: string;
		VSCODE_INSPECTOR_OPTIONS: string;
		WINDIR: string;
		[key: `PUBLIC_${string}`]: undefined;
		[key: `${string}`]: string | undefined;
	}
}

/**
 * Similar to [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), but only includes variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`), and can therefore safely be exposed to client-side code.
 * 
 * Note that public dynamic environment variables must all be sent from the server to the client, causing larger network requests — when possible, use `$env/static/public` instead.
 * 
 * Dynamic environment variables cannot be used during prerendering.
 * 
 * ```ts
 * import { env } from '$env/dynamic/public';
 * console.log(env.PUBLIC_DEPLOYMENT_SPECIFIC_VARIABLE);
 * ```
 */
declare module '$env/dynamic/public' {
	export const env: {
		PUBLIC_API_URL: string;
		PUBLIC_RAG_SERVICE_URL: string;
		PUBLIC_UPLOAD_SERVICE_URL: string;
		PUBLIC_VECTOR_SERVICE_URL: string;
		PUBLIC_CUDA_SERVICE_URL: string;
		PUBLIC_LOAD_BALANCER_URL: string;
		PUBLIC_ENHANCED_API_URL: string;
		PUBLIC_MAX_FILE_SIZE: string;
		PUBLIC_MAX_FILES_PER_UPLOAD: string;
		PUBLIC_REQUEST_TIMEOUT: string;
		[key: `PUBLIC_${string}`]: string | undefined;
	}
}
