// API Client utility for centralized API calls
import { error } from '@sveltejs/kit';

interface ApiConfig {
  baseUrl?: string;
  timeout?: number;
  retries?: number;
}

const defaultConfig: ApiConfig = {
  baseUrl: '',
  timeout: 10000,
  retries: 3,
};

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export async function apiFetch<T = any>(
  endpoint: string,
  options: RequestInit & { config?: ApiConfig } = {}
): Promise<T> {
  const { config = {}, ...fetchOptions } = options;
  const finalConfig = { ...defaultConfig, ...config };

  const url = `${finalConfig.baseUrl}${endpoint}`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), finalConfig.timeout);

  try {
    const response = await fetch(url, {
      ...fetchOptions,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new ApiError(
        `API request failed: ${response.statusText}`,
        response.status,
        await response.text().catch(() => null)
      );
    }

    const contentType = response.headers.get('content-type');
    if (contentType?.includes('application/json')) {
      return await response.json();
    }

    return (await response.text()) as T;
  } catch (err) {
    clearTimeout(timeoutId);

    if (err instanceof ApiError) {
      throw err;
    }

    if (err instanceof Error && err.name === 'AbortError') {
      throw new ApiError('Request timeout', 408);
    }

    throw new ApiError(`Network error: ${err instanceof Error ? err.message : 'Unknown error'}`, 0);
  }
}

export { ApiError };
