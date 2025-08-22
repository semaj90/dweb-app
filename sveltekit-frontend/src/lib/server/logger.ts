
/**
 * Server-side logger utility
 */

export class Logger {
  private static instance: Logger;

  private constructor() {}

  static getInstance(): Logger {
    if (!Logger.instance) {
      Logger.instance = new Logger();
    }
    return Logger.instance;
  }

  info(message: string, meta?: unknown) {
    console.log(`[INFO] ${new Date().toISOString()} - ${message}`, meta || "");
  }

  error(message: string, error?: unknown) {
    console.error(
      `[ERROR] ${new Date().toISOString()} - ${message}`,
      error || ""
    );
  }

  warn(message: string, meta?: unknown) {
    console.warn(`[WARN] ${new Date().toISOString()} - ${message}`, meta || "");
  }

  debug(message: string, meta?: unknown) {
    if (import.meta.env.NODE_ENV === "development") {
      console.debug(
        `[DEBUG] ${new Date().toISOString()} - ${message}`,
        meta || ""
      );
    }
  }
}

export const logger = Logger.getInstance();
export default logger;
