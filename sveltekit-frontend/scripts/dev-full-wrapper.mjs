#!/usr/bin/env node
/**
 * dev-full-wrapper.mjs
 * Runs the full dev stack. If any child errors, writes a timestamped TODO file
 * without deleting anything, then exits with the original code.
 */

import { spawn } from "node:child_process";
import { writeFile } from "node:fs/promises";
import { mkdirSync } from "node:fs";
import { join } from "node:path";

const isWin = process.platform === "win32";

function start(cmd, args, name) {
  const child = spawn(cmd, args, { stdio: "inherit", shell: isWin });
  child._name = name;
  return child;
}

function timestamp() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const yyyy = d.getFullYear();
  const mm = pad(d.getMonth() + 1);
  const dd = pad(d.getDate());
  const hh = pad(d.getHours());
  const mi = pad(d.getMinutes());
  const ss = pad(d.getSeconds());
  return `${yyyy}${mm}${dd}-${hh}${mi}${ss}`;
}

function banner(line) {
  return `**** ${line} ****`;
}

async function writeTodo(message, meta = {}) {
  try {
    const outDir = join(process.cwd(), "todos");
    mkdirSync(outDir, { recursive: true });
    const file = join(outDir, `todo-${timestamp()}.txt`);
    const starify = (str) =>
      str
        .split("")
        .map((ch) => (ch === "\n" ? "\n" : `*${ch}*`))
        .join("");
    const header = banner(
      `TODO ${new Date().toISOString()} (proc=${meta.proc ?? "n/a"} code=${meta.code ?? "n/a"} signal=${meta.signal ?? "n/a"})`
    );
    const body = process.env.STARIFY ? starify(message) : message;
    await writeFile(file, `${header}\n${body}\n`);
    console.error(`Wrote TODO: ${file}`);
  } catch (err) {
    console.error("Failed to write TODO file:", err);
  }
}

async function main() {
  const children = [
    start("npm", ["run", "ollama:start"], "ollama:start"),
    start("npm", ["run", "dev"], "dev"),
    start("npm", ["run", "db:studio"], "db:studio"),
  ];

  let wroteTodo = false;

  const onFailure = async (msg, info = {}) => {
    if (wroteTodo) return;
    wroteTodo = true;
    await writeTodo(msg, info);
  };

  for (const child of children) {
    child.on("error", async (err) => {
      console.error(`[${child._name}] spawn error:`, err);
      await onFailure(`${child._name} spawn error: ${err?.message || err}`, {
        proc: child._name,
      });
    });
    child.on("exit", async (code, signal) => {
      if (code && code !== 0) {
        console.error(`[${child._name}] exited with code ${code}`);
        await onFailure(`${child._name} exited with code ${code}`, {
          proc: child._name,
          code,
        });
      } else if (signal) {
        console.error(`[${child._name}] exited via signal ${signal}`);
        await onFailure(`${child._name} exited via signal ${signal}`, {
          proc: child._name,
          signal,
        });
      } else {
        console.log(`[${child._name}] exited normally (code 0)`);
      }
      // Do not kill other processes; keep running as requested
    });
  }

  // Keep process alive while children are running
  const handleSignal = (sig) => {
    console.log(`dev:full wrapper received ${sig}, forwarding to children...`);
    for (const c of children) {
      try {
        c.kill(sig);
      } catch (e) {
        /* ignore */
      }
    }
    process.exit(0);
  };
  process.on("SIGINT", handleSignal);
  process.on("SIGTERM", handleSignal);
}

main().catch(async (err) => {
  await writeTodo(`dev:full wrapper threw: ${err?.stack || err}`, {
    proc: "dev-full-wrapper",
  });
  // Keep process alive briefly so the message is visible
  setTimeout(() => process.exit(1), 100);
});
