#!/usr/bin/env node
/**
 * review-todos.mjs
 * Lists and optionally opens TODO files created by dev:full errors
 */

import { readdir, readFile, stat } from "node:fs/promises";
import { join } from "node:path";
import { spawn } from "node:child_process";

const isWin = process.platform === "win32";

async function listTodos() {
  const todosDir = join(process.cwd(), "todos");

  try {
    const files = await readdir(todosDir);
    const todoFiles = files
      .filter((f) => f.startsWith("todo-") && f.endsWith(".txt"))
      .sort()
      .reverse(); // newest first

    if (todoFiles.length === 0) {
      console.log("No TODO files found in todos/");
      return [];
    }

    const todos = [];
    for (const file of todoFiles) {
      const path = join(todosDir, file);
      const stats = await stat(path);
      const content = await readFile(path, "utf8");
      const firstLine = content.split("\n")[0];

      todos.push({
        file,
        path,
        mtime: stats.mtime,
        preview: firstLine.slice(0, 80) + (firstLine.length > 80 ? "..." : ""),
      });
    }

    return todos;
  } catch (err) {
    if (err.code === "ENOENT") {
      console.log("No todos/ directory found");
      return [];
    }
    throw err;
  }
}

function openFile(path) {
  const cmd = isWin
    ? "start"
    : process.platform === "darwin"
      ? "open"
      : "xdg-open";
  spawn(cmd, [path], { detached: true, stdio: "ignore" });
}

async function main() {
  const args = process.argv.slice(2);
  const showLatest = args.includes("--latest");
  const openFlag = args.includes("--open");

  const todos = await listTodos();

  if (todos.length === 0) return;

  if (showLatest) {
    const latest = todos[0];
    console.log(`Latest TODO: ${latest.file} (${latest.mtime.toISOString()})`);
    console.log(latest.preview);
    if (openFlag) {
      openFile(latest.path);
    }
    return;
  }

  console.log(`Found ${todos.length} TODO files:\n`);

  for (const todo of todos) {
    console.log(`${todo.file} (${todo.mtime.toLocaleString()})`);
    console.log(`  ${todo.preview}\n`);
  }

  if (openFlag && todos.length > 0) {
    openFile(todos[0].path);
  }
}

main().catch(console.error);
