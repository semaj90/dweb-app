import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';
import assert from 'node:assert';

// Basic smoke test for scripts/is-ready.mjs JSON output mode
const script = resolve(process.cwd(), 'scripts/is-ready.mjs');

function run(args=[]) {
  const res = spawnSync(process.execPath, [script, '--json', '--wait', '0', '--minimal', ...args], { encoding: 'utf8' });
  if (res.error) throw res.error;
  return { stdout: res.stdout, stderr: res.stderr, status: res.status };
}

// 1. Should produce JSON parsable output
const { stdout, status } = run();
assert.equal(status, 0, 'is-ready script should exit 0 for no required checks by default');
let json;
try { json = JSON.parse(stdout.trim().split('\n').slice(-1)[0]); } catch (e) { assert.fail('Output not valid JSON: ' + e.message + '\n' + stdout); }
assert.ok(json.timestamp, 'JSON should include timestamp');
assert.ok(Array.isArray(json.results), 'JSON should include results array');

// 2. Port check that should succeed quickly (loopback 127.0.0.1: "any" not guaranteed). Use 65535 expecting closed -> script should still exit 0 (non-fatal)
// ensure structure stable on second run
const { stdout: stdout2 } = run();
let json2;
try { json2 = JSON.parse(stdout2.trim().split('\n').slice(-1)[0]); } catch (e) { assert.fail('Second output not valid JSON'); }
assert.ok(Array.isArray(json2.results), 'Second run should also contain results');

console.log('is-ready script tests passed');
