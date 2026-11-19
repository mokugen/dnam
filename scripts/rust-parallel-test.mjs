// @ts-check
/**
 * Parallel Rust test-binary runner (compiles once via `cargo test --no-run --message-format=json`
 * and runs all compiled test executables concurrently with awaited sibling `SIGTERM` cleanup).
 */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const CHILD_KILL_GRACE_MS = 2_000;

/**
 * @typedef {{ pkg: string, allFeatures: boolean, bins: boolean }} ParallelTestOptions
 */

/** @type {Set<import("node:child_process").ChildProcess>} */
const GLOBAL_ACTIVE_CHILDREN = new Set();

/**
 * Sends `SIGTERM` to live tracked children and awaits their exit (with `SIGKILL` fallback).
 * @param {Set<import("node:child_process").ChildProcess>} children
 * @returns {Promise<void>}
 */
async function terminateAndWait(children) {
  /** @type {import("node:child_process").ChildProcess[]} */
  const alive = [];
  for (const child of children) {
    if (!child.killed && child.exitCode === null) {
      child.kill("SIGTERM");
      alive.push(child);
    }
  }
  if (alive.length === 0) {
    return;
  }
  await new Promise((resolve) => {
    let remaining = alive.length;
    const timer = setTimeout(() => {
      for (const child of alive) {
        if (child.exitCode === null) {
          child.kill("SIGKILL");
        }
      }
      resolve(undefined);
    }, CHILD_KILL_GRACE_MS);
    timer.unref();
    for (const child of alive) {
      child.once("close", () => {
        remaining -= 1;
        if (remaining <= 0) {
          clearTimeout(timer);
          resolve(undefined);
        }
      });
    }
  });
}

function registerSignalCleanup() {
  for (const sig of /** @type {const} */ (["SIGINT", "SIGTERM"])) {
    process.once(sig, () => {
      terminateAndWait(GLOBAL_ACTIVE_CHILDREN).finally(() => {
        process.exit(sig === "SIGINT" ? 130 : 143);
      });
    });
  }
}

/**
 * Parses and validates CLI flags, exiting with code 2 on invalid arguments.
 * @param {string[]} argv
 * @returns {ParallelTestOptions}
 */
function parseCliArgs(argv) {
  const cleanArgv = argv[0] === "--" ? argv.slice(1) : argv;
  /** @type {ReturnType<typeof parseArgs>} */
  let parsed;
  try {
    parsed = parseArgs({
      args: cleanArgv,
      strict: true,
      allowPositionals: false,
      options: {
        package: { type: "string" },
        "all-features": { type: "boolean", default: false },
        bins: { type: "boolean", default: false },
      },
    });
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err));
    process.exit(2);
  }

  const vals = parsed.values;
  const pkg = typeof vals["package"] === "string" ? vals["package"].trim() : "";
  if (pkg.length === 0) {
    console.error(
      "Usage: node scripts/rust-parallel-test.mjs --package <crate> [--all-features] [--bins]",
    );
    process.exit(2);
  }

  return {
    pkg,
    allFeatures: Boolean(vals["all-features"]),
    bins: Boolean(vals["bins"]),
  };
}

/**
 * Spawns a subprocess, captures stdout/stderr, rejects on spawn error, and registers active child.
 * @param {string} cmd
 * @param {string[]} args
 * @param {string} cwd
 * @returns {Promise<{ code: number, stdout: string, stderr: string, elapsedSec: number }>}
 */
function captureProcess(cmd, args, cwd) {
  return new Promise((resolve, reject) => {
    const started = performance.now();
    const child = spawn(cmd, args, { cwd, stdio: ["ignore", "pipe", "pipe"] });
    GLOBAL_ACTIVE_CHILDREN.add(child);
    /** @type {Buffer[]} */
    const stdoutChunks = [];
    /** @type {Buffer[]} */
    const stderrChunks = [];
    let settled = false;
    child.stdout.on("data", (chunk) => stdoutChunks.push(chunk));
    child.stderr.on("data", (chunk) => stderrChunks.push(chunk));
    child.once("error", (err) => {
      GLOBAL_ACTIVE_CHILDREN.delete(child);
      if (!settled) {
        settled = true;
        reject(err);
      }
    });
    child.once("close", (code) => {
      GLOBAL_ACTIVE_CHILDREN.delete(child);
      if (!settled) {
        settled = true;
        resolve({
          code: code ?? 1,
          stdout: Buffer.concat(stdoutChunks).toString("utf8"),
          stderr: Buffer.concat(stderrChunks).toString("utf8"),
          elapsedSec: (performance.now() - started) / 1000,
        });
      }
    });
  });
}

/**
 * Collects unique compiled test binary paths from `cargo test --no-run --message-format=json` output.
 * @param {string} stdout
 * @returns {string[]}
 */
function collectTestExecutables(stdout) {
  /** @type {string[]} */
  const bins = [];
  /** @type {Set<string>} */
  const seen = new Set();
  for (const line of stdout.split(/\r?\n/)) {
    if (!line.startsWith("{")) {
      continue;
    }
    /** @type {{ reason?: string, executable?: unknown, profile?: { test?: boolean } }} */
    const obj = JSON.parse(line);
    const exe = obj.executable;
    if (
      obj.reason === "compiler-artifact" &&
      typeof exe === "string" &&
      obj.profile?.test &&
      !seen.has(exe) &&
      existsSync(exe)
    ) {
      seen.add(exe);
      bins.push(exe);
    }
  }
  return bins;
}

async function main() {
  registerSignalCleanup();
  const opts = parseCliArgs(process.argv.slice(2));
  const scriptDir = path.dirname(fileURLToPath(import.meta.url));
  const repoRoot = path.resolve(scriptDir, "..");

  const cargoArgs = ["test", "--package", opts.pkg];
  if (opts.allFeatures) {
    cargoArgs.push("--all-features");
  }
  cargoArgs.push("--lib");
  if (opts.bins) {
    cargoArgs.push("--bins");
  }
  cargoArgs.push("--tests", "--no-run", "--message-format=json");

  const startedAll = performance.now();
  const build = await captureProcess("cargo", cargoArgs, repoRoot);
  if (build.code !== 0) {
    process.stderr.write(build.stdout);
    process.stderr.write(build.stderr);
    process.exitCode = build.code;
    return;
  }

  const bins = collectTestExecutables(build.stdout);
  if (bins.length === 0) {
    throw new Error(`No test binaries found in cargo test --no-run output for ${opts.pkg}`);
  }

  /** @type {Array<{ name: string, elapsedSec: number }>} */
  const runs = [];
  /** @type {Error | null} */
  let firstError = null;

  /** @param {Error} err */
  function abortSiblings(err) {
    firstError ??= err;
    for (const ch of GLOBAL_ACTIVE_CHILDREN) {
      if (!ch.killed && ch.exitCode === null) {
        ch.kill("SIGTERM");
      }
    }
  }

  /** @param {string} exe */
  async function runOneBinary(exe) {
    const base = path.basename(exe);
    const threads =
      base.includes("solver_suite") || base.includes("stimulus_and_align_suite") ? "4" : "2";
    try {
      const res = await captureProcess(exe, [`--test-threads=${threads}`], repoRoot);
      if (res.code !== 0) {
        abortSiblings(
          new Error(`FAILED ${base} (${res.elapsedSec.toFixed(2)}s):\n${res.stdout}${res.stderr}`),
        );
        return;
      }
      runs.push({ name: base, elapsedSec: res.elapsedSec });
    } catch (err) {
      abortSiblings(err instanceof Error ? err : new Error(String(err)));
    }
  }

  try {
    await Promise.allSettled(bins.map(runOneBinary));
  } finally {
    await terminateAndWait(GLOBAL_ACTIVE_CHILDREN);
  }
  if (firstError) {
    throw firstError;
  }

  const totalSec = (performance.now() - startedAll) / 1000;
  const summary = [...runs]
    .sort((a, b) => b.elapsedSec - a.elapsedSec)
    .map((r) => `${r.name}=${r.elapsedSec.toFixed(2)}s`)
    .join(", ");
  console.log(
    `All ${bins.length} ${opts.pkg} test binaries passed in ${totalSec.toFixed(2)}s (${summary})`,
  );
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  terminateAndWait(GLOBAL_ACTIVE_CHILDREN).finally(() => {
    process.exit(1);
  });
});
