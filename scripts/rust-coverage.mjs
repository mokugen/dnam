// @ts-check
/**
 * Shared parallel LLVM coverage runner for Rust crates in the DNAM monorepo.
 * Compiles instrumented test binaries once via `cargo llvm-cov --no-report`,
 * executes binaries or sharded test jobs concurrently, merges scoped `.profraw`
 * files via `llvm-profdata`, prints `llvm-cov report`, and enforces `--min-lines`.
 */

import { execFileSync, spawn } from "node:child_process";
import { closeSync, existsSync, mkdirSync, openSync, readdirSync, rmSync, statSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const MAX_STDIO_BUFFER_BYTES = 64 * 1024 * 1024;
const CHILD_KILL_GRACE_MS = 2_000;
const PROJ_NAMED_PREFIXES = ["adversarial_regressions::", "bundle_and_vault_edges::", "security::"];

const HEAVY_CAPTURE_KEYS = [
  "mode1_streaming",
  "mode2_batch_96khz",
  "chains_and_watermarks",
  "48k_window",
  "silent_muted",
  "steep_",
  "live_null",
  "multirate_bitdepth",
  "case_09",
  "routing_topologies",
  "adversaries_and_topologies",
];

/**
 * @typedef {{
 *   pkg: string,
 *   sources: string,
 *   profile: "cov" | "default",
 *   allFeatures: boolean,
 *   bins: boolean,
 *   mode: "bins" | "sharded-capture",
 *   minLines: number,
 * }} CoverageCliOptions
 *
 * @typedef {{
 *   label: string,
 *   cmd: string[],
 * }} CoverageJob
 */

/** @type {Set<import("node:child_process").ChildProcess>} */
const GLOBAL_ACTIVE_CHILDREN = new Set();

/**
 * Sends `SIGTERM` to all live tracked child processes and waits briefly before `SIGKILL`.
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
 * Parses and strictly validates CLI flags, exiting with code 2 on invalid input.
 * @param {string[]} argv
 * @returns {CoverageCliOptions}
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
        sources: { type: "string" },
        profile: { type: "string", default: "cov" },
        "all-features": { type: "boolean", default: false },
        bins: { type: "boolean", default: false },
        mode: { type: "string", default: "bins" },
        "min-lines": { type: "string" },
        threshold: { type: "string" },
      },
    });
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err));
    process.exit(2);
  }

  const vals = parsed.values;
  const pkg = typeof vals["package"] === "string" ? vals["package"].trim() : "";
  const sources = typeof vals["sources"] === "string" ? vals["sources"].trim() : "";
  const profile = typeof vals["profile"] === "string" ? vals["profile"] : "cov";
  const mode = typeof vals["mode"] === "string" ? vals["mode"] : "bins";
  const thresholdRaw =
    typeof vals["min-lines"] === "string"
      ? vals["min-lines"]
      : typeof vals["threshold"] === "string"
        ? vals["threshold"]
        : "95";
  const thresholdTrimmed = thresholdRaw.trim();
  const minLines = Number(thresholdTrimmed);

  if (
    pkg.length === 0 ||
    sources.length === 0 ||
    thresholdTrimmed.length === 0 ||
    (profile !== "cov" && profile !== "default") ||
    (mode !== "bins" && mode !== "sharded-capture") ||
    !Number.isFinite(minLines) ||
    minLines < 0 ||
    minLines > 100
  ) {
    console.error(
      "Usage: node scripts/rust-coverage.mjs --package <crate> --sources <dir> [--profile cov|default] [--all-features] [--bins] [--mode bins|sharded-capture] [--min-lines 95]",
    );
    process.exit(2);
  }

  return {
    pkg,
    sources,
    profile,
    allFeatures: Boolean(vals["all-features"]),
    bins: Boolean(vals["bins"]),
    mode,
    minLines,
  };
}

/**
 * Resolves `llvm-profdata` and `llvm-cov` binaries from the active `rustc` sysroot.
 * @returns {{ llvmProfdata: string, llvmCov: string }}
 */
function resolveLlvmTools() {
  const sysroot = execFileSync("rustc", ["--print", "sysroot"], {
    encoding: "utf8",
    maxBuffer: MAX_STDIO_BUFFER_BYTES,
  }).trim();
  const vv = execFileSync("rustc", ["-vV"], {
    encoding: "utf8",
    maxBuffer: MAX_STDIO_BUFFER_BYTES,
  });
  const hostLine = vv.split(/\r?\n/).find((line) => line.startsWith("host: "));
  if (!hostLine) {
    throw new Error("Could not determine rustc host triple from `rustc -vV`");
  }
  const host = hostLine.slice("host: ".length).trim();
  const ext = process.platform === "win32" ? ".exe" : "";
  const llvmBin = path.join(sysroot, "lib", "rustlib", host, "bin");
  return {
    llvmProfdata: path.join(llvmBin, `llvm-profdata${ext}`),
    llvmCov: path.join(llvmBin, `llvm-cov${ext}`),
  };
}

/**
 * Spawns a subprocess, registers it for signal cleanup, and rejects on spawn error.
 * @param {string} cmd
 * @param {string[]} args
 * @param {string} cwd
 * @returns {Promise<{ code: number, output: string }>}
 */
function runCommandCombined(cmd, args, cwd) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, {
      cwd,
      stdio: ["ignore", "pipe", "pipe"],
    });
    GLOBAL_ACTIVE_CHILDREN.add(child);
    /** @type {Buffer[]} */
    const chunks = [];
    let settled = false;
    child.stdout.on("data", (chunk) => chunks.push(chunk));
    child.stderr.on("data", (chunk) => chunks.push(chunk));
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
          output: Buffer.concat(chunks).toString("utf8"),
        });
      }
    });
  });
}

/**
 * Executes a single coverage test job, registering the child for sibling abort handling.
 * @param {CoverageJob} job
 * @param {NodeJS.ProcessEnv} env
 * @param {string} cwd
 * @param {Set<import("node:child_process").ChildProcess>} poolChildren
 * @returns {Promise<{ label: string, code: number, output: string, elapsedSec: number }>}
 */
function runJobAsync(job, env, cwd, poolChildren) {
  return new Promise((resolve, reject) => {
    const bin = job.cmd[0];
    if (!bin) {
      reject(new Error(`Empty command for job ${job.label}`));
      return;
    }
    const started = performance.now();
    const child = spawn(bin, job.cmd.slice(1), {
      cwd,
      env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    GLOBAL_ACTIVE_CHILDREN.add(child);
    poolChildren.add(child);
    /** @type {Buffer[]} */
    const chunks = [];
    let settled = false;
    child.stdout.on("data", (chunk) => chunks.push(chunk));
    child.stderr.on("data", (chunk) => chunks.push(chunk));
    child.once("error", (err) => {
      GLOBAL_ACTIVE_CHILDREN.delete(child);
      poolChildren.delete(child);
      if (!settled) {
        settled = true;
        reject(err);
      }
    });
    child.once("close", (code) => {
      GLOBAL_ACTIVE_CHILDREN.delete(child);
      poolChildren.delete(child);
      if (!settled) {
        settled = true;
        resolve({
          label: job.label,
          code: code ?? 1,
          output: Buffer.concat(chunks).toString("utf8"),
          elapsedSec: (performance.now() - started) / 1000,
        });
      }
    });
  });
}

/**
 * Sends `SIGTERM` to every still-live child in `children`.
 * @param {Set<import("node:child_process").ChildProcess>} children
 */
function signalChildrenTerm(children) {
  for (const ch of children) {
    if (!ch.killed && ch.exitCode === null) {
      ch.kill("SIGTERM");
    }
  }
}

/**
 * Runs jobs concurrently across up to `maxWorkers`, terminating and awaiting siblings if any fails.
 * @param {CoverageJob[]} jobs
 * @param {number} maxWorkers
 * @param {NodeJS.ProcessEnv} env
 * @param {string} cwd
 * @returns {Promise<Array<{ label: string, code: number, output: string, elapsedSec: number }>>}
 */
async function runPool(jobs, maxWorkers, env, cwd) {
  /** @type {Array<{ label: string, code: number, output: string, elapsedSec: number }>} */
  const results = [];
  /** @type {Set<import("node:child_process").ChildProcess>} */
  const poolChildren = new Set();
  let cursor = 0;
  let aborted = false;
  /** @type {Error | null} */
  let firstError = null;

  /** @param {Error} err */
  function abortPool(err) {
    aborted = true;
    firstError ??= err;
    signalChildrenTerm(poolChildren);
  }

  async function worker() {
    while (!aborted) {
      const idx = cursor;
      cursor += 1;
      const job = jobs[idx];
      if (!job) {
        return;
      }
      try {
        const res = await runJobAsync(job, env, cwd, poolChildren);
        results.push(res);
        if (res.code !== 0) {
          abortPool(
            new Error(`FAILED ${res.label} (${res.elapsedSec.toFixed(2)}s):\n${res.output}`),
          );
          return;
        }
      } catch (err) {
        abortPool(err instanceof Error ? err : new Error(String(err)));
        return;
      }
    }
  }

  try {
    const workerCount = Math.max(1, Math.min(maxWorkers, jobs.length));
    await Promise.allSettled(Array.from({ length: workerCount }, () => worker()));
  } finally {
    await terminateAndWait(poolChildren);
  }
  if (firstError) {
    throw firstError;
  }
  return results;
}

/**
 * Extracts space-safe executable paths printed by `cargo llvm-cov --no-report`.
 * @param {string} output
 * @param {string} repoRoot
 * @returns {string[]}
 */
function extractTestBinaries(output, repoRoot) {
  /** @type {string[]} */
  const bins = [];
  /** @type {Set<string>} */
  const seen = new Set();
  for (const match of output.matchAll(/^\s*(?:Running|Executable)\s+[^\n]*\(([^)\r\n]+)\)\s*$/gm)) {
    const raw = match[1]?.trim();
    if (!raw || raw.endsWith(".rs")) {
      continue;
    }
    const abs = path.isAbsolute(raw) ? raw : path.join(repoRoot, raw);
    if (!seen.has(abs) && existsSync(abs) && statSync(abs).isFile()) {
      seen.add(abs);
      bins.push(abs);
    }
  }
  return bins;
}

/**
 * Enumerates tests inside a compiled Rust test binary without writing profraw noise or piping stderr.
 * @param {string} binPath
 * @param {string} repoRoot
 * @param {string[]} extraArgs
 * @returns {string[]}
 */
function listBinaryTests(binPath, repoRoot, extraArgs) {
  const devNullFd = openSync(os.devNull, "w");
  try {
    const out = execFileSync(binPath, ["--list", ...extraArgs], {
      cwd: repoRoot,
      env: { ...process.env, LLVM_PROFILE_FILE: os.devNull },
      stdio: ["ignore", "pipe", devNullFd],
      encoding: "utf8",
      maxBuffer: MAX_STDIO_BUFFER_BYTES,
    });
    return out.split(/\r?\n/);
  } finally {
    closeSync(devNullFd);
  }
}

/**
 * Builds parallel execution jobs across unit tests, integration tests, and capture shards.
 * @param {string[]} bins
 * @param {CoverageCliOptions} opts
 * @param {string} repoRoot
 * @returns {CoverageJob[]}
 */
function buildJobs(bins, opts, repoRoot) {
  const crateSnake = opts.pkg.replaceAll("-", "_");
  if (opts.mode !== "sharded-capture") {
    return bins.map((bin) => {
      const name = path.basename(bin);
      const threads = name.startsWith(`${crateSnake}-`) ? "4" : "2";
      return { label: name, cmd: [bin, `--test-threads=${threads}`] };
    });
  }

  /** @type {CoverageJob[]} */
  const libJobs = [];
  /** @type {CoverageJob[]} */
  const heavyJobs = [];
  /** @type {CoverageJob[]} */
  const midJobs = [];
  /** @type {CoverageJob[]} */
  const projJobs = [];

  for (const bin of bins) {
    const name = path.basename(bin);
    if (name.startsWith(`${crateSnake}-`)) {
      libJobs.push(
        { label: "lib:v3_varpro", cmd: [bin, "solver::v3_varpro::", "--test-threads=1"] },
        {
          label: "lib:solver_core",
          cmd: [bin, "solver::", "--skip", "solver::v3_varpro::", "--test-threads=1"],
        },
        { label: "lib:non_solver", cmd: [bin, "--skip", "solver::", "--test-threads=1"] },
      );
      continue;
    }
    if (name.includes("proj_suite")) {
      const skipArgs = PROJ_NAMED_PREFIXES.flatMap((prefix) => ["--skip", prefix]);
      projJobs.push(
        { label: "proj:adversarial", cmd: [bin, "adversarial_regressions::", "--test-threads=1"] },
        { label: "proj:bundle_vault", cmd: [bin, "bundle_and_vault_edges::", "--test-threads=1"] },
        { label: "proj:dau1_security", cmd: [bin, "dau1_security::", "--test-threads=1"] },
        {
          label: "proj:security",
          cmd: [bin, "security::", "--skip", "dau1_security::", "--test-threads=1"],
        },
        {
          label: "proj:other",
          cmd: [bin, ...skipArgs, "--skip", "dau1_security::", "--test-threads=1"],
        },
      );
      continue;
    }
    const allLines = listBinaryTests(bin, repoRoot, []);
    const ignored = new Set(listBinaryTests(bin, repoRoot, ["--ignored"]));
    for (const line of allLines) {
      if (!line.endsWith(": test") || ignored.has(line)) {
        continue;
      }
      const testName = line.slice(0, -": test".length);
      /** @type {CoverageJob} */
      const entry = {
        label: testName,
        cmd: [bin, "--exact", testName, "--test-threads=1"],
      };
      if (HEAVY_CAPTURE_KEYS.some((key) => testName.includes(key))) {
        heavyJobs.push(entry);
      } else {
        midJobs.push(entry);
      }
    }
  }
  return [...libJobs, ...heavyJobs, ...projJobs, ...midJobs];
}

/**
 * Removes files in `dir` matching `predicate`.
 * @param {string} dir
 * @param {(name: string) => boolean} predicate
 */
function removeMatchingFiles(dir, predicate) {
  if (!existsSync(dir)) {
    return;
  }
  for (const name of readdirSync(dir)) {
    if (predicate(name)) {
      rmSync(path.join(dir, name), { force: true });
    }
  }
}

async function main() {
  registerSignalCleanup();
  const opts = parseCliArgs(process.argv.slice(2));
  const scriptDir = path.dirname(fileURLToPath(import.meta.url));
  const repoRoot = path.resolve(scriptDir, "..");
  const srcDir = path.resolve(repoRoot, opts.sources);
  const profDir = path.join(repoRoot, "target", "llvm-cov-target");
  const { llvmProfdata, llvmCov } = resolveLlvmTools();

  mkdirSync(profDir, { recursive: true });
  const profPrefix = `${opts.pkg}-cov-`;
  const profdataFile = `${opts.pkg}-cov.profdata`;
  const profdataPath = path.join(profDir, profdataFile);
  removeMatchingFiles(
    profDir,
    (name) => (name.startsWith(profPrefix) && name.endsWith(".profraw")) || name === profdataFile,
  );

  const cargoArgs = ["llvm-cov"];
  if (opts.profile !== "default") {
    cargoArgs.push("--profile", opts.profile);
  }
  cargoArgs.push("--package", opts.pkg);
  if (opts.allFeatures) {
    cargoArgs.push("--all-features");
  }
  cargoArgs.push("--lib");
  if (opts.bins) {
    cargoArgs.push("--bins");
  }
  cargoArgs.push("--tests", "--no-report", "--", "--exact", "__none__");

  const buildRes = await runCommandCombined("cargo", cargoArgs, repoRoot);
  if (buildRes.code !== 0) {
    process.stderr.write(buildRes.output);
    process.exitCode = buildRes.code;
    return;
  }

  const bins = extractTestBinaries(buildRes.output, repoRoot);
  const firstBin = bins[0];
  if (!firstBin) {
    process.stderr.write(buildRes.output);
    throw new Error(`No instrumented ${opts.pkg} test binaries found`);
  }

  const jobs = buildJobs(bins, opts, repoRoot);
  const workerCap =
    opts.mode === "sharded-capture"
      ? Math.max(4, Math.min(8, os.availableParallelism()))
      : Math.min(8, jobs.length);
  const runEnv = {
    ...process.env,
    LLVM_PROFILE_FILE: path.join(profDir, `${profPrefix}%p-%8m.profraw`),
  };

  const startedAll = performance.now();
  const timings = await runPool(jobs, workerCap, runEnv, repoRoot);
  const totalSec = (performance.now() - startedAll) / 1000;

  if (opts.mode === "sharded-capture") {
    const top5 = [...timings]
      .sort((a, b) => b.elapsedSec - a.elapsedSec)
      .slice(0, 5)
      .map((item) => `${item.label}=${item.elapsedSec.toFixed(1)}s`)
      .join(", ");
    console.log(
      `Sharded coverage tests passed (${jobs.length} jobs across ${workerCap}w in ${totalSec.toFixed(2)}s; top5: ${top5})`,
    );
  }

  const profraws = readdirSync(profDir)
    .filter((name) => name.startsWith(profPrefix) && name.endsWith(".profraw"))
    .map((name) => path.join(profDir, name));
  if (profraws.length === 0) {
    throw new Error(`No ${profPrefix}*.profraw files generated`);
  }

  execFileSync(llvmProfdata, ["merge", "-sparse", ...profraws, "-o", profdataPath], {
    cwd: repoRoot,
    stdio: "inherit",
    maxBuffer: MAX_STDIO_BUFFER_BYTES,
  });
  const objectArgs = [firstBin, ...bins.slice(1).map((bin) => `--object=${bin}`)];
  execFileSync(
    llvmCov,
    ["report", `--instr-profile=${profdataPath}`, ...objectArgs, "--sources", srcDir],
    { cwd: repoRoot, stdio: "inherit", maxBuffer: MAX_STDIO_BUFFER_BYTES },
  );
  const exportJson = execFileSync(
    llvmCov,
    [
      "export",
      "--summary-only",
      `--instr-profile=${profdataPath}`,
      ...objectArgs,
      "--sources",
      srcDir,
    ],
    { cwd: repoRoot, encoding: "utf8", maxBuffer: MAX_STDIO_BUFFER_BYTES },
  );
  /** @type {{ data?: Array<{ totals?: { lines?: { percent?: number } } }> }} */
  const parsedExport = JSON.parse(exportJson);
  const pct = parsedExport.data?.[0]?.totals?.lines?.percent;
  if (typeof pct !== "number") {
    throw new Error("Could not parse line coverage percentage from llvm-cov export JSON");
  }

  removeMatchingFiles(
    profDir,
    (name) => (name.startsWith(profPrefix) && name.endsWith(".profraw")) || name === profdataFile,
  );

  console.log(
    `${opts.pkg} line coverage: ${pct.toFixed(2)}% (tests ran in ${totalSec.toFixed(2)}s)`,
  );
  if (pct < opts.minLines) {
    throw new Error(`Line coverage ${pct.toFixed(2)}% < ${opts.minLines.toFixed(1)}%`);
  }
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  terminateAndWait(GLOBAL_ACTIVE_CHILDREN).finally(() => {
    process.exit(1);
  });
});
