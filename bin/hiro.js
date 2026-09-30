#!/usr/bin/env node

/**
 * Hiro — Terminal AI Coding Assistant
 * Node.js entry point & Python bootstrapper for NPM distribution.
 */

const { spawn, spawnSync } = require("child_process");
const path = require("path");
const fs = require("fs");
const os = require("os");

const ROOT_DIR = path.resolve(__dirname, "..");

// Candidates for Python executable
const PYTHON_CANDIDATES = process.platform === "win32"
  ? [
      path.join(os.homedir(), "AppData", "Local", "Python", "bin", "python.exe"),
      "python",
      "py",
      "python3"
    ]
  : ["python3", "python", "python3.12", "python3.11", "python3.13"];

function findPython() {
  for (const cmd of PYTHON_CANDIDATES) {
    try {
      const res = spawnSync(cmd, ["--version"], { encoding: "utf-8", stdio: ["ignore", "pipe", "ignore"] });
      if (res.status === 0 && res.stdout) {
        const match = res.stdout.match(/Python (\d+)\.(\d+)/);
        if (match) {
          const major = parseInt(match[1], 10);
          const minor = parseInt(match[2], 10);
          if (major >= 3 && minor >= 10) {
            return cmd;
          }
        }
      }
    } catch {
      // Continue searching
    }
  }
  return null;
}

function ensureDependencies(pythonCmd) {
  // Check if core packages (prompt_toolkit, rich, httpx) can be imported
  const testScript = "import prompt_toolkit, rich, httpx, pydantic; print('OK')";
  const check = spawnSync(pythonCmd, ["-c", testScript], {
    encoding: "utf-8",
    stdio: ["ignore", "pipe", "ignore"]
  });

  if (check.status !== 0 || !check.stdout.includes("OK")) {
    console.log("\x1b[36m[hiro]\x1b[0m Installing Python dependencies on first run…");
    const pipArgs = ["-m", "pip", "install", "-e", ROOT_DIR];
    const install = spawnSync(pythonCmd, pipArgs, {
      cwd: ROOT_DIR,
      stdio: "inherit",
      env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONUTF8: "1" }
    });

    if (install.status !== 0) {
      console.error("\x1b[31m[hiro error]\x1b[0m Failed to install Python dependencies.");
      console.error("Please run manually: pip install -e " + ROOT_DIR);
    }
  }
}

function main() {
  const pythonCmd = findPython();
  if (!pythonCmd) {
    console.error("\x1b[31m[hiro error]\x1b[0m Python 3.10+ is required but was not found on your PATH.");
    console.error("Please install Python 3.10+ from https://www.python.org/ or via your package manager.");
    process.exit(1);
  }

  ensureDependencies(pythonCmd);

  const env = {
    ...process.env,
    PYTHONIOENCODING: "utf-8",
    PYTHONUTF8: "1",
    PYTHONPATH: ROOT_DIR + (process.env.PYTHONPATH ? path.delimiter + process.env.PYTHONPATH : "")
  };

  const args = ["-m", "hiro.main", ...process.argv.slice(2)];

  const child = spawn(pythonCmd, args, {
    cwd: process.cwd(),
    stdio: "inherit",
    env: env
  });

  child.on("exit", (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
    } else {
      process.exit(code ?? 0);
    }
  });

  // Relay signals to child
  process.on("SIGINT", () => {});
  process.on("SIGTERM", () => {
    child.kill("SIGTERM");
  });
}

main();
