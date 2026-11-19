// End-to-end test: real browser + real FastAPI backend (mock provider) + Vite dev server.
//
//   npm run e2e                     (from frontend/; needs ../backend/.venv set up)
//   SCREENSHOT_DIR=/some/dir npm run e2e   to also save screenshots
//
// It starts both servers on spare ports with a throwaway storage directory, runs the
// flows below in Chromium (desktop + mobile), and always shuts the servers down.
import { spawn, spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, devices } from "playwright";

const HERE = dirname(fileURLToPath(import.meta.url));
const FRONTEND = resolve(HERE, "..");
const BACKEND = resolve(FRONTEND, "../backend");
const API_PORT = 8765;
const WEB_PORT = 5199;
const WEB = `http://127.0.0.1:${WEB_PORT}`;
const SHOTS = process.env.SCREENSHOT_DIR || null;

const work = mkdtempSync(join(tmpdir(), "vidsgen-e2e-"));
const procs = [];
const results = [];
let failed = 0;

// ------------------------------------------------------------------ helpers
function start(cmd, args, opts) {
  const p = spawn(cmd, args, { ...opts, stdio: ["ignore", "pipe", "pipe"] });
  let log = "";
  p.stdout.on("data", (d) => (log += d));
  p.stderr.on("data", (d) => (log += d));
  p.logs = () => log;
  procs.push(p);
  return p;
}

async function waitFor(url, timeoutMs = 30000) {
  const end = Date.now() + timeoutMs;
  while (Date.now() < end) {
    try {
      const r = await fetch(url);
      if (r.ok) return;
    } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`Timed out waiting for ${url}`);
}

function check(cond, msg) {
  if (!cond) throw new Error(msg);
}

async function test(name, fn) {
  const t0 = Date.now();
  try {
    await fn();
    results.push(`  ✓ ${name} (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
  } catch (e) {
    failed++;
    results.push(`  ✗ ${name}\n      ${String(e.message || e).split("\n")[0]}`);
  }
}

async function shot(page, name) {
  if (SHOTS) await page.screenshot({ path: join(SHOTS, `${name}.png`), fullPage: false });
}

function makeImage(path, w, h) {
  // Uses the backend's Pillow to create a test image (patterned so frames are visible).
  const py = `from PIL import Image, ImageDraw
im = Image.new("RGB", (${w}, ${h}), (24, 30, 46)); d = ImageDraw.Draw(im)
for i in range(0, ${w}, 48): d.rectangle([i, 0, i + 22, ${h}], fill=(240, 180, 90))
d.ellipse([${w}//2-80, ${h}//2-80, ${w}//2+80, ${h}//2+80], fill=(138, 168, 255))
im.save(${JSON.stringify(path)})`;
  const r = spawnSync(join(BACKEND, ".venv/bin/python"), ["-c", py]);
  check(r.status === 0, `could not create test image: ${r.stderr}`);
  return path;
}

async function waitStatus(page, label, timeout = 30000) {
  await page.locator(".stage .badge", { hasText: label }).first().waitFor({ timeout });
}

async function videoPlays(page, selector = ".player video") {
  const v = page.locator(selector).first();
  await v.waitFor();
  await page.waitForFunction((sel) => {
    const el = document.querySelector(sel);
    return el && el.readyState >= 2 && el.duration > 0;
  }, selector, { timeout: 15000 });
  return v.evaluate(async (el) => {
    el.muted = true;
    await el.play();
    const t = el.currentTime;
    await new Promise((r) => setTimeout(r, 400));
    const ok = el.currentTime > t;
    el.pause();
    return { ok, duration: el.duration, w: el.videoWidth, h: el.videoHeight };
  });
}

async function noHorizontalOverflow(page) {
  return page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1);
}

// ------------------------------------------------------------------ run
async function main() {
  console.log(`Starting backend (mock provider) on :${API_PORT} and Vite on :${WEB_PORT}`);
  start(join(BACKEND, ".venv/bin/uvicorn"), ["app.main:app", "--port", String(API_PORT), "--log-level", "warning"], {
    cwd: BACKEND,
    env: { ...process.env, STORAGE_LOCAL_ROOT: join(work, "storage"), MOCK_STEP_DELAY_SECONDS: "0.05", APP_ENV: "test" },
  });
  start(join(FRONTEND, "node_modules/.bin/vite"), ["--port", String(WEB_PORT), "--strictPort", "--host", "127.0.0.1"], {
    cwd: FRONTEND,
    env: { ...process.env, VITE_DEV_API_PROXY: `http://127.0.0.1:${API_PORT}` },
  });
  await waitFor(`http://127.0.0.1:${API_PORT}/api/health`);
  await waitFor(WEB);

  const goodImg = makeImage(join(work, "start.png"), 832, 480);
  const tinyImg = makeImage(join(work, "tiny.png"), 120, 120);

  const browser = await chromium.launch();
  const desktop = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: SHOTS ? 2 : 1, acceptDownloads: true });
  const page = await desktop.newPage();
  const consoleErrors = [];
  page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });
  page.on("pageerror", (e) => consoleErrors.push(String(e)));

  // Count job-polling requests to verify polling stops after completion.
  const polls = new Map();
  page.on("request", (r) => {
    const m = r.url().match(/\/api\/jobs\/(job_[a-f0-9]+)$/);
    if (m && r.method() === "GET") polls.set(m[1], (polls.get(m[1]) ?? 0) + 1);
  });

  let t2vJobId = null;

  await test("History shows an empty state before any generation", async () => {
    await page.goto(`${WEB}/generations`);
    await page.getByRole("heading", { name: "No generations yet" }).waitFor();
    await shot(page, "desktop-history-empty");
  });

  await test("Create page loads with backend health and supported options only", async () => {
    await page.goto(WEB);
    await page.getByRole("heading", { name: "Create video" }).waitFor();
    await page.locator(".health--ok", { hasText: "Online" }).first().waitFor();
    const res = await page.locator(".res-card .res-card__value").allTextContents();
    check(JSON.stringify(res) === JSON.stringify(["1280×704", "704×1280", "832×480", "480×832", "640×640"]), `resolutions: ${res}`);
    const frames = await page.locator("#frames option").evaluateAll((os) => os.map((o) => Number(o.value)));
    check(frames[0] === 17 && frames.at(-1) === 121 && frames.every((n) => (n - 1) % 4 === 0), `frames: ${frames}`);
    await shot(page, "desktop-create-empty");
  });

  await test("Client-side validation blocks an empty prompt", async () => {
    await page.getByRole("button", { name: "Generate", exact: true }).click();
    await page.getByText("Describe the video you want to create.").waitFor();
  });

  await test("T2V: submit, live progress, completion, playback", async () => {
    await page.fill("#prompt", "A paper boat drifting down a rainy neon street at night, cinematic dolly shot");
    await page.selectOption("#frames", "49");
    await page.locator(".res-card", { hasText: "1280×704" }).click();
    await page.getByRole("button", { name: "Generate", exact: true }).click();
    await page.waitForURL(/\?job=job_/);
    t2vJobId = new URL(page.url()).searchParams.get("job");
    await waitStatus(page, "Generating");
    await page.waitForFunction(() => {
      const el = document.querySelector('.stage [role="progressbar"]');
      return el && Number(el.getAttribute("aria-valuenow")) > 0;
    });
    await shot(page, "desktop-create-processing");
    await waitStatus(page, "Completed");
    const v = await videoPlays(page);
    check(v.ok, "video did not advance when played");
    check(v.w === 1280 && v.h === 704, `video size ${v.w}x${v.h}`);
    check(Math.abs(v.duration - 49 / 24) < 0.15, `duration ${v.duration}`);
    await page.locator(".player video").evaluate((el) => { el.currentTime = el.duration * 0.4; });
    await page.waitForTimeout(300);
    await shot(page, "desktop-create-completed");
  });

  await test("Polling is moderate and stops once the job is finished", async () => {
    const n = polls.get(t2vJobId) ?? 0;
    check(n > 0 && n < 20, `unexpected poll count ${n}`);
    await page.waitForTimeout(4000);
    check((polls.get(t2vJobId) ?? 0) === n, "kept polling after completion");
  });

  await test("Download delivers a valid MP4", async () => {
    const [dl] = await Promise.all([page.waitForEvent("download"), page.getByRole("link", { name: "Download" }).click()]);
    const path = join(work, "download.mp4");
    await dl.saveAs(path);
    const buf = readFileSync(path);
    check(dl.suggestedFilename() === `vidsgen-${t2vJobId}.mp4`, `filename ${dl.suggestedFilename()}`);
    check(buf.subarray(4, 8).toString() === "ftyp" && statSync(path).size > 10000, "not an MP4");
  });

  await test("I2V: requires an image, rejects a too-small one, previews and removes", async () => {
    await page.goto(WEB);
    await page.getByRole("radio", { name: "Image to Video" }).click();
    await page.fill("#prompt", "The scene slowly comes alive, gentle camera push-in");
    await page.getByRole("button", { name: "Generate", exact: true }).click();
    await page.getByText("Add a start image for image-to-video.").waitFor();
    await page.locator('input[type="file"]').setInputFiles(tinyImg);
    await page.getByText(/shortest side must be at least 256px/).waitFor();
    await page.locator('input[type="file"]').setInputFiles(goodImg);
    await page.getByRole("img", { name: "Selected start image" }).waitFor();
    await page.locator(".dropzone__sub", { hasText: "832×480 ·" }).waitFor();
    await page.getByRole("button", { name: "Remove image" }).click();
    await page.getByText(/Drop a start image/).waitFor();
    await page.locator('input[type="file"]').setInputFiles(goodImg);
    await page.getByRole("img", { name: "Selected start image" }).waitFor();
    await shot(page, "desktop-create-i2v-ready");
  });

  await test("I2V: generation completes and plays", async () => {
    await page.selectOption("#frames", "17");
    await page.getByRole("button", { name: "Generate", exact: true }).click();
    await waitStatus(page, "Completed");
    const v = await videoPlays(page);
    check(v.ok && v.w === 832 && v.h === 480, `i2v video ${JSON.stringify(v)}`);
  });

  await test("Cancel stops a running generation", async () => {
    await page.goto(WEB);
    await page.fill("#prompt", "A long, slow timelapse of clouds over mountains");
    await page.locator("details.section--collapsible summary").click();
    await page.locator("#steps").fill("60");
    await page.getByRole("button", { name: "Generate", exact: true }).click();
    await waitStatus(page, "Generating");
    await page.getByRole("button", { name: "Cancel" }).click();
    await waitStatus(page, "Cancelled");
    await page.getByText(/Stopped at \d+%/).waitFor();
    check(!(await page.getByRole("button", { name: "Cancel" }).isVisible()), "cancel button still shown");
  });

  await test("Failed generations show the error from the API", async () => {
    await page.goto(WEB);
    await page.fill("#prompt", "Show me an error [mock:fail]");
    await page.getByRole("button", { name: "Generate", exact: true }).click();
    await waitStatus(page, "Failed");
    await page.getByText("Simulated provider failure").waitFor();
    await page.getByText("mock_failure").waitFor();
    await shot(page, "desktop-create-failed");
  });

  await test("Fixed seed is sent and recorded", async () => {
    await page.goto(WEB);
    await page.fill("#prompt", "Ink drops blooming in clear water, macro");
    await page.locator("details.section--collapsible summary").click();
    await page.locator(".switch").click();           // turn off random
    await page.fill("#seed", "424242");
    await page.selectOption("#frames", "17");
    await page.getByRole("button", { name: "Generate", exact: true }).click();
    await waitStatus(page, "Completed");
    await page.getByRole("link", { name: "Details" }).click();
    await page.locator(".kv__row", { hasText: "Seed" }).getByText("424242").waitFor();
  });

  await test("History lists generations, filters, and reopens a video", async () => {
    await page.goto(`${WEB}/generations`);
    await page.locator(".card:not(.card--skeleton)").first().waitFor();
    const count = await page.locator(".card").count();
    check(count === 5, `expected 5 cards, got ${count}`);
    await page.waitForFunction(() => [...document.querySelectorAll(".card video")].every((v) => v.readyState >= 2));
    await shot(page, "desktop-history");
    await page.getByRole("tab", { name: "Completed" }).click();
    await page.waitForFunction(() => document.querySelectorAll(".card").length === 3);
    await page.getByRole("tab", { name: "Failed" }).click();
    await page.waitForFunction(() => document.querySelectorAll(".card").length === 1);
    await page.getByRole("tab", { name: "All" }).click();
    await page.locator(`a.card[href="/generations/${t2vJobId}"]`).click();
    await page.getByRole("heading", { name: "Details" }).waitFor();
    const v = await videoPlays(page);
    check(v.ok, "detail video did not play");
    await page.getByText("A paper boat drifting").first().waitFor();
    await page.locator(".player video").evaluate((el) => { el.currentTime = el.duration * 0.6; });
    await page.waitForTimeout(300);
    await shot(page, "desktop-detail");
    const [dl] = await Promise.all([page.waitForEvent("download"), page.getByRole("link", { name: "Download" }).click()]);
    check(dl.suggestedFilename().endsWith(".mp4"), "detail download");
  });

  await test("Reuse settings pre-fills the create form", async () => {
    await page.getByRole("button", { name: "Reuse settings" }).click();
    await page.waitForURL(`${WEB}/`);
    check((await page.inputValue("#prompt")).startsWith("A paper boat"), "prompt not reused");
    check((await page.inputValue("#frames")) === "49", "frames not reused");
    check(await page.locator(".res-card.is-active", { hasText: "1280×704" }).isVisible(), "resolution not reused");
  });

  await test("Unknown generation shows a not-found state", async () => {
    await page.goto(`${WEB}/generations/job_doesnotexist`);
    await page.getByRole("heading", { name: "Generation not found" }).waitFor();
  });

  await test("Settings shows live API status", async () => {
    await page.goto(`${WEB}/settings`);
    await page.getByText("Healthy").waitFor();
    await page.getByText("mock provider ready (no GPU)").waitFor();
    await shot(page, "desktop-settings");
  });

  await test("No browser console errors", async () => {
    const relevant = consoleErrors.filter((e) => !/404 \(Not Found\)/.test(e)); // expected from the not-found test
    check(relevant.length === 0, relevant.join(" | "));
  });

  // ---------------------------------------------------------------- mobile
  const mobile = await browser.newContext({ ...devices["iPhone 13"], acceptDownloads: true });
  const m = await mobile.newPage();

  await test("Mobile: tab bar navigation, no horizontal overflow on any page", async () => {
    await m.goto(WEB);
    check(!(await m.locator(".sidebar").isVisible()), "sidebar visible on mobile");
    check(await m.locator(".tabbar").isVisible(), "tab bar missing");
    for (const path of ["/", "/generations", `/generations/${t2vJobId}`, "/settings"]) {
      await m.goto(WEB + path);
      await m.waitForLoadState("networkidle");
      check(await noHorizontalOverflow(m), `horizontal overflow on ${path}`);
    }
    await m.goto(WEB);
    await shot(m, "mobile-create");
  });

  await test("Mobile: T2V generation end to end", async () => {
    await m.goto(WEB);
    await m.fill("#prompt", "Golden hour over sand dunes, wind lifting fine sand");
    await m.selectOption("#frames", "17");
    await m.getByRole("button", { name: "Generate", exact: true }).click();
    await m.locator(".stage .badge", { hasText: "Completed" }).waitFor({ timeout: 30000 });
    const v = await videoPlays(m);
    check(v.ok, "mobile video did not play");
    check(await noHorizontalOverflow(m), "overflow after generation");
    await m.locator(".stage").scrollIntoViewIfNeeded();
    await shot(m, "mobile-completed");
    await m.getByRole("link", { name: "Library" }).click();
    await m.locator(".card").first().waitFor();
    await m.waitForFunction(() => [...document.querySelectorAll(".card video")].slice(0, 2).every((v) => v.readyState >= 2));
    await shot(m, "mobile-history");
  });

  await browser.close();
}

try {
  await main();
} catch (e) {
  failed++;
  results.push(`  ✗ setup: ${e.message}`);
  for (const p of procs) console.error(p.logs().slice(-2000));
} finally {
  for (const p of procs) p.kill("SIGTERM");
  rmSync(work, { recursive: true, force: true });
}

console.log(results.join("\n"));
console.log(`\n${results.length - failed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
