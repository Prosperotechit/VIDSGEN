// Regenerates the API-format workflow templates from the OFFICIAL Comfy-Org UI workflow,
// using ComfyUI's own frontend exporter (the same code as "Export (API)" in the UI).
//
// Why: the official template is a UI-format graph, and ComfyUI's node inputs change between
// versions (SaveVideo's format/codec became nested "DynamicCombo" inputs, CreateVideo gained
// new optional inputs). Hand-converting would be guesswork; the real exporter is not.
//
// Requirements (none of this touches model weights):
//   * a running ComfyUI (any machine, CPU is fine) whose models folders contain files with the
//     three official names. Empty placeholder files are enough for exporting and validation.
//   * Playwright from the frontend's dev dependencies (cd frontend && npm install).
//
// Usage (from the repo root):
//   node workflows/comfyui/tools/export_api_workflows.mjs \
//        --official path/to/video_wan2_2_5B_ti2v.json --comfyui http://127.0.0.1:8188
//
// Writes:
//   workflows/comfyui/wan22_ti2v_5b_t2v.json         (Load Image bypassed: text-to-video)
//   workflows/comfyui/wan22_ti2v_5b_i2v.json         (Load Image enabled: image-to-video)
//   backend/tests/fixtures/comfyui/object_info.json  (schemas of the node classes used, for offline tests)
//   workflows/comfyui/export_provenance.json         (versions used for this export)
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "../../..");
const require = createRequire(resolve(REPO, "frontend/package.json"));
const { chromium } = require("playwright");

const arg = (name, def) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : def;
};
const officialPath = arg("official");
const comfy = (arg("comfyui", "http://127.0.0.1:8188")).replace(/\/+$/, "");
if (!officialPath) {
  console.error("Missing --official <path to video_wan2_2_5B_ti2v.json>");
  process.exit(2);
}

const officialRaw = readFileSync(officialPath);
const official = JSON.parse(officialRaw);
if (!official.nodes?.some((n) => n.type === "Wan22ImageToVideoLatent")) {
  throw new Error("This does not look like the official Wan 2.2 5B TI2V workflow (no Wan22ImageToVideoLatent node).");
}

// Node ids in the official template (verified 2026-11-19; the export re-checks them below).
const LOAD_IMAGE = 56;
const SAVE_VIDEO = 58;

const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto(`${comfy}/`);
await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 60000 });

const exported = await page.evaluate(
  async ({ wf, LOAD_IMAGE, SAVE_VIDEO }) => {
    const setWidget = async (node, name, value) => {
      const w = node.widgets.find((x) => x.name === name);
      if (!w) throw new Error(`widget ${name} not found on ${node.type}`);
      if (w.options?.values && !w.options.values.includes(value)) {
        throw new Error(`${node.type}.${name} does not offer ${value}; options: ${w.options.values}`);
      }
      w.value = value;
      w.callback?.(value); // lets DynamicCombo add its nested sub-inputs, exactly like the UI
      await new Promise((r) => setTimeout(r, 200));
    };
    const out = {};
    for (const mode of ["t2v", "i2v"]) {
      await window.app.loadGraphData(structuredClone(wf));
      const g = window.app.graph;
      const load = g.getNodeById(LOAD_IMAGE);
      const save = g.getNodeById(SAVE_VIDEO);
      if (load?.type !== "LoadImage" || save?.type !== "SaveVideo") throw new Error("Unexpected node ids in official workflow");
      load.mode = mode === "i2v" ? 0 : 4; // 0 = always, 4 = bypass (as in the official template)
      // Browser-playable output: MP4 container, H.264 video (verified to be offered by this version).
      await setWidget(save, "format", "mp4");
      await setWidget(save, "format.codec", "h264");
      const { output } = await window.app.graphToPrompt();
      out[mode] = output;
    }
    return out;
  },
  { wf: official, LOAD_IMAGE, SAVE_VIDEO },
);
await browser.close();

const sortKeys = (o) => Object.fromEntries(Object.keys(o).sort((a, b) => Number(a) - Number(b)).map((k) => [k, o[k]]));
writeFileSync(resolve(REPO, "workflows/comfyui/wan22_ti2v_5b_t2v.json"), JSON.stringify(sortKeys(exported.t2v), null, 2) + "\n");
writeFileSync(resolve(REPO, "workflows/comfyui/wan22_ti2v_5b_i2v.json"), JSON.stringify(sortKeys(exported.i2v), null, 2) + "\n");

// Schemas of every node class used, for offline validation in backend tests.
const classes = [...new Set(Object.values(exported.i2v).map((n) => n.class_type))].sort();
const objectInfo = {};
for (const c of classes) {
  const r = await fetch(`${comfy}/object_info/${encodeURIComponent(c)}`);
  if (!r.ok) throw new Error(`object_info for ${c}: HTTP ${r.status}`);
  Object.assign(objectInfo, await r.json());
}
const fixtureDir = resolve(REPO, "backend/tests/fixtures/comfyui");
mkdirSync(fixtureDir, { recursive: true });
writeFileSync(resolve(fixtureDir, "object_info.json"), JSON.stringify(objectInfo, null, 2) + "\n");

const stats = await (await fetch(`${comfy}/system_stats`)).json();
const provenance = {
  _comment: "Written by workflows/comfyui/tools/export_api_workflows.mjs. Records exactly what the templates were exported from.",
  official_workflow: {
    repo: "Comfy-Org/workflow_templates",
    path: "templates/video_wan2_2_5B_ti2v.json",
    sha256: createHash("sha256").update(officialRaw).digest("hex"),
    saved_with_frontend: official.extra?.frontendVersion ?? null,
  },
  exported_with: {
    comfyui_version: stats.system?.comfyui_version,
    frontend_version: stats.system?.required_frontend_version,
  },
  changes_from_official: [
    "Node 56 (LoadImage) bypassed for t2v, enabled for i2v (as the official docs describe).",
    "Node 58 (SaveVideo) format=mp4, format.codec=h264 (was auto) for browser-compatible output.",
  ],
  node_classes: classes,
};
writeFileSync(resolve(REPO, "workflows/comfyui/export_provenance.json"), JSON.stringify(provenance, null, 2) + "\n");
console.log(`Exported t2v (${Object.keys(exported.t2v).length} nodes) and i2v (${Object.keys(exported.i2v).length} nodes) with ComfyUI ${provenance.exported_with.comfyui_version}`);
