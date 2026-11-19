import { Check, Copy } from "lucide-react";
import { useState, type ReactNode } from "react";
import { formatBytes } from "../api/constraints";
import type { Job } from "../api/types";
import { elapsed, formatDateTime, modeLabel } from "../lib/format";

function CopyButton({ text, label }: { text: string; label: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      className="icon-btn icon-btn--sm"
      aria-label={label}
      title={label}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setDone(true);
          setTimeout(() => setDone(false), 1500);
        } catch {
          /* clipboard unavailable (e.g. insecure context) */
        }
      }}
    >
      {done ? <Check size={14} /> : <Copy size={14} />}
    </button>
  );
}

/** Full record of how a video was made: prompt, all parameters, timings and output. */
export function JobParams({ job }: { job: Job }) {
  const p = job.params;
  const rows: [string, ReactNode][] = [
    ["Mode", modeLabel(job.mode)],
    ["Resolution", `${p.width}×${p.height}`],
    ["Frames", `${p.num_frames} (${p.duration_seconds}s at ${p.fps} fps)`],
    ["Steps", p.steps],
    ["Guidance scale", p.guidance_scale],
    ["Seed", <span className="kv__seed"><span className="mono">{p.seed}</span><CopyButton text={String(p.seed)} label="Copy seed" /></span>],
    ["Provider", job.provider],
    ["Created", formatDateTime(job.created_at)],
    ["Finished", formatDateTime(job.completed_at)],
  ];
  const took = elapsed(job.started_at, job.completed_at);
  if (took) rows.push(["Generation time", took]);
  if (job.result?.size_bytes) rows.push(["File size", formatBytes(job.result.size_bytes)]);
  if (job.input_image) {
    rows.push(["Start image", `${job.input_image.width}×${job.input_image.height} ${job.input_image.original_format.toUpperCase()}`]);
  }

  return (
    <div className="params">
      <div className="params__block">
        <div className="params__label">
          Prompt <CopyButton text={p.prompt} label="Copy prompt" />
        </div>
        <p className="params__prompt">{p.prompt}</p>
      </div>
      {p.negative_prompt && (
        <div className="params__block">
          <div className="params__label">Negative prompt</div>
          <p className="params__prompt params__prompt--muted">{p.negative_prompt}</p>
        </div>
      )}
      <dl className="kv">
        {rows.map(([k, v]) => (
          <div key={k} className="kv__row">
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
      <p className="params__id mono">{job.id}</p>
    </div>
  );
}
