import { ImageIcon, Loader2, Sparkles, Type, Wand2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { ApiError } from "../api/client";
import { constraints, durationSeconds, frameOptions } from "../api/constraints";
import { cancelJob, createJob } from "../api/jobs";
import type { GenerationMode, Job, JobParams } from "../api/types";
import { EmptyState, ErrorBanner } from "../components/Feedback";
import { Field, SliderField } from "../components/Field";
import { ImageDropzone } from "../components/ImageDropzone";
import { JobStage } from "../components/JobStage";
import { ResolutionPicker } from "../components/ResolutionPicker";
import { SeedField, seedError, type SeedState } from "../components/SeedField";
import { Segmented } from "../components/Segmented";
import { useJobPolling } from "../hooks/useJobPolling";

type Errors = Partial<Record<"prompt" | "negative_prompt" | "image" | "seed" | "form", string>>;

export interface ReuseState {
  reuse: { mode: GenerationMode; params: JobParams };
}

const PROMPT_IDEAS = [
  "A slow dolly shot through a rain-soaked neon alley at night, reflections rippling in puddles",
  "Golden hour over rolling sand dunes, wind lifting fine sand, wide cinematic shot",
  "Macro shot of ink drops blooming in clear water, soft studio light",
];

export function CreatePage() {
  const navigate = useNavigate();
  const location = useLocation();
  const [search, setSearch] = useSearchParams();
  const stageRef = useRef<HTMLDivElement>(null);

  // ---------------------------------------------------------------- form state
  const reuse = (location.state as ReuseState | null)?.reuse;
  const [mode, setMode] = useState<GenerationMode>(reuse?.mode ?? "t2v");
  const [prompt, setPrompt] = useState(reuse?.params.prompt ?? "");
  const [negative, setNegative] = useState(reuse?.params.negative_prompt ?? "");
  const [showNegative, setShowNegative] = useState(!!reuse?.params.negative_prompt);
  const [image, setImage] = useState<File | null>(null);
  const [resolution, setResolution] = useState(reuse?.params.resolution ?? constraints.defaultResolution);
  const [frames, setFrames] = useState(reuse?.params.num_frames ?? constraints.numFrames.default);
  const [fps, setFps] = useState(reuse?.params.fps ?? constraints.fps.default);
  const [steps, setSteps] = useState(reuse?.params.steps ?? constraints.steps.default);
  const [guidance, setGuidance] = useState(reuse?.params.guidance_scale ?? constraints.guidanceScale.default);
  const [seed, setSeed] = useState<SeedState>(
    reuse ? { random: false, value: String(reuse.params.seed) } : { random: true, value: "" },
  );
  const [errors, setErrors] = useState<Errors>({});
  const [submitting, setSubmitting] = useState(false);

  // Clear router state after reading it, so a refresh doesn't re-apply it.
  useEffect(() => {
    if (reuse) navigate(location.pathname + location.search, { replace: true, state: null });
    // Run once on mount only.
  }, []);

  // ---------------------------------------------------------------- active job (?job=<id>)
  const activeId = search.get("job");
  const [created, setCreated] = useState<Job | null>(null);
  const { job, error: pollError, notFound, update } = useJobPolling(activeId, created);
  const [cancelling, setCancelling] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const promptLen = prompt.length;
  const selectedRes = constraints.resolutions.find((r) => r.value === resolution)!;
  const summary = useMemo(
    () => `${selectedRes.width}×${selectedRes.height} · ${durationSeconds(frames, fps)}s · ${frames} frames`,
    [selectedRes, frames, fps],
  );

  function validate(): Errors {
    const e: Errors = {};
    if (!prompt.trim()) e.prompt = "Describe the video you want to create.";
    else if (promptLen > constraints.prompt.maxChars) e.prompt = `Keep the prompt under ${constraints.prompt.maxChars} characters.`;
    if (negative.length > constraints.prompt.maxChars) e.negative_prompt = `Keep it under ${constraints.prompt.maxChars} characters.`;
    if (mode === "i2v" && !image) e.image = "Add a start image for image-to-video.";
    const se = seedError(seed);
    if (se) e.seed = se;
    return e;
  }

  async function onSubmit(ev?: FormEvent) {
    ev?.preventDefault();
    const e = validate();
    setErrors(e);
    if (Object.keys(e).length) return;

    setSubmitting(true);
    try {
      const newJob = await createJob({
        mode,
        prompt: prompt.trim(),
        negative_prompt: showNegative ? negative.trim() : "",
        resolution,
        num_frames: frames,
        fps,
        steps,
        guidance_scale: guidance,
        seed: seed.random ? null : Number(seed.value),
        image: mode === "i2v" ? image : null,
      });
      setCreated(newJob);
      setActionError(null);
      setSearch({ job: newJob.id });
      // On narrow screens the stage is below the form: bring it into view.
      if (window.matchMedia("(max-width: 1099px)").matches) {
        requestAnimationFrame(() => stageRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
      }
    } catch (err) {
      setErrors(mapApiError(err));
    } finally {
      setSubmitting(false);
    }
  }

  async function onCancel() {
    if (!job) return;
    setCancelling(true);
    setActionError(null);
    try {
      update(await cancelJob(job.id));
    } catch (err) {
      setActionError(err instanceof ApiError ? err.message : "Couldn't cancel the job.");
    } finally {
      setCancelling(false);
    }
  }

  return (
    <div className="create">
      <form className="panel create__form" onSubmit={onSubmit} noValidate aria-labelledby="create-title">
        <div className="panel__head">
          <h1 id="create-title" className="page-title">Create video</h1>
          <p className="page-sub">Wan 2.2 TI2V-5B · text or image to video</p>
        </div>

        <Segmented<GenerationMode>
          label="Generation mode"
          value={mode}
          onChange={(m) => { setMode(m); setErrors((e) => ({ ...e, image: undefined })); }}
          options={[
            { value: "t2v", label: "Text to Video", icon: Type },
            { value: "i2v", label: "Image to Video", icon: ImageIcon },
          ]}
        />

        {mode === "i2v" && (
          <Field id="image" label="Start image" error={null}
            hint={reuse?.mode === "i2v" && !image ? "Start images aren't stored for reuse. Please upload it again." : undefined}>
            <ImageDropzone file={image} onChange={(f) => { setImage(f); setErrors((e) => ({ ...e, image: undefined })); }} error={errors.image} />
          </Field>
        )}

        <Field
          id="prompt"
          label="Prompt"
          error={errors.prompt}
          hint={mode === "i2v" ? "Describe how the image should move and change." : "Describe the scene, subject, motion and camera."}
          aside={<span className={promptLen > constraints.prompt.maxChars ? "count count--over" : "count"}>{promptLen}/{constraints.prompt.maxChars}</span>}
        >
          <textarea
            id="prompt"
            className="input textarea"
            rows={5}
            placeholder={PROMPT_IDEAS[0]}
            value={prompt}
            aria-invalid={!!errors.prompt}
            aria-describedby={errors.prompt ? "prompt-error" : "prompt-hint"}
            onChange={(e) => { setPrompt(e.target.value); if (errors.prompt) setErrors((x) => ({ ...x, prompt: undefined })); }}
            onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void onSubmit(); }}
          />
        </Field>
        {!prompt && (
          <div className="ideas" aria-label="Prompt ideas">
            {PROMPT_IDEAS.slice(1).map((idea) => (
              <button key={idea} type="button" className="chip" onClick={() => setPrompt(idea)}>
                <Sparkles size={12} aria-hidden="true" /> {idea.split(",")[0]}
              </button>
            ))}
          </div>
        )}

        {showNegative ? (
          <Field id="negative" label="Negative prompt" error={errors.negative_prompt} hint="What to avoid, e.g. blurry, distorted, text, watermark."
            aside={<button type="button" className="link" onClick={() => { setShowNegative(false); setNegative(""); }}>Remove</button>}>
            <textarea id="negative" className="input textarea" rows={2} value={negative} onChange={(e) => setNegative(e.target.value)} />
          </Field>
        ) : (
          <button type="button" className="link link--add" onClick={() => setShowNegative(true)}>+ Add negative prompt</button>
        )}

        <div className="section">
          <h2 className="section__title">Format</h2>
          <Field id="resolution" label="Resolution" hint="1280×704 and 704×1280 are the model's native 720p sizes.">
            <ResolutionPicker value={resolution} onChange={setResolution} />
          </Field>
          <div className="grid-2">
            <Field id="frames" label="Length" hint={`${frames} frames`}>
              <select id="frames" className="input select" value={frames} onChange={(e) => setFrames(Number(e.target.value))}>
                {frameOptions.map((n) => (
                  <option key={n} value={n}>{durationSeconds(n, fps)} s · {n} frames</option>
                ))}
              </select>
            </Field>
            <SliderField id="fps" label="Frame rate" value={fps} min={constraints.fps.min} max={constraints.fps.max} step={1}
              onChange={setFps} format={(v) => `${v} fps`} hint="The model is trained at 24 fps." />
          </div>
        </div>

        <details className="section section--collapsible">
          <summary className="section__title">
            Advanced <span className="section__summary">{steps} steps · guidance {guidance} · {seed.random ? "random seed" : `seed ${seed.value || "-"}`}</span>
          </summary>
          <div className="section__body">
            <SliderField id="steps" label="Sampling steps" value={steps} min={constraints.steps.min} max={constraints.steps.max} step={1}
              onChange={setSteps} hint="More steps can add detail but take longer." />
            <SliderField id="guidance" label="Guidance scale" value={guidance} min={constraints.guidanceScale.min}
              max={constraints.guidanceScale.max} step={0.5} onChange={setGuidance} format={(v) => v.toFixed(1)}
              hint="How strictly to follow the prompt. Around 5 is a good balance." />
            <SeedField seed={seed} onChange={(s) => { setSeed(s); setErrors((x) => ({ ...x, seed: undefined })); }} error={errors.seed} />
          </div>
        </details>

        {errors.form && <ErrorBanner title="Couldn't start generation">{errors.form}</ErrorBanner>}

        <div className="create__submit">
          <span className="create__summary">{summary}</span>
          <button type="submit" className="btn btn--primary btn--lg" disabled={submitting}>
            {submitting ? <Loader2 size={18} className="spin" aria-hidden="true" /> : <Wand2 size={18} aria-hidden="true" />}
            {submitting ? "Submitting…" : "Generate"}
          </button>
        </div>
      </form>

      <div className="create__stage" ref={stageRef}>
        {activeId && notFound ? (
          <div className="panel">
            <EmptyState icon={Wand2} title="Generation not found" action={
              <button type="button" className="btn btn--ghost" onClick={() => setSearch({})}>Start a new one</button>}>
              It may have been removed. Create a new video instead.
            </EmptyState>
          </div>
        ) : activeId && job ? (
          <div className="panel">
            <JobStage job={job} onCancel={onCancel} cancelling={cancelling} actionError={actionError}
              pollError={pollError} showDetailsLink />
          </div>
        ) : activeId ? (
          <div className="panel panel--loading" aria-busy="true"><Loader2 className="spin" aria-hidden="true" /> Loading generation…</div>
        ) : (
          <div className="panel stage-empty">
            <EmptyState icon={Wand2} title="Your video will appear here">
              Write a prompt{mode === "i2v" ? ", add a start image" : ""} and press <strong>Generate</strong>. Progress is
              shown live, and you can preview and download the result here or in <em>My Generations</em>.
            </EmptyState>
          </div>
        )}
      </div>
    </div>
  );
}

/** Maps API errors to form fields. */
function mapApiError(err: unknown): Errors {
  if (!(err instanceof ApiError)) return { form: "Something went wrong. Please try again." };
  if (err.code === "validation_error" && err.fieldErrors.length) {
    const out: Errors = {};
    for (const fe of err.fieldErrors) {
      const key = (["prompt", "negative_prompt", "image", "seed"] as const).find((k) => k === fe.field);
      if (key) out[key] = fe.message;
      else out.form = `${fe.field}: ${fe.message}`;
    }
    return out;
  }
  if (err.code === "invalid_image" || err.code === "request_too_large") return { image: err.message };
  return { form: err.message };
}
