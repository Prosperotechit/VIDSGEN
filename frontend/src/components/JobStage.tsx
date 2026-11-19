import { Ban, Clock, Download, Film, Loader2, RotateCcw, XCircle } from "lucide-react";
import { Link } from "react-router-dom";
import type { ApiError } from "../api/client";
import { videoUrls } from "../api/jobs";
import type { Job } from "../api/types";
import { elapsed, modeLabel, timeAgo } from "../lib/format";
import { ErrorBanner } from "./Feedback";
import { ProgressBar } from "./ProgressBar";
import { StatusBadge } from "./StatusBadge";
import { VideoPlayer } from "./VideoPlayer";

interface Props {
  job: Job;
  onCancel?: () => void;
  cancelling?: boolean;
  actionError?: string | null;
  pollError?: ApiError | null;
  showDetailsLink?: boolean;
  onReuse?: () => void;
}

/** The "screen": progress while generating, the video once completed, and the outcome otherwise. */
export function JobStage({ job, onCancel, cancelling, actionError, pollError, showDetailsLink, onReuse }: Props) {
  const urls = videoUrls(job);
  const { width, height } = job.params;
  const active = job.status === "queued" || job.status === "processing";

  return (
    <section className="stage" aria-labelledby={`stage-${job.id}`}>
      <header className="stage__head">
        <div className="stage__title">
          <StatusBadge status={job.status} progress={job.progress} />
          <h2 id={`stage-${job.id}`} className="stage__mode">{modeLabel(job.mode)}</h2>
          <span className="stage__time">{timeAgo(job.created_at)}</span>
        </div>
        <div className="stage__actions">
          {active && onCancel && (
            <button type="button" className="btn btn--ghost btn--sm" onClick={onCancel} disabled={cancelling}>
              {cancelling ? <Loader2 size={14} className="spin" aria-hidden="true" /> : <Ban size={14} aria-hidden="true" />}
              Cancel
            </button>
          )}
          {onReuse && !active && (
            <button type="button" className="btn btn--ghost btn--sm" onClick={onReuse}>
              <RotateCcw size={14} aria-hidden="true" /> Reuse settings
            </button>
          )}
          {urls && (
            <a className="btn btn--primary btn--sm" href={urls.download} download={`vidsgen-${job.id}.mp4`}>
              <Download size={14} aria-hidden="true" /> Download
            </a>
          )}
        </div>
      </header>

      {pollError && active && (
        <ErrorBanner title="Connection problem">{pollError.message} Retrying automatically…</ErrorBanner>
      )}
      {actionError && <ErrorBanner>{actionError}</ErrorBanner>}

      {job.status === "completed" && urls ? (
        <VideoPlayer src={urls.preview} width={width} height={height} label={`Generated video: ${job.params.prompt}`} />
      ) : (
        <div className={`screen screen--${job.status}`} style={{ aspectRatio: `${width} / ${height}` }}>
          <ScreenContent job={job} />
        </div>
      )}

      {active && <ProgressBar value={job.progress} status={job.status} />}

      <p className="stage__prompt">{job.params.prompt}</p>
      <div className="stage__foot">
        <span>{width}×{height} · {job.params.num_frames} frames · {job.params.duration_seconds}s @ {job.params.fps} fps</span>
        {job.status === "completed" && elapsed(job.started_at, job.completed_at) && (
          <span>Generated in {elapsed(job.started_at, job.completed_at)}</span>
        )}
        {showDetailsLink && <Link to={`/generations/${job.id}`} className="link">Details</Link>}
      </div>
    </section>
  );
}

function ScreenContent({ job }: { job: Job }) {
  switch (job.status) {
    case "queued":
      return (
        <div className="screen__msg" aria-live="polite">
          <Clock size={26} aria-hidden="true" />
          <strong>Waiting in queue</strong>
          <span>Your generation will start as soon as the GPU is free.</span>
        </div>
      );
    case "processing":
      return (
        <div className="screen__msg" aria-live="polite">
          <span className="screen__pct">{Math.round(job.progress)}<small>%</small></span>
          <span>Generating your video…</span>
        </div>
      );
    case "failed":
      return (
        <div className="screen__msg screen__msg--error">
          <XCircle size={26} aria-hidden="true" />
          <strong>Generation failed</strong>
          <span>{job.error?.message ?? "Unknown error"}</span>
          {job.error?.code && <code className="mono">{job.error.code}</code>}
        </div>
      );
    case "cancelled":
      return (
        <div className="screen__msg">
          <Ban size={26} aria-hidden="true" />
          <strong>Cancelled</strong>
          <span>Stopped at {Math.round(job.progress)}%. No video was produced.</span>
        </div>
      );
    default:
      return (
        <div className="screen__msg">
          <Film size={26} aria-hidden="true" />
          <span>Video unavailable</span>
        </div>
      );
  }
}
