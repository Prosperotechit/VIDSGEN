import { Ban, Clock, ImageIcon, Type, XCircle } from "lucide-react";
import { Link } from "react-router-dom";
import { videoUrls } from "../api/jobs";
import type { Job } from "../api/types";
import { timeAgo } from "../lib/format";
import { ProgressBar } from "./ProgressBar";
import { StatusBadge } from "./StatusBadge";

/** History grid card. Completed jobs show a real frame of the video as their thumbnail. */
export function JobCard({ job }: { job: Job }) {
  const urls = videoUrls(job);
  const { width, height } = job.params;
  const active = job.status === "queued" || job.status === "processing";
  const ModeIcon = job.mode === "i2v" ? ImageIcon : Type;

  return (
    <Link to={`/generations/${job.id}`} className="card" aria-label={`${job.status} generation: ${job.params.prompt}`}>
      <div className="card__thumb">
        {urls ? (
          // #t=0.1 makes browsers paint an early frame as a poster without downloading the whole file.
          <video src={`${urls.preview}#t=0.1`} muted playsInline preload="metadata" tabIndex={-1} aria-hidden="true"
            style={{ aspectRatio: `${width} / ${height}` }} />
        ) : (
          <div className={`card__placeholder card__placeholder--${job.status}`}>
            {job.status === "failed" ? <XCircle size={22} /> : job.status === "cancelled" ? <Ban size={22} /> : active && job.status === "processing" ? (
              <span className="card__pct">{Math.round(job.progress)}%</span>
            ) : <Clock size={22} />}
          </div>
        )}
        <span className="card__dims">{width}×{height}</span>
      </div>
      <div className="card__body">
        <div className="card__row">
          <StatusBadge status={job.status} progress={job.progress} />
          <span className="card__time">{timeAgo(job.created_at)}</span>
        </div>
        <p className="card__prompt">{job.params.prompt}</p>
        <div className="card__meta">
          <ModeIcon size={13} aria-hidden="true" />
          <span>{job.mode === "i2v" ? "Image → Video" : "Text → Video"}</span>
          <span aria-hidden="true">·</span>
          <span>{job.params.duration_seconds}s</span>
        </div>
        {active && <ProgressBar value={job.progress} status={job.status} />}
      </div>
    </Link>
  );
}
