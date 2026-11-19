import type { JobStatus } from "../api/types";

interface Props {
  value: number;
  status: JobStatus;
  label?: string;
}

/** Accessible progress bar. Queued shows an indeterminate shimmer. */
export function ProgressBar({ value, status, label = "Generation progress" }: Props) {
  const indeterminate = status === "queued";
  const pct = Math.max(0, Math.min(100, value));
  return (
    <div
      className={`progress progress--${status}${indeterminate ? " progress--indeterminate" : ""}`}
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={indeterminate ? undefined : Math.round(pct)}
      aria-valuetext={indeterminate ? "Waiting in queue" : `${Math.round(pct)}%`}
    >
      <div className="progress__fill" style={indeterminate ? undefined : { width: `${pct}%` }} />
    </div>
  );
}
