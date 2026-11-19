import { Ban, CheckCircle2, Clock, Loader2, XCircle } from "lucide-react";
import type { JobStatus } from "../api/types";

const META: Record<JobStatus, { label: string; Icon: typeof Clock }> = {
  queued: { label: "Queued", Icon: Clock },
  processing: { label: "Generating", Icon: Loader2 },
  completed: { label: "Completed", Icon: CheckCircle2 },
  failed: { label: "Failed", Icon: XCircle },
  cancelled: { label: "Cancelled", Icon: Ban },
};

export const statusLabel = (s: JobStatus) => META[s].label;

export function StatusBadge({ status, progress }: { status: JobStatus; progress?: number }) {
  const { label, Icon } = META[status];
  return (
    <span className={`badge badge--${status}`}>
      <Icon size={13} aria-hidden="true" className={status === "processing" ? "spin" : undefined} />
      {label}
      {status === "processing" && progress !== undefined && <span className="badge__pct">{Math.round(progress)}%</span>}
    </span>
  );
}
