import { ArrowLeft, Loader2, SearchX } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ApiError } from "../api/client";
import { cancelJob } from "../api/jobs";
import { EmptyState, ErrorBanner } from "../components/Feedback";
import { JobParams } from "../components/JobParams";
import { JobStage } from "../components/JobStage";
import { useJobPolling } from "../hooks/useJobPolling";
import type { ReuseState } from "./CreatePage";

export function JobPage() {
  const { jobId = "" } = useParams();
  const navigate = useNavigate();
  const { job, error, notFound, update } = useJobPolling(jobId);
  const [cancelling, setCancelling] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  async function onCancel() {
    setCancelling(true);
    setActionError(null);
    try {
      update(await cancelJob(jobId));
    } catch (e) {
      setActionError(e instanceof ApiError ? e.message : "Couldn't cancel the job.");
    } finally {
      setCancelling(false);
    }
  }

  const back = (
    <Link to="/generations" className="back-link"><ArrowLeft size={16} aria-hidden="true" /> My generations</Link>
  );

  if (notFound) {
    return (
      <div className="page">
        {back}
        <div className="panel">
          <EmptyState icon={SearchX} title="Generation not found" action={<Link to="/generations" className="btn btn--ghost">Back to library</Link>}>
            This generation doesn't exist or has been removed.
          </EmptyState>
        </div>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="page">
        {back}
        {error ? (
          <ErrorBanner title="Couldn't load this generation">{error.message}</ErrorBanner>
        ) : (
          <div className="panel panel--loading" aria-busy="true"><Loader2 className="spin" aria-hidden="true" /> Loading…</div>
        )}
      </div>
    );
  }

  const reuse = () => navigate("/", { state: { reuse: { mode: job.mode, params: job.params } } satisfies ReuseState });

  return (
    <div className="page">
      {back}
      <div className="detail">
        <div className="panel detail__main">
          <JobStage job={job} onCancel={onCancel} cancelling={cancelling} actionError={actionError} pollError={error} onReuse={reuse} />
        </div>
        <aside className="panel detail__side" aria-label="Generation details">
          <h2 className="section__title">Details</h2>
          <JobParams job={job} />
        </aside>
      </div>
    </div>
  );
}
