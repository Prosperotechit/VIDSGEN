import { Clapperboard, LayoutGrid, Loader2, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ApiError } from "../api/client";
import { listJobs } from "../api/jobs";
import { isTerminal, type Job, type JobStatus } from "../api/types";
import { EmptyState, ErrorBanner } from "../components/Feedback";
import { JobCard } from "../components/JobCard";

const PAGE = 12;
const FILTERS: { value: JobStatus | "all"; label: string }[] = [
  { value: "all", label: "All" },
  { value: "completed", label: "Completed" },
  { value: "processing", label: "Generating" },
  { value: "queued", label: "Queued" },
  { value: "failed", label: "Failed" },
  { value: "cancelled", label: "Cancelled" },
];

export function GenerationsPage() {
  const [filter, setFilter] = useState<JobStatus | "all">("all");
  const [items, setItems] = useState<Job[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const itemsRef = useRef<Job[]>([]);
  itemsRef.current = items;

  const status = filter === "all" ? undefined : filter;

  const load = useCallback(async (signal?: AbortSignal) => {
    setError(null);
    try {
      const res = await listJobs({ limit: Math.max(PAGE, itemsRef.current.length), offset: 0, status }, signal);
      setItems(res.items);
      setTotal(res.total);
    } catch (e) {
      if (signal?.aborted) return;
      setError(e instanceof ApiError ? e.message : "Couldn't load your generations.");
    } finally {
      setLoading(false);
    }
  }, [status]);

  // Initial load and whenever the filter changes.
  useEffect(() => {
    const c = new AbortController();
    setLoading(true);
    setItems([]);
    itemsRef.current = [];
    void load(c.signal);
    return () => c.abort();
  }, [load]);

  // Refresh gently (every 4 s) only while something on screen is still running.
  const hasActive = items.some((j) => !isTerminal(j.status));
  useEffect(() => {
    if (!hasActive) return;
    const t = window.setInterval(() => { if (!document.hidden) void load(); }, 4000);
    return () => window.clearInterval(t);
  }, [hasActive, load]);

  async function loadMore() {
    setLoadingMore(true);
    try {
      const res = await listJobs({ limit: PAGE, offset: items.length, status });
      setItems((prev) => [...prev, ...res.items.filter((j) => !prev.some((p) => p.id === j.id))]);
      setTotal(res.total);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Couldn't load more.");
    } finally {
      setLoadingMore(false);
    }
  }

  return (
    <div className="page">
      <header className="page__head">
        <div>
          <h1 className="page-title">My generations</h1>
          <p className="page-sub">{loading ? "Loading…" : `${total} ${total === 1 ? "video" : "videos"}${filter === "all" ? "" : ` · ${FILTERS.find((f) => f.value === filter)!.label.toLowerCase()}`}`}</p>
        </div>
        <div className="page__actions">
          <button type="button" className="icon-btn icon-btn--bordered" aria-label="Refresh" onClick={() => void load()}>
            <RefreshCw size={16} />
          </button>
          <Link to="/" className="btn btn--primary"><Clapperboard size={16} aria-hidden="true" /> New video</Link>
        </div>
      </header>

      <div className="filters" role="tablist" aria-label="Filter by status">
        {FILTERS.map((f) => (
          <button key={f.value} type="button" role="tab" aria-selected={filter === f.value}
            className={`chip chip--filter${filter === f.value ? " is-active" : ""}`} onClick={() => setFilter(f.value)}>
            {f.label}
          </button>
        ))}
      </div>

      {error && <ErrorBanner action={<button type="button" className="btn btn--ghost btn--sm" onClick={() => void load()}>Retry</button>}>{error}</ErrorBanner>}

      {loading ? (
        <div className="grid" aria-busy="true">
          {Array.from({ length: 6 }, (_, i) => <div key={i} className="card card--skeleton" />)}
        </div>
      ) : items.length === 0 && !error ? (
        <div className="panel">
          {filter === "all" ? (
            <EmptyState icon={LayoutGrid} title="No generations yet"
              action={<Link to="/" className="btn btn--primary">Create your first video</Link>}>
              Videos you generate will be collected here so you can watch and download them anytime.
            </EmptyState>
          ) : (
            <EmptyState icon={LayoutGrid} title={`No ${FILTERS.find((f) => f.value === filter)!.label.toLowerCase()} generations`}
              action={<button type="button" className="btn btn--ghost" onClick={() => setFilter("all")}>Show all</button>} />
          )}
        </div>
      ) : (
        <>
          <div className="grid">{items.map((j) => <JobCard key={j.id} job={j} />)}</div>
          {items.length < total && (
            <div className="load-more">
              <button type="button" className="btn btn--ghost" onClick={loadMore} disabled={loadingMore}>
                {loadingMore && <Loader2 size={16} className="spin" aria-hidden="true" />} Load more
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
