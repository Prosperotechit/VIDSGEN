import { Cpu, HardDrive, Server, UserRound } from "lucide-react";
import { API_BASE_URL } from "../api/client";
import { useHealth } from "../hooks/useHealth";

export function SettingsPage() {
  const health = useHealth(10_000);
  const h = health.kind === "ok" || health.kind === "degraded" ? health.health : null;

  return (
    <div className="page page--narrow">
      <header className="page__head">
        <div>
          <h1 className="page-title">Settings</h1>
          <p className="page-sub">Connection status and workspace preferences.</p>
        </div>
      </header>

      <section className="panel settings-card" aria-labelledby="conn">
        <h2 id="conn" className="section__title"><Server size={16} aria-hidden="true" /> API connection</h2>
        <dl className="kv">
          <div className="kv__row"><dt>API endpoint</dt><dd className="mono">{API_BASE_URL || `${window.location.origin} (same origin)`}</dd></div>
          <div className="kv__row"><dt>Status</dt><dd>
            <span className={`health health--${health.kind === "ok" ? "ok" : health.kind === "degraded" ? "warn" : health.kind === "loading" ? "neutral" : "danger"}`}>
              <span className="health__dot" aria-hidden="true" />
              {health.kind === "loading" ? "Checking…" : health.kind === "offline" ? health.message : health.kind === "ok" ? "Healthy" : "Degraded"}
            </span>
          </dd></div>
          {h && <div className="kv__row"><dt>API version</dt><dd>{h.version} ({h.environment})</dd></div>}
        </dl>
      </section>

      {h && (
        <div className="grid-2">
          <section className="panel settings-card" aria-labelledby="gpu">
            <h2 id="gpu" className="section__title"><Cpu size={16} aria-hidden="true" /> Inference provider</h2>
            <dl className="kv">
              <div className="kv__row"><dt>Provider</dt><dd>{h.provider.name}</dd></div>
              <div className="kv__row"><dt>Available</dt><dd>{h.provider.ok ? "Yes" : "No"}</dd></div>
              {h.provider.detail && <div className="kv__row"><dt>Detail</dt><dd>{h.provider.detail}</dd></div>}
            </dl>
          </section>
          <section className="panel settings-card" aria-labelledby="store">
            <h2 id="store" className="section__title"><HardDrive size={16} aria-hidden="true" /> Storage &amp; queue</h2>
            <dl className="kv">
              <div className="kv__row"><dt>Storage</dt><dd>{h.storage.name} · {h.storage.ok ? "writable" : "unavailable"}</dd></div>
              <div className="kv__row"><dt>Jobs</dt><dd>{h.jobs.queued ?? 0} queued · {h.jobs.processing ?? 0} generating · {h.jobs.completed ?? 0} completed</dd></div>
            </dl>
          </section>
        </div>
      )}

      <section className="panel settings-card settings-card--muted" aria-labelledby="acct">
        <h2 id="acct" className="section__title"><UserRound size={16} aria-hidden="true" /> Account &amp; preferences</h2>
        <p className="page-sub">Sign-in, default generation settings and storage preferences will appear here in a later release.
          For now, VIDSGEN runs as a single local workspace.</p>
      </section>
    </div>
  );
}
