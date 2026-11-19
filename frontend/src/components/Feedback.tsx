import { AlertTriangle, type LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

export function ErrorBanner({ title, children, action }: { title?: string; children: ReactNode; action?: ReactNode }) {
  return (
    <div className="banner banner--error" role="alert">
      <AlertTriangle size={16} aria-hidden="true" />
      <div className="banner__body">
        {title && <strong>{title}</strong>}
        <span>{children}</span>
      </div>
      {action}
    </div>
  );
}

export function EmptyState({ icon: Icon, title, children, action }: {
  icon: LucideIcon; title: string; children?: ReactNode; action?: ReactNode;
}) {
  return (
    <div className="empty">
      <div className="empty__icon"><Icon size={22} aria-hidden="true" /></div>
      <h3>{title}</h3>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <span className="spinner" role="status">
      <span className="visually-hidden">{label}</span>
    </span>
  );
}
