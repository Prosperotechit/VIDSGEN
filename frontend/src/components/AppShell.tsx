import { Clapperboard, LayoutGrid, Settings } from "lucide-react";
import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { useHealth, type HealthState } from "../hooks/useHealth";
import { Logo, LogoMark } from "./Logo";

const NAV = [
  { to: "/", label: "Create Video", short: "Create", icon: Clapperboard, end: true },
  { to: "/generations", label: "My Generations", short: "Library", icon: LayoutGrid, end: false },
  { to: "/settings", label: "Settings", short: "Settings", icon: Settings, end: false },
];

function healthText(h: HealthState): { tone: string; text: string } {
  switch (h.kind) {
    case "loading": return { tone: "neutral", text: "Connecting…" };
    case "ok": return { tone: "ok", text: h.health.provider.name === "mock" ? "Online · mock GPU" : "Online" };
    case "degraded": return { tone: "warn", text: "GPU server not ready · see Settings" };
    case "offline": return { tone: "danger", text: "API unreachable" };
  }
}

export function AppShell({ children }: { children: ReactNode }) {
  const health = useHealth();
  const h = healthText(health);

  return (
    <div className="shell">
      <a href="#main" className="skip-link">Skip to content</a>

      <aside className="sidebar" aria-label="Primary">
        <div className="sidebar__brand"><Logo /></div>
        <nav className="sidebar__nav">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end} className={({ isActive }) => `nav-item${isActive ? " is-active" : ""}`}>
              <Icon size={18} aria-hidden="true" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar__foot">
          <div className={`health health--${h.tone}`} role="status" aria-live="polite">
            <span className="health__dot" aria-hidden="true" />
            {h.text}
          </div>
          <div className="profile">
            <span className="profile__avatar" aria-hidden="true">G</span>
            <span className="profile__text">
              <span className="profile__name">Guest</span>
              <span className="profile__sub">Local workspace</span>
            </span>
          </div>
        </div>
      </aside>

      <header className="topbar">
        <span className="logo"><LogoMark size={24} /><span className="logo__word">VIDSGEN</span></span>
        <span className={`health health--${h.tone} health--compact`} role="status" title={h.text}>
          <span className="health__dot" aria-hidden="true" />
          <span className="visually-hidden">{h.text}</span>
        </span>
      </header>

      <main id="main" className="main">{children}</main>

      <nav className="tabbar" aria-label="Primary">
        {NAV.map(({ to, short, icon: Icon, end }) => (
          <NavLink key={to} to={to} end={end} className={({ isActive }) => `tab${isActive ? " is-active" : ""}`}>
            <Icon size={20} aria-hidden="true" />
            <span>{short}</span>
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
