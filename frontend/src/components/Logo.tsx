/** Original VIDSGEN mark: two offset frames (a sequence) with a forward wedge (motion). */
export function LogoMark({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true">
      <rect x="2.5" y="6.5" width="19" height="19" rx="5" stroke="currentColor" strokeOpacity="0.35" />
      <rect x="7.5" y="3.5" width="21" height="21" rx="5.5" stroke="currentColor" strokeOpacity="0.9" />
      <path d="M15 9.6v9.8a.8.8 0 0 0 1.22.68l7.6-4.9a.8.8 0 0 0 0-1.36l-7.6-4.9A.8.8 0 0 0 15 9.6Z" fill="var(--accent)" />
    </svg>
  );
}

export function Logo() {
  return (
    <span className="logo">
      <LogoMark />
      <span className="logo__word">VIDSGEN</span>
    </span>
  );
}
