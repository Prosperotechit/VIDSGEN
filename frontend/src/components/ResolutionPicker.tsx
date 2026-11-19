import { aspectLabel, constraints, isNative } from "../api/constraints";

/** Card-style radio group of the backend's supported resolutions, with aspect previews. */
export function ResolutionPicker({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <div className="res-grid" role="radiogroup" aria-label="Resolution">
      {constraints.resolutions.map((r) => {
        const active = r.value === value;
        const scale = 28 / Math.max(r.width, r.height);
        return (
          <button
            key={r.value}
            type="button"
            role="radio"
            aria-checked={active}
            className={`res-card${active ? " is-active" : ""}`}
            onClick={() => onChange(r.value)}
          >
            <span className="res-card__slot" aria-hidden="true">
              <span className="res-card__shape" style={{ width: r.width * scale, height: r.height * scale }} />
            </span>
            <span className="res-card__value">{r.width}×{r.height}</span>
            <span className="res-card__label">
              {aspectLabel(r)}
              {isNative(r) && <span className="res-card__tag">720p</span>}
            </span>
          </button>
        );
      })}
    </div>
  );
}
