import type { CSSProperties, ReactNode } from "react";

/** Label + control + hint/error, wired up for screen readers via ids. */
export function Field({ id, label, hint, error, aside, children }: {
  id: string;
  label: ReactNode;
  hint?: ReactNode;
  error?: string | null;
  aside?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className={`field${error ? " has-error" : ""}`}>
      <div className="field__head">
        <label htmlFor={id} className="field__label">{label}</label>
        {aside && <span className="field__aside">{aside}</span>}
      </div>
      {children}
      {error ? (
        <p id={`${id}-error`} className="field__error" role="alert">{error}</p>
      ) : hint ? (
        <p id={`${id}-hint`} className="field__hint">{hint}</p>
      ) : null}
    </div>
  );
}

/** Range slider with a live value readout. */
export function SliderField({ id, label, value, min, max, step, onChange, format, hint, error }: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  hint?: ReactNode;
  error?: string | null;
}) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <Field id={id} label={label} hint={hint} error={error}
      aside={<output htmlFor={id} className="field__value">{format ? format(value) : value}</output>}>
      <input
        id={id}
        type="range"
        className="slider"
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ "--fill": `${pct}%` } as CSSProperties}
        aria-describedby={error ? `${id}-error` : hint ? `${id}-hint` : undefined}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </Field>
  );
}
