import { Dices } from "lucide-react";
import { constraints, randomSeed } from "../api/constraints";
import { Field } from "./Field";

export interface SeedState {
  random: boolean;  // true: server picks a new seed for every generation
  value: string;    // used when random is false
}

export function seedError(s: SeedState): string | null {
  if (s.random) return null;
  if (!/^\d+$/.test(s.value)) return "Enter a whole number, or switch to random.";
  const n = Number(s.value);
  if (n < constraints.seed.min || n > constraints.seed.max) return `Seed must be between 0 and ${constraints.seed.max}.`;
  return null;
}

export function SeedField({ seed, onChange, error }: { seed: SeedState; onChange: (s: SeedState) => void; error?: string | null }) {
  const err = error ?? seedError(seed);
  return (
    <Field
      id="seed"
      label="Seed"
      error={err}
      hint={seed.random ? "A new random seed is used for each generation (it's saved with the result)." : "The same seed and settings reproduce the same video."}
      aside={
        <label className="switch">
          <input type="checkbox" checked={seed.random} onChange={(e) => onChange({ ...seed, random: e.target.checked })} />
          <span className="switch__track" aria-hidden="true" />
          Random
        </label>
      }
    >
      <div className="input-group">
        <input
          id="seed"
          className="input mono"
          inputMode="numeric"
          placeholder={seed.random ? "Random" : "e.g. 42"}
          disabled={seed.random}
          value={seed.random ? "" : seed.value}
          aria-invalid={!!err}
          aria-describedby={err ? "seed-error" : "seed-hint"}
          onChange={(e) => onChange({ ...seed, value: e.target.value.trim() })}
        />
        <button
          type="button"
          className="icon-btn icon-btn--bordered"
          aria-label="Randomize seed"
          title="Pick a random seed"
          onClick={() => onChange({ random: false, value: String(randomSeed()) })}
        >
          <Dices size={16} />
        </button>
      </div>
    </Field>
  );
}
