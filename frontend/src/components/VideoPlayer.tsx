import { RefreshCw, VideoOff } from "lucide-react";
import { useState } from "react";
import { Spinner } from "./Feedback";

/** HTML5 player with loading and "video unavailable" states. */
export function VideoPlayer({ src, width, height, label }: { src: string; width: number; height: number; label: string }) {
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const [attempt, setAttempt] = useState(0);

  return (
    <div className="player" style={{ aspectRatio: `${width} / ${height}` }}>
      {state !== "error" && (
        <video
          key={attempt}
          src={src}
          controls
          playsInline
          loop
          preload="metadata"
          aria-label={label}
          onLoadedData={() => setState("ready")}
          onError={() => setState("error")}
        />
      )}
      {state === "loading" && (
        <div className="player__overlay"><Spinner label="Loading video" /></div>
      )}
      {state === "error" && (
        <div className="player__overlay player__overlay--error">
          <VideoOff size={24} aria-hidden="true" />
          <p>This video couldn't be loaded. It may have been removed from storage.</p>
          <button type="button" className="btn btn--ghost btn--sm" onClick={() => { setState("loading"); setAttempt((a) => a + 1); }}>
            <RefreshCw size={14} aria-hidden="true" /> Try again
          </button>
        </div>
      )}
    </div>
  );
}
