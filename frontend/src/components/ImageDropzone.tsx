import { ImagePlus, Loader2, X } from "lucide-react";
import { useEffect, useId, useRef, useState, type DragEvent } from "react";
import { checkImageFile, constraints, formatBytes } from "../api/constraints";

interface Props {
  file: File | null;
  onChange: (file: File | null) => void;
  error?: string | null;
}

/** Drag-and-drop / click-to-browse image picker with preview, remove and validation. */
export function ImageDropzone({ file, onChange, error: externalError }: Props) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      setDims(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  async function accept(f: File | undefined) {
    if (!f) return;
    setChecking(true);
    const problem = await checkImageFile(f);
    setChecking(false);
    setLocalError(problem);
    if (!problem) onChange(f);
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    void accept(e.dataTransfer.files[0]);
  };

  const error = localError ?? externalError;
  const { minSide, maxSide, maxBytes } = constraints.image;

  if (file && preview) {
    return (
      <div className="dropzone dropzone--filled">
        <img
          src={preview}
          alt="Selected start image"
          className="dropzone__img"
          onLoad={(e) => setDims({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
        />
        <div className="dropzone__meta">
          <span className="dropzone__name" title={file.name}>{file.name}</span>
          <span className="dropzone__sub">
            {dims ? `${dims.w}×${dims.h} · ` : ""}{formatBytes(file.size)}
          </span>
        </div>
        <div className="dropzone__actions">
          <button type="button" className="btn btn--ghost btn--sm" onClick={() => inputRef.current?.click()}>
            Replace
          </button>
          <button type="button" className="icon-btn" aria-label="Remove image" onClick={() => { onChange(null); setLocalError(null); }}>
            <X size={16} />
          </button>
        </div>
        <input ref={inputRef} type="file" accept={constraints.image.mimeTypes.join(",")} hidden
          onChange={(e) => { void accept(e.target.files?.[0]); e.target.value = ""; }} />
      </div>
    );
  }

  return (
    <div>
      <label
        htmlFor={inputId}
        className={`dropzone${dragging ? " is-dragging" : ""}${error ? " has-error" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        {checking ? <Loader2 size={22} className="spin" aria-hidden="true" /> : <ImagePlus size={22} aria-hidden="true" />}
        <span className="dropzone__title">Drop a start image or <u>browse</u></span>
        <span className="dropzone__sub">
          JPEG, PNG or WEBP · up to {formatBytes(maxBytes)} · {minSide}–{maxSide}px
        </span>
        <input
          id={inputId}
          ref={inputRef}
          type="file"
          accept={constraints.image.mimeTypes.join(",")}
          className="visually-hidden"
          aria-describedby={error ? `${inputId}-err` : undefined}
          onChange={(e) => { void accept(e.target.files?.[0]); e.target.value = ""; }}
        />
      </label>
      {error && <p id={`${inputId}-err`} className="field__error" role="alert">{error}</p>}
    </div>
  );
}
