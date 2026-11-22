// Keeps credentials and internal infrastructure details out of anything shown to a person.
//
// Server messages are meant for people, but a message can still carry a URL, an address, a file path, a token or a setting name from
// somewhere down the stack. Everything that reaches the UI passes through here first. The rules err on the side of hiding.

const RULES: ReadonlyArray<readonly [RegExp, string]> = [
  [/\bBearer\s+[A-Za-z0-9._~+/=-]{6,}/gi, "Bearer [hidden]"],
  [/\b(?:token|secret|password|passwd|api[_-]?key|authorization)\s*[=:]\s*\S+/gi, "[credential hidden]"],
  [/\b[a-z][a-z0-9+.-]*:\/\/[^\s"'<>)\]]+/gi, "[address hidden]"],                      // URLs of any scheme
  [/\b[A-Za-z]:\\[^\s"'<>]+/g, "[path hidden]"],                                          // Windows paths
  [/(^|[\s(=:"'])((?:\/[\w.@+~-]+){2,}\/?)/g, "$1[path hidden]"],                         // /unix/style/paths (two or more segments)
  [/\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b/g, "[address hidden]"],                      // IPv4 (with optional port)
  [/\blocalhost(?::\d{1,5})?\b/gi, "[address hidden]"],
  [/\b(?:[a-z0-9-]+\.)+(?:com|net|org|io|dev|app|cloud|local|internal|lan|xyz|ai)(?::\d{1,5})?\b/gi, "[address hidden]"],
  [/\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*_(?:TOKEN|KEY|SECRET|PASSWORD)\b/g, "[setting hidden]"],  // setting names such as XYZ_API_TOKEN
  [/\b(?:hf|ghp|gho|sk|AKIA|xox[a-z])[_-]?[A-Za-z0-9]{12,}\b/g, "[credential hidden]"],     // well-known token prefixes
  [/\b[A-Za-z0-9_-]{32,}\b/g, "[credential hidden]"],                                     // any other long opaque string
];

export const REDACTION_MARKER = /\[(?:address|path|credential|setting) hidden\]|Bearer \[hidden\]/;

/** `text` with credentials, URLs, addresses, file paths and long opaque strings replaced by short markers. */
export function redactSensitive(text: string): string {
  let out = String(text);
  for (const [rx, replacement] of RULES) out = out.replace(rx, replacement);
  return out;
}

/** True if redaction changed anything. A message that needed redaction is not shown as written: callers use their own wording instead. */
export function wasRedacted(text: string): boolean {
  return REDACTION_MARKER.test(redactSensitive(text));
}

/** The server's own words if they are short and clean, otherwise `fallback`. */
export function safeMessage(text: unknown, fallback: string, maxChars = 240): string {
  if (typeof text !== "string") return fallback;
  const trimmed = text.trim();
  if (!trimmed || trimmed.length > maxChars || wasRedacted(trimmed)) return fallback;
  return trimmed;
}
