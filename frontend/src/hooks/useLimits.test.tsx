// @vitest-environment jsdom
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/client";
import { createLimitsStore } from "../api/limits";
import type { Limits } from "../api/types";
import { DEFAULT_LIMITS, T4_LIMITS } from "../api/limits.fixtures";
import { useLimits } from "./useLimits";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function makeStore() {
  const calls: { resolve: (l: Limits) => void; reject: (e: unknown) => void }[] = [];
  const fetcher = vi.fn(() => new Promise<Limits>((resolve, reject) => { calls.push({ resolve, reject }); }));
  return { fetcher, calls, store: createLimitsStore({ fetcher }) };
}

let root: Root | null = null;
let host: HTMLElement | null = null;
beforeEach(() => { host = document.createElement("div"); document.body.appendChild(host); });
afterEach(() => { act(() => root?.unmount()); root = null; host?.remove(); });

function Probe({ store, label = "probe" }: { store: ReturnType<typeof makeStore>["store"]; label?: string }) {
  const { state, canGenerate, limits, retry } = useLimits(store);
  return createElement("div", { "data-testid": label },
    createElement("span", { className: "kind" }, state.kind),
    createElement("span", { className: "can" }, String(canGenerate)),
    createElement("span", { className: "profile" }, limits?.profile ?? "-"),
    createElement("span", { className: "err" }, state.kind === "unavailable" ? state.error.title : "-"),
    createElement("button", { className: "retry", onClick: retry }, "retry"));
}

const text = (sel: string, label = "probe") => host!.querySelector(`[data-testid=${label}] ${sel}`)?.textContent;
const render = (el: ReturnType<typeof createElement>) => { root = createRoot(host!); act(() => root!.render(el)); };
const settle = async () => { await act(async () => { for (let i = 0; i < 5; i++) await Promise.resolve(); }); };

describe("useLimits", () => {
  it("starts loading with generation disabled, then exposes the active limits", async () => {
    const { store, calls } = makeStore();
    render(createElement(Probe, { store }));
    expect(text(".kind")).toBe("loading");
    expect(text(".can")).toBe("false");
    expect(text(".profile")).toBe("-");
    act(() => calls[0]!.resolve(T4_LIMITS));
    await settle();
    expect(text(".kind")).toBe("ready");
    expect(text(".can")).toBe("true");
    expect(text(".profile")).toBe("t4_verified");
  });

  it("reports unavailable with a friendly title, keeps generation off, and recovers via retry", async () => {
    const { store, calls, fetcher } = makeStore();
    render(createElement(Probe, { store }));
    act(() => calls[0]!.reject(new ApiError("x", { status: 0, code: "network_error" })));
    await settle();
    expect(text(".kind")).toBe("unavailable");
    expect(text(".can")).toBe("false");
    expect(text(".err")).toBe("Can't reach the VIDSGEN server");
    expect(text(".profile")).toBe("-");
    act(() => host!.querySelector<HTMLButtonElement>(".retry")!.click());
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(text(".kind")).toBe("loading");
    act(() => calls[1]!.resolve(DEFAULT_LIMITS));
    await settle();
    expect(text(".kind")).toBe("ready");
    expect(text(".can")).toBe("true");
  });

  it("never offers limits for a response that fails verification", async () => {
    const { store, calls } = makeStore();
    render(createElement(Probe, { store }));
    const bad = JSON.parse(JSON.stringify(T4_LIMITS));
    bad.modes.forEach((m: { enabled: boolean }) => (m.enabled = false));
    act(() => calls[0]!.resolve(bad));
    await settle();
    expect(text(".kind")).toBe("unavailable");
    expect(text(".can")).toBe("false");
    expect(text(".profile")).toBe("-");
  });

  it("several components share one request and one answer", async () => {
    const { store, calls, fetcher } = makeStore();
    render(createElement("div", null, createElement(Probe, { store, label: "a" }), createElement(Probe, { store, label: "b" })));
    expect(fetcher).toHaveBeenCalledTimes(1);
    act(() => calls[0]!.resolve(DEFAULT_LIMITS));
    await settle();
    expect(text(".profile", "a")).toBe("unrestricted");
    expect(text(".profile", "b")).toBe("unrestricted");
  });

  it("uses the cached answer when a component mounts later (no new request)", async () => {
    const { store, calls, fetcher } = makeStore();
    render(createElement(Probe, { store, label: "a" }));
    act(() => calls[0]!.resolve(DEFAULT_LIMITS));
    await settle();
    const lateHost = document.createElement("div");
    document.body.appendChild(lateHost);
    const lateRoot = createRoot(lateHost);
    act(() => lateRoot.render(createElement(Probe, { store, label: "late" })));
    expect(fetcher).toHaveBeenCalledTimes(1);                                    // served from the cache
    expect(lateHost.querySelector("[data-testid=late] .kind")?.textContent).toBe("ready");
    act(() => lateRoot.unmount());
    lateHost.remove();
  });

  it("unmounting the last component cancels the request in flight", async () => {
    const { store, fetcher } = makeStore();
    render(createElement(Probe, { store }));
    expect(fetcher).toHaveBeenCalledTimes(1);
    const signal = (fetcher.mock.calls[0] as unknown as [AbortSignal])[0];
    act(() => root!.unmount());
    root = null;
    expect(signal.aborted).toBe(true);
  });

  it("exposes limits only while ready", async () => {
    const { store, calls } = makeStore();
    render(createElement(Probe, { store }));
    expect(text(".profile")).toBe("-");
    act(() => calls[0]!.resolve(T4_LIMITS));
    await settle();
    expect(text(".profile")).toBe("t4_verified");
    act(() => store.refresh(true));
    act(() => calls[1]!.reject(new ApiError("x", { status: 503, code: "http_503" })));
    await settle();
    expect(text(".profile")).toBe("-");                                          // a failed re-check removes the stale limits
    expect(text(".can")).toBe("false");
  });
});

