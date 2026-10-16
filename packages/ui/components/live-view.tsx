import { useEffect, useRef, useState } from "react";
import { argusApiUrl } from "@/lib/api";
import "./live-view.css";

interface LiveFrameMessage {
  kind: "frame";
  sequence?: number;
  data: string;
  metadata?: { deviceWidth?: number; deviceHeight?: number };
}

interface LiveActionMessage {
  kind: "action";
  action?: string;
  locator?: string;
  rationale?: string;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  viewport_width?: number;
  viewport_height?: number;
  sequence?: number;
}

interface CursorAction extends LiveActionMessage {
  key: number;
}

interface LiveViewProps {
  runId: string;
  fixtureSrc?: string;
  fixtureAlt?: string;
  demoCursorRun?: number;
}

export function LiveView({
  runId,
  fixtureSrc,
  fixtureAlt = "Recorded browser fixture frame",
  demoCursorRun = 0,
}: LiveViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const demoCursorRef = useRef<HTMLSpanElement>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const renderedSequenceRef = useRef(0);
  const [connection, setConnection] = useState("Connecting to browser stream…");
  const [cursorAction, setCursorAction] = useState<CursorAction | null>(null);
  const [frameSize, setFrameSize] = useState({ width: 1280, height: 720 });
  const [retry, setRetry] = useState(0);

  useEffect(() => {
    if (fixtureSrc) {
      setConnection("Fixture frame · no browser session");
      return;
    }

    if (!runId) {
      setConnection("Waiting for a run to start…");
      return;
    }

    let disposed = false;
    let reconnectTimer: number | undefined;
    let completed = false;
    renderedSequenceRef.current = 0;
    setCursorAction(null);
    setConnection("Connecting to browser stream…");
    const socketUrl = `${argusApiUrl.replace(/^http/, "ws")}/runs/${encodeURIComponent(runId)}/screencast`;
    const socket = new WebSocket(socketUrl);
    socketRef.current = socket;

    socket.addEventListener("open", () => {
      socket.send(JSON.stringify({
        type: "auth",
        token: import.meta.env.VITE_ARGUS_API_TOKEN ?? "",
      }));
      setConnection("Connected · waiting for browser frames");
    });

    socket.addEventListener("message", (message) => {
      let payload: unknown;
      try {
        payload = JSON.parse(String(message.data));
      } catch {
        return;
      }
      if (typeof payload !== "object" || payload === null || !("kind" in payload)) return;
      if (payload.kind === "ready") {
        setConnection("Connected · waiting for browser frames");
      } else if (payload.kind === "frame" && "data" in payload && typeof payload.data === "string") {
        drawFrame(payload as LiveFrameMessage);
        setConnection("Live browser stream · Chromium · up to 720p");
      } else if (payload.kind === "action") {
        const action = payload as LiveActionMessage;
        setCursorAction({ ...action, key: Number(action.sequence ?? Date.now()) });
      } else if (payload.kind === "closed") {
        completed = true;
        setConnection("Browser stream ended");
        socket.close();
      }
    });

    socket.addEventListener("error", () => {
      if (!disposed) setConnection("Browser stream unavailable · reconnecting…");
    });

    socket.addEventListener("close", () => {
      if (disposed || completed) return;
      setConnection("Browser stream interrupted · reconnecting…");
      reconnectTimer = window.setTimeout(() => setRetry((value) => value + 1), 1000);
    });

    function drawFrame(frame: LiveFrameMessage): void {
      const image = new Image();
      image.onload = () => {
        if (disposed) return;
        const sequence = frame.sequence ?? 0;
        if (sequence < renderedSequenceRef.current) return;
        renderedSequenceRef.current = sequence;
        const canvas = canvasRef.current;
        const context = canvas?.getContext("2d");
        if (!canvas || !context) return;
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        context.imageSmoothingEnabled = true;
        context.imageSmoothingQuality = "high";
        context.drawImage(image, 0, 0);
        setFrameSize({ width: image.naturalWidth, height: image.naturalHeight });
      };
      image.src = `data:image/jpeg;base64,${frame.data}`;
    }

    return () => {
      disposed = true;
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
      socketRef.current = null;
      socket.close();
    };
  }, [fixtureSrc, retry, runId]);

  useEffect(() => {
    const cursor = demoCursorRef.current;
    if (!cursor || demoCursorRun === 0) return;
    const animation = cursor.animate(
      [{ left: "2%", top: "2%" }, { left: "96%", top: "94%" }],
      { duration: 5000, easing: "linear", fill: "forwards" },
    );
    return () => animation.cancel();
  }, [demoCursorRun]);

  const viewportWidth = cursorAction?.viewport_width ?? frameSize.width;
  const viewportHeight = cursorAction?.viewport_height ?? frameSize.height;
  const hasPoint = typeof cursorAction?.x === "number" && typeof cursorAction.y === "number";
  const hasDetectionArea = hasPoint &&
    typeof cursorAction?.width === "number" && cursorAction.width > 0 &&
    typeof cursorAction?.height === "number" && cursorAction.height > 0;
  // CDP can capture a browser surface wider than the page viewport. In that
  // case the page is centered in the frame, so include the letterbox offset
  // when mapping Playwright viewport coordinates onto the overlay.
  const pageScale = Math.min(frameSize.width / viewportWidth, frameSize.height / viewportHeight);
  const pageOffsetX = (frameSize.width - viewportWidth * pageScale) / 2;
  const pageOffsetY = (frameSize.height - viewportHeight * pageScale) / 2;
  const frameX = (value: number) => ((pageOffsetX + value * pageScale) / frameSize.width) * 100;
  const frameY = (value: number) => ((pageOffsetY + value * pageScale) / frameSize.height) * 100;

  return (
    <section className="argus-live-view" aria-label="Live browser view">
      <header className="argus-live-header">
        <div><p className="eyebrow">BROWSER VIEW</p><strong>{fixtureSrc ? "Fixture replay" : "Live target"}</strong></div>
        <span className={`argus-live-indicator ${fixtureSrc ? "fixture" : "live"}`}><i />{connection}</span>
      </header>
      <div
        className="argus-live-stage"
        style={{ aspectRatio: `${frameSize.width} / ${frameSize.height}` }}
        aria-label={fixtureSrc ? fixtureAlt : "Chromium screencast canvas"}
      >
        {fixtureSrc ? (
          <img className="argus-live-fixture" src={fixtureSrc} alt={fixtureAlt} />
        ) : (
          <canvas ref={canvasRef} className="argus-live-canvas" aria-hidden="true" />
        )}
        {fixtureSrc && demoCursorRun > 0 && (
          <span ref={demoCursorRef} className="argus-live-cursor argus-live-demo-cursor" aria-hidden="true">
            <CursorMark />
          </span>
        )}
        {!fixtureSrc && cursorAction && hasPoint && (
          <>
            {hasDetectionArea && (
              <span
                key={`area-${cursorAction.key}`}
                className="argus-detection-area"
                style={{
                  left: `${frameX(cursorAction.x! - cursorAction.width! / 2)}%`,
                  top: `${frameY(cursorAction.y! - cursorAction.height! / 2)}%`,
                  width: `${(cursorAction.width! * pageScale / frameSize.width) * 100}%`,
                  height: `${(cursorAction.height! * pageScale / frameSize.height) * 100}%`,
                  position: "absolute",
                  zIndex: 1,
                  boxSizing: "border-box",
                  border: "2px solid #67e2b2",
                  borderRadius: 4,
                  background: "rgba(75, 215, 162, 0.12)",
                  boxShadow: "0 0 0 1px rgba(4, 12, 8, 0.8), 0 0 16px rgba(75, 215, 162, 0.32)",
                  pointerEvents: "none",
                }}
                aria-label={`Detected area: ${cursorAction.locator ?? cursorAction.action ?? "target"}`}
              >
                <span style={{
                  position: "absolute",
                  left: 0,
                  top: 0,
                  transform: "translateY(-100%)",
                  maxWidth: 260,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  padding: "3px 6px",
                  borderRadius: "4px 4px 0 0",
                  background: "#67e2b2",
                  color: "#102017",
                  font: "600 10px var(--mono)",
                }}>{cursorAction.action ?? "detected"} · {cursorAction.locator ?? "target"}</span>
              </span>
            )}
            <span
              key={cursorAction.key}
              className={`argus-live-cursor ${cursorAction.action === "click" ? "clicked" : ""}`}
              style={{ left: `${frameX(cursorAction.x!)}%`, top: `${frameY(cursorAction.y!)}%` }}
              aria-hidden="true"
            ><CursorMark />{cursorAction.action === "click" && <i className="argus-click-ripple" />}</span>
          </>
        )}
        {!fixtureSrc && <span className="argus-live-resolution">{frameSize.width} × {frameSize.height}</span>}
      </div>
      <footer className="argus-live-caption" aria-live="polite">
        <span className="argus-live-action">{demoCursorRun > 0 ? "Demo" : cursorAction?.action ?? "Waiting"}</span>
        <span>{demoCursorRun > 0 ? "Illustrative overlay path: top-left to bottom-right in 5 seconds; it does not control the target page." : cursorAction?.rationale ?? (fixtureSrc ? "Recorded fixture frame; no target site is being controlled." : "Action rationale will appear here when the agent interacts with the page.")}</span>
      </footer>
    </section>
  );
}

function CursorMark() {
  return (
    <svg className="argus-cursor-pointer" viewBox="0 0 24 24" fill="#0f1712" aria-hidden="true">
      <path
        d="M20.5056 10.7754C21.1225 10.5355 21.431 10.4155 21.5176 10.2459C21.5926 10.099 21.5903 9.92446 21.5115 9.77954C21.4205 9.61226 21.109 9.50044 20.486 9.2768L4.59629 3.5728C4.0866 3.38983 3.83175 3.29835 3.66514 3.35605C3.52029 3.40621 3.40645 3.52004 3.35629 3.6649C3.29859 3.8315 3.39008 4.08635 3.57304 4.59605L9.277 20.4858C9.50064 21.1088 9.61246 21.4203 9.77973 21.5113C9.92465 21.5901 10.0991 21.5924 10.2461 21.5174C10.4157 21.4308 10.5356 21.1223 10.7756 20.5054L13.3724 13.8278C13.4194 13.707 13.4429 13.6466 13.4792 13.5957C13.5114 13.5506 13.5508 13.5112 13.5959 13.479C13.6468 13.4427 13.7072 13.4192 13.828 13.3722L20.5056 10.7754Z"
        fill="#0f1712"
        stroke="#5eead4"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
