"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import HlsPlayer from "./HlsPlayer";

const views = ["cameras", "activity", "system"];

const emptyCamera = {
  name: "",
  location: "",
  host: "",
  port: 554,
  rtspPath: "/axis-media/media.amp",
  username: "",
  password: "",
  enabled: true,
};

async function jsonRequest(url, options = {}, token = "") {
  const response = await fetch(url, {
    ...options,
    headers: {
      ...options.headers,
      ...(options.body ? { "content-type": "application/json" } : {}),
      ...(token ? { authorization: `Bearer ${token}` } : {}),
    },
    cache: "no-store",
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const error = new Error(
      body.error || `Request failed (${response.status})`,
    );
    error.status = response.status;
    throw error;
  }
  return response.status === 204 ? null : response.json();
}

export default function CameraDashboard() {
  const [cameras, setCameras] = useState([]);
  const [streams, setStreams] = useState([]);
  const [inference, setInference] = useState({ ready: false, loaded: false });
  const [token, setToken] = useState("");
  const [user, setUser] = useState(null);
  const [selected, setSelected] = useState(null);
  const [dialog, setDialog] = useState(null);
  const [activeView, setActiveView] = useState("cameras");
  const [menuOpen, setMenuOpen] = useState(false);
  const [services, setServices] = useState({
    cameraApi: null,
    inferenceApi: null,
    lastChecked: null,
  });
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState("");

  const load = useCallback(async () => {
    const [cameraResult, streamResult, inferenceResult] =
      await Promise.allSettled([
        jsonRequest("/api/cameras"),
        jsonRequest("/api/streams"),
        jsonRequest("/api/inference/status"),
      ]);
    if (cameraResult.status === "fulfilled") setCameras(cameraResult.value);
    if (streamResult.status === "fulfilled") setStreams(streamResult.value);
    if (inferenceResult.status === "fulfilled")
      setInference(inferenceResult.value);
    setServices({
      cameraApi:
        cameraResult.status === "fulfilled" &&
        streamResult.status === "fulfilled",
      inferenceApi: inferenceResult.status === "fulfilled",
      lastChecked: new Date(),
    });
  }, []);

  useEffect(() => {
    const saved = window.localStorage.getItem("camera-session");
    if (saved) {
      try {
        const session = JSON.parse(saved);
        if (
          !session.token ||
          !session.user ||
          (session.expiresAt && new Date(session.expiresAt) <= new Date())
        ) {
          throw new Error("Expired session");
        }
        setToken(session.token);
        setUser(session.user);
        jsonRequest("/api/auth/me", {}, session.token)
          .then((currentUser) => setUser(currentUser))
          .catch((error) => {
            if (error.status === 401) {
              window.localStorage.removeItem("camera-session");
              setToken("");
              setUser(null);
            }
          });
      } catch {
        window.localStorage.removeItem("camera-session");
      }
    }

    const syncViewFromHash = () => {
      const hash = window.location.hash.slice(1);
      if (views.includes(hash)) setActiveView(hash);
    };
    syncViewFromHash();
    window.addEventListener("hashchange", syncViewFromHash);
    window.addEventListener("popstate", syncViewFromHash);
    load();
    const interval = window.setInterval(load, 5000);
    return () => {
      window.clearInterval(interval);
      window.removeEventListener("hashchange", syncViewFromHash);
      window.removeEventListener("popstate", syncViewFromHash);
    };
  }, [load]);

  useEffect(() => {
    if (!dialog && !selected && !menuOpen) return undefined;
    const handleKeyDown = (event) => {
      if (event.key !== "Escape") return;
      if (selected) setSelected(null);
      else if (dialog) setDialog(null);
      else setMenuOpen(false);
    };
    window.addEventListener("keydown", handleKeyDown);
    if (dialog || selected) document.body.classList.add("modal-open");
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.body.classList.remove("modal-open");
    };
  }, [dialog, menuOpen, selected]);

  const liveByCamera = useMemo(
    () =>
      Object.fromEntries(streams.map((stream) => [stream.cameraId, stream])),
    [streams],
  );

  async function toggleStream(camera) {
    if (!token) return openDialog("auth");
    const live = liveByCamera[camera.id];
    setBusy(camera.id);
    setMessage("");
    try {
      await jsonRequest(
        `/api/streams/${camera.id}/${live ? "stop" : "start"}`,
        { method: "POST" },
        token,
      );
      await load();
    } catch (error) {
      handleRequestError(error);
    } finally {
      setBusy("");
    }
  }

  async function authenticate(event, mode) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const payload = Object.fromEntries(form.entries());
    setBusy("auth");
    setMessage("");
    try {
      const session = await jsonRequest(`/api/auth/${mode}`, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      window.localStorage.setItem("camera-session", JSON.stringify(session));
      setToken(session.token);
      setUser(session.user);
      setDialog(null);
    } catch (error) {
      setMessage(error.message || "Unable to sign in. Please try again.");
    } finally {
      setBusy("");
    }
  }

  async function addCamera(event) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const body = {
      ...emptyCamera,
      ...Object.fromEntries(form.entries()),
      port: Number(form.get("port")),
    };
    setBusy("camera");
    setMessage("");
    try {
      await jsonRequest(
        "/api/cameras",
        { method: "POST", body: JSON.stringify(body) },
        token,
      );
      setDialog(null);
      await load();
    } catch (error) {
      handleRequestError(error);
    } finally {
      setBusy("");
    }
  }

  function signOut() {
    window.localStorage.removeItem("camera-session");
    setToken("");
    setUser(null);
    setMenuOpen(false);
    setMessage("");
  }

  function openDialog(name) {
    setMessage("");
    setMenuOpen(false);
    setDialog(name);
  }

  function closeDialog() {
    setDialog(null);
    setMessage("");
  }

  function handleRequestError(error) {
    if (error.status === 401) {
      window.localStorage.removeItem("camera-session");
      setToken("");
      setUser(null);
      setDialog("auth");
      setMessage("Your session expired. Sign in again to continue.");
      return;
    }
    setMessage(error.message || "Something went wrong. Please try again.");
  }

  function activateView(view) {
    if (!views.includes(view)) return;
    setActiveView(view);
    setMenuOpen(false);
    window.history.pushState(null, "", `#${view}`);
    window.requestAnimationFrame(() => {
      document
        .getElementById("workspace")
        ?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Sentinel home">
          <span className="brand-mark">S</span>
          <span>Sentinel</span>
        </a>
        <nav
          className="desktop-nav"
          aria-label="Dashboard views"
          role="tablist"
        >
          {views.map((view) => (
            <button
              id={`tab-${view}`}
              type="button"
              role="tab"
              aria-selected={activeView === view}
              aria-controls={`panel-${view}`}
              className={activeView === view ? "active" : ""}
              key={view}
              onClick={() => activateView(view)}
            >
              {view[0].toUpperCase() + view.slice(1)}
            </button>
          ))}
        </nav>
        <div className="account desktop-account">
          {user ? (
            <>
              <span className="user-name">{user.displayName}</span>
              <button className="ghost" onClick={signOut}>
                Sign out
              </button>
            </>
          ) : (
            <button className="ghost" onClick={() => openDialog("auth")}>
              Sign in
            </button>
          )}
          <button
            className="primary"
            onClick={() => openDialog(token ? "camera" : "auth")}
          >
            Add camera
          </button>
        </div>
        <button
          type="button"
          className={`mobile-menu-toggle ${menuOpen ? "open" : ""}`}
          aria-label={menuOpen ? "Close menu" : "Open menu"}
          aria-expanded={menuOpen}
          aria-controls="mobile-menu"
          onClick={() => setMenuOpen((open) => !open)}
        >
          <span />
          <span />
          <span />
        </button>
        {menuOpen && (
          <div className="mobile-menu" id="mobile-menu">
            <div className="mobile-menu-links" role="tablist">
              {views.map((view) => (
                <button
                  type="button"
                  role="tab"
                  aria-selected={activeView === view}
                  className={activeView === view ? "active" : ""}
                  key={view}
                  onClick={() => activateView(view)}
                >
                  <span>{view[0].toUpperCase() + view.slice(1)}</span>
                  <span aria-hidden="true">↗</span>
                </button>
              ))}
            </div>
            <div className="mobile-menu-account">
              {user ? (
                <>
                  <span className="mobile-user-name">
                    Signed in as {user.displayName}
                  </span>
                  <button type="button" className="ghost" onClick={signOut}>
                    Sign out
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  className="ghost"
                  onClick={() => openDialog("auth")}
                >
                  Sign in
                </button>
              )}
              <button
                type="button"
                className="primary"
                onClick={() => openDialog(token ? "camera" : "auth")}
              >
                Add camera
              </button>
            </div>
          </div>
        )}
      </header>

      <section className="intro" id="top">
        <div>
          <p className="kicker">Live operations</p>
          <h1>
            Your spaces,
            <br />
            <em>in clear view.</em>
          </h1>
          <p className="intro-copy">
            Secure local camera streaming with a dedicated inference plane.
            Credentials stay encrypted and video stays on your network.
          </p>
        </div>
        <div className="system-card">
          <div>
            <span className="pulse" /> System overview
          </div>
          <dl>
            <div>
              <dt>Camera API</dt>
              <dd className={services.cameraApi === false ? "warn" : "good"}>
                {services.cameraApi === null
                  ? "Checking"
                  : services.cameraApi
                    ? "Online"
                    : "Unavailable"}
              </dd>
            </div>
            <div>
              <dt>Live feeds</dt>
              <dd>{streams.length}</dd>
            </div>
            <div>
              <dt>Inference</dt>
              <dd
                className={
                  services.inferenceApi === false
                    ? "warn"
                    : inference.ready
                      ? "good"
                      : "warn"
                }
              >
                {services.inferenceApi === false
                  ? "Unavailable"
                  : inference.ready
                    ? "Ready"
                    : "Needs model"}
              </dd>
            </div>
          </dl>
        </div>
      </section>

      {message && (
        <div className="notice" role="alert">
          {message}
          <button onClick={() => setMessage("")}>×</button>
        </div>
      )}

      <div className="workspace" id="workspace">
        {activeView === "cameras" && (
          <section
            className="section workspace-panel"
            id="panel-cameras"
            role="tabpanel"
            aria-labelledby="tab-cameras"
          >
            <div className="section-heading">
              <div>
                <p className="kicker">Camera wall</p>
                <h2>
                  {cameras.length
                    ? `${cameras.length} connected spaces`
                    : "Build your camera wall"}
                </h2>
              </div>
              <button className="text-button" onClick={load}>
                Refresh status ↗
              </button>
            </div>
            {cameras.length ? (
              <div className="camera-grid">
                {cameras.map((camera, index) => {
                  const stream = liveByCamera[camera.id];
                  return (
                    <article
                      className="camera-card"
                      key={camera.id}
                      style={{ "--delay": `${index * 60}ms` }}
                    >
                      <div className="viewport">
                        {stream ? (
                          <HlsPlayer
                            source={stream.streamUrl}
                            name={camera.name}
                          />
                        ) : (
                          <div className="standby">
                            <span>{String(index + 1).padStart(2, "0")}</span>
                            <p>Stream on standby</p>
                          </div>
                        )}
                        <div className="viewport-top">
                          <span className={`status ${stream ? "live" : ""}`}>
                            {stream ? "Live" : "Offline"}
                          </span>
                          <button
                            onClick={() => setSelected({ camera, stream })}
                            aria-label={`Expand ${camera.name}`}
                          >
                            ↗
                          </button>
                        </div>
                      </div>
                      <div className="camera-meta">
                        <div>
                          <h3>{camera.name}</h3>
                          <p>{camera.location || camera.host}</p>
                        </div>
                        <button
                          className={stream ? "stop" : "play"}
                          disabled={busy === camera.id}
                          onClick={() => toggleStream(camera)}
                        >
                          {busy === camera.id ? "…" : stream ? "Stop" : "Start"}
                        </button>
                      </div>
                    </article>
                  );
                })}
              </div>
            ) : (
              <div className="empty-state">
                <span>01</span>
                <h3>No cameras configured yet</h3>
                <p>
                  Create an account, add an RTSP camera, and Sentinel will
                  prepare its HLS stream here.
                </p>
                <button
                  className="primary"
                  onClick={() => openDialog(token ? "camera" : "auth")}
                >
                  Add your first camera
                </button>
              </div>
            )}
          </section>
        )}

        {activeView === "activity" && (
          <section
            className="activity workspace-panel"
            id="panel-activity"
            role="tabpanel"
            aria-labelledby="tab-activity"
          >
            <div>
              <p className="kicker">Live activity</p>
              <h2>
                Built for signal,
                <br />
                not noise.
              </h2>
            </div>
            <div className="activity-copy">
              <p>
                Sentinel reports active streams and keeps the optimized OpenVINO
                detector ready for person detection on your network.
              </p>
              <div className="model-row">
                <span>Active model</span>
                <strong>{inference.model || "Awaiting optimized model"}</strong>
                <span>{inference.device || "CPU"}</span>
              </div>
              <div className="activity-list" aria-live="polite">
                <h3>Current stream activity</h3>
                {streams.length ? (
                  streams.map((stream) => (
                    <div className="activity-item" key={stream.cameraId}>
                      <span className="status live">Live</span>
                      <strong>{stream.cameraName}</strong>
                      <span>
                        {stream.startedAt
                          ? `Started ${new Date(stream.startedAt).toLocaleString()}`
                          : "Stream active"}
                      </span>
                    </div>
                  ))
                ) : (
                  <p className="muted-copy">
                    No live streams right now. Start a camera to see it here.
                  </p>
                )}
              </div>
            </div>
          </section>
        )}

        {activeView === "system" && (
          <section
            className="section system-panel workspace-panel"
            id="panel-system"
            role="tabpanel"
            aria-labelledby="tab-system"
          >
            <div className="section-heading">
              <div>
                <p className="kicker">Service health</p>
                <h2>Your local stack at a glance.</h2>
              </div>
              <button className="text-button" onClick={load}>
                Check now ↗
              </button>
            </div>
            <div className="service-grid">
              <article className="service-item">
                <span className="service-label">Camera API</span>
                <strong>
                  {services.cameraApi === null
                    ? "Checking"
                    : services.cameraApi
                      ? "Online"
                      : "Unavailable"}
                </strong>
                <p>Authentication, cameras, and stream control.</p>
              </article>
              <article className="service-item">
                <span className="service-label">Inference API</span>
                <strong>
                  {services.inferenceApi === null
                    ? "Checking"
                    : services.inferenceApi
                      ? "Online"
                      : "Unavailable"}
                </strong>
                <p>
                  {services.inferenceApi === false
                    ? "The inference service could not be reached."
                    : inference.ready
                      ? "The optimized model is loaded and ready."
                      : "The service is online but still needs a model."}
                </p>
              </article>
              <article className="service-item">
                <span className="service-label">Live feeds</span>
                <strong>{streams.length}</strong>
                <p>{cameras.length} cameras configured on this system.</p>
              </article>
            </div>
            <p className="last-checked">
              {services.lastChecked
                ? `Last checked ${services.lastChecked.toLocaleTimeString()}`
                : "Running initial health check…"}
            </p>
          </section>
        )}
      </div>

      <footer>
        <span>Sentinel camera operations</span>
        <span>Local-first · encrypted credentials · OpenVINO inference</span>
      </footer>

      {selected && (
        <div className="overlay" onClick={() => setSelected(null)}>
          <div
            className="viewer"
            role="dialog"
            aria-modal="true"
            aria-label={`${selected.camera.name} viewer`}
            onClick={(event) => event.stopPropagation()}
          >
            <button className="close" onClick={() => setSelected(null)}>
              Close ×
            </button>
            <div className="viewer-video">
              {selected.stream ? (
                <HlsPlayer
                  source={selected.stream.streamUrl}
                  name={selected.camera.name}
                />
              ) : (
                <div className="standby">
                  <p>Stream is not running</p>
                </div>
              )}
            </div>
            <h2>{selected.camera.name}</h2>
            <p>{selected.camera.location}</p>
          </div>
        </div>
      )}

      {dialog === "auth" && (
        <AuthDialog
          busy={busy === "auth"}
          message={message}
          close={closeDialog}
          clearMessage={() => setMessage("")}
          submit={authenticate}
        />
      )}
      {dialog === "camera" && (
        <CameraDialog
          busy={busy === "camera"}
          message={message}
          close={closeDialog}
          submit={addCamera}
        />
      )}
    </main>
  );
}

function AuthDialog({ busy, message, close, clearMessage, submit }) {
  const [mode, setMode] = useState("login");
  return (
    <div className="overlay" onMouseDown={close}>
      <form
        className="dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="auth-title"
        onMouseDown={(event) => event.stopPropagation()}
        onSubmit={(event) => submit(event, mode)}
      >
        <button type="button" className="close" onClick={close}>
          Close ×
        </button>
        <p className="kicker">Operator access</p>
        <h2 id="auth-title">
          {mode === "login" ? "Welcome back." : "Create your account."}
        </h2>
        {mode === "register" && (
          <input
            name="displayName"
            autoComplete="name"
            placeholder="Display name"
            required
          />
        )}
        <input
          name="email"
          type="email"
          autoComplete="email"
          placeholder="Email address"
          required
        />
        <input
          name="password"
          type="password"
          autoComplete={mode === "login" ? "current-password" : "new-password"}
          minLength={12}
          placeholder="Password · 12 characters minimum"
          required
        />
        {message && (
          <p className="form-error" role="alert">
            {message}
          </p>
        )}
        <button type="submit" className="primary wide" disabled={busy}>
          {busy ? "Working…" : mode === "login" ? "Sign in" : "Create account"}
        </button>
        <button
          type="button"
          className="switch"
          onClick={() => {
            clearMessage();
            setMode(mode === "login" ? "register" : "login");
          }}
        >
          {mode === "login"
            ? "Need an account? Register"
            : "Already registered? Sign in"}
        </button>
      </form>
    </div>
  );
}

function CameraDialog({ busy, message, close, submit }) {
  return (
    <div className="overlay" onMouseDown={close}>
      <form
        className="dialog camera-form"
        role="dialog"
        aria-modal="true"
        aria-labelledby="camera-title"
        onMouseDown={(event) => event.stopPropagation()}
        onSubmit={submit}
      >
        <button type="button" className="close" onClick={close}>
          Close ×
        </button>
        <p className="kicker">New source</p>
        <h2 id="camera-title">Add an RTSP camera.</h2>
        <div className="form-grid">
          <input name="name" placeholder="Camera name" required />
          <input name="location" placeholder="Location" />
          <input name="host" placeholder="192.168.1.20" required />
          <input
            name="port"
            type="number"
            defaultValue="554"
            min="1"
            max="65535"
            required
          />
          <input
            className="full"
            name="rtspPath"
            defaultValue="/axis-media/media.amp"
            placeholder="RTSP path"
            required
          />
          <input
            name="username"
            autoComplete="username"
            placeholder="Username"
            required
          />
          <input
            name="password"
            type="password"
            autoComplete="current-password"
            placeholder="Camera password"
            required
          />
        </div>
        {message && (
          <p className="form-error" role="alert">
            {message}
          </p>
        )}
        <button type="submit" className="primary wide" disabled={busy}>
          {busy ? "Saving…" : "Save camera"}
        </button>
      </form>
    </div>
  );
}
