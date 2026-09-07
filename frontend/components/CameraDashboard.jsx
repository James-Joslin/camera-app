"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import HlsPlayer from "./HlsPlayer";

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
      ...(options.body ? { "content-type": "application/json" } : {}),
      ...(token ? { authorization: `Bearer ${token}` } : {}),
    },
    cache: "no-store",
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `Request failed (${response.status})`);
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
  }, []);

  useEffect(() => {
    const saved = window.localStorage.getItem("camera-session");
    if (saved) {
      const session = JSON.parse(saved);
      setToken(session.token);
      setUser(session.user);
    }
    load();
    const interval = window.setInterval(load, 5000);
    return () => window.clearInterval(interval);
  }, [load]);

  const liveByCamera = useMemo(
    () =>
      Object.fromEntries(streams.map((stream) => [stream.cameraId, stream])),
    [streams],
  );

  async function toggleStream(camera) {
    if (!token) return setDialog("auth");
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
      setMessage(error.message);
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
      setMessage(error.message);
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
      setMessage(error.message);
    } finally {
      setBusy("");
    }
  }

  function signOut() {
    window.localStorage.removeItem("camera-session");
    setToken("");
    setUser(null);
  }

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Sentinel home">
          <span className="brand-mark">S</span>
          <span>Sentinel</span>
        </a>
        <nav>
          <a href="#cameras">Cameras</a>
          <a href="#activity">Activity</a>
          <a href="#system">System</a>
        </nav>
        <div className="account">
          {user ? (
            <>
              <span className="user-name">{user.displayName}</span>
              <button className="ghost" onClick={signOut}>
                Sign out
              </button>
            </>
          ) : (
            <button className="ghost" onClick={() => setDialog("auth")}>
              Sign in
            </button>
          )}
          <button
            className="primary"
            onClick={() => (token ? setDialog("camera") : setDialog("auth"))}
          >
            Add camera
          </button>
        </div>
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
        <div className="system-card" id="system">
          <div>
            <span className="pulse" /> System overview
          </div>
          <dl>
            <div>
              <dt>Camera API</dt>
              <dd>Online</dd>
            </div>
            <div>
              <dt>Live feeds</dt>
              <dd>{streams.length}</dd>
            </div>
            <div>
              <dt>Inference</dt>
              <dd className={inference.ready ? "good" : "warn"}>
                {inference.ready ? "Ready" : "Needs model"}
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

      <section className="section" id="cameras">
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
                      <HlsPlayer source={stream.streamUrl} name={camera.name} />
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
              Create an account, add an RTSP camera, and Sentinel will prepare
              its HLS stream here.
            </p>
            <button
              className="primary"
              onClick={() => (token ? setDialog("camera") : setDialog("auth"))}
            >
              Add your first camera
            </button>
          </div>
        )}
      </section>

      <section className="activity" id="activity">
        <div>
          <p className="kicker">Inference plane</p>
          <h2>
            Built for signal,
            <br />
            not noise.
          </h2>
        </div>
        <div className="activity-copy">
          <p>
            The Python service loads the optimized OpenVINO detector lazily,
            keeps it warm, and returns structured person detections with
            per-request latency.
          </p>
          <div className="model-row">
            <span>Active model</span>
            <strong>{inference.model || "Awaiting optimized model"}</strong>
            <span>{inference.device || "CPU"}</span>
          </div>
        </div>
      </section>

      <footer>
        <span>Sentinel camera operations</span>
        <span>Local-first · encrypted credentials · OpenVINO inference</span>
      </footer>

      {selected && (
        <div className="overlay" onClick={() => setSelected(null)}>
          <div className="viewer" onClick={(event) => event.stopPropagation()}>
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
          close={() => setDialog(null)}
          submit={authenticate}
        />
      )}
      {dialog === "camera" && (
        <CameraDialog
          busy={busy === "camera"}
          close={() => setDialog(null)}
          submit={addCamera}
        />
      )}
    </main>
  );
}

function AuthDialog({ busy, close, submit }) {
  const [mode, setMode] = useState("login");
  return (
    <div className="overlay">
      <form className="dialog" onSubmit={(event) => submit(event, mode)}>
        <button type="button" className="close" onClick={close}>
          Close ×
        </button>
        <p className="kicker">Operator access</p>
        <h2>{mode === "login" ? "Welcome back." : "Create your account."}</h2>
        {mode === "register" && (
          <input name="displayName" placeholder="Display name" required />
        )}
        <input name="email" type="email" placeholder="Email address" required />
        <input
          name="password"
          type="password"
          minLength={12}
          placeholder="Password · 12 characters minimum"
          required
        />
        <button className="primary wide" disabled={busy}>
          {busy ? "Working…" : mode === "login" ? "Sign in" : "Create account"}
        </button>
        <button
          type="button"
          className="switch"
          onClick={() => setMode(mode === "login" ? "register" : "login")}
        >
          {mode === "login"
            ? "Need an account? Register"
            : "Already registered? Sign in"}
        </button>
      </form>
    </div>
  );
}

function CameraDialog({ busy, close, submit }) {
  return (
    <div className="overlay">
      <form className="dialog camera-form" onSubmit={submit}>
        <button type="button" className="close" onClick={close}>
          Close ×
        </button>
        <p className="kicker">New source</p>
        <h2>Add an RTSP camera.</h2>
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
          <input name="username" placeholder="Username" required />
          <input
            name="password"
            type="password"
            placeholder="Camera password"
            required
          />
        </div>
        <button className="primary wide" disabled={busy}>
          {busy ? "Saving…" : "Save camera"}
        </button>
      </form>
    </div>
  );
}
