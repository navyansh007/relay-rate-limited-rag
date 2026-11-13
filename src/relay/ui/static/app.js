/* Relay UI. Replays recorded real runs or streams live ones; every view is a
   pure function of (run, time), so replay and live share one renderer. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const fmt = (n) => Math.round(n).toLocaleString("en-US");
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

const STEPS = [
  ["Understand", "gpt-oss-20b reads the question: which paper, what to search."],
  ["Retrieve", "Local hybrid search: BM25 + embeddings, fused."],
  ["Decide", "Weak evidence or out of scope: refuse before spending a draft."],
  ["Draft", "gpt-oss-120b writes from the passages, streamed."],
  ["Verify", "Each sentence: cited, numbers found in the source, right paper."],
];

let DEMO = null;
let current = null; // the run being shown
let playing = null; // {start, run}
let mode = "replay";

/* ------------------------------------------------------------------ runs */

function typeMs(q) { return clamp(q.length * 26, 700, 1700); }

/** Turns a recorded session into a run: timed events the renderer understands. */
function runFromSession(s) {
  const spanEnd = (pred) => { const sp = s.spans.filter(pred); return sp.length ? Math.max(...sp.map((x) => x.end_ms)) : null; };
  const readings = [];
  let draftIdx = 0;
  const draftSpans = s.spans.filter((x) => x.name.startsWith("draft.attempt"));
  for (const b of s.budget) {
    let t = 0;
    if (b.step === "understand") t = spanEnd((x) => x.name.startsWith("understand")) ?? 0;
    else { const sp = draftSpans[draftIdx++]; t = sp ? sp.end_ms : s.total_ms; }
    readings.push({ t, model: b.model, remaining: b.remaining, limit: b.limit });
  }
  const retrieveEnd = spanEnd((x) => x.name.startsWith("retrieve")) ?? 0;
  return {
    question: s.question, source: s.source, setting: s.setting, spans: s.spans,
    texts: s.text ? [{ t: s.first_token_ms ?? s.total_ms, text: s.text }] : [],
    passages: s.passages, passagesAt: retrieveEnd, readings,
    first_token_ms: s.first_token_ms, total_ms: s.total_ms, tokens_in: s.tokens_in,
    tokens_out: s.tokens_out, status: s.status, answer_kind: s.answer_kind,
    evidence: s.evidence, understanding: s.understanding, held: s.held, degraded: s.degraded,
    complete: true, doneAt: s.total_ms,
  };
}

function emptyRun(question, source) {
  return { question, source, setting: "", spans: [], texts: [], passages: [], passagesAt: 0, readings: [],
    first_token_ms: null, total_ms: null, tokens_in: 0, tokens_out: 0, status: "", answer_kind: "",
    evidence: null, understanding: {}, held: 0, degraded: [], complete: false, doneAt: null };
}

/* ------------------------------------------------------------------ ask view */

function formatAnswer(text) {
  const cite = (s) => s.replace(/\[((?:transformer|moshi|hnet) [^\]]+)\]/g, '<span class="cite">$1</span>');
  const out = [];
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    if (line.startsWith(">")) {
      const m = line.replace(/^>\s*/, "");
      out.push(`<blockquote>${cite(esc(m))}</blockquote>`);
    } else out.push(`<p>${cite(esc(line))}</p>`);
  }
  return out.join("");
}

function sentenceCount(text) {
  return text.trim().split(/(?<=[.!?\]])\s+(?=[A-Z\u2018"])/).filter((x) => /[a-z]/i.test(x)).length;
}

function stepInfo(run, r) {
  const sp = run.spans.filter((x) => x.start_ms <= r);
  const by = (p) => sp.filter((x) => x.name.startsWith(p));
  const ended = (arr) => arr.length && arr.every((x) => x.end_ms <= r);
  const dur = (arr) => arr.reduce((a, x) => a + (Math.min(x.end_ms, r) - x.start_ms), 0);
  const finished = run.complete && r >= run.doneAt;
  const info = STEPS.map(() => ({ state: "", d: "", ms: "" }));

  const u = by("understand");
  if (u.length) {
    info[0] = { state: ended(u) ? "done" : "active", ms: `${fmt(dur(u))} ms`,
      d: ended(u) ? (run.understanding?.papers?.length ? `Paper: ${run.understanding.papers.join(", ")} · query "${run.understanding.query}"` : "Routed to a paper and a search query.") : STEPS[0][1] };
  } else if (r > 0 && run.spans.some((x) => x.name.startsWith("retrieve"))) {
    info[0] = { state: "done", ms: "0 ms", d: "Keyword routing (model budget saved)." };
  }
  const rt = by("retrieve");
  if (rt.length) {
    const done = ended(rt);
    info[1] = { state: done ? "done" : "active", ms: `${fmt(dur(rt))} ms`,
      d: done ? `${run.passages.length} passages · setting ${run.setting || "s2_big_few"}` : STEPS[1][1] };
    if (done) {
      const refused = run.answer_kind === "refusal" && !run.spans.some((x) => x.name.startsWith("draft"));
      info[2] = { state: "done", ms: "0 ms",
        d: run.evidence != null ? `Best match ${run.evidence.toFixed(2)} · ${refused ? "refuse" : "draft an answer"}` : "Evidence checked · draft an answer" };
    }
  }
  const dw = by("draft").filter((x) => x.kind === "wait");
  const da = by("draft.attempt");
  if (dw.length || da.length) {
    const w = dw.find((x) => x.end_ms > r);
    const refused = dw.find((x) => x.name.includes("refused") && x.end_ms <= r);
    if (refused) info[3] = { state: "done waiting", ms: `${fmt(dur(dw))} ms`, d: "Over budget for this deadline → quote the sources." };
    else if (w) info[3] = { state: "active waiting", ms: `${fmt(r - w.start_ms)} ms`, d: `Waiting for token budget · ${((r - w.start_ms) / 1000).toFixed(1)} s` };
    else if (da.length && !ended(da)) info[3] = { state: "active", ms: `${fmt(dur(da) + dur(dw))} ms`, d: dw.length ? `Waited ${(dur(dw) / 1000).toFixed(1)} s for budget · drafting` : "gpt-oss-120b drafting…" };
    else if (da.length) {
      const ttft = da[da.length - 1].attrs?.ttft_ms;
      info[3] = { state: "done", ms: `${fmt(dur(da) + dur(dw))} ms`,
        d: `${dw.length ? `Waited ${(dur(dw) / 1000).toFixed(1)} s for budget · ` : ""}model first token ${ttft ? fmt(ttft) + " ms" : "–"}` };
    } else info[3] = { state: "active waiting", ms: "", d: "Waiting for token budget" };
  }
  const shownText = run.texts.some((x) => x.t <= r);
  if (shownText || finished) {
    const k = run.answer_kind;
    info[4] = { state: "done", ms: "",
      d: k === "answer" ? ((n) => `${n} sentence${n === 1 ? "" : "s"} passed · ${run.held} held`)(sentenceCount(run.texts.map((x) => x.text).join(" ")))
        : k === "refusal" ? "Nothing in the passages answers it → refuse."
          : k === "extractive" ? "No model text: exact quotes, nothing to verify." : "–" };
  }
  // live mode: spans arrive when finished, so infer the step in progress
  if (!run.complete && !info.some((x) => x.state.includes("active"))) {
    const next = info.findIndex((x) => !x.state);
    if (next >= 0) info[next] = { state: "active", ms: "", d: STEPS[next][1] };
  }
  return info;
}

function renderAsk(run, r, typedMs = Infinity) {
  if (!run) return;
  // question (typed out during replay)
  const qEl = $("#q-replay");
  if (mode === "replay") {
    const n = typedMs === Infinity ? run.question.length : clamp(Math.floor(typedMs / (typeMs(run.question) / run.question.length)), 0, run.question.length);
    const caret = r < 0 && Math.floor(typedMs / 450) % 2 === 0 ? '<span class="caret"></span>' : "";
    qEl.innerHTML = esc(run.question.slice(0, n)) + caret;
  }
  // steps
  const info = stepInfo(run, r);
  $("#steps").innerHTML = STEPS.map(([name], i) => {
    const s = info[i];
    return `<li class="step ${s.state}"><span class="n">0${i + 1}</span><span class="name">${name}</span><span class="ms">${s.ms}</span><span class="d">${esc(s.d || (r < 0 || !s.state ? "" : ""))}</span></li>`;
  }).join("");
  // answer
  const shown = run.texts.filter((x) => x.t <= r).map((x) => x.text).join("");
  $("#answer").innerHTML = shown ? formatAnswer(shown)
    : `<span class="pending">${r < 0 ? "The answer appears here, one verified sentence at a time." : "Waiting for the first verified sentence…"}</span>`;
  // meta
  const done = run.complete && r >= run.doneAt;
  const ft = run.first_token_ms;
  $("#m-ttft").innerHTML = ft != null && r >= ft ? `${fmt(ft)}<small>ms</small>` : "–";
  $("#m-total").innerHTML = done ? `${fmt(run.total_ms)}<small>ms</small>` : r > 0 ? `${fmt(r)}<small>ms</small>` : "–";
  $("#m-tokens").innerHTML = done ? `${fmt(run.tokens_in + run.tokens_out)}` : "–";
  const out = { answer: "Answered", refusal: "Refused", extractive: "Quoted", none: "Gave up" }[run.answer_kind] || "–";
  $("#m-out").innerHTML = done ? `<span class="${run.answer_kind === "answer" ? "" : "red"}">${out}</span>` : "–";
  // passages
  $("#passages").innerHTML = run.passages.length && r >= run.passagesAt
    ? `<span class="label">Retrieved passages</span>` + run.passages.map((p) => `<div class="passage"><span class="c">${esc(p.cite)}</span><span class="x">${esc(p.text)}</span></div>`).join("")
    : "";
  renderWaterfall(run, r);
  renderGauges(run, r);
}

function renderWaterfall(run, r) {
  const scale = Math.max(run.complete ? run.total_ms : Math.max(r, 3000), 1000);
  const rows = run.spans.filter((x) => x.start_ms <= r);
  const label = (n) => n.replace(".attempt0", "").replace(".attempt", " retry ").replace(".limiter_refused", " · refused").replace(".limiter", " · wait").replace(".0", "").replace(".1", " again");
  const html = rows.map((x) => {
    const left = (x.start_ms / scale) * 100;
    const w = Math.max(0.4, ((Math.min(x.end_ms, r) - x.start_ms) / scale) * 100);
    const cls = x.kind === "work" ? "" : x.kind;
    return `<div class="wf-row"><span>${esc(label(x.name))}</span><div class="wf-track"><div class="wf-bar ${cls}" style="left:${left}%;width:${w}%"></div></div></div>`;
  }).join("");
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => `<span>${fmt((scale * f) / 1000 * 10) / 10}s</span>`).join("");
  $("#wf").innerHTML = (html || `<div class="wf-row"><span>–</span><div class="wf-track"></div></div>`) + `<div class="wf-axis"><span></span><div class="ticks">${ticks}</div></div>`;
}

function renderGauges(run, r) {
  const models = ["gpt-oss-20b", "gpt-oss-120b"];
  const role = { "gpt-oss-20b": "understand", "gpt-oss-120b": "draft" };
  $("#gauges").innerHTML = models.map((m) => {
    const rd = run.readings.filter((x) => x.model === m && x.t <= r).pop();
    const pct = rd ? (rd.remaining / rd.limit) * 100 : 0;
    return `<div class="gauge"><div class="top"><span>${m} <span class="label">· ${role[m]}</span></span><b>${rd ? fmt(rd.remaining) + " / " + fmt(rd.limit) : "–"}</b></div>
      <div class="bar"><div class="fill ${pct < 20 ? "low" : ""}" style="width:${pct}%"></div></div></div>`;
  }).join("");
}

/* ------------------------------------------------------------------ burst view */

const BURST_MAX = 18000;
function buildBurst() {
  const lanes = DEMO.burst.lanes;
  $("#lanes").innerHTML = lanes.map((l, i) => `<div class="lane" data-i="${i}"><div class="q"><b>${String(i + 1).padStart(2, "0")}</b>${esc(l.question)}</div><div class="track"></div><div class="out"></div></div>`).join("");
  $("#ticks").innerHTML = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18].map((s) => `<span style="left:${(s * 1000 / BURST_MAX) * 100}%">${s}s</span>`).join("");
}

function laneSegments(l) {
  // recorded spans, plus any untraced gap before the end: those requests were
  // waiting in the limiter (the trace bug fixed later did not record it)
  const segs = l.spans.map((s) => ({ kind: s.name.includes("refused") ? "refused" : s.kind, a: s.start_ms, b: s.end_ms }));
  const last = segs.length ? Math.max(...segs.map((s) => s.b)) : 0;
  if (l.total_ms - last > 50) segs.push({ kind: l.status === "timeout" ? "wait" : "wait", a: last, b: l.total_ms });
  return segs;
}

function renderBurst(r) {
  $("#clock").innerHTML = `${(clamp(r, 0, BURST_MAX) / 1000).toFixed(1)}<small>s</small>`;
  DEMO.burst.lanes.forEach((l, i) => {
    const el = $(`.lane[data-i="${i}"]`);
    const local = r - l.start_ms;
    const track = $(".track", el);
    const x = (ms) => ((l.start_ms + ms) / BURST_MAX) * 100;
    if (local <= 0) { track.innerHTML = ""; $(".out", el).textContent = ""; $(".out", el).className = "out"; return; }
    track.innerHTML = laneSegments(l).filter((s) => s.a < local).map((s) => {
      const w = ((Math.min(s.b, local) - s.a) / BURST_MAX) * 100;
      return `<div class="seg ${s.kind === "work" ? "" : s.kind}" style="left:${x(s.a)}%;width:${Math.max(0.15, w)}%"></div>`;
    }).join("") + (local >= l.total_ms ? `<div class="end" style="left:${x(l.total_ms)}%"></div>` : "");
    const out = $(".out", el);
    if (local >= l.total_ms) {
      const k = l.status === "ok" ? "answered" : l.status === "rate_limited" ? "quoted" : "gave_up";
      out.className = `out ${k}`;
      out.textContent = { answered: `answered ${(l.total_ms / 1000).toFixed(1)} s`, quoted: "quoted sources", gave_up: "gave up" }[k];
    } else { out.className = "out"; out.textContent = ""; }
  });
  // tokens-left line
  const pts = DEMO.burst.tpm.filter((p) => p[0] <= r);
  const X = (t) => (t / BURST_MAX) * 1000, Y = (v) => 120 - (v / 8000) * 120;
  let d = "";
  pts.forEach((p, i) => { d += i ? ` H ${X(p[0])} V ${Y(p[1])}` : `M ${X(p[0])} ${Y(p[1])}`; });
  if (pts.length) d += ` H ${X(Math.min(r, BURST_MAX))}`;
  const lastPt = pts[pts.length - 1];
  $("#tpm").innerHTML = `<line x1="0" y1="120" x2="1000" y2="120" stroke="#0e0e0e" stroke-width="1.5"/>
    <line x1="0" y1="0" x2="1000" y2="0" stroke="#cfccc4" stroke-width="1" stroke-dasharray="4 4"/>
    <path d="${d}" fill="none" stroke="#e2231a" stroke-width="3" vector-effect="non-scaling-stroke"/>`;
  $("#tpm-now").innerHTML = lastPt ? `<span class="num" style="font:800 40px/1 var(--display)">${fmt(lastPt[1])}</span><br>of 8,000 left` : "";
}

/* ------------------------------------------------------------------ measured view */

const FIGS = (s) => [
  [s.tpm, "", "tokens per minute, per model", "The real ceiling. About 2,150 tokens per answer means 3.7 answers a minute on one key."],
  [s.tpd, "", "tokens per day, found by spending them", "The daily 429 said: Limit 200000, retry-after 262 s. It refills continuously, not at midnight."],
  [s.steady_ttft_p50, "ms", "first verified sentence, median", `Real Groq, steady load. p95 ${fmt(s.steady_ttft_p95)} ms over 17 answers.`],
  [s.refused_correctly, `/${s.unanswerable_asked}`, "unanswerable questions refused", "8 questions the papers do not answer × 4 retrieval settings, on real Groq."],
  [s.real_429s_in_burst, "", "429 errors in a real 12-question burst", "The limiter waited for the budget instead of letting the provider refuse."],
  [22, "/24", "real answers arrived all at once", "gpt-oss reasons silently, then sends the whole answer. Streaming did not shorten the wait."],
];

function buildMeasured() {
  $("#figures").innerHTML = FIGS(DEMO.stats).map((f, i) => `<div class="fig"><div class="v num" data-i="${i}">0</div><div class="k">${f[2]}</div><div class="s">${f[3]}</div></div>`).join("");
  const names = { real_steady: "Steady", real_burst: "Burst", steady: "Steady", burst: "Burst, 100 in 10 s", burst_cache: "Same burst, cache on", overload: "Overload, 4× capacity", rpm_halved: "Limits halved", day_exhausted: "Daily budget runs out" };
  $("#outcomes").innerHTML = `<span class="label">Outcomes by load shape · ■ answered  <span class="red">■ cited sources at once</span>  ▨ gave up</span>` +
    DEMO.stats.outcomes.map((o, i) => `<div class="oc" data-i="${i}"><span>${names[o.scenario]}</span><span class="src">${o.source === "real" ? "real Groq" : "stand-in"} · ${o.n}</span><div class="stack"></div></div>`).join("");
}

function renderMeasured(r) {
  FIGS(DEMO.stats).forEach((f, i) => {
    const p = clamp((r - i * 180) / 900, 0, 1);
    const e = 1 - Math.pow(1 - p, 3);
    $(`.fig .v[data-i="${i}"]`).innerHTML = `${fmt(f[0] * e)}<small>${f[1]}</small>`;
  });
  DEMO.stats.outcomes.forEach((o, i) => {
    const p = clamp((r - 900 - i * 120) / 700, 0, 1);
    const e = 1 - Math.pow(1 - p, 3);
    const w = (k) => ((o[k] / o.n) * 100 * e).toFixed(2);
    $(`.oc[data-i="${i}"] .stack`).innerHTML = `<i class="a" style="width:${w("answered")}%"></i><i class="qq" style="width:${w("quoted")}%"></i><i class="g" style="width:${w("gave_up")}%"></i>`;
  });
}

/* ------------------------------------------------------------------ replay + live control */

function sessionList() {
  $("#sessions").innerHTML = DEMO.sessions.map((s, i) => `<li><button data-id="${s.id}"><span class="i">${String(i + 1).padStart(2, "0")}</span><span class="t">${esc(s.label)}</span><span class="b">${esc(s.blurb)}</span></button></li>`).join("");
  $("#sessions").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-id]");
    if (b) playSession(b.dataset.id);
  });
}

function playSession(id) {
  const s = DEMO.sessions.find((x) => x.id === id);
  document.querySelectorAll("#sessions button").forEach((b) => b.setAttribute("aria-current", String(b.dataset.id === id)));
  current = runFromSession(s);
  $("#src-label").textContent = `Recorded on Groq · ${s.setting} · ${s.id}`;
  playing = { start: performance.now(), run: current, typed: typeMs(s.question) };
}

function tick() {
  if (playing && mode === "replay") {
    const el = performance.now() - playing.start;
    const r = el - playing.typed - 250;
    renderAsk(playing.run, r, el);
    if (r > playing.run.total_ms + 400) playing = null;
  } else if (playing && mode === "live") {
    renderAsk(playing.run, performance.now() - playing.start);
    if (playing.run.complete) { renderAsk(playing.run, playing.run.doneAt + 1); playing = null; $("#go").disabled = false; }
  }
  if (burstPlay) {
    const r = performance.now() - burstPlay;
    renderBurst(r);
    if (r > BURST_MAX) burstPlay = null;
  }
  if (measuredPlay) {
    const r = performance.now() - measuredPlay;
    renderMeasured(r);
    if (r > 3000) measuredPlay = null;
  }
  requestAnimationFrame(tick);
}
let burstPlay = null;
let measuredPlay = null;

async function askLive() {
  const q = $("#q-live").value.trim();
  if (!q) return;
  $("#go").disabled = true;
  const run = emptyRun(q, "live");
  current = run;
  const t0 = performance.now();
  playing = { start: t0, run };
  try {
    const resp = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q }) });
    if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let k;
      while ((k = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, k).replace(/^data: /, "");
        buf = buf.slice(k + 2);
        const ev = JSON.parse(line);
        const t = performance.now() - t0;
        if (ev.type === "start") $("#src-label").textContent = `Live · ${ev.source}`;
        if (ev.type === "span") {
          run.spans.push(ev);
          if (ev.passages) { run.passages = ev.passages; run.passagesAt = ev.end_ms; }
          if (ev.name.startsWith("retrieve") && ev.attrs?.hits && !run.setting) run.setting = "";
        }
        if (ev.type === "text") { run.texts.push({ t: ev.t, text: ev.text }); if (run.first_token_ms == null) run.first_token_ms = ev.t; }
        if (ev.type === "budget") for (const [m, b] of Object.entries(ev.models)) run.readings.push({ t: ev.t, model: m, remaining: b.remaining, limit: b.limit });
        if (ev.type === "done") {
          Object.assign(run, { status: ev.status, answer_kind: ev.answer_kind || "none", total_ms: ev.total_ms ?? t, tokens_in: ev.tokens_in || 0,
            tokens_out: ev.tokens_out || 0, setting: ev.setting || run.setting, first_token_ms: ev.first_token_ms ?? run.first_token_ms, complete: true, doneAt: ev.total_ms ?? t });
          if (ev.error) run.texts.push({ t, text: `Error: ${ev.error}` });
        }
      }
    }
  } catch (e) {
    run.texts.push({ t: performance.now() - t0, text: `Could not run live: ${e.message}` });
    Object.assign(run, { complete: true, doneAt: performance.now() - t0, total_ms: performance.now() - t0, answer_kind: "none" });
  }
}

function setMode(m) {
  mode = m;
  $("#mode-replay").setAttribute("aria-pressed", String(m === "replay"));
  $("#mode-live").setAttribute("aria-pressed", String(m === "live"));
  $("#sessions").hidden = m !== "replay";
  $("#q-replay").hidden = m !== "replay";
  $("#q-live").hidden = m !== "live";
  $("#go").textContent = m === "replay" ? "Replay" : "Ask";
  if (m === "live") { $("#src-label").textContent = "Live"; $("#q-live").focus(); }
}

/* ------------------------------------------------------------------ boot */

async function boot() {
  DEMO = await (await fetch("/static/demo.json")).json();
  sessionList();
  buildBurst();
  buildMeasured();
  renderBurst(0);
  renderMeasured(0);

  $("#mode-replay").onclick = () => setMode("replay");
  $("#mode-live").onclick = () => setMode("live");
  $("#go").onclick = () => (mode === "replay" ? playSession(current ? DEMO.sessions.find((x) => x.question === current.question)?.id || DEMO.sessions[0].id : DEMO.sessions[0].id) : askLive());
  $("#q-live").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); askLive(); } });
  try {
    const st = await (await fetch("/api/status")).json();
    $("#mode-live").textContent = `Live · ${st.source}`;
    if (!st.live) { $("#mode-live").disabled = true; $("#mode-live").title = st.reason; }
  } catch { $("#mode-live").disabled = true; }

  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      if (e.target.id === "burst" && !burstPlay) burstPlay = performance.now();
      if (e.target.id === "measured" && !measuredPlay) measuredPlay = performance.now();
      io.unobserve(e.target);
    }
  }, { threshold: 0.35 });
  io.observe($("#burst"));
  io.observe($("#measured"));
  $("#clock").style.cursor = "pointer";
  $("#clock").title = "Replay the burst";
  $("#clock").onclick = () => (burstPlay = performance.now());

  playSession(DEMO.sessions[0].id);
  requestAnimationFrame(tick);
}

boot();
