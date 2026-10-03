// Chat client. Keeps the conversation history client-side and renders the
// execution trace the API returns, so the tool calls behind an answer are
// visible rather than implied.

const messagesEl = document.getElementById("messages");
const traceEl = document.getElementById("trace");
const form = document.getElementById("composer");
const input = document.getElementById("input");
// Restored by "New conversation" so the chat never ends up blank.
const welcome = messagesEl.querySelector(".welcome")?.cloneNode(true);

// Sent back with each turn so follow-ups ("yes, file it") keep their context.
// Only the plain user/assistant turns are kept; tool traffic is reconstructed
// server-side each turn and would bloat the payload for no benefit.
let history = [];
let busy = false;

const STEP_LABELS = {
  tool_calls: "Tool calls",
  final: "Final answer",
  validation: "Evidence check",
  error: "Error",
  truncated: "Step limit reached",
};

// Fields surfaced first in a one-line tool output summary. Other fields still
// follow, and the full JSON is one click away.
const KEY_FIELDS = [
  "ticket_id", "draft_id", "employee_name", "full_name", "available_days", "available_hours",
  "days_used", "days_remaining", "annual_limit_days", "employment_type", "benefits_eligible",
  "short_term_disability", "work_arrangement", "count", "status", "citation",
];

const PREVIEW_ORDER = {
  create_hr_ticket: ["employee_name", "category", "priority", "would_route_to", "subject", "body"],
  draft_hr_email: ["to", "subject", "body"],
};

const PREVIEW_LABELS = {
  employee_name: "Employee", category: "Category", priority: "Priority",
  would_route_to: "Routes to", subject: "Subject", body: "Details", to: "To",
};

const CATEGORY_LABELS = {
  international_remote_work: "International remote work", remote_work: "Remote work",
  pto: "PTO", benefits: "Benefits", equipment: "Equipment", expense: "Expense",
  security: "Security",
};

function headers() {
  return {
    "Content-Type": "application/json",
    "X-Demo-Code": document.getElementById("demo-code")?.value || "",
  };
}

function setBusy(value) {
  busy = value;
  document.querySelectorAll("button").forEach((button) => {
    button.disabled = value || button.dataset.completed === "true";
  });
}

const escape = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function formatMs(ms) {
  if (!ms) return "";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

function shorten(value, max = 60) {
  const text = String(value);
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function formatValue(value) {
  if (Array.isArray(value)) return plural(value.length, "item");
  return typeof value === "string" ? `"${shorten(value)}"` : String(value);
}

// `key: value` pairs for the scalar fields of an object; lists show a count.
function compact(obj, max = 6, preferred = []) {
  if (!obj || typeof obj !== "object") return "";
  const shown = (key) => {
    const value = obj[key];
    return value !== null && value !== undefined && value !== "" &&
      (typeof value !== "object" || Array.isArray(value));
  };
  const keys = [
    ...preferred.filter((k) => k in obj && shown(k)),
    ...Object.keys(obj).filter((k) => !preferred.includes(k) && shown(k)),
  ];
  const more = keys.length > max ? ` · +${keys.length - max} more` : "";
  return keys.slice(0, max).map((k) => `${k}: ${formatValue(obj[k])}`).join(" · ") + more;
}

function addMessage(role, text, options = {}) {
  const { citations = [], sources = [], html = "", cached = false } = options;
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML =
    (cached
      ? `<span class="badge" title="Returned from the 10-minute cache; no model call was made for this request">Cached answer</span>`
      : "") +
    (html ||
      escape(text)
        .split(/\n{2,}/)
        .map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`)
        .join(""));
  if (citations.length) {
    // The label says where a claim came from; the snippet shows what was
    // there. Collapsed by default so the answer stays readable, but present,
    // so a citation can be checked without leaving the page.
    const byLabel = new Map(sources.map((s) => [s.citation, s]));
    el.innerHTML +=
      `<div class="cites"><span class="cites-label">Sources</span>` +
      citations
        .map((c) => {
          const source = byLabel.get(c);
          if (!source) return `<span class="cite">${escape(c)}</span>`;
          return `<details class="cite"><summary title="${escape(source.doc_title)}">${escape(c)}</summary>
            <p class="snippet">${escape(source.snippet)}</p></details>`;
        })
        .join("") +
      `</div>`;
  }
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function addDivider(text) {
  const el = document.createElement("div");
  el.className = "divider";
  el.textContent = text;
  messagesEl.appendChild(el);
}

// ---------------------------------------------------------------------------
// Execution trace
// ---------------------------------------------------------------------------

function outputSummary(call) {
  const r = call.result;
  if (call.is_error) {
    const message = typeof r === "string" ? r : r?.error || r?.message || "the tool returned an error";
    return `Error: ${shorten(message, 160)}`;
  }
  if (!r || typeof r !== "object") return shorten(String(r ?? ""), 160);
  if (r.requires_confirmation) return "Preview only: nothing is created until you confirm it";
  if (Array.isArray(r.hits)) {
    const best = typeof r.best_similarity === "number"
      ? ` · best similarity ${r.best_similarity.toFixed(2)}`
      : "";
    return r.grounded === false
      ? `${plural(r.hits.length, "passage")} retrieved, none relevant enough to answer from${best}`
      : `${plural(r.hits.length, "passage")} retrieved${best}`;
  }
  if (typeof r.compliant === "boolean") {
    const blocking = (r.blocking_reasons || []).length;
    const review = (r.findings || []).filter((f) => f.status === "review_required").length;
    const options = (r.alternatives || []).length;
    return [
      r.compliant ? "Compliant" : "Not compliant",
      blocking && plural(blocking, "blocking issue"),
      review && `${review} for HR review`,
      options && plural(options, "alternative"),
    ].filter(Boolean).join(" · ");
  }
  return compact(r, 4, KEY_FIELDS);
}

function renderToolCall(call) {
  return `
    <div class="tool${call.is_error ? " err" : ""}">
      <div class="tool-head">
        <code class="name">${escape(call.name)}</code>
        <span class="muted">${escape(formatMs(call.latency_ms))}</span>
      </div>
      <div class="kv"><span class="k">Arguments</span><code>${escape(compact(call.arguments, 8) || "none")}</code></div>
      <div class="kv"><span class="k">Output</span><span>${escape(outputSummary(call))}</span></div>
      <details>
        <summary>Full arguments and output (JSON)</summary>
        <pre>${escape(JSON.stringify(call.arguments, null, 1))}</pre>
        <pre>${escape(JSON.stringify(call.result, null, 1))}</pre>
      </details>
    </div>`;
}

function renderStep(step) {
  const calls = step.tool_calls || [];
  const label = STEP_LABELS[step.kind] || step.kind;
  const parallel = calls.length > 1 ? ` · ${calls.length} in parallel` : "";
  // Tool steps are described by their calls below; other steps carry an
  // operational one-liner from the server, never the model's own narration.
  const note = step.kind === "tool_calls"
    ? ""
    : step.summary || (step.kind === "truncated" ? "The agent reached its step limit." : "");
  let timing = formatMs(step.latency_ms);
  if (timing) timing = `model ${timing}`;
  if (timing && step.throttle_ms > 1000) timing += ` (incl. ${formatMs(step.throttle_ms)} rate-limit wait)`;
  return `
    <div class="step">
      <div class="head">
        <span>Step ${escape(step.index)} · ${escape(label)}${parallel}</span>
        ${timing ? `<span class="muted" title="Model response time for this step">${escape(timing)}</span>` : ""}
      </div>
      <div class="body">${note ? `<p class="muted">${escape(note)}</p>` : ""}${calls.map(renderToolCall).join("")}</div>
    </div>`;
}

function toolPath(steps) {
  return steps
    .filter((s) => (s.tool_calls || []).length)
    .map((s) => {
      const counts = new Map();
      s.tool_calls.forEach((c) => counts.set(c.name, (counts.get(c.name) || 0) + 1));
      return [...counts].map(([name, n]) => (n > 1 ? `${name} ×${n}` : name)).join(" + ");
    })
    .join(" → ");
}

function outcome(trace) {
  if (trace.error) return ["err", `Stopped with an error: ${shorten(trace.error, 200)}`];
  if (trace.truncated) return ["warn", "Stopped at the step limit before a final answer"];
  if (trace.grounded === false) return ["warn", "Declined: the policy documents do not cover this question"];
  const cited = (trace.citations || []).length;
  const actions = (trace.pending_actions || []).length;
  let tone = "ok";
  let text = `Answered from ${plural(cited, "cited policy section")}`;
  if (!cited) {
    tone = "";
    text = (trace.tools_used || []).length
      ? "Answered from tool results, without policy citations"
      : "Replied without tools or citations";
  }
  if (actions) text += `; ${plural(actions, "mock action")} awaiting your confirmation`;
  return [tone, text];
}

function sourcesRow(trace) {
  const cited = trace.citations || [];
  const others = (trace.retrieved_citations || []).filter((c) => !cited.includes(c));
  if (!cited.length && !others.length) return `<span class="muted">No policy sources retrieved</span>`;
  const chips = cited.length
    ? `<div class="chips">${cited.map((c) => `<span class="tag">${escape(c)}</span>`).join("")}</div>`
    : `<span class="muted">None cited in the answer</span>`;
  const rest = others.length
    ? `<details class="more"><summary>Also retrieved, not cited (${others.length})</summary>
        <div class="chips">${others.map((c) => `<span class="tag quiet">${escape(c)}</span>`).join("")}</div>
      </details>`
    : "";
  return chips + rest;
}

function evidenceCheck(trace) {
  let text = "No policy retrieval in this turn";
  if (trace.grounded === true) text = "Policy search found relevant passages";
  else if (trace.grounded === false) text = "Policy search found no relevant passage";
  else if ((trace.sources || []).length) text = "Policy sections retrieved directly";
  if ((trace.steps || []).some((s) => s.kind === "validation")) {
    text += "; the draft answer was sent back once to add missing evidence";
  }
  return text;
}

function escalation(trace) {
  const items = (trace.pending_actions || []).map((a) => {
    if (a.tool === "create_hr_ticket") {
      return `Mock HR ticket prepared for ${a.preview?.would_route_to || "HR"}, awaiting your confirmation`;
    }
    if (a.tool === "draft_hr_email") {
      return `Mock email draft to ${a.preview?.to || "the employee"}, awaiting your confirmation`;
    }
    return `${a.tool} preview awaiting your confirmation`;
  });
  if (!trace.error && !trace.truncated && trace.grounded === false) items.push("Redirected to People Operations");
  if (trace.truncated) items.push("Advised to narrow the question or contact HR");
  return items.length ? items.join("; ") : "None in this turn (no ticket, email draft or HR redirect)";
}

function usage(trace) {
  const run = [formatMs(trace.total_ms) || "0 ms"];
  if (trace.throttle_ms > 1000) run.push(`including ${formatMs(trace.throttle_ms)} waiting on provider rate limits`);
  const tokens = (trace.prompt_tokens || 0) + (trace.completion_tokens || 0);
  if (tokens) run.push(`${tokens.toLocaleString()} tokens`);
  if (trace.provider) run.push(trace.model ? `${trace.provider} / ${trace.model}` : trace.provider);
  if (trace.fell_back) run.push("fallback model used");
  if (trace.cached) return `Cached answer: no model call for this request. Original run: ${run.join(" · ")}`;
  return [plural(trace.api_calls ?? 0, "model call"), ...run].join(" · ");
}

function renderTrace(trace) {
  const steps = trace.steps || [];
  const [tone, outcomeText] = outcome(trace);
  const path = toolPath(steps);
  const calls = steps.reduce((n, s) => n + (s.tool_calls || []).length, 0);
  const toolSteps = steps.filter((s) => (s.tool_calls || []).length).length;
  const rows = [
    ["Outcome", `<span class="dot ${tone}" aria-hidden="true"></span>${escape(outcomeText)}`],
    ["Selected tools", path
      ? `<code>${escape(path)}</code><div class="muted">${plural(calls, "call")} in ${plural(toolSteps, "step")}</div>`
      : `<span class="muted">No tools selected</span>`],
    ["Policy sources", sourcesRow(trace)],
    ["Evidence check", escape(evidenceCheck(trace))],
    ["Escalation", escape(escalation(trace))],
    ["Usage", escape(usage(trace))],
  ];
  traceEl.innerHTML = `
    <section class="basis" aria-label="Final answer basis">
      <h3>Final answer basis</h3>
      <dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>
    </section>
    <h3>Steps</h3>
    ${steps.map(renderStep).join("") || `<p class="muted">No steps recorded.</p>`}`;
  traceEl.scrollTop = 0;
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

async function send(question) {
  if (busy) return;
  setBusy(true);
  addMessage("user", question);
  input.value = "";

  const pending = document.createElement("div");
  pending.className = "msg assistant pending";
  messagesEl.appendChild(pending);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  const started = Date.now();
  const tick = () => {
    const s = Math.round((Date.now() - started) / 1000);
    pending.textContent = s < 45
      ? `Running the agent… ${s} s`
      : `Still running… ${s} s. Multi-step tasks and provider rate-limit waits can take a few minutes.`;
  };
  tick();
  const timer = setInterval(tick, 1000);
  traceEl.innerHTML = `<p class="muted empty">Running… the trace appears when this turn finishes.</p>`;

  try {
    let response;
    try {
      response = await fetch("/chat", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify({
          message: question, history, fresh: document.getElementById("fresh").checked,
        }),
      });
    } catch {
      throw new Error("the service could not be reached. If it was asleep, wait a minute and try again.");
    }

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(typeof detail.detail === "string" ? detail.detail : `request failed (HTTP ${response.status})`);
    }

    const trace = await response.json();
    pending.remove();
    addMessage("assistant", trace.answer || "(no answer returned)", {
      citations: trace.citations || [],
      sources: trace.sources || [],
      html: trace.answer_html,
      cached: trace.cached,
    });
    renderTrace(trace);
    for (const action of trace.pending_actions || []) renderAction(action);

    history.push({ role: "user", content: question });
    history.push({ role: "assistant", content: trace.answer || "" });
    history = history.slice(-8);
  } catch (err) {
    pending.remove();
    addMessage("error", `Could not complete that request: ${err.message}`);
    traceEl.innerHTML = `<p class="muted empty">No trace: the request did not complete.</p>`;
  } finally {
    clearInterval(timer);
    setBusy(false);
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (question) send(question);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("button.demo").forEach((button) => {
  button.addEventListener("click", () => {
    if (busy) return;
    // Each demo task is a self-contained scenario, so it starts a fresh
    // conversation; the divider makes that reset visible when there is
    // earlier conversation to separate it from.
    history = [];
    if (messagesEl.querySelector(".msg:not(.welcome)")) {
      addDivider(`New conversation · ${button.textContent.trim()}`);
    }
    send(button.dataset.q);
  });
});

document.getElementById("new-chat").addEventListener("click", () => {
  if (busy) return;
  history = [];
  messagesEl.replaceChildren(...(welcome ? [welcome.cloneNode(true)] : []));
  traceEl.innerHTML = `<p class="muted empty">New conversation started. No model call was made.</p>`;
  input.focus();
});

// ---------------------------------------------------------------------------
// Mock actions: preview, then a separate confirmation with no model call
// ---------------------------------------------------------------------------

function renderAction(action) {
  const preview = action.preview || {};
  const isTicket = action.tool === "create_hr_ticket";
  const isEmail = action.tool === "draft_hr_email";
  const card = document.createElement("div");
  card.className = "msg assistant action-card";

  const title = document.createElement("p");
  title.className = "action-title";
  title.textContent = isTicket
    ? "Review the mock HR ticket"
    : isEmail ? "Review the mock email draft" : `Review the mock action: ${action.tool}`;

  const fields = document.createElement("dl");
  fields.className = "fields";
  const order = PREVIEW_ORDER[action.tool] || [];
  const keys = [
    ...order.filter((k) => k in preview),
    ...Object.keys(preview).filter(
      (k) => !order.includes(k) && !(k === "employee_id" && "employee_name" in preview)
    ),
  ];
  for (const key of keys) {
    let value = preview[key];
    if (key === "employee_name" && preview.employee_id) value = `${value} (${preview.employee_id})`;
    if (key === "category") value = CATEGORY_LABELS[value] || value;
    if (key === "priority" && typeof value === "string") value = value.charAt(0).toUpperCase() + value.slice(1);
    const dt = document.createElement("dt");
    dt.textContent = PREVIEW_LABELS[key] || key.replace(/_/g, " ");
    const dd = document.createElement("dd");
    dd.textContent = typeof value === "object" ? JSON.stringify(value) : String(value);
    if (key === "body") dd.className = "body-text";
    fields.append(dt, dd);
  }

  const note = document.createElement("p");
  note.className = "muted";
  note.textContent =
    (isEmail
      ? "Mock draft, held in memory only; no email is sent. "
      : "Mock action, held in memory only; nothing reaches a real HR system. ") +
    "Confirming runs this exact preview through the MCP server, without a model call.";

  const confirm = document.createElement("button");
  confirm.type = "button";
  confirm.className = "primary";
  confirm.textContent = isTicket ? "Create mock ticket" : isEmail ? "Save mock draft" : "Confirm mock action";
  const discard = document.createElement("button");
  discard.type = "button";
  discard.textContent = "Discard";
  const state = document.createElement("span");
  state.className = "muted";
  const close = (message) => {
    confirm.dataset.completed = discard.dataset.completed = "true";
    confirm.disabled = discard.disabled = true;
    state.textContent = message;
  };

  discard.addEventListener("click", () => {
    if (busy || discard.dataset.completed === "true") return;
    close("Discarded. Nothing was created.");
  });

  confirm.addEventListener("click", async () => {
    if (busy || confirm.dataset.completed === "true") return;
    setBusy(true);
    try {
      let response;
      try {
        response = await fetch(`/actions/${encodeURIComponent(action.id)}/confirm`, {
          method: "POST", headers: headers(),
        });
      } catch {
        throw Object.assign(new Error("the service could not be reached; try again."), { retry: true });
      }
      const result = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = typeof result.detail === "string"
          ? result.detail
          : result.detail?.error || (result.detail ? JSON.stringify(result.detail) : `HTTP ${response.status}`);
        // A missing access code leaves the preview valid, so allow a retry.
        throw Object.assign(new Error(detail), { retry: response.status === 403 });
      }
      close("Confirmed.");
      addConfirmation(action.tool, result);
      history.push({ role: "assistant", content: `Confirmed mock action result: ${JSON.stringify(result.result)}` });
      history = history.slice(-8);
    } catch (error) {
      if (!error.retry) close("Not created. Ask again for a fresh preview.");
      addMessage("error", `Confirmation failed: ${error.message}`);
    } finally {
      setBusy(false);
    }
  });

  const actions = document.createElement("div");
  actions.className = "actions";
  actions.append(confirm, discard, state);
  card.append(title, fields, note, actions);
  messagesEl.appendChild(card);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addConfirmation(tool, payload) {
  const r = payload.result && typeof payload.result === "object" ? payload.result : {};
  let text = "Mock action completed.";
  if (tool === "create_hr_ticket" && r.ticket_id) {
    text = `Mock ticket ${r.ticket_id} created for ${r.employee_name}: ${r.status}, ` +
      `${r.priority} priority, assigned to ${r.assigned_team}.`;
  } else if (tool === "draft_hr_email" && r.draft_id) {
    text = `Mock email draft ${r.draft_id} saved for ${r.to_name || r.to}. It was not sent.`;
  }
  const el = addMessage("assistant", `${text} Model calls for this confirmation: ${payload.api_calls ?? 0}.`);
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "Full result (JSON)";
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(payload, null, 2);
  details.append(summary, pre);
  el.appendChild(details);

  // The confirmation is a real MCP call too, so it belongs in the trace.
  traceEl.insertAdjacentHTML("beforeend", `
    <div class="step confirm">
      <div class="head"><span>Confirmation · executed via MCP</span><span class="muted">${plural(payload.api_calls ?? 0, "model call")}</span></div>
      <div class="body">${renderToolCall({
        name: payload.tool || tool, arguments: payload.arguments, result: payload.result,
        is_error: false, latency_ms: 0,
      })}</div>
    </div>`);
}

// ---------------------------------------------------------------------------
// Service status (no model call)
// ---------------------------------------------------------------------------

(async function pollHealth() {
  const link = document.getElementById("status");
  const dot = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  try {
    const response = await fetch("/health");
    const health = await response.json();
    const ok = health.status === "ok";
    const llm = health.llm || {};
    const providers = llm.providers_configured || [];
    const model = !llm.enabled
      ? "model calls disabled"
      : providers.length
        ? `LLM ${providers.includes(llm.primary) ? llm.primary : providers[0]}`
        : "no LLM key";
    const sha = health.build_sha && health.build_sha !== "local" ? `build ${health.build_sha.slice(0, 7)}` : "";
    dot.className = `dot ${ok ? "ok" : "warn"}`;
    text.textContent = [
      ok ? "Ready" : "Degraded",
      health.mcp?.connected ? `MCP connected, ${health.mcp.tool_count} tools` : "MCP not connected",
      health.rag_index?.ok ? `${health.rag_index.chunks} indexed chunks` : "index unavailable",
      model,
      sha,
    ].filter(Boolean).join(" · ");
    const q = health.quota_limits || {};
    link.title =
      `Model-call limits: ${q.per_turn} per turn, ${q.per_hour} per hour, ${q.per_day} per day. ` +
      `Data snapshot: ${health.as_of_date}. Click to open the /health JSON.`;
  } catch {
    dot.className = "dot err";
    text.textContent = "Service unreachable. If it was asleep, wait a minute and reload.";
  }
})();
