// Chat client. Keeps the conversation history client-side and renders the
// execution trace the API returns, so the tool calls behind an answer are
// visible rather than implied.

const messagesEl = document.getElementById("messages");
const traceEl = document.getElementById("trace");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");

// Sent back with each turn so follow-ups ("yes, file it") keep their context.
// Only the plain user/assistant turns are kept; tool traffic is reconstructed
// server-side each turn and would bloat the payload for no benefit.
let history = [];
let busy = false;

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

function addMessage(role, text, citations, sources, answerHtml) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = answerHtml || escape(text)
    .split(/\n{2,}/)
    .map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`)
    .join("");
  if (citations && citations.length) {
    // The label says where a claim came from; the snippet shows what was
    // there. Collapsed by default so the answer stays readable, but present,
    // so a citation can be checked without leaving the page.
    const byLabel = new Map((sources || []).map((s) => [s.citation, s]));
    el.innerHTML +=
      `<div class="cites">` +
      citations
        .map((c) => {
          const source = byLabel.get(c);
          if (!source) return `<span class="cite">${escape(c)}</span>`;
          return `<details class="cite"><summary>${escape(c)}</summary>
            <p class="snippet">${escape(source.snippet)}</p></details>`;
        })
        .join("") +
      `</div>`;
  }
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function renderTrace(trace) {
  const tags = [];
  tags.push(`<span class="tag">${escape(trace.provider || "no provider")}</span>`);
  if (trace.model) tags.push(`<span class="tag">${escape(trace.model)}</span>`);
  tags.push(`<span class="tag">${trace.steps.length} step(s)</span>`);
  tags.push(`<span class="tag">${Math.round(trace.total_ms)} ms</span>`);
  tags.push(`<span class="tag">${trace.api_calls ?? 0} API calls this request</span>`);
  if (trace.cached) tags.push(`<span class="tag warn">Cached result: trace and timings are from the original run</span>`);
  if (trace.grounded === true) tags.push(`<span class="tag ok">retrieval accepted</span>`);
  if (trace.grounded === false) tags.push(`<span class="tag warn">not grounded</span>`);
  if (trace.fell_back) tags.push(`<span class="tag warn">provider fallback</span>`);
  if (trace.truncated) tags.push(`<span class="tag warn">step limit reached</span>`);
  if (trace.error) tags.push(`<span class="tag err">error</span>`);

  const steps = trace.steps
    .map((step) => {
      const calls = (step.tool_calls || [])
        .map(
          (call) => `
        <div class="tool ${call.is_error ? "err" : ""}">
          <span class="name">${escape(call.name)}</span>
          <span class="muted"> ${Math.round(call.latency_ms)} ms${call.is_error ? " &middot; error" : ""}</span>
          <details>
            <summary>arguments &amp; result</summary>
            <pre>${escape(JSON.stringify(call.arguments, null, 1))}</pre>
            <pre>${escape(JSON.stringify(call.result, null, 1))}</pre>
          </details>
        </div>`
        )
        .join("");

      // The summary leads, then the per-tool detail. It is an operational line
      // -- which tools were selected -- never the model's own narration.
      const body =
        (step.summary ? `<p class="muted">${escape(step.summary)}</p>` : "") +
        (calls || (step.summary ? "" : `<p class="muted">${escape(step.kind)}</p>`));

      return `
      <div class="step">
        <div class="head">
          <span>step ${step.index} &middot; ${escape(step.kind)}</span>
          <span>${step.latency_ms ? Math.round(step.latency_ms) + " ms" : ""}</span>
        </div>
        <div class="body">${body}</div>
      </div>`;
    })
    .join("");

  traceEl.innerHTML = `<div class="summary-row">${tags.join("")}</div>${steps}`;
}

async function send(question) {
  if (busy) return;
  setBusy(true);
  addMessage("user", question);
  const pending = addMessage("assistant", "Working");
  pending.classList.add("typing");
  input.value = "";

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: headers(),
      body: JSON.stringify({
        message: question, history, fresh: document.getElementById("fresh").checked,
      }),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `request failed (${response.status})`);
    }

    const trace = await response.json();
    pending.remove();
    addMessage("assistant", trace.answer || "(no answer returned)", trace.citations, trace.sources, trace.answer_html);
    renderTrace(trace);
    for (const action of trace.pending_actions || []) renderAction(action);

    history.push({ role: "user", content: question });
    history.push({ role: "assistant", content: trace.answer || "" });
    history = history.slice(-8);
  } catch (err) {
    pending.remove();
    addMessage("error", `Could not complete that request: ${err.message}`);
  } finally {
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
    history = [];
    send(button.dataset.q);
  });
});

document.getElementById("new-chat").addEventListener("click", () => {
  if (busy) return;
  history = [];
  messagesEl.replaceChildren();
  traceEl.textContent = "New conversation. No model call made.";
});

function renderAction(action) {
  const card = document.createElement("div");
  card.className = "msg assistant";
  const title = document.createElement("p");
  title.textContent = `Mock action preview: ${action.tool}`;
  const preview = document.createElement("pre");
  preview.textContent = JSON.stringify(action.preview, null, 2);
  const button = document.createElement("button");
  button.textContent = "Confirm this exact mock action (no LLM call)";
  button.addEventListener("click", async () => {
    if (busy || button.dataset.completed === "true") return;
    setBusy(true);
    try {
      const response = await fetch(`/actions/${encodeURIComponent(action.id)}/confirm`, {
        method: "POST", headers: headers(),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(JSON.stringify(result.detail || result));
      button.dataset.completed = "true";
      button.textContent = "Mock action completed";
      addMessage("assistant", JSON.stringify(result, null, 2));
      history.push({ role: "assistant", content: `Confirmed mock action result: ${JSON.stringify(result.result)}` });
      history = history.slice(-8);
    } catch (error) {
      button.dataset.completed = "true";
      button.textContent = "Confirmation failed; request a new preview";
      addMessage("error", error.message);
    } finally {
      setBusy(false);
    }
  });
  card.append(title, preview, button);
  messagesEl.appendChild(card);
}

(async function pollHealth() {
  const dot = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  try {
    const response = await fetch("/health");
    const health = await response.json();
    dot.className = `dot ${health.status === "ok" ? "ok" : "degraded"}`;
    const bits = [
      `${health.mcp.tool_count} MCP tools`,
      `${health.rag_index.chunks ?? "?"} chunks`,
      health.llm.providers_configured.length
        ? health.llm.providers_configured.join(" / ")
        : "no LLM key",
    ];
    text.textContent = `${health.status} &middot; ${bits.join(" · ")}`.replace("&middot;", "·");
  } catch {
    dot.className = "dot down";
    text.textContent = "unreachable";
  }
})();
