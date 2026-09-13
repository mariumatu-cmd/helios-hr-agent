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

const escape = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );

function addMessage(role, text, citations) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = escape(text)
    .split(/\n{2,}/)
    .map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`)
    .join("");
  if (citations && citations.length) {
    el.innerHTML +=
      `<div class="cites">` +
      citations.map((c) => `<span class="cite">${escape(c)}</span>`).join("") +
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
  if (trace.grounded === true) tags.push(`<span class="tag ok">grounded</span>`);
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
            <pre>${escape(JSON.stringify(call.result, null, 1).slice(0, 4000))}</pre>
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
  addMessage("user", question);
  const pending = addMessage("assistant", "Working");
  pending.classList.add("typing");
  sendBtn.disabled = true;
  input.value = "";

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question, history }),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `request failed (${response.status})`);
    }

    const trace = await response.json();
    pending.remove();
    addMessage("assistant", trace.answer || "(no answer returned)", trace.citations);
    renderTrace(trace);

    history.push({ role: "user", content: question });
    history.push({ role: "assistant", content: trace.answer || "" });
    history = history.slice(-8);
  } catch (err) {
    pending.remove();
    addMessage("error", `Could not complete that request: ${err.message}`);
  } finally {
    sendBtn.disabled = false;
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
  button.addEventListener("click", () => send(button.dataset.q));
});

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
