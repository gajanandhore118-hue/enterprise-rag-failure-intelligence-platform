const state = { conversation: [] };

const chat          = document.getElementById("chat");
const form          = document.getElementById("chatForm");
const question      = document.getElementById("question");
const productFamily = document.getElementById("productFamily");
const searchScope   = document.getElementById("searchScope");
const sources       = document.getElementById("sources");
const intentVal     = document.getElementById("intentVal");
const queryVal      = document.getElementById("queryVal");
const statusBox     = document.getElementById("status");
const statusDot     = document.getElementById("statusDot");
const clearBtn      = document.getElementById("clearBtn");
const sendBtn       = document.getElementById("sendBtn");
const sendLabel     = document.getElementById("sendLabel");


// ── Helpers ───────────────────────────────────────────────────────────────────

function addMessage(role, content, isError = false) {
  const wrap = document.createElement("div");
  wrap.className = `message ${role}${isError ? " error" : ""}`;

  const label = document.createElement("div");
  label.className = "role-label";
  label.textContent = role === "user" ? "You" : "Assistant";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;

  wrap.appendChild(label);
  wrap.appendChild(bubble);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  return wrap;
}

function showTyping() {
  const wrap = document.createElement("div");
  wrap.className = "message assistant";
  wrap.id = "typing";

  const label = document.createElement("div");
  label.className = "role-label";
  label.textContent = "Assistant";

  const bubble = document.createElement("div");
  bubble.className = "bubble typing-dots";
  bubble.innerHTML = "<span></span><span></span><span></span>";

  wrap.appendChild(label);
  wrap.appendChild(bubble);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
}

function removeTyping() {
  document.getElementById("typing")?.remove();
}

function renderSources(items) {
  sources.innerHTML = "";
  if (!items?.length) {
    sources.innerHTML = `<div style="color:var(--muted);font-size:13px;padding:8px 0">No sources retrieved.</div>`;
    return;
  }
  items.forEach((item, idx) => {
    const card = document.createElement("div");
    card.className = "source-card";

    const score = item.score != null ? item.score.toFixed(3) : "—";
    const docId = item.document_id || item.source_file || "Source";

    card.innerHTML = `
      <div class="source-header">
        <span class="source-badge">S${idx + 1}</span>
        <span class="source-title">${docId}</span>
        <span class="source-score">${score}</span>
      </div>
      <div class="source-meta">
        <span>${item.document_type || "unknown"}</span>
        ${item.revision ? `<span>Rev ${item.revision}</span>` : ""}
        ${item.page_number != null ? `<span>Page ${item.page_number}</span>` : ""}
      </div>
      ${item.excerpt ? `<div class="source-excerpt">${item.excerpt}</div>` : ""}
    `;
    sources.appendChild(card);
  });
}

function setStatus(online) {
  statusDot.className = `status-dot ${online ? "online" : "offline"}`;
  statusBox.textContent = online ? "API online" : "API unavailable";
}

// ── Init ──────────────────────────────────────────────────────────────────────

async function loadProductFamilies() {
  const res  = await fetch("/api/product-families");
  const data = await res.json();
  productFamily.innerHTML = "";
  data.items.forEach(item => {
    const opt = document.createElement("option");
    opt.value = opt.textContent = item;
    productFamily.appendChild(opt);
  });
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    setStatus(res.ok);
  } catch {
    setStatus(false);
  }
}

// ── Demo questions ────────────────────────────────────────────────────────────

document.querySelectorAll(".demo-q").forEach(el => {
  el.addEventListener("click", () => {
    question.value = el.dataset.q;
    question.focus();
  });
});

// ── Submit ────────────────────────────────────────────────────────────────────

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = question.value.trim();
  if (!q) return;

  addMessage("user", q);
  const prevConversation = [...state.conversation];
  state.conversation.push({ role: "user", content: q });
  question.value = "";

  sendBtn.disabled = true;
  sendLabel.textContent = "Thinking…";
  showTyping();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        product_family: productFamily.value,
        search_scope:   searchScope.value,
        question:       q,
        conversation:   prevConversation,
      }),
    });

    const data = await res.json();
    removeTyping();

    if (!res.ok) throw new Error(data.detail || "Request failed");

    addMessage("assistant", data.answer);
    state.conversation.push({ role: "assistant", content: data.answer });

    intentVal.textContent = data.intent.replaceAll("_", " ");
    queryVal.textContent  = data.contextualized_query;
    renderSources(data.sources);

  } catch (err) {
    removeTyping();
    addMessage("assistant", `Error: ${err.message}`, true);
  } finally {
    sendBtn.disabled = false;
    sendLabel.textContent = "Send";
  }
});

// ── Clear ─────────────────────────────────────────────────────────────────────

clearBtn.addEventListener("click", () => {
  state.conversation = [];
  chat.innerHTML = "";
  sources.innerHTML = "";
  intentVal.textContent = "—";
  queryVal.textContent  = "—";
  addMessage("assistant", "Conversation cleared. Select a product family and start a new investigation.");
});

// ── Auto-resize textarea ──────────────────────────────────────────────────────

question.addEventListener("input", () => {
  question.style.height = "auto";
  question.style.height = question.scrollHeight + "px";
});

question.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

// ── Boot ──────────────────────────────────────────────────────────────────────

(async function init() {
  await Promise.all([loadProductFamilies(), checkHealth()]);
  addMessage("assistant", "Select a product family and ask about an engineering change, historical validation, supplier issue, or product failure.\n\nTip: click a demo question in the sidebar to get started.");
})();
