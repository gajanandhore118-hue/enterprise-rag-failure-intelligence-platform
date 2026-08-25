const state = {
  conversation: []
};

const chat = document.getElementById("chat");
const form = document.getElementById("chatForm");
const question = document.getElementById("question");
const productFamily = document.getElementById("productFamily");
const searchScope = document.getElementById("searchScope");
const sources = document.getElementById("sources");
const intentBox = document.getElementById("intentBox");
const queryBox = document.getElementById("queryBox");
const statusBox = document.getElementById("status");
const clearBtn = document.getElementById("clearBtn");
const sendBtn = document.getElementById("sendBtn");


function addMessage(role, content) {
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;

  const roleEl = document.createElement("div");
  roleEl.className = "role";
  roleEl.textContent = role;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;

  wrap.appendChild(roleEl);
  wrap.appendChild(bubble);

  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
}


function renderSources(items) {
  sources.innerHTML = "";

  if (!items || items.length === 0) {
    sources.textContent = "No sources retrieved.";
    return;
  }

  items.forEach((item, idx) => {
    const card = document.createElement("div");
    card.className = "source-card";

    const title = document.createElement("div");
    title.className = "source-title";
    title.textContent = `[S${idx + 1}] ${item.document_id || item.source_file || "Source"}`;

    const meta = document.createElement("div");
    meta.className = "source-meta";
    meta.textContent =
      `${item.document_type || "unknown"} | ` +
      `Rev: ${item.revision || "—"} | ` +
      `Page: ${item.page_number ?? "—"} | ` +
      `Score: ${item.score != null ? item.score.toFixed(3) : "—"}`;

    const excerpt = document.createElement("div");
    excerpt.className = "source-excerpt";
    excerpt.textContent = item.excerpt || "";

    card.appendChild(title);
    card.appendChild(meta);
    card.appendChild(excerpt);
    sources.appendChild(card);
  });
}


async function loadProductFamilies() {
  const res = await fetch("/api/product-families");
  const data = await res.json();

  productFamily.innerHTML = "";
  data.items.forEach((item) => {
    const opt = document.createElement("option");
    opt.value = item;
    opt.textContent = item;
    productFamily.appendChild(opt);
  });
}


async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error();
    statusBox.textContent = "API online";
  } catch {
    statusBox.textContent = "API unavailable";
  }
}


form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const q = question.value.trim();
  if (!q) return;

  addMessage("user", q);

  const previousConversation = [...state.conversation];
  state.conversation.push({ role: "user", content: q });

  question.value = "";
  sendBtn.disabled = true;
  sendBtn.textContent = "Thinking...";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        product_family: productFamily.value,
        search_scope: searchScope.value,
        question: q,
        conversation: previousConversation
      })
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Request failed");
    }

    addMessage("assistant", data.answer);
    state.conversation.push({ role: "assistant", content: data.answer });

    intentBox.textContent = `Intent: ${data.intent}`;
    queryBox.textContent = `Standalone query: ${data.contextualized_query}`;
    renderSources(data.sources);

  } catch (err) {
    addMessage("assistant", `Error: ${err.message}`);
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = "Ask";
  }
});


clearBtn.addEventListener("click", () => {
  state.conversation = [];
  chat.innerHTML = "";
  sources.innerHTML = "";
  intentBox.textContent = "Intent: —";
  queryBox.textContent = "Standalone query: —";
  addMessage(
    "assistant",
    "Conversation cleared. Select a product family and start a new engineering investigation."
  );
});


(async function init() {
  await Promise.all([loadProductFamilies(), checkHealth()]);
  addMessage(
    "assistant",
    "Select a product family and ask about an engineering change, historical validation, supplier issue, or product failure."
  );
})();
