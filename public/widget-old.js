(function () {
  "use strict";

  const config = window.chatbotConfig || {};
  const agentId      = config.agentId      || "";
  const primaryColor = config.primaryColor || "#0f172a";
  const appearance   = config.appearance   || "light";
  const apiBase      = config.apiBase      || "http://127.0.0.1:8001";
  const position     = config.position     || "bottom-right"; // bottom-right | bottom-left

  if (!agentId) {
    console.warn("[ChatWidget] agentId is required in window.chatbotConfig");
    return;
  }

  // ── Styles ──────────────────────────────────────────────────────────────────
  const isDark        = appearance === "dark";
  const bgColor       = isDark ? "#0f172a" : "#ffffff";
  const textColor     = isDark ? "#f1f5f9" : "#1e293b";
  const bubbleBg      = isDark ? "#1e293b" : "#f1f5f9";
  const bubbleBorder  = isDark ? "#334155" : "#e2e8f0";
  const inputBg       = isDark ? "#1e293b" : "#f8fafc";
  const inputBorder   = isDark ? "#334155" : "#e2e8f0";
  const subText       = isDark ? "#94a3b8" : "#64748b";

  const posRight = position === "bottom-right";

  const css = `
    #cw-bubble {
      position: fixed;
      ${posRight ? "right: 24px" : "left: 24px"};
      bottom: 24px;
      width: 52px;
      height: 52px;
      border-radius: 50%;
      background: ${primaryColor};
      border: none;
      cursor: pointer;
      box-shadow: 0 4px 14px rgba(0,0,0,.25);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 99998;
      transition: transform .2s, box-shadow .2s;
    }
    #cw-bubble:hover { transform: scale(1.08); box-shadow: 0 6px 20px rgba(0,0,0,.3); }
    #cw-bubble svg   { width: 24px; height: 24px; color: #fff; }

    #cw-panel {
      position: fixed;
      ${posRight ? "right: 24px" : "left: 24px"};
      bottom: 90px;
      width: 370px;
      height: 560px;
      max-height: calc(100vh - 110px);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 8px 32px rgba(0,0,0,.18);
      display: flex;
      flex-direction: column;
      z-index: 99999;
      background: ${bgColor};
      border: 1px solid ${bubbleBorder};
      transform: scale(.92) translateY(12px);
      opacity: 0;
      pointer-events: none;
      transition: transform .22s cubic-bezier(.34,1.56,.64,1), opacity .18s;
    }
    #cw-panel.open {
      transform: scale(1) translateY(0);
      opacity: 1;
      pointer-events: all;
    }

    #cw-header {
      padding: 14px 16px;
      background: ${primaryColor};
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
    }
    #cw-header-left { display: flex; align-items: center; gap: 10px; }
    #cw-avatar {
      width: 32px; height: 32px; border-radius: 50%;
      background: rgba(255,255,255,.2);
      display: flex; align-items: center; justify-content: center;
    }
    #cw-avatar svg { width: 18px; height: 18px; color: #fff; }
    #cw-title { font-size: 14px; font-weight: 600; color: #fff; font-family: system-ui, sans-serif; }
    #cw-close {
      background: none; border: none; cursor: pointer;
      color: rgba(255,255,255,.7); padding: 4px;
      border-radius: 6px; display: flex;
      transition: background .15s;
    }
    #cw-close:hover { background: rgba(255,255,255,.15); color: #fff; }
    #cw-close svg { width: 16px; height: 16px; }

    #cw-messages {
      flex: 1;
      overflow-y: auto;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      background: ${isDark ? "#0f172a" : "#f8fafc"};
      background-image: radial-gradient(circle, ${isDark ? "rgba(148,163,184,.1)" : "rgba(203,213,225,.5)"} 1px, transparent 1px);
      background-size: 20px 20px;
    }
    #cw-messages::-webkit-scrollbar { width: 4px; }
    #cw-messages::-webkit-scrollbar-track { background: transparent; }
    #cw-messages::-webkit-scrollbar-thumb { background: ${bubbleBorder}; border-radius: 4px; }

    .cw-msg { display: flex; max-width: 84%; }
    .cw-msg.user  { align-self: flex-end; justify-content: flex-end; }
    .cw-msg.bot   { align-self: flex-start; }
    .cw-bubble-text {
      padding: 9px 13px;
      font-size: 13px;
      line-height: 1.55;
      font-family: system-ui, sans-serif;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .cw-msg.user .cw-bubble-text {
      background: ${primaryColor};
      color: #fff;
      border-radius: 16px 16px 4px 16px;
    }
    .cw-msg.bot .cw-bubble-text {
      background: ${bubbleBg};
      color: ${textColor};
      border: 1px solid ${bubbleBorder};
      border-radius: 4px 16px 16px 16px;
    }

    .cw-typing {
      display: flex; align-items: center; gap: 4px;
      padding: 10px 14px;
      background: ${bubbleBg};
      border: 1px solid ${bubbleBorder};
      border-radius: 4px 16px 16px 16px;
      align-self: flex-start;
    }
    .cw-dot {
      width: 6px; height: 6px; border-radius: 50%;
      background: ${subText};
      animation: cw-bounce .9s infinite;
    }
    .cw-dot:nth-child(2) { animation-delay: .15s; }
    .cw-dot:nth-child(3) { animation-delay: .3s; }
    @keyframes cw-bounce {
      0%,60%,100% { transform: translateY(0); }
      30% { transform: translateY(-5px); }
    }

    #cw-powered {
      text-align: center;
      font-size: 10px;
      color: ${subText};
      padding: 4px 0;
      background: ${bgColor};
      border-top: 1px solid ${bubbleBorder};
      font-family: system-ui, sans-serif;
    }

    #cw-input-row {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 10px 12px;
      background: ${bgColor};
      border-top: 1px solid ${bubbleBorder};
      flex-shrink: 0;
    }
    #cw-input {
      flex: 1;
      border: 1px solid ${inputBorder};
      background: ${inputBg};
      color: ${textColor};
      border-radius: 20px;
      padding: 8px 14px;
      font-size: 13px;
      font-family: system-ui, sans-serif;
      outline: none;
      transition: border-color .15s;
    }
    #cw-input:focus { border-color: ${primaryColor}; }
    #cw-input::placeholder { color: ${subText}; }
    #cw-send {
      width: 34px; height: 34px; border-radius: 50%;
      background: ${primaryColor};
      border: none; cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      flex-shrink: 0;
      transition: opacity .15s;
    }
    #cw-send:disabled { opacity: .4; cursor: not-allowed; }
    #cw-send svg { width: 16px; height: 16px; color: #fff; }

    @media (max-width: 420px) {
      #cw-panel {
        width: calc(100vw - 24px);
        ${posRight ? "right: 12px" : "left: 12px"};
        bottom: 80px;
      }
      #cw-bubble { ${posRight ? "right: 16px" : "left: 16px"}; bottom: 16px; }
    }
  `;

  // ── Inject styles ────────────────────────────────────────────────────────────
  const style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  // ── Build DOM ────────────────────────────────────────────────────────────────
  // Bubble button
  const bubble = document.createElement("button");
  bubble.id = "cw-bubble";
  bubble.setAttribute("aria-label", "Open chat");
  bubble.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`;

  // Panel
  const panel = document.createElement("div");
  panel.id = "cw-panel";
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-modal", "true");
  panel.setAttribute("aria-label", "Chat");

  panel.innerHTML = `
    <div id="cw-header">
      <div id="cw-header-left">
        <div id="cw-avatar">
          <svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 2a8 8 0 1 0 0 16A8 8 0 0 0 10 2Zm3.707 6.293a1 1 0 0 0-1.414 0L9 11.586 7.707 10.293a1 1 0 0 0-1.414 1.414l2 2a1 1 0 0 0 1.414 0l4-4a1 1 0 0 0 0-1.414Z" clip-rule="evenodd"/></svg>
        </div>
        <span id="cw-title">Assistant</span>
      </div>
      <button id="cw-close" aria-label="Close chat">
        <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>
      </button>
    </div>
    <div id="cw-messages"></div>
    <div id="cw-powered">Powered by your AI assistant</div>
    <div id="cw-input-row">
      <input id="cw-input" type="text" placeholder="Message..." autocomplete="off" />
      <button id="cw-send" disabled aria-label="Send">
        <svg viewBox="0 0 16 16" fill="currentColor"><path fill-rule="evenodd" d="M8 2a.75.75 0 0 1 .75.75v8.69l3.22-3.22a.75.75 0 1 1 1.06 1.06l-4.5 4.5a.75.75 0 0 1-1.06 0l-4.5-4.5a.75.75 0 0 1 1.06-1.06l3.22 3.22V2.75A.75.75 0 0 1 8 2Z" clip-rule="evenodd" transform="rotate(-90 8 8)"/></svg>
      </button>
    </div>
  `;

  document.body.appendChild(bubble);
  document.body.appendChild(panel);

  // ── State ────────────────────────────────────────────────────────────────────
  let isOpen         = false;
  let isLoading      = false;
  let conversationId = null;

  const messagesEl = document.getElementById("cw-messages");
  const inputEl    = document.getElementById("cw-input");
  const sendEl     = document.getElementById("cw-send");

  // ── Helpers ──────────────────────────────────────────────────────────────────
  function addMessage(role, text) {
    const wrap = document.createElement("div");
    wrap.className = "cw-msg " + (role === "user" ? "user" : "bot");
    const bub = document.createElement("div");
    bub.className = "cw-bubble-text";
    bub.textContent = text;
    wrap.appendChild(bub);
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return wrap;
  }

  function showTyping() {
    const wrap = document.createElement("div");
    wrap.className = "cw-typing";
    wrap.id = "cw-typing";
    wrap.innerHTML = `<div class="cw-dot"></div><div class="cw-dot"></div><div class="cw-dot"></div>`;
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function removeTyping() {
    const el = document.getElementById("cw-typing");
    if (el) el.remove();
  }

  async function fetchSettings() {
    // Uses public endpoint — no JWT required, safe for any website visitor
    try {
      const res = await fetch(`${apiBase}/widget/settings/${agentId}`);
      if (res.ok) {
        const s = await res.json();
        const titleEl = document.getElementById("cw-title");
        if (titleEl && s.display_name) titleEl.textContent = s.display_name;
        addMessage("bot", s.welcome_message || "Hi! What can I help you with?");
      } else {
        addMessage("bot", "Hi! What can I help you with?");
      }
    } catch {
      addMessage("bot", "Hi! What can I help you with?");
    }
  }

  async function sendMessage(text) {
    if (!text.trim() || isLoading) return;
    isLoading = true;
    sendEl.disabled = true;
    inputEl.disabled = true;

    addMessage("user", text);
    inputEl.value = "";
    showTyping();

    try {
      const res = await fetch(`${apiBase}/widget/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          agent_id: agentId,
          question: text,
          conversation_id: conversationId,
        }),
      });

      removeTyping();

      if (res.ok) {
        const data = await res.json();
        conversationId = data.conversation_id;
        addMessage("bot", data.answer || "I don't have enough information to answer that.");
      } else {
        addMessage("bot", "Sorry, I'm having trouble connecting. Please try again.");
      }
    } catch {
      removeTyping();
      addMessage("bot", "Sorry, I'm having trouble connecting. Please try again.");
    } finally {
      isLoading = false;
      sendEl.disabled = false;
      inputEl.disabled = false;
      inputEl.focus();
    }
  }

  // ── Events ───────────────────────────────────────────────────────────────────
  bubble.addEventListener("click", () => {
    isOpen = !isOpen;
    panel.classList.toggle("open", isOpen);
    bubble.setAttribute("aria-expanded", String(isOpen));
    if (isOpen) {
      if (messagesEl.children.length === 0) fetchSettings();
      setTimeout(() => inputEl.focus(), 250);
    }
  });

  document.getElementById("cw-close").addEventListener("click", () => {
    isOpen = false;
    panel.classList.remove("open");
    bubble.setAttribute("aria-expanded", "false");
  });

  inputEl.addEventListener("input", () => {
    sendEl.disabled = !inputEl.value.trim() || isLoading;
  });

  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !isLoading) {
      e.preventDefault();
      sendMessage(inputEl.value.trim());
    }
  });

  sendEl.addEventListener("click", () => sendMessage(inputEl.value.trim()));

  // Close on outside click
  document.addEventListener("click", (e) => {
    if (isOpen && !panel.contains(e.target) && e.target !== bubble) {
      isOpen = false;
      panel.classList.remove("open");
    }
  });

  // Close on Escape
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && isOpen) {
      isOpen = false;
      panel.classList.remove("open");
    }
  });

})();
