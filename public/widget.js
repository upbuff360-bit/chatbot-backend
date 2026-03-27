(function () {
  "use strict";

  function init() {
  const config = window.chatbotConfig || {};
  const agentId      = config.agentId      || "";
  const primaryColor = config.primaryColor || "#0f172a";
  const appearance   = config.appearance   || "light";
  const apiBase      = config.apiBase      || "http://127.0.0.1:8001";
  const position     = config.position     || "bottom-right";
  const STORAGE_KEY  = "cw_history_" + agentId;

  if (!agentId) { console.warn("[ChatWidget] agentId is required"); return; }

  const isDark       = appearance === "dark";
  const bgColor      = isDark ? "#0f172a" : "#ffffff";
  const textColor    = isDark ? "#f1f5f9" : "#1e293b";
  const bubbleBg     = isDark ? "#1e293b" : "#f1f5f9";
  const bubbleBorder = isDark ? "#334155" : "#e2e8f0";
  const inputBg      = isDark ? "#1e293b" : "#f8fafc";
  const inputBorder  = isDark ? "#334155" : "#e2e8f0";
  const subText      = isDark ? "#94a3b8" : "#64748b";
  const dividerColor = isDark ? "#334155" : "#e2e8f0";
  const dividerText  = isDark ? "#64748b"  : "#94a3b8";
  const posRight     = position === "bottom-right";

  // ── CSS ───────────────────────────────────────────────────────────────────────
  const css = `
    #cw-bubble {
      position:fixed;${posRight?"right:24px":"left:24px"};bottom:24px;
      width:52px;height:52px;border-radius:50%;background:${primaryColor};
      border:none;cursor:pointer;box-shadow:0 4px 14px rgba(0,0,0,.25);
      display:flex;align-items:center;justify-content:center;
      z-index:2147483646;transition:transform .2s,box-shadow .2s;
    }
    #cw-bubble:hover{transform:scale(1.08);box-shadow:0 6px 20px rgba(0,0,0,.3);}
    #cw-bubble svg{width:24px;height:24px;color:#fff;}

    #cw-panel {
      position:fixed;${posRight?"right:24px":"left:24px"};bottom:90px;
      width:370px;height:560px;max-height:calc(100vh - 110px);
      border-radius:16px;overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,.18);
      display:none;flex-direction:column;z-index:2147483647;
      background:${bgColor};border:1px solid ${bubbleBorder};
    }
    #cw-panel, #cw-panel *{box-sizing:border-box;}
    #cw-panel.open{display:flex;}

    #cw-header {
      padding:14px 16px;background:${primaryColor};
      display:flex;align-items:center;justify-content:space-between;flex-shrink:0;
    }
    #cw-header-left{display:flex;align-items:center;gap:10px;}
    #cw-avatar{width:32px;height:32px;border-radius:50%;background:rgba(255,255,255,.2);display:flex;align-items:center;justify-content:center;}
    #cw-avatar svg{width:18px;height:18px;color:#fff;}
    #cw-title{font-size:14px;font-weight:600;color:#fff;font-family:system-ui,sans-serif;}
    #cw-header-right{display:flex;align-items:center;gap:4px;position:relative;}
    .cw-hbtn{background:none;border:none;cursor:pointer;color:rgba(255,255,255,.7);padding:4px;border-radius:6px;display:flex;transition:background .15s;}
    .cw-hbtn:hover{background:rgba(255,255,255,.15);color:#fff;}
    .cw-hbtn svg{width:16px;height:16px;}

    /* Kebab dropdown */
    #cw-menu {
      position:absolute;top:calc(100% + 6px);right:0;
      background:${bgColor};border:1px solid ${bubbleBorder};
      border-radius:10px;box-shadow:0 8px 24px rgba(0,0,0,.15);
      min-width:180px;overflow:hidden;z-index:10;
      display:none;flex-direction:column;
    }
    #cw-menu.open{display:flex;}
    .cw-menu-item {
      display:flex;align-items:center;gap:10px;
      padding:10px 14px;font-size:13px;font-family:system-ui,sans-serif;
      color:${textColor};background:none;border:none;cursor:pointer;
      text-align:left;transition:background .12s;
    }
    .cw-menu-item:hover{background:${bubbleBg};}
    .cw-menu-item.disabled{opacity:.4;cursor:not-allowed;pointer-events:none;}
    .cw-menu-item svg{width:14px;height:14px;flex-shrink:0;color:${subText};}
    .cw-menu-sep{height:1px;background:${bubbleBorder};margin:2px 0;}

    /* Messages pane */
    #cw-messages {
      flex:1;overflow-y:auto;padding:14px;
      display:flex;flex-direction:column;gap:10px;
      background:${isDark?"#0f172a":"#f8fafc"};
      background-image:radial-gradient(circle,${isDark?"rgba(148,163,184,.1)":"rgba(203,213,225,.5)"} 1px,transparent 1px);
      background-size:20px 20px;
    }
    #cw-messages::-webkit-scrollbar{width:4px;}
    #cw-messages::-webkit-scrollbar-thumb{background:${bubbleBorder};border-radius:4px;}

    .cw-msg{display:flex;max-width:84%;}
    .cw-msg.user{align-self:flex-end;justify-content:flex-end;}
    .cw-msg.bot{align-self:flex-start;}
    .cw-bubble-text{padding:9px 13px;font-size:13px;line-height:1.55;font-family:system-ui,sans-serif;white-space:pre-wrap;word-break:break-word;}
    .cw-msg.user .cw-bubble-text{background:${primaryColor};color:#fff;border-radius:16px 16px 4px 16px;}
    .cw-msg.bot  .cw-bubble-text{background:${bubbleBg};color:${textColor};border:1px solid ${bubbleBorder};border-radius:4px 16px 16px 16px;}

    .cw-typing{display:flex;align-items:center;gap:4px;padding:10px 14px;background:${bubbleBg};border:1px solid ${bubbleBorder};border-radius:4px 16px 16px 16px;align-self:flex-start;}
    .cw-dot{width:6px;height:6px;border-radius:50%;background:${subText};animation:cw-bounce .9s infinite;}
    .cw-dot:nth-child(2){animation-delay:.15s;}
    .cw-dot:nth-child(3){animation-delay:.3s;}
    @keyframes cw-bounce{0%,60%,100%{transform:translateY(0);}30%{transform:translateY(-5px);}}

    /* Recent chats pane */
    #cw-history{display:none;flex-direction:column;flex:1;overflow:hidden;}
    #cw-history.active{display:flex;}
    #cw-history-header{
      padding:12px 14px;font-size:12px;font-weight:600;color:${subText};
      font-family:system-ui,sans-serif;border-bottom:1px solid ${bubbleBorder};
      flex-shrink:0;display:flex;align-items:center;justify-content:space-between;
    }
    #cw-history-new{
      display:flex;align-items:center;gap:5px;background:${primaryColor};
      color:#fff;border:none;border-radius:8px;padding:5px 10px;
      font-size:11px;font-weight:600;font-family:system-ui,sans-serif;
      cursor:pointer;transition:opacity .15s;
    }
    #cw-history-new:hover{opacity:.85;}
    #cw-history-new svg{width:11px;height:11px;}
    #cw-history-list{flex:1;overflow-y:auto;}
    .cw-hist-item{padding:12px 14px;border-bottom:1px solid ${bubbleBorder};cursor:pointer;transition:background .12s;}
    .cw-hist-item:hover{background:${bubbleBg};}
    .cw-hist-title{font-size:13px;font-weight:500;color:${textColor};font-family:system-ui,sans-serif;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
    .cw-hist-meta{font-size:10px;color:${subText};font-family:system-ui,sans-serif;margin-top:2px;}
    .cw-hist-empty{padding:40px 14px;text-align:center;font-size:13px;color:${subText};font-family:system-ui,sans-serif;}

    #cw-powered{text-align:center;font-size:10px;color:${subText};padding:4px 0;background:${bgColor};border-top:1px solid ${bubbleBorder};font-family:system-ui,sans-serif;flex-shrink:0;}
    #cw-input-row{display:flex;align-items:center;gap:8px;padding:10px 12px;background:${bgColor};border-top:1px solid ${bubbleBorder};flex-shrink:0;}
    #cw-input{flex:1;border:1px solid ${inputBorder};background:${inputBg};color:${textColor};border-radius:20px;padding:8px 14px;font-size:13px;font-family:system-ui,sans-serif;outline:none;transition:border-color .15s;}
    #cw-input:focus{border-color:${primaryColor};}
    #cw-input::placeholder{color:${subText};}
    #cw-send{width:34px;height:34px;border-radius:50%;background:${primaryColor};border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:opacity .15s;}
    #cw-send:disabled{opacity:.4;cursor:not-allowed;}
    #cw-send svg{width:16px;height:16px;color:#fff;}

    /* Confirmation dialog */
    #cw-confirm-box{
      background:${bgColor};border-radius:12px;padding:20px;
      width:260px;box-shadow:0 8px 32px rgba(0,0,0,.2);
      font-family:system-ui,sans-serif;
    }
    #cw-confirm-box p{font-size:13px;color:${textColor};margin:0 0 6px;font-weight:600;}
    #cw-confirm-box span{font-size:12px;color:${subText};line-height:1.5;}
    #cw-confirm-btns{display:flex;gap:8px;margin-top:16px;}
    .cw-confirm-cancel{
      flex:1;padding:8px;border:1px solid ${bubbleBorder};
      border-radius:8px;background:none;color:${textColor};
      font-size:12px;font-weight:500;cursor:pointer;
      font-family:system-ui,sans-serif;transition:background .12s;
    }
    .cw-confirm-cancel:hover{background:${bubbleBg};}
    .cw-confirm-ok{
      flex:1;padding:8px;border:none;border-radius:8px;
      background:${primaryColor};color:#fff;
      font-size:12px;font-weight:600;cursor:pointer;
      font-family:system-ui,sans-serif;transition:opacity .12s;
    }
    .cw-confirm-ok:hover{opacity:.85;}

    @media(max-width:420px){
      #cw-panel{width:calc(100vw - 24px);${posRight?"right:12px":"left:12px"};bottom:80px;}
      #cw-bubble{${posRight?"right:16px":"left:16px"};bottom:16px;}
    }
  `;
  const style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  // ── DOM ───────────────────────────────────────────────────────────────────────
  const bubble = document.createElement("button");
  bubble.id = "cw-bubble";
  bubble.setAttribute("aria-label", "Open chat");
  bubble.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`;

  const panel = document.createElement("div");
  panel.id = "cw-panel";
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-modal", "true");
  panel.innerHTML = `
    <div id="cw-header">
      <div id="cw-header-left">
        <div id="cw-avatar">
          <svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 2a8 8 0 1 0 0 16A8 8 0 0 0 10 2Zm3.707 6.293a1 1 0 0 0-1.414 0L9 11.586 7.707 10.293a1 1 0 0 0-1.414 1.414l2 2a1 1 0 0 0 1.414 0l4-4a1 1 0 0 0 0-1.414Z" clip-rule="evenodd"/></svg>
        </div>
        <span id="cw-title">Assistant</span>
      </div>
      <div id="cw-header-right">
        <button class="cw-hbtn" id="cw-kebab" aria-label="More options">
          <svg viewBox="0 0 16 16" fill="currentColor"><circle cx="8" cy="3" r="1.2"/><circle cx="8" cy="8" r="1.2"/><circle cx="8" cy="13" r="1.2"/></svg>
        </button>
        <div id="cw-menu">
          <button class="cw-menu-item" id="cw-menu-new">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M8 3H4a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V8M10 2l2 2-5 5H5V7l5-5Z"/></svg>
            Start a new chat
          </button>
          <button class="cw-menu-item disabled" id="cw-menu-end">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>
            End chat
          </button>
          <div class="cw-menu-sep"></div>
          <button class="cw-menu-item" id="cw-menu-history">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><circle cx="8" cy="8" r="6"/><path d="M8 5v3.5l2 2"/></svg>
            View recent chats
          </button>
        </div>
        <button class="cw-hbtn" id="cw-close" aria-label="Close chat">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>
        </button>
      </div>
    </div>

    <!-- Chat view -->
    <div id="cw-messages"></div>

    <!-- Recent chats view -->
    <div id="cw-history">
      <div id="cw-history-header">
        <span>Recent chats</span>
        <button id="cw-history-new" aria-label="Start new chat">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M8 3H4a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V8M10 2l2 2-5 5H5V7l5-5Z"/></svg>
          New chat
        </button>
      </div>
      <div id="cw-history-list"></div>
    </div>

    <!-- Confirmation dialog -->
    <div id="cw-confirm" style="display:none;position:absolute;inset:0;background:rgba(0,0,0,.45);z-index:20;align-items:center;justify-content:center;">
      <div id="cw-confirm-box"></div>
    </div>
    <div id="cw-powered">Powered by Upbuff AI</div>
    <div id="cw-input-row">
      <input id="cw-input" type="text" placeholder="Message..." autocomplete="off" />
      <button id="cw-send" disabled aria-label="Send">
        <svg viewBox="0 0 16 16" fill="currentColor"><path fill-rule="evenodd" d="M8 2a.75.75 0 0 1 .75.75v8.69l3.22-3.22a.75.75 0 1 1 1.06 1.06l-4.5 4.5a.75.75 0 0 1-1.06 0l-4.5-4.5a.75.75 0 0 1 1.06-1.06l3.22 3.22V2.75A.75.75 0 0 1 8 2Z" clip-rule="evenodd" transform="rotate(-90 8 8)"/></svg>
      </button>
    </div>
  `;

  // Preview bubble
  const preview = document.createElement("div");
  preview.id = "cw-preview";
  preview.style.cssText = `position:fixed;${posRight?"right:30px":"left:80px"};bottom:90px;max-width:260px;background:#ffffff;color:#1e293b;font-family:system-ui,sans-serif;font-size:13px;line-height:1.5;padding:10px 14px;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,.18);cursor:pointer;z-index:2147483645;border:1px solid #e2e8f0;opacity:0;transform:translateY(8px) scale(0.95);transition:opacity .3s,transform .3s;word-break:break-word;`;
  const previewClose = document.createElement("button");
  previewClose.innerHTML = "&#x2715;";
  previewClose.style.cssText = `position:absolute;top:-8px;right:-8px;width:20px;height:20px;border-radius:50%;background:#64748b;border:none;color:#fff;font-size:9px;cursor:pointer;display:flex;align-items:center;justify-content:center;padding:0;z-index:1;`;
  preview.appendChild(previewClose);

  document.body.appendChild(bubble);
  document.body.appendChild(preview);
  document.body.appendChild(panel);

  // ── State ─────────────────────────────────────────────────────────────────────
  let isOpen          = false;
  let isLoading       = false;
  let conversationId  = null;
  let _localSessionId = "local_" + Date.now(); // stable ID for this session
  let _welcomeMsg     = "Hi! What can I help you with?";
  let _lastDay        = "";
  let _showingHistory = false;

  const messagesEl    = document.getElementById("cw-messages");
  const historyEl     = document.getElementById("cw-history");
  const historyListEl = document.getElementById("cw-history-list");
  const inputEl       = document.getElementById("cw-input");
  const sendEl        = document.getElementById("cw-send");
  const menuEl        = document.getElementById("cw-menu");

  // ── Local history storage ─────────────────────────────────────────────────────
  function loadHistory() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); } catch { return []; }
  }

  function saveHistory(history) {
    // Deduplicate by id — keep latest version of each session
    const seen = new Map();
    for (const s of history) {
      if (!seen.has(s.id) || new Date(s.updatedAt) > new Date(seen.get(s.id).updatedAt)) {
        seen.set(s.id, s);
      }
    }
    // Sort by updatedAt descending, keep max 20
    const deduped = [...seen.values()]
      .sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt))
      .slice(0, 20);
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(deduped)); } catch {}
  }

  function saveCurrentSession(messages) {
    if (!messages.length) return;
    // ALWAYS use _localSessionId as the stable key for localStorage
    // Server conversationId is only used for API calls, not storage
    const history  = loadHistory();
    const existing = history.findIndex(s => s.id === _localSessionId);
    const session  = {
      id:           _localSessionId,
      conversationId: conversationId || null, // keep for reference
      title:        messages.find(m => m.role === "user")?.text || "Chat session",
      updatedAt:    new Date().toISOString(),
      messages:     messages.map(m => ({ role: m.role, text: m.text, timestamp: m.timestamp })),
    };
    if (existing >= 0) history[existing] = session;
    else history.unshift(session);
    saveHistory(history);
  }

  // Track messages for local storage
  let _sessionMessages = [];

  // ── Time helpers ──────────────────────────────────────────────────────────────
  function _timeAgo(date) {
    if (typeof date === "string") date = new Date(date);
    const diff = Math.floor((Date.now() - date.getTime()) / 1000);
    if (diff < 60)     return "just now";
    if (diff < 3600)   return Math.floor(diff / 60) + " min ago";
    if (diff < 86400)  return Math.floor(diff / 3600) + " hr ago";
    if (diff < 172800) return "yesterday";
    return date.toLocaleDateString("en", { month: "short", day: "numeric", year: "numeric" });
  }

  function _dayLabel(date) {
    if (typeof date === "string") date = new Date(date);
    const now = new Date(); const y = new Date(); y.setDate(now.getDate() - 1);
    if (date.toDateString() === now.toDateString()) return "Today";
    if (date.toDateString() === y.toDateString())   return "Yesterday";
    return date.toLocaleDateString("en", { weekday: "long", month: "long", day: "numeric", year: "numeric" });
  }

  function _insertDayDivider(date) {
    const label = _dayLabel(date);
    if (label === _lastDay) return;
    _lastDay = label;
    const d = document.createElement("div");
    d.style.cssText = "display:flex;align-items:center;gap:8px;margin:6px 0;";
    d.innerHTML = `<div style="flex:1;height:1px;background:${dividerColor}"></div><span style="font-size:10px;font-weight:500;color:${dividerText};white-space:nowrap;font-family:system-ui,sans-serif;">${label}</span><div style="flex:1;height:1px;background:${dividerColor}"></div>`;
    messagesEl.appendChild(d);
  }

  // ── Message helpers ───────────────────────────────────────────────────────────
  function addMessage(role, text, timestamp) {
    const now = timestamp ? new Date(timestamp) : new Date();
    _insertDayDivider(now);

    const wrap = document.createElement("div");
    wrap.className = "cw-msg " + (role === "user" ? "user" : "bot");

    const bub = document.createElement("div");
    bub.className = "cw-bubble-text";

    const textEl = document.createElement("p");
    textEl.style.margin = "0";
    textEl.textContent = text;

    const timeEl = document.createElement("p");
    timeEl.title = now.toLocaleString();
    timeEl.style.cssText = `margin:4px 0 0;font-size:10px;opacity:0.55;text-align:${role==="user"?"right":"left"};font-family:system-ui,sans-serif;`;
    timeEl.textContent = _timeAgo(now);

    if (!timestamp) {
      // Save immediately for both user and bot messages
      _sessionMessages.push({ role, text, timestamp: now.toISOString() });
      saveCurrentSession(_sessionMessages);
      const timer = setInterval(() => { timeEl.textContent = _timeAgo(now); }, 60000);
      setTimeout(() => clearInterval(timer), 3600000);
    }

    bub.appendChild(textEl);
    bub.appendChild(timeEl);
    wrap.appendChild(bub);
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return wrap;
  }

  function typeMessage(role, text) {
    const now = new Date();
    _insertDayDivider(now);

    const wrap = document.createElement("div");
    wrap.className = "cw-msg " + (role === "user" ? "user" : "bot");

    const bub = document.createElement("div");
    bub.className = "cw-bubble-text";

    const textEl = document.createElement("p");
    textEl.style.margin = "0";
    bub.appendChild(textEl);

    const timeEl = document.createElement("p");
    timeEl.title = now.toLocaleString();
    timeEl.style.cssText = `margin:4px 0 0;font-size:10px;opacity:0;text-align:${role==="user"?"right":"left"};font-family:system-ui,sans-serif;transition:opacity .3s;`;
    timeEl.textContent = _timeAgo(now);
    bub.appendChild(timeEl);

    wrap.appendChild(bub);
    messagesEl.appendChild(wrap);

    // Save to session immediately — don't wait for typing to finish
    _sessionMessages.push({ role, text, timestamp: now.toISOString() });
    saveCurrentSession(_sessionMessages);

    // Type characters one by one
    let i = 0;
    const speed = 18; // ms per character — adjust for faster/slower
    function typeNext() {
      if (i < text.length) {
        textEl.textContent += text[i++];
        messagesEl.scrollTop = messagesEl.scrollHeight;
        setTimeout(typeNext, speed);
      } else {
        // Typing done — show timestamp
        timeEl.style.opacity = "0.55";
        const timer = setInterval(() => { timeEl.textContent = _timeAgo(now); }, 60000);
        setTimeout(() => clearInterval(timer), 3600000);
      }
    }
    typeNext();

    return wrap;
  }

  function showTyping() {
    const w = document.createElement("div");
    w.className = "cw-typing"; w.id = "cw-typing";
    w.innerHTML = `<div class="cw-dot"></div><div class="cw-dot"></div><div class="cw-dot"></div>`;
    messagesEl.appendChild(w);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function removeTyping() {
    const el = document.getElementById("cw-typing");
    if (el) el.remove();
  }

  // ── Views ─────────────────────────────────────────────────────────────────────
  function showChatView() {
    _showingHistory = false;
    messagesEl.style.display = "";
    historyEl.classList.remove("active");
    inputEl.disabled = false;
    sendEl.style.display = "";
    document.getElementById("cw-input-row").style.display = "";
  }

  function showHistoryView() {
    _showingHistory = true;
    messagesEl.style.display = "none";
    historyEl.classList.add("active");
    document.getElementById("cw-input-row").style.display = "none";
    renderHistory();
  }

  function renderHistory() {
    // Save current session first so it appears in history
    if (_sessionMessages.length) saveCurrentSession(_sessionMessages);

    const history = loadHistory();
    historyListEl.innerHTML = "";

    if (!history.length) {
      historyListEl.innerHTML = `<div class="cw-hist-empty">No previous chats yet.</div>`;
      return;
    }

    // Sort descending by updatedAt — most recent first
    const sorted = [...history].sort((a, b) =>
      new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
    );

    sorted.forEach(session => {
      const item = document.createElement("div");
      item.className = "cw-hist-item";
      const msgCount = session.messages.length;
      // Highlight current active session
      if (session.id === conversationId || 
          (!conversationId && session.id === sorted[0].id)) {
        item.style.borderLeft = `3px solid ${primaryColor}`;
        item.style.paddingLeft = "11px";
      }
      item.innerHTML = `
        <div class="cw-hist-title">${escHtml(session.title)}</div>
        <div class="cw-hist-meta">${msgCount} message${msgCount !== 1 ? "s" : ""} · ${_timeAgo(session.updatedAt)}</div>
      `;
      item.addEventListener("click", () => loadSession(session));
      historyListEl.appendChild(item);
    });
  }

  function loadSession(session) {
    startNewChat(false);
    _localSessionId = session.id;
    conversationId  = session.conversationId || null;
    _sessionMessages = session.messages.slice();
    session.messages.forEach(m => addMessage(m.role, m.text, m.timestamp));
    showChatView();
    // Enable end chat
    document.getElementById("cw-menu-end").classList.remove("disabled");
  }

  function escHtml(s) {
    return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  }

  // ── Preview ───────────────────────────────────────────────────────────────────
  function showPreview(text) {
    const el = document.getElementById("cw-preview");
    if (!el) return;
    const btn = el.querySelector("button");
    Array.from(el.childNodes).forEach(n => { if (n.nodeType === 3) el.removeChild(n); });
    el.insertBefore(document.createTextNode(text), btn);
    setTimeout(() => { el.style.opacity = "1"; el.style.transform = "translateY(0) scale(1)"; }, 800);
  }

  function hidePreview() {
    const el = document.getElementById("cw-preview");
    if (!el) return;
    el.style.opacity = "0"; el.style.transform = "translateY(8px) scale(0.95)";
    setTimeout(() => { el.style.display = "none"; }, 300);
  }

  // ── Chat core ─────────────────────────────────────────────────────────────────
  function startNewChat(focus = true) {
    conversationId  = null;
    _localSessionId = "local_" + Date.now(); // fresh ID for new session
    _sessionMessages = [];
    _lastDay = "";
    messagesEl.innerHTML = "";
    inputEl.value = "";
    sendEl.disabled = true;
    document.getElementById("cw-menu-end").classList.add("disabled");
    if (focus) {
      typeMessage("bot", _welcomeMsg);
      setTimeout(() => inputEl.focus(), 100);
    }
    showChatView();
  }

  async function fetchSettings() {
    _lastDay = "";
    _sessionMessages = [];
    typeMessage("bot", _welcomeMsg);
    document.getElementById("cw-menu-end").classList.remove("disabled");
    setTimeout(() => inputEl.focus(), 100);
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
        body: JSON.stringify({ agent_id: agentId, question: text, conversation_id: conversationId }),
      });
      removeTyping();
      if (res.ok) {
        const data = await res.json();
        conversationId = data.conversation_id;
        typeMessage("bot", data.answer || "I don't have enough information to answer that.");
        document.getElementById("cw-menu-end").classList.remove("disabled");
      } else {
        typeMessage("bot", "Sorry, I'm having trouble connecting. Please try again.");
      }
    } catch {
      removeTyping();
      typeMessage("bot", "Sorry, I'm having trouble connecting. Please try again.");
    } finally {
      isLoading = false;
      sendEl.disabled = false;
      inputEl.disabled = false;
      inputEl.focus();
    }
  }

  // ── Confirmation dialog ───────────────────────────────────────────────────────
  function showConfirm({ title, message, okLabel, okDanger, onOk }) {
    const overlay  = document.getElementById("cw-confirm");
    const box      = document.getElementById("cw-confirm-box");
    if (!overlay || !box) return;

    box.innerHTML = `
      <p>${title}</p>
      <span>${message}</span>
      <div id="cw-confirm-btns">
        <button class="cw-confirm-cancel">Cancel</button>
        <button class="cw-confirm-ok" style="${okDanger ? "background:#ef4444;" : ""}">${okLabel}</button>
      </div>
    `;

    overlay.style.display = "flex";

    box.querySelector(".cw-confirm-cancel").addEventListener("click", () => {
      overlay.style.display = "none";
    });
    box.querySelector(".cw-confirm-ok").addEventListener("click", () => {
      overlay.style.display = "none";
      onOk();
    });
  }

  function showEndChatConfirm(closeAfter = false) {
    const overlay = document.getElementById("cw-confirm");
    const box     = document.getElementById("cw-confirm-box");
    if (!overlay || !box) return;

    box.innerHTML = `
      <div style="text-align:center;padding:8px 0 4px;">
        <svg viewBox="0 0 40 40" fill="none" style="width:40px;height:40px;margin:0 auto 12px;display:block;">
          <circle cx="20" cy="20" r="18" stroke="${subText}" stroke-width="1.5"/>
          <path d="M14 14l12 12M26 14L14 26" stroke="${subText}" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
        <p style="font-size:15px;font-weight:700;color:${textColor};margin:0 0 6px;font-family:system-ui,sans-serif;">End chat</p>
        <span style="font-size:12px;color:${subText};font-family:system-ui,sans-serif;">Do you want to end this chat?</span>
      </div>
      <div style="display:flex;flex-direction:column;gap:8px;margin-top:20px;">
        <button id="cw-end-confirm-yes" style="width:100%;padding:11px;border:none;border-radius:10px;background:${textColor};color:${bgColor};font-size:13px;font-weight:700;cursor:pointer;font-family:system-ui,sans-serif;">Yes, end chat</button>
        <button id="cw-end-confirm-no" style="width:100%;padding:11px;border:1px solid ${bubbleBorder};border-radius:10px;background:none;color:${textColor};font-size:13px;font-weight:500;cursor:pointer;font-family:system-ui,sans-serif;">Cancel</button>
      </div>
    `;

    overlay.style.display = "flex";

    document.getElementById("cw-end-confirm-no").addEventListener("click", () => {
      overlay.style.display = "none";
    });

    document.getElementById("cw-end-confirm-yes").addEventListener("click", () => {
      overlay.style.display = "none";
      saveCurrentSession(_sessionMessages);
      // Clear chat so next open starts fresh
      messagesEl.innerHTML = "";
      conversationId = null;
      _sessionMessages = [];
      _lastDay = "";
      inputEl.disabled = false;
      sendEl.disabled = true;
      inputEl.value = "";
      showChatView();
      // Close widget
      isOpen = false;
      panel.classList.remove("open");
      bubble.setAttribute("aria-expanded", "false");
    });
  }

  // ── Kebab menu ────────────────────────────────────────────────────────────────
  function closeMenu() { menuEl.classList.remove("open"); }

  document.getElementById("cw-kebab").addEventListener("click", (e) => {
    e.stopPropagation();
    menuEl.classList.toggle("open");
  });

  document.getElementById("cw-menu-new").addEventListener("click", () => {
    closeMenu();
    startNewChat();
  });

  document.getElementById("cw-menu-end").addEventListener("click", () => {
    if (document.getElementById("cw-menu-end").classList.contains("disabled")) return;
    closeMenu();
    showEndChatConfirm();
  });

  document.getElementById("cw-menu-history").addEventListener("click", () => {
    closeMenu();
    if (_showingHistory) showChatView();
    else showHistoryView();
  });

  // History panel "New chat" button
  document.getElementById("cw-history-new").addEventListener("click", () => {
    startNewChat();
  });

  // ── Panel open/close ──────────────────────────────────────────────────────────
  function openPanel() {
    isOpen = true;
    panel.classList.add("open");
    bubble.setAttribute("aria-expanded", "true");
    hidePreview();
    if (messagesEl.children.length === 0) {
      // Try to restore last session from localStorage
      const history = loadHistory();
      if (history.length > 0) {
        const last = history[0]; // most recent session
        _lastDay         = "";
        _sessionMessages = last.messages.slice();
        _localSessionId  = last.id; // restore stable local key
        conversationId   = last.conversationId || null; // restore server ID if saved
        last.messages.forEach(m => addMessage(m.role, m.text, m.timestamp));
        showChatView();
        setTimeout(() => {
          messagesEl.scrollTop = messagesEl.scrollHeight;
          inputEl.focus();
        }, 100);
      } else {
        fetchSettings();
      }
    } else {
      setTimeout(() => inputEl.focus(), 250);
    }
  }

  bubble.addEventListener("mousedown", (e) => {
    e.preventDefault(); e.stopPropagation();
    if (isOpen) {
      isOpen = false; panel.classList.remove("open");
      bubble.setAttribute("aria-expanded", "false");
      closeMenu();
    } else {
      openPanel();
      setTimeout(() => inputEl.focus(), 250);
    }
  });

  preview.addEventListener("mousedown", (e) => {
    if (e.target.tagName === "BUTTON") return;
    e.preventDefault(); e.stopPropagation();
    openPanel();
    setTimeout(() => inputEl.focus(), 250);
  });

  previewClose.addEventListener("mousedown", (e) => {
    e.preventDefault(); e.stopPropagation();
    hidePreview();
  });

  document.getElementById("cw-close").addEventListener("click", () => {
    closeMenu();
    // Just close — no confirmation needed. Chat is auto-saved.
    if (_sessionMessages.length) saveCurrentSession(_sessionMessages);
    isOpen = false;
    panel.classList.remove("open");
    bubble.setAttribute("aria-expanded", "false");
  });

  inputEl.addEventListener("input", () => {
    sendEl.disabled = !inputEl.value.trim() || isLoading;
  });

  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !isLoading) {
      e.preventDefault(); sendMessage(inputEl.value.trim());
    }
  });

  sendEl.addEventListener("click", () => sendMessage(inputEl.value.trim()));

  document.addEventListener("mousedown", (e) => {
    if (!panel.contains(e.target)) closeMenu();
    if (isOpen && !panel.contains(e.target) && e.target !== bubble) {
      isOpen = false; panel.classList.remove("open");
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (menuEl.classList.contains("open")) { closeMenu(); return; }
      if (_showingHistory) { showChatView(); return; }
      if (isOpen) { isOpen = false; panel.classList.remove("open"); }
    }
  });

  // ── Pre-fetch settings → show preview ────────────────────────────────────────
  (async () => {
    try {
      const res = await fetch(`${apiBase}/widget/settings/${agentId}`);
      if (res.ok) {
        const s = await res.json();
        _welcomeMsg = s.welcome_message || "Hi! What can I help you with?";
        const titleEl = document.getElementById("cw-title");
        if (titleEl && s.display_name) titleEl.textContent = s.display_name;
        // Apply dynamic color from settings
        if (s.primary_color) {
          const dynColor = s.primary_color;
          const bubbleEl = document.getElementById("cw-bubble");
          const headerEl = document.getElementById("cw-header");
          const sendEl2  = document.getElementById("cw-send");
          if (bubbleEl) bubbleEl.style.background = dynColor;
          if (headerEl) headerEl.style.background = dynColor;
          if (sendEl2)  sendEl2.style.background  = dynColor;
        }
        showPreview(_welcomeMsg);
      } else { showPreview(_welcomeMsg); }
    } catch { showPreview(_welcomeMsg); }
  })();

  } // end init

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    setTimeout(init, 0);
  }

})();