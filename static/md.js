/* md.js — общие утилиты страниц: мини-рендерер Markdown (без внешних
   библиотек — контейнер живёт офлайн) и копирование в буфер с фолбэком.

   Фолбэк нужен, потому что navigator.clipboard существует только в secure
   context (HTTPS/localhost), а приложение живёт и на http://…ts.net. */

"use strict";

function mdEsc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, c => (
    {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
}

/* Срезать YAML-frontmatter и (опционально) первый H1 — на экранах заголовок
   встречи и так показан шапкой, дублировать его в тексте незачем. */
function mdStrip(src, dropFirstH1) {
  let s = String(src || "");
  if (s.startsWith("---")) {
    const e = s.indexOf("\n---", 3);
    if (e !== -1) s = s.slice(e + 4).replace(/^\s+/, "");
  }
  if (dropFirstH1) s = s.replace(/^#\s[^\n]*\n+/, "");
  return s;
}

function mdRender(src, opts) {
  const o = opts || {};
  const lines = mdStrip(src, o.dropFirstH1 !== false).split(/\r?\n/);
  let out = "", inUl = false, inBq = false;
  const inline = s => mdEsc(s)
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[\[([^\]]+)\]\]/g, '<span style="color:var(--accent)">$1</span>');
  const closeAll = () => {
    if (inUl) { out += "</ul>"; inUl = false; }
    if (inBq) { out += "</blockquote>"; inBq = false; }
  };
  for (const raw of lines) {
    const l = raw.trimEnd();
    const h = l.match(/^(#{1,4})\s+(.*)/);
    if (h) { closeAll(); const n = Math.min(h[1].length, 3);
      out += `<h${n}>${inline(h[2])}</h${n}>`; continue; }
    if (/^\s*\d+\.\s+/.test(l)) { if (!inUl) { closeAll(); out += "<ul>"; inUl = true; }
      out += "<li>" + inline(l.replace(/^\s*\d+\.\s+/, "")) + "</li>"; continue; }
    if (/^\s*[-*]\s+/.test(l)) { if (!inUl) { closeAll(); out += "<ul>"; inUl = true; }
      out += "<li>" + inline(l.replace(/^\s*[-*]\s+/, "")) + "</li>"; continue; }
    if (/^>\s?/.test(l)) { if (!inBq) { closeAll(); out += "<blockquote>"; inBq = true; }
      out += inline(l.replace(/^>\s?/, "")) + "<br>"; continue; }
    if (/^\s*\|.*\|\s*$/.test(l)) { closeAll();
      if (/^\s*\|[\s:|-]+\|\s*$/.test(l)) continue;
      out += "<table><tr>" + l.split("|").slice(1, -1)
        .map(c => "<td>" + inline(c.trim()) + "</td>").join("") + "</tr></table>";
      continue; }
    if (l === "---" || l === "") { closeAll(); continue; }
    closeAll(); out += "<p>" + inline(l) + "</p>";
  }
  closeAll();
  return out.replace(/<\/table><table>/g, "");
}

/* Копирование: clipboard API в secure context, иначе textarea+execCommand. */
function copyToClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text).then(() => true).catch(() => _copyFallback(text));
  }
  return Promise.resolve(_copyFallback(text));
}
function _copyFallback(text) {
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.cssText = "position:fixed;left:-9999px;top:0";
    document.body.appendChild(ta);
    ta.select();
    ta.setSelectionRange(0, ta.value.length);
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  } catch (e) {
    return false;
  }
}
