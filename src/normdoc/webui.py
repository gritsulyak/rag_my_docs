"""Веб-интерфейс RAG-ассистента: чат в браузере + консольный (без-JS) fallback.

Модуль содержит только HTML-шаблоны (без внешних CDN — всё офлайн):

* :func:`chat_page`   — страница чата ``GET /`` с JS-формой (fetch → ``POST /answer``)
  и нативным fallback-ом на ``GET /ask`` для браузеров без JavaScript
  (``w3m``, ``lynx``, ``curl``);
* :func:`answer_page` — серверный рендер ответа ``GET /ask?q=…`` для консольных
  браузеров и просмотра ответа без JS.

Плейсхолдеры ``__COLLECTION__`` / ``__MODEL__`` / ``__TOP_K__`` подставляются без
f-строк, чтобы не конфликтовать с фигурными скобками в CSS/JS.
"""

from __future__ import annotations

import re
from html import escape

__all__ = ["chat_page", "answer_page"]


CHAT_PAGE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>normdoc-assistant — чат по нормативным документам</title>
<style>
  :root { color-scheme: light dark; --bg:#f5f6f8; --fg:#1c1e21; --card:#fff;
          --mut:#6b7280; --acc:#1f6feb; --ok:#1a7f37; --warn:#b45309; --bd:#e2e4e8; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14161a; --fg:#e8eaed; --card:#1d2025; --mut:#9aa0a6;
            --acc:#5b9dff; --ok:#57d38c; --warn:#f0b429; --bd:#2a2e35; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Ubuntu,sans-serif; }
  header { padding:14px 22px; border-bottom:1px solid var(--bd); display:flex;
           gap:12px; align-items:baseline; flex-wrap:wrap; }
  header h1 { font-size:18px; margin:0; }
  header .meta { color:var(--mut); font-size:13px; }
  .wrap { display:grid; grid-template-columns: 300px 1fr; gap:18px;
          padding:18px 22px; max-width:1200px; margin:0 auto; }
  @media (max-width: 820px) { .wrap { grid-template-columns:1fr; } aside { order:2; } }
  aside h2, main h2 { font-size:14px; color:var(--mut); text-transform:uppercase;
                      letter-spacing:.04em; margin:0 0 8px; }
  .hist { list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:6px; }
  .hist button { width:100%; text-align:left; background:var(--card); color:var(--fg);
                 border:1px solid var(--bd); border-radius:8px; padding:8px 10px;
                 font-size:13px; cursor:pointer; }
  .hist button:hover { border-color:var(--acc); }
  .hist button.active { border-color:var(--acc); box-shadow:0 0 0 1px var(--acc) inset; }
  form { display:flex; flex-direction:column; gap:10px; }
  textarea, input[type=number] { width:100%; background:var(--card); color:var(--fg);
        border:1px solid var(--bd); border-radius:10px; padding:10px 12px; font:inherit; resize:vertical; }
  textarea { min-height:76px; }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  button.primary { background:var(--acc); color:#fff; border:0; border-radius:10px;
                   padding:10px 18px; font-weight:600; cursor:pointer; }
  button.primary:disabled { opacity:.55; cursor:progress; }
  .hint { color:var(--mut); font-size:12px; }
  .status { font-size:13px; color:var(--mut); }
  .status b.ok { color:var(--ok); } .status b.err { color:var(--warn); }
  .answer { background:var(--card); border:1px solid var(--bd); border-radius:12px;
            padding:16px 18px; white-space:pre-wrap; word-wrap:break-word; }
  .stats { color:var(--mut); font-size:13px; margin:8px 0 0; }
  .warn { color:var(--warn); font-size:13px; margin:8px 0 0; }
  .sources { margin:14px 0 0; padding:0; list-style:none; display:flex;
             flex-direction:column; gap:8px; }
  .sources li { background:var(--card); border:1px solid var(--bd); border-radius:10px;
                padding:10px 12px; font-size:13px; }
  .sources .src { font-weight:600; }
  .sources .m { color:var(--mut); }
  .sources .snip { color:var(--mut); margin-top:4px; }
  .empty { color:var(--mut); }
  a { color:var(--acc); }
</style>
</head>
<body>
<header>
  <h1>normdoc-assistant</h1>
  <span class="meta">коллекция <code>__COLLECTION__</code> · модель <code>__MODEL__</code></span>
  <span class="status" id="status">проверка…</span>
</header>
<div class="wrap">
  <aside>
    <h2>История</h2>
    <ul class="hist" id="hist"><li class="empty">пока пусто</li></ul>
    <p class="hint" style="margin-top:12px">
      Без JavaScript работает как обычная форма (GET&nbsp;<code>/ask</code>) —
      годится для <code>w3m</code>/<code>lynx</code>/<code>curl</code>.
      JSON API: <a href="/docs">/docs</a> · <a href="/info">/info</a>.
    </p>
  </aside>
  <main>
    <h2>Вопрос по документам</h2>
    <form id="ask" method="get" action="/ask">
      <textarea id="q" name="q" placeholder="Например: Сколько классов защиты устанавливается для средств контейнеризации Требованиями ФСТЭК России №118?"
                required></textarea>
      <div class="row">
        <label>top-k <input type="number" id="topk" name="top_k" min="1" max="20" value="__TOP_K__" style="width:80px"></label>
        <button class="primary" type="submit" id="send">Спросить</button>
        <span class="hint">Ctrl/⌘+Enter — отправить</span>
      </div>
    </form>
    <div id="out" style="margin-top:16px">
      <p class="empty">Задайте вопрос — ответ появится здесь со ссылками на источники.</p>
    </div>
  </main>
</div>
<script>
(function () {
  var form = document.getElementById('ask');
  var q = document.getElementById('q');
  var topk = document.getElementById('topk');
  var out = document.getElementById('out');
  var send = document.getElementById('send');
  var hist = document.getElementById('hist');
  var status = document.getElementById('status');
  var items = [];
  var active = -1;

  function esc(s) { return String(s == null ? '' : s); }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function linkify(text, sources) {
    var map = {};
    (sources || []).forEach(function (s) { if (s.url && s.source) map[s.source] = s.url; });
    var labels = Object.keys(map);
    var frag = document.createDocumentFragment();
    String(text == null ? '' : text).split(/(\\[[^\\]\\n]+\\])/).forEach(function (part) {
      if (part.charAt(0) === '[' && part.charAt(part.length - 1) === ']') {
        for (var i = 0; i < labels.length; i++) {
          if (part.indexOf(labels[i]) !== -1) {
            var a = el('a', null, part);
            a.href = map[labels[i]]; a.target = '_blank'; a.rel = 'noopener';
            frag.appendChild(a);
            return;
          }
        }
      }
      if (part) frag.appendChild(document.createTextNode(part));
    });
    return frag;
  }

  function render(item) {
    out.textContent = '';
    var ans = el('div', 'answer');
    ans.appendChild(linkify(item.answer, item.sources));
    out.appendChild(ans);
    var stats = 'Итого ' + item.latency_s.toFixed(2) + ' с · поиск ' + item.retrieval_s.toFixed(2) +
                ' с · генерация ' + item.generation_s.toFixed(2) + ' с · коллекция ' + item.collection;
    if (item.usage && item.usage.completion_tokens) stats += ' · токенов ' + item.usage.completion_tokens;
    out.appendChild(el('p', 'stats', stats));
    if (item.dropped_citations && item.dropped_citations.length) {
      out.appendChild(el('p', 'warn', 'Отброшены несуществующие ссылки: ' +
        item.dropped_citations.join(', ') + ' (защита от галлюцинаций).'));
    }
    if (item.sources && item.sources.length) {
      out.appendChild(el('h2', null, 'Источники'));
      var ul = el('ul', 'sources');
      item.sources.forEach(function (s, i) {
        var li = el('li');
        var head = el('div');
        var nameText = '[' + (i + 1) + '] ' + esc(s.source || 'документ');
        if (s.url) {
          var link = el('a', 'src', nameText);
          link.href = s.url; link.target = '_blank'; link.rel = 'noopener';
          head.appendChild(link);
        } else {
          head.appendChild(el('span', 'src', nameText));
        }
        var loc = (s.page != null ? ', стр. ' + s.page : '') +
                  (s.score != null ? ' · релевантность ' + s.score.toFixed(3) : '');
        head.appendChild(el('span', 'm', loc));
        li.appendChild(head);
        li.appendChild(el('div', 'snip', s.snippet));
        ul.appendChild(li);
      });
      out.appendChild(ul);
    }
  }

  function renderHistory() {
    hist.textContent = '';
    if (!items.length) { hist.appendChild(el('li', 'empty', 'пока пусто')); return; }
    items.forEach(function (it, i) {
      var b = el('button', i === active ? 'active' : null, it.question);
      b.title = it.question;
      b.addEventListener('click', function () { active = i; render(it); renderHistory(); });
      var li = el('li'); li.appendChild(b); hist.appendChild(li);
    });
  }

  function ask() {
    var question = q.value.trim();
    if (!question) return;
    send.disabled = true;
    status.textContent = 'выполняется…';
    fetch('/answer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: question, top_k: parseInt(topk.value, 10) || null })
    }).then(function (r) {
      if (!r.ok) return r.text().then(function (t) { throw new Error(t); });
      return r.json();
    }).then(function (data) {
      data.question = question;
      items.unshift(data);
      active = 0;
      render(data);
      renderHistory();
      status.innerHTML = '<b class="ok">готово</b>';
    }).catch(function (e) {
      out.textContent = '';
      out.appendChild(el('p', 'warn', 'Ошибка: ' + e.message));
      status.innerHTML = '<b class="err">ошибка</b>';
    }).finally(function () { send.disabled = false; });
  }

  form.addEventListener('submit', function (ev) { ev.preventDefault(); ask(); });
  q.addEventListener('keydown', function (ev) {
    if ((ev.ctrlKey || ev.metaKey) && ev.key === 'Enter') { ev.preventDefault(); ask(); }
  });

  fetch('/health').then(function (r) { return r.json(); }).then(function (h) {
    status.innerHTML = '<b class="ok">API: ' + h.status + '</b>';
  }).catch(function () { status.innerHTML = '<b class="err">API недоступен</b>'; });
})();
</script>
</body>
</html>
"""


ANSWER_PAGE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ответ — normdoc-assistant</title>
<style>
  body { margin:0 auto; max-width:820px; padding:20px;
         font:15px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  h1 { font-size:18px; } h2 { font-size:14px; margin-top:22px; }
  .answer { white-space:pre-wrap; border:1px solid currentColor; border-radius:8px; padding:12px 14px; }
  .stats, .m { opacity:.7; font-size:13px; }
  .warn { font-weight:700; }
  ul { padding-left:18px; } li { margin-bottom:8px; }
  .snip { opacity:.7; font-size:13px; }
  form { margin:16px 0; display:flex; gap:8px; align-items:flex-start; }
  textarea, input { font:inherit; padding:6px 8px; }
  textarea { width:70%; min-height:56px; }
</style>
</head>
<body>
<h1>normdoc-assistant</h1>
<form method="get" action="/ask">
  <textarea name="q" placeholder="Следующий вопрос…" required>__QUESTION__</textarea>
  <input type="number" name="top_k" min="1" max="20" value="__TOP_K__" style="width:70px">
  <button type="submit">Спросить</button>
</form>
__BODY__
<p class="stats"><a href="/">← чат</a> · <a href="/docs">/docs</a> · <a href="/info">/info</a></p>
</body>
</html>
"""


def chat_page(collection: str, model: str, top_k: int) -> str:
    """HTML-страница чата."""
    return (
        CHAT_PAGE.replace("__COLLECTION__", escape(str(collection)))
        .replace("__MODEL__", escape(str(model)))
        .replace("__TOP_K__", str(int(top_k)))
    )


def _source_links(sources: list[dict]) -> dict[str, str]:
    """Карта «метка источника -> ссылка» для подстановки в цитаты ответа."""
    links: dict[str, str] = {}
    for src in sources or []:
        url = src.get("url")
        label = src.get("source")
        if url and label:
            links.setdefault(str(label), str(url))
    return links


def _answer_html(answer: str, sources: list[dict]) -> str:
    """Экранирует текст ответа и превращает цитаты ``[источник, стр. N]`` в ссылки."""
    links = _source_links(sources)
    if not links:
        return escape(answer)
    labels = list(links)
    parts: list[str] = []
    pos = 0
    for match in re.finditer(r"\[[^\]\n]+\]", answer):
        inner = match.group(0)
        url = next((links[label] for label in labels if label in inner), None)
        parts.append(escape(answer[pos:match.start()]))
        if url:
            href = escape(url, quote=True)
            parts.append(f"<a href='{href}' target='_blank' rel='noopener'>{escape(inner)}</a>")
        else:
            parts.append(escape(inner))
        pos = match.end()
    parts.append(escape(answer[pos:]))
    return "".join(parts)


def _sources_html(sources: list[dict]) -> str:
    if not sources:
        return ""
    parts = ["<h2>Источники</h2>", "<ul>"]
    for i, src in enumerate(sources, start=1):
        name = escape(str(src.get("source") or "документ"))
        url = src.get("url")
        if url:
            href = escape(str(url), quote=True)
            name = f"<a href='{href}' target='_blank' rel='noopener'>{name}</a>"
        page = src.get("page")
        score = src.get("score")
        loc = ""
        if page is not None:
            loc += f", стр. {escape(str(page))}"
        if isinstance(score, (int, float)):
            loc += f" · релевантность {score:.3f}"
        snippet = escape(str(src.get("snippet", "")))
        parts.append(f"<li><b>[{i}] {name}</b><span class='m'>{loc}</span><div class='snip'>{snippet}</div></li>")
    parts.append("</ul>")
    return "\n".join(parts)


def answer_page(
    *,
    question: str,
    body_html: str,
    top_k: int,
) -> str:
    """HTML-страница с готовым ответом (для ``GET /ask?q=…`` без JS)."""
    return (
        ANSWER_PAGE.replace("__QUESTION__", escape(str(question)))
        .replace("__TOP_K__", str(int(top_k)))
        .replace("__BODY__", body_html)
    )


def render_answer(result, sources: list[dict], collection: str, model: str) -> str:
    """Собирает HTML-блок ответа + источников для серверной страницы ``/ask``."""
    parts = [f"<div class='answer'>{_answer_html(str(result.answer), sources)}</div>"]
    stats = (
        f"Итого {result.latency_s:.2f} с · поиск {result.retrieval_s:.2f} с · "
        f"генерация {result.generation_s:.2f} с · коллекция {escape(str(collection))} · "
        f"модель {escape(str(model))}"
    )
    completion = (result.usage or {}).get("completion_tokens")
    if completion:
        stats += f" · токенов {escape(str(completion))}"
    parts.append(f"<p class='stats'>{stats}</p>")
    dropped = getattr(result, "dropped_citations", None)
    if dropped:
        parts.append(
            "<p class='warn'>Отброшены несуществующие ссылки: "
            + escape(", ".join(dropped))
            + " (защита от галлюцинаций).</p>"
        )
    parts.append(_sources_html(sources))
    return "\n".join(parts)
