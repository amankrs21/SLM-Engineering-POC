const API = '/api';
const BRANCHES = ['slm', 'mistral'];
const STEP_LABELS = {
  embedding: 'Embedded search query',
  search: 'Searched vector knowledge base',
  rerank: 'Reranked candidate passages',
  cache: 'Checked LLM response cache',
  verify: 'Validated citations and confidence',
  citation_repair: 'Model is revising missing or invalid citations',
  direct: 'Model chose a direct reply',
  limit: 'Tool budget reached; preparing final answer',
  error: 'Run failed'
};
const state = { sessionId: null, busy: false, live: {} };
const $ = (id) => document.getElementById(id);
let toastTimer;
let pendingDelete = null;

function sessionFromPath() {
  return location.pathname.match(/^\/chat\/([0-9a-f-]{36})\/?$/i)?.[1] || null;
}

function setSessionId(id, navigation = 'push') {
  state.sessionId = id;
  const path = id ? `/chat/${encodeURIComponent(id)}` : '/chat/new';
  if (navigation !== 'none' && location.pathname !== path) {
    history[`${navigation}State`]({ sessionId: id }, '', path);
  }
  if (id) localStorage.setItem('rag-lab-session', id);
  else localStorage.removeItem('rag-lab-session');
}

function toast(message) {
  const node = $('toast');
  node.textContent = message;
  node.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove('show'), 4500);
}

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { const body = await response.json(); detail = body.detail || detail; } catch (_) { /* no JSON body */ }
    throw new Error(detail);
  }
  return response.json();
}

function setBusy(busy) {
  state.busy = busy;
  $('send-button').disabled = busy;
  $('new-chat').disabled = busy;
  $('question').disabled = busy;
  document.querySelectorAll('.session-item, .delete-session').forEach((button) => { button.disabled = busy; });
  $('send-button').querySelector('span:first-child').textContent = busy ? 'Working…' : 'Send to both';
}

function setStatus(branch, label, mode = '') {
  const node = $(`${branch}-status`);
  node.textContent = label;
  node.className = `model-status ${mode}`;
}

function clearNode(node) { while (node.firstChild) node.removeChild(node.firstChild); }

function emptyState(branch) {
  const container = $(`${branch}-messages`);
  clearNode(container);
  const wrap = document.createElement('div');
  wrap.className = 'empty-state';
  const icon = document.createElement('div');
  icon.className = 'empty-orbit';
  icon.textContent = branch === 'slm' ? '✦' : '◇';
  const title = document.createElement('strong');
  title.textContent = branch === 'slm' ? 'Your local model is ready' : 'Your hosted model is ready';
  const copy = document.createElement('p');
  copy.textContent = 'Ask about your documents or data. Open Thinking trace on any answer to see its tool activity.';
  wrap.append(icon, title, copy);
  container.append(wrap);
}

function clearEmpty(branch) {
  $(`${branch}-messages`).querySelector('.empty-state')?.remove();
}

function createTurn(branch, question) {
  clearEmpty(branch);
  const container = $(`${branch}-messages`);
  const group = document.createElement('div');
  group.className = 'turn-group';
  const user = document.createElement('div');
  user.className = 'message user';
  user.textContent = question;
  const assistant = document.createElement('div');
  assistant.className = 'message assistant';
  const kicker = document.createElement('div');
  kicker.className = 'message-kicker';
  kicker.textContent = branch === 'slm' ? 'Qwen3-4B-Q4_K_M' : 'mistral-large-4';
  const thinking = document.createElement('details');
  thinking.className = 'thinking-trace';
  thinking.open = true;
  const traceSummary = document.createElement('summary');
  const traceTitle = document.createElement('span');
  traceTitle.textContent = 'Thinking trace';
  const traceCount = document.createElement('span');
  traceCount.className = 'trace-count';
  traceCount.textContent = 'Waiting for model';
  traceSummary.append(traceTitle, traceCount);
  const trace = document.createElement('ol');
  trace.className = 'trace-list';
  thinking.append(traceSummary, trace);
  const answer = document.createElement('div');
  answer.className = 'answer-text';
  answer.textContent = 'Understanding your request…';
  assistant.append(kicker, thinking, answer);
  group.append(user, assistant);
  container.append(group);
  container.scrollTop = container.scrollHeight;
  return { assistant, answer, kicker, thinking, trace, traceCount };
}

function appendMeta(target, result) {
  target.querySelector('.answer-meta')?.remove();
  target.querySelector('.sources')?.remove();
  const meta = document.createElement('div');
  meta.className = 'answer-meta';
  const confidence = result.confidence || {};
  const metrics = result.metrics || {};
  const values = [
    ['EVIDENCE', confidence.score == null ? (confidence.method === 'direct_response' ? 'N/A · direct response' : 'Not assessed') : `${Math.round(confidence.score * 100)}% · ${confidence.label || 'low'}`],
    ['LATENCY', metrics.total_ms == null ? '—' : `${(metrics.total_ms / 1000).toFixed(1)}s`],
    ['SOURCES', String((result.citations || []).length)]
  ];
  values.forEach(([label, value]) => {
    const chip = document.createElement('span');
    chip.className = 'metric';
    const strong = document.createElement('strong');
    strong.textContent = `${label} `;
    chip.append(strong, document.createTextNode(value));
    meta.append(chip);
  });
  target.append(meta);
  const citations = result.citations || [];
  if (citations.length) {
    const details = document.createElement('details');
    details.className = 'sources';
    const summary = document.createElement('summary');
    summary.textContent = `View ${citations.length} cited source${citations.length === 1 ? '' : 's'}`;
    const list = document.createElement('ul');
    citations.forEach((item) => {
      const row = document.createElement('li');
      const label = `[${item.ref}] ${item.source} · ${item.location}`;
      if (item.source_url && /^https?:\/\//i.test(item.source_url)) {
        const link = document.createElement('a');
        link.href = item.source_url;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.textContent = label;
        row.append(link);
      } else { row.textContent = label; }
      list.append(row);
    });
    details.append(summary, list);
    target.append(details);
  }
}

function formatAnswer(target, text) {
  clearNode(target);
  for (const piece of text.split(/(\*\*[^*]+\*\*)/g)) {
    if (!piece) continue;
    if (piece.startsWith('**') && piece.endsWith('**')) {
      const strong = document.createElement('strong');
      strong.textContent = piece.slice(2, -2);
      target.append(strong);
    } else { target.append(document.createTextNode(piece)); }
  }
}

function showResult(branch, target, result) {
  target.answer.classList.remove('streaming');
  if (result.metrics?.model_name) target.kicker.textContent = result.metrics.model_name;
  if (result.metrics?.trace) {
    clearNode(target.trace);
    result.metrics.trace.forEach((event) => updateTrace(branch, event, target, false));
  }
  target.traceCount.textContent = target.trace.children.length ? `${target.trace.children.length} activities` : 'No activity recorded';
  target.thinking.open = false;
  if (result.status === 'ok') {
    formatAnswer(target.answer, result.answer || 'No answer returned.');
    appendMeta(target.assistant, result);
    setStatus(branch, 'Complete', 'complete');
  } else if (result.status === 'running') {
    target.answer.textContent = 'Run is still in progress. Reload this chat shortly to see the result.';
    setStatus(branch, 'In progress', 'running');
  } else {
    target.assistant.classList.add('error');
    target.answer.textContent = result.error || 'The model could not complete this run.';
    setStatus(branch, 'Error', 'error');
  }
  const container = $(`${branch}-messages`);
  container.scrollTop = container.scrollHeight;
}

function traceDetail(step, detail = {}) {
  if (step === 'embedding' && detail.dimension) return `${detail.dimension} dimensions`;
  if (step === 'search' && detail.candidates != null) return `${detail.candidates} candidates`;
  if (step === 'rerank' && detail.selected != null) return `top ${detail.selected} selected`;
  if (step === 'cache' && detail.hit != null) return detail.hit ? 'cache hit' : 'new generation';
  if (step === 'verify') return detail.direct ? 'direct response · no source checks needed' : `${detail.citations || 0} citations`;
  if (step === 'error' && detail.message) return detail.message;
  return '';
}

function updateTrace(branch, event, target = state.live[branch], live = true) {
  if (!target) return;
  const list = target.trace;
  const key = `${event.round || 0}:${event.call_id || ''}:${event.step}`;
  let item = Array.from(list.children).find((node) => node.dataset.key === key);
  if (!item) {
    item = document.createElement('li');
    item.dataset.key = key;
    const label = document.createElement('span');
    label.className = 'trace-label';
    const time = document.createElement('span');
    time.className = 'trace-time';
    item.append(label, time);
    list.append(item);
  }
  item.className = event.status === 'start' ? 'active' : event.status === 'error' ? 'error' : 'done';
  const detail = event.detail || {};
  let label = STEP_LABELS[event.step] || event.step;
  if (event.step === 'model') {
    label = event.status === 'start' ? `${event.model} is choosing the next action` :
      detail.decision === 'tools' ? `Model selected ${(detail.tools || []).join(', ')}` : 'Model is answering';
  } else if (event.step === 'tool') {
    label = `${event.status === 'start' ? 'Calling' : event.status === 'error' ? 'Tool error:' : 'Completed'} ${detail.tool}`;
  } else {
    const extra = traceDetail(event.step, detail);
    if (extra) label += ` · ${extra}`;
  }
  item.querySelector('.trace-label').textContent = label;
  item.querySelector('.trace-time').textContent = `${(event.elapsed_ms / 1000).toFixed(1)}s`;
  if (event.step === 'tool') {
    item.querySelector('.tool-data')?.remove();
    const data = document.createElement('details');
    data.className = 'tool-data';
    const summary = document.createElement('summary');
    summary.textContent = detail.output ? 'Tool input & result' : 'Tool input';
    const text = document.createElement('pre');
    let input = detail.input;
    try { input = JSON.parse(input); } catch (_) { /* malformed model arguments remain visible */ }
    text.textContent = JSON.stringify({ input, ...(detail.output ? { output: detail.output } : {}) }, null, 2);
    data.append(summary, text);
    item.append(data);
  }
  target.traceCount.textContent = `${list.children.length} activities`;
  if (event.model) target.kicker.textContent = event.model;
  if (live && event.status === 'start') setStatus(branch, event.step === 'model' ? 'Thinking' : 'Using tool', 'running');
}

function renderStored(branch, conversation) {
  const messages = conversation.branches?.[branch] || [];
  const runs = new Map((conversation.runs?.[branch] || []).map((run) => [run.turn_no, run]));
  clearNode($(`${branch}-messages`));
  if (!messages.length) { emptyState(branch); setStatus(branch, 'Ready'); return; }
  messages.filter((message) => message.role === 'user').forEach((message) => {
    const target = createTurn(branch, message.content);
    const run = runs.get(message.turn_no);
    if (run) showResult(branch, target, run);
  });
}

async function loadSessions() {
  const sessions = await request('/v1/conversations');
  $('session-count').textContent = sessions.length;
  const list = $('session-list');
  clearNode(list);
  if (!sessions.length) {
    const empty = document.createElement('div');
    empty.className = 'sidebar-empty';
    empty.textContent = 'No conversations yet. Start a new chat.';
    list.append(empty);
  }
  sessions.forEach((session) => {
    const row = document.createElement('div');
    row.className = 'session-row';
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `session-item${session.conversation_id === state.sessionId ? ' active' : ''}`;
    button.disabled = state.busy;
    const icon = document.createElement('span');
    icon.className = 'session-icon';
    icon.textContent = '◌';
    const content = document.createElement('span');
    const title = document.createElement('span');
    title.className = 'session-title';
    title.textContent = session.title || 'New chat';
    const date = document.createElement('span');
    date.className = 'session-date';
    date.textContent = new Intl.DateTimeFormat('en-IN', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', timeZone: 'Asia/Kolkata' }).format(new Date(session.created_at));
    content.append(title, date);
    button.append(icon, content);
    button.addEventListener('click', () => selectSession(session.conversation_id).catch((error) => toast(error.message)));
    const deleteButton = document.createElement('button');
    deleteButton.type = 'button';
    deleteButton.className = 'delete-session';
    deleteButton.disabled = state.busy;
    deleteButton.setAttribute('aria-label', `Delete ${session.title || 'chat'}`);
    deleteButton.title = 'Delete chat';
    deleteButton.textContent = '×';
    deleteButton.addEventListener('click', () => {
      if (state.busy) return;
      pendingDelete = session;
      $('delete-chat-title').textContent = session.title || 'New chat';
      $('delete-dialog').returnValue = 'cancel';
      $('delete-dialog').showModal();
    });
    row.append(button, deleteButton);
    list.append(row);
  });
  return sessions;
}

async function selectSession(id, navigation = 'push') {
  if (state.busy) return;
  const conversation = await request(`/v1/conversations/${encodeURIComponent(id)}`);
  setSessionId(id, navigation);
  BRANCHES.forEach((branch) => renderStored(branch, conversation));
  await loadSessions();
}

async function newChat(navigation = 'push') {
  if (state.busy) return;
  setSessionId(null, navigation);
  $('question').value = '';
  state.live = {};
  BRANCHES.forEach((branch) => { emptyState(branch); setStatus(branch, 'Ready'); });
  document.querySelectorAll('.session-item').forEach((button) => button.classList.remove('active'));
  $('question').focus();
}

function selectModelTab(branch) {
  BRANCHES.forEach((name) => {
    document.querySelector(`.${name}-panel`).classList.toggle('active-mobile', name === branch);
    $(`tab-${name}`).setAttribute('aria-pressed', String(name === branch));
  });
}

function handleEvent(event, live) {
  const branch = event.branch;
  if (event.type === 'started') {
    loadSessions().catch((error) => toast(error.message));
  } else if (event.type === 'step' && branch) {
    updateTrace(branch, event, live[branch]);
  } else if (event.type === 'answer_reset' && branch) {
    live[branch].answer.textContent = 'Checking tool results…';
    live[branch].answer.classList.remove('streaming');
    live[branch].started = false;
  } else if (event.type === 'token' && branch) {
    const target = live[branch];
    if (!target) return;
    if (!target.started) { target.started = true; target.answer.textContent = ''; target.answer.classList.add('streaming'); }
    target.answer.textContent += event.delta || '';
    const container = $(`${branch}-messages`);
    container.scrollTop = container.scrollHeight;
    setStatus(branch, 'Streaming', 'running');
  } else if (event.type === 'result' && branch) {
    live[branch].finished = true;
    showResult(branch, live[branch], event.result);
  }
}

async function sendQuestion(question) {
  if (state.busy || !question.trim()) return;
  if (!state.sessionId) {
    const created = await request('/v1/conversations', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}'
    });
    setSessionId(created.conversation_id, 'replace');
  }
  const sessionId = state.sessionId;
  const text = question.trim();
  $('question').value = '';
  BRANCHES.forEach((branch) => { setStatus(branch, 'Queued', 'running'); });
  const live = Object.fromEntries(BRANCHES.map((branch) => [branch, { ...createTurn(branch, text), started: false, finished: false }]));
  state.live = live;
  setBusy(true);
  try {
    const response = await fetch(`${API}/v1/conversations/${encodeURIComponent(sessionId)}/turns/stream`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body: JSON.stringify({ question: text })
    });
    if (!response.ok) {
      let detail = `Request failed (${response.status})`;
      try { detail = (await response.json()).detail || detail; } catch (_) { /* no JSON body */ }
      throw new Error(detail);
    }
    if (!response.body) throw new Error('Streaming response is unavailable in this browser.');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const blocks = buffer.split('\n\n');
      buffer = blocks.pop() || '';
      for (const block of blocks) {
        const line = block.split('\n').find((entry) => entry.startsWith('data: '));
        if (!line) continue;
        handleEvent(JSON.parse(line.slice(6)), live);
      }
      if (done) break;
    }
    await loadSessions();
  } catch (error) {
    toast(error.message);
    BRANCHES.forEach((branch) => {
      if (!live[branch].finished) showResult(branch, live[branch], { status: 'error', error: error.message });
    });
  } finally {
    setBusy(false);
  }
}

async function checkHealth() {
  try {
    await request('/health');
    $('api-status').className = 'connection online';
    $('api-status').lastElementChild.textContent = 'Services online';
  } catch (_) {
    $('api-status').className = 'connection offline';
    $('api-status').lastElementChild.textContent = 'API offline';
  }
}

async function initialize() {
  BRANCHES.forEach((branch) => { emptyState(branch); });
  await checkHealth();
  try {
    const sessions = await loadSessions();
    if (location.pathname === '/chat/new') { await newChat('none'); return; }
    const pathId = sessionFromPath();
    if (pathId) { await selectSession(pathId, 'none'); return; }
    const remembered = localStorage.getItem('rag-lab-session');
    const selected = sessions.find((session) => session.conversation_id === remembered) || sessions[0];
    if (selected) await selectSession(selected.conversation_id, 'replace');
    else await newChat('replace');
  } catch (error) {
    toast(`Could not load chats: ${error.message}`);
    await newChat('replace');
  }
}

$('new-chat').addEventListener('click', () => newChat().catch((error) => toast(error.message)));
$('delete-dialog').addEventListener('close', () => { pendingDelete = null; });
$('confirm-delete').addEventListener('click', async (event) => {
  event.preventDefault();
  const session = pendingDelete;
  pendingDelete = null;
  $('delete-dialog').close('confirm');
  if (!session) return;
  try {
    await request(`/v1/conversations/${encodeURIComponent(session.conversation_id)}`, { method: 'DELETE' });
    if (state.sessionId === session.conversation_id) await newChat('replace');
    await loadSessions();
    toast('Chat deleted');
  } catch (error) { toast(`Delete failed: ${error.message}`); }
});
window.addEventListener('popstate', () => {
  const id = sessionFromPath();
  (id ? selectSession(id, 'none') : newChat('none')).catch((error) => toast(error.message));
});
$('tab-slm').addEventListener('click', () => selectModelTab('slm'));
$('tab-mistral').addEventListener('click', () => selectModelTab('mistral'));
$('composer').addEventListener('submit', (event) => {
  event.preventDefault();
  sendQuestion($('question').value).catch((error) => toast(error.message));
});
$('question').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); $('composer').requestSubmit(); }
});
document.addEventListener('keydown', (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault(); newChat().catch((error) => toast(error.message));
  }
});
initialize();
setInterval(checkHealth, 30000);
