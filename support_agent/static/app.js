"use strict";

const element = (id) => document.getElementById(id);
const ui = {
  user: element("user-select"), summary: element("user-summary"),
  tickets: element("ticket-list"), count: element("ticket-count"),
  title: element("conversation-title"), subtitle: element("conversation-subtitle"),
  reset: element("reset-button"), error: element("error-banner"),
  errorMessage: element("error-message"), retry: element("retry-button"),
  loading: element("loading-state"), scroll: element("chat-scroll"),
  empty: element("empty-state"), emptyDescription: element("empty-description"),
  examples: element("examples"), messages: element("messages"),
  thinking: element("thinking"), form: element("chat-form"),
  input: element("message-input"), send: element("send-button"),
};

const state = {
  users: [], tickets: [], userId: null, ticketId: null, turns: [],
  pending: false, ready: false, version: 0, controller: null,
};

function node(tag, className, text) {
  const result = document.createElement(tag);
  if (className) result.className = className;
  if (text !== undefined && text !== null) result.textContent = String(text);
  return result;
}

function contextQuery() {
  const params = new URLSearchParams({ user_id: state.userId });
  if (state.ticketId) params.set("ticket_id", state.ticketId);
  return params.toString();
}

async function request(path, options = {}) {
  const response = await fetch(path, { credentials: "same-origin", ...options });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = typeof data?.detail === "string" ? data.detail : "Не удалось выполнить запрос. Попробуйте ещё раз.";
    throw new Error(detail);
  }
  if (data === null) throw new Error("Сервис вернул пустой ответ. Попробуйте ещё раз.");
  return data;
}

function showError(error, canRetry = false) {
  ui.errorMessage.textContent = error instanceof TypeError
    ? "Не удалось связаться с сервисом. Проверьте соединение и повторите попытку."
    : error.message || "Не удалось выполнить запрос. Попробуйте ещё раз.";
  ui.retry.hidden = !canRetry;
  ui.error.hidden = false;
}

function clearError() {
  ui.error.hidden = true;
  ui.errorMessage.textContent = "";
  ui.retry.hidden = true;
}

function syncControls() {
  ui.user.disabled = state.pending || !state.users.length;
  ui.tickets.querySelectorAll("button").forEach((button) => { button.disabled = state.pending; });
  ui.examples.querySelectorAll("button").forEach((button) => { button.disabled = state.pending || !state.ready; });
  ui.input.disabled = state.pending || !state.ready;
  ui.send.disabled = state.pending || !state.ready || !ui.input.value.trim();
  ui.reset.disabled = state.pending || !state.ready || !state.turns.length;
  ui.retry.disabled = state.pending;
  ui.form.setAttribute("aria-busy", String(state.pending));
}

function beginRequest({ loading = false } = {}) {
  state.controller?.abort();
  state.controller = new AbortController();
  state.version += 1;
  state.pending = true;
  clearError();
  ui.loading.hidden = !loading;
  syncControls();
  return { version: state.version, signal: state.controller.signal };
}

function finishRequest(version) {
  if (version !== state.version) return;
  state.pending = false;
  ui.loading.hidden = true;
  ui.thinking.hidden = true;
  syncControls();
}

const statusLabels = {
  open: "Открыто", in_progress: "В работе", pending: "Ожидает ответа",
  waiting: "Ожидает ответа", waiting_user: "Ожидает ответа", resolved: "Решено", closed: "Закрыто",
};

function renderTickets() {
  ui.tickets.replaceChildren();
  ui.count.textContent = state.tickets.length;
  const choices = [{ id: null, subject: "Общий вопрос", status: null }, ...state.tickets];
  for (const ticket of choices) {
    const button = node("button", "ticket-button");
    button.type = "button";
    const active = ticket.id === state.ticketId;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
    button.append(node("span", "ticket-subject", ticket.subject));
    const meta = node("span", "ticket-meta");
    if (ticket.id) {
      meta.append(node("span", "ticket-id", ticket.id));
      meta.append(node("span", "ticket-status", statusLabels[ticket.status] || ticket.status || "Обращение"));
    } else {
      meta.textContent = "Помощь по работе с продуктом";
    }
    button.append(meta);
    button.addEventListener("click", () => {
      if (!state.pending && ticket.id !== state.ticketId) selectTicket(ticket.id);
    });
    ui.tickets.append(button);
  }
  if (!state.tickets.length) ui.tickets.append(node("p", "no-tickets", "У этого пользователя пока нет обращений."));
}

function renderContextHeader() {
  const user = state.users.find((item) => item.id === state.userId);
  const ticket = state.tickets.find((item) => item.id === state.ticketId);
  ui.summary.textContent = user?.summary || "";
  ui.title.textContent = ticket?.subject || "Общий вопрос";
  ui.subtitle.textContent = [user?.name, ticket?.id || "По документации LLMAgent"].filter(Boolean).join(" · ");
  ui.emptyDescription.textContent = ticket
    ? "Данные этого обращения помогут разобраться в проблеме. Задайте вопрос или уточните, что уже пробовали."
    : "Расскажите, что хотите сделать и что происходит. Ассистент найдёт подходящую инструкцию в документации.";
  const examples = ticket
    ? ["Помоги разобраться с проблемой из этого обращения", "Что мне проверить в первую очередь?", "Какие данные нужны для диагностики?"]
    : ["Как подключить MCP-сервер?", "Почему агент не находит ответ в документации?", "Как использовать локальную модель?"];
  ui.examples.replaceChildren();
  for (const question of examples) {
    const button = node("button", "example-button");
    button.type = "button";
    button.append(node("span", "", question), node("span", "example-arrow", "↗"));
    button.lastChild.setAttribute("aria-hidden", "true");
    button.addEventListener("click", () => {
      ui.input.value = question;
      syncControls();
      ui.input.focus();
    });
    ui.examples.append(button);
  }
}

const fieldLabels = {
  id: "ID", name: "Имя", display_name: "Имя", summary: "Описание",
  user_id: "Пользователь", subject: "Тема", title: "Тема", status: "Статус",
  description: "Описание", error: "Ошибка", error_code: "Код ошибки",
  created_at: "Создано", updated_at: "Обновлено", platform: "Платформа",
  os: "ОС", python_version: "Версия Python", app_version: "Версия приложения",
  model: "Модель", provider: "Провайдер", configuration: "Конфигурация",
  config: "Конфигурация", settings: "Настройки", environment: "Окружение",
  setup: "Настройка", facts: "Известные факты", messages: "Переписка",
  events: "События", attempts: "Что уже пробовали", attempted_steps: "Что уже пробовали",
  resolution: "Решение", notes: "Примечания", role: "Автор", text: "Текст",
  content: "Текст", message: "Сообщение", at: "Время", timestamp: "Время",
  from: "Автор", category: "Категория", rag_enabled: "RAG включён",
  mcp_enabled: "MCP включён", index_status: "Состояние индекса",
  interface: "Интерфейс", observed_at: "Время наблюдения",
};

function readableValue(value) {
  if (value === null || value === undefined || value === "") return "Не указано";
  if (typeof value === "boolean") return value ? "Да" : "Нет";
  return String(value);
}

function renderData(value, depth = 0, field = "") {
  if (typeof value !== "object" || value === null) {
    const labels = {
      status: statusLabels,
      index_status: { ready: "Готов", missing: "Отсутствует" },
      role: { user: "Пользователь", assistant: "Ассистент", support: "Поддержка" },
    };
    return node("span", "", labels[field]?.[value] || readableValue(value));
  }
  if (depth > 4) return node("pre", "context-text", JSON.stringify(value, null, 2));
  if (Array.isArray(value)) {
    if (!value.length) return node("span", "muted", "Нет записей");
    const list = node("ul", "context-list");
    for (const item of value) {
      const entry = node("li");
      entry.append(renderData(item, depth + 1));
      list.append(entry);
    }
    return list;
  }
  const list = node("dl", "context-fields");
  for (const [key, item] of Object.entries(value)) {
    list.append(node("dt", "", fieldLabels[key] || key.replaceAll("_", " ")));
    const description = node("dd");
    description.append(renderData(item, depth + 1, key));
    list.append(description);
  }
  return list;
}

function renderEvidence(turn) {
  const wrapper = node("div", "evidence");
  const sources = Array.isArray(turn.sources) ? turn.sources : [];
  if (sources.length) {
    const details = node("details", "evidence-section");
    details.append(node("summary", "", `Источники документации · ${sources.length}`));
    const list = node("div", "source-list");
    for (const source of sources) {
      const article = node("article", "source-card");
      article.append(node("h4", "source-title", source.section || source.source || "Документация"));
      article.append(node("p", "source-location", [source.id, source.source].filter(Boolean).join(" · ")));
      if (source.text) article.append(node("p", "source-text", source.text));
      list.append(article);
    }
    details.append(list);
    wrapper.append(details);
  }
  if (turn.context?.user || turn.context?.ticket) {
    const details = node("details", "evidence-section");
    details.append(node("summary", "", "Какие данные учтены"));
    const content = node("div", "context-content");
    for (const [key, label] of [["user", "Профиль пользователя"], ["ticket", "Обращение"]]) {
      if (!turn.context[key]) continue;
      content.append(node("h4", "", label), renderData(turn.context[key]));
    }
    if (turn.context.demo) content.append(node("p", "context-demo", "Демонстрационные данные"));
    details.append(content);
    wrapper.append(details);
  }
  return wrapper;
}

function renderQuestion(question) {
  const article = node("article", "message message-user");
  article.append(node("div", "message-author", "Вы"), node("p", "message-text", question));
  return article;
}

function renderTurn(turn) {
  const group = node("div", "turn");
  group.append(renderQuestion(turn.question));
  const answer = node("article", "message message-assistant");
  const author = node("div", "message-author");
  author.append(node("span", "assistant-avatar", "LA"), node("span", "", "Поддержка LLMAgent"));
  answer.append(author, node("p", "message-text", turn.answer));
  for (const warning of Array.isArray(turn.warnings) ? turn.warnings : []) {
    answer.append(node("p", "answer-warning", warning));
  }
  answer.append(renderEvidence(turn));
  group.append(answer);
  return group;
}

function renderHistory() {
  ui.messages.replaceChildren(...state.turns.map(renderTurn));
  ui.empty.hidden = !state.ready || state.turns.length > 0;
}

function scrollToLatest() {
  requestAnimationFrame(() => { ui.scroll.scrollTop = ui.scroll.scrollHeight; });
}

async function fetchHistory(signal) {
  const data = await request(`/api/history?${contextQuery()}`, { signal });
  return Array.isArray(data.turns) ? data.turns : [];
}

async function selectTicket(ticketId) {
  state.ticketId = ticketId;
  state.ready = false;
  state.turns = [];
  ui.input.value = "";
  renderTickets();
  renderContextHeader();
  renderHistory();
  const { version, signal } = beginRequest({ loading: true });
  try {
    const turns = await fetchHistory(signal);
    if (version !== state.version) return;
    state.turns = turns;
    state.ready = true;
    renderHistory();
    scrollToLatest();
  } catch (error) {
    if (version === state.version && error.name !== "AbortError") showError(error, true);
  } finally {
    finishRequest(version);
  }
}

async function selectUser(userId) {
  state.userId = userId;
  state.ticketId = null;
  state.tickets = [];
  state.turns = [];
  state.ready = false;
  ui.input.value = "";
  renderContextHeader();
  renderHistory();
  ui.tickets.replaceChildren(node("p", "muted", "Загружаем обращения…"));
  ui.count.textContent = "0";
  const { version, signal } = beginRequest({ loading: true });
  try {
    const [ticketsData, turns] = await Promise.all([
      request(`/api/tickets?${new URLSearchParams({ user_id: userId })}`, { signal }),
      fetchHistory(signal),
    ]);
    if (version !== state.version) return;
    state.tickets = Array.isArray(ticketsData.tickets) ? ticketsData.tickets : [];
    state.turns = turns;
    state.ready = true;
    renderTickets();
    renderContextHeader();
    renderHistory();
    scrollToLatest();
  } catch (error) {
    if (version === state.version && error.name !== "AbortError") showError(error, true);
  } finally {
    finishRequest(version);
  }
}

async function initialize() {
  const { version, signal } = beginRequest({ loading: true });
  let selectedUser = null;
  try {
    const data = await request("/api/users", { signal });
    if (version !== state.version) return;
    state.users = Array.isArray(data.users) ? data.users : [];
    if (!state.users.length) throw new Error("Демонстрационные пользователи пока не настроены.");
    ui.user.replaceChildren(...state.users.map((user) => {
      const option = node("option", "", user.name);
      option.value = user.id;
      return option;
    }));
    selectedUser = state.users[0].id;
  } catch (error) {
    if (version === state.version && error.name !== "AbortError") showError(error, true);
  } finally {
    finishRequest(version);
  }
  if (selectedUser && version === state.version) await selectUser(selectedUser);
}

async function sendMessage(event) {
  event.preventDefault();
  const message = ui.input.value.trim();
  if (!message || state.pending || !state.ready) return;
  const { version, signal } = beginRequest();
  const draft = renderQuestion(message);
  ui.messages.append(draft);
  ui.empty.hidden = true;
  ui.thinking.hidden = false;
  scrollToLatest();
  try {
    const turn = await request("/api/chat", {
      method: "POST", signal, headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: state.userId, ticket_id: state.ticketId, message }),
    });
    if (version !== state.version) return;
    state.turns = [...state.turns, turn].slice(-5);
    ui.input.value = "";
    renderHistory();
    scrollToLatest();
  } catch (error) {
    if (version === state.version && error.name !== "AbortError") {
      draft.remove();
      ui.empty.hidden = state.turns.length > 0;
      showError(error);
    }
  } finally {
    finishRequest(version);
    if (version === state.version) ui.input.focus();
  }
}

async function resetHistory() {
  if (state.pending || !state.ready || !state.turns.length) return;
  const { version, signal } = beginRequest();
  try {
    await request(`/api/history?${contextQuery()}`, { method: "DELETE", signal });
    if (version !== state.version) return;
    state.turns = [];
    ui.input.value = "";
    renderHistory();
  } catch (error) {
    if (version === state.version && error.name !== "AbortError") showError(error);
  } finally {
    finishRequest(version);
    if (version === state.version) ui.input.focus();
  }
}

ui.user.addEventListener("change", () => {
  if (!state.pending) selectUser(ui.user.value);
});
ui.retry.addEventListener("click", () => {
  if (state.pending) return;
  if (!state.users.length) initialize();
  else if (!state.tickets.length) selectUser(state.userId);
  else selectTicket(state.ticketId);
});
ui.input.addEventListener("input", syncControls);
ui.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!ui.send.disabled) ui.form.requestSubmit();
  }
});
ui.form.addEventListener("submit", sendMessage);
ui.reset.addEventListener("click", resetHistory);
initialize();
