// MangARTI web translator

const DIRECTIONS = {
  "id-mdo": {
    source: "Bahasa Indonesia",
    target: "Bahasa Manado",
    placeholder: "Ketik satu kalimat, misalnya: Saya mau makan nasi goreng.",
    examples: [
      "Saya mau makan nasi goreng.",
      "Kamu mau pergi ke mana?",
      "Mereka belum tidur dari tadi.",
      "Kami sedang menonton televisi.",
      "Saya tidak mengerti maksudmu.",
    ],
  },
  "mdo-id": {
    source: "Bahasa Manado",
    target: "Bahasa Indonesia",
    placeholder: "Ketik satu kalimat, misalnya: Ngana mo pigi ka mana?",
    examples: [
      "Ngana mo pigi ka mana?",
      "Dorang balum tidor dari tadi.",
      "Torang musti pulang skarang.",
      "Kyapa ngana marah pa kita?",
      "Hari ini cuaca panas skali.",
    ],
  },
};

const DEFAULT_SETTINGS = { maxNewTokens: 128, doSample: false, temperature: 0.7, topP: 0.9 };
const MAX_CHARS = 500;
const STORAGE_KEY = "mangarti:prefs";
const IS_MAC = /Mac|iPhone|iPad/.test(navigator.platform);

const $ = (id) => document.getElementById(id);
const els = {
  status: $("status"),
  statusLabel: document.querySelector("#status .status-label"),
  sourceLang: $("source-lang"),
  targetLang: $("target-lang"),
  swap: $("swap"),
  source: $("source"),
  charCount: $("char-count"),
  clear: $("clear"),
  output: $("output"),
  outputMeta: $("output-meta"),
  copy: $("copy"),
  hint: $("hint"),
  translate: $("translate"),
  translateLabel: document.querySelector("#translate .btn-label"),
  examples: $("examples"),
  maxTokens: $("max-tokens"),
  maxTokensValue: $("max-tokens-value"),
  doSample: $("do-sample"),
  temperature: $("temperature"),
  temperatureValue: $("temperature-value"),
  topP: $("top-p"),
  topPValue: $("top-p-value"),
  resetSettings: $("reset-settings"),
};

const state = {
  direction: "id-mdo",
  modelState: "loading",
  busy: false,
  translation: "",
  settings: { ...DEFAULT_SETTINGS },
};

// ---------- Preferences (per browser, optional) ----------

function loadPrefs() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    if (saved.direction in DIRECTIONS) state.direction = saved.direction;
    if (saved.settings) state.settings = { ...DEFAULT_SETTINGS, ...saved.settings };
  } catch { /* storage unavailable: use defaults */ }
}

function savePrefs() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ direction: state.direction, settings: state.settings }));
  } catch { /* ignore */ }
}

// ---------- Rendering ----------

const DEFAULT_HINT = `<span class="kbd-hint"><kbd>${IS_MAC ? "⌘" : "Ctrl"}</kbd> + <kbd>Enter</kbd> untuk menerjemahkan</span>`;

function setHint(html, tone) {
  els.hint.innerHTML = html;
  if (tone) els.hint.dataset.tone = tone;
  else delete els.hint.dataset.tone;
}

function updateHint() {
  if (state.modelState === "loading") {
    return setHint("Model sedang dimuat. Saat pertama kali, ini bisa memakan waktu beberapa menit.");
  }
  if (state.modelState === "error") {
    return setHint("Model gagal dimuat. Periksa terminal tempat server dijalankan.", "error");
  }
  const sentences = els.source.value.trim().match(/[^.!?]+[.!?]+(\s|$)/g) || [];
  if (sentences.length > 1) {
    return setHint("Terdeteksi lebih dari satu kalimat. Model paling akurat untuk satu kalimat per terjemahan.", "warning");
  }
  setHint(DEFAULT_HINT);
}

function renderDirection() {
  const dir = DIRECTIONS[state.direction];
  els.sourceLang.textContent = dir.source;
  els.targetLang.textContent = dir.target;
  els.source.placeholder = dir.placeholder;
  els.source.lang = state.direction === "id-mdo" ? "id" : "xmm";

  els.examples.replaceChildren(
    ...dir.examples.map((text) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "btn chip";
      chip.textContent = text;
      chip.addEventListener("click", () => {
        els.source.value = text;
        onSourceInput();
        translate();
      });
      return chip;
    }),
  );
}

function renderSourceMeta() {
  const length = els.source.value.length;
  els.charCount.textContent = `${length} / ${MAX_CHARS}`;
  if (length >= MAX_CHARS * 0.9) els.charCount.dataset.limit = "near";
  else delete els.charCount.dataset.limit;
  els.clear.hidden = length === 0;
}

function renderTranslateButton() {
  const ready = state.modelState === "ready" || state.modelState === "mock";
  els.translate.disabled = !ready || state.busy || !els.source.value.trim();
  els.translate.setAttribute("aria-busy", String(state.busy));
  els.translateLabel.textContent = state.busy ? "Menerjemahkan…" : "Terjemahkan";
}

function renderOutput(kind, payload = {}) {
  els.output.dataset.state = kind;
  els.output.setAttribute("aria-busy", String(kind === "loading"));
  els.copy.hidden = kind !== "result";
  els.outputMeta.textContent = "";

  if (kind === "empty") {
    els.output.innerHTML = '<p class="output-placeholder">Terjemahan akan muncul di sini.</p>';
  } else if (kind === "loading") {
    els.output.innerHTML = '<div class="skeleton" aria-label="Menerjemahkan"><span></span><span></span></div>';
  } else if (kind === "error") {
    const p = document.createElement("p");
    p.className = "output-error";
    p.textContent = payload.message;
    els.output.replaceChildren(p);
  } else if (kind === "result") {
    const p = document.createElement("p");
    p.className = "output-text";
    p.lang = state.direction === "id-mdo" ? "xmm" : "id";
    p.textContent = payload.text;
    els.output.replaceChildren(p);
    els.outputMeta.textContent = `${payload.seconds.toLocaleString("id-ID")} dtk`;
  }
}

function renderSettings() {
  const s = state.settings;
  els.maxTokens.value = s.maxNewTokens;
  els.maxTokensValue.textContent = s.maxNewTokens;
  els.doSample.checked = s.doSample;
  els.doSample.setAttribute("aria-checked", String(s.doSample));
  els.temperature.value = s.temperature;
  els.temperatureValue.textContent = Number(s.temperature).toFixed(1);
  els.topP.value = s.topP;
  els.topPValue.textContent = Number(s.topP).toFixed(2);
  for (const range of [els.maxTokens, els.temperature, els.topP]) {
    const pct = ((range.value - range.min) / (range.max - range.min)) * 100;
    range.style.setProperty("--fill", `${pct}%`);
  }
  for (const field of document.querySelectorAll("[data-requires-sampling]")) {
    field.dataset.disabled = String(!s.doSample);
    field.querySelector("input").disabled = !s.doSample;
  }
}

function renderStatus() {
  const labels = {
    loading: "Memuat model…",
    ready: "Model siap",
    mock: "Mode uji",
    error: "Model gagal dimuat",
  };
  els.status.dataset.state = state.modelState;
  els.statusLabel.textContent = labels[state.modelState];
}

// ---------- Actions ----------

function onSourceInput() {
  renderSourceMeta();
  renderTranslateButton();
  updateHint();
}

async function translate() {
  const text = els.source.value.trim();
  if (!text || state.busy || els.translate.disabled) return;

  state.busy = true;
  renderTranslateButton();
  renderOutput("loading");

  try {
    const s = state.settings;
    const response = await fetch("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        direction: state.direction,
        max_new_tokens: Number(s.maxNewTokens),
        do_sample: s.doSample,
        temperature: Number(s.temperature),
        top_p: Number(s.topP),
      }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = typeof data.detail === "string" ? data.detail : "Permintaan tidak valid.";
      throw new Error(detail);
    }
    state.translation = data.translation;
    renderOutput("result", { text: data.translation || "(hasil kosong)", seconds: data.seconds });
  } catch (error) {
    state.translation = "";
    const message = error instanceof TypeError
      ? "Tidak dapat terhubung ke server. Pastikan app.py masih berjalan."
      : error.message;
    renderOutput("error", { message });
  } finally {
    state.busy = false;
    renderTranslateButton();
  }
}

function swapDirection() {
  state.direction = state.direction === "id-mdo" ? "mdo-id" : "id-mdo";
  if (state.translation) {
    els.source.value = state.translation;
    state.translation = "";
    renderOutput("empty");
  }
  renderDirection();
  onSourceInput();
  savePrefs();
  els.source.focus();
}

async function copyTranslation() {
  const label = els.copy.querySelector("span");
  try {
    await navigator.clipboard.writeText(state.translation);
    label.textContent = "Tersalin";
  } catch {
    label.textContent = "Gagal menyalin";
  }
  setTimeout(() => { label.textContent = "Salin"; }, 1600);
}

function updateSetting(key, value) {
  state.settings[key] = value;
  renderSettings();
  savePrefs();
}

// ---------- Model status polling ----------

async function pollStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    state.modelState = data.state === "ready" && data.mock ? "mock" : data.state;
    if (data.state === "error") els.status.title = data.error || "";
  } catch {
    state.modelState = "error";
  }
  renderStatus();
  renderTranslateButton();
  updateHint();
  if (state.modelState === "loading") setTimeout(pollStatus, 2000);
}

// ---------- Wire up ----------

loadPrefs();
renderDirection();
renderSettings();
renderSourceMeta();
renderStatus();
renderTranslateButton();
updateHint();
pollStatus();

els.source.addEventListener("input", onSourceInput);
els.source.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    translate();
  }
});
els.clear.addEventListener("click", () => {
  els.source.value = "";
  state.translation = "";
  renderOutput("empty");
  onSourceInput();
  els.source.focus();
});
els.translate.addEventListener("click", translate);
els.swap.addEventListener("click", swapDirection);
els.copy.addEventListener("click", copyTranslation);

els.maxTokens.addEventListener("input", (e) => updateSetting("maxNewTokens", Number(e.target.value)));
els.doSample.addEventListener("change", (e) => updateSetting("doSample", e.target.checked));
els.temperature.addEventListener("input", (e) => updateSetting("temperature", Number(e.target.value)));
els.topP.addEventListener("input", (e) => updateSetting("topP", Number(e.target.value)));
els.resetSettings.addEventListener("click", () => {
  state.settings = { ...DEFAULT_SETTINGS };
  renderSettings();
  savePrefs();
});
