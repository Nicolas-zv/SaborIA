/**
 * SaborIA – Lógica Frontend y Conexión con API REST
 * Desarrollado por Roberto (@Roberto) para SaborIA
 */

// URL base del backend API REST (a desarrollar por Nicolás en el puerto 8000)
const API_BASE = "http://localhost:8000";

// Datos por defecto (fallback precargado de models/metricas_regresion_logistica.json)
const METRICAS_FALLBACK = {
  accuracy: 0.6986,
  precision_weighted: 0.6989,
  recall_weighted: 0.6986,
  f1_weighted: 0.6980,
  matriz_confusion: [
    [27, 6, 9],
    [9, 12, 5],
    [10, 5, 63]
  ],
  baseline_dummy: {
    accuracy: 0.5342,
    f1_weighted: 0.3721
  }
};

const ERRORES_FALLBACK = [
  {
    etiqueta_real: "neutral",
    etiqueta_predicha: "negativo",
    confianza_predicha: 0.550,
    texto_resena: "Nos ofrecieron una mesa junto a la puerta que resultaba muy incómoda por las corrientes, aunque la comida estuvo aceptable.",
    explicacion: "El léxico negativo inicial ('muy incómoda') sobrepesó el término templado 'aceptable', llevando al clasificador hacia negativo."
  },
  {
    etiqueta_real: "positivo",
    etiqueta_predicha: "negativo",
    confianza_predicha: 0.557,
    texto_resena: "La carne estaba tan tierna que sospecho que fue masajeada por ángeles antes de llegar a la parrilla.",
    explicacion: "Uso de lenguaje hiperbólico y metafórico ('sospecho', 'masajeada') sin los adjetivos típicos de elogio directo, confundiendo la polaridad."
  },
  {
    etiqueta_real: "neutral",
    etiqueta_predicha: "negativo",
    confianza_predicha: 0.451,
    texto_resena: "La relación calidad-precio no es del todo competitiva comparada con locales similares.",
    explicacion: "La presencia de la negación ('no es del todo competitiva') fue interpretada como una crítica severa más que una valoración neutra/moderada."
  }
];

// Inicialización cuando carga el DOM
document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupPresets();
  setupPrediction();
  setupBatch();
  setupPipeline();
  checkApiHealth();
  renderMetrics(METRICAS_FALLBACK);
  renderErrorCases(ERRORES_FALLBACK);
  initVisualEffects();
});

/* ==========================================================================
   1. Navegación por Pestañas (Tabs)
   ========================================================================== */
function setupTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  const panes = document.querySelectorAll(".tab-pane");

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetId = tab.getAttribute("data-tab");

      tabs.forEach(t => t.classList.remove("active"));
      panes.forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add("active");
      }
    });
  });

  // Contador de caracteres
  const textarea = document.getElementById("input-review");
  const counter = document.getElementById("char-counter");
  if (textarea && counter) {
    textarea.addEventListener("input", () => {
      counter.textContent = `${textarea.value.length} caracteres`;
    });
  }

  // Botón para saltar desde el resultado al visualizador de pipeline
  const btnJumpPipeline = document.getElementById("btn-ver-en-pipeline");
  if (btnJumpPipeline) {
    btnJumpPipeline.addEventListener("click", () => {
      const reviewText = document.getElementById("input-review").value.trim();
      document.getElementById("tab-btn-pipeline").click();
      if (reviewText) {
        const pipelineInput = document.getElementById("pipeline-text-input");
        pipelineInput.value = reviewText;
        document.getElementById("btn-ejecutar-pipeline").click();
      }
    });
  }
}

/* ==========================================================================
   2. Verificación de Salud de la API
   ========================================================================== */
async function checkApiHealth() {
  const badge = document.getElementById("api-status-badge");
  const statusText = document.getElementById("api-status-text");

  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET" });
    if (res.ok) {
      const data = await res.json();
      badge.style.borderColor = "rgba(16, 185, 129, 0.4)";
      statusText.innerHTML = `API Conectada • ${data.modelo_cargado ? "Modelo Activo" : "Sin Modelo"}`;
      fetchRemoteMetrics();
      fetchRemoteErrors();
    } else {
      throw new Error("HTTP error " + res.status);
    }
  } catch (err) {
    badge.style.borderColor = "rgba(245, 158, 11, 0.4)";
    statusText.textContent = "API Desconectada (Modo Local / Demo)";
  }
}

/* ==========================================================================
   3. Botones de Ejemplos Rápidos
   ========================================================================== */
function setupPresets() {
  const presetButtons = document.querySelectorAll(".btn-preset");
  const inputReview = document.getElementById("input-review");

  presetButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const text = btn.getAttribute("data-text");
      inputReview.value = text;
      document.getElementById("char-counter").textContent = `${text.length} caracteres`;
      ejecutarPrediccion(text);
    });
  });

  const btnLimpiar = document.getElementById("btn-limpiar");
  if (btnLimpiar) {
    btnLimpiar.addEventListener("click", () => {
      inputReview.value = "";
      document.getElementById("char-counter").textContent = "0 caracteres";
      document.getElementById("result-placeholder").classList.remove("hidden");
      document.getElementById("result-display").classList.add("hidden");
      resetAtmosphereTheme();
    });
  }
}

/* ==========================================================================
   4. Inferencia Individual
   ========================================================================== */
function setupPrediction() {
  const btnPredecir = document.getElementById("btn-predecir");
  const inputReview = document.getElementById("input-review");

  btnPredecir.addEventListener("click", () => {
    const texto = inputReview.value.trim();
    if (!texto) {
      alert("Por favor escribe una reseña antes de analizar.");
      inputReview.focus();
      return;
    }
    ejecutarPrediccion(texto);
  });
}

async function ejecutarPrediccion(texto) {
  const btn = document.getElementById("btn-predecir");
  const placeholder = document.getElementById("result-placeholder");
  const display = document.getElementById("result-display");

  btn.disabled = true;
  btn.innerHTML = `<span class="btn-icon">⏳</span> Evaluando...`;

  try {
      const res = await fetch(`${API_BASE}/predict/logistic`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texto }),
    });

    if (!res.ok) {
      throw new Error(`Error en servidor: ${res.status}`);
    }

    const data = await res.json();
    renderResultadoInferencia(data);
  } catch (err) {
    console.warn("Fallo API, usando heurística local:", err);
    // Simulación local heurística elegante si la API no está en línea
    const dataLocal = inferenciaLocalSimulada(texto);
    renderResultadoInferencia(dataLocal);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span class="btn-icon">⚡</span> Analizar Sentimiento`;
    placeholder.classList.add("hidden");
    display.classList.remove("hidden");
  }
}

function renderResultadoInferencia(data) {
  const badge = document.getElementById("sentiment-badge");
  const icon = document.getElementById("sentiment-icon");
  const name = document.getElementById("sentiment-name");
  const conf = document.getElementById("confidence-value");

  const sent = data.sentimiento_predicho.toLowerCase();
  name.textContent = sent.toUpperCase();
  conf.textContent = `${(data.confianza * 100).toFixed(1)}%`;

  badge.className = "sentiment-badge";
  if (sent === "positivo") {
    badge.classList.add("badge-positivo");
    icon.textContent = "😋";
  } else if (sent === "negativo") {
    badge.classList.add("badge-negativo");
    icon.textContent = "😡";
  } else {
    badge.classList.add("badge-neutral");
    icon.textContent = "⚖️";
  }
  updateAtmosphereTheme(sent);

  // Actualizar barras de probabilidades
  const probs = data.probabilidades || { positivo: 0.33, neutral: 0.33, negativo: 0.33 };
  const pPos = (probs.positivo || 0) * 100;
  const pNeu = (probs.neutral || 0) * 100;
  const pNeg = (probs.negativo || 0) * 100;

  document.getElementById("prob-val-positivo").textContent = `${pPos.toFixed(1)}%`;
  document.getElementById("prob-fill-positivo").style.width = `${pPos}%`;

  document.getElementById("prob-val-neutral").textContent = `${pNeu.toFixed(1)}%`;
  document.getElementById("prob-fill-neutral").style.width = `${pNeu}%`;

  document.getElementById("prob-val-negativo").textContent = `${pNeg.toFixed(1)}%`;
  document.getElementById("prob-fill-negativo").style.width = `${pNeg}%`;
}

function inferenciaLocalSimulada(texto) {
  const lower = texto.toLowerCase();
  let posScore = 0.33;
  let negScore = 0.33;
  let neuScore = 0.34;

  const palabrasPos = ["exquisita", "rico", "bueno", "excelente", "delicioso", "maravilloso", "impecable", "diez", "perfecto", "me encanto", "encantó", "genial"];
  const palabrasNeg = ["pésimo", "pesimo", "horrible", "malo", "mala", "frio", "fría", "tardaron", "groseros", "nunca", "desastre", "cruda", "sucio"];

  palabrasPos.forEach(p => { if (lower.includes(p)) posScore += 0.25; });
  palabrasNeg.forEach(p => { if (lower.includes(p)) negScore += 0.25; });

  if (lower.includes("no") || lower.includes("tampoco")) {
    negScore += 0.15;
  }

  const suma = posScore + negScore + neuScore;
  const probPos = posScore / suma;
  const probNeg = negScore / suma;
  const probNeu = neuScore / suma;

  let maxSent = "neutral";
  let maxConf = probNeu;
  if (probPos > probNeg && probPos > probNeu) {
    maxSent = "positivo";
    maxConf = probPos;
  } else if (probNeg > probPos && probNeg > probNeu) {
    maxSent = "negativo";
    maxConf = probNeg;
  }

  return {
    texto,
    sentimiento_predicho: maxSent,
    confianza: maxConf,
    probabilidades: {
      positivo: Number(probPos.toFixed(4)),
      neutral: Number(probNeu.toFixed(4)),
      negativo: Number(probNeg.toFixed(4)),
    }
  };
}

/* ==========================================================================
   5. Inferencia por Lotes (Masiva)
   ========================================================================== */
function setupBatch() {
  const toggle = document.getElementById("batch-toggle");
  const content = document.getElementById("batch-content");
  const btnUpload = document.getElementById("btn-upload-file");
  const fileInput = document.getElementById("batch-file-input");
  const btnBatchPredict = document.getElementById("btn-batch-predict");
  const batchTextarea = document.getElementById("batch-input");

  if (toggle && content) {
    toggle.addEventListener("click", () => {
      content.classList.toggle("collapsed");
      toggle.querySelector(".toggle-icon").textContent = content.classList.contains("collapsed") ? "▼" : "▲";
    });
  }

  if (btnUpload && fileInput) {
    btnUpload.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (evt) => {
          batchTextarea.value = evt.target.result;
        };
        reader.readAsText(file);
      }
    });
  }

  if (btnBatchPredict) {
    btnBatchPredict.addEventListener("click", async () => {
      const lines = batchTextarea.value
        .split("\n")
        .map(l => l.trim())
        .filter(l => l.length > 2);

      if (lines.length === 0) {
        alert("Ingresa al menos una reseña en el área de texto por lotes.");
        return;
      }

      btnBatchPredict.disabled = true;
      btnBatchPredict.textContent = "Procesando Lote...";

      try {
          const res = await fetch(`${API_BASE}/predict/logistic/batch`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ textos: lines }),
        });

        let data;
        if (res.ok) {
          data = await res.json();
        } else {
          data = lines.map(t => inferenciaLocalSimulada(t));
        }

        renderBatchTable(data);
      } catch (err) {
        const data = lines.map(t => inferenciaLocalSimulada(t));
        renderBatchTable(data);
      } finally {
        btnBatchPredict.disabled = false;
        btnBatchPredict.textContent = "🚀 Analizar Lote";
      }
    });
  }
}

function renderBatchTable(items) {
  const container = document.getElementById("batch-results-container");
  const tbody = document.getElementById("batch-table-body");
  const countBadge = document.getElementById("batch-count-badge");

  container.classList.remove("hidden");
  countBadge.textContent = `${items.length} reseñas procesadas`;
  tbody.innerHTML = "";

  items.forEach((item, idx) => {
    const tr = document.createElement("tr");
    const sent = item.sentimiento_predicho;
    const badgeClass = sent === "positivo" ? "badge-pos" : sent === "negativo" ? "badge-neg" : "badge-neu";

    tr.innerHTML = `
      <td><strong>${idx + 1}</strong></td>
      <td>${escapeHtml(item.texto)}</td>
      <td><span class="badge ${badgeClass}">${sent.toUpperCase()}</span></td>
      <td><strong>${(item.confianza * 100).toFixed(1)}%</strong></td>
      <td><code>+${((item.probabilidades?.positivo || 0) * 100).toFixed(0)}% / -${((item.probabilidades?.negativo || 0) * 100).toFixed(0)}%</code></td>
    `;
    tbody.appendChild(tr);
  });

  container.scrollIntoView({ behavior: "smooth" });
}

/* ==========================================================================
   6. Visualizador del Pipeline NLP Paso a Paso
   ========================================================================== */
function setupPipeline() {
  const btn = document.getElementById("btn-ejecutar-pipeline");
  const input = document.getElementById("pipeline-text-input");

  btn.addEventListener("click", () => {
    const texto = input.value.trim();
    if (!texto) {
      alert("Por favor ingresa un texto para ver la descomposición paso a paso.");
      input.focus();
      return;
    }
    ejecutarPipelineNLP(texto);
  });
}

async function ejecutarPipelineNLP(texto) {
  const btn = document.getElementById("btn-ejecutar-pipeline");
  btn.disabled = true;
  btn.textContent = "Procesando...";

  try {
    const res = await fetch(`${API_BASE}/preprocess`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texto }),
    });

    if (!res.ok) throw new Error("Error HTTP " + res.status);
    const data = await res.json();
    renderPipelineData(data);
  } catch (err) {
    console.warn("Fallo backend /preprocess, usando visualización local:", err);
    renderPipelineDataLocal(texto);
  } finally {
    btn.disabled = false;
    btn.textContent = "🔬 Inspeccionar NLP";
  }
}

function renderPipelineData(data) {
  document.getElementById("step-1-output").textContent = data.paso_1_normalizacion || "";
  renderTokenCloud("step-2-output", data.paso_2_tokenizacion || []);
  renderTokenCloud("step-3-output", data.paso_3_sin_stopwords || []);
  renderTokenCloud("step-4-output", data.paso_4_reduccion?.tokens || []);
  document.getElementById("step-5-output").textContent = data.paso_5_texto_final || "";

  const stats = data.estadisticas || {};
  document.getElementById("step-stats").innerHTML = `
    <span><strong>Tokens Iniciales:</strong> ${stats.total_tokens_iniciales || 0}</span>
    <span><strong>Tokens Filtrados:</strong> ${stats.total_tokens_filtrados || 0}</span>
    <span><strong>Reducción de Dimensionalidad:</strong> ${stats.reduccion_porcentual || 0}%</span>
  `;
}

function renderPipelineDataLocal(texto) {
  // Normalización simulada
  let norm = texto.toLowerCase()
    .replace(/[¡!]+/g, " [EXCLAMACION] ")
    .replace(/[¿?]+/g, " [INTERROGACION] ")
    .replace(/😋|😍|🍷/g, " delicioso ")
    .replace(/😡|🤬/g, " pésimo ");

  const tokens = norm.match(/[a-záéíóúüñ\[\]]+/gi) || [];
  const stopwords = new Set(["la", "el", "los", "las", "un", "una", "de", "en", "y", "a", "por", "con", "pero", "estuvo", "fue"]);
  const negaciones = new Set(["no", "ni", "nunca", "sin", "tampoco"]);

  const sinStopwords = tokens.filter(t => !stopwords.has(t));
  const lemas = sinStopwords.map(t => t.replace(/s$/, ""));

  renderPipelineData({
    paso_1_normalizacion: norm,
    paso_2_tokenizacion: tokens,
    paso_3_sin_stopwords: sinStopwords,
    paso_4_reduccion: { metodo: "lematizacion", tokens: lemas },
    paso_5_texto_final: lemas.join(" "),
    estadisticas: {
      total_tokens_iniciales: tokens.length,
      total_tokens_filtrados: sinStopwords.length,
      reduccion_porcentual: Math.round((1 - sinStopwords.length / (tokens.length || 1)) * 100),
    }
  });
}

function renderTokenCloud(elementId, tokens) {
  const container = document.getElementById(elementId);
  container.innerHTML = "";

  const negaciones = new Set(["no", "nunca", "ni", "sin", "tampoco"]);

  tokens.forEach(tok => {
    const chip = document.createElement("span");
    chip.className = "token-chip";
    chip.textContent = tok;

    if (negaciones.has(tok.toLowerCase())) {
      chip.classList.add("negation");
      chip.title = "Negación protegida en stopwords";
    } else if (tok.includes("[") && tok.includes("]")) {
      chip.classList.add("emoji-token");
    }

    container.appendChild(chip);
  });
}

/* ==========================================================================
   7. Dashboard de Métricas y Análisis de Errores
   ========================================================================== */
async function fetchRemoteMetrics() {
  try {
    const res = await fetch(`${API_BASE}/metrics`);
    if (res.ok) {
      const data = await res.json();
      renderMetrics(data.metricas_test);
    }
  } catch (e) {
    console.log("Usando métricas locales precargadas.");
  }
}

function renderMetrics(metricas) {
  if (!metricas) return;

  const acc = (metricas.accuracy * 100).toFixed(2);
  const f1 = (metricas.f1_weighted * 100).toFixed(2);
  const prec = (metricas.precision_weighted * 100).toFixed(2);
  const rec = (metricas.recall_weighted * 100).toFixed(2);

  document.getElementById("kpi-accuracy").textContent = `${acc}%`;
  document.getElementById("kpi-f1").textContent = `${f1}%`;
  document.getElementById("kpi-precision").textContent = `${prec}%`;
  document.getElementById("kpi-recall").textContent = `${rec}%`;

  if (metricas.matriz_confusion) {
    const cm = metricas.matriz_confusion;
    document.getElementById("cell-0-0").textContent = cm[0][0];
    document.getElementById("cell-0-1").textContent = cm[0][1];
    document.getElementById("cell-0-2").textContent = cm[0][2];

    document.getElementById("cell-1-0").textContent = cm[1][0];
    document.getElementById("cell-1-1").textContent = cm[1][1];
    document.getElementById("cell-1-2").textContent = cm[1][2];

    document.getElementById("cell-2-0").textContent = cm[2][0];
    document.getElementById("cell-2-1").textContent = cm[2][1];
    document.getElementById("cell-2-2").textContent = cm[2][2];
  }
}

async function fetchRemoteErrors() {
  try {
    const res = await fetch(`${API_BASE}/errors`);
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data) && data.length > 0) {
        renderErrorCases(data);
      }
    }
  } catch (e) {
    console.log("Usando casos de error precargados.");
  }
}

function renderErrorCases(casos) {
  const container = document.getElementById("error-cases-container");
  if (!container) return;

  container.innerHTML = "";
  casos.forEach(item => {
    const card = document.createElement("div");
    card.className = "error-card";

    const badgeReal = item.etiqueta_real === "positivo" ? "badge-pos" : item.etiqueta_real === "negativo" ? "badge-neg" : "badge-neu";
    const badgePred = item.etiqueta_predicha === "positivo" ? "badge-pos" : item.etiqueta_predicha === "negativo" ? "badge-neg" : "badge-neu";

    card.innerHTML = `
      <div>
        <div class="error-tags-row">
          <span>Real: <span class="badge ${badgeReal}">${item.etiqueta_real.toUpperCase()}</span></span>
          <span>Predicho: <span class="badge ${badgePred}">${item.etiqueta_predicha.toUpperCase()}</span></span>
        </div>
        <p class="error-quote">"${escapeHtml(item.texto_resena)}"</p>
      </div>
      <div class="error-explanation">
        <strong>💡 Causa del fallo:</strong> ${item.explicacion || "Discrepancia léxica entre modismos sutiles o sobrepeso de términos individuales."}
      </div>
    `;
    container.appendChild(card);
  });
}

function escapeHtml(text) {
  const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
  return String(text).replace(/[&<>"']/g, m => map[m]);
}

/* ==========================================================================
   Visual Effects: Canvas Partículas Reactivas, Cursor Spotlight y 3D Tilt
   ========================================================================== */
let canvasAnimationId = null;
let particleColorTheme = {
  primary: "rgba(99, 102, 241, ",
  secondary: "rgba(168, 85, 247, ",
  glowHex: "#6366f1"
};

function initVisualEffects() {
  initCursorSpotlight();
  initCardSpotlight();
  initInteractiveCanvas();
  initHologramReactor();
}

// 1. Cursor Spotlight Follower
function initCursorSpotlight() {
  const cursor = document.getElementById("cursor-glow");
  if (!cursor) return;

  let mouseX = window.innerWidth / 2;
  let mouseY = window.innerHeight / 2;
  let curX = mouseX;
  let curY = mouseY;

  window.addEventListener("mousemove", (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
  });

  function renderCursor() {
    curX += (mouseX - curX) * 0.12;
    curY += (mouseY - curY) * 0.12;

    cursor.style.left = `${curX}px`;
    cursor.style.top = `${curY}px`;

    requestAnimationFrame(renderCursor);
  }
  renderCursor();
}

// 2. Card Spotlight (Borde perimetral interactivo 2D, TOTALMENTE PLANO y SIN distorsión 3D)
function initCardSpotlight() {
  const cards = document.querySelectorAll(".glass-card, .kpi-card, .pipeline-step-card");

  cards.forEach(card => {
    card.addEventListener("mousemove", (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      card.style.setProperty("--mouse-x", `${x}px`);
      card.style.setProperty("--mouse-y", `${y}px`);
    });

    card.addEventListener("mouseleave", () => {
      card.style.setProperty("--mouse-x", "-500px");
      card.style.setProperty("--mouse-y", "-500px");
    });
  });
}

// 3. Canvas de Fondo con Partículas Conectadas y Física de Repulsión
function initInteractiveCanvas() {
  const canvas = document.getElementById("bg-canvas");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  let width = (canvas.width = window.innerWidth);
  let height = (canvas.height = window.innerHeight);

  window.addEventListener("resize", () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  });

  const mouse = { x: -1000, y: -1000, radius: 150 };

  window.addEventListener("mousemove", (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
  });

  window.addEventListener("mouseleave", () => {
    mouse.x = -1000;
    mouse.y = -1000;
  });

  // Crear partículas
  const numParticles = Math.min(Math.floor((width * height) / 22000), 55);
  const particles = [];

  class Particle {
    constructor() {
      this.x = Math.random() * width;
      this.y = Math.random() * height;
      this.size = Math.random() * 2.2 + 1;
      this.baseX = this.x;
      this.baseY = this.y;
      this.density = Math.random() * 20 + 2;
      this.vx = (Math.random() - 0.5) * 0.6;
      this.vy = (Math.random() - 0.5) * 0.6;
      this.alpha = Math.random() * 0.5 + 0.3;
    }

    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
      ctx.fillStyle = `${particleColorTheme.primary}${this.alpha})`;
      ctx.shadowBlur = 8;
      ctx.shadowColor = particleColorTheme.glowHex;
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    update() {
      // Movimiento autónomo suave
      this.x += this.vx;
      this.y += this.vy;

      if (this.x < 0 || this.x > width) this.vx *= -1;
      if (this.y < 0 || this.y > height) this.vy *= -1;

      // Interacción con el cursor (repulsión elástica)
      const dx = mouse.x - this.x;
      const dy = mouse.y - this.y;
      const distance = Math.sqrt(dx * dx + dy * dy);

      if (distance < mouse.radius) {
        const forceDirectionX = dx / distance;
        const forceDirectionY = dy / distance;
        const maxDistance = mouse.radius;
        const force = (maxDistance - distance) / maxDistance;
        const directionX = forceDirectionX * force * this.density * 0.6;
        const directionY = forceDirectionY * force * this.density * 0.6;

        this.x -= directionX;
        this.y -= directionY;
      }
    }
  }

  for (let i = 0; i < numParticles; i++) {
    particles.push(new Particle());
  }

  function connectParticles() {
    for (let a = 0; a < particles.length; a++) {
      for (let b = a + 1; b < particles.length; b++) {
        const dx = particles[a].x - particles[b].x;
        const dy = particles[a].y - particles[b].y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < 130) {
          const opacity = (1 - distance / 130) * 0.22;
          ctx.strokeStyle = `${particleColorTheme.primary}${opacity})`;
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          ctx.moveTo(particles[a].x, particles[a].y);
          ctx.lineTo(particles[b].x, particles[b].y);
          ctx.stroke();
        }
      }
    }
  }

  function animate() {
    ctx.clearRect(0, 0, width, height);

    for (let i = 0; i < particles.length; i++) {
      particles[i].draw();
      particles[i].update();
    }
    connectParticles();

    canvasAnimationId = requestAnimationFrame(animate);
  }

  animate();
}

// 4. Reacción Dinámica de Iluminación Ambiental según el Sentimiento
function updateAtmosphereTheme(sentiment) {
  const glowTop = document.querySelector(".background-glow.top-left");
  const glowBottom = document.querySelector(".background-glow.bottom-right");
  const cursor = document.getElementById("cursor-glow");

  if (sentiment === "positivo") {
    particleColorTheme = {
      primary: "rgba(16, 185, 129, ",
      secondary: "rgba(52, 211, 153, ",
      glowHex: "#10b981"
    };
    if (glowTop) glowTop.style.background = "radial-gradient(circle, #059669 0%, rgba(16, 185, 129, 0.4) 40%, transparent 70%)";
    if (glowBottom) glowBottom.style.background = "radial-gradient(circle, #10b981 0%, rgba(52, 211, 153, 0.3) 40%, transparent 70%)";
    if (cursor) cursor.style.background = "radial-gradient(circle, rgba(16, 185, 129, 0.22) 0%, rgba(52, 211, 153, 0.08) 40%, transparent 70%)";
  } else if (sentiment === "negativo") {
    particleColorTheme = {
      primary: "rgba(239, 68, 68, ",
      secondary: "rgba(248, 113, 113, ",
      glowHex: "#ef4444"
    };
    if (glowTop) glowTop.style.background = "radial-gradient(circle, #dc2626 0%, rgba(239, 68, 68, 0.4) 40%, transparent 70%)";
    if (glowBottom) glowBottom.style.background = "radial-gradient(circle, #ef4444 0%, rgba(244, 63, 94, 0.3) 40%, transparent 70%)";
    if (cursor) cursor.style.background = "radial-gradient(circle, rgba(239, 68, 68, 0.22) 0%, rgba(248, 113, 113, 0.08) 40%, transparent 70%)";
  } else {
    particleColorTheme = {
      primary: "rgba(245, 158, 11, ",
      secondary: "rgba(251, 191, 36, ",
      glowHex: "#f59e0b"
    };
    if (glowTop) glowTop.style.background = "radial-gradient(circle, #d97706 0%, rgba(245, 158, 11, 0.4) 40%, transparent 70%)";
    if (glowBottom) glowBottom.style.background = "radial-gradient(circle, #f59e0b 0%, rgba(251, 191, 36, 0.3) 40%, transparent 70%)";
    if (cursor) cursor.style.background = "radial-gradient(circle, rgba(245, 158, 11, 0.22) 0%, rgba(251, 191, 36, 0.08) 40%, transparent 70%)";
  }
}

function resetAtmosphereTheme() {
  particleColorTheme = {
    primary: "rgba(99, 102, 241, ",
    secondary: "rgba(168, 85, 247, ",
    glowHex: "#6366f1"
  };
  const glowTop = document.querySelector(".background-glow.top-left");
  const glowBottom = document.querySelector(".background-glow.bottom-right");
  const cursor = document.getElementById("cursor-glow");

  if (glowTop) glowTop.style.background = "radial-gradient(circle, #6366f1 0%, rgba(139, 92, 246, 0.6) 40%, transparent 70%)";
  if (glowBottom) glowBottom.style.background = "radial-gradient(circle, #ec4899 0%, rgba(244, 63, 94, 0.5) 40%, transparent 70%)";
  if (cursor) cursor.style.background = "radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.08) 35%, transparent 70%)";
}

// 5. Reactor Holográfico Cuántico: Emblema Gastronómico con Desintegración y Reconstrucción en Barra
function initHologramReactor() {
  const canvas = document.getElementById("reactor-hologram-canvas");
  const wrapper = document.getElementById("reactor-canvas-wrapper");
  const btnTrigger = document.getElementById("btn-trigger-disintegrate");
  const btnText = document.getElementById("reactor-btn-text");
  const statusText = document.getElementById("reactor-status-text");
  const statusDot = document.querySelector(".reactor-status-dot");
  const brandInteractive = document.getElementById("brand-interactive");
  const brandIconBase = document.getElementById("brand-icon-base");

  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 2;
  const logicalW = 300;
  const logicalH = 64;

  canvas.width = logicalW * dpr;
  canvas.height = logicalH * dpr;
  ctx.scale(dpr, dpr);

  const cx = 56;  // Centro del emblema gastronómico
  const cy = 32;

  // Estado del Reactor: "assembled" | "charging" | "disintegrating" | "vortex" | "magnetic" | "shockwave"
  let reactorState = "assembled";
  let stateStartTime = performance.now();
  let shockwaveR = 0;
  let shockwaveAlpha = 0;
  let flashAlpha = 0;
  let shakeX = 0;
  let shakeY = 0;

  // Paleta de partículas cuánticas (ámbar culinario, neón cian, púrpura IA, magenta, esmeralda, destello blanco)
  const colors = ["#f59e0b", "#fbbf24", "#38bdf8", "#818cf8", "#ec4899", "#10b981", "#ffffff"];

  // Generador de mapa de anclaje (Target Anchors) para el emblema y visualizador
  const particles = [];

  class ReactorParticle {
    constructor(originX, originY, color, size, layer = "emblem") {
      this.originX = originX;
      this.originY = originY;
      this.x = originX;
      this.y = originY;
      this.color = color || colors[Math.floor(Math.random() * colors.length)];
      this.size = size || (Math.random() * 1.6 + 1.2);
      this.vx = 0;
      this.vy = 0;
      this.layer = layer;
      this.alpha = 1;
      this.trail = [];
    }

    explode(centerX, centerY) {
      const angle = Math.atan2(this.y - centerY, this.x - centerX) + (Math.random() - 0.5) * 1.1;
      const speed = Math.random() * 4.8 + 2.2;
      this.vx = Math.cos(angle) * speed;
      this.vy = Math.sin(angle) * speed;
      this.trail = [];
    }

    updatePhysics(phase) {
      if (phase === "disintegrating" || phase === "vortex") {
        this.x += this.vx;
        this.y += this.vy;
        this.vx *= 0.94;
        this.vy *= 0.94;

        // Fuerza de remolino orbital en vórtice
        if (phase === "vortex") {
          const dx = this.x - cx;
          const dy = this.y - cy;
          this.vx += -dy * 0.045;
          this.vy += dx * 0.045;
        }

        // Rebote elástico suave en bordes del canvas
        if (this.x < 4 || this.x > logicalW - 4) this.vx *= -0.7;
        if (this.y < 4 || this.y > logicalH - 4) this.vy *= -0.7;

      } else if (phase === "magnetic") {
        // Atracción gravitacional / resorte hacia su anclaje exacto
        const dx = this.originX - this.x;
        const dy = this.originY - this.y;
        this.vx += dx * 0.18;
        this.vy += dy * 0.18;
        this.vx *= 0.81;
        this.vy *= 0.81;
        this.x += this.vx;
        this.y += this.vy;
      } else {
        // Assembled / Idle: pequeña oscilación cuántica sutil
        this.x = this.originX;
        this.y = this.originY;
      }
    }

    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
      ctx.fillStyle = this.color;
      ctx.shadowBlur = 5;
      ctx.shadowColor = this.color;
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  }

  // Tokens flotantes de NLP ("delicioso", "★5", "TF-IDF", etc.)
  let nlpTokens = [];

  class NlpFloatToken {
    constructor(text, color, vx, vy) {
      this.text = text;
      this.color = color;
      this.x = cx;
      this.y = cy;
      this.vx = vx;
      this.vy = vy;
      this.alpha = 1;
      this.scale = 1;
    }

    update(phase) {
      if (phase === "disintegrating" || phase === "vortex") {
        this.x += this.vx;
        this.y += this.vy;
        this.vx *= 0.95;
        this.vy *= 0.95;
        this.scale = Math.min(1.2, this.scale + 0.01);
      } else if (phase === "magnetic") {
        const dx = cx - this.x;
        const dy = cy - this.y;
        this.x += dx * 0.15;
        this.y += dy * 0.15;
        this.scale *= 0.92;
        this.alpha *= 0.90;
      }
    }

    draw() {
      if (this.alpha <= 0.05) return;
      ctx.save();
      ctx.font = `bold ${Math.round(8.5 * this.scale)}px 'JetBrains Mono', monospace`;
      ctx.fillStyle = this.color;
      ctx.globalAlpha = this.alpha;
      ctx.shadowBlur = 8;
      ctx.shadowColor = this.color;
      ctx.fillText(this.text, this.x, this.y);
      ctx.restore();
    }
  }

  // Inicializar puntos de anclaje geométrico del emblema y visualizador
  function generateParticleAnchors() {
    particles.length = 0;

    // 1. Cúpula Cloche Gourmet (Arco semicircular)
    const domeRadius = 18;
    for (let i = 0; i <= 24; i++) {
      const angle = Math.PI + (i / 24) * Math.PI;
      const px = cx + Math.cos(angle) * domeRadius;
      const py = cy + 10 + Math.sin(angle) * (domeRadius * 0.92);
      particles.push(new ReactorParticle(px, py, "#f59e0b", 1.8, "dome"));
    }

    // 2. Plato base (Elipse horizontal)
    for (let i = 0; i <= 20; i++) {
      const angle = (i / 20) * Math.PI * 2;
      const px = cx + Math.cos(angle) * 28;
      const py = cy + 11 + Math.sin(angle) * 3.5;
      particles.push(new ReactorParticle(px, py, "#e2e8f0", 1.6, "plate"));
    }

    // 3. Asa del Cloche (Pequeña perla dorada en la cima)
    for (let i = 0; i < 6; i++) {
      const angle = (i / 6) * Math.PI * 2;
      const px = cx + Math.cos(angle) * 2.8;
      const py = cy - 8 + Math.sin(angle) * 2.8;
      particles.push(new ReactorParticle(px, py, "#fde68a", 1.9, "handle"));
    }

    // 4. Cubiertos holográficos (Tenedor a la izquierda, Cuchillo a la derecha)
    // Tenedor (cx - 36)
    for (let y = cy - 6; y <= cy + 16; y += 3) {
      particles.push(new ReactorParticle(cx - 35, y, "#38bdf8", 1.4, "cutlery"));
    }
    particles.push(new ReactorParticle(cx - 38, cy - 9, "#38bdf8", 1.4, "cutlery"));
    particles.push(new ReactorParticle(cx - 35, cy - 10, "#38bdf8", 1.4, "cutlery"));
    particles.push(new ReactorParticle(cx - 32, cy - 9, "#38bdf8", 1.4, "cutlery"));

    // Cuchillo (cx + 36)
    for (let y = cy - 8; y <= cy + 16; y += 3) {
      particles.push(new ReactorParticle(cx + 35, y, "#fbbf24", 1.4, "cutlery"));
    }
    particles.push(new ReactorParticle(cx + 37, cy - 5, "#fbbf24", 1.3, "cutlery"));

    // 5. Estrellas de Reseña (5 micro-estrellas arqueadas sobre el plato)
    const starOffsets = [-22, -11, 0, 11, 22];
    starOffsets.forEach(ox => {
      const oy = cy - 18 - (1 - Math.abs(ox) / 28) * 4;
      particles.push(new ReactorParticle(cx + ox, oy, "#facc15", 2.2, "star"));
    });

    // 6. Nodos de Red Neuronal / Synapses NLP
    const neuralNodes = [
      { x: cx - 22, y: cy - 2 },
      { x: cx + 22, y: cy - 2 },
      { x: cx - 14, y: cy - 16 },
      { x: cx + 14, y: cy - 16 },
      { x: cx, y: cy + 3 }
    ];
    neuralNodes.forEach(n => {
      particles.push(new ReactorParticle(n.x, n.y, "#818cf8", 2.0, "neural"));
    });

    // 7. Vapor Aromático (Partículas ondulantes)
    for (let i = 0; i < 9; i++) {
      const ox = (i % 3 - 1) * 8;
      const oy = cy - 12 - (Math.floor(i / 3) * 6);
      particles.push(new ReactorParticle(cx + ox, oy, "#fde68a", 1.2, "steam"));
    }

    // 8. Visualizador de Espectro NLP en el lado derecho (x: 114 a 286)
    const rightStartX = 118;
    for (let bar = 0; bar < 12; bar++) {
      const bx = rightStartX + bar * 14;
      const barColor = bar < 7 ? "#10b981" : (bar < 10 ? "#f59e0b" : "#ef4444");
      const hMax = (bar % 3 === 0 ? 18 : (bar % 2 === 0 ? 14 : 22));
      for (let dotY = cy + 14; dotY >= cy + 14 - hMax; dotY -= 5) {
        particles.push(new ReactorParticle(bx, dotY, barColor, 1.4, "spectrum"));
      }
    }
  }

  generateParticleAnchors();

  // Función de disparo del ciclo de desintegración & reconstrucción
  function triggerDisintegration() {
    if (reactorState !== "assembled") return;

    reactorState = "charging";
    stateStartTime = performance.now();

    if (btnText) btnText.textContent = "Sobrecarga...";
    if (statusText) statusText.textContent = "⚡ Sobrecarga Cuántica...";
    if (statusDot) statusDot.style.background = "#ec4899";

    // Efecto sutil en el icono de marca adyacente
    if (brandIconBase) {
      brandIconBase.style.transform = "scale(0.88)";
      brandIconBase.style.opacity = "0.4";
    }

    // Fase 1: Carga/Glitch por 200ms -> Detonación
    setTimeout(() => {
      reactorState = "disintegrating";
      stateStartTime = performance.now();

      if (btnText) btnText.textContent = "Desintegrando...";
      if (statusText) statusText.textContent = "💥 Desintegración: 200 Partículas NLP";
      if (statusDot) statusDot.style.background = "#ef4444";

      // Dispersión explosiva de las partículas
      particles.forEach(p => p.explode(cx, cy));

      // Crear tokens flotantes de NLP
      nlpTokens = [
        new NlpFloatToken("delicioso", "#10b981", -1.4, -0.9),
        new NlpFloatToken("★ 5.0", "#f59e0b", 1.5, -1.2),
        new NlpFloatToken("TF-IDF", "#ec4899", 1.2, 0.8),
        new NlpFloatToken("NLP", "#38bdf8", -0.9, 1.1),
        new NlpFloatToken("LogReg", "#818cf8", -1.5, 0.3),
        new NlpFloatToken("sabor", "#fbbf24", 0.6, -1.4)
      ];

      // Fase 2: Vórtice Cuántico a los 550ms
      setTimeout(() => {
        if (reactorState === "disintegrating") {
          reactorState = "vortex";
          if (btnText) btnText.textContent = "Vórtice...";
          if (statusText) statusText.textContent = "🌀 Vórtice de Datos en Órbita...";
        }
      }, 500);

      // Fase 3: Atracción Magnética / Reensamblaje a los 1150ms
      setTimeout(() => {
        reactorState = "magnetic";
        if (btnText) btnText.textContent = "Reensamblando...";
        if (statusText) statusText.textContent = "🧲 Reconstrucción Magnética...";
        if (statusDot) statusDot.style.background = "#38bdf8";
      }, 1150);

      // Fase 4: Impacto & Onda Expansiva (Shockwave) a los 1750ms
      setTimeout(() => {
        reactorState = "shockwave";
        shockwaveR = 6;
        shockwaveAlpha = 1;
        flashAlpha = 0.75;

        if (btnText) btnText.textContent = "¡Completado!";
        if (statusText) statusText.textContent = "✨ Emblema Reconstruido al 100%";
        if (statusDot) statusDot.style.background = "#10b981";

        if (brandIconBase) {
          brandIconBase.style.transform = "scale(1)";
          brandIconBase.style.opacity = "1";
        }

        // Fin del ciclo y regreso a Assembled
        setTimeout(() => {
          reactorState = "assembled";
          if (btnText) btnText.textContent = "Desintegrar";
          setTimeout(() => {
            if (statusText && reactorState === "assembled") {
              statusText.textContent = "Núcleo NLP Activo • 800 Tokens";
              if (statusDot) statusDot.style.background = "#f59e0b";
            }
          }, 2400);
        }, 600);

      }, 1750);

    }, 220);
  }

  // Renderizado del Emblema Assembled (Dibujo vectorial fino con gradientes)
  function drawAssembledGraphics(time) {
    ctx.save();

    // 1. Plato Base Gourmet
    ctx.beginPath();
    ctx.ellipse(cx, cy + 11, 28, 4, 0, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(226, 232, 240, 0.85)";
    ctx.shadowBlur = 10;
    ctx.shadowColor = "rgba(245, 158, 11, 0.4)";
    ctx.fill();
    ctx.shadowBlur = 0;

    // 2. Cúpula Cloche
    const gradDome = ctx.createLinearGradient(cx - 20, cy - 8, cx + 20, cy + 10);
    gradDome.addColorStop(0, "rgba(251, 191, 36, 0.85)");
    gradDome.addColorStop(0.5, "rgba(245, 158, 11, 0.95)");
    gradDome.addColorStop(1, "rgba(217, 119, 6, 0.8)");

    ctx.beginPath();
    ctx.arc(cx, cy + 10, 19, Math.PI, 0);
    ctx.fillStyle = gradDome;
    ctx.shadowBlur = 12;
    ctx.shadowColor = "#f59e0b";
    ctx.fill();
    ctx.shadowBlur = 0;

    // Borde brillante de la cúpula
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "rgba(254, 243, 199, 0.9)";
    ctx.stroke();

    // Asa del Cloche
    ctx.beginPath();
    ctx.arc(cx, cy - 9, 3.2, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.shadowBlur = 8;
    ctx.shadowColor = "#fff";
    ctx.fill();
    ctx.shadowBlur = 0;

    // 3. Ondas de Vapor Aromático oscilando con el tiempo
    for (let i = -1; i <= 1; i++) {
      const ox = cx + i * 8;
      const wave = Math.sin(time * 0.003 + i * 1.5) * 2.5;
      ctx.beginPath();
      ctx.moveTo(ox, cy - 11);
      ctx.quadraticCurveTo(ox + wave, cy - 18, ox - wave, cy - 24);
      ctx.strokeStyle = `rgba(253, 230, 138, ${0.45 + Math.sin(time * 0.004 + i) * 0.25})`;
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }

    // 4. Cubiertos Tenores y Cuchillo
    // Tenedor
    ctx.beginPath();
    ctx.moveTo(cx - 35, cy - 8);
    ctx.lineTo(cx - 35, cy + 15);
    ctx.moveTo(cx - 38, cy - 8);
    ctx.lineTo(cx - 38, cy - 2);
    ctx.moveTo(cx - 32, cy - 8);
    ctx.lineTo(cx - 32, cy - 2);
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 1.4;
    ctx.shadowBlur = 6;
    ctx.shadowColor = "#38bdf8";
    ctx.stroke();

    // Cuchillo
    ctx.beginPath();
    ctx.moveTo(cx + 35, cy - 8);
    ctx.lineTo(cx + 35, cy + 15);
    ctx.moveTo(cx + 37, cy - 4);
    ctx.lineTo(cx + 35, cy - 8);
    ctx.strokeStyle = "#fbbf24";
    ctx.lineWidth = 1.4;
    ctx.shadowBlur = 6;
    ctx.shadowColor = "#fbbf24";
    ctx.stroke();
    ctx.shadowBlur = 0;

    // 5. Estrellas de Reseña (5★ con resplandor)
    const starOffsets = [-22, -11, 0, 11, 22];
    starOffsets.forEach((ox, idx) => {
      const oy = cy - 18 - (1 - Math.abs(ox) / 28) * 4;
      const starGlow = 0.7 + Math.sin(time * 0.004 + idx * 0.8) * 0.3;
      ctx.font = "8px sans-serif";
      ctx.fillStyle = `rgba(250, 204, 21, ${starGlow})`;
      ctx.shadowBlur = 8;
      ctx.shadowColor = "#facc15";
      ctx.fillText("★", cx + ox - 4, oy + 3);
    });

    // 6. Nodos y Conexiones Neuronales
    const nodes = [
      { x: cx - 22, y: cy - 2 },
      { x: cx + 22, y: cy - 2 },
      { x: cx - 14, y: cy - 16 },
      { x: cx + 14, y: cy - 16 }
    ];

    ctx.beginPath();
    ctx.moveTo(nodes[0].x, nodes[0].y);
    ctx.lineTo(nodes[2].x, nodes[2].y);
    ctx.lineTo(nodes[3].x, nodes[3].y);
    ctx.lineTo(nodes[1].x, nodes[1].y);
    ctx.strokeStyle = "rgba(129, 140, 248, 0.4)";
    ctx.lineWidth = 0.9;
    ctx.stroke();

    nodes.forEach((n, idx) => {
      ctx.beginPath();
      ctx.arc(n.x, n.y, 2, 0, Math.PI * 2);
      ctx.fillStyle = idx % 2 === 0 ? "#818cf8" : "#ec4899";
      ctx.shadowBlur = 8;
      ctx.shadowColor = ctx.fillStyle;
      ctx.fill();
    });

    // 7. Textos y Métricas en el lado derecho
    ctx.shadowBlur = 0;
    ctx.font = "bold 9px 'Plus Jakarta Sans', sans-serif";
    ctx.fillStyle = "#ffffff";
    ctx.letterSpacing = "0.5px";
    ctx.fillText("SABOR·IA NLP CORE", 120, cy - 8);

    ctx.font = "500 7.5px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#94a3b8";
    ctx.fillText("TF-IDF 800 N-GRAMS • REG. LOGÍSTICA", 120, cy + 3);

    // 8. Barras de Espectro Dinámico de Sentimiento
    const rightStartX = 120;
    const barWidth = 7;
    const barGap = 6;
    for (let bar = 0; bar < 12; bar++) {
      const bx = rightStartX + bar * (barWidth + barGap);
      const waveVal = Math.sin(time * 0.005 + bar * 0.45) * 0.5 + 0.5;
      const barH = 4 + waveVal * 14;
      const barColor = bar < 7 ? "#10b981" : (bar < 10 ? "#f59e0b" : "#ef4444");

      ctx.fillStyle = barColor;
      ctx.shadowBlur = 4;
      ctx.shadowColor = barColor;
      ctx.beginPath();
      ctx.roundRect(bx, cy + 22 - barH, barWidth, barH, 2);
      ctx.fill();
    }

    ctx.restore();
  }

  // Bucle de animación continuo (60 FPS)
  function animateFrame(now) {
    ctx.clearRect(0, 0, logicalW, logicalH);

    // Glitch / vibración en estado Charging
    if (reactorState === "charging") {
      shakeX = (Math.random() - 0.5) * 4;
      shakeY = (Math.random() - 0.5) * 4;
    } else {
      shakeX = 0;
      shakeY = 0;
    }

    ctx.save();
    if (shakeX !== 0 || shakeY !== 0) {
      ctx.translate(shakeX, shakeY);
    }

    if (reactorState === "assembled") {
      // Estado normal: dibujo vectorial fino + ligeras partículas cuánticas de halo
      drawAssembledGraphics(now);

    } else if (reactorState === "charging") {
      // Estado de carga: dibujo con aberración cromática
      ctx.globalAlpha = 0.5;
      ctx.fillStyle = "#38bdf8";
      ctx.fillRect(cx - 30, cy - 20, 60, 40);
      drawAssembledGraphics(now);
      ctx.globalAlpha = 1.0;

    } else {
      // Estados de Desintegración, Vórtice y Reconstrucción Cuántica
      for (let p of particles) {
        p.updatePhysics(reactorState);
        p.draw();
      }

      // Dibujar tokens NLP flotantes
      for (let token of nlpTokens) {
        token.update(reactorState);
        token.draw();
      }

      // Onda expansiva luminosa en Shockwave
      if (reactorState === "shockwave" && shockwaveAlpha > 0.02) {
        ctx.beginPath();
        ctx.arc(cx, cy, shockwaveR, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(245, 158, 11, ${shockwaveAlpha})`;
        ctx.lineWidth = 2.8;
        ctx.shadowBlur = 16;
        ctx.shadowColor = "#f59e0b";
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(cx, cy, shockwaveR * 0.65, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(56, 189, 248, ${shockwaveAlpha * 0.8})`;
        ctx.lineWidth = 1.6;
        ctx.stroke();

        shockwaveR += 3.8;
        shockwaveAlpha -= 0.05;

        // Reincorporación progresiva de gráficos montados
        ctx.globalAlpha = Math.min(1, 1 - shockwaveAlpha * 0.8);
        drawAssembledGraphics(now);
        ctx.globalAlpha = 1.0;
      }
    }

    // Flash suave de luz
    if (flashAlpha > 0.02) {
      ctx.fillStyle = `rgba(255, 255, 255, ${flashAlpha})`;
      ctx.fillRect(0, 0, logicalW, logicalH);
      flashAlpha -= 0.04;
    }

    ctx.restore();
    requestAnimationFrame(animateFrame);
  }

  requestAnimationFrame(animateFrame);

  // Escuchadores de eventos para interactividad
  if (btnTrigger) {
    btnTrigger.addEventListener("click", triggerDisintegration);
  }

  if (wrapper) {
    wrapper.addEventListener("click", triggerDisintegration);
  }

  if (brandInteractive) {
    brandInteractive.addEventListener("click", triggerDisintegration);
  }

  // Activación automática cíclica cada 12 segundos
  setInterval(() => {
    if (reactorState === "assembled" && document.visibilityState === "visible") {
      triggerDisintegration();
    }
  }, 12000);
}


