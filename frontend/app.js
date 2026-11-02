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
    const res = await fetch(`${API_BASE}/predict`, {
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
        const res = await fetch(`${API_BASE}/predict/batch`, {
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
