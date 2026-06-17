const fileInput = document.querySelector("#fileInput");
const dropZone = document.querySelector("#dropZone");
const dropTitle = document.querySelector("#dropTitle");
const fileMeta = document.querySelector("#fileMeta");
const previewFrame = document.querySelector("#previewFrame");
const previewImage = document.querySelector("#previewImage");
const analyzeButton = document.querySelector("#analyzeButton");
const gradcamButton = document.querySelector("#gradcamButton");
const clearButton = document.querySelector("#clearButton");
const message = document.querySelector("#message");
const apiStatus = document.querySelector("#apiStatus");
const apiStatusText = document.querySelector("#apiStatusText");
const emptyState = document.querySelector("#emptyState");
const resultContent = document.querySelector("#resultContent");
const resultLabel = document.querySelector("#resultLabel");
const confidenceValue = document.querySelector("#confidenceValue");
const confidenceGauge = document.querySelector(".confidence-gauge");
const severityStrip = document.querySelector("#severityStrip");
const probabilityList = document.querySelector("#probabilityList");
const gradcamSection = document.querySelector("#gradcamSection");
const gradcamImage = document.querySelector("#gradcamImage");
const gradcamMeta = document.querySelector("#gradcamMeta");

const API_BASE = window.location.protocol === "file:"
  ? "http://127.0.0.1:3000/api"
  : "/api";

const labels = {
  "No DR": "No DR",
  Mild: "Mild",
  Moderate: "Moderate",
  Severe: "Severe",
  "Proliferative DR": "Proliferative DR"
};

const severityIndex = {
  "No DR": 0,
  Mild: 1,
  Moderate: 2,
  Severe: 3,
  "Proliferative DR": 4
};

let selectedFile = null;
let previewUrl = null;
let gradcamUrl = null;

function setMessage(text, type = "") {
  message.textContent = text;
  message.className = `message ${type ? `is-${type}` : ""}`;
}

function formatPercent(value) {
  return `${Math.round(value * 1000) / 10}%`;
}

function setApiStatus(state, text) {
  apiStatus.dataset.state = state;
  apiStatusText.textContent = text;
}

async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`, { cache: "no-store" });
    if (!response.ok) throw new Error("Health check failed");
    setApiStatus("online", "API Online");
  } catch {
    setApiStatus("offline", "API Offline");
  }
}

function resetResult() {
  emptyState.hidden = false;
  resultContent.hidden = true;
  resultLabel.textContent = "-";
  confidenceValue.textContent = "0%";
  confidenceGauge.style.setProperty("--confidence", "0%");
  severityStrip.replaceChildren();
  probabilityList.replaceChildren();
  resetGradcam();
}

function resetGradcam() {
  gradcamSection.hidden = true;
  gradcamImage.removeAttribute("src");
  gradcamMeta.textContent = "";

  if (gradcamUrl) {
    URL.revokeObjectURL(gradcamUrl);
    gradcamUrl = null;
  }
}

function setSelectedFile(file) {
  if (!file) return;

  if (!file.type.startsWith("image/")) {
    setMessage("Please choose a valid image file.", "error");
    return;
  }

  selectedFile = file;
  analyzeButton.disabled = false;
  gradcamButton.disabled = false;
  clearButton.disabled = false;
  dropTitle.textContent = file.name;
  fileMeta.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewFrame.hidden = false;
  setMessage("Image ready for analysis.");
  resetResult();
}

function clearSelection() {
  selectedFile = null;
  fileInput.value = "";
  analyzeButton.disabled = true;
  gradcamButton.disabled = true;
  clearButton.disabled = true;
  dropTitle.textContent = "Choose an image or drag it here";
  fileMeta.textContent = "PNG, JPG, WEBP";
  previewFrame.hidden = true;
  previewImage.removeAttribute("src");

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  setMessage("");
  resetResult();
}

function renderResult(result) {
  const confidence = Number(result.confidence || 0);
  const classLabel = result.label || "Unknown";
  const className = labels[classLabel] || classLabel;
  const classIndex = severityIndex[classLabel] ?? result.predicted_class ?? 0;

  emptyState.hidden = true;
  resultContent.hidden = false;
  resultLabel.textContent = className;
  confidenceValue.textContent = formatPercent(confidence);
  confidenceGauge.style.setProperty("--confidence", `${Math.max(0, Math.min(100, confidence * 100))}%`);

  severityStrip.replaceChildren();
  const chip = document.createElement("span");
  chip.className = "severity-chip";
  chip.dataset.severity = classIndex;
  chip.textContent = `Class ${result.predicted_class}: ${classLabel}`;
  severityStrip.appendChild(chip);

  probabilityList.replaceChildren();
  Object.entries(result.probabilities || {}).forEach(([label, value]) => {
    const row = document.createElement("div");
    const severity = severityIndex[label] ?? 0;
    row.className = "probability-row";
    row.dataset.severity = severity;

    const head = document.createElement("div");
    head.className = "probability-head";

    const labelSpan = document.createElement("span");
    labelSpan.textContent = labels[label] || label;

    const valueSpan = document.createElement("span");
    valueSpan.textContent = formatPercent(Number(value));

    const track = document.createElement("div");
    track.className = "probability-track";

    const fill = document.createElement("div");
    fill.className = "probability-fill";
    fill.style.width = `${Math.max(0, Math.min(100, Number(value) * 100))}%`;

    head.append(labelSpan, valueSpan);
    track.appendChild(fill);
    row.append(head, track);
    probabilityList.appendChild(row);
  });
}

async function generateGradcam() {
  if (!selectedFile) return;

  analyzeButton.disabled = true;
  gradcamButton.disabled = true;
  clearButton.disabled = true;
  setMessage("Generating Grad-CAM overlay...");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch(`${API_BASE}/gradcam`, {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || "Could not generate Grad-CAM.");
    }

    const blob = await response.blob();
    if (gradcamUrl) URL.revokeObjectURL(gradcamUrl);
    gradcamUrl = URL.createObjectURL(blob);

    const predictedClass = response.headers.get("X-Predicted-Class") || "-";
    const predictedLabel = response.headers.get("X-Predicted-Label") || "Unknown";
    const confidence = Number(response.headers.get("X-Confidence") || 0);

    emptyState.hidden = true;
    resultContent.hidden = false;
    gradcamImage.src = gradcamUrl;
    gradcamSection.hidden = false;
    gradcamMeta.textContent = `Explained class ${predictedClass}: ${predictedLabel} (${formatPercent(confidence)} confidence)`;
    setMessage("Grad-CAM overlay generated.", "success");
  } catch (error) {
    setMessage(error.message, "error");
  } finally {
    analyzeButton.disabled = false;
    gradcamButton.disabled = false;
    clearButton.disabled = false;
    checkHealth();
  }
}

async function analyzeImage() {
  if (!selectedFile) return;

  analyzeButton.disabled = true;
  gradcamButton.disabled = true;
  clearButton.disabled = true;
  setMessage("Analyzing image...");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: formData
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Could not analyze the image.");
    }

    renderResult(payload);
    setMessage("Analysis complete.", "success");
  } catch (error) {
    setMessage(error.message, "error");
  } finally {
    analyzeButton.disabled = false;
    gradcamButton.disabled = false;
    clearButton.disabled = false;
    checkHealth();
  }
}

fileInput.addEventListener("change", () => setSelectedFile(fileInput.files[0]));
clearButton.addEventListener("click", clearSelection);
analyzeButton.addEventListener("click", analyzeImage);
gradcamButton.addEventListener("click", generateGradcam);

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
});

dropZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  setSelectedFile(file);
});

checkHealth();
setInterval(checkHealth, 15000);
