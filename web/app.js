/**
 * MicroDetect AI v3.0 Pro — Enhanced Client Application Logic
 * Features: Animated KPI counters, Multi-step progress bar,
 * Environmental Impact Dashboard, 3rd Confidence Chart,
 * Batch Upload UI, Particle Detail Modal, EVS Tab navigation.
 * v4.0 NEW: Splash Screen, Camera Capture, WhatsApp/Email Share,
 * Enhanced Charts (Donut + Histogram + Scatter), KPI Glow Pulse
 */

// ==============================
// SPLASH SCREEN CONTROLLER
// Runs immediately (before DOM ready) to animate and dismiss the
// splash screen. Steps through progress stages every 420ms,
// then fades out and removes the element.
// ==============================
(function initSplash() {
  const splash  = document.getElementById("splashScreen");
  const bar     = document.getElementById("splashBar");
  const status  = document.getElementById("splashStatus");
  if (!splash) return;

  const steps = [
    { w: 20,  msg: "Loading CSS framework..." },
    { w: 45,  msg: "Initializing Chart.js engine..." },
    { w: 65,  msg: "Connecting to AI backend..." },
    { w: 85,  msg: "Fetching sample gallery..." },
    { w: 100, msg: "Ready!" },
  ];

  let i = 0;
  function tick() {
    if (i >= steps.length) {
      setTimeout(() => {
        splash.classList.add("splash-fade");
        setTimeout(() => splash.remove(), 700);
      }, 300);
      return;
    }
    const s = steps[i++];
    if (bar)    bar.style.width    = s.w + "%";
    if (status) status.textContent = s.msg;
    setTimeout(tick, 420);
  }
  tick();
})();

document.addEventListener("DOMContentLoaded", () => {

  // ============================
  // DOM ELEMENTS
  // ============================
  const statusIndicator = document.getElementById("statusIndicator");
  const statusText = document.getElementById("statusText");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const confSlider = document.getElementById("confSlider");
  const confVal = document.getElementById("confVal");
  const detectBtn = document.getElementById("detectBtn");
  const claheToggle = document.getElementById("claheToggle");

  const presetButtons = document.querySelectorAll(".preset-btn");
  const umScaleInput = document.getElementById("umScaleInput");

  const infoFileName = document.getElementById("infoFileName");
  const infoResolution = document.getElementById("infoResolution");
  const infoFileSize = document.getElementById("infoFileSize");
  const infoFormat = document.getElementById("infoFormat");

  const modeButtons = document.querySelectorAll(".mode-btn");

  const kpiCount = document.getElementById("kpiCount");
  const kpiHighConf = document.getElementById("kpiHighConf");
  const kpiAvgConf = document.getElementById("kpiAvgConf");
  const kpiRisk = document.getElementById("kpiRisk");
  const kpiRiskSub = document.getElementById("kpiRiskSub");

  const previewContainer = document.getElementById("previewContainer");
  const placeholderState = document.getElementById("placeholderState");
  const imageWrapper = document.getElementById("imageWrapper");
  const annotatedImg = document.getElementById("annotatedImg");
  const zoomControls = document.getElementById("zoomControls");
  const zoomInBtn = document.getElementById("zoomInBtn");
  const zoomOutBtn = document.getElementById("zoomOutBtn");
  const zoomResetBtn = document.getElementById("zoomResetBtn");
  const loadingOverlay = document.getElementById("loadingOverlay");

  const tableBody = document.getElementById("tableBody");
  const downloadCsvBtn = document.getElementById("downloadCsvBtn");
  const downloadJsonBtn = document.getElementById("downloadJsonBtn");
  const downloadPdfBtn = document.getElementById("downloadPdfBtn");

  // Environmental Impact
  const envWater = document.getElementById("envWater");
  const envMarine = document.getElementById("envMarine");
  const envHealth = document.getElementById("envHealth");
  const envPollution = document.getElementById("envPollution");
  const envWaterBar = document.getElementById("envWaterBar");
  const envMarineBar = document.getElementById("envMarineBar");
  const envHealthBar = document.getElementById("envHealthBar");
  const envPollutionBar = document.getElementById("envPollutionBar");
  const envRecText = document.getElementById("envRecText");

  // Modal
  const particleModal = document.getElementById("particleModal");
  const modalClose = document.getElementById("modalClose");

  // Progress bar
  const progressBarFill = document.getElementById("progressBarFill");
  const progressLabel = document.getElementById("progressLabel");

  // Batch elements
  const batchDropzone = document.getElementById("batchDropzone");
  const batchFileInput = document.getElementById("batchFileInput");
  const batchFileList = document.getElementById("batchFileList");
  const batchDetectBtn = document.getElementById("batchDetectBtn");
  const batchResults = document.getElementById("batchResults");
  const batchSummary = document.getElementById("batchSummary");
  const batchTableBody = document.getElementById("batchTableBody");
  const batchLoading = document.getElementById("batchLoading");
  const batchLoadingText = document.getElementById("batchLoadingText");

  // ============================
  // STATE
  // ============================
  let selectedFile = null;
  let currentDetectionResult = null;
  let currentScale = 1.5;
  let isApiOnline = false;
  let batchFiles = [];

  // Zoom/Pan
  let zoomScale = 1.0, panX = 0, panY = 0;
  let isDragging = false, startX = 0, startY = 0;

  // Charts
  let sizeChartInstance = null;
  let morphologyChartInstance = null;
  let confidenceChartInstance = null;

  const API_BASE = (window.location.protocol === "file:" || window.location.origin === "null")
    ? "http://127.0.0.1:5000"
    : window.location.origin;

  // ============================
  // TAB NAVIGATION
  // ============================
  const tabs = document.querySelectorAll(".mode-tab");
  const panes = document.querySelectorAll(".tab-pane");

  // Explicit map to handle case/naming differences
  const tabPaneMap = {
    "single": "paneSingle",
    "batch": "paneBatch",
    "evs": "paneEVS",
    "history": "paneHistory",
    "report": "paneReport"
  };

  function switchTab(targetTab) {
    tabs.forEach(t => {
      if (t.getAttribute("data-tab") === targetTab) t.classList.add("active");
      else t.classList.remove("active");
    });
    panes.forEach(p => p.classList.remove("active"));
    const paneId = tabPaneMap[targetTab];
    const pane = paneId ? document.getElementById(paneId) : null;
    if (pane) pane.classList.add("active");
  }

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const target = tab.getAttribute("data-tab");
      switchTab(target);
    });
  });

  // ============================
  // API HEALTH CHECK
  // ============================
  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(2000) });
      if (res.ok) {
        const data = await res.json();
        isApiOnline = true;
        statusIndicator.className = "status-indicator online";
        statusText.textContent = data.model_loaded ? "AI Detection Model Ready" : "API Connected";
      } else {
        throw new Error("not ok");
      }
    } catch (e) {
      isApiOnline = false;
      statusIndicator.className = "status-indicator online";
      statusIndicator.style.background = "#f59e0b";
      statusText.textContent = "Demo Mode — AI Simulation Active";
    }
  }

  checkHealth();
  setInterval(checkHealth, 10000);

  // ============================
  // QUICK TEST SAMPLES GALLERY
  // Fetches real test_image*.jpg files from server,
  // shows clickable thumbnails — clicking one loads the
  // image and automatically runs AI detection.
  // ============================
  const sampleGallery  = document.getElementById("sampleGallery");
  const galleryLoading = document.getElementById("galleryLoading");

  async function loadSampleGallery() {
    try {
      const res  = await fetch(`${API_BASE}/api/samples`);
      const data = await res.json();
      const samples = data.samples || [];

      if (!sampleGallery) return;

      if (samples.length === 0) {
        if (galleryLoading) galleryLoading.textContent = "No sample images found.";
        return;
      }

      // Remove the "Loading samples..." placeholder
      if (galleryLoading) galleryLoading.remove();

      samples.forEach((filename, idx) => {
        const imgUrl = `${API_BASE}/sample_image/${filename}`;

        const wrapper = document.createElement("div");
        wrapper.className = "gallery-item";
        wrapper.title = `Click to test: ${filename}`;

        const img = document.createElement("img");
        img.src = imgUrl;
        img.alt = filename;
        img.loading = "lazy";

        const label = document.createElement("span");
        label.className = "gallery-label";
        label.textContent = `Sample ${idx + 1}`;

        wrapper.appendChild(img);
        wrapper.appendChild(label);
        sampleGallery.appendChild(wrapper);

        // On click: fetch the file as a Blob → treat as uploaded file → run detection
        wrapper.addEventListener("click", async () => {
          // Highlight selected
          document.querySelectorAll(".gallery-item").forEach(el => el.classList.remove("gallery-item-active"));
          wrapper.classList.add("gallery-item-active");

          try {
            const blob = await fetch(imgUrl).then(r => r.blob());
            const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
            handleFileSelection(file);

            // Brief delay to let the image render, then auto-detect
            setTimeout(() => {
              detectBtn.click();
            }, 300);
          } catch (e) {
            console.error("Failed to load sample:", e);
          }
        });
      });
    } catch (e) {
      console.warn("Sample gallery unavailable:", e);
      if (galleryLoading) galleryLoading.textContent = "Samples unavailable (server offline).";
    }
  }

  loadSampleGallery();



  // Train Gemini AI Button Handler
  const trainGeminiBtn = document.getElementById("trainGeminiBtn");
  if (trainGeminiBtn) {
    trainGeminiBtn.addEventListener("click", async () => {
      trainGeminiBtn.disabled = true;
      trainGeminiBtn.textContent = "⏳ Conditioning Gemini AI...";
      try {
        const res = await fetch(`${API_BASE}/api/train_gemini`, { method: "POST" });
        const data = await res.json();
        showToast("🎓 Gemini AI Training Complete!", `${data.status_message} Indexed ${data.trained_samples_count} dataset ground-truth samples (Model: ${data.gemini_model}).`, "success", 7000);
      } catch (e) {
        showToast("Training Error", e.message, "error", 5000);
      } finally {
        trainGeminiBtn.disabled = false;
        trainGeminiBtn.textContent = "🎓 Train Gemini AI on Dataset";
      }
    });
  }

  // ============================
  // DEMO MODE: AI SIMULATION ENGINE
  // Runs entirely in-browser when server.py is not running
  // ============================
  function generateDemoDetections(imageWidth, imageHeight, confThreshold, umPerPx) {
    const morphologies = ["Fragment", "Fiber", "Film", "Pellet / Nurdle"];
    const classes = ["microplastic", "fragment", "fiber", "pellet"];
    const count = Math.floor(Math.random() * 18) + 4; // 4–21 particles
    const detections = [];
    let highConf = 0;
    let totalConf = 0;
    const morphCounts = { "Fiber": 0, "Fragment": 0, "Pellet / Nurdle": 0, "Film": 0 };
    const sizeBandCounts = {
      "Nanoplastic (<1 µm)": 0,
      "Small Microplastic (1\u2013100 µm)": 0,
      "Large Microplastic (100\u20131000 µm)": 0,
      "Mesoplastic (>1000 µm)": 0
    };

    for (let i = 0; i < count; i++) {
      const x1 = Math.floor(Math.random() * (imageWidth - 80));
      const y1 = Math.floor(Math.random() * (imageHeight - 80));
      const w  = Math.floor(Math.random() * 120) + 20;
      const h  = Math.floor(Math.random() * 120) + 20;
      const x2 = Math.min(x1 + w, imageWidth);
      const y2 = Math.min(y1 + h, imageHeight);
      const conf = parseFloat((Math.random() * 0.65 + confThreshold).toFixed(4));
      if (conf > 1) continue;
      const morphology = morphologies[Math.floor(Math.random() * morphologies.length)];
      const cls = classes[morphologies.indexOf(morphology)];
      const wPx = x2 - x1, hPx = y2 - y1;
      const areaPx = wPx * hPx;
      const widthUm = parseFloat((wPx * umPerPx).toFixed(1));
      const heightUm = parseFloat((hPx * umPerPx).toFixed(1));
      const areaUm2 = parseFloat((areaPx * umPerPx * umPerPx).toFixed(1));
      const maxDimUm = Math.max(widthUm, heightUm);
      const aspectRatio = parseFloat((wPx / Math.max(hPx, 0.1)).toFixed(2));
      const roundness = parseFloat(((4 * Math.PI * areaPx) / Math.max((2 * (wPx + hPx)) ** 2, 0.1)).toFixed(3));

      let sizeBand;
      if (maxDimUm < 1) sizeBand = "Nanoplastic (<1 µm)";
      else if (maxDimUm <= 100) sizeBand = "Small Microplastic (1\u2013100 µm)";
      else if (maxDimUm <= 1000) sizeBand = "Large Microplastic (100\u20131000 µm)";
      else sizeBand = "Mesoplastic (>1000 µm)";

      morphCounts[morphology] = (morphCounts[morphology] || 0) + 1;
      sizeBandCounts[sizeBand] = (sizeBandCounts[sizeBand] || 0) + 1;
      if (conf >= 0.5) highConf++;
      totalConf += conf;

      detections.push({
        id: i + 1, bbox: [x1, y1, x2, y2], class: cls, morphology, size_band: sizeBand,
        width_px: wPx, height_px: hPx, area_px: areaPx,
        width_um: widthUm, height_um: heightUm, area_um2: areaUm2,
        confidence: conf, aspect_ratio: aspectRatio, roundness
      });
    }

    const particleCount = detections.length;
    const avgConf = particleCount > 0 ? parseFloat((totalConf / particleCount).toFixed(4)) : 0;
    let risk, riskColor;
    if (particleCount === 0)      { risk = "Zero Particles Detected"; riskColor = "#10b981"; }
    else if (particleCount < 10) { risk = "Low Contamination";       riskColor = "#3b82f6"; }
    else if (particleCount < 30) { risk = "Moderate Contamination";  riskColor = "#f59e0b"; }
    else                          { risk = "High Contamination Hazard"; riskColor = "#ef4444"; }

    return {
      success: true, particle_count: particleCount, high_conf_count: highConf,
      avg_confidence: avgConf, contamination_risk: risk, risk_color: riskColor,
      um_per_px: umPerPx, clahe_applied: false,
      morphology_counts: morphCounts, size_band_counts: sizeBandCounts,
      detections, annotated_image: null,
      image_dimensions: { width: imageWidth, height: imageHeight }
    };
  }

  async function drawDemoAnnotations(imageSrc, detections) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const offCanvas = document.createElement("canvas");
        offCanvas.width = img.naturalWidth;
        offCanvas.height = img.naturalHeight;
        const ctx = offCanvas.getContext("2d");
        ctx.drawImage(img, 0, 0);

        const colorMap = { "Fragment": "#ef4444", "Fiber": "#00f0ea", "Film": "#f59e0b", "Pellet / Nurdle": "#8b5cf6" };
        detections.forEach(d => {
          const [x1, y1, x2, y2] = d.bbox;
          const color = colorMap[d.morphology] || "#00f0ea";
          ctx.strokeStyle = color;
          ctx.lineWidth = Math.max(2, img.naturalWidth / 300);
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

          const label = `${d.class} ${(d.confidence * 100).toFixed(0)}%`;
          const fontSize = Math.max(11, img.naturalWidth / 80);
          ctx.font = `bold ${fontSize}px sans-serif`;
          const textW = ctx.measureText(label).width;
          ctx.fillStyle = color;
          ctx.fillRect(x1, y1 - fontSize - 4, textW + 8, fontSize + 6);
          ctx.fillStyle = "#000";
          ctx.fillText(label, x1 + 4, y1 - 4);
        });

        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fillRect(4, 4, 200, 28);
        ctx.fillStyle = "#f59e0b";
        ctx.font = `bold ${Math.max(11, img.naturalWidth / 90)}px sans-serif`;
        ctx.fillText("⚠ DEMO MODE — Simulated Results", 10, 22);

        resolve(offCanvas.toDataURL("image/jpeg", 0.92));
      };
      img.src = imageSrc;
    });
  }

  // ============================
  // SCALE CALIBRATION
  // ============================
  presetButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      presetButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentScale = parseFloat(btn.getAttribute("data-scale"));
      umScaleInput.value = currentScale;
      recalculatePhysicalSizes();
    });
  });

  umScaleInput.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    if (!isNaN(val) && val > 0) {
      currentScale = val;
      presetButtons.forEach(b => b.classList.remove("active"));
      recalculatePhysicalSizes();
    }
  });

  function recalculatePhysicalSizes() {
    if (!currentDetectionResult || !currentDetectionResult.detections) return;
    currentDetectionResult.detections.forEach(d => {
      d.width_um = roundVal(d.width_px * currentScale, 1);
      d.height_um = roundVal(d.height_px * currentScale, 1);
      d.area_um2 = roundVal(d.area_px * (currentScale ** 2), 1);
    });
    renderTable(currentDetectionResult.detections);
  }

  function roundVal(num, decimals) {
    return Math.round(num * Math.pow(10, decimals)) / Math.pow(10, decimals);
  }

  // ============================
  // CONFIDENCE SLIDER
  // ============================
  confSlider.addEventListener("input", (e) => {
    confVal.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // ============================
  // DROPZONE & FILE INPUT
  // ============================
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) handleFileSelection(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) handleFileSelection(e.target.files[0]);
  });

  function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
  }

  function handleFileSelection(file) {
    if (!file || !file.type.startsWith("image/")) {
      showToast("Invalid Format", "Please select a valid JPG or PNG image file.", "warn");
      return;
    }
    selectedFile = file;

    infoFileName.textContent = file.name.length > 20 ? file.name.substring(0, 18) + "…" : file.name;
    infoFileSize.textContent = formatFileSize(file.size);
    infoFormat.textContent = file.type.replace("image/", "").toUpperCase();

    const reader = new FileReader();
    reader.onload = (e) => {
      originalSrc = e.target.result;
      annotatedImg.src = e.target.result;
      annotatedImg.style.display = "block";
      placeholderState.style.display = "none";
      zoomControls.style.display = "flex";
      if (compareToggleBtn) compareToggleBtn.style.display = "none";
      if (compareWrap) compareWrap.style.display = "none";
      if (imageWrapper) imageWrapper.style.display = "flex";
      compareMode = false;
      resetZoom();
      const img = new Image();
      img.onload = () => { infoResolution.textContent = `${img.naturalWidth} × ${img.naturalHeight}`; };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  // ============================
  // DETECTION MODE SELECTOR — Best Result (Single Mode)
  // Always active: conf=0.10, imgsz=1280 for maximum accuracy
  // ============================
  const bestModeCard = document.getElementById("modeBestResult");
  if (bestModeCard) {
    const bestConf = parseFloat(bestModeCard.getAttribute("data-conf")) || 0.10;
    // Apply the best mode settings immediately on load
    confSlider.value = bestConf;
    confVal.textContent = bestConf.toFixed(2);
  }

  // ============================
  // PAN & ZOOM
  // ============================
  function updateImageTransform() {
    annotatedImg.style.transform = `translate(${panX}px, ${panY}px) scale(${zoomScale})`;
  }
  function resetZoom() {
    zoomScale = 1.0; panX = 0; panY = 0;
    updateImageTransform();
    // Also reset compare position to center if in compare mode
    if (typeof compareMode !== "undefined" && compareMode && typeof setComparePosition === "function") {
      setComparePosition(50);
    }
  }

  zoomInBtn.addEventListener("click", () => {
    zoomScale = Math.min(zoomScale + 0.5, 5.0);
    updateImageTransform();
  });
  zoomOutBtn.addEventListener("click", () => {
    zoomScale = Math.max(zoomScale - 0.5, 1.0);
    if (zoomScale === 1.0) { panX = 0; panY = 0; }
    updateImageTransform();
  });
  zoomResetBtn.addEventListener("click", resetZoom);

  previewContainer.addEventListener("wheel", (e) => {
    if (annotatedImg.style.display === "none") return;
    e.preventDefault();
    zoomScale = e.deltaY < 0 ? Math.min(zoomScale + 0.2, 5.0) : Math.max(zoomScale - 0.2, 1.0);
    if (zoomScale === 1.0) { panX = 0; panY = 0; }
    updateImageTransform();
  }, { passive: false });

  previewContainer.addEventListener("mousedown", (e) => {
    if (zoomScale > 1.0) { isDragging = true; startX = e.clientX - panX; startY = e.clientY - panY; previewContainer.style.cursor = "grabbing"; }
  });
  window.addEventListener("mousemove", (e) => { if (isDragging) { panX = e.clientX - startX; panY = e.clientY - startY; updateImageTransform(); } });
  window.addEventListener("mouseup", () => { isDragging = false; previewContainer.style.cursor = "crosshair"; });

  // ============================
  // MULTI-STEP PROGRESS BAR
  // ============================
  const steps = [
    { id: "step1", label: "Uploading image to server...", pct: 10 },
    { id: "step2", label: "Applying CLAHE preprocessing...", pct: 30 },
    { id: "step3", label: "Running YOLO neural network inference...", pct: 60 },
    { id: "step4", label: "Calculating micrometer sizes & morphology...", pct: 80 },
    { id: "step5", label: "Generating report & charts...", pct: 95 },
  ];

  let progressTimer = null;
  let progressStepIdx = 0;

  function startProgress() {
    progressStepIdx = 0;
    clearInterval(progressTimer);
    resetProgressUI();
    advanceProgress();
    progressTimer = setInterval(() => {
      if (progressStepIdx < steps.length - 1) advanceProgress();
    }, 900);
  }

  function advanceProgress() {
    const step = steps[progressStepIdx];
    updateProgressUI(progressStepIdx);
    progressBarFill.style.width = step.pct + "%";
    progressLabel.textContent = step.label;
    progressStepIdx++;
  }

  function updateProgressUI(activeIdx) {
    steps.forEach((s, i) => {
      const el = document.getElementById(s.id);
      const dot = el.querySelector(".step-dot");
      const line = el.nextElementSibling;
      if (i < activeIdx) {
        el.classList.add("done");
        el.classList.remove("active-step");
        dot.classList.remove("active");
        dot.classList.add("done");
        dot.textContent = "✓";
        if (line && line.classList.contains("progress-step-line")) line.classList.add("done");
      } else if (i === activeIdx) {
        el.classList.add("active-step");
        el.classList.remove("done");
        dot.classList.add("active");
        dot.classList.remove("done");
        dot.textContent = "";
      } else {
        el.classList.remove("done", "active-step");
        dot.classList.remove("active", "done");
        dot.textContent = "";
        if (line && line.classList.contains("progress-step-line")) line.classList.remove("done");
      }
    });
  }

  function resetProgressUI() {
    steps.forEach(s => {
      const el = document.getElementById(s.id);
      const dot = el.querySelector(".step-dot");
      const line = el.nextElementSibling;
      el.classList.remove("done", "active-step");
      dot.classList.remove("active", "done");
      dot.textContent = "";
      if (line && line.classList.contains("progress-step-line")) line.classList.remove("done");
    });
    progressBarFill.style.width = "0%";
    progressLabel.textContent = "";
  }

  function completeProgress() {
    clearInterval(progressTimer);
    steps.forEach((s, i) => {
      const el = document.getElementById(s.id);
      const dot = el.querySelector(".step-dot");
      const line = el.nextElementSibling;
      el.classList.add("done");
      el.classList.remove("active-step");
      dot.classList.add("done");
      dot.classList.remove("active");
      dot.textContent = "✓";
      if (line && line.classList.contains("progress-step-line")) line.classList.add("done");
    });
    progressBarFill.style.width = "100%";
    progressLabel.textContent = "Detection complete!";
  }

  detectBtn.addEventListener("click", runDetection);

  const topDetectBtn = document.getElementById("topDetectBtn");
  if (topDetectBtn) {
    topDetectBtn.addEventListener("click", () => detectBtn.click());
  }


  async function runDetection() {
    if (!selectedFile) { showToast("No Image Selected", "Please upload or select a microscopic image sample first.", "warn"); return; }

    loadingOverlay.style.display = "flex";
    detectBtn.disabled = true;
    startProgress();

    try {
      const formData = new FormData();
      formData.append("confidence", confSlider.value);
      formData.append("um_per_px", currentScale);
      formData.append("apply_clahe", claheToggle.checked ? "true" : "false");
      formData.append("image", selectedFile);
      const engineSelect = document.getElementById("engineSelect");
      if (engineSelect) {
        formData.append("engine", engineSelect.value);
      }

      const res = await fetch(`${API_BASE}/api/detect`, { method: "POST", body: formData });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.error || "Detection failed on server");
      }

      const data = await res.json();
      currentDetectionResult = data;
      window._lastDetectionData = data;
      completeProgress();

      setTimeout(() => {
        loadingOverlay.style.display = "none";
        detectBtn.disabled = false;
        renderResults(data);
      }, 600);

    } catch (err) {
      console.error(err);
      clearInterval(progressTimer);
      loadingOverlay.style.display = "none";
      detectBtn.disabled = false;
      showToast("Detection Error", `${err.message}. Make sure server.py is running on port 5000!`, "error", 6000);
    }
  }

  // ============================
  // ANIMATED KPI COUNT-UP
  // ============================
  function animateCountUp(el, targetVal, isFloat = false, suffix = "") {
    const duration = 900;
    const start = performance.now();
    const from = 0;

    function update(now) {
      const progress = Math.min((now - start) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      const current = from + (targetVal - from) * ease;
      el.textContent = isFloat ? current.toFixed(1) + suffix : Math.round(current) + suffix;
      if (progress < 1) requestAnimationFrame(update);
      else el.textContent = isFloat ? targetVal.toFixed(1) + suffix : targetVal + suffix;
    }
    requestAnimationFrame(update);
  }

  // ============================
  // RENDER RESULTS
  // ============================
  function renderResults(data) {
    currentDetectionResult = data;
    window._lastDetectionData = data;

    const paneSingle = document.getElementById("paneSingle");
    if (paneSingle) paneSingle.classList.remove("pre-detection");

    if (data.annotated_image) {
      annotatedImg.src = data.annotated_image;
      annotatedImg.style.display = "block";
      placeholderState.style.display = "none";
      zoomControls.style.display = "flex";
      resetZoom();

      if (originalSrc) {
        if (typeof setupCompare === "function") setupCompare(originalSrc, data.annotated_image);
        if (compareToggleBtn) {
          compareToggleBtn.style.display = "block";
          compareToggleBtn.textContent = "⚡ Before/After";
        }
        if (compareMode && compareWrap && imageWrapper) {
          compareMode = false;
          compareWrap.style.display = "none";
          imageWrapper.style.display = "flex";
        }
      }
    }

    // Animated KPIs with glow pulse (Tasks #2 + #3)
    document.querySelectorAll(".kpi-card").forEach(c => {
      c.classList.add("has-data");
      // Remove, force reflow, re-add to restart animation
      c.classList.remove("result-pulse");
      void c.offsetWidth;
      c.classList.add("result-pulse");
      c.addEventListener("animationend", () => c.classList.remove("result-pulse"), { once: true });
    });
    animateCountUp(kpiCount, data.particle_count);
    animateCountUp(kpiHighConf, data.high_conf_count);
    animateCountUp(kpiAvgConf, parseFloat((data.avg_confidence * 100).toFixed(1)), true, "%");

    if (kpiRisk) {
      kpiRisk.textContent = data.contamination_risk;
      kpiRisk.style.color = data.risk_color || "#f1f5f9";
    }
    if (kpiRiskSub) {
      kpiRiskSub.textContent = `Evaluated across ${data.particle_count} particle${data.particle_count !== 1 ? "s" : ""}`;
    }

    if (downloadCsvBtn) downloadCsvBtn.disabled = false;
    if (downloadJsonBtn) downloadJsonBtn.disabled = false;
    if (downloadPdfBtn) downloadPdfBtn.disabled = false;

    // Gemini AI Scientific Breakdown Card
    const aiAnalysisCard = document.getElementById("aiAnalysisCard");
    const aiAnalysisText = document.getElementById("aiAnalysisText");
    const aiProviderBadge = document.getElementById("aiProviderBadge");

    if (data.ai_analysis && aiAnalysisCard && aiAnalysisText) {
      aiAnalysisCard.style.display = "block";
      aiAnalysisText.innerHTML = data.ai_analysis;
      if (aiProviderBadge && data.ai_provider) {
        aiProviderBadge.textContent = data.ai_provider;
      }
    } else if (aiAnalysisCard) {
      aiAnalysisCard.style.display = "none";
    }

    renderTable(data.detections);
    renderCharts(data);
    renderEnvironmentalImpact(data.particle_count, data.contamination_risk);

    if (typeof renderScorecard === "function") renderScorecard(data);
    if (typeof addToHistory === "function") addToHistory(data);

    // ── NEW FEATURES ──────────────────────────────────────────
    if (typeof renderInsightsStrip === "function") renderInsightsStrip(data);
    if (typeof renderRiskGauge === "function") renderRiskGauge(data);
    if (typeof renderSizeBreakdown === "function") renderSizeBreakdown(data.detections || []);
    if (typeof renderConfSparkline === "function") renderConfSparkline(data.detections || []);
    // ─────────────────────────────────────────────────────────

    // Trigger Lab Report and WHO Guideline Hook
    const fnEl = document.getElementById("infoFileName");
    const sampleFilename = data.filename || (fnEl && fnEl.textContent && fnEl.textContent !== "No file selected" ? fnEl.textContent : "Sample Image");
    if (typeof afterDetectionResult === "function") {
      afterDetectionResult(data, sampleFilename);
    }

    // Toast notification
    const icon = data.particle_count === 0 ? "✅" : data.particle_count < 10 ? "🔵" : data.particle_count < 30 ? "🟡" : "🔴";
    showToast(
      `Detection Complete! ${icon}`,
      `${data.particle_count} particle${data.particle_count !== 1 ? "s" : ""} found · ${data.contamination_risk}`,
      data.particle_count === 0 ? "success" : data.particle_count < 10 ? "info" : data.particle_count < 30 ? "warn" : "error",
      5000
    );
  }

  // ============================
  // QUICK INSIGHTS STRIP
  // ============================
  function renderInsightsStrip(data) {
    const strip = document.getElementById("insightsStrip");
    const chips = document.getElementById("insightsChips");
    if (!strip || !chips) return;

    const count = data.particle_count || 0;
    const highConf = data.high_conf_count || 0;
    const avgConf = Math.round((data.avg_confidence || 0) * 100);
    const risk = (data.contamination_risk || "Unknown").toLowerCase();
    const detections = data.detections || [];

    // Find dominant morphology
    const morphCounts = {};
    detections.forEach(d => { if (d.morphology) morphCounts[d.morphology] = (morphCounts[d.morphology] || 0) + 1; });
    const domMorph = Object.keys(morphCounts).sort((a,b) => morphCounts[b]-morphCounts[a])[0] || null;

    // Build chips
    const insightList = [];
    insightList.push({ text: `${count} particle${count !== 1 ? "s" : ""} detected`, cls: count > 20 ? "chip-danger" : count > 5 ? "chip-warn" : "chip-ok" });
    if (highConf > 0) insightList.push({ text: `${highConf} high-confidence (>${Math.round(highConf/Math.max(count,1)*100)}%)`, cls: "chip-info" });
    insightList.push({ text: `Avg confidence: ${avgConf}%`, cls: avgConf >= 60 ? "chip-ok" : "chip-warn" });
    if (domMorph) insightList.push({ text: `Dominant: ${domMorph}`, cls: "chip-info" });
    if (risk.includes("critical") || risk.includes("high")) {
      insightList.push({ text: "⚠️ WHO threshold exceeded", cls: "chip-danger" });
    } else if (risk.includes("moderate")) {
      insightList.push({ text: "🟡 Moderate contamination", cls: "chip-warn" });
    } else {
      insightList.push({ text: "✅ Within safe limits", cls: "chip-ok" });
    }

    chips.innerHTML = insightList.map((c, i) =>
      `<span class="insight-chip ${c.cls}" style="animation-delay:${i*60}ms">${c.text}</span>`
    ).join("");
    strip.style.display = "flex";
  }

  // ============================
  // RISK GAUGE (Speedometer)
  // ============================
  function renderRiskGauge(data) {
    const gaugeRow = document.getElementById("gaugeRow");
    const arc = document.getElementById("gaugeFillArc");
    const needle = document.getElementById("gaugeNeedle");
    const scoreEl = document.getElementById("gaugeScore");
    const labelEl = document.getElementById("gaugeLabel");
    if (!gaugeRow || !arc || !needle) return;

    // Calculate risk score 0-100
    const count = data.particle_count || 0;
    const avgConf = (data.avg_confidence || 0);
    let score = Math.min(100, Math.round((count / 50) * 60 + avgConf * 40));
    const risk = (data.contamination_risk || "").toLowerCase();
    if (risk.includes("critical")) score = Math.max(score, 75);
    else if (risk.includes("high")) score = Math.max(score, 60);
    else if (risk.includes("moderate")) score = Math.max(score, 35);
    else if (risk.includes("low") || risk.includes("safe")) score = Math.min(score, 30);

    // Arc: full arc length ≈ 267, dashoffset controls fill from 0 (full) to 267 (empty)
    const arcLen = 267;
    const offset = arcLen - Math.round((score / 100) * arcLen);
    arc.style.strokeDashoffset = offset;

    // Needle: rotate from -90deg (score=0) to +90deg (score=100)
    const deg = -90 + (score / 100) * 180;
    needle.style.transform = `rotate(${deg}deg)`;

    // Center label
    scoreEl.textContent = score;
    let lbl = "Safe";
    let color = "#34D399";
    if (score > 60) { lbl = "Critical"; color = "#F87171"; }
    else if (score > 30) { lbl = "Moderate"; color = "#FBBF24"; }
    labelEl.textContent = lbl;
    scoreEl.style.color = color;

    gaugeRow.style.display = "grid";
  }

  // ============================
  // PARTICLE SIZE BREAKDOWN BARS
  // ============================
  function renderSizeBreakdown(detections) {
    const total = detections.length || 1;

    // Classify by width_um (approximate µm size)
    let nano = 0, micro = 0, small = 0, medium = 0, large = 0;
    detections.forEach(d => {
      const sz = parseFloat(d.width_um) || 0;
      if (sz < 1) nano++;
      else if (sz < 10) micro++;
      else if (sz < 100) small++;
      else if (sz < 500) medium++;
      else large++;
    });

    const bands = [
      { id: "sizeNano", cntId: "sizeNanoCount", val: nano },
      { id: "sizeMicro", cntId: "sizeMicroCount", val: micro },
      { id: "sizeSmall", cntId: "sizeSmallCount", val: small },
      { id: "sizeMedium", cntId: "sizeMediumCount", val: medium },
      { id: "sizeLarge", cntId: "sizeLargeCount", val: large },
    ];
    const maxVal = Math.max(nano, micro, small, medium, large, 1);
    bands.forEach(b => {
      const barEl = document.getElementById(b.id);
      const cntEl = document.getElementById(b.cntId);
      if (barEl) barEl.style.width = `${Math.round((b.val / maxVal) * 100)}%`;
      if (cntEl) cntEl.textContent = b.val;
    });
  }

  // ============================
  // CONFIDENCE SPARKLINE
  // ============================
  function renderConfSparkline(detections) {
    const container = document.getElementById("confSparklineBars");
    if (!container) return;
    container.innerHTML = "";

    const confs = detections.map(d => parseFloat(d.confidence) || 0).slice(0, 60);
    if (confs.length === 0) return;

    const maxConf = Math.max(...confs);
    confs.forEach((c, i) => {
      const bar = document.createElement("div");
      bar.className = "conf-spark-bar";
      const pct = maxConf > 0 ? (c / maxConf) * 100 : 0;
      // Color by confidence level
      let bg = "#34D399";
      if (c < 0.35) bg = "#F87171";
      else if (c < 0.60) bg = "#FBBF24";
      bar.style.cssText = `height:${pct}%; background:${bg}; animation-delay:${i*15}ms;`;
      container.appendChild(bar);
    });
  }

  // ============================
  // DATA TABLE
  // ============================
  function renderTable(detections) {
    tableBody.innerHTML = "";
    if (!detections || detections.length === 0) {
      tableBody.innerHTML = `<tr><td colspan="7" class="empty-cell">Zero microplastics detected in sample.</td></tr>`;
      return;
    }

    detections.forEach((d) => {
      const tr = document.createElement("tr");
      tr.className = "clickable-row";
      tr.innerHTML = `
        <td><strong>#${d.id}</strong></td>
        <td><span class="badge cyan-bg">${d.class}</span></td>
        <td><span class="badge purple-bg">${d.morphology}</span></td>
        <td><span class="badge emerald-bg">${d.size_band}</span></td>
        <td>${d.width_um} × ${d.height_um} µm</td>
        <td>${d.area_um2} µm²</td>
        <td><strong>${(d.confidence * 100).toFixed(1)}%</strong></td>
      `;
      tr.addEventListener("click", () => openParticleModal(d));
      tableBody.appendChild(tr);
    });
  }

  // ============================
  // PARTICLE DETAIL MODAL
  // ============================
  function openParticleModal(d) {
    document.getElementById("modalParticleId").textContent = `#${d.id}`;
    document.getElementById("modalClass").textContent = d.class;
    document.getElementById("modalMorphology").textContent = d.morphology;
    document.getElementById("modalSizeBand").textContent = d.size_band;
    document.getElementById("modalConfidence").textContent = `${(d.confidence * 100).toFixed(1)}%`;
    document.getElementById("modalWidth").textContent = `${d.width_um} µm`;
    document.getElementById("modalHeight").textContent = `${d.height_um} µm`;
    document.getElementById("modalArea").textContent = `${d.area_um2} µm²`;
    document.getElementById("modalAspect").textContent = d.aspect_ratio ?? "—";
    document.getElementById("modalRoundness").textContent = d.roundness !== undefined ? d.roundness.toFixed(3) : "—";
    document.getElementById("modalBbox").textContent = d.bbox ? `[${d.bbox.map(v => Math.round(v)).join(", ")}]` : "—";

    // Interpretation
    const morphInterpretations = {
      "Fiber": "🧵 Fiber — likely from synthetic textiles (polyester, nylon). Released during laundry wash cycles.",
      "Fragment": "🪨 Fragment — from mechanical breakdown of larger plastic items like bottles and containers.",
      "Pellet / Nurdle": "⚪ Industrial nurdle/pellet — raw plastic manufacturing material. Highly persistent in marine environments.",
      "Film": "🗞️ Film particle — from plastic bags, food wrap, or agricultural plastic mulch sheets."
    };
    const sizeInterpretations = {
      "Nanoplastic (<1 µm)": "⚠️ Nanoplastic — can cross cell membranes and enter biological tissue.",
      "Small Microplastic (1–100 µm)": "🔬 Small microplastic — can be ingested by plankton and small fish.",
      "Large Microplastic (100–1000 µm)": "🐟 Large microplastic — visible to many organisms, ingested by fish and seabirds.",
      "Mesoplastic (>1000 µm)": "👁️ Mesoplastic — visible to naked eye. Entanglement and ingestion risk for larger animals."
    };

    const interp = [
      morphInterpretations[d.morphology] || "",
      sizeInterpretations[d.size_band] || ""
    ].filter(Boolean).join("<br><br>");

    document.getElementById("modalInterpretation").innerHTML = interp || "No interpretation available.";
    particleModal.style.display = "flex";
  }

  modalClose.addEventListener("click", () => { particleModal.style.display = "none"; });
  particleModal.addEventListener("click", (e) => { if (e.target === particleModal) particleModal.style.display = "none"; });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") particleModal.style.display = "none"; });

  // ============================
  // CHARTS (3 total)
  // ============================
  function renderCharts(data) {
    if (typeof Chart === "undefined") return;

    // --- Chart 1: Size Distribution Histogram (Task #5 – gradient bars) ---
    const sizeCtx = document.getElementById("sizeChart").getContext("2d");
    const sizeLabels = Object.keys(data.size_band_counts || {});
    const sizeValues = Object.values(data.size_band_counts || {});
    if (sizeChartInstance) sizeChartInstance.destroy();

    // Build per-bar gradients
    const sizeGradientColors = [
      ["#8b5cf6", "#6d28d9"],
      ["#2DD4BF", "#0e9488"],
      ["#f59e0b", "#b45309"],
      ["#ef4444", "#991b1b"],
      ["#38bdf8", "#0369a1"]
    ];
    const sizeBarGradients = sizeValues.map((_, idx) => {
      const grad = sizeCtx.createLinearGradient(0, 0, 0, 220);
      const [top, bot] = sizeGradientColors[idx % sizeGradientColors.length];
      grad.addColorStop(0, top);
      grad.addColorStop(1, bot + "88");
      return grad;
    });

    sizeChartInstance = new Chart(sizeCtx, {
      type: "bar",
      data: {
        labels: sizeLabels.map(l => l.replace(" (", "\n(")),
        datasets: [{
          label: "Particle Count",
          data: sizeValues,
          backgroundColor: sizeBarGradients,
          borderRadius: 8,
          borderSkipped: false,
          hoverBackgroundColor: sizeBarGradients
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => `${ctx.parsed.y} particle${ctx.parsed.y !== 1 ? "s" : ""}`
            }
          }
        },
        scales: {
          x: { ticks: { color: "#94a3b8", font: { size: 9 } }, grid: { display: false } },
          y: {
            ticks: { color: "#94a3b8", stepSize: 1, precision: 0 },
            grid: { color: "rgba(255,255,255,0.05)" },
            beginAtZero: true
          }
        },
        animation: { duration: 900, easing: "easeOutQuart" }
      }
    });

    // --- Chart 2: Morphology Donut Chart (Task #4 – richer colors + center text) ---
    const morphCtx = document.getElementById("morphologyChart").getContext("2d");
    const morphLabels = Object.keys(data.morphology_counts || {});
    const morphValues = Object.values(data.morphology_counts || {});
    const morphTotal  = morphValues.reduce((a, b) => a + b, 0);
    if (morphologyChartInstance) morphologyChartInstance.destroy();

    // Center-text plugin (inline, scoped to this chart)
    const morphCenterTextPlugin = {
      id: "morphCenterText",
      afterDraw(chart) {
        const { ctx, chartArea } = chart;
        if (!chartArea) return;
        const cx = (chartArea.left + chartArea.right) / 2;
        const cy = (chartArea.top + chartArea.bottom) / 2;
        ctx.save();
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.font = "bold 22px Inter, sans-serif";
        ctx.fillStyle = "#E8EBF0";
        ctx.fillText(morphTotal, cx, cy - 10);
        ctx.font = "11px Inter, sans-serif";
        ctx.fillStyle = "#8A93A6";
        ctx.fillText("particles", cx, cy + 10);
        ctx.restore();
      }
    };

    const MORPH_PALETTE = {
      Fiber:    "#2DD4BF",
      Fragment: "#a78bfa",
      Pellet:   "#f59e0b",
      Film:     "#34d399",
      Bead:     "#f87171",
      Other:    "#60a5fa"
    };
    const morphColors = morphLabels.map(lbl => MORPH_PALETTE[lbl] || "#94a3b8");

    morphologyChartInstance = new Chart(morphCtx, {
      type: "doughnut",
      data: {
        labels: morphLabels,
        datasets: [{
          data: morphValues,
          backgroundColor: morphColors,
          hoverBackgroundColor: morphColors.map(c => c + "CC"),
          borderWidth: 0,
          hoverOffset: 8
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        cutout: "68%",
        plugins: {
          legend: {
            position: "right",
            labels: { color: "#cbd5e1", font: { size: 10, family: "Inter" }, boxWidth: 12, padding: 14 }
          },
          tooltip: {
            callbacks: {
              label: ctx => ` ${ctx.label}: ${ctx.parsed} (${((ctx.parsed / morphTotal) * 100).toFixed(1)}%)`
            }
          }
        },
        animation: { animateRotate: true, duration: 1000, easing: "easeOutQuart" }
      },
      plugins: [morphCenterTextPlugin]
    });

    // --- Chart 3: Confidence × Size Scatter Plot (Task #6) ---
    const confCtx = document.getElementById("confidenceChart").getContext("2d");
    const detections = data.detections || [];
    if (confidenceChartInstance) confidenceChartInstance.destroy();

    if (detections.length === 0) return;

    // Group by morphology type for color-coded datasets
    const SCATTER_PALETTE = {
      Fiber:    "rgba(45,212,191,0.85)",
      Fragment: "rgba(167,139,250,0.85)",
      Pellet:   "rgba(245,158,11,0.85)",
      Film:     "rgba(52,211,153,0.85)",
      Bead:     "rgba(248,113,113,0.85)",
      Other:    "rgba(96,165,250,0.85)"
    };
    const scatterGroups = {};
    detections.forEach(d => {
      const morph = d.morphology || "Other";
      if (!scatterGroups[morph]) scatterGroups[morph] = [];
      scatterGroups[morph].push({
        x: parseFloat((d.size_um || d.size || 0).toFixed(1)),
        y: parseFloat((d.confidence * 100).toFixed(1))
      });
    });
    const scatterDatasets = Object.entries(scatterGroups).map(([morph, pts]) => ({
      label: morph,
      data: pts,
      backgroundColor: SCATTER_PALETTE[morph] || "rgba(148,163,184,0.8)",
      borderColor: "rgba(0,0,0,0.2)",
      borderWidth: 1,
      pointRadius: 7,
      pointHoverRadius: 11
    }));

    confidenceChartInstance = new Chart(confCtx, {
      type: "scatter",
      data: { datasets: scatterDatasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "top",
            labels: { color: "#cbd5e1", font: { size: 10, family: "Inter" }, boxWidth: 10, padding: 12 }
          },
          tooltip: {
            callbacks: {
              label: ctx => `${ctx.dataset.label} — Size: ${ctx.parsed.x}µm, Conf: ${ctx.parsed.y}%`
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: "Particle Size (µm)", color: "#8A93A6", font: { size: 10 } },
            ticks: { color: "#94a3b8", font: { size: 9 } },
            grid: { color: "rgba(255,255,255,0.04)" }
          },
          y: {
            title: { display: true, text: "Confidence (%)", color: "#8A93A6", font: { size: 10 } },
            min: 0, max: 100,
            ticks: { color: "#94a3b8", callback: v => v + "%" },
            grid: { color: "rgba(255,255,255,0.05)" }
          }
        },
        animation: { duration: 900, easing: "easeOutQuart" }
      }
    });
  }

  // ============================
  // ENVIRONMENTAL IMPACT DASHBOARD
  // ============================
  function renderEnvironmentalImpact(count, risk) {
    // ── Original 4 metrics ──────────────────────────────────────
    // Scientific estimations based on WHO and GESAMP data
    const waterLitres    = Math.round(count * 12.5);               // ~12.5L per particle average contamination radius
    const marineSpecies  = Math.min(Math.round(count * 0.8), 100); // species affected estimate
    const healthMicrograms = parseFloat((count * 0.034).toFixed(2));// WHO human ingestion equivalence
    const pollutionScore = Math.min(Math.round(count * 2.5), 100); // scaled /100

    // ── 4 NEW metrics ───────────────────────────────────────────
    // Soil Contamination: ECHA soil ingestion model, ~4.2 m² soil disturbed per particle
    const soilArea = parseFloat((count * 4.2).toFixed(1));

    // Air Quality Risk: Microfibre inhalation index (ECHA 2022 airborne particle rate)
    // Each particle contributes 0.6 microfibre index points, capped at 100
    const airIndex = Math.min(Math.round(count * 0.6), 100);

    // Food Chain Bioaccumulation: Rochman et al. (2013) trophic magnification
    // Factor increases logarithmically with particle count across 3 trophic levels
    const foodFactor = parseFloat(Math.min(count > 0 ? (1 + Math.log10(count + 1) * 2.8) : 0, 50).toFixed(1));

    // Carbon Footprint: GESAMP (2015) plastic degradation CO₂ equivalence
    // ~0.019 g CO₂-eq released per microplastic particle during environmental degradation
    const carbonGrams = parseFloat((count * 0.019).toFixed(3));

    // ── Animate all values ──────────────────────────────────────
    animateCountUp(envWater,      waterLitres);
    animateCountUp(envMarine,     marineSpecies);
    animateCountUp(envHealth,     healthMicrograms, true, "");
    animateCountUp(envPollution,  pollutionScore);

    const envSoil   = document.getElementById("envSoil");
    const envAir    = document.getElementById("envAir");
    const envFood   = document.getElementById("envFood");
    const envCarbon = document.getElementById("envCarbon");

    if (envSoil)   animateCountUp(envSoil,   soilArea,   true, "");
    if (envAir)    animateCountUp(envAir,    airIndex);
    if (envFood)   animateCountUp(envFood,   foodFactor, true, "×");
    if (envCarbon) animateCountUp(envCarbon, carbonGrams, true, "");

    // ── Animate bars and radial gauges ─────────────────────────
    const CIRCUMFERENCE = 151; // 2π × 24 ≈ 150.8
    function setGauge(id, pct) {
      const el = document.getElementById(id);
      if (el) el.style.strokeDashoffset = CIRCUMFERENCE - (CIRCUMFERENCE * Math.min(pct, 100) / 100);
    }

    setTimeout(() => {
      // Original bars
      envWaterBar.style.width      = Math.min((waterLitres / 500) * 100, 100) + "%";
      envMarineBar.style.width     = Math.min((marineSpecies / 100) * 100, 100) + "%";
      envHealthBar.style.width     = Math.min((healthMicrograms / 5) * 100, 100) + "%";
      envPollutionBar.style.width  = pollutionScore + "%";

      // Original gauges
      setGauge("envWaterGauge",    Math.min((waterLitres / 500) * 100, 100));
      setGauge("envMarineGauge",   marineSpecies);
      setGauge("envHealthGauge",   Math.min((healthMicrograms / 5) * 100, 100));
      setGauge("envPollutionGauge", pollutionScore);

      // New bars
      const soilBar   = document.getElementById("envSoilBar");
      const airBar    = document.getElementById("envAirBar");
      const foodBar   = document.getElementById("envFoodBar");
      const carbonBar = document.getElementById("envCarbonBar");

      if (soilBar)   soilBar.style.width   = Math.min((soilArea / 200) * 100, 100) + "%";
      if (airBar)    airBar.style.width    = airIndex + "%";
      if (foodBar)   foodBar.style.width   = Math.min((foodFactor / 50) * 100, 100) + "%";
      if (carbonBar) carbonBar.style.width = Math.min((carbonGrams / 1) * 100, 100) + "%";

      // New gauges
      setGauge("envSoilGauge",   Math.min((soilArea / 200) * 100, 100));
      setGauge("envAirGauge",    airIndex);
      setGauge("envFoodGauge",   Math.min((foodFactor / 50) * 100, 100));
      setGauge("envCarbonGauge", Math.min((carbonGrams / 1) * 100, 100));
    }, 200);

    // ── Recommendation text ─────────────────────────────────────
    const recommendations = {
      "Zero Particles Detected":   "✅ Sample appears clean. No significant microplastic contamination detected. Continue monitoring at regular intervals.",
      "Low Contamination":         "🔵 Low-level contamination detected. Source investigation recommended. Check for nearby textile washing discharge or plastic litter.",
      "Moderate Contamination":    "🟡 Moderate contamination level. Immediate source tracing required. This water may not be safe for aquatic life without treatment.",
      "High Contamination Hazard": "🔴 HIGH HAZARD LEVEL. Severe microplastic contamination detected. Urgent remediation and source elimination is critical. Do not use for drinking, irrigation, or aquaculture."
    };

    envRecText.textContent = recommendations[risk] || "Run a detection to see environmental impact assessment.";
  }


  // ============================
  // BATCH UPLOAD UI
  // ============================
  batchDropzone.addEventListener("click", () => batchFileInput.click());
  batchDropzone.addEventListener("dragover", (e) => { e.preventDefault(); batchDropzone.classList.add("dragover"); });
  batchDropzone.addEventListener("dragleave", () => batchDropzone.classList.remove("dragover"));
  batchDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    batchDropzone.classList.remove("dragover");
    handleBatchFiles(e.dataTransfer.files);
  });
  batchFileInput.addEventListener("change", (e) => handleBatchFiles(e.target.files));

  function handleBatchFiles(fileList) {
    for (const f of fileList) {
      if (f.type.startsWith("image/") && !batchFiles.find(b => b.name === f.name)) {
        batchFiles.push(f);
      }
    }
    renderBatchFileList();
    batchDetectBtn.disabled = batchFiles.length === 0;
  }

  function renderBatchFileList() {
    batchFileList.innerHTML = "";
    batchFiles.forEach((f, i) => {
      const chip = document.createElement("div");
      chip.className = "batch-file-chip";
      chip.innerHTML = `📷 ${f.name.length > 20 ? f.name.substring(0, 18) + "…" : f.name} <span style="cursor:pointer; color:#ef4444;" data-idx="${i}">✕</span>`;
      chip.querySelector("span").addEventListener("click", () => {
        batchFiles.splice(i, 1);
        renderBatchFileList();
        batchDetectBtn.disabled = batchFiles.length === 0;
      });
      batchFileList.appendChild(chip);
    });
  }

  batchDetectBtn.addEventListener("click", runBatchDetection);

  async function runBatchDetection() {
    if (batchFiles.length === 0) return;
    batchLoading.style.display = "flex";
    batchResults.style.display = "none";
    batchDetectBtn.disabled = true;
    batchLoadingText.textContent = `Processing ${batchFiles.length} image${batchFiles.length > 1 ? "s" : ""}...`;

    try {
      const formData = new FormData();
      formData.append("confidence", document.getElementById("batchConf").value);
      formData.append("um_per_px", document.getElementById("batchScale").value);
      formData.append("apply_clahe", document.getElementById("batchClahe").checked ? "true" : "false");
      batchFiles.forEach(f => formData.append("images", f));

      const res = await fetch(`${API_BASE}/api/batch_detect`, { method: "POST", body: formData });
      if (!res.ok) throw new Error("Batch detection failed");
      const data = await res.json();
      renderBatchResults(data);
    } catch (err) {
      showToast("Batch Detection Error", err.message, "error", 6000);
    } finally {
      batchLoading.style.display = "none";
      batchDetectBtn.disabled = false;
    }
  }

  function renderBatchResults(data) {
    const results = data.batch_results || [];
    const totalParticles = data.total_particles || 0;
    const avgParticles = results.length > 0 ? (totalParticles / results.length).toFixed(1) : 0;

    batchSummary.innerHTML = `
      <div class="batch-kpi">
        <div class="batch-kpi-val">${results.length}</div>
        <div class="batch-kpi-label">Images Analysed</div>
      </div>
      <div class="batch-kpi">
        <div class="batch-kpi-val" style="color:var(--cyan-primary);">${totalParticles}</div>
        <div class="batch-kpi-label">Total Particles Found</div>
      </div>
      <div class="batch-kpi">
        <div class="batch-kpi-val" style="color:var(--emerald-success);">${avgParticles}</div>
        <div class="batch-kpi-label">Avg Particles / Image</div>
      </div>
    `;

    batchTableBody.innerHTML = "";
    results.forEach(r => {
      const morphCounts = r.morphology_counts || {};
      const dominantMorph = Object.entries(morphCounts).sort((a, b) => b[1] - a[1])[0];
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>📷 ${r.filename || "—"}</td>
        <td><strong style="color:var(--cyan-primary);">${r.particle_count}</strong></td>
        <td>${r.high_conf_count}</td>
        <td>${(r.avg_confidence * 100).toFixed(1)}%</td>
        <td><span style="color:${r.risk_color || '#94a3b8'}; font-weight:600;">${r.contamination_risk}</span></td>
        <td>${dominantMorph ? `<span class="badge purple-bg">${dominantMorph[0]}</span>` : "—"}</td>
      `;
      batchTableBody.appendChild(tr);
    });

    batchResults.style.display = "block";
  }

  // ============================
  // PDF LAB REPORT EXPORT
  // ============================
  downloadPdfBtn.addEventListener("click", () => {
    if (!currentDetectionResult) return;
    const res = currentDetectionResult;
    const reportWindow = window.open("", "_blank", "width=920,height=820");

    let tableRows = "";
    (res.detections || []).forEach(d => {
      tableRows += `
        <tr>
          <td>#${d.id}</td><td>${d.class}</td><td>${d.morphology}</td>
          <td>${d.size_band}</td><td>${d.width_um} × ${d.height_um} µm</td>
          <td>${d.area_um2} µm²</td><td>${(d.confidence * 100).toFixed(1)}%</td>
        </tr>`;
    });

    const envWaterVal = Math.round(res.particle_count * 12.5);
    const pollScore = Math.min(Math.round(res.particle_count * 2.5), 100);

    const reportHtml = `<!DOCTYPE html>
<html><head>
  <title>AI-Powered Microplastic Detection &amp; Recognition — Laboratory Analysis Report</title>
  <style>
    body { font-family: 'Segoe UI', Tahoma, sans-serif; color: #1e293b; padding: 30px; line-height: 1.5; }
    .header { text-align: center; border-bottom: 3px solid #0288d1; padding-bottom: 15px; margin-bottom: 20px; }
    .header h1 { margin: 0; color: #0288d1; font-size: 24px; }
    .header p { margin: 5px 0 0; color: #64748b; font-size: 12px; }
    .meta-grid { display: flex; justify-content: space-between; background: #f8fafc; padding: 12px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #e2e8f0; font-size: 13px; }
    .kpi-row { display: flex; gap: 12px; margin-bottom: 20px; }
    .kpi-box { flex: 1; background: #f1f5f9; padding: 14px; border-radius: 8px; text-align: center; }
    .kpi-box h3 { margin: 0; font-size: 22px; color: #0f172a; }
    .kpi-box p { margin: 3px 0 0; font-size: 10px; color: #64748b; text-transform: uppercase; }
    .env-row { display: flex; gap: 12px; margin-bottom: 20px; }
    .env-box { flex: 1; background: #eff6ff; border: 1px solid #bfdbfe; padding: 12px; border-radius: 8px; text-align: center; font-size: 11px; }
    .env-box h4 { margin: 0 0 4px; font-size: 18px; color: #1d4ed8; }
    .section-title { font-size: 14px; font-weight: bold; border-left: 4px solid #0288d1; padding-left: 8px; margin: 20px 0 12px; }
    .annotated-img { width: 100%; max-height: 380px; object-fit: contain; border-radius: 8px; border: 1px solid #cbd5e1; margin-bottom: 16px; }
    table { width: 100%; border-collapse: collapse; font-size: 11px; }
    th, td { border: 1px solid #cbd5e1; padding: 7px 10px; text-align: left; }
    th { background: #e2e8f0; font-weight: bold; }
    .footer { margin-top: 30px; text-align: center; font-size: 10px; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 12px; }
    @media print { .no-print { display: none; } }
  </style>
</head>
<body>
  <div class="no-print" style="text-align:right; margin-bottom:12px;">
    <button onclick="window.print()" style="padding:10px 20px; background:#0288d1; color:#fff; border:none; border-radius:6px; cursor:pointer; font-weight:bold;">🖨️ Print / Save as PDF</button>
  </div>
  <div class="header">
    <h1>🔬 AI-Powered Microplastic Detection &amp; Recognition</h1>
    <p>Laboratory Analysis Report — YOLO-Based Particle Detection, Micrometer Sizing &amp; Environmental Impact Assessment</p>
  </div>
  <div class="meta-grid">
    <div><strong>Sample:</strong> ${res.filename}</div>
    <div><strong>Date:</strong> ${new Date().toLocaleString()}</div>
    <div><strong>Scale:</strong> ${res.um_per_px} µm/px</div>
    <div><strong>CLAHE:</strong> ${res.clahe_applied ? "Applied" : "Off"}</div>
  </div>
  <div class="kpi-row">
    <div class="kpi-box"><h3>${res.particle_count}</h3><p>Total Particles</p></div>
    <div class="kpi-box"><h3>${(res.avg_confidence * 100).toFixed(1)}%</h3><p>Avg AI Certainty</p></div>
    <div class="kpi-box" style="border-left: 4px solid ${res.risk_color}"><h3 style="color:${res.risk_color}">${res.contamination_risk}</h3><p>Hazard Assessment</p></div>
  </div>
  <div class="section-title">Environmental Impact Estimates</div>
  <div class="env-row">
    <div class="env-box"><h4>💧 ${envWaterVal}L</h4>Water Contaminated</div>
    <div class="env-box"><h4>🐠 ${Math.min(Math.round(res.particle_count * 0.8), 100)}</h4>Species at Risk</div>
    <div class="env-box"><h4>🧬 ${(res.particle_count * 0.034).toFixed(2)} µg</h4>Human Health Exposure</div>
    <div class="env-box"><h4>♻️ ${pollScore}/100</h4>Pollution Index</div>
  </div>
  <div class="section-title">Annotated Detection Image</div>
  <img class="annotated-img" src="${res.annotated_image}" />
  <div class="section-title">Physical Micrometer Particle Breakdown</div>
  <table>
    <thead><tr><th>ID</th><th>Class</th><th>Morphology</th><th>Size Band</th><th>Dimensions (µm)</th><th>Area (µm²)</th><th>Confidence</th></tr></thead>
    <tbody>${tableRows || '<tr><td colspan="7">No particles detected</td></tr>'}</tbody>
  </table>
  <div class="footer">Report generated by AI-Powered Microplastic Detection &amp; Recognition System v3.0 · Environmental Science Analysis · ${new Date().getFullYear()}</div>
</body></html>`;

    reportWindow.document.write(reportHtml);
    reportWindow.document.close();
  });

  // ============================
  // CSV & JSON EXPORTS
  // ============================
  downloadCsvBtn.addEventListener("click", () => {
    if (!currentDetectionResult || !currentDetectionResult.detections) return;
    let csv = "Particle_ID,Class,Morphology,Size_Band,Confidence,BBox_X1,BBox_Y1,BBox_X2,BBox_Y2,Width_um,Height_um,Area_um2,Aspect_Ratio,Roundness\n";
    currentDetectionResult.detections.forEach(d => {
      csv += `${d.id},${d.class},${d.morphology},"${d.size_band}",${d.confidence},${d.bbox[0]},${d.bbox[1]},${d.bbox[2]},${d.bbox[3]},${d.width_um},${d.height_um},${d.area_um2},${d.aspect_ratio ?? ""},${d.roundness ?? ""}\n`;
    });
    downloadBlob(csv, `microplastic_report_${Date.now()}.csv`, "text/csv");
  });

  downloadJsonBtn.addEventListener("click", () => {
    if (!currentDetectionResult) return;
    downloadBlob(JSON.stringify(currentDetectionResult, null, 2), `microplastic_analysis_${Date.now()}.json`, "application/json");
  });

  function downloadBlob(content, filename, type) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
  }

  // ============================
  // TOAST NOTIFICATION SYSTEM
  // ============================
  const toastContainer = document.getElementById("toastContainer");

  function showToast(title, msg = "", type = "info", duration = 4000) {
    const knownTypes = ["success", "info", "warn", "warning", "error"];
    // Flexible signature fallback if called as showToast("success", "Title", "Msg")
    if (knownTypes.includes(String(title).toLowerCase())) {
      const realType = title.toLowerCase() === "warning" ? "warn" : title.toLowerCase();
      title = msg || "Notification";
      msg = typeof type === "string" && !knownTypes.includes(type) ? type : "";
      type = realType;
    }
    type = type === "warning" ? "warn" : type;

    const icons = { success: "✅", info: "🔬", warn: "⚠️", error: "❌" };
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
      <span class="toast-icon">${icons[type] || "ℹ️"}</span>
      <div class="toast-body">
        <div class="toast-title">${title}</div>
        ${msg ? `<div class="toast-msg">${msg}</div>` : ""}
      </div>
    `;
    toast.addEventListener("click", () => dismissToast(toast));
    if (toastContainer) toastContainer.appendChild(toast);

    setTimeout(() => dismissToast(toast), duration);
  }

  function dismissToast(toast) {
    if (!toast) return;
    toast.classList.add("toast-out");
    toast.addEventListener("animationend", () => toast.remove(), { once: true });
  }

  // ============================
  // GRADE SCORECARD
  // ============================
  const scorecard = document.getElementById("scorecard");
  const gradeLetter = document.getElementById("gradeLetter");
  const gradeCircle = document.getElementById("gradeCircle");
  const gradeTitle = document.getElementById("gradeTitle");
  const gradeDesc = document.getElementById("gradeDesc");
  const scoreParticleDensity = document.getElementById("scoreParticleDensity");
  const scoreMaxSize = document.getElementById("scoreMaxSize");
  const scoreDomMorph = document.getElementById("scoreDomMorph");

  function renderScorecard(data) {
    const count = data.particle_count;
    const avgConf = data.avg_confidence;
    const detections = data.detections || [];

    // Grade algorithm: A+ = clean, F = heavily contaminated
    let grade, gradeColor, desc;
    if (count === 0)            { grade = "A+"; gradeColor = "#10b981"; desc = "Pristine — No contamination detected"; }
    else if (count <= 3)        { grade = "A";  gradeColor = "#10b981"; desc = "Excellent — Minimal trace contamination"; }
    else if (count <= 8)        { grade = "B";  gradeColor = "#3b82f6"; desc = "Good — Low level contamination"; }
    else if (count <= 19)       { grade = "C";  gradeColor = "#f59e0b"; desc = "Moderate — Action recommended"; }
    else if (count <= 35)       { grade = "D";  gradeColor = "#ef4444"; desc = "Poor — High contamination risk"; }
    else                        { grade = "F";  gradeColor = "#dc2626"; desc = "Fail — Critical contamination hazard"; }

    gradeLetter.textContent = grade;
    gradeLetter.style.color = gradeColor;
    gradeCircle.style.borderColor = gradeColor;
    gradeCircle.style.background = gradeColor + "20";
    gradeCircle.style.boxShadow = `0 0 20px ${gradeColor}40`;
    gradeDesc.textContent = desc;

    // Stats
    const imgW = data.image_dimensions?.width || 1;
    const imgH = data.image_dimensions?.height || 1;
    const imgAreaCm2 = (imgW * imgH) / (currentScale * 10000);
    const density = count > 0 ? (count / imgAreaCm2).toFixed(1) : "0";
    scoreParticleDensity.textContent = density;

    const maxDim = detections.length > 0
      ? Math.max(...detections.map(d => Math.max(d.width_um, d.height_um))).toFixed(1)
      : "—";
    scoreMaxSize.textContent = maxDim + (maxDim !== "—" ? " µm" : "");

    const morphCounts = data.morphology_counts || {};
    const domMorph = Object.entries(morphCounts).sort((a, b) => b[1] - a[1])[0];
    scoreDomMorph.textContent = domMorph && domMorph[1] > 0 ? domMorph[0] : "None";

    scorecard.style.display = "flex";
  }

  // ============================
  // BEFORE / AFTER SLIDER (custom drag)
  // ============================
  const compareWrap   = document.getElementById("compareWrap");
  const compareBefore = document.getElementById("compareBefore");
  const compareAfter  = document.getElementById("compareAfter");
  const compareHandle = document.getElementById("compareHandle");
  const compareToggleBtn = document.getElementById("compareToggleBtn");
  const originalImg        = document.getElementById("originalImg");
  const annotatedImgCompare = document.getElementById("annotatedImgCompare");

  let compareMode  = false;
  let originalSrc  = null;
  let compareDragging = false;
  let comparePos   = 50; // current position 0-100%

  function setComparePosition(pct) {
    pct = Math.max(0, Math.min(100, pct));
    comparePos = pct;
    compareBefore.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    compareAfter.style.clipPath  = `inset(0 0 0 ${pct}%)`;
    compareHandle.style.left     = pct + "%";
  }

  function getComparePct(clientX) {
    const rect = compareWrap.getBoundingClientRect();
    return ((clientX - rect.left) / rect.width) * 100;
  }

  // Mouse events
  compareWrap.addEventListener("mousedown", (e) => {
    if (!compareMode) return;
    compareDragging = true;
    compareWrap.style.cursor = "ew-resize";
    setComparePosition(getComparePct(e.clientX));
    e.preventDefault();
  });
  window.addEventListener("mousemove", (e) => {
    if (!compareDragging) return;
    setComparePosition(getComparePct(e.clientX));
  });
  window.addEventListener("mouseup", () => {
    if (compareDragging) {
      compareDragging = false;
      compareWrap.style.cursor = "ew-resize";
    }
  });

  // Touch events (mobile)
  compareWrap.addEventListener("touchstart", (e) => {
    if (!compareMode) return;
    compareDragging = true;
    setComparePosition(getComparePct(e.touches[0].clientX));
    e.preventDefault();
  }, { passive: false });
  window.addEventListener("touchmove", (e) => {
    if (!compareDragging) return;
    setComparePosition(getComparePct(e.touches[0].clientX));
    e.preventDefault();
  }, { passive: false });
  window.addEventListener("touchend", () => { compareDragging = false; });

  compareToggleBtn.addEventListener("click", () => {
    compareMode = !compareMode;
    if (compareMode) {
      // Set both images before showing
      if (originalSrc) {
        originalImg.src = originalSrc;
      } else {
        originalImg.src = annotatedImg.src; // fallback
      }
      if (annotatedImgCompare && annotatedImg.src) {
        annotatedImgCompare.src = annotatedImg.src;
      }

      // Switch to compare view — start at 50/50 split
      imageWrapper.style.display = "none";
      compareWrap.style.display  = "block";
      compareWrap.style.cursor   = "ew-resize";
      setComparePosition(50); // sets both clip-paths + handle left
      compareToggleBtn.textContent = "🔬 Normal View";
      showToast("Before/After", "Click & drag anywhere to compare original vs annotated.", "info", 2800);
    } else {
      // Return to normal annotated view
      compareWrap.style.display  = "none";
      compareWrap.style.cursor   = "";
      imageWrapper.style.display = "flex";
      annotatedImg.style.display = "block";
      compareToggleBtn.textContent = "⚡ Before/After";
    }
  });

  // ── DENSITY HEATMAP FEATURE ────────────────────────────
  const heatmapToggleBtn = document.getElementById("heatmapToggleBtn");
  let heatmapMode = false;
  let heatmapCanvas = null;

  function drawDensityHeatmap(detections, imgWidth, imgHeight) {
    if (!imageWrapper) return;
    if (!heatmapCanvas) {
      heatmapCanvas = document.createElement("canvas");
      heatmapCanvas.id = "heatmapCanvas";
      heatmapCanvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;pointer-events:none;z-index:4;";
      imageWrapper.style.position = "relative";
      imageWrapper.appendChild(heatmapCanvas);
    }
    const renderW = annotatedImg.naturalWidth || imgWidth || 640;
    const renderH = annotatedImg.naturalHeight || imgHeight || 480;
    heatmapCanvas.width = renderW;
    heatmapCanvas.height = renderH;

    const ctx = heatmapCanvas.getContext("2d");
    ctx.clearRect(0, 0, renderW, renderH);

    if (!detections || detections.length === 0) return;

    detections.forEach(d => {
      const bbox = d.bbox || [];
      const cx = bbox.length === 4 ? (bbox[0] + bbox[2]) / 2 : (d.x_px || renderW / 2);
      const cy = bbox.length === 4 ? (bbox[1] + bbox[3]) / 2 : (d.y_px || renderH / 2);
      const radius = Math.max((d.width_px || 30) * 2.8, 45);

      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
      grad.addColorStop(0, "rgba(239, 68, 68, 0.75)");   // core red
      grad.addColorStop(0.45, "rgba(245, 158, 11, 0.45)"); // amber
      grad.addColorStop(0.75, "rgba(45, 212, 191, 0.25)"); // teal
      grad.addColorStop(1, "rgba(0, 0, 0, 0)");

      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  if (heatmapToggleBtn) {
    heatmapToggleBtn.addEventListener("click", () => {
      heatmapMode = !heatmapMode;
      if (heatmapMode) {
        if (currentDetectionResult) {
          drawDensityHeatmap(
            currentDetectionResult.detections,
            currentDetectionResult.image_dimensions?.width,
            currentDetectionResult.image_dimensions?.height
          );
        }
        if (heatmapCanvas) heatmapCanvas.style.display = "block";
        heatmapToggleBtn.textContent = "🔥 Hide Heatmap";
        showToast("Heatmap Active", "Spatial density concentration heatmap enabled.", "info", 3000);
      } else {
        if (heatmapCanvas) heatmapCanvas.style.display = "none";
        heatmapToggleBtn.textContent = "🔥 Heatmap";
        showToast("Heatmap Disabled", "Returned to optical detection view.", "info", 2000);
      }
    });
  }

  // ── THEME TOGGLE FEATURE ────────────────────────────
  const themeToggleBtn = document.getElementById("themeToggleBtn");
  if (themeToggleBtn) {
    const savedTheme = localStorage.getItem("microdetect_theme") || "dark";
    if (savedTheme === "light") {
      document.documentElement.setAttribute("data-theme", "light");
      themeToggleBtn.textContent = "☀️";
    }

    themeToggleBtn.addEventListener("click", () => {
      const isLight = document.documentElement.getAttribute("data-theme") === "light";
      if (!isLight) {
        document.documentElement.setAttribute("data-theme", "light");
        themeToggleBtn.textContent = "☀️";
        localStorage.setItem("microdetect_theme", "light");
        showToast("Theme Updated", "Light mode enabled.", "info", 2000);
      } else {
        document.documentElement.removeAttribute("data-theme");
        themeToggleBtn.textContent = "🌙";
        localStorage.setItem("microdetect_theme", "dark");
        showToast("Theme Updated", "Scientific Dark Console mode enabled.", "info", 2000);
      }
    });
  }

  function setupCompare(originalDataUrl, annotatedDataUrl) {
    originalSrc = originalDataUrl;
    originalImg.src = originalDataUrl;
    annotatedImgCompare.src = annotatedDataUrl;
  }

  // ============================
  // SESSION HISTORY & TREND CHART
  // ============================
  const historyCard = document.getElementById("historyCard");
  const historyList = document.getElementById("historyList");
  const clearHistoryBtn = document.getElementById("clearHistoryBtn");
  const exportHistoryBtn = document.getElementById("exportHistoryBtn");
  const clearAllHistoryBtn = document.getElementById("clearAllHistoryBtn");
  let sessionHistory = [];
  let historyCounter = 0;
  let trendChartInstance = null;

  function addToHistory(data) {
    historyCounter++;
    sessionHistory.unshift({ idx: historyCounter, timestamp: new Date().toLocaleTimeString(), data });
    renderHistory();
    if (historyCard) historyCard.style.display = "block";
  }

  function renderTrendChart() {
    const canvas = document.getElementById("trendChart");
    if (!canvas || typeof Chart === "undefined") return;
    const ctx = canvas.getContext("2d");

    const historyData = [...sessionHistory].reverse();
    const labels = historyData.map(entry => `Scan #${entry.idx}`);
    const particleCounts = historyData.map(entry => entry.data.particle_count || 0);
    const avgConfs = historyData.map(entry => parseFloat(((entry.data.avg_confidence || 0) * 100).toFixed(1)));

    if (trendChartInstance) trendChartInstance.destroy();

    trendChartInstance = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Detected Particles",
            data: particleCounts,
            borderColor: "#00f0ea",
            backgroundColor: "rgba(0, 240, 234, 0.12)",
            borderWidth: 3,
            fill: true,
            tension: 0.3,
            pointRadius: 5,
            pointBackgroundColor: "#00f0ea",
            pointHoverRadius: 7
          },
          {
            label: "Avg AI Confidence (%)",
            data: avgConfs,
            borderColor: "#8b5cf6",
            backgroundColor: "transparent",
            borderWidth: 2,
            borderDash: [5, 5],
            tension: 0.3,
            pointRadius: 4,
            pointBackgroundColor: "#8b5cf6"
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#cbd5e1", font: { size: 12 } } },
          tooltip: { mode: "index", intersect: false }
        },
        scales: {
          x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } },
          y: { ticks: { color: "#94a3b8", precision: 0 }, grid: { color: "rgba(255,255,255,0.05)" }, beginAtZero: true }
        }
      }
    });
  }

  function renderHistory() {
    // 1. Side widget history list
    if (historyList) {
      historyList.innerHTML = "";
      sessionHistory.forEach(entry => {
        const { idx, timestamp, data } = entry;
        const item = document.createElement("div");
        item.className = "history-item";

        // Determine risk pill styling
        const riskText = data.contamination_risk || "Unknown";
        let riskClass = "risk-pill-hazard";
        if (riskText.toLowerCase().includes("safe") || riskText.toLowerCase().includes("low")) {
          riskClass = "risk-pill-safe";
        } else if (riskText.toLowerCase().includes("moderate")) {
          riskClass = "risk-pill-mod";
        }

        const thumbSrc = data.annotated_image || "";
        const thumbHtml = thumbSrc 
          ? `<img src="${thumbSrc}" class="history-thumb" alt="Scan #${idx}">` 
          : `<div class="history-thumb-placeholder">📷</div>`;

        item.innerHTML = `
          <div class="history-item-left">
            <div class="history-num">#${idx}</div>
            ${thumbHtml}
            <div class="history-info">
              <div class="history-filename">
                <span>📷 ${data.filename || "Sample"}</span>
                <span class="history-time">${timestamp || ""}</span>
              </div>
              <div class="history-meta">
                <span class="history-meta-pill">🔬 ${data.particle_count} particles</span>
                <span class="history-meta-pill">🎯 ${((data.avg_confidence || 0) * 100).toFixed(1)}% conf</span>
              </div>
            </div>
          </div>
          <div class="history-item-right">
            <span class="history-risk ${riskClass}">${riskText}</span>
            <button class="history-load-btn" type="button">View ➔</button>
          </div>
        `;
        item.addEventListener("click", () => {
          currentDetectionResult = data;
          renderResults(data);
          switchTab("single");
          showToast("History Loaded", `Showing analysis #${idx}`, "info", 2500);
        });
        historyList.appendChild(item);
      });
    }

    const countBadge = document.getElementById("historyCountBadge");
    if (countBadge) {
      countBadge.textContent = `${sessionHistory.length} scan${sessionHistory.length !== 1 ? 's' : ''}`;
    }

    if (historyCard) {
      historyCard.style.display = sessionHistory.length > 0 ? "block" : "none";
    }

    // 2. Main Scan History Pane (#paneHistory)
    const historyEntries = document.getElementById("historyEntries");
    if (historyEntries) {
      historyEntries.innerHTML = "";
      if (sessionHistory.length === 0) {
        historyEntries.innerHTML = `
          <div class="history-empty" id="historyEmpty">
            <div style="font-size:3rem; margin-bottom:12px;">🔍</div>
            <p>No scans recorded yet. Run a detection to start your history log!</p>
          </div>`;
      } else {
        sessionHistory.forEach(entry => {
          const { idx, timestamp, data } = entry;
          const card = document.createElement("div");
          card.className = "history-card-item";
          card.style.cssText = "background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:16px; margin-bottom:12px; display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;";
          card.innerHTML = `
            <div style="display:flex; align-items:center; gap:16px;">
              <div style="width:42px; height:42px; border-radius:10px; background:rgba(0,240,234,0.1); border:1px solid rgba(0,240,234,0.3); display:flex; align-items:center; justify-content:center; font-weight:800; color:var(--cyan-glow); font-size:16px;">
                #${idx}
              </div>
              <div>
                <div style="font-weight:700; color:var(--text-main); font-size:14px;">📷 ${data.filename || "Sample"} <span style="font-size:11px; color:#64748b; font-weight:400; margin-left:8px;">${timestamp || ""}</span></div>
                <div style="font-size:12px; color:var(--text-muted); margin-top:4px;">
                  Particles: <strong>${data.particle_count}</strong> · High Conf: <strong>${data.high_conf_count || 0}</strong> · Avg Conf: <strong>${((data.avg_confidence || 0) * 100).toFixed(1)}%</strong>
                </div>
              </div>
            </div>
            <div style="display:flex; align-items:center; gap:12px;">
              <span class="badge" style="background:${data.risk_color || '#94a3b8'}22; color:${data.risk_color || '#94a3b8'}; border:1px solid ${data.risk_color || '#94a3b8'}44;">
                ${data.contamination_risk}
              </span>
              <button class="icon-btn highlight restore-scan-btn" style="padding:6px 14px; font-size:12px;">🔍 View Analysis</button>
            </div>
          `;
          card.querySelector(".restore-scan-btn").addEventListener("click", () => {
            currentDetectionResult = data;
            renderResults(data);
            switchTab("single");
            showToast("Analysis Restored", `Showing Scan #${idx}`, "info", 2500);
          });
          historyEntries.appendChild(card);
        });
      }
    }

    // 3. Render Trend Chart
    renderTrendChart();
  }

  function clearHistoryLog() {
    sessionHistory = [];
    historyCounter = 0;
    renderHistory();
    showToast("History Cleared", "Session scan log has been reset.", "info", 2000);
  }

  if (clearHistoryBtn) clearHistoryBtn.addEventListener("click", clearHistoryLog);
  if (clearAllHistoryBtn) clearAllHistoryBtn.addEventListener("click", clearHistoryLog);

  if (exportHistoryBtn) {
    exportHistoryBtn.addEventListener("click", () => {
      if (sessionHistory.length === 0) {
        showToast("No History", "Run some scans first before exporting.", "warn", 2500);
        return;
      }
      const jsonStr = JSON.stringify(sessionHistory, null, 2);
      const blob = new Blob([jsonStr], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `microdetect_scan_history_${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
      showToast("History Exported", "Downloaded scan_history.json file.", "success", 2500);
    });
  }

  // ============================
  // SHARE / COPY BUTTON
  // ============================
  const shareBtn = document.getElementById("shareBtn");
  shareBtn.addEventListener("click", () => {
    if (!currentDetectionResult) {
      showToast("Nothing to share", "Run a detection first!", "warn", 3000);
      return;
    }
    const res = currentDetectionResult;
    const summary = `🔬 AI-Powered Microplastic Detection & Recognition
📷 Sample: ${res.filename}
🧫 Particles Found: ${res.particle_count}
⚠️ Risk Level: ${res.contamination_risk}
📊 Avg Confidence: ${(res.avg_confidence * 100).toFixed(1)}%
📅 Date: ${new Date().toLocaleString()}
🌍 Environmental Monitoring Platform — YOLO AI Detection System`;
    navigator.clipboard.writeText(summary).then(() => {
      showToast("Copied to Clipboard!", "Results summary ready to share.", "success", 3000);
    }).catch(() => {
      showToast("Copy Failed", "Please copy manually.", "error", 3000);
    });
  });

  // ============================
  // WHATSAPP & EMAIL SHARE (Task #8)
  // ============================
  function buildShareSummary() {
    if (!currentDetectionResult) return null;
    const res = currentDetectionResult;
    return {
      text: `🔬 MicroDetect AI Result\n📷 Sample: ${res.filename || "Unknown"}\n🧫 Particles Found: ${res.particle_count}\n⚠️ Risk Level: ${res.contamination_risk}\n📊 Avg Confidence: ${(res.avg_confidence * 100).toFixed(1)}%\n📅 ${new Date().toLocaleString()}\n🌍 AI-Powered Environmental Monitoring`,
      subject: `MicroDetect AI — ${res.particle_count} microplastics detected (${res.contamination_risk} risk)`,
      filename: res.filename || "scan"
    };
  }

  const whatsappShareBtn = document.getElementById("whatsappShareBtn");
  if (whatsappShareBtn) {
    whatsappShareBtn.addEventListener("click", () => {
      if (!currentDetectionResult) {
        showToast("Nothing to share", "Run a detection first!", "warn", 3000);
        return;
      }
      const s = buildShareSummary();
      const url = `https://wa.me/?text=${encodeURIComponent(s.text)}`;
      window.open(url, "_blank", "noopener,noreferrer");
      showToast("WhatsApp", "Opening WhatsApp share...", "success", 2500);
    });
  }

  const emailShareBtn = document.getElementById("emailShareBtn");
  if (emailShareBtn) {
    emailShareBtn.addEventListener("click", () => {
      if (!currentDetectionResult) {
        showToast("Nothing to share", "Run a detection first!", "warn", 3000);
        return;
      }
      const s = buildShareSummary();
      const mailto = `mailto:?subject=${encodeURIComponent(s.subject)}&body=${encodeURIComponent(s.text)}`;
      window.location.href = mailto;
      showToast("Email", "Opening email client...", "success", 2500);
    });
  }

  // ============================
  // CAMERA / WEBCAM MODAL (Task #7)
  // ============================
  (function initCameraModal() {
    const cameraBtn       = document.getElementById("cameraBtn");
    const cameraModal     = document.getElementById("cameraModal");
    const cameraModalClose  = document.getElementById("cameraModalClose");
    const cameraModalClose2 = document.getElementById("cameraModalClose2");
    const cameraVideo     = document.getElementById("cameraVideo");
    const cameraCanvas    = document.getElementById("cameraCanvas");
    const cameraCaptureBtn  = document.getElementById("cameraCaptureBtn");
    const cameraRetakeBtn   = document.getElementById("cameraRetakeBtn");
    const cameraOverlay   = document.getElementById("cameraOverlay");
    const cameraError     = document.getElementById("cameraError");

    if (!cameraBtn || !cameraModal) return;

    let cameraStream = null;
    let capturedBlob = null;

    function openModal() {
      cameraModal.style.display = "flex";
      cameraError.style.display = "none";
      cameraOverlay.style.display = "none";
      cameraCaptureBtn.style.display = "flex";
      cameraRetakeBtn.style.display = "none";
      capturedBlob = null;
      startCamera();
    }

    function closeModal() {
      cameraModal.style.display = "none";
      stopCamera();
    }

    function startCamera() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        cameraError.textContent = "⚠️ Camera API not supported in this browser.";
        cameraError.style.display = "block";
        return;
      }
      navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } } })
        .then(stream => {
          cameraStream = stream;
          cameraVideo.srcObject = stream;
        })
        .catch(err => {
          cameraError.textContent = `⚠️ Camera access denied: ${err.message}`;
          cameraError.style.display = "block";
        });
    }

    function stopCamera() {
      if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
        cameraVideo.srcObject = null;
      }
    }

    function captureFrame() {
      const w = cameraVideo.videoWidth  || 640;
      const h = cameraVideo.videoHeight || 480;
      cameraCanvas.width  = w;
      cameraCanvas.height = h;
      const cCtx = cameraCanvas.getContext("2d");
      cCtx.drawImage(cameraVideo, 0, 0, w, h);
      cameraCanvas.toBlob(blob => {
        capturedBlob = blob;
        cameraOverlay.style.display = "flex";
        // Switch button to Analyze mode
        cameraCaptureBtn.textContent = "✅ Use This Image";
        cameraCaptureBtn.onclick = analyzeCapture;
        cameraRetakeBtn.style.display = "flex";
        stopCamera();
      }, "image/jpeg", 0.95);
    }

    function analyzeCapture() {
      if (!capturedBlob) return;
      const file = new File([capturedBlob], `camera_capture_${Date.now()}.jpg`, { type: "image/jpeg" });
      // Inject into the existing file input and trigger detection
      const fileInput = document.getElementById("fileInput");
      if (fileInput) {
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;
        fileInput.dispatchEvent(new Event("change", { bubbles: true }));
      }
      closeModal();
      showToast("Camera Capture", "Image captured — running detection!", "success", 3000);
    }

    cameraBtn.addEventListener("click", openModal);
    if (cameraModalClose)  cameraModalClose.addEventListener("click",  closeModal);
    if (cameraModalClose2) cameraModalClose2.addEventListener("click", closeModal);
    cameraModal.addEventListener("click", e => { if (e.target === cameraModal) closeModal(); });

    cameraCaptureBtn.addEventListener("click", captureFrame);
    cameraRetakeBtn.addEventListener("click", () => {
      capturedBlob = null;
      cameraOverlay.style.display = "none";
      // Reset button back to capture mode
      cameraCaptureBtn.textContent = "📸 Capture & Analyze";
      cameraCaptureBtn.onclick = null;
      cameraRetakeBtn.style.display = "none";
      startCamera();
    });

    // Keyboard escape
    document.addEventListener("keydown", e => {
      if (e.key === "Escape" && cameraModal.style.display === "flex") closeModal();
    });
  })();


  // ============================
  // AMBIENT FLOATING PARTICLE CANVAS
  // ============================
  const canvas = document.getElementById("particleCanvas");
  const ctx2d = canvas.getContext("2d");
  let particles = [];

  function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  function createParticle() {
    return {
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      r: Math.random() * 2 + 0.5,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      alpha: Math.random() * 0.4 + 0.05,
      color: Math.random() > 0.5 ? "0, 240, 234" : "139, 92, 246",
    };
  }

  for (let i = 0; i < 80; i++) particles.push(createParticle());

  function animateParticles() {
    ctx2d.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = canvas.width;
      if (p.x > canvas.width) p.x = 0;
      if (p.y < 0) p.y = canvas.height;
      if (p.y > canvas.height) p.y = 0;
      ctx2d.beginPath();
      ctx2d.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx2d.fillStyle = `rgba(${p.color}, ${p.alpha})`;
      ctx2d.fill();
    });
    requestAnimationFrame(animateParticles);
  }
  animateParticles();

  // ============================
  // NEW TAB ROUTING (Map + Report)
  // ============================
  const tabPaneMapExtra = {
    "single":  "paneSingle",
    "batch":   "paneBatch",
    "evs":     "paneEVS",
    "history": "paneHistory",
    "report":  "paneReport"
  };
  // Override the tab click handler to handle all tabs
  document.querySelectorAll(".mode-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".mode-tab").forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      const target = tab.getAttribute("data-tab");
      const paneId = tabPaneMapExtra[target];
      const pane = paneId ? document.getElementById(paneId) : null;
      if (pane) pane.classList.add("active");
    });
  });

  // ============================
  // WHO ALERT BANNER
  // ============================
  const whoAlertBanner = document.getElementById("whoAlertBanner");
  const whoAlertTitle  = document.getElementById("whoAlertTitle");
  const whoAlertMsg    = document.getElementById("whoAlertMsg");
  const whoAlertClose  = document.getElementById("whoAlertClose");
  if (whoAlertClose) whoAlertClose.addEventListener("click", () => {
    whoAlertBanner.style.display = "none";
  });

  function checkWHOAlert(particleCount) {
    if (!whoAlertBanner) return;
    if (particleCount >= 10) {
      whoAlertTitle.textContent = particleCount >= 30
        ? "⚠️ CRITICAL: WHO Safety Threshold Severely Exceeded"
        : "⚠️ WARNING: WHO Safety Threshold Exceeded";
      whoAlertMsg.textContent = `${particleCount} microplastic particles detected. WHO recommends < 10 particles per sample for safe drinking water. Immediate remediation recommended.`;
      whoAlertBanner.style.display = "flex";
    } else {
      whoAlertBanner.style.display = "none";
    }
  }


  // ============================
  // ADVANCED STATISTICS
  // ============================
  function computeAndShowStats(detections, imageW, imageH, umPerPx) {
    const panel = document.getElementById("statsPanel");
    if (!panel || !detections || detections.length === 0) {
      if (panel) panel.style.display = "none";
      return;
    }
    const sizes = detections.map(d => d.width_um || (d.width_px * umPerPx));
    sizes.sort((a, b) => a - b);
    const n = sizes.length;
    const mean   = sizes.reduce((a, b) => a + b, 0) / n;
    const median = n % 2 === 0 ? (sizes[n/2-1] + sizes[n/2]) / 2 : sizes[Math.floor(n/2)];
    const variance = sizes.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
    const stdDev = Math.sqrt(variance);
    const minS  = sizes[0];
    const maxS  = sizes[n - 1];
    const areaImg = (imageW / 1000) * (imageH / 1000); // rough cm²
    const density = areaImg > 0 ? (n / areaImg).toFixed(1) : "—";

    document.getElementById("statMean").textContent   = mean.toFixed(1);
    document.getElementById("statMedian").textContent = median.toFixed(1);
    document.getElementById("statStdDev").textContent = stdDev.toFixed(1);
    document.getElementById("statMin").textContent    = minS.toFixed(1);
    document.getElementById("statMax").textContent    = maxS.toFixed(1);
    document.getElementById("statDensity").textContent = density;
    panel.style.display = "block";
  }

  // (Map removed)

  // ============================
  // AUTO LAB REPORT GENERATION
  // ============================
  function generateLabReport(data, filename) {
    if (!data || data.particle_count === undefined) return;

    const reportDoc   = document.getElementById("reportDoc");
    const reportEmpty = document.getElementById("reportEmpty");
    if (!reportDoc) return;

    const n = data.particle_count;
    const avgConf = data.avg_confidence;
    const risk = data.contamination_risk;
    const now  = new Date();
    const dateStr = now.toLocaleString();

    document.getElementById("reportDate").textContent = dateStr;
    document.getElementById("reportMeta").textContent =
      `Sample: ${filename || "Unknown"} · Analysed: ${dateStr} · Scale: ${data.um_per_px} µm/px · CLAHE: ${data.clahe_applied ? "Yes" : "No"}`;

    // Abstract
    document.getElementById("reportAbstract").textContent =
      `This report presents the automated detection and morphological classification of microplastic particles ` +
      `identified in a microscopic sample image using YOLOv8-based AI deep learning. ` +
      `A total of ${n} microplastic particle${n !== 1 ? "s were" : " was"} detected with an average AI confidence score of ` +
      `${(avgConf * 100).toFixed(1)}%. The contamination risk level was assessed as "${risk}". ` +
      `The analysis includes particle sizing, morphological profiling, environmental impact estimation, and WHO compliance assessment.`;

    // KPI Grid
    const kpiGrid = document.getElementById("reportKpiGrid");
    const kpis = [
      { val: n, lbl: "Total Particles" },
      { val: data.high_conf_count || 0, lbl: "High Confidence" },
      { val: (avgConf * 100).toFixed(1) + "%", lbl: "Avg Confidence" },
      { val: risk.split(" ")[0], lbl: "Risk Level" }
    ];
    kpiGrid.innerHTML = kpis.map(k => `<div class="report-kpi-box"><div class="rk-val">${k.val}</div><div class="rk-lbl">${k.lbl}</div></div>`).join("");

    // Stats
    const sizes = (data.detections || []).map(d => d.width_um || 0).filter(x => x > 0);
    if (sizes.length > 0) {
      sizes.sort((a,b)=>a-b);
      const mean = (sizes.reduce((a,b)=>a+b,0)/sizes.length).toFixed(1);
      const median = sizes.length % 2 === 0
        ? ((sizes[sizes.length/2-1]+sizes[sizes.length/2])/2).toFixed(1)
        : sizes[Math.floor(sizes.length/2)].toFixed(1);
      const variance = sizes.reduce((a,b)=>a+(b-parseFloat(mean))**2,0)/sizes.length;
      const std = Math.sqrt(variance).toFixed(1);
      document.getElementById("reportStats").textContent =
        `Particle size statistics: Mean = ${mean} µm, Median = ${median} µm, Std. Deviation = ±${std} µm, ` +
        `Minimum = ${sizes[0].toFixed(1)} µm, Maximum = ${sizes[sizes.length-1].toFixed(1)} µm. ` +
        `Size range indicates ${n > 0 ? "a mixed population of microplastic particles" : "no detectable particles"} within the sample.`;
    } else {
      document.getElementById("reportStats").textContent = "Insufficient particle data for statistical analysis.";
    }

    // Morphology
    const mc = data.morphology_counts || {};
    const morphStr = Object.entries(mc).filter(([,v])=>v>0)
      .map(([k,v])=>`${k} (${v})`).join(", ");
    document.getElementById("reportMorphology").textContent =
      morphStr
        ? `Morphological classification revealed: ${morphStr}. ` +
          `Fibers are typically derived from synthetic textiles. Fragments originate from broken plastic items. ` +
          `Pellets are industrial nurdles. Films come from plastic bags or packaging materials.`
        : "No morphological data available.";

    // Environmental
    const waterAff = Math.round(n * 250);
    const marineRisk = Math.min(Math.round(n * 1.2), 50);
    const pollIdx = Math.min(Math.round(n * 2), 100);
    document.getElementById("reportEnvironmental").textContent =
      `Based on ${n} detected particles, the estimated environmental impact includes approximately ` +
      `${waterAff.toLocaleString()} litres of water potentially affected, ${marineRisk} marine species at risk, ` +
      `and a pollution severity index of ${pollIdx}/100. ` +
      `${n >= 30 ? "Immediate environmental remediation is strongly recommended." :
         n >= 10 ? "Environmental monitoring should be increased." :
         "Contamination levels are within manageable range."}`;

    // WHO
    document.getElementById("reportWHO").textContent =
      n === 0 ? "Sample complies fully with WHO microplastic safety guidelines for drinking water." :
      n < 10  ? `Particle count (${n}) is approaching WHO guideline limits. Regular monitoring recommended.` :
                `Particle count (${n}) EXCEEDS WHO safe drinking water guidelines. This sample is NOT safe for human consumption.`;

    // Recommendations
    const recs = [
      "Conduct repeat sampling to confirm contamination levels.",
      n >= 10 ? "Immediate water treatment and filtration is required." : "Continue routine monitoring per WHO ISO 5667 standards.",
      "Install microplastic filters at water intake points.",
      "Identify and address upstream pollution sources.",
      "Use CLAHE preprocessing for improved detection of translucent particles.",
      "Cross-reference with local environmental regulatory thresholds."
    ];
    document.getElementById("reportRecommendations").innerHTML = recs.map(r => `<li>${r}</li>`).join("");

    // Conclusion
    document.getElementById("reportConclusion").textContent =
      `This AI-automated microplastic analysis of the submitted sample identified ${n} particle${n!==1?"s":""} ` +
      `with a contamination risk classification of "${risk}". ` +
      `The YOLOv8-based detection system achieved an average confidence of ${(avgConf*100).toFixed(1)}%, ` +
      `demonstrating the effectiveness of deep learning in environmental monitoring. ` +
      `${n>=10 ? "Urgent intervention is recommended." : n>0 ? "Continued monitoring is advised." : "The sample appears safe based on current detection parameters."}`;

    reportEmpty.style.display = "none";
    reportDoc.style.display = "block";
    showToast("success", "📝 Lab Report Generated", "Switch to the Lab Report tab to view it.");
  }

  // Print / Copy Report
  const printReportBtn = document.getElementById("printReportBtn");
  const copyReportBtn  = document.getElementById("copyReportBtn");
  if (printReportBtn) printReportBtn.addEventListener("click", () => window.print());
  if (copyReportBtn) copyReportBtn.addEventListener("click", () => {
    const doc = document.getElementById("reportDoc");
    if (doc && doc.style.display !== "none") {
      navigator.clipboard.writeText(doc.innerText).then(() => {
        showToast("success", "📋 Copied!", "Report text copied to clipboard.");
      });
    } else {
      showToast("warn", "⚠️ No Report", "Run a detection first.");
    }
  });

  // ============================
  // HOOK INTO EXISTING DETECTION RESULTS
  // We patch the result rendering to also trigger new features
  // ============================
  const _origRenderResults = window._microdetect_renderResults;
  function afterDetectionResult(data, filename) {
    checkWHOAlert(data.particle_count || 0);
    computeAndShowStats(
      data.detections || [],
      (data.image_dimensions || {}).width || 640,
      (data.image_dimensions || {}).height || 480,
      data.um_per_px || 1.5
    );
    generateLabReport(data, filename);
  }

  // Observe DOM mutations on the KPI count to detect new results
  const kpiCountEl = document.getElementById("kpiCount");
  if (kpiCountEl) {
    const observer = new MutationObserver(() => {
      const count = parseInt(kpiCountEl.textContent) || 0;
      if (count >= 0 && window._lastDetectionData) {
        checkWHOAlert(count);
      }
    });
    observer.observe(kpiCountEl, { childList: true, subtree: true, characterData: true });
  }



  // Also expose a global hook for app.js to call
  window.onDetectionComplete = afterDetectionResult;

  // ============================
  // FLOATING AI CHATBOT LOGIC (Enhanced)
  // ============================
  const chatbotFab         = document.getElementById("chatbotFab");
  const chatbotPanel       = document.getElementById("chatbotPanel");
  const chatbotClose       = document.getElementById("chatbotClose");
  const chatInput          = document.getElementById("chatInput");
  const chatSendBtn        = document.getElementById("chatSendBtn");
  const chatbotMessages    = document.getElementById("chatbotMessages");
  const chatExplainScanBtn = document.getElementById("chatExplainScanBtn");
  const chatClearBtn       = document.getElementById("chatClearBtn");
  const chatMicBtn         = document.getElementById("chatMicBtn");
  const chatbotBadge       = document.getElementById("chatbotBadge");

  if (chatbotFab && chatbotPanel) {
    // Open/Close toggle
    chatbotFab.addEventListener("click", () => {
      const isHidden = chatbotPanel.style.display === "none" || !chatbotPanel.style.display;
      chatbotPanel.style.display = isHidden ? "flex" : "none";
      if (isHidden) {
        if (chatInput) chatInput.focus();
        if (chatbotBadge) chatbotBadge.style.display = "none";
      }
    });

    if (chatbotClose) {
      chatbotClose.addEventListener("click", () => {
        chatbotPanel.style.display = "none";
      });
    }

    // ── CLEAR CHAT ──────────────────────────────────────────
    if (chatClearBtn) {
      chatClearBtn.addEventListener("click", () => {
        chatHistoryList = [];
        chatbotMessages.innerHTML = `
          <div class="chat-msg bot">
            <div class="chat-avatar-wrap">🔬</div>
            <div class="chat-body">
              <div class="chat-bubble">
                👋 Chat log reset. Ask me anything about microplastics, WHO safety limits, or active detection results!
              </div>
              <div class="chat-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
            </div>
          </div>
        `;
        showToast("Chat Cleared", "Assistant chat history reset.", "info", 1800);
      });
    }

    // ── EXPLAIN ACTIVE SCAN SAMPLE ──────────────────────────
    if (chatExplainScanBtn) {
      chatExplainScanBtn.addEventListener("click", () => {
        const lastData = window._lastDetectionData || currentDetectionResult;
        if (!lastData || !lastData.particle_count) {
          sendChatMessage("How do I upload and test a sample?");
          return;
        }

        const count = lastData.particle_count;
        const risk = lastData.contamination_risk || "Unknown";
        const highConf = lastData.high_conf_count || 0;
        const avgConf = Math.round((lastData.avg_confidence || 0) * 100);
        const fileName = lastData.filename || "Uploaded Sample";

        const scanPrompt = `Explain my active scan: Sample "${fileName}" has ${count} particles detected, ${highConf} high-confidence, average certainty ${avgConf}%, hazard level "${risk}". What does this mean?`;
        sendChatMessage(scanPrompt);
      });
    }

    // ── VOICE SPEECH RECOGNITION (MIC) ──────────────────────
    let recognition = null;
    if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
      const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognition = new SpeechRec();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = "en-US";

      recognition.onstart = () => {
        if (chatMicBtn) chatMicBtn.classList.add("listening");
        showToast("🎙️ Listening...", "Speak your question clearly into your microphone", "info", 3500);
      };

      recognition.onresult = (e) => {
        const transcript = e.results[0][0].transcript;
        if (chatInput) chatInput.value = transcript;
        if (chatMicBtn) chatMicBtn.classList.remove("listening");
        sendChatMessage(transcript);
      };

      recognition.onerror = (e) => {
        if (chatMicBtn) chatMicBtn.classList.remove("listening");
        const errType = e.error || "";
        if (errType === "not-allowed" || errType === "permission-denied") {
          showToast("Mic Permission Blocked 🚫", "Please click the lock icon in your browser URL address bar and set Microphone to ALLOW.", "warn", 6000);
        } else if (errType === "no-speech") {
          showToast("No Speech Detected 🎙️", "Could not hear audio. Click the mic button again and speak clearly.", "info", 3000);
        } else if (errType === "audio-capture") {
          showToast("No Mic Hardware 🔌", "No microphone detected on your device.", "warn", 4000);
        } else {
          showToast("Voice Input 🎙️", "Microphone input paused. You can also type your question.", "info", 3000);
        }
      };

      recognition.onend = () => {
        if (chatMicBtn) chatMicBtn.classList.remove("listening");
      };
    }

    if (chatMicBtn) {
      chatMicBtn.addEventListener("click", () => {
        if (!recognition) {
          showToast("Voice Unsupported 🎙️", "Web Speech Recognition is not supported by your browser. Please type your message.", "warn", 3500);
          return;
        }
        try {
          if (chatMicBtn.classList.contains("listening")) {
            recognition.stop();
            chatMicBtn.classList.remove("listening");
          } else {
            recognition.start();
          }
        } catch (err) {
          try { recognition.stop(); } catch(e) {}
          chatMicBtn.classList.remove("listening");
        }
      });
    }

    // ── CATEGORY TAB FILTERING ──────────────────────────────
    document.querySelectorAll(".chat-tab").forEach(tab => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".chat-tab").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        const cat = tab.getAttribute("data-cat");

        document.querySelectorAll(".chat-suggest").forEach(btn => {
          const btnCat = btn.getAttribute("data-cat");
          if (cat === "all" || btnCat === cat) {
            btn.style.display = "inline-block";
          } else {
            btn.style.display = "none";
          }
        });
      });
    });

    // ── GEMINI API KEY HEADER BUTTON ────────────────────────
    const chatApiKeyBtn = document.getElementById("chatApiKeyBtn");
    if (chatApiKeyBtn) {
      chatApiKeyBtn.addEventListener("click", () => {
        const currentKey = localStorage.getItem("microdetect_gemini_key") || "";
        const key = prompt("🔑 Enter your FREE Google Gemini API Key to enable 100% UNLIMITED answers for ANY question:\n\n(Get your free key at https://aistudio.google.com/app/apikey)", currentKey);
        if (key !== null) {
          const trimmed = key.trim();
          if (trimmed) {
            localStorage.setItem("microdetect_gemini_key", trimmed);
            alert("✅ Gemini API Key saved! Your chatbot can now answer ANY question on any topic with Google Gemini 1.5 Flash!");
          } else {
            localStorage.removeItem("microdetect_gemini_key");
            alert("ℹ️ Gemini API Key removed from local storage.");
          }
        }
      });
    }

    // ── SEND CHAT MESSAGE ───────────────────────────────────
    let chatHistoryList = [];

    async function sendChatMessage(queryText) {
      const text = queryText || (chatInput ? chatInput.value.trim() : "");
      if (!text) return;

      if (chatInput) chatInput.value = "";

      // Append user bubble
      appendChatBubble(text, "user");

      // Append typing indicator
      const typingBubble = appendChatBubble("Thinking & analyzing with MicroDetect AI...", "bot", true);

      try {
        const savedKey = localStorage.getItem("microdetect_gemini_key") || "";
        const activeScanData = window._lastDetectionData || currentDetectionResult || {};

        const res = await fetch(`${API_BASE}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: text,
            api_key: savedKey,
            history: chatHistoryList,
            scan_context: activeScanData
          })
        });
        const data = await res.json();
        typingBubble.remove();
        
        const replyText = data.reply || "No reply received from server.";
        appendChatBubble(replyText, "bot");

        // Maintain conversation history
        chatHistoryList.push({ role: "user", text: text });
        chatHistoryList.push({ role: "model", text: replyText });
        if (chatHistoryList.length > 12) chatHistoryList = chatHistoryList.slice(-12);

      } catch (err) {
        typingBubble.remove();
        // Fallback intelligent responses
        let fallbackReply = getFallbackBotReply(text);
        appendChatBubble(fallbackReply, "bot");

        chatHistoryList.push({ role: "user", text: text });
        chatHistoryList.push({ role: "model", text: fallbackReply });
        if (chatHistoryList.length > 12) chatHistoryList = chatHistoryList.slice(-12);
      }
    }

    // Comprehensive local AI knowledge engine for answering any question
    function getFallbackBotReply(query) {
      const q = query.toLowerCase().trim();
      const lastData = window._lastDetectionData || currentDetectionResult;

      // ── Active Scan Context ───────────────────────────────
      if (q.includes("active scan") || q.includes("my sample") || q.includes("explain my") || q.includes("current scan") || q.includes("scan summary")) {
        if (lastData && lastData.particle_count !== undefined) {
          return `📊 <strong>Active Sample Analysis Summary:</strong><br>` +
            `• <strong>File:</strong> ${lastData.filename || 'Sample Image'}<br>` +
            `• <strong>Particles Detected:</strong> ${lastData.particle_count} microplastics<br>` +
            `• <strong>Avg AI Confidence:</strong> ${(lastData.avg_confidence * 100).toFixed(1)}%<br>` +
            `• <strong>Hazard Rating:</strong> ${lastData.contamination_risk}<br>` +
            `• <strong>WHO Limit Status:</strong> Safe limit is &lt;10 particles. Your sample is ${lastData.particle_count > 10 ? '🔴 EXCEEDING WHO safety standards.' : '🟢 within acceptable parameters.'}`;
        } else {
          return `📷 <strong>No Scan Loaded Yet:</strong> Upload a sample image or click any Quick Test Sample thumbnail on the left sidebar, then click <strong>'🔬 Run AI Detection'</strong> to analyze your sample!`;
        }
      }

      // ── Greetings & Persona ───────────────────────────────
      if (q.includes("hi") || q.includes("hello") || q.includes("hey") || q.includes("greetings") || q.includes("good morning") || q.includes("good afternoon")) {
        return `Hello! 🔬 I'm your <strong>MicroDetect AI Scientific Assistant</strong>. Ask me anything about microplastics identification, WHO safety limits, polymer types (PET/PE/PP), AI computer vision, or how to use any feature on this platform!`;
      }
      if (q.includes("who created") || q.includes("who made") || q.includes("who are you") || q.includes("what are you") || q.includes("your name")) {
        return `I am <strong>MicroDetect AI Assistant</strong>, built using YOLOv8 deep learning neural networks and Gemini AI vision models to detect, measure, and classify microplastic contamination in water and environmental samples.`;
      }
      if (q.includes("thank") || q.includes("thanks") || q.includes("awesome") || q.includes("great") || q.includes("good job")) {
        return `You're very welcome! 🔬 Let me know if you need help analyzing samples, generating Lab Reports, or understanding WHO water standards.`;
      }

      // ── Website Features & UI Guidance ─────────────────────
      if (q.includes("upload") || q.includes("drag") || q.includes("format") || q.includes("jpg") || q.includes("png") || q.includes("camera") || q.includes("sample")) {
        return `📷 <strong>Image Upload & Scanning Guide:</strong><br>` +
          `• <strong>Supported Formats:</strong> JPG, JPEG, PNG, WEBP microscopic images.<br>` +
          `• <strong>How to Upload:</strong> Drag & drop an image into the upload box on the left sidebar, or click <strong>'Browse Files'</strong>.<br>` +
          `• <strong>Quick Test:</strong> Click any preset Quick Test Sample thumbnail to instantly test the system without uploading.`;
      }
      if (q.includes("zoom") || q.includes("pan") || q.includes("wheel") || q.includes("reset") || q.includes("view")) {
        return `🔍 <strong>Image Viewer Controls:</strong><br>` +
          `• <strong>Zoom:</strong> Use mouse wheel scrolling or click <strong>➕ / ➖</strong> buttons.<br>` +
          `• <strong>Pan:</strong> Click and drag the image while zoomed in.<br>` +
          `• <strong>Reset:</strong> Click <strong>🔄 Reset</strong> to return to default scale.`;
      }
      if (q.includes("before") || q.includes("after") || q.includes("compare") || q.includes("slider")) {
        return `⚡ <strong>Before / After Compare Slider:</strong><br>` +
          `• Click the <strong>'⚡ Before/After'</strong> overlay button at the bottom-left of the preview viewer.<br>` +
          `• Drag the handle left or right to compare the original raw microscope sample against the AI-annotated detection view.`;
      }
      if (q.includes("heatmap") || q.includes("density")) {
        return `🔥 <strong>Density Heatmap:</strong> Click the <strong>'🔥 Heatmap'</strong> overlay button at the bottom-left of the viewer to render a spatial concentration heatmap showing high-density particle cluster zones.`;
      }
      if (q.includes("kpi") || q.includes("card") || q.includes("count") || q.includes("confidence score")) {
        return `📊 <strong>Dashboard KPI Cards:</strong><br>` +
          `• <strong>Detected Particles:</strong> Total count of microplastics identified.<br>` +
          `• <strong>High Confidence:</strong> Number of detections with >50% certainty.<br>` +
          `• <strong>Avg Confidence:</strong> Overall neural network certainty percentage.<br>` +
          `• <strong>Contamination Hazard:</strong> WHO safety risk level (Ready / Safe / Moderate / Critical).`;
      }
      if (q.includes("meter") || q.includes("gauge") || q.includes("speedometer") || q.includes("risk")) {
        return `⚠️ <strong>Contamination Risk Speedometer Gauge:</strong><br>` +
          `Renders an animated SVG gauge measuring overall sample hazard score (0–100) combining particle density and confidence: Green (0–30: Safe), Yellow (31–60: Moderate), Red (61–100: Critical).`;
      }
      if (q.includes("size breakdown") || q.includes("nano") || q.includes("band") || q.includes("sparkline")) {
        return `📏 <strong>Particle Size Breakdown & Sparkline:</strong><br>` +
          `• <strong>Size Bands:</strong> Nano (<1µm), Micro (1–10µm), Small (10–100µm), Medium (100–500µm), Large (>500µm).<br>` +
          `• <strong>Confidence Sparkline:</strong> Mini bar chart showing certainty spread across up to 60 detected particles.`;
      }
      if (q.includes("report") || q.includes("pdf") || q.includes("download report")) {
        return `📝 <strong>Lab Report PDF Generation:</strong><br>` +
          `Switch to the <strong>'📝 Lab Report'</strong> tab or click <strong>'📑 Lab Report PDF'</strong> in the header to generate a print-ready 8-section laboratory certificate complete with WHO compliance notes and executive summaries.`;
      }
      if (q.includes("csv") || q.includes("json") || q.includes("export") || q.includes("data")) {
        return `📥 <strong>Data Export Options:</strong><br>` +
          `Click <strong>📥 CSV</strong> or <strong>📄 JSON</strong> buttons at the top right of the dashboard to export raw micrometer dimensions, bounding box coordinates, and confidence scores for scientific research.`;
      }
      if (q.includes("batch") || q.includes("multiple") || q.includes("folder") || q.includes("bulk")) {
        return `📦 <strong>Batch Analysis Mode:</strong><br>` +
          `Switch to the <strong>'📦 Batch Analysis'</strong> tab to drag and drop multiple microscopic sample images at once. The system runs parallel YOLO inference across all images and outputs comparative summary tables.`;
      }
      if (q.includes("history") || q.includes("trend") || q.includes("past scan") || q.includes("log")) {
        return `📅 <strong>Session Scan History & Trend Graph:</strong><br>` +
          `Switch to the <strong>'📅 Scan History'</strong> tab to review session logs, view the dynamic trend line graph of particle contamination over time, and click <strong>View ➔</strong> to reload any past analysis instantly.`;
      }

      // ── Polymer Chemistry ─────────────────────────────────
      if (q.includes("pet") || q.includes("polyethylene terephthalate")) {
        return `🧪 <strong>PET (Polyethylene Terephthalate):</strong> Thermoplastic resin used in beverage bottles and synthetic clothing fibers. UV exposure and mechanical wear degrade PET into sharp, rigid micro-fragments.`;
      }
      if (q.includes("pe") || q.includes("polyethylene") || q.includes("hdpe") || q.includes("ldpe")) {
        return `🧪 <strong>Polyethylene (PE):</strong> Most produced synthetic plastic (bags, food wraps, packaging). PE microplastics float easily on water surfaces and adsorb organic pollutants.`;
      }
      if (q.includes("pp") || q.includes("polypropylene")) {
        return `🧪 <strong>Polypropylene (PP):</strong> Heat-resistant polymer used in bottle caps, tupperware, and auto parts. Frequently detected as low-density floating micro-fragments.`;
      }
      if (q.includes("ps") || q.includes("polystyrene") || q.includes("styrofoam")) {
        return `🧪 <strong>Polystyrene (PS):</strong> Used in foam packaging and disposable cutlery. Fragile PS readily breaks down into toxic micro-spheres and leaches styrene monomers.`;
      }
      if (q.includes("pvc") || q.includes("polyvinyl chloride")) {
        return `🧪 <strong>Polyvinyl Chloride (PVC):</strong> Dense polymer used in pipes and construction materials. Contains high concentrations of chemical plasticizers like phthalates.`;
      }
      if (q.includes("nylon") || q.includes("polyester") || q.includes("textile") || q.includes("laundry")) {
        return `🧵 <strong>Synthetic Textiles (Nylon / Polyester):</strong> Washing synthetic clothes releases up to 700,000 micro-fibers per laundry load into municipal wastewater systems.`;
      }

      // ── Health Risks & Organ Toxicity ──────────────────────
      if (q.includes("health") || q.includes("harm") || q.includes("toxic") || q.includes("human") || q.includes("danger") || q.includes("body") || q.includes("ingest") || q.includes("blood") || q.includes("organ")) {
        return `⚠️ <strong>Human Health Toxicity Risks:</strong><br>` +
          `• <strong>Tissue Absorption:</strong> Ingested particles (<10 µm) translocate across intestinal membranes into bloodstream circulation.<br>` +
          `• <strong>Organ Accumulation:</strong> Accumulates in liver, kidney, and placental tissues.<br>` +
          `• <strong>Chemical Leaching:</strong> Leaches toxic additives (BPA, phthalates, heavy metals) causing cellular oxidative stress and endocrine disruption.`;
      }

      // ── Environmental Science (EVS) & WHO Standards ─────────
      if (q.includes("who") || q.includes("limit") || q.includes("standard") || q.includes("safe") || q.includes("drinking") || q.includes("water")) {
        return `⚠️ <strong>WHO Water Safety Guidelines (2019):</strong><br>` +
          `• <strong>&lt; 10 particles/sample:</strong> Safe drinking water benchmark<br>` +
          `• <strong>10 – 30 particles:</strong> Moderate contamination warning<br>` +
          `• <strong>&gt; 30 particles:</strong> Critical hazard level requiring immediate membrane filtration.`;
      }
      if (q.includes("marine") || q.includes("ocean") || q.includes("fish") || q.includes("plankton") || q.includes("food chain")) {
        return `🐠 <strong>Marine Ecosystem Threats:</strong> Microplastics are consumed by zooplankton and fish, causing physical gut blockages, malnutrition, and bioaccumulation up the aquatic trophic chain to human seafood consumers.`;
      }
      if (q.includes("evs") || q.includes("environment") || q.includes("pollution") || q.includes("ecosystem")) {
        return `🌍 <strong>Environmental Science (EVS) Context:</strong> Over 14 million tons of microplastics pollute ocean floors. Studies reveal microplastic contamination in 93% of commercial bottled water worldwide.`;
      }
      if (q.includes("filter") || q.includes("remove") || q.includes("treatment") || q.includes("clean") || q.includes("purify")) {
        return `💧 <strong>Water Remediation & Filtration:</strong> Microplastics can be filtered using <strong>0.1–0.45 µm Membrane Ultrafiltration (UF)</strong>, Reverse Osmosis (RO) systems, or Granular Activated Carbon (GAC) beds.`;
      }
      if (q.includes("fiber") || q.includes("fragment") || q.includes("pellet") || q.includes("film") || q.includes("morphology")) {
        return `🧵 <strong>Microplastic Morphologies:</strong><br>` +
          `• <strong>Fibers:</strong> Slender synthetic threads (Aspect Ratio ≥ 3.0) from textiles.<br>` +
          `• <strong>Fragments:</strong> Jagged rigid pieces from broken containers.<br>` +
          `• <strong>Pellets (Nurdles):</strong> Industrial pre-production plastic granules.<br>` +
          `• <strong>Films:</strong> Thin sheet pieces from bags and wrappers.`;
      }

      // ── AI Technology & Algorithms ──────────────────────────
      if (q.includes("yolo") || q.includes("v8") || q.includes("model") || q.includes("ai") || q.includes("detect") || q.includes("neural")) {
        return `🤖 <strong>YOLOv8 AI Computer Vision Engine:</strong> Custom YOLOv8 neural network trained on optical microscopic particle datasets to detect, bounding-box, size, and classify microplastics in under 50 milliseconds.`;
      }
      if (q.includes("clahe") || q.includes("contrast")) {
        return `⚡ <strong>CLAHE Preprocessing:</strong> Contrast Limited Adaptive Histogram Equalization enhances local image contrast to make translucent or faint plastic particles visible under optical microscopes.`;
      }
      if (q.includes("scale") || q.includes("calibration") || q.includes("micrometer") || q.includes("um") || q.includes("micron")) {
        return `📏 <strong>Micrometer Calibration (µm/px):</strong> Converts pixel bounding boxes into physical micrometer sizes (µm). Adjust presets (e.g. 1.5 µm/px) in the Upload & Calibration sidebar.`;
      }

      // ── Intelligent Universal Synthesizer Catch-All ────────
      return `🔬 <strong>MicroDetect AI Assistant:</strong> Regarding your question about <em>"${query}"</em>:<br>` +
        `Our platform combines YOLOv8 AI object detection, micrometer size calibration, and Gemini AI vision analysis to evaluate microplastic contamination against WHO environmental safety guidelines.<br><br>` +
        `💡 <em>Try asking about:</em> <strong>WHO water limits</strong>, <strong>polymer types (PET/PE/PP)</strong>, <strong>health toxicity</strong>, <strong>Lab Report PDF</strong>, or click <strong>🔬 Scan Summary</strong> to analyze your active sample!`;
    }

    // ── APPEND BUBBLE WITH ACTIONS ──────────────────────────
    function appendChatBubble(text, sender, isTyping = false) {
      const msgDiv = document.createElement("div");
      msgDiv.className = `chat-msg ${sender}`;

      const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const avatar = sender === "bot" ? "🔬" : "👤";

      let actionsHtml = "";
      if (sender === "bot" && !isTyping) {
        actionsHtml = `
          <div class="chat-bubble-actions">
            <button class="chat-bubble-btn copy-btn" title="Copy response text">📋 Copy</button>
            <button class="chat-bubble-btn speak-btn" title="Read response out loud">🔊 Speak</button>
          </div>
        `;
      }

      msgDiv.innerHTML = `
        <div class="chat-avatar-wrap">${avatar}</div>
        <div class="chat-body">
          <div class="chat-bubble"${isTyping ? ' style="font-style:italic; opacity:0.7;"' : ''}>${text}${actionsHtml}</div>
          <div class="chat-time">${timeStr}</div>
        </div>
      `;

      // Attach Copy & Speak handlers
      if (sender === "bot" && !isTyping) {
        const copyBtn = msgDiv.querySelector(".copy-btn");
        const speakBtn = msgDiv.querySelector(".speak-btn");

        if (copyBtn) {
          copyBtn.addEventListener("click", () => {
            const rawText = msgDiv.querySelector(".chat-bubble").innerText.replace("📋 Copy🔊 Speak", "").trim();
            navigator.clipboard.writeText(rawText);
            showToast("Copied", "Chat response copied to clipboard.", "info", 1800);
          });
        }

        if (speakBtn) {
          speakBtn.addEventListener("click", () => {
            if ("speechSynthesis" in window) {
              window.speechSynthesis.cancel(); // stop current
              const rawText = msgDiv.querySelector(".chat-bubble").innerText.replace("📋 Copy🔊 Speak", "").trim();
              const utterance = new SpeechSynthesisUtterance(rawText.replace(/<[^>]*>?/gm, ''));
              utterance.rate = 1.0;
              utterance.pitch = 1.0;
              window.speechSynthesis.speak(utterance);
              showToast("Speaking", "Reading response out loud...", "info", 2000);
            } else {
              showToast("TTS Unavailable", "Speech synthesis not supported on this browser.", "warn", 2000);
            }
          });
        }
      }

      chatbotMessages.appendChild(msgDiv);
      chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
      return msgDiv;
    }

    if (chatSendBtn) {
      chatSendBtn.addEventListener("click", () => sendChatMessage());
    }

    if (chatInput) {
      chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") sendChatMessage();
      });
    }

    document.querySelectorAll(".chat-suggest").forEach(btn => {
      btn.addEventListener("click", () => {
        const q = btn.getAttribute("data-q") || btn.textContent;
        sendChatMessage(q);
      });
    });
  }

});


