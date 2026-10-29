/**
 * NapoVision Client-side Application
 * Biomechanical Sleep Posture Monitoring & Clinical Ergonomics Engine
 */

document.addEventListener('DOMContentLoaded', () => {
  // Application State
  const state = {
    selectedPreset: 'prone',
    currentImageBlob: null,
    currentImageUrl: null,
    currentFilename: 'prone.jpg',
    currentFilesize: 'Sample Reference Image',
    activeViewMode: 'blended',
    groqApiKey: localStorage.getItem('napovision_groq_key') || '',
    latestDiagnosticData: null
  };

  // DOM Elements
  const dropzone = document.getElementById('dropzone');
  const dropzoneUnloaded = document.getElementById('dropzoneUnloaded');
  const dropzoneLoaded = document.getElementById('dropzoneLoaded');
  const fileInput = document.getElementById('fileInput');
  const browseFileBtn = document.getElementById('browseFileBtn');
  const previewImage = document.getElementById('previewImage');
  const previewFilename = document.getElementById('previewFilename');
  const previewFilesize = document.getElementById('previewFilesize');
  const reselectBtn = document.getElementById('reselectBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const diagnosticLoader = document.getElementById('diagnosticLoader');

  // Diagnostic UI Elements
  const diagClassBadge = document.getElementById('diagClassBadge');
  const diagConfidenceVal = document.getElementById('diagConfidenceVal');
  const probValSupine = document.getElementById('probValSupine');
  const probValSide = document.getElementById('probValSide');
  const probValFetal = document.getElementById('probValFetal');
  const probValProne = document.getElementById('probValProne');
  const probBarSupine = document.getElementById('probBarSupine');
  const probBarSide = document.getElementById('probBarSide');
  const probBarFetal = document.getElementById('probBarFetal');
  const probBarProne = document.getElementById('probBarProne');

  // Risk Gauge Elements
  const riskScoreNumber = document.getElementById('riskScoreNumber');
  const riskLevelBadge = document.getElementById('riskLevelBadge');
  const gaugeProgressCircle = document.getElementById('gaugeProgressCircle');
  const riskStatusSummary = document.getElementById('riskStatusSummary');
  const anatCervicalVal = document.getElementById('anatCervicalVal');
  const anatCervicalBar = document.getElementById('anatCervicalBar');
  const anatThoracicVal = document.getElementById('anatThoracicVal');
  const anatThoracicBar = document.getElementById('anatThoracicBar');
  const anatLumbarVal = document.getElementById('anatLumbarVal');
  const anatLumbarBar = document.getElementById('anatLumbarBar');
  const anatPelvicVal = document.getElementById('anatPelvicVal');
  const anatPelvicBar = document.getElementById('anatPelvicBar');

  // Explainable AI Elements
  const xaiBaseImage = document.getElementById('xaiBaseImage');
  const xaiHeatmapLayer = document.getElementById('xaiHeatmapLayer');
  const xaiBlendedView = document.getElementById('xaiBlendedView');
  const xaiSplitView = document.getElementById('xaiSplitView');
  const xaiSplitOriginal = document.getElementById('xaiSplitOriginal');
  const xaiSplitHeatmap = document.getElementById('xaiSplitHeatmap');
  const opacityRange = document.getElementById('opacityRange');
  const opacitySliderVal = document.getElementById('opacitySliderVal');
  const opacitySliderGroup = document.getElementById('opacitySliderGroup');

  // Clinical Advice Elements
  const adviceEngineSourceBadge = document.getElementById('adviceEngineSourceBadge');
  const headerEngineSource = document.getElementById('headerEngineSource');
  const adviceClinicalSummary = document.getElementById('adviceClinicalSummary');
  const advicePathomechanics = document.getElementById('advicePathomechanics');
  const adviceCervical = document.getElementById('adviceCervical');
  const adviceLumbar = document.getElementById('adviceLumbar');
  const adviceEquipment = document.getElementById('adviceEquipment');
  const exercisesGrid = document.getElementById('exercisesGrid');
  const advicePrognosis = document.getElementById('advicePrognosis');
  const copyAdviceBtn = document.getElementById('copyAdviceBtn');
  const printAdviceBtn = document.getElementById('printAdviceBtn');

  // API Modal Elements
  const openApiKeyModalBtn = document.getElementById('openApiKeyModalBtn');
  const closeApiKeyModalBtn = document.getElementById('closeApiKeyModalBtn');
  const apiKeyModal = document.getElementById('apiKeyModal');
  const groqApiKeyInput = document.getElementById('groqApiKeyInput');
  const saveKeyBtn = document.getElementById('saveKeyBtn');
  const clearKeyBtn = document.getElementById('clearKeyBtn');
  const keyStatusText = document.getElementById('keyStatusText');
  const apiKeyBtnText = document.getElementById('apiKeyBtnText');

  // Circumference for Gauge: 2 * pi * 68 = 427.26
  const GAUGE_CIRCUMFERENCE = 427.26;

  // Initialize UI & Settings
  function init() {
    updateApiKeyStatus();
    setupEventListeners();
    setupPresetClickHandlers();
    
    // Auto-analyze default preset on startup
    loadPreset('prone', true);
  }

  // Update API Key Display State
  function updateApiKeyStatus() {
    if (state.groqApiKey && state.groqApiKey.startsWith('gsk_')) {
      apiKeyBtnText.textContent = 'Groq: Active';
      headerEngineSource.textContent = 'Groq Llama-3.3-70B';
      keyStatusText.textContent = `Custom Key Active (${state.groqApiKey.substring(0, 8)}...)`;
      if (groqApiKeyInput) groqApiKeyInput.value = state.groqApiKey;
    } else {
      apiKeyBtnText.textContent = 'Groq Key';
      headerEngineSource.textContent = 'Llama-3.3-70B / Clinical Engine';
      keyStatusText.textContent = 'Using NapoVision Clinical Ergonomics Engine';
      if (groqApiKeyInput) groqApiKeyInput.value = '';
    }
  }

  // Event Listeners
  function setupEventListeners() {
    // Browse button trigger
    browseFileBtn.addEventListener('click', () => fileInput.click());
    
    // File input change
    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) handleSelectedFile(file);
    });

    // Drag and drop handling
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) {
        handleSelectedFile(file);
      }
    });

    // Reselect action
    reselectBtn.addEventListener('click', () => {
      fileInput.click();
    });

    // Analyze action
    analyzeBtn.addEventListener('click', () => {
      runDiagnosis();
    });

    // XAI View Mode switcher
    document.querySelectorAll('.view-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.view-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        state.activeViewMode = tab.dataset.view;
        applyViewMode(state.activeViewMode);
      });
    });

    // Opacity Range Slider
    opacityRange.addEventListener('input', (e) => {
      const val = e.target.value;
      opacitySliderVal.textContent = `${val}%`;
      xaiHeatmapLayer.style.opacity = (val / 100).toString();
    });

    // Advice Tab Switcher
    document.querySelectorAll('.advice-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.advice-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        
        tab.classList.add('active');
        const targetPanel = document.getElementById(`tab${tab.dataset.tab.charAt(0).toUpperCase() + tab.dataset.tab.slice(1)}`);
        if (targetPanel) targetPanel.classList.add('active');
      });
    });

    // Copy advice button
    copyAdviceBtn.addEventListener('click', copyClinicalAdvice);

    // Print button
    printAdviceBtn.addEventListener('click', () => window.print());

    // API Key Modal
    openApiKeyModalBtn.addEventListener('click', () => {
      apiKeyModal.style.display = 'flex';
    });

    closeApiKeyModalBtn.addEventListener('click', () => {
      apiKeyModal.style.display = 'none';
    });

    apiKeyModal.addEventListener('click', (e) => {
      if (e.target === apiKeyModal) apiKeyModal.style.display = 'none';
    });

    saveKeyBtn.addEventListener('click', () => {
      const key = groqApiKeyInput.value.trim();
      state.groqApiKey = key;
      localStorage.setItem('napovision_groq_key', key);
      updateApiKeyStatus();
      apiKeyModal.style.display = 'none';
    });

    clearKeyBtn.addEventListener('click', () => {
      state.groqApiKey = '';
      localStorage.removeItem('napovision_groq_key');
      updateApiKeyStatus();
      apiKeyModal.style.display = 'none';
    });
  }

  // Handle Preset Clicks
  function setupPresetClickHandlers() {
    document.querySelectorAll('.preset-card').forEach(btn => {
      btn.addEventListener('click', () => {
        const presetKey = btn.dataset.preset;
        loadPreset(presetKey, true);
      });
    });
  }

  // Load a preset sleep posture image
  async function loadPreset(presetKey, autoAnalyze = false) {
    document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active-preset'));
    const activeBtn = document.querySelector(`.preset-card[data-preset="${presetKey}"]`);
    if (activeBtn) activeBtn.classList.add('active-preset');

    state.selectedPreset = presetKey;
    const url = `/static/presets/${presetKey}.jpg`;
    state.currentImageUrl = url;
    state.currentFilename = `${presetKey}_posture.jpg`;
    state.currentFilesize = 'Standardized Anchor Sample';

    // Fetch as blob for consistent transmission
    try {
      const resp = await fetch(url);
      state.currentImageBlob = await resp.blob();

      // Show preview in dropzone
      previewImage.src = url;
      previewFilename.textContent = state.currentFilename;
      previewFilesize.textContent = state.currentFilesize;
      dropzoneUnloaded.style.display = 'none';
      dropzoneLoaded.style.display = 'flex';

      // Update XAI original image stage immediately
      xaiBaseImage.src = url;
      xaiSplitOriginal.src = url;

      if (autoAnalyze) {
        runDiagnosis();
      }
    } catch (err) {
      console.error('Failed to load preset:', err);
    }
  }

  // Handle User File Upload
  function handleSelectedFile(file) {
    document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active-preset'));
    state.selectedPreset = null;
    state.currentImageBlob = file;
    state.currentFilename = file.name;
    state.currentFilesize = formatBytes(file.size);

    const objectUrl = URL.createObjectURL(file);
    state.currentImageUrl = objectUrl;

    previewImage.src = objectUrl;
    previewFilename.textContent = state.currentFilename;
    previewFilesize.textContent = state.currentFilesize;
    dropzoneUnloaded.style.display = 'none';
    dropzoneLoaded.style.display = 'flex';

    xaiBaseImage.src = objectUrl;
    xaiSplitOriginal.src = objectUrl;

    // Trigger analysis
    runDiagnosis();
  }

  // Main Diagnosis Async Function
  async function runDiagnosis() {
    if (!state.currentImageBlob) return;

    showLoader(true);
    simulateProgressSteps();

    try {
      const formData = new FormData();
      formData.append('image', state.currentImageBlob, state.currentFilename);

      const headers = {};
      if (state.groqApiKey) {
        headers['X-Groq-Api-Key'] = state.groqApiKey;
      }

      const response = await fetch('/predict', {
        method: 'POST',
        headers: headers,
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();
      state.latestDiagnosticData = data;
      renderDiagnosticResults(data);

    } catch (error) {
      console.error('Diagnosis failed:', error);
      alert(`Posture analysis error: ${error.message}. Please check your connection and try again.`);
    } finally {
      showLoader(false);
    }
  }

  // Simulate Progressive Diagnostic Pipeline Steps
  function simulateProgressSteps() {
    const steps = [
      document.getElementById('step1'),
      document.getElementById('step2'),
      document.getElementById('step3'),
      document.getElementById('step4')
    ];

    steps.forEach((s, idx) => {
      s.classList.remove('step-active');
      if (idx === 0) s.classList.add('step-active');
    });

    setTimeout(() => {
      if (steps[1]) steps[1].classList.add('step-active');
    }, 450);

    setTimeout(() => {
      if (steps[2]) steps[2].classList.add('step-active');
    }, 900);

    setTimeout(() => {
      if (steps[3]) steps[3].classList.add('step-active');
    }, 1400);
  }

  function showLoader(isLoading) {
    if (isLoading) {
      diagnosticLoader.style.display = 'flex';
      analyzeBtn.disabled = true;
    } else {
      diagnosticLoader.style.display = 'none';
      analyzeBtn.disabled = false;
    }
  }

  // Render Full Diagnostic Results into UI
  function renderDiagnosticResults(data) {
    const pred = data.prediction;
    const risk = data.risk_assessment;
    const xai = data.explainable_ai;
    const advice = data.clinical_advice;

    // 1. Column 1: Primary Diagnosis
    diagClassBadge.textContent = pred.display_name;
    diagConfidenceVal.textContent = `${pred.confidence_percent}%`;

    const probs = pred.all_probabilities;
    updateProbBar('Supine', probs['Supine/Back'] || 0, probValSupine, probBarSupine);
    updateProbBar('Side', probs['Side'] || 0, probValSide, probBarSide);
    updateProbBar('Fetal', probs['Fetal'] || 0, probValFetal, probBarFetal);
    updateProbBar('Prone', probs['Prone/Stomach'] || 0, probValProne, probBarProne);

    // 2. Column 2: Biomechanical Risk Meter
    animateRiskGauge(risk.score, risk.level, risk.color);
    riskStatusSummary.textContent = risk.status_text;

    // Anatomical Multi-Zone Breakdown
    const anat = risk.breakdown;
    anatCervicalVal.textContent = `${anat.cervical}/100`;
    anatCervicalBar.style.width = `${anat.cervical}%`;

    anatThoracicVal.textContent = `${anat.thoracic}/100`;
    anatThoracicBar.style.width = `${anat.thoracic}%`;

    anatLumbarVal.textContent = `${anat.lumbar}/100`;
    anatLumbarBar.style.width = `${anat.lumbar}%`;

    anatPelvicVal.textContent = `${anat.pelvic}/100`;
    anatPelvicBar.style.width = `${anat.pelvic}%`;

    // 3. Column 3: Explainable AI Heatmaps
    if (xai.gradcam_heatmap) {
      xaiHeatmapLayer.src = xai.gradcam_heatmap;
      xaiSplitHeatmap.src = xai.gradcam_heatmap;
    }

    // 4. Clinical Ergonomic Prescription Dossier
    adviceEngineSourceBadge.textContent = advice.engine_source || 'NapoVision Clinical Engine';
    adviceClinicalSummary.textContent = advice.clinical_summary || 'Evaluation summary generated.';
    advicePathomechanics.textContent = advice.pathomechanics || 'Pathomechanics details analyzed.';
    adviceCervical.textContent = advice.cervical_prescription || 'Cervical alignment protocol ready.';
    adviceLumbar.textContent = advice.lumbar_prescription || 'Lumbar decompression protocol ready.';
    adviceEquipment.textContent = advice.ergonomic_equipment || 'Sleep equipment specifications ready.';
    advicePrognosis.textContent = advice.long_term_prognosis || 'Long-term prognostic evaluation complete.';

    // Populate Exercises Cards Grid
    exercisesGrid.innerHTML = '';
    if (advice.corrective_exercises && Array.isArray(advice.corrective_exercises)) {
      advice.corrective_exercises.forEach((ex, idx) => {
        const card = document.createElement('div');
        card.className = 'exercise-item-card';
        card.innerHTML = `
          <span class="ex-badge">Protocol #${idx + 1}</span>
          <h5 class="ex-title">${escapeHtml(ex.name)}</h5>
          <span class="ex-dosage">${escapeHtml(ex.dosage)}</span>
          <p class="ex-desc">${escapeHtml(ex.instruction)}</p>
        `;
        exercisesGrid.appendChild(card);
      });
    }
  }

  // Update Probability Progress Row
  function updateProbBar(name, val, textElem, barElem) {
    const pct = (val * 100).toFixed(1);
    textElem.textContent = `${pct}%`;
    barElem.style.width = `${pct}%`;
  }

  // Animate Circular Risk Gauge
  function animateRiskGauge(score, level, color) {
    riskScoreNumber.textContent = score;
    riskLevelBadge.textContent = level;

    // Stroke Dash Offset Formula
    const offset = GAUGE_CIRCUMFERENCE * (1 - score / 100);
    gaugeProgressCircle.style.strokeDashoffset = offset.toString();
    gaugeProgressCircle.style.stroke = color;

    // Badge colors
    if (score >= 65) {
      riskLevelBadge.className = 'gauge-status-badge high-tag';
    } else if (score >= 45) {
      riskLevelBadge.className = 'gauge-status-badge mod-tag';
    } else if (score >= 25) {
      riskLevelBadge.className = 'gauge-status-badge mild-tag';
    } else {
      riskLevelBadge.className = 'gauge-status-badge low-tag';
    }
  }

  // Switch XAI View Modes
  function applyViewMode(mode) {
    if (mode === 'blended') {
      xaiBlendedView.style.display = 'block';
      xaiSplitView.style.display = 'none';
      xaiBaseImage.style.display = 'block';
      xaiHeatmapLayer.style.display = 'block';
      opacitySliderGroup.style.display = 'flex';
      xaiHeatmapLayer.style.opacity = (opacityRange.value / 100).toString();
    } else if (mode === 'sidebyside') {
      xaiBlendedView.style.display = 'none';
      xaiSplitView.style.display = 'grid';
      opacitySliderGroup.style.display = 'none';
    } else if (mode === 'heatmap') {
      xaiBlendedView.style.display = 'block';
      xaiSplitView.style.display = 'none';
      xaiBaseImage.style.display = 'none';
      xaiHeatmapLayer.style.display = 'block';
      xaiHeatmapLayer.style.opacity = '1';
      opacitySliderGroup.style.display = 'none';
    }
  }

  // Copy Full Clinical Advice to Clipboard
  function copyClinicalAdvice() {
    if (!state.latestDiagnosticData) return;

    const advice = state.latestDiagnosticData.clinical_advice;
    const pred = state.latestDiagnosticData.prediction;
    const risk = state.latestDiagnosticData.risk_assessment;

    let text = `NAPOVISION CLINICAL ERGONOMIC REPORT\n`;
    text += `====================================\n`;
    text += `Diagnosis: ${pred.display_name} (Confidence: ${pred.confidence_percent}%)\n`;
    text += `Biomechanical Strain Risk: ${risk.score}/100 (${risk.level})\n`;
    text += `Primary Stress: ${risk.status_text}\n\n`;
    text += `EXECUTIVE CLINICAL ASSESSMENT:\n${advice.clinical_summary}\n\n`;
    text += `BIOMECHANICAL PATHOMECHANICS:\n${advice.pathomechanics}\n\n`;
    text += `CERVICAL SPINE PRESCRIPTION:\n${advice.cervical_prescription}\n\n`;
    text += `LUMBAR & PELVIC DECOMPRESSION:\n${advice.lumbar_prescription}\n\n`;
    text += `SLEEP ERGONOMICS & MATTRESS SPECS:\n${advice.ergonomic_equipment}\n\n`;
    text += `PRESCRIBED REHABILITATION STRETCHES:\n`;
    if (advice.corrective_exercises) {
      advice.corrective_exercises.forEach((ex, i) => {
        text += `${i + 1}. ${ex.name} (${ex.dosage}): ${ex.instruction}\n`;
      });
    }
    text += `\nLONG-TERM PROGNOSIS:\n${advice.long_term_prognosis}\n`;
    text += `\nGenerated via: ${advice.engine_source || 'NapoVision AI'}\n`;

    navigator.clipboard.writeText(text).then(() => {
      const origText = copyAdviceBtn.querySelector('span').textContent;
      copyAdviceBtn.querySelector('span').textContent = 'Copied to Clipboard!';
      setTimeout(() => {
        copyAdviceBtn.querySelector('span').textContent = origText;
      }, 2000);
    }).catch(err => {
      console.error('Clipboard copy failed:', err);
    });
  }

  // Utility: Format File Size
  function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  // Utility: HTML Escaping
  function escapeHtml(text) {
    if (!text) return '';
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Boot Application
  init();
});
