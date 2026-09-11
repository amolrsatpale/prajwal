/**
 * PhishNet – Frontend Application Logic
 * Handles URL scanning, results display, history, and UI interactions.
 */

// ──────────────────────────── DOM Elements ────────────────────────────
const els = {
  form: document.getElementById('scan-form'),
  urlInput: document.getElementById('url-input'),
  scanBtn: document.getElementById('scan-btn'),
  scanProgress: document.getElementById('scan-progress'),
  resultsSection: document.getElementById('results-section'),
  verdictCard: document.getElementById('verdict-card'),
  ringFill: document.getElementById('ring-fill'),
  verdictScore: document.getElementById('verdict-score'),
  verdictLabel: document.getElementById('verdict-label'),
  verdictUrl: document.getElementById('verdict-url'),
  riskLevel: document.getElementById('risk-level'),
  confidence: document.getElementById('confidence'),
  modelType: document.getElementById('model-type'),
  featureBars: document.getElementById('feature-bars'),
  anatomyDisplay: document.getElementById('anatomy-display'),
  featureGrid: document.getElementById('feature-grid'),
  historyList: document.getElementById('history-list'),
  clearHistory: document.getElementById('clear-history'),
  navStatus: document.getElementById('nav-status'),
  scannerMode: document.getElementById('scanner-mode'),
};

// ──────────────────────────── State ────────────────────────────
const RING_CIRCUMFERENCE = 2 * Math.PI * 54; // r=54 from SVG
let scanHistory = JSON.parse(localStorage.getItem('phishnet_history') || '[]');
let isScanning = false;

// ──────────────────────────── Init ────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  checkServerStatus();
  renderHistory();
  setupEventListeners();
});

// ──────────────────────────── Event Listeners ────────────────────────────
function setupEventListeners() {
  els.form.addEventListener('submit', (e) => {
    e.preventDefault();
    const url = els.urlInput.value.trim();
    if (url && !isScanning) {
      analyzeURL(url);
    }
  });

  // Quick test buttons
  document.querySelectorAll('.quick-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const url = btn.dataset.url;
      els.urlInput.value = url;
      analyzeURL(url);
    });
  });

  // Clear history
  els.clearHistory.addEventListener('click', () => {
    scanHistory = [];
    localStorage.removeItem('phishnet_history');
    renderHistory();
  });

  // History item click → re-scan
  els.historyList.addEventListener('click', (e) => {
    const item = e.target.closest('.history-item');
    if (item) {
      const url = item.dataset.url;
      els.urlInput.value = url;
      analyzeURL(url);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  });

  // Navbar scroll
  window.addEventListener('scroll', () => {
    document.getElementById('navbar').classList.toggle('scrolled', window.scrollY > 20);
  });
}

// ──────────────────────────── Server Status ────────────────────────────
async function checkServerStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();

    const dot = els.navStatus.querySelector('.status-dot');
    const text = els.navStatus.querySelector('.status-text');

    dot.classList.add('online');
    if (data.model_type === 'deep_learning') {
      text.textContent = 'DL Model Active';
      els.scannerMode.textContent = 'Deep Learning Mode';
      els.scannerMode.classList.add('dl-mode');
    } else if (data.model_type === 'random_forest') {
      text.textContent = 'RF Model Active';
      els.scannerMode.textContent = 'Random Forest Mode';
      els.scannerMode.classList.add('rf-mode');
    } else {
      text.textContent = 'Heuristic Mode';
      els.scannerMode.textContent = 'Heuristic Mode';
    }
  } catch {
    const text = els.navStatus.querySelector('.status-text');
    text.textContent = 'Server Offline';
  }
}

// ──────────────────────────── URL Analysis ────────────────────────────
async function analyzeURL(url) {
  if (isScanning) return;
  isScanning = true;

  // UI: loading state
  els.scanBtn.classList.add('loading');
  els.scanProgress.classList.add('active');
  els.resultsSection.classList.add('hidden');

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Server error');
    }

    const data = await res.json();

    // Short delay for scan effect
    await delay(600);

    displayResults(data);
    addToHistory(data);

  } catch (err) {
    alert('Analysis failed: ' + err.message);
  } finally {
    isScanning = false;
    els.scanBtn.classList.remove('loading');
    els.scanProgress.classList.remove('active');
  }
}

// ──────────────────────────── Display Results ────────────────────────────
function displayResults(data) {
  // Show section
  els.resultsSection.classList.remove('hidden');

  // Scroll to results
  setTimeout(() => {
    els.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 100);

  const pct = Math.round(data.probability * 100);

  // --- Verdict card ---
  els.verdictCard.setAttribute('data-risk', data.risk_level);
  els.verdictUrl.textContent = truncate(data.url, 50);
  els.verdictLabel.textContent = data.verdict;
  els.riskLevel.textContent = data.risk_level;
  els.riskLevel.style.color = riskColor(data.risk_level);
  els.confidence.textContent = pct + '%';
  els.modelType.textContent = data.using_dl ? 'CNN-BiLSTM' : (data.using_rf ? 'Random Forest' : 'Heuristic');

  // Animate ring
  const offset = RING_CIRCUMFERENCE - (pct / 100) * RING_CIRCUMFERENCE;
  els.ringFill.style.strokeDashoffset = RING_CIRCUMFERENCE; // reset
  requestAnimationFrame(() => {
    els.ringFill.style.strokeDashoffset = offset;
  });

  // Animate score counter
  animateCounter(els.verdictScore, 0, pct, 1000, (v) => v + '%');

  // --- Feature importance bars ---
  els.featureBars.innerHTML = '';
  (data.feature_importance || []).forEach((fi, i) => {
    const impPct = Math.round(fi.impact * 100);
    const levelClass = impPct >= 70 ? 'high' : impPct >= 40 ? 'medium' : '';
    const item = document.createElement('div');
    item.className = 'feature-bar-item';
    item.style.animationDelay = `${i * 0.05}s`;
    item.innerHTML = `
      <div class="feature-bar-label">
        <span class="feature-bar-name">${formatFeatureName(fi.feature)}</span>
        <span class="feature-bar-value">${formatFeatureValue(fi.feature, fi.value)}</span>
      </div>
      <div class="feature-bar-track">
        <div class="feature-bar-fill ${levelClass}" style="width: 0%"></div>
      </div>
    `;
    els.featureBars.appendChild(item);

    // Animate bar fill
    setTimeout(() => {
      item.querySelector('.feature-bar-fill').style.width = impPct + '%';
    }, 100 + i * 60);
  });

  // --- URL Anatomy ---
  renderAnatomy(data.url, data.features);

  // --- Feature grid ---
  els.featureGrid.innerHTML = '';
  const featureEntries = Object.entries(data.features || {});
  featureEntries.forEach(([name, value]) => {
    const chip = document.createElement('div');
    chip.className = 'feature-chip';
    chip.innerHTML = `
      <span class="feature-chip-name">${formatFeatureName(name)}</span>
      <span class="feature-chip-value">${typeof value === 'number' ? (Number.isInteger(value) ? value : value.toFixed(3)) : value}</span>
    `;
    els.featureGrid.appendChild(chip);
  });
}

// ──────────────────────────── URL Anatomy Rendering ────────────────────────────
function renderAnatomy(url, features) {
  try {
    let protocolAdded = false;
    let displayUrl = url;
    if (!url.match(/^https?:\/\//)) {
      displayUrl = 'http://' + url;
      protocolAdded = true;
    }

    const parsed = new URL(displayUrl);
    let html = '';

    // Scheme
    html += `<span class="url-part scheme" title="Protocol">${parsed.protocol}//</span>`;

    // Host
    const host = parsed.host;
    const isIP = /^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/.test(parsed.hostname);
    const hostClass = isIP ? 'suspicious' : 'host';
    html += `<span class="url-part ${hostClass}" title="${isIP ? '⚠ IP Address in URL' : 'Hostname'}">${host}</span>`;

    // Path
    if (parsed.pathname && parsed.pathname !== '/') {
      const pathParts = parsed.pathname.split('/').filter(Boolean);
      const suspiciousKeywords = ['login', 'verify', 'secure', 'account', 'update', 'confirm', 'signin', 'password', 'credential', 'billing'];

      html += '<span class="url-part path" title="Path">/';
      pathParts.forEach((part, i) => {
        const isSuspicious = suspiciousKeywords.some((kw) => part.toLowerCase().includes(kw));
        if (isSuspicious) {
          html += `<span class="url-part suspicious" title="⚠ Suspicious keyword: ${part}">${part}</span>`;
        } else {
          html += part;
        }
        if (i < pathParts.length - 1) html += '/';
      });
      html += '</span>';
    }

    // Query
    if (parsed.search) {
      html += `<span class="url-part query" title="Query Parameters">${parsed.search}</span>`;
    }

    els.anatomyDisplay.innerHTML = html;
  } catch {
    els.anatomyDisplay.textContent = url;
  }
}

// ──────────────────────────── History ────────────────────────────
function addToHistory(data) {
  const entry = {
    url: data.url,
    probability: data.probability,
    risk_level: data.risk_level,
    verdict: data.verdict,
    timestamp: Date.now(),
  };

  // Remove duplicate
  scanHistory = scanHistory.filter((h) => h.url !== data.url);
  scanHistory.unshift(entry);
  scanHistory = scanHistory.slice(0, 20); // keep last 20

  localStorage.setItem('phishnet_history', JSON.stringify(scanHistory));
  renderHistory();
}

function renderHistory() {
  if (scanHistory.length === 0) {
    els.historyList.innerHTML = '<div class="empty-history">No scans yet. Enter a URL above to get started.</div>';
    return;
  }

  els.historyList.innerHTML = '';
  scanHistory.forEach((entry) => {
    const dotClass = entry.risk_level === 'LOW' ? 'safe' : entry.risk_level === 'MEDIUM' ? 'suspicious' : 'phishing';
    const verdictClass = dotClass;
    const item = document.createElement('div');
    item.className = 'history-item';
    item.dataset.url = entry.url;
    item.innerHTML = `
      <div class="history-dot ${dotClass}"></div>
      <div class="history-url">${entry.url}</div>
      <div class="history-verdict ${verdictClass}">${entry.verdict}</div>
      <div class="history-score">${Math.round(entry.probability * 100)}%</div>
    `;
    els.historyList.appendChild(item);
  });
}

// ──────────────────────────── Utilities ────────────────────────────
function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function truncate(str, len) {
  return str.length > len ? str.slice(0, len) + '...' : str;
}

function riskColor(level) {
  switch (level) {
    case 'LOW': return '#34d399';
    case 'MEDIUM': return '#fb923c';
    case 'HIGH': return '#f87171';
    default: return '#94a3b8';
  }
}

function formatFeatureName(name) {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace('Url', 'URL')
    .replace('Ip', 'IP')
    .replace('Https', 'HTTPS')
    .replace('Tld', 'TLD');
}

function formatFeatureValue(name, value) {
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (value === 0 || value === 1) {
    if (name.startsWith('has_') || name.startsWith('is_')) {
      return value === 1 ? '✓ Yes' : '✗ No';
    }
  }
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  }
  return String(value);
}

function animateCounter(element, from, to, duration, formatter) {
  const start = performance.now();
  const step = (now) => {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    // Ease out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(from + (to - from) * eased);
    element.textContent = formatter ? formatter(current) : current;
    if (progress < 1) {
      requestAnimationFrame(step);
    }
  };
  requestAnimationFrame(step);
}
