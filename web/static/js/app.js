/* ─────────────────────────────────────────────────────────────
   FZL Color Matching AI — Frontend Logic
   Handles: API calls, autocomplete, result rendering, copy
───────────────────────────────────────────────────────────── */

const API = '';  // Same origin (Flask serves this)
let allColors   = { tape: [], thread: [] };
let currentSub  = 'tape';

// ── Init ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  await loadColorLists();
  setupSubstrateToggle();
  setupAutocomplete();
  setupEnterKey();
});

// ── Load color lists for both substrates ─────────────────────
async function loadColorLists() {
  try {
    const [tapeRes, threadRes] = await Promise.all([
      fetch(`${API}/api/colors?substrate=tape`),
      fetch(`${API}/api/colors?substrate=thread`),
    ]);
    const tapeData   = await tapeRes.json();
    const threadData = await threadRes.json();

    allColors.tape   = tapeData.colors   || [];
    allColors.thread = threadData.colors || [];

    document.getElementById('tapeCount').innerHTML =
      `<span class="dot tape"></span>Tape: ${allColors.tape.length} colors`;
    document.getElementById('threadCount').innerHTML =
      `<span class="dot thread"></span>Thread: ${allColors.thread.length} colors`;
  } catch (e) {
    console.warn('Could not load color lists:', e);
  }
}

// ── Substrate toggle ─────────────────────────────────────────
function setupSubstrateToggle() {
  document.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentSub = btn.dataset.substrate;
      document.getElementById('colorInput').value = '';
      hideSuggestions();
    });
  });
}

// ── Autocomplete ──────────────────────────────────────────────
function setupAutocomplete() {
  const input = document.getElementById('colorInput');
  const box   = document.getElementById('suggestions');

  input.addEventListener('input', () => {
    const q = input.value.trim().toUpperCase();
    if (!q) { hideSuggestions(); return; }

    const matches = allColors[currentSub]
      .filter(c => c.toUpperCase().includes(q))
      .slice(0, 30);

    if (!matches.length) { hideSuggestions(); return; }

    box.innerHTML = matches.map(c => `
      <div class="suggestion-item" onclick="selectColor('${c.replace(/'/g,"\\'")}')">
        <div class="suggestion-swatch" style="background:${colorNameToApprox(c)}"></div>
        ${c}
      </div>
    `).join('');
    box.classList.add('open');
  });

  // Close on outside click
  document.addEventListener('click', e => {
    if (!e.target.closest('.search-input-wrap')) hideSuggestions();
  });
}

function selectColor(name) {
  document.getElementById('colorInput').value = name;
  hideSuggestions();
  // Clear lab inputs
  ['labL','labA','labB'].forEach(id => document.getElementById(id).value = '');
}

function hideSuggestions() {
  document.getElementById('suggestions').classList.remove('open');
}

// Very rough hue hint for autocomplete swatches (cosmetic only)
function colorNameToApprox(name) {
  const n = name.toUpperCase();
  if (n.includes('BLACK')) return '#2a2a2a';
  if (n.includes('NAVY')) return '#1e3a5f';
  if (n.includes('BLUE')) return '#2563eb';
  if (n.includes('TURQ')) return '#0d9488';
  if (n.includes('GREEN')) return '#16a34a';
  if (n.includes('YELLOW')) return '#ca8a04';
  if (n.includes('ORANGE')) return '#ea580c';
  if (n.includes('RED')) return '#dc2626';
  if (n.includes('VIOLET') || n.includes('PURPLE')) return '#7c3aed';
  if (n.includes('WHITE')) return '#f0f0f0';
  return '#6b7280';
}

// ── Enter key shortcut ────────────────────────────────────────
function setupEnterKey() {
  ['colorInput','labL','labA','labB'].forEach(id => {
    document.getElementById(id).addEventListener('keydown', e => {
      if (e.key === 'Enter') runPrediction();
    });
  });
}

// ── Main Prediction ───────────────────────────────────────────
async function runPrediction() {
  const colorName = document.getElementById('colorInput').value.trim();
  const L = parseFloat(document.getElementById('labL').value);
  const a = parseFloat(document.getElementById('labA').value);
  const b = parseFloat(document.getElementById('labB').value);
  const model = document.getElementById('modelSelect').value;
  const topN  = parseInt(document.getElementById('topN').value);

  // Validation
  if (!colorName && (isNaN(L) || isNaN(a) || isNaN(b))) {
    showToast('Please enter a color name or L*a*b* values.', 'error');
    return;
  }

  const body = { substrate: currentSub, model, top_n: topN };
  if (colorName) {
    body.color_name = colorName;
  } else {
    body.lab = [L, a, b];
  }

  showLoading(true);

  try {
    const resp = await fetch(`${API}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const data = await resp.json();

    if (!resp.ok || data.error) {
      throw new Error(data.error || 'Prediction failed');
    }

    renderResults(data);
  } catch (err) {
    showToast(err.message || 'Server error', 'error');
  } finally {
    showLoading(false);
  }
}

// ── Render Results ────────────────────────────────────────────
function renderResults(data) {
  const panel = document.getElementById('resultsPanel');
  panel.style.display = 'flex';
  panel.style.flexDirection = 'column';
  panel.style.gap = '20px';

  // Target color card
  document.getElementById('targetSwatch').style.backgroundColor = data.target_hex;
  document.getElementById('targetName').textContent  = data.color_name;
  document.getElementById('targetLab').textContent   =
    `L* ${data.target_lab[0].toFixed(1)}  a* ${data.target_lab[1].toFixed(1)}  b* ${data.target_lab[2].toFixed(1)}`;
  document.getElementById('targetSubstrate').textContent =
    `Polyester ${capitalize(data.substrate)} — ${data.model.replace(/_/g,' ').toUpperCase()}`;

  // Recipe cards
  const grid = document.getElementById('recipesGrid');
  grid.innerHTML = '';

  // Find max concentration for bar scaling
  const allConcs = data.recipes.flatMap(r => Object.values(r.recipe));
  const maxConc  = Math.max(...allConcs, 0.001);

  data.recipes.forEach((recipe, idx) => {
    const deClass = getDeClass(recipe.dE2000);
    const card = document.createElement('div');
    card.className = 'recipe-card glass-card';

    const dyeRows = Object.entries(recipe.recipe)
      .sort((a,b) => b[1] - a[1])
      .map(([dye, conc]) => {
        const pct = Math.min((conc / maxConc) * 100, 100);
        return `
          <div class="dye-item">
            <div class="dye-item-header">
              <span class="dye-name">${dye}</span>
              <span class="dye-conc">${conc.toFixed(4)}% owf</span>
            </div>
            <div class="dye-bar-wrap">
              <div class="dye-bar" style="width:${pct}%"></div>
            </div>
          </div>
        `;
      }).join('');

    card.innerHTML = `
      <div class="recipe-header">
        <span class="recipe-rank">Recipe #${recipe.rank}</span>
        <span class="grade-badge grade-${recipe.grade}">${recipe.grade}</span>
      </div>

      <div class="de-row">
        <span class="de-value ${deClass}">${recipe.dE2000.toFixed(4)}</span>
        <span class="de-label">ΔE₂₀₀₀</span>
      </div>
      <div class="pred-swatch" style="--swatch-color: ${recipe.predicted_hex}; background-color: ${recipe.predicted_hex}; opacity:0.7; height:4px; border-radius:2px;"></div>

      <div class="dye-list">${dyeRows}</div>

      <div class="recipe-footer">
        <span class="ndyes-chip">${recipe.n_dyes} Dyes Used</span>
        <button class="copy-btn" onclick="copyRecipe(${idx}, event)">
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
          Copy
        </button>
      </div>
    `;

    grid.appendChild(card);
  });

  // Scroll to results
  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Store for copy
  window._lastResults = data;
}

function getDeClass(dE) {
  if (dE < 1.0) return 'de-excellent';
  if (dE < 2.0) return 'de-good';
  if (dE < 3.5) return 'de-acceptable';
  return 'de-reject';
}

// ── Copy Recipe ───────────────────────────────────────────────
function copyRecipe(idx, e) {
  e.stopPropagation();
  const data   = window._lastResults;
  const recipe = data.recipes[idx];

  const lines = [
    `Color: ${data.color_name}`,
    `Substrate: Polyester ${data.substrate}`,
    `ΔE₂₀₀₀: ${recipe.dE2000.toFixed(4)} (${recipe.grade})`,
    `Target Lab: L=${data.target_lab[0].toFixed(1)} a=${data.target_lab[1].toFixed(1)} b=${data.target_lab[2].toFixed(1)}`,
    ``,
    `Recipe (${recipe.n_dyes} dyes):`,
    ...Object.entries(recipe.recipe)
        .sort((a,b) => b[1]-a[1])
        .map(([d,c]) => `  ${d.padEnd(22)}: ${c.toFixed(4)}% owf`),
  ];

  navigator.clipboard.writeText(lines.join('\n'))
    .then(() => showToast('Recipe copied to clipboard!', 'success'))
    .catch(() => showToast('Copy failed — try manually', 'error'));
}

// ── Loading ───────────────────────────────────────────────────
function showLoading(show) {
  document.getElementById('loadingOverlay').style.display = show ? 'flex' : 'none';
  document.getElementById('predictBtn').disabled = show;
}

// ── Toast ─────────────────────────────────────────────────────
function showToast(msg, type = 'info') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className   = `toast ${type} show`;
  setTimeout(() => t.classList.remove('show'), 3500);
}

// ── Helpers ───────────────────────────────────────────────────
function capitalize(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : ''; }
