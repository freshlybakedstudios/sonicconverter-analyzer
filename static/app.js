// SonicConverter Analyzer — Frontend logic
// Handles: registration, file upload, analysis, results rendering

const API_URL = ''; // Same origin when served by FastAPI; set to ngrok URL for remote

let accessToken = null;
let selectedFile = null;
let currentJobId = null;
let allMatches = [];      // Current view match list
let tierMatches = [];     // Tier-filtered matches
let fullPoolMatches = []; // All artists (no tier filter)
let totalAllMatches = 0;  // Real total before cap
let matchesShown = 20;    // Current pagination offset
const MATCHES_PER_PAGE = 20;
let eventSource = null;   // SSE connection
let inputMode = 'file';   // 'file' or 'url'
let matchView = 'tier';   // 'tier' or 'all'
let userTier = '';        // User's tier label
let storedPlaylists = {}; // match_key -> playlist SSE data (preserved across re-renders)
let storedConfidence = {}; // match_key -> confidence level

// -------------------------------------------------------
// Helpers
// -------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const show = (el) => el.classList.remove('hidden');
const hide = (el) => el.classList.add('hidden');

function scrollToRegister() {
  $('#register-section').scrollIntoView({ behavior: 'smooth' });
}

function resetAll() {
  hide($('#results-section'));
  hide($('#floating-cta'));
  show($('#upload-section'));
  selectedFile = null;
  allMatches = [];
  matchesShown = 20;
  currentJobId = null;
  if (eventSource) { eventSource.close(); eventSource = null; }
  hide($('#selected-file'));
  $('#file-input').value = '';
}

// -------------------------------------------------------
// Registration
// -------------------------------------------------------
$('#register-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = $('#register-btn');
  const name = $('#name').value.trim();
  const email = $('#email').value.trim();
  if (!name || !email) return;

  btn.disabled = true;
  btn.textContent = 'Registering...';

  try {
    const form = new FormData();
    form.append('name', name);
    form.append('email', email);
    const spotifyUrl = $('#spotify-url').value.trim();
    if (spotifyUrl) form.append('spotify_url', spotifyUrl);
    const monthlyListeners = $('#monthly-listeners').value.trim();
    if (monthlyListeners) form.append('monthly_listeners', monthlyListeners);

    const res = await fetch(`${API_URL}/api/register`, { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Registration failed');
    }
    const data = await res.json();
    accessToken = data.token;

    // Track lead conversion
    if (typeof gtag === 'function') {
      gtag('event', 'generate_lead', {
        currency: 'USD',
        value: 1.00
      });
    }
    if (typeof fbq === 'function') {
      fbq('track', 'Lead');
    }

    // Transition
    hide($('#register-section'));
    hide($('#hero'));
    show($('#upload-section'));
  } catch (err) {
    alert(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Unlock Analyzer';
  }
});

// -------------------------------------------------------
// File Upload
// -------------------------------------------------------
const uploadZone = $('#upload-zone');
const fileInput = $('#file-input');

uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('dragover');
});
uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('dragover');
});
uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) {
    handleFile(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) {
    handleFile(fileInput.files[0]);
  }
});

function handleFile(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  const allowed = ['mp3', 'wav', 'flac', 'ogg', 'm4a', 'aac'];
  if (!allowed.includes(ext)) {
    alert('Please upload an audio file (MP3, WAV, FLAC, OGG, M4A, AAC)');
    return;
  }
  if (file.size > 50 * 1024 * 1024) {
    alert('File too large (max 50 MB)');
    return;
  }
  selectedFile = file;
  $('#file-name').textContent = file.name;
  show($('#selected-file'));
  updateAnalyzeButton();
}

// Enable analyze button only when input is ready (genre optional for URL mode)
function updateAnalyzeButton() {
  const btn = $('#analyze-btn');
  const hasGenre = !!$('#genre-select').value;
  if (inputMode === 'url') {
    const urlEl = $('#spotify-track-url');
    const hasUrl = urlEl && urlEl.value.trim().includes('spotify');
    btn.disabled = !hasUrl;  // Genre optional — CM provides it
  } else {
    const hasFile = !!selectedFile;
    btn.disabled = !(hasFile && hasGenre);
  }
}

// Listen for genre changes
$('#genre-select').addEventListener('change', updateAnalyzeButton);

// Listen for URL input changes
document.addEventListener('DOMContentLoaded', () => {
  const urlInput = $('#spotify-track-url');
  if (urlInput) urlInput.addEventListener('input', updateAnalyzeButton);
});

// -------------------------------------------------------
// Input Mode Toggle (File vs URL)
// -------------------------------------------------------
function setInputMode(mode) {
  inputMode = mode;
  const fileToggle = $('#mode-file');
  const urlToggle = $('#mode-url');
  const fileZone = $('#upload-zone');
  const urlInput = $('#url-input-zone');

  const genreLabel = document.querySelector('.genre-picker label');
  if (mode === 'file') {
    fileToggle && fileToggle.classList.add('active');
    urlToggle && urlToggle.classList.remove('active');
    fileZone && show(fileZone);
    urlInput && hide(urlInput);
    if (genreLabel) genreLabel.textContent = 'Genre';
  } else {
    fileToggle && fileToggle.classList.remove('active');
    urlToggle && urlToggle.classList.add('active');
    fileZone && hide(fileZone);
    urlInput && show(urlInput);
    if (genreLabel) genreLabel.textContent = 'Genre (optional — auto-detected from Spotify)';
  }
  updateAnalyzeButton();
}

// Wire up mode toggles (if elements exist)
document.addEventListener('DOMContentLoaded', () => {
  const modeFile = $('#mode-file');
  const modeUrl = $('#mode-url');
  if (modeFile) modeFile.addEventListener('click', () => setInputMode('file'));
  if (modeUrl) modeUrl.addEventListener('click', () => setInputMode('url'));
});

// -------------------------------------------------------
// Analyze
// -------------------------------------------------------
$('#analyze-btn').addEventListener('click', analyzeTrack);

async function analyzeTrack() {
  if (!accessToken) return;
  const genre = $('#genre-select').value;

  if (inputMode === 'file') {
    if (!selectedFile) return;
    if (!genre) {
      alert('Please select a genre before analyzing.');
      return;
    }
  } else {
    const urlVal = ($('#spotify-track-url') || {}).value || '';
    if (!urlVal.includes('spotify.com/track/') && !urlVal.includes('spotify:track:')) {
      alert('Please enter a valid Spotify track URL');
      return;
    }
    // Genre is optional for URL mode — CM will provide it
  }

  // Show loading
  hide($('#upload-section'));
  show($('#loading-section'));

  const statuses = inputMode === 'url'
    ? ['Fetching track from Spotify', 'Extracting audio features', 'Matching against 140,000+ tracks', 'Generating recommendations']
    : ['Extracting audio features', 'Analyzing frequency spectrum', 'Detecting emotional character', 'Matching against 140,000+ tracks', 'Generating recommendations'];
  let statusIdx = 0;
  const statusInterval = setInterval(() => {
    statusIdx = Math.min(statusIdx + 1, statuses.length - 1);
    $('#loader-status').textContent = statuses[statusIdx];
  }, 3000);

  try {
    let res;
    if (inputMode === 'url') {
      const form = new FormData();
      form.append('spotify_url', $('#spotify-track-url').value.trim());
      form.append('token', accessToken);
      if (genre) form.append('genre', genre);
      res = await fetch(`${API_URL}/api/analyze-url`, { method: 'POST', body: form });
    } else {
      const form = new FormData();
      form.append('file', selectedFile);
      form.append('token', accessToken);
      if (genre) form.append('genre', genre);
      res = await fetch(`${API_URL}/api/analyze`, { method: 'POST', body: form });
    }
    clearInterval(statusInterval);

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Analysis failed');
    }

    const data = await res.json();
    currentJobId = data.job_id;

    // Track analysis completion
    if (typeof gtag === 'function') {
      gtag('event', 'analysis_complete', { event_category: 'engagement' });
    }

    renderResults(data);

    hide($('#loading-section'));
    show($('#results-section'));
    show($('#floating-cta'));
    $('#results-section').scrollIntoView({ behavior: 'smooth' });

    // Start SSE for background enrichment
    if (currentJobId) {
      startSSE(currentJobId);
    }

  } catch (err) {
    clearInterval(statusInterval);
    hide($('#loading-section'));
    show($('#upload-section'));
    alert('Analysis failed: ' + err.message);
  }
}

// -------------------------------------------------------
// SSE Client for background enrichment
// -------------------------------------------------------
let sseComplete = false;
function startSSE(jobId) {
  if (eventSource) eventSource.close();
  sseComplete = false;
  eventSource = new EventSource(`${API_URL}/api/analysis/${jobId}/stream`);

  // Show enrichment status
  const enrichEl = $('#enrichment-status');
  if (enrichEl) show(enrichEl);

  eventSource.addEventListener('playlists', (e) => {
    const data = JSON.parse(e.data);
    storedPlaylists[data.match_key] = data;
    appendPlaylistData(data);
    updateEnrichmentProgress('playlists', data.progress);
  });

  eventSource.addEventListener('all_playlists', (e) => {
    const data = JSON.parse(e.data);
    renderAllPlaylists(data.playlists, data.total);
  });

  eventSource.addEventListener('confidence', (e) => {
    const data = JSON.parse(e.data);
    storedConfidence = data.confidence_map || {};
    renderConfidenceBadges(storedConfidence);
  });

  eventSource.addEventListener('related_artists', (e) => {
    const data = JSON.parse(e.data);
    updateEnrichmentProgress('related_artists', `${data.count} found`);
  });

  eventSource.addEventListener('credits', (e) => {
    const data = JSON.parse(e.data);
    appendCreditsData(data);
  });

  eventSource.addEventListener('curator_emails', (e) => {
    const data = JSON.parse(e.data);
    appendCuratorEmail(data);
    updateEnrichmentProgress('emails', data.progress);
  });

  eventSource.addEventListener('enrichment_progress', (e) => {
    const data = JSON.parse(e.data);
    const pct = data.total_batches ? Math.round(data.batch / data.total_batches * 100) : 0;
    const msg = `${pct}% — ${data.curators_found} curators found`;
    updateEnrichmentProgress('playlists', msg);
  });

  eventSource.addEventListener('complete', () => {
    sseComplete = true;
    const enrichEl = $('#enrichment-status');
    if (enrichEl) {
      enrichEl.innerHTML = '<span class="enrichment-done">Enrichment complete</span>';
      setTimeout(() => hide(enrichEl), 3000);
    }
    eventSource.close();
    eventSource = null;
  });

  eventSource.addEventListener('error', () => {
    if (eventSource) { eventSource.close(); eventSource = null; }
    // Auto-reconnect after 2s (Railway proxy can drop long SSE connections)
    if (!sseComplete) {
      console.log('SSE disconnected, reconnecting in 2s...');
      setTimeout(() => { if (!sseComplete) startSSE(jobId); }, 2000);
    }
  });
}

function updateEnrichmentProgress(type, value) {
  if (!value) return;
  const el = $(`#enrichment-${type}`);
  if (el) el.textContent = value;
  const statusEl = $('#enrichment-status');
  if (statusEl && !statusEl.classList.contains('hidden')) {
    const progress = statusEl.querySelector('.enrichment-progress');
    if (progress) progress.textContent = `Enriching: ${type} ${value}`;
  }
}

// -------------------------------------------------------
// Results Rendering
// -------------------------------------------------------
const EMOTION_LABELS = {
  power: 'Power', nostalgia: 'Nostalgia', tension: 'Tension',
  aggressive: 'Aggression', intense: 'Intensity', dark: 'Darkness',
  brooding: 'Brooding', wonder: 'Wonder', tenderness: 'Tenderness',
  joyfulness: 'Joyfulness', sadness: 'Sadness', peacefulness: 'Peacefulness',
  transcendence: 'Transcendence',
};

function renderResults(data) {
  const f = data.features || {};
  const matches = data.matches || [];
  const recs = data.recommendations || [];
  const genreAlignment = data.genre_alignment || null;
  const userProfile = data.user_profile || null;
  const flatteryMatches = data.flattery_matches || [];

  // Descriptive labels for numeric features
  function energyLabel(v) {
    if (v < 0.05) return 'Intimate';
    if (v < 0.12) return 'Mellow';
    if (v < 0.25) return 'Moderate';
    if (v < 0.45) return 'Driving';
    return 'Explosive';
  }
  function compressionLabel(v) {
    if (v < 0.3) return 'Open';
    if (v < 0.5) return 'Light';
    if (v < 0.65) return 'Balanced';
    if (v < 0.8) return 'Punchy';
    return 'Slammed';
  }
  function danceabilityLabel(v) {
    if (v < 0.9) return 'Freeform';
    if (v < 1.05) return 'Laid-back';
    if (v < 1.2) return 'Rhythmic';
    if (v < 1.4) return 'Groovy';
    return 'Club-ready';
  }

  // Artist card — show when we have artist data from source or user_profile
  const artistCard = $('#artist-card');
  const src = data.source || {};
  const up = userProfile || {};
  if (src.artist_name || up.name) {
    const name = src.artist_name || up.name || '';
    const listeners = up.listeners || src.artist_listeners || 0;
    const followers = up.followers || 0;
    const conversion = up.conversion_rate;
    const genres = src.artist_genres || '';
    const tier = src.artist_tier || data.user_tier || '';

    const trackName = src.track_name || '';
    $('#artist-card-name').textContent = name + (trackName ? ` — ${trackName}` : '');
    $('#artist-card-tier').textContent = tier ? tier.charAt(0).toUpperCase() + tier.slice(1) : '';

    function fmtNum(n) {
      if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
      if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
      return n.toString();
    }
    $('#artist-card-listeners').textContent = listeners > 0 ? fmtNum(listeners) : '-';
    $('#artist-card-followers').textContent = followers > 0 ? fmtNum(followers) : '-';
    $('#artist-card-conversion').textContent = conversion != null ? conversion.toFixed(2) + '%' : '-';
    $('#artist-card-genres').textContent = genres || '-';
    show(artistCard);
  } else {
    hide(artistCard);
  }

  // Key stats
  const statsGrid = $('#stats-grid');
  statsGrid.innerHTML = '';
  const stats = [
    { value: (f.bpm || 0).toFixed(0), label: 'BPM' },
    { value: `${f.key || '?'} ${f.scale || ''}`, label: 'KEY' },
    { value: (f.lufs_integrated || 0).toFixed(1), label: 'LUFS', subtitle: 'Integrated Loudness' },
    { value: energyLabel(f.energy || 0), label: 'ENERGY' },
    { value: compressionLabel(f.compression_amount || 0), label: 'COMPRESSION' },
    { value: danceabilityLabel(f.danceability || 0), label: 'DANCEABILITY' },
  ];
  stats.forEach(s => {
    const div = document.createElement('div');
    div.className = 'stat-card';
    const subtitleHtml = s.subtitle ? `<div class="stat-subtitle">${s.subtitle}</div>` : '';
    div.innerHTML = `<div class="stat-value">${s.value}</div><div class="stat-label">${s.label}</div>${subtitleHtml}`;
    statsGrid.appendChild(div);
  });

  // EQ chart
  const eqChart = $('#eq-chart');
  eqChart.innerHTML = '';
  const bands = [
    { label: 'Sub', key: 'sub_ratio' },
    { label: 'Bass', key: 'bass_ratio' },
    { label: 'Low-Mid', key: 'low_mid_ratio' },
    { label: 'Mid', key: 'mid_ratio' },
    { label: 'Hi-Mid', key: 'high_mid_ratio' },
    { label: 'Presence', key: 'presence_ratio' },
    { label: 'Air', key: 'air_ratio' },
  ];
  const maxBand = Math.max(...bands.map(b => f[b.key] || 0), 0.01);
  bands.forEach(b => {
    const val = f[b.key] || 0;
    const pct = Math.min(100, (val / (maxBand * 1.2)) * 100);
    const row = document.createElement('div');
    row.className = 'eq-row';
    row.innerHTML = `
      <span class="eq-label">${b.label}</span>
      <div class="eq-bar-bg"><div class="eq-bar-fill" style="width:${pct}%"></div></div>
      <span class="eq-value">${(val * 100).toFixed(1)}%</span>
    `;
    eqChart.appendChild(row);
  });

  // Emotions
  const emotionDiv = $('#emotion-tags');
  emotionDiv.innerHTML = '';
  const emotions = (f.emotion_summary && f.emotion_summary.emotions) || [];
  if (emotions.length === 0) {
    for (let i = 1; i <= 4; i++) {
      const e = f[`emotion_${i}`];
      const s = f[`emotion_${i}_score`] || 0;
      if (e && e !== 'neutral') emotions.push([e, s]);
    }
  }
  emotions.forEach(([emo, score]) => {
    const tag = document.createElement('span');
    tag.className = 'emotion-tag';
    tag.innerHTML = `${EMOTION_LABELS[emo] || emo} <span class="emotion-score">${(score * 100).toFixed(0)}%</span>`;
    emotionDiv.appendChild(tag);
  });

  // Genre alignment callout
  const genreCallout = $('#genre-callout');
  if (genreAlignment && genreAlignment.genre) {
    $('#genre-callout-value').textContent = genreAlignment.genre;
    $('#genre-callout-detail').textContent =
      `${genreAlignment.count} of ${genreAlignment.total} matches (${genreAlignment.percentage.toFixed(0)}%) share this genre`;
    show(genreCallout);
  } else {
    hide(genreCallout);
  }

  // Where You Stand — conversion comparison card
  const convCard = $('#conversion-card');
  const comp = (userProfile && userProfile.conversion_comparison) || {};
  const hasUserRate = userProfile && userProfile.conversion_rate != null;

  if (userProfile && (hasUserRate || comp.peer_median != null)) {
    const cr = userProfile.conversion_rate;
    const p25 = comp.peer_bottom_25 || 0;
    const median = comp.peer_median || 0;
    const p75 = comp.peer_top_25 || 0;
    const p99 = Math.min(comp.peer_p99 || p75 * 2, p75 * 3);
    const atTop = cr >= p75;
    const target = atTop && p99 > cr ? p99 : p75; // Target p99 if already above p75
    const fans = userProfile.additional_fans || 0;
    const scaleMax = Math.max(p99, cr, target) * 1.05;
    const toPos = v => Math.max(0, Math.min(100, (v / scaleMax) * 100));

    // Bar: solid fill up to "You"
    if (hasUserRate) {
      $('#conv-bar-fill').style.width = toPos(cr) + '%';
      $('#conv-bar-dot').style.left = toPos(cr) + '%';

      // Opportunity zone (striped) from You to Target
      const oppBar = $('#conv-bar-opportunity');
      if (target > cr) {
        oppBar.style.left = toPos(cr) + '%';
        oppBar.style.width = (toPos(target) - toPos(cr)) + '%';
        oppBar.style.display = 'block';
      } else {
        oppBar.style.display = 'none';
      }
    }

    // Tick marks
    $('#conv-tick-p25').style.left = toPos(p25) + '%';
    $('#conv-tick-median').style.left = toPos(median) + '%';
    $('#conv-tick-p75').style.left = toPos(p75) + '%';
    $('#conv-tick-p99').style.left = toPos(p99) + '%';
    if (target > cr) {
      $('#conv-tick-target').style.left = toPos(target) + '%';
      $('#conv-tick-target').style.display = 'block';
    } else {
      $('#conv-tick-target').style.display = 'none';
    }

    // Fan label on the gap
    const fanLabel = $('#conv-fan-label');
    if (hasUserRate && fans > 0 && target > cr) {
      fanLabel.textContent = '+' + fans.toLocaleString() + ' fans';
      fanLabel.style.marginLeft = toPos(cr) + '%';
      fanLabel.style.width = (toPos(target) - toPos(cr)) + '%';
      show(fanLabel);
    } else {
      hide(fanLabel);
    }

    // Labels below bar — collision-aware (higher priority labels kept)
    const labelsEl = $('#conv-bar-labels');
    labelsEl.innerHTML = '';
    const MIN_GAP = 8;
    const allLabels = [
      { pos: toPos(p25), name: 'p25', val: p25.toFixed(2) + '%', cls: '', priority: 1 },
      { pos: toPos(median), name: 'Median', val: median.toFixed(2) + '%', cls: '', priority: 2 },
      { pos: toPos(p75), name: 'p75', val: p75.toFixed(2) + '%', cls: '', priority: 1 },
    ];
    if (hasUserRate) {
      allLabels.push({ pos: toPos(cr), name: 'You', val: cr.toFixed(2) + '%', cls: 'conv-bar-label-you', priority: 5 });
    }
    if (target > cr) {
      const targetName = atTop ? 'Top 1%' : 'Top 25%';
      allLabels.push({ pos: toPos(target), name: targetName, val: target.toFixed(2) + '%', cls: 'conv-bar-label-target', priority: 4 });
    }
    if (!atTop || target !== p99) {
      allLabels.push({ pos: toPos(p99), name: 'p99', val: p99.toFixed(2) + '%', cls: '', priority: 1 });
    }
    // Sort by priority (highest kept), filter overlaps
    const sorted = [...allLabels].sort((a, b) => b.priority - a.priority);
    const kept = [];
    for (const label of sorted) {
      if (kept.every(k => Math.abs(k.pos - label.pos) >= MIN_GAP)) {
        kept.push(label);
      }
    }
    for (const l of kept) {
      const div = document.createElement('div');
      div.className = 'conv-bar-label ' + l.cls;
      div.style.left = l.pos + '%';
      div.innerHTML = `<span>${l.name}</span><span>${l.val}</span>`;
      labelsEl.appendChild(div);
    }

    // Combined contextual text + revenue projection
    const oppEl = $('#conv-opportunity');
    const revEl = $('#conv-revenue');
    const REVENUE_PER_LISTENER = 0.13; // ~$0.13/listener/year (Loud & Clear + published artist data)
    const listeners = (userProfile.listeners || 0);

    if (hasUserRate) {
      const peerCount = (comp.peer_count || 0).toLocaleString();

      if (fans > 0 && atTop) {
        oppEl.innerHTML = `You're already in the <span class="fan-number">top 25%</span> of ${peerCount} sonic peers. Reaching the top 1% (${target.toFixed(1)}%) would convert an estimated <span class="fan-number">${fans.toLocaleString()} additional listeners into followers</span> — each one a direct line to your releases, merch drops, and tour dates via Spotify push notifications.`;
      } else if (fans > 0) {
        oppEl.innerHTML = `Across ${peerCount} sonic peers, the top 25% convert at <span class="fan-number">${target.toFixed(1)}%</span>. Closing that gap means an estimated <span class="fan-number">${fans.toLocaleString()} new followers</span> — each one receiving push notifications for your releases, merch drops, and tour dates.`;
      } else {
        oppEl.innerHTML = `You're converting at <span class="fan-number">${cr.toFixed(1)}%</span> — above the top 1% of ${peerCount} sonic peers. Your listener-to-follower conversion is exceptional.`;
      }
      show(oppEl);

      // Revenue projection using per-listener rate
      if (listeners > 0) {
        function fmtNum(n) {
          if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
          if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
          return n.toFixed(0);
        }
        function fmtRev(n) { return '$' + fmtNum(n); }
        const currentRev = Math.round(listeners * REVENUE_PER_LISTENER);
        const projectedRev = Math.round((listeners + fans) * REVENUE_PER_LISTENER);

        $('#conv-revenue-title').textContent = 'Estimated Spotify Revenue';
        const earningsEl = $('#conv-tier-earnings');

        if (fans > 0) {
          earningsEl.innerHTML = `
            <div class="conv-tier-row">
              <span class="conv-tier-label">Current (${fmtNum(listeners)} listeners)</span>
              <span class="conv-tier-val">${fmtRev(currentRev)}/year</span>
            </div>
            <div class="conv-tier-row">
              <span class="conv-tier-label">With gap closed (+${fans.toLocaleString()} fans)</span>
              <span class="conv-tier-val conv-tier-val-projected">${fmtRev(projectedRev)}/year</span>
            </div>
          `;
        } else {
          earningsEl.innerHTML = `
            <div class="conv-tier-row">
              <span class="conv-tier-label">Estimated from ${fmtNum(listeners)} listeners</span>
              <span class="conv-tier-val">${fmtRev(currentRev)}/year</span>
            </div>
          `;
        }
        $('#conv-sources').innerHTML = 'Derived from <a href="https://loudandclear.byspotify.com/" target="_blank" rel="noopener noreferrer">Spotify Loud & Clear 2025</a> and <a href="https://andrewsouthworth.com/how-many-monthly-listeners-to-make-a-living-on-spotify/" target="_blank" rel="noopener noreferrer">published artist earnings data</a>. Streaming revenue only — does not include merch, touring, or sync.';
        show(revEl);
      } else {
        hide(revEl);
      }
    } else {
      hide(oppEl);
      hide(revEl);
    }

    show(convCard);
  } else {
    hide(convCard);
  }

  // Trajectory targets (flattery matches) with Show More
  const flatteryCard = $('#flattery-card');
  const flatteryList = $('#flattery-list');
  if (flatteryMatches.length > 0) {
    flatteryList.innerHTML = '';
    const FLATTERY_PAGE = 3;
    let flatteryShown = 0;

    function renderFlatteryBatch() {
      const end = Math.min(flatteryShown + FLATTERY_PAGE, flatteryMatches.length);
      for (let i = flatteryShown; i < end; i++) {
        const m = flatteryMatches[i];
        const trackLink = m.track_url
          ? `<a href="${m.track_url}" target="_blank" rel="noopener">${m.track_name || 'Listen'}</a>`
          : (m.track_name || '');
        const artistLink = m.spotify_url
          ? `<a href="${m.spotify_url}" target="_blank" rel="noopener">${m.name}</a>`
          : m.name;
        const sim = (m.similarity * 100).toFixed(1);
        const tier = (m.tier || '').charAt(0).toUpperCase() + (m.tier || '').slice(1);
        const listeners = m.listeners ? Math.round(m.listeners).toLocaleString() : '';
        const listenerStr = listeners ? `<span class="flattery-listeners">${listeners} listeners</span>` : '';
        const div = document.createElement('div');
        div.className = 'flattery-match';
        div.innerHTML = `
          <div class="flattery-rank">${i + 1}</div>
          <div class="flattery-info">
            <div class="flattery-name">${artistLink} <span class="flattery-tier">${tier}</span></div>
            <div class="flattery-track">${trackLink} ${listenerStr}</div>
          </div>
          <div class="flattery-sim">${sim}%</div>
        `;
        flatteryList.appendChild(div);
      }
      flatteryShown = end;
      updateFlatteryShowMore();
    }

    function updateFlatteryShowMore() {
      let btn = $('#flattery-show-more');
      if (!btn) {
        btn = document.createElement('button');
        btn.id = 'flattery-show-more';
        btn.className = 'btn-secondary';
        btn.style.cssText = 'margin-top:12px;width:100%';
        btn.onclick = renderFlatteryBatch;
        flatteryList.parentElement.appendChild(btn);
      }
      if (flatteryShown < flatteryMatches.length) {
        btn.textContent = `Show ${Math.min(FLATTERY_PAGE, flatteryMatches.length - flatteryShown)} More`;
        btn.style.display = '';
      } else {
        btn.style.display = 'none';
      }
    }

    renderFlatteryBatch();
    show(flatteryCard);
  } else {
    hide(flatteryCard);
  }

  // Matches table with pagination
  storedPlaylists = {};
  storedConfidence = {};
  seenCurators.clear();
  curatorRows.length = 0;
  sseComplete = false;
  // Filter out matches with no genre data
  function hasGenre(m) {
    if (m.primary_genre && m.primary_genre.toLowerCase() !== 'unknown') return true;
    if (m.secondary_genre && m.secondary_genre.toLowerCase() !== 'unknown') return true;
    if (m.artist_genres && m.artist_genres.length > 0) return true;
    return false;
  }
  tierMatches = matches.filter(hasGenre);
  fullPoolMatches = (data.all_matches || matches).filter(hasGenre);
  totalAllMatches = data.total_all_matches || fullPoolMatches.length;
  userTier = data.user_tier || '';
  matchView = 'tier';
  allMatches = tierMatches;
  matchesShown = MATCHES_PER_PAGE;

  // Tier toggle
  const tierToggle = $('#tier-toggle');
  if (tierToggle && fullPoolMatches.length > tierMatches.length) {
    const tierLabel = userTier ? userTier.charAt(0).toUpperCase() + userTier.slice(1) : 'My Tier';
    $('#tier-btn-mine').textContent = tierLabel;
    show(tierToggle);
  }

  renderMatchView();

  // Recommendations
  const recList = $('#rec-list');
  recList.innerHTML = '';
  recs.forEach(r => {
    const li = document.createElement('li');
    const parts = r.split('\n');
    if (parts.length === 2) {
      li.innerHTML = `<div class="rec-action">${parts[0]}</div><div class="rec-consensus">${parts[1]}</div>`;
    } else {
      li.textContent = r;
    }
    recList.appendChild(li);
  });
}

// -------------------------------------------------------
// Match row rendering + pagination
// -------------------------------------------------------
function renderMatchRows(matchSlice, startIdx) {
  const tbody = $('#matches-body');
  matchSlice.forEach((m, i) => {
    const idx = startIdx + i;
    const tr = document.createElement('tr');
    tr.setAttribute('data-match-key', String(m.artist_id || m.name || idx));
    const linkUrl = m.track_url || m.spotify_url;
    const artistLink = linkUrl
      ? `<a href="${linkUrl}" target="_blank" rel="noopener">${m.name}</a>`
      : m.name;
    const emos = (m.emotions || []).filter(e => e && e !== 'neutral').slice(0, 3);
    const emoTags = emos.map(e => `<span class="mini-tag">${EMOTION_LABELS[e] || e}</span>`).join('');
    const convRate = m.conversion_rate != null ? m.conversion_rate.toFixed(1) + '%' : '-';
    const allGenres = new Set();
    if (m.primary_genre && m.primary_genre.toLowerCase() !== 'unknown') allGenres.add(m.primary_genre);
    if (m.secondary_genre && m.secondary_genre.toLowerCase() !== 'unknown') allGenres.add(m.secondary_genre);
    (m.artist_genres || []).forEach(g => { if (g) allGenres.add(g); });
    const genreStr = allGenres.size > 0 ? [...allGenres].join(', ') : '-';
    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td class="match-artist">${artistLink}</td>
      <td class="match-sim">${(m.similarity * 100).toFixed(1)}%</td>
      <td class="match-conversion">${convRate}</td>
      <td class="match-tier">${m.tier || '-'}</td>
      <td class="match-genre">${genreStr}</td>
      <td class="match-emotions">${emoTags}</td>
    `;
    // Collapsible playlist section (populated via SSE)
    const plRow = document.createElement('tr');
    plRow.className = 'playlist-row hidden';
    plRow.setAttribute('data-playlist-for', String(m.artist_id || m.name || idx));
    plRow.innerHTML = `<td colspan="7"><div class="match-playlists"></div></td>`;

    tbody.appendChild(tr);
    tbody.appendChild(plRow);
  });
}

function showMoreMatches() {
  const nextBatch = allMatches.slice(matchesShown, matchesShown + MATCHES_PER_PAGE);
  if (nextBatch.length === 0) return;
  renderMatchRows(nextBatch, matchesShown);
  matchesShown += nextBatch.length;

  // Re-apply enrichment to newly rendered rows
  _reapplyEnrichment();

  const matchCounter = $('#match-counter');
  if (matchCounter) {
    matchCounter.textContent = `Showing ${matchesShown} of ${allMatches.length} matches`;
  }
  const showMoreBtn = $('#show-more-btn');
  if (showMoreBtn) {
    if (matchesShown >= allMatches.length) {
      hide(showMoreBtn);
    } else {
      showMoreBtn.textContent = `Show ${Math.min(MATCHES_PER_PAGE, allMatches.length - matchesShown)} More`;
    }
  }
}

// Render match table for current view (tier or all)
function renderMatchView() {
  const tbody = $('#matches-body');
  tbody.innerHTML = '';
  matchesShown = MATCHES_PER_PAGE;

  const totalCount = allMatches.length;
  const matchCounter = $('#match-counter');
  if (matchCounter) {
    let label;
    if (matchView === 'tier' && userTier) {
      label = `Showing ${Math.min(matchesShown, totalCount)} of ${totalCount} ${userTier} matches`;
    } else {
      const realTotal = totalAllMatches > totalCount ? ` (${totalAllMatches.toLocaleString()} total)` : '';
      label = `Showing ${Math.min(matchesShown, totalCount)} of ${totalCount.toLocaleString()} matches${realTotal}`;
    }
    matchCounter.textContent = label;
    show(matchCounter);
  }

  renderMatchRows(allMatches.slice(0, MATCHES_PER_PAGE), 0);

  // Re-apply stored enrichment data (playlists + confidence)
  _reapplyEnrichment();

  const showMoreBtn = $('#show-more-btn');
  if (showMoreBtn) {
    if (allMatches.length > MATCHES_PER_PAGE) {
      show(showMoreBtn);
      showMoreBtn.textContent = `Show ${MATCHES_PER_PAGE} More`;
    } else {
      hide(showMoreBtn);
    }
  }
}

function _reapplyEnrichment() {
  // Re-apply playlists to visible rows
  for (const [matchKey, data] of Object.entries(storedPlaylists)) {
    appendPlaylistData(data);
  }
  // Re-apply confidence badges
  if (Object.keys(storedConfidence).length > 0) {
    renderConfidenceBadges(storedConfidence);
  }
}

function switchMatchView(view) {
  matchView = view;
  allMatches = view === 'tier' ? tierMatches : fullPoolMatches;
  const btnMine = $('#tier-btn-mine');
  const btnAll = $('#tier-btn-all');
  if (btnMine) btnMine.classList.toggle('active', view === 'tier');
  if (btnAll) btnAll.classList.toggle('active', view === 'all');
  renderMatchView();
}

// Wire up Show More and Tier Toggle buttons
document.addEventListener('DOMContentLoaded', () => {
  const btn = $('#show-more-btn');
  if (btn) btn.addEventListener('click', showMoreMatches);
  const btnMine = $('#tier-btn-mine');
  const btnAll = $('#tier-btn-all');
  if (btnMine) btnMine.addEventListener('click', () => switchMatchView('tier'));
  if (btnAll) btnAll.addEventListener('click', () => switchMatchView('all'));
});

// -------------------------------------------------------
// SSE data rendering helpers
// -------------------------------------------------------
function appendPlaylistData(data) {
  const matchKey = data.match_key;
  const plRow = document.querySelector(`tr[data-playlist-for="${matchKey}"]`);
  if (!plRow) return;

  const container = plRow.querySelector('.match-playlists');
  if (!container) return;

  const playlists = data.playlists || [];
  if (playlists.length === 0) return;

  // Show the row
  plRow.classList.remove('hidden');

  // Make the parent match row clickable to toggle
  const matchRow = document.querySelector(`tr[data-match-key="${matchKey}"]`);
  // Render playlists — dedupe by playlist_id, show all
  const seen = new Set();
  const deduped = [];
  const sorted = [...playlists].sort((a, b) => (b.score || 0) - (a.score || 0));
  for (const pl of sorted) {
    const pid = pl.playlist_id || pl.name;
    if (!seen.has(pid)) { seen.add(pid); deduped.push(pl); }
  }

  if (matchRow && !matchRow.classList.contains('has-playlists')) {
    matchRow.classList.add('has-playlists');
    const badge = document.createElement('span');
    badge.className = 'playlist-count-badge';
    badge.textContent = `${deduped.length} playlists`;
    const artistCell = matchRow.querySelector('.match-artist');
    if (artistCell) artistCell.appendChild(badge);

    matchRow.style.cursor = 'pointer';
    matchRow.addEventListener('click', () => {
      plRow.classList.toggle('hidden');
    });
    // Start collapsed
    plRow.classList.add('hidden');
  }

  container.innerHTML = deduped.map(pl => {
    const freshDate = pl.added_at || pl.last_updated || '';
    const isRecent = _isRecentlyActive(freshDate, 30);
    return `
    <div class="playlist-item ${pl.editorial ? 'editorial' : ''} ${pl.double_validated ? 'double-validated' : ''} ${isRecent ? 'recently-active' : ''}">
      <a href="${pl.link}" target="_blank" rel="noopener">${pl.name}</a>
      <span class="playlist-followers">${(pl.followers || 0).toLocaleString()} followers</span>
      ${pl.editorial ? '<span class="playlist-badge editorial-badge">Editorial</span>' : ''}
      ${pl.status === 'current' ? '<span class="playlist-badge current-badge">Current</span>' : ''}
      ${isRecent ? '<span class="playlist-badge active-badge">Active</span>' : ''}
      ${freshDate ? `<span class="playlist-freshness">added ${_formatFreshness(freshDate)}</span>` : ''}
      ${pl.curator_name ? `<span class="playlist-curator">${pl.curator_name}</span>` : ''}
    </div>`;
  }).join('');
}

function _formatFreshness(dateStr) {
  if (!dateStr) return '';
  try {
    const d = new Date(dateStr);
    const now = new Date();
    const days = Math.floor((now - d) / (1000 * 60 * 60 * 24));
    if (days < 0) return 'just now';
    if (days === 0) return 'today';
    if (days < 7) return `${days}d ago`;
    if (days < 30) return `${Math.floor(days / 7)}w ago`;
    if (days < 365) return `${Math.floor(days / 30)}mo ago`;
    return `${Math.floor(days / 365)}y ago`;
  } catch { return ''; }
}

function _isRecentlyActive(dateStr, withinDays) {
  if (!dateStr) return false;
  try {
    const d = new Date(dateStr);
    const days = Math.floor((new Date() - d) / (1000 * 60 * 60 * 24));
    return days >= 0 && days <= withinDays;
  } catch { return false; }
}

function renderAllPlaylists(playlists, total) {
  const container = $('#all-playlists-list');
  const card = $('#all-playlists-card');
  if (!container || !card) return;

  show(card);
  const countEl = $('#all-playlists-count');
  const recentCount = playlists.filter(pl => _isRecentlyActive(pl.added_at || pl.last_updated || '', 90)).length;
  if (countEl) countEl.textContent = `${total} unique playlists found across wider pool — ${recentCount} active in last 90 days`;

  // Show CSV download button
  const csvBtn = $('#csv-download-btn');
  if (csvBtn && currentJobId) {
    csvBtn.classList.remove('hidden');
    csvBtn.onclick = () => {
      window.open(`${API_URL}/api/analysis/${currentJobId}/csv`, '_blank');
    };
  }

  container.innerHTML = playlists.map((pl, i) => {
    const freshDate = pl.added_at || pl.last_updated || '';
    const isRecent = _isRecentlyActive(freshDate, 30);
    return `
    <tr class="${pl.double_validated ? 'double-validated-row' : ''} ${isRecent ? 'recently-active-row' : ''}">
      <td>${i + 1}</td>
      <td><a href="${pl.link}" target="_blank" rel="noopener">${pl.name}</a></td>
      <td>${pl.sonic_match || ''}</td>
      <td>${(pl.followers || 0).toLocaleString()}</td>
      <td>${pl.editorial ? 'Editorial' : 'Indie'}</td>
      <td>${pl.status === 'current' ? '<span class="current-badge-sm">Current</span>' : 'Past'}</td>
      <td>${freshDate ? (isRecent ? `<span class="active-badge-sm">${_formatFreshness(freshDate)}</span>` : _formatFreshness(freshDate)) : '-'}</td>
      <td>${(pl.score || 0).toFixed(3)}</td>
      <td>${pl.curator_name || ''}</td>
    </tr>`;
  }).join('');
}

function renderConfidenceBadges(confidenceMap) {
  const audienceCard = $('#audience-match-card');
  const audienceList = $('#audience-match-list');

  // Find all double-validated matches from the current match data
  const dvKeys = new Set(
    Object.entries(confidenceMap)
      .filter(([, v]) => v === 'double_validated')
      .map(([k]) => k)
  );

  // Badge on match table rows
  for (const [key, value] of Object.entries(confidenceMap)) {
    const row = document.querySelector(`tr[data-match-key="${key}"]`);
    if (row && value === 'double_validated') {
      const artistCell = row.querySelector('.match-artist');
      if (artistCell && !artistCell.querySelector('.confidence-badge')) {
        const badge = document.createElement('span');
        badge.className = 'confidence-badge';
        badge.title = 'Also appears in CM Related Artists — audience overlap confirmed';
        badge.textContent = 'Audience Match';
        artistCell.appendChild(badge);
      }
      row.classList.add('double-validated-row');
    }
  }

  // Build Audience Match card from all match pools
  if (dvKeys.size > 0 && audienceCard && audienceList) {
    const allPool = [...(tierMatches || []), ...(fullPoolMatches || [])];
    const seen = new Set();
    const dvMatches = [];
    for (const m of allPool) {
      const key = String(m.artist_id || m.name || '');
      if (dvKeys.has(key) && !seen.has(key)) {
        seen.add(key);
        dvMatches.push(m);
      }
    }

    if (dvMatches.length > 0) {
      // Sort by similarity descending
      dvMatches.sort((a, b) => (b.similarity || 0) - (a.similarity || 0));

      audienceList.innerHTML = '';
      dvMatches.forEach((m, i) => {
        const linkUrl = m.track_url || m.spotify_url;
        const artistLink = linkUrl
          ? `<a href="${linkUrl}" target="_blank" rel="noopener">${m.name}</a>`
          : m.name;
        const allGenres = new Set();
        if (m.primary_genre && m.primary_genre.toLowerCase() !== 'unknown') allGenres.add(m.primary_genre);
        if (m.secondary_genre && m.secondary_genre.toLowerCase() !== 'unknown') allGenres.add(m.secondary_genre);
        (m.artist_genres || []).forEach(g => { if (g) allGenres.add(g); });
        const genreStr = allGenres.size > 0 ? [...allGenres].join(', ') : '';
        const emos = (m.emotions || []).filter(e => e && e !== 'neutral').slice(0, 3);
        const emoTags = emos.map(e => `<span class="mini-tag">${EMOTION_LABELS[e] || e}</span>`).join('');
        const listeners = m.listeners ? Math.round(m.listeners).toLocaleString() : '';

        const div = document.createElement('div');
        div.className = 'audience-match-item';
        div.innerHTML = `
          <div class="audience-match-rank">${i + 1}</div>
          <div class="audience-match-info">
            <div class="audience-match-name">${artistLink} <span class="audience-match-tier">${m.tier || ''}</span></div>
            <div class="audience-match-meta">
              ${(m.similarity * 100).toFixed(1)}% sonic match
              ${listeners ? ` · ${listeners} listeners` : ''}
              ${genreStr ? ` · ${genreStr}` : ''}
            </div>
            <div class="audience-match-emotions">${emoTags}</div>
          </div>
        `;
        audienceList.appendChild(div);
      });
      show(audienceCard);
    }
  }
}

function appendCreditsData(data) {
  const container = $('#credits-list');
  const card = $('#credits-card');
  if (!container || !card) return;

  show(card);
  const credits = data.credits || {};
  const artistName = data.artist_name || '';

  const producers = credits.producers || [];
  const writers = credits.writers || [];

  if (producers.length === 0 && writers.length === 0) return;

  const row = document.createElement('div');
  row.className = 'credits-entry';
  row.innerHTML = `
    <div class="credits-artist">${artistName}</div>
    ${producers.length ? `<div class="credits-role">Producers: ${producers.map(p => p.name).join(', ')}</div>` : ''}
    ${writers.length ? `<div class="credits-role">Writers: ${writers.map(w => w.name).join(', ')}</div>` : ''}
  `;
  container.appendChild(row);

  // Update aggregated counts
  updateCreditsSummary();
}

// Track all credits for aggregation
const creditCounts = { producers: {}, writers: {} };

function updateCreditsSummary() {
  const container = $('#credits-list');
  if (!container) return;

  // Rebuild from all credits entries
  const entries = container.querySelectorAll('.credits-entry');
  const pCounts = {};
  const wCounts = {};

  entries.forEach(entry => {
    const roles = entry.querySelectorAll('.credits-role');
    roles.forEach(role => {
      const text = role.textContent;
      const names = text.replace(/^(Producers|Writers):\s*/, '').split(', ');
      const isProducer = text.startsWith('Producers');
      names.forEach(name => {
        const n = name.trim();
        if (!n) return;
        if (isProducer) pCounts[n] = (pCounts[n] || 0) + 1;
        else wCounts[n] = (wCounts[n] || 0) + 1;
      });
    });
  });

  // Show top credits summary
  const summaryEl = $('#credits-summary');
  if (summaryEl) {
    const topProducers = Object.entries(pCounts).sort((a, b) => b[1] - a[1]).slice(0, 5);
    const topWriters = Object.entries(wCounts).sort((a, b) => b[1] - a[1]).slice(0, 5);
    let html = '';
    if (topProducers.length) {
      html += '<div class="credits-top"><strong>Top Producers:</strong> ' +
        topProducers.map(([n, c]) => `${n} <span class="credit-count">(${c} tracks)</span>`).join(', ') + '</div>';
    }
    if (topWriters.length) {
      html += '<div class="credits-top"><strong>Top Writers:</strong> ' +
        topWriters.map(([n, c]) => `${n} <span class="credit-count">(${c} tracks)</span>`).join(', ') + '</div>';
    }
    summaryEl.innerHTML = html;
  }
}

const seenCurators = new Set();
const curatorRows = [];  // accumulate for CSV export
function appendCuratorEmail(data) {
  const container = $('#curator-emails-body');
  const card = $('#curator-emails-card');
  if (!container || !card) return;

  show(card);
  const curator = data.curator || {};

  // Dedup — SSE reconnect replays all curators
  const curatorKey = `${curator.name}::${curator.playlist_name}`;
  if (seenCurators.has(curatorKey)) return;
  seenCurators.add(curatorKey);

  // Store for CSV export
  curatorRows.push(curator);
  const csvBtn = $('#curator-csv-btn');
  if (csvBtn) csvBtn.classList.remove('hidden');

  const countEl = $('#curator-contacts-count');
  if (countEl && data.progress) countEl.textContent = data.progress;

  // Build social links
  const links = [];
  if (curator.instagram_url) links.push(`<a href="${curator.instagram_url}" target="_blank" rel="noopener" title="Instagram" class="social-link">IG</a>`);
  if (curator.facebook_url) links.push(`<a href="${curator.facebook_url}" target="_blank" rel="noopener" title="Facebook" class="social-link">FB</a>`);
  if (curator.website_url) links.push(`<a href="${curator.website_url}" target="_blank" rel="noopener" title="Website" class="social-link">Web</a>`);
  if (curator.groover_url) links.push(`<a href="${curator.groover_url}" target="_blank" rel="noopener" title="Groover" class="social-link social-groover">Groover</a>`);
  if (curator.submithub_url) links.push(`<a href="${curator.submithub_url}" target="_blank" rel="noopener" title="SubmitHub" class="social-link social-submithub">SubmitHub</a>`);
  if (curator.submission_url) links.push(`<a href="${curator.submission_url}" target="_blank" rel="noopener" title="Submission" class="social-link social-submit">Submit</a>`);
  if (curator.twitter_url) links.push(`<a href="${curator.twitter_url}" target="_blank" rel="noopener" title="X / Twitter" class="social-link">X</a>`);

  const emailCell = curator.email
    ? `<a href="mailto:${curator.email}">${curator.email}</a>`
    : '<span class="no-email">—</span>';

  // Reference artist/track that appeared on this playlist
  const refArtist = curator.sonic_match || '';
  const refTrack = curator.track_name || '';
  const refCell = refTrack && refArtist
    ? `${refArtist} — ${refTrack}`
    : refArtist || refTrack || '—';

  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td>${curator.name || ''}</td>
    <td><a href="${curator.playlist_link || '#'}" target="_blank" rel="noopener">${curator.playlist_name || ''}</a></td>
    <td class="ref-artist-cell">${refCell}</td>
    <td>${(curator.followers || 0).toLocaleString()}</td>
    <td>${emailCell}</td>
    <td class="social-links-cell">${links.join(' ') || '—'}</td>
  `;
  container.appendChild(tr);
}

// Curator CSV download
function downloadCuratorCSV() {
  if (!curatorRows.length) return;
  const headers = ['Curator', 'Playlist', 'Reference Artist', 'Followers', 'Email', 'Instagram', 'Facebook', 'Website', 'Groover', 'SubmitHub', 'Submission URL', 'Twitter'];
  const escape = v => `"${String(v || '').replace(/"/g, '""')}"`;
  const rows = curatorRows.map(c => {
    const ref = (c.track_name && c.sonic_match) ? `${c.sonic_match} — ${c.track_name}` : (c.sonic_match || c.track_name || '');
    return [c.name, c.playlist_name, ref, c.followers || 0, c.email, c.instagram_url, c.facebook_url, c.website_url, c.groover_url, c.submithub_url, c.submission_url, c.twitter_url].map(escape).join(',');
  });
  const csv = [headers.join(','), ...rows].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'curator_contacts.csv';
  a.click();
  URL.revokeObjectURL(a.href);
}

(function() {
  const btn = document.getElementById('curator-csv-btn');
  if (btn) btn.addEventListener('click', downloadCuratorCSV);
})();

// -------------------------------------------------------
// Floating Pies — Asteroids-style with physics & mouse repel
// -------------------------------------------------------
const pieFiles = [
  '/static/pies/allfbsgraphices-02.svg',
  '/static/pies/allfbsgraphices-03.svg',
  '/static/pies/allfbsgraphices-04.svg',
  '/static/pies/allfbsgraphices-05.svg',
  '/static/pies/allfbsgraphices-06.svg',
  '/static/pies/allfbsgraphices-07.svg',
  '/static/pies/allfbsgraphices-09.svg',
  '/static/pies/allfbsgraphices-10.svg',
  '/static/pies/allfbsgraphices-11.svg',
  '/static/pies/allfbsgraphices-13.svg',
  '/static/pies/allfbsgraphices-14.svg',
  '/static/pies/allfbsgraphices-15.svg',
  '/static/pies/allfbsgraphices-18.svg',
];

const floatingPies = [];
let mouseX = -1000;
let mouseY = -1000;

// Physics constants
const FRICTION = 0.995;
const REPEL_RADIUS = 100;
const REPEL_STRENGTH = 0.8;

function initFloatingPies() {
  const numPies = 6;
  const shuffled = [...pieFiles].sort(() => Math.random() - 0.5);

  // 6 pies with varying sizes (slightly larger)
  const sizes = [55, 75, 95, 115, 140, 170];

  for (let i = 0; i < numPies; i++) {
    const src = shuffled[i % shuffled.length];
    const img = document.createElement('img');
    img.src = src;
    img.className = 'floating-pie';

    // Random starting position
    const x = Math.random() * window.innerWidth;
    const y = Math.random() * window.innerHeight;

    // Faster drift
    const speed = 0.3 + Math.random() * 0.4;
    const angle = Math.random() * Math.PI * 2;
    const baseVx = Math.cos(angle) * speed;
    const baseVy = Math.sin(angle) * speed;

    // Get size from predefined distribution
    const size = sizes[i];

    // Slow rotation
    const rotationSpeed = (Math.random() - 0.5) * 0.2;

    // Opacity: smaller = more visible, bigger = more ghostly
    const opacity = 0.06 + (1 - size / 180) * 0.12;

    img.style.width = `${size}px`;
    img.style.height = `${size}px`;
    img.style.opacity = opacity;

    document.body.appendChild(img);

    floatingPies.push({
      el: img,
      x, y,
      vx: baseVx,
      vy: baseVy,
      baseVx, baseVy,
      rotation: Math.random() * 360,
      rotationSpeed,
      baseRotationSpeed: rotationSpeed,
      size,
      radius: size / 2,
      mass: size * size, // Mass scales with area
    });
  }

  document.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
  });

  requestAnimationFrame(animateFloatingPies);
}

function animateFloatingPies() {
  const w = window.innerWidth;
  const h = window.innerHeight;

  // Collision detection between all pie pairs
  for (let i = 0; i < floatingPies.length; i++) {
    const a = floatingPies[i];
    const ax = a.x + a.radius;
    const ay = a.y + a.radius;

    for (let j = i + 1; j < floatingPies.length; j++) {
      const b = floatingPies[j];
      const bx = b.x + b.radius;
      const by = b.y + b.radius;

      const dx = bx - ax;
      const dy = by - ay;
      const distSq = dx * dx + dy * dy;
      const minDist = a.radius + b.radius;

      if (distSq < minDist * minDist && distSq > 0) {
        const dist = Math.sqrt(distSq);
        const nx = dx / dist;
        const ny = dy / dist;

        // Separate them first (push apart)
        const overlap = minDist - dist;
        const totalMass = a.mass + b.mass;
        const aRatio = b.mass / totalMass;
        const bRatio = a.mass / totalMass;

        a.x -= overlap * nx * aRatio;
        a.y -= overlap * ny * aRatio;
        b.x += overlap * nx * bRatio;
        b.y += overlap * ny * bRatio;

        // Calculate bounce velocity
        const relVelX = a.vx - b.vx;
        const relVelY = a.vy - b.vy;
        const relVelDotNormal = relVelX * nx + relVelY * ny;

        // Only resolve if moving toward each other
        if (relVelDotNormal > 0) {
          const restitution = 0.7; // Bounciness
          const impulse = (1 + restitution) * relVelDotNormal / totalMass;

          a.vx -= impulse * b.mass * nx;
          a.vy -= impulse * b.mass * ny;
          b.vx += impulse * a.mass * nx;
          b.vy += impulse * a.mass * ny;

          // Spin on impact
          a.rotationSpeed += (Math.random() - 0.5) * 0.8;
          b.rotationSpeed += (Math.random() - 0.5) * 0.8;
        }
      }
    }
  }

  floatingPies.forEach(pie => {
    // Mouse repulsion
    const cx = pie.x + pie.radius;
    const cy = pie.y + pie.radius;
    const dx = cx - mouseX;
    const dy = cy - mouseY;
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist < REPEL_RADIUS && dist > 0) {
      const t = 1 - dist / REPEL_RADIUS;
      const force = t * t * REPEL_STRENGTH;
      pie.vx += (dx / dist) * force;
      pie.vy += (dy / dist) * force;
      pie.rotationSpeed += (Math.random() - 0.5) * 0.3;
    }

    // Friction
    pie.vx *= FRICTION;
    pie.vy *= FRICTION;

    // Drift back to base velocity
    pie.vx += (pie.baseVx - pie.vx) * 0.002;
    pie.vy += (pie.baseVy - pie.vy) * 0.002;

    // Return rotation to base
    pie.rotationSpeed += (pie.baseRotationSpeed - pie.rotationSpeed) * 0.005;

    // Update position
    pie.x += pie.vx;
    pie.y += pie.vy;
    pie.rotation += pie.rotationSpeed;

    // Wrap edges
    if (pie.x < -pie.size) pie.x = w;
    if (pie.x > w) pie.x = -pie.size;
    if (pie.y < -pie.size) pie.y = h;
    if (pie.y > h) pie.y = -pie.size;

    pie.el.style.transform = `translate(${pie.x}px, ${pie.y}px) rotate(${pie.rotation}deg)`;
  });

  requestAnimationFrame(animateFloatingPies);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', initFloatingPies);
