// Sonic Analyzer — Frontend logic
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
let enrichmentPct = 0;    // enrichment progress percentage

// -------------------------------------------------------
// Helpers
// -------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const show = (el) => el.classList.remove('hidden');
const hide = (el) => el.classList.add('hidden');

function scrollToRegister() {
  const el = $('#auth-section');
  if (el) el.scrollIntoView({ behavior: 'smooth' });
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
// Auth: Login / Signup / Session
// -------------------------------------------------------
function _authSuccess(data) {
  accessToken = data.token;
  localStorage.setItem('sc_token', data.token);
  localStorage.setItem('sc_name', data.name || '');
  _updateScans(data.scans_remaining);
  hide($('#auth-section'));
  hide($('#hero'));
  show($('#upload-section'));
  // Default to Spotify URL tab
  if (typeof setInputMode === 'function') setInputMode('url');
}

function _updateScans(remaining) {
  const badge = $('#scans-badge');
  if (!badge) return;
  if (remaining == null || remaining >= 999) {
    badge.classList.add('hidden');
    badge.textContent = '';
    return;
  }
  badge.classList.remove('hidden');
  if (remaining <= 0) {
    badge.innerHTML = '0 scans remaining — <a href="mailto:almgren@freshlybakedstudios.com" style="color:#4ecdc4">contact us</a>';
  } else {
    badge.textContent = `${remaining} scans remaining`;
  }
}

function _showView(viewId) {
  ['login-view', 'signup-view', 'forgot-view', 'reset-view'].forEach(id => {
    const el = $('#' + id);
    if (el) el.classList.toggle('hidden', id !== viewId);
  });
}

// View toggles
document.addEventListener('DOMContentLoaded', () => {
  const showSignup = $('#show-signup');
  const showLogin = $('#show-login');
  const showForgot = $('#show-forgot');
  const backToLogin = $('#back-to-login');
  if (showSignup) showSignup.addEventListener('click', (e) => { e.preventDefault(); _showView('signup-view'); });
  if (showLogin) showLogin.addEventListener('click', (e) => { e.preventDefault(); _showView('login-view'); });
  if (showForgot) showForgot.addEventListener('click', (e) => { e.preventDefault(); _showView('forgot-view'); });
  if (backToLogin) backToLogin.addEventListener('click', (e) => { e.preventDefault(); _showView('login-view'); });

  // NDA checkbox enables signup button
  const ndaBox = $('#nda-agreed');
  const signupBtn = $('#signup-btn');
  if (ndaBox && signupBtn) {
    ndaBox.addEventListener('change', () => {
      signupBtn.disabled = !ndaBox.checked;
      signupBtn.style.opacity = ndaBox.checked ? '1' : '0.5';
    });
  }

  // Logout
  const logoutBtn = $('#logout-btn');
  if (logoutBtn) logoutBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    if (accessToken) {
      const form = new FormData();
      form.append('token', accessToken);
      fetch(`${API_URL}/api/logout`, { method: 'POST', body: form }).catch(() => {});
    }
    accessToken = null;
    localStorage.removeItem('sc_token');
    localStorage.removeItem('sc_name');
    hide($('#upload-section'));
    hide($('#results-section'));
    show($('#auth-section'));
    show($('#hero'));
    _showView('login-view');
  });

  // Check for password reset token in URL
  const params = new URLSearchParams(window.location.search);
  const resetToken = params.get('reset');
  if (resetToken) {
    _showView('reset-view');
    // Store for form submission
    window._resetToken = resetToken;
  }

  // Check for saved session on page load
  const savedToken = localStorage.getItem('sc_token');
  if (savedToken) {
    fetch(`${API_URL}/api/me?token=${savedToken}`)
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(data => {
        accessToken = savedToken;
        _updateScans(data.scans_remaining);
        hide($('#auth-section'));
        hide($('#hero'));
        show($('#upload-section'));
        if (typeof setInputMode === 'function') setInputMode('url');
      })
      .catch(() => {
        localStorage.removeItem('sc_token');
        localStorage.removeItem('sc_name');
      });
  }
});

// Login form
$('#login-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = $('#login-btn');
  const email = $('#login-email').value.trim();
  const password = $('#login-password').value;
  if (!email || !password) return;

  btn.disabled = true;
  btn.textContent = 'Logging in...';
  try {
    const form = new FormData();
    form.append('email', email);
    form.append('password', password);
    const res = await fetch(`${API_URL}/api/login`, { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Login failed');
    }
    _authSuccess(await res.json());
  } catch (err) {
    alert(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Log In';
  }
});

// Signup form
$('#signup-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = $('#signup-btn');
  const name = $('#signup-name').value.trim();
  const email = $('#signup-email').value.trim();
  const password = $('#signup-password').value;
  const ndaAgreed = $('#nda-agreed').checked;
  if (!name || !email || !password || !ndaAgreed) return;

  btn.disabled = true;
  btn.textContent = 'Creating account...';
  try {
    const form = new FormData();
    form.append('name', name);
    form.append('email', email);
    form.append('password', password);
    form.append('nda_agreed', 'true');
    const spotifyUrl = $('#signup-spotify-url').value.trim();
    if (spotifyUrl) form.append('spotify_url', spotifyUrl);
    const listeners = $('#signup-listeners').value.trim();
    if (listeners) form.append('monthly_listeners', listeners);

    const res = await fetch(`${API_URL}/api/signup`, { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Signup failed');
    }
    const data = await res.json();

    // Track lead conversion
    if (typeof gtag === 'function') {
      gtag('event', 'generate_lead', { currency: 'USD', value: 1.00 });
    }
    if (typeof fbq === 'function') {
      fbq('track', 'Lead');
    }

    _authSuccess(data);
  } catch (err) {
    alert(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Create Account';
  }
});

// Forgot password form
$('#forgot-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const email = $('#forgot-email').value.trim();
  if (!email) return;
  const form = new FormData();
  form.append('email', email);
  await fetch(`${API_URL}/api/forgot-password`, { method: 'POST', body: form }).catch(() => {});
  hide($('#forgot-form'));
  show($('#forgot-success'));
});

// Reset password form
$('#reset-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const pw = $('#reset-password').value;
  const confirm = $('#reset-confirm').value;
  if (pw !== confirm) { alert('Passwords do not match.'); return; }
  if (pw.length < 8) { alert('Password must be at least 8 characters.'); return; }

  const form = new FormData();
  form.append('token', window._resetToken || '');
  form.append('new_password', pw);
  try {
    const res = await fetch(`${API_URL}/api/reset-password`, { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Reset failed');
    }
    alert('Password reset! You can now log in.');
    window.history.replaceState({}, '', window.location.pathname);
    _showView('login-view');
  } catch (err) {
    alert(err.message);
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
  if (file.size > 100 * 1024 * 1024) {
    alert('File too large (max 100 MB)');
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
    ? ['Fetching track from Spotify', 'Recording from Spotify desktop', 'Extracting audio features', 'Matching against 210,000+ tracks', 'Generating recommendations']
    : ['Extracting audio features', 'Analyzing frequency spectrum', 'Detecting emotional character', 'Matching against 210,000+ tracks', 'Generating recommendations'];
  let statusIdx = 0;
  const statusInterval = setInterval(() => {
    statusIdx = Math.min(statusIdx + 1, statuses.length - 1);
    const el = $('#loader-status');
    // Don't overwrite queue message
    if (el && !el.textContent.includes('in queue')) {
      el.textContent = statuses[statusIdx];
    }
  }, 4000);

  // Poll queue position during URL scans
  let queueInterval = null;
  if (inputMode === 'url') {
    queueInterval = setInterval(async () => {
      try {
        const qr = await fetch(`${API_URL}/api/queue-status`);
        if (qr.ok) {
          const qd = await qr.json();
          const el = $('#loader-status');
          if (qd.queue_length > 1 && el) {
            el.textContent = `You're #${qd.queue_length} in queue — another scan is in progress`;
          }
        }
      } catch {}
    }, 5000);
  }

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
      const artistUrl = ($('#upload-artist-url') && $('#upload-artist-url').value.trim()) || '';
      if (artistUrl) form.append('artist_spotify_url', artistUrl);
      res = await fetch(`${API_URL}/api/analyze`, { method: 'POST', body: form });
    }
    clearInterval(statusInterval);
    if (queueInterval) clearInterval(queueInterval);

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
    if (queueInterval) clearInterval(queueInterval);
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
    renderEditorialPlaylists(data.playlists);
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

  eventSource.addEventListener('campaign_forecast', (e) => {
    const data = JSON.parse(e.data);
    data._complete = true;
    window._lastCampaignForecast = data;  // for PDF export
    renderCampaignForecast(data);
  });

  eventSource.addEventListener('enrichment_progress', (e) => {
    const data = JSON.parse(e.data);
    enrichmentPct = data.total_batches ? Math.round(data.batch / data.total_batches * 100) : 0;
    let msg = `${enrichmentPct}% — ${data.curators_found} curators found`;
    if (data.checking) msg += ` · checking ${data.checking}`;
    updateEnrichmentProgress('playlists', msg);
    // Update live forecast from accumulated curators
    if (curatorRows.length > 0) {
      renderLiveForecast();
    }
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

  // Stash the full result so the PDF export can rebuild the metric cards
  // without recomputing anything (campaign forecast is stashed separately
  // when its SSE event arrives — see startSSE).
  window._lastAnalysisResult = data;
  const pdfBtn = $('#pdf-download-btn');
  if (pdfBtn) pdfBtn.classList.remove('hidden');

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
    const splitGenres = (s) => (s || '').split(',').map(x => x.trim()).filter(Boolean);
    const trackTags = splitGenres(src.track_genres);
    const artistTags = splitGenres(src.artist_genres);
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
    const escGenre = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const genreEl = $('#artist-card-genres');
    if (!trackTags.length && !artistTags.length) {
      genreEl.textContent = '-';
    } else {
      // Track genres (green chips) always visible — they're the song-specific
      // identity that drives the lane. Artist back-catalog tags (the soup,
      // often 10-20 items) collapse to first 3 + a "+N more" toggle so the
      // card doesn't get visually swamped.
      const COLLAPSED_ARTIST = 3;
      const trackChips = trackTags.map(g =>
        `<span class="genre-tag genre-tag-track" title="Track genre">${escGenre(g)}</span>`
      ).join('');
      const artistChipsHTML = (tags) => tags.map(g =>
        `<span class="genre-tag genre-tag-artist" title="Artist genre">${escGenre(g)}</span>`
      ).join('');
      let html = trackChips;
      if (artistTags.length <= COLLAPSED_ARTIST) {
        html += artistChipsHTML(artistTags);
      } else {
        const head = artistTags.slice(0, COLLAPSED_ARTIST);
        const tail = artistTags.slice(COLLAPSED_ARTIST);
        const more = tail.length;
        html +=
          artistChipsHTML(head) +
          `<span class="genre-tag genre-tag-rest" data-expanded="false" style="display:none;">${artistChipsHTML(tail)}</span>` +
          `<button type="button" class="genre-toggle" data-state="collapsed">+${more} more ▾</button>`;
      }
      genreEl.innerHTML = html;
      const toggle = genreEl.querySelector('.genre-toggle');
      if (toggle) {
        toggle.addEventListener('click', () => {
          const rest = genreEl.querySelector('.genre-tag-rest');
          const isCollapsed = toggle.dataset.state === 'collapsed';
          if (isCollapsed) {
            rest.style.display = 'inline';
            toggle.textContent = 'Hide ▴';
            toggle.dataset.state = 'expanded';
          } else {
            rest.style.display = 'none';
            const remaining = artistTags.length - COLLAPSED_ARTIST;
            toggle.textContent = `+${remaining} more ▾`;
            toggle.dataset.state = 'collapsed';
          }
        });
      }
    }
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

  // Track-level momentum panel — shows the scanned track's standing vs sonic peers
  // using popularity / CM score / playlist count, with a composite-percentile bar
  // styled like the artist-level one.
  const tm = userProfile && userProfile.track_momentum;
  const tmPanel = $('#track-momentum');
  if (tm && tmPanel) {
    function fmtNum(n) {
      if (n == null) return 'N/A';
      if (typeof n === 'number' && n > 0 && n < 1) return n.toFixed(2);
      if (typeof n === 'number' && n >= 1000) return n.toLocaleString(undefined, {maximumFractionDigits: 0});
      return Math.round(n).toLocaleString();
    }
    function pctLabel(pct) {
      // Plain-English bucketing for the percentile number on each row.
      if (pct == null) return 'no peer data';
      const p = Math.round(pct * 100);
      if (p >= 99) return `top 1%`;
      if (p >= 90) return `top 10%`;
      if (p >= 75) return `top 25%`;
      if (p >= 50) return `above average`;
      if (p >= 25) return `below average`;
      return `bottom 25%`;
    }

    // === Composite percentile bar (0–100 scale, fixed ticks) ===
    const compPct = Math.round((tm.composite_percentile || 0) * 100);
    $('#tm-bar-fill').style.width = compPct + '%';
    $('#tm-bar-dot').style.left = compPct + '%';
    // Build labels row
    const tmLabels = $('#tm-bar-labels');
    tmLabels.innerHTML = '';
    const labels = [
      { pos: 25, name: 'p25',   priority: 1 },
      { pos: 50, name: 'Median', priority: 2 },
      { pos: 75, name: 'Top 25%', priority: 4 },
      { pos: 99, name: 'Top 1%', priority: 3 },
      { pos: compPct, name: 'You', cls: 'conv-bar-label-you', priority: 5 },
    ];
    const MIN_GAP = 9;
    const sorted = [...labels].sort((a, b) => b.priority - a.priority);
    const kept = [];
    for (const l of sorted) {
      if (kept.every(k => Math.abs(k.pos - l.pos) >= MIN_GAP)) kept.push(l);
    }
    for (const l of kept) {
      const div = document.createElement('div');
      div.className = 'conv-bar-label ' + (l.cls || '');
      div.style.left = l.pos + '%';
      div.innerHTML = `<span>${l.name}</span><span>${l.pos}</span>`;
      tmLabels.appendChild(div);
    }

    // === Plain-English summary line ===
    let summary;
    if (compPct >= 90) summary = `Your track is in the <strong>top 10%</strong> of its sonic cohort — performing better than nearly every track that sounds like it.`;
    else if (compPct >= 75) summary = `Your track is in the <strong>top 25%</strong> of its sonic cohort — outperforming most tracks that sound like it.`;
    else if (compPct >= 50) summary = `Your track is <strong>above average</strong> for its sonic cohort — doing better than most similar-sounding tracks.`;
    else if (compPct >= 25) summary = `Your track is <strong>below average</strong> for its sonic cohort — there's headroom on the momentum side.`;
    else summary = `Your track is in the <strong>bottom 25%</strong> of its sonic cohort — the biggest lifts here are playlist pitching and Spotify popularity growth.`;
    $('#tm-summary').innerHTML = summary;

    // === Three metric rows (cleaner labels) ===
    const tmRows = $('#track-momentum-rows');
    tmRows.innerHTML = '';
    function row(label, sublabel, scanned, stats, pct) {
      const div = document.createElement('div');
      div.className = 'track-momentum-row';
      const valCls = (scanned == null) ? 'tm-value na' : 'tm-value';
      const pctStr = pct != null
        ? `<span class="tm-pct">${pctLabel(pct)}</span> of ${stats ? stats.count.toLocaleString() : '?'} sonic peers`
        : 'no comparable peer data';
      const peerStr = stats
        ? `<br>peer median ${fmtNum(stats.median)} · top 25% ${fmtNum(stats.p75)} · top 1% ${fmtNum(stats.p99)}`
        : '';
      div.innerHTML = `
        <span class="tm-label">${label}<small>${sublabel}</small></span>
        <span class="${valCls}">${scanned != null ? fmtNum(scanned) : 'N/A'}</span>
        <span class="tm-peers">${pctStr}${peerStr}</span>`;
      tmRows.appendChild(div);
    }
    row('Spotify Popularity', 'Spotify’s 0–100 recency-weighted score', tm.scanned_popularity, tm.pop_stats, tm.percentile_popularity);
    row('Chartmetric Score',  'Multi-platform composite (0–100)', tm.scanned_cm_score, tm.cm_stats, tm.percentile_cm_score);
    row('Playlist Placements', 'Editorial + user playlists combined', tm.scanned_playlists, tm.playlists_stats, tm.percentile_playlists);

    $('#tm-peer-count').textContent = `vs ${tm.peer_count.toLocaleString()} tracks that sound like yours`;

    // === Gap line: only shown for tracks BELOW top 25% of cohort ===
    const gapEl = $('#track-momentum-gap');
    if (tm.gap_additional_revenue && tm.gap_additional_revenue > 0 && tm.gap_target_listeners && compPct < 75) {
      const cur = (tm.gap_current_revenue || 0).toLocaleString();
      const tgt = (tm.gap_target_revenue || 0).toLocaleString();
      const tgtListeners = tm.gap_target_listeners.toLocaleString();
      const additional = tm.gap_additional_revenue.toLocaleString();
      gapEl.innerHTML = `<strong>What "closing the gap" looks like:</strong> tracks in the top 25% of this sonic cohort belong to artists with a median of <strong>${tgtListeners} monthly listeners</strong> — typically earning <strong>$${tgt}/year</strong> in Spotify streaming royalties (at $${tm.revenue_per_listener.toFixed(2)}/listener, Loud &amp; Clear 2025). You're currently at $${cur}/year. <strong>+$${additional}/year potential</strong> if your track reached that peer tier.
        <span class="gap-note">Peer-typical correlation from your actual sonic cohort — what artists with tracks at this level typically have. Not a personal forecast.</span>`;
      gapEl.style.display = 'block';
    } else {
      gapEl.style.display = 'none';
    }
    tmPanel.classList.remove('hidden');
  } else if (tmPanel) {
    tmPanel.classList.add('hidden');
  }

  // === Sonic Originality panel ===
  // Same data architecture as the track-momentum panel: composite bar (0-100),
  // headline summary, two row-based breakdowns. Different math — distance from
  // cohort centroid in z-space rather than peer percentile on momentum signals.
  const orig = userProfile && userProfile.sonic_originality;
  const quadrant = userProfile && userProfile.quadrant;
  const origCard = $('#sonic-originality-card');
  if (orig && origCard) {
    // Plain-English direction labels (mirrors the backend ORIGINALITY_DIRECTION_LABELS
    // for the deviation list — describes what direction the user sits relative to cohort).
    const FEATURE_DIRECTION_LABELS = {
      // Frequency spectrum
      sub_ratio:           {high: 'heavier sub-bass than',         low: 'lighter sub-bass than'},
      bass_ratio:          {high: 'heavier bass than',              low: 'lighter bass than'},
      low_mid_ratio:       {high: 'thicker low-mids than',          low: 'cleaner low-mids than'},
      mid_ratio:           {high: 'more forward mids than',         low: 'softer mids than'},
      high_mid_ratio:      {high: 'more presence / edge than',      low: 'softer upper-mids than'},
      presence_ratio:      {high: 'brighter presence than',         low: 'darker presence than'},
      air_ratio:           {high: 'more high-end air than',         low: 'less high-end air than'},
      // Brightness / spectral shape
      brightness:          {high: 'brighter spectral center than',  low: 'darker spectral center than'},
      spectral_rolloff:    {high: 'more high-frequency rolloff than', low: 'less high-frequency content than'},
      brightness_variance: {high: 'more brightness movement than',  low: 'flatter brightness curve than'},
      // Dynamics
      energy:              {high: 'higher energy than',             low: 'more restrained than'},
      dynamic_range:       {high: 'more dynamic contrast than',     low: 'flatter dynamics than'},
      loudness_range:      {high: 'wider loudness variation than',  low: 'tighter loudness than'},
      lufs_integrated:     {high: 'louder master than',             low: 'quieter master than'},
      compression_amount:  {high: 'more compressed than',           low: 'more open / less compressed than'},
      crest_factor:        {high: 'punchier peaks than',            low: 'flatter peaks than'},
      true_peak_dbfs:      {high: 'higher peak level than',         low: 'lower peak level than'},
      // Rhythm / transients
      beat_strength:       {high: 'stronger beat than',             low: 'softer beat than'},
      onset_rate:          {high: 'denser percussion than',         low: 'sparser percussion than'},
      attack_time:         {high: 'slower attacks than',            low: 'sharper attacks than'},
      danceability:        {high: 'more rhythmic pull than',        low: 'looser groove than'},
      // Tonal character
      spectral_complexity: {high: 'more spectral complexity than',  low: 'simpler spectrum than'},
      dissonance:          {high: 'more dissonant / edgy than',     low: 'more consonant / clean than'},
      key_strength:        {high: 'more tonally anchored than',     low: 'more tonally ambiguous than'},
      zcr:                 {high: 'brighter / noisier than',        low: 'mellower / cleaner than'},
      spectral_flux:       {high: 'more spectral movement than',    low: 'more static spectrum than'},
      harmonic_distortion: {high: 'more harmonic saturation than',  low: 'cleaner harmonics than'},
      // Stereo imaging
      stereo_width:        {high: 'wider stereo image than',        low: 'narrower stereo than'},
      mid_side_ratio:      {high: 'more side energy than',          low: 'more centered mix than'},
      stereo_correlation:  {high: 'more decorrelated stereo than',  low: 'more correlated stereo than'},
    };
    const FEATURE_PRETTY = {
      sub_ratio: 'Sub-bass',
      bass_ratio: 'Bass',
      low_mid_ratio: 'Low-mids',
      mid_ratio: 'Mids',
      high_mid_ratio: 'High-mids',
      presence_ratio: 'Presence',
      air_ratio: 'Air',
      brightness: 'Brightness (spectral centroid)',
      spectral_rolloff: 'Spectral rolloff',
      brightness_variance: 'Brightness movement',
      energy: 'Energy',
      dynamic_range: 'Dynamic range',
      loudness_range: 'Loudness range',
      lufs_integrated: 'Master loudness',
      compression_amount: 'Compression',
      crest_factor: 'Crest factor',
      true_peak_dbfs: 'True peak',
      beat_strength: 'Beat strength',
      onset_rate: 'Onset density',
      attack_time: 'Attack time',
      danceability: 'Danceability',
      spectral_complexity: 'Spectral complexity',
      dissonance: 'Dissonance',
      key_strength: 'Tonal anchoring',
      zcr: 'Brightness (ZCR)',
      spectral_flux: 'Spectral flux',
      harmonic_distortion: 'Harmonic distortion',
      stereo_width: 'Stereo width',
      mid_side_ratio: 'Mid/side ratio',
      stereo_correlation: 'Stereo correlation',
    };

    // Quadrant headline
    const qEl = $('#orig-quadrant');
    if (quadrant) {
      const qClass = quadrant.quadrant === 'signature_of_success' ? 'q-signature'
                   : quadrant.quadrant === 'stuck_in_pack'         ? 'q-stuck'
                   : '';
      qEl.className = 'orig-quadrant ' + qClass;
      qEl.innerHTML = `<div class="orig-quadrant-label">${quadrant.label}</div>
                       <div class="orig-quadrant-message">${quadrant.message}</div>`;
      qEl.style.display = 'block';
    } else {
      qEl.style.display = 'none';
    }

    // Composite originality bar (0–100)
    const origScore = orig.composite_score || 0;
    $('#orig-bar-fill').style.width = origScore + '%';
    $('#orig-bar-dot').style.left = origScore + '%';

    const origLabels = $('#orig-bar-labels');
    origLabels.innerHTML = '';
    const olabels = [
      { pos: 25, name: 'Low orig',  priority: 1 },
      { pos: 50, name: 'Typical',   priority: 2 },
      { pos: 75, name: 'Distinct',  priority: 3 },
      { pos: 99, name: 'Singular',  priority: 2 },
      { pos: origScore, name: 'You', cls: 'conv-bar-label-you', priority: 5 },
    ];
    const oMIN_GAP = 9;
    const oSorted = [...olabels].sort((a, b) => b.priority - a.priority);
    const oKept = [];
    for (const l of oSorted) {
      if (oKept.every(k => Math.abs(k.pos - l.pos) >= oMIN_GAP)) oKept.push(l);
    }
    for (const l of oKept) {
      const div = document.createElement('div');
      div.className = 'conv-bar-label ' + (l.cls || '');
      div.style.left = l.pos + '%';
      div.innerHTML = `<span>${l.name}</span><span>${l.pos}</span>`;
      origLabels.appendChild(div);
    }

    // Summary line
    let origSummary;
    if (origScore >= 75)      origSummary = `Your sound is in the <strong>top 25% of sonically distinct tracks</strong> within your cohort. The deviations below are your signature.`;
    else if (origScore >= 50) origSummary = `Your sound is <strong>moderately distinct</strong> from your sonic cohort — some signature features, mostly within consensus.`;
    else if (origScore >= 25) origSummary = `Your sound <strong>mostly follows cohort consensus</strong>. You're executing the genre playbook more than reinventing it.`;
    else                       origSummary = `Your sound <strong>closely matches cohort consensus</strong> on most dimensions. Strong commercial fit; low sonic differentiation.`;
    $('#orig-summary').innerHTML = origSummary;

    // Deviation rows ("Where your sound stands out")
    const devEl = $('#orig-deviations');
    devEl.innerHTML = '';
    const deviations = (orig.top_deviations || []);
    if (deviations.length === 0) {
      devEl.innerHTML = '<div class="orig-context" style="padding:10px 0">No strongly distinctive features — every dimension is within 1σ of your cohort consensus.</div>';
    } else {
      for (const d of deviations) {
        const labels = FEATURE_DIRECTION_LABELS[d.feature] || {high: 'higher than', low: 'lower than'};
        const pretty = FEATURE_PRETTY[d.feature] || d.feature;
        const dirLabel = d.direction === 'high' ? labels.high : labels.low;
        const zStr = (d.z > 0 ? '+' : '') + d.z.toFixed(2) + 'σ';
        const div = document.createElement('div');
        div.className = 'orig-row';
        div.innerHTML = `
          <span class="orig-label">${pretty}</span>
          <span class="orig-z">${zStr}</span>
          <span class="orig-context"><span class="orig-direction">${dirLabel}</span> cohort consensus &middot; you ${d.user_val} vs median ${d.cohort_mean}</span>`;
        devEl.appendChild(div);
      }
    }

    // Fits-consensus rows ("Where you match the consensus")
    const fitsEl = $('#orig-fits');
    fitsEl.innerHTML = '';
    const fits = (orig.fits_consensus || []);
    if (fits.length === 0) {
      fitsEl.innerHTML = '<div class="orig-context" style="padding:10px 0">No close-consensus dimensions — your sound deviates from cohort on every measured feature.</div>';
    } else {
      for (const f of fits) {
        const pretty = FEATURE_PRETTY[f.feature] || f.feature;
        const zStr = (f.z > 0 ? '+' : '') + f.z.toFixed(2) + 'σ';
        const div = document.createElement('div');
        div.className = 'orig-row';
        div.innerHTML = `
          <span class="orig-label">${pretty}</span>
          <span class="orig-z fits">${zStr}</span>
          <span class="orig-context">matches cohort &middot; you ${f.user_val} vs median ${f.cohort_mean}</span>`;
        fitsEl.appendChild(div);
      }
    }

    origCard.classList.remove('hidden');
  } else if (origCard) {
    origCard.classList.add('hidden');
  }

  // === Sonic Quadrant scatter plot ===
  // 2D plot: X = performance percentile, Y = originality score. Quadrant cuts
  // at 75/75. User's dot prominent; top pitch comparables labeled.
  const sqCard = $('#sonic-quadrant-card');
  const sqWrap = $('#sonic-quadrant-svg-wrap');
  const sqTm = userProfile && userProfile.track_momentum;
  const sqOrig = userProfile && userProfile.sonic_originality;
  const sqPitch = userProfile && userProfile.pitch_comparables;
  if (sqCard && sqWrap && sqTm && sqOrig) {
    // Coordinates: 0-100 on both axes
    const userPerf = Math.round((sqTm.composite_percentile || 0) * 100);
    const userOrig = sqOrig.composite_score || 0;

    // SVG geometry (responsive viewBox)
    const W = 640, H = 460;
    const M_L = 60, M_R = 30, M_T = 30, M_B = 50;
    const innerW = W - M_L - M_R;
    const innerH = H - M_T - M_B;
    const xToPx = v => M_L + (v / 100) * innerW;
    const yToPx = v => M_T + ((100 - v) / 100) * innerH;  // invert: high orig = top

    let svg = `<svg class="sq-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">`;

    // Background grid every 25
    for (let g = 25; g < 100; g += 25) {
      svg += `<line class="sq-bg-grid" x1="${xToPx(g)}" y1="${M_T}" x2="${xToPx(g)}" y2="${M_T+innerH}"/>`;
      svg += `<line class="sq-bg-grid" x1="${M_L}" y1="${yToPx(g)}" x2="${M_L+innerW}" y2="${yToPx(g)}"/>`;
    }

    // Quadrant cut lines at 75/75
    svg += `<line class="sq-quad-line" x1="${xToPx(75)}" y1="${M_T}" x2="${xToPx(75)}" y2="${M_T+innerH}"/>`;
    svg += `<line class="sq-quad-line" x1="${M_L}" y1="${yToPx(75)}" x2="${M_L+innerW}" y2="${yToPx(75)}"/>`;

    // Quadrant corner labels
    svg += `<text class="sq-quad-label" x="${xToPx(50)}" y="${yToPx(95)}" text-anchor="middle">Ahead of the Curve</text>`;
    svg += `<text class="sq-quad-label success" x="${xToPx(88)}" y="${yToPx(95)}" text-anchor="middle">Signature of Success</text>`;
    svg += `<text class="sq-quad-label stuck" x="${xToPx(50)}" y="${yToPx(8)}" text-anchor="middle">Stuck in the Pack</text>`;
    svg += `<text class="sq-quad-label" x="${xToPx(88)}" y="${yToPx(8)}" text-anchor="middle">Genre-Playbook Winner</text>`;

    // Axes
    svg += `<line class="sq-axis" x1="${M_L}" y1="${M_T+innerH}" x2="${M_L+innerW}" y2="${M_T+innerH}"/>`;
    svg += `<line class="sq-axis" x1="${M_L}" y1="${M_T}" x2="${M_L}" y2="${M_T+innerH}"/>`;
    // Tick labels
    for (let t = 0; t <= 100; t += 25) {
      svg += `<text class="sq-axis-label" x="${xToPx(t)}" y="${M_T+innerH+18}" text-anchor="middle">${t}</text>`;
      svg += `<text class="sq-axis-label" x="${M_L-10}" y="${yToPx(t)+4}" text-anchor="end">${t}</text>`;
    }
    // Axis titles
    svg += `<text class="sq-axis-title" x="${M_L+innerW/2}" y="${H-10}" text-anchor="middle">Performance percentile →</text>`;
    svg += `<text class="sq-axis-title" x="${15}" y="${M_T+innerH/2}" text-anchor="middle" transform="rotate(-90 15 ${M_T+innerH/2})">Originality score →</text>`;

    // Cohort cloud — every same-tier peer plotted as a dim small dot.
    // Exclude the pitch-comparables (we'll draw those labeled on top).
    const pitchNames = new Set((sqPitch || []).slice(0, 5).map(p => p.name));
    const sqCloud = userProfile && userProfile.cohort_scatter;
    if (sqCloud && sqCloud.length > 0) {
      sqCloud.forEach(p => {
        if (pitchNames.has(p.name)) return;  // skip — drawn labeled below
        const px = xToPx(Math.round((p.perf_pct || 0) * 100));
        const py = yToPx(p.orig_score || 0);
        svg += `<circle class="sq-cloud-dot" cx="${px}" cy="${py}" r="2.5"/>`;
      });
    }

    // Pitch comparables as labeled dots (drawn over the cloud)
    if (sqPitch && sqPitch.length > 0) {
      sqPitch.slice(0, 5).forEach((p, i) => {
        const px = xToPx(Math.round((p.perf_pct || 0) * 100));
        const py = yToPx(p.orig_score || 0);
        svg += `<circle class="sq-peer-dot" cx="${px}" cy="${py}" r="5"/>`;
        // Offset label to avoid dot overlap
        const labelOffsetX = (i % 2 === 0) ? 8 : -8;
        const anchor = (i % 2 === 0) ? 'start' : 'end';
        svg += `<text class="sq-peer-label" x="${px+labelOffsetX}" y="${py+3}" text-anchor="${anchor}">${(p.name || '').replace(/[<>&]/g,'')}</text>`;
      });
    }

    // User dot (drawn last so it's on top)
    const ux = xToPx(userPerf);
    const uy = yToPx(userOrig);
    svg += `<circle class="sq-user-dot" cx="${ux}" cy="${uy}" r="10"/>`;
    svg += `<text class="sq-user-label" x="${ux+14}" y="${uy+4}" text-anchor="start">You (${userPerf}, ${userOrig})</text>`;

    svg += `</svg>`;
    sqWrap.innerHTML = svg;
    sqCard.classList.remove('hidden');
  } else if (sqCard) {
    sqCard.classList.add('hidden');
  }

  // === A&R Pitch Comparables panel ===
  // Top 5 same-tier sonic peers in Signature of Success (high perf + high originality).
  const pitchList = userProfile && userProfile.pitch_comparables;
  const pitchCard = $('#pitch-comparables-card');
  if (pitchList && pitchList.length > 0 && pitchCard) {
    const pitchRows = $('#pitch-rows');
    pitchRows.innerHTML = '';
    function fmtListeners(n) {
      if (!n) return '—';
      if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M monthly listeners';
      if (n >= 1000) return (n / 1000).toFixed(0) + 'K monthly listeners';
      return n.toLocaleString() + ' monthly listeners';
    }
    pitchList.forEach((p, i) => {
      const div = document.createElement('div');
      div.className = 'pitch-row';
      // Link to the specific TRACK that scored — not the artist profile.
      // A&R wants to hear the cut that proves the comparable, not just the artist.
      const linkTarget = p.track_url || p.spotify_url;
      const linkLabel = p.track_name ? `${p.name} — ${p.track_name}` : p.name;
      const nameLink = linkTarget
        ? `<a href="${linkTarget}" target="_blank" rel="noopener">${linkLabel}</a>`
        : linkLabel;
      const simPct = Math.round((p.similarity || 0) * 100);
      const perfPct = Math.round((p.perf_pct || 0) * 100);
      const cmStr = (p.cm_track_score != null) ? Math.round(p.cm_track_score) : 'N/A';
      // Generate a plain-English pitch angle
      let angle;
      if (p.orig_score >= 80 && perfPct >= 80) {
        angle = `Distinctive sonic profile AND scaled performance — clean proof your sound rewards distinctiveness in this lane.`;
      } else if (p.orig_score >= 80) {
        angle = `Strong sonic distinctiveness for the tier — proof your kind of sonic edge has commercial traction.`;
      } else if (perfPct >= 80) {
        angle = `Top performer in your sonic neighborhood — a comparable on numbers, not just sound.`;
      } else {
        angle = `Sits in your sonic lane with both distinctiveness and traction above the cohort floor.`;
      }
      div.innerHTML = `
        <div class="pitch-name">${i + 1}. ${nameLink}</div>
        <div class="pitch-listeners">${fmtListeners(p.listeners)}</div>
        <div class="pitch-stats">
          <span class="pitch-stat-pill">${simPct}% sonic match</span>
          <span class="pitch-stat-pill">Popularity ${p.sp_track_popularity ?? '—'}</span>
          <span class="pitch-stat-pill">CM ${cmStr}</span>
          <span class="pitch-stat-pill">${p.playlists_total.toLocaleString()} playlists</span>
          <span class="pitch-stat-pill">Originality ${p.orig_score}</span>
        </div>
        <div class="pitch-angle">${angle}</div>`;
      pitchRows.appendChild(div);
    });
    pitchCard.classList.remove('hidden');
  } else if (pitchCard) {
    pitchCard.classList.add('hidden');
  }

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

    // Fan label on the gap — suppressed when track-momentum panel is rendering
    // (the new panel's gap line covers this story more honestly).
    const fanLabel = $('#conv-fan-label');
    if (hasUserRate && fans > 0 && target > cr && !tm) {
      fanLabel.textContent = '+' + fans.toLocaleString() + ' fans';
      fanLabel.style.marginLeft = toPos(cr) + '%';
      fanLabel.style.width = (toPos(target) - toPos(cr)) + '%';
      show(fanLabel);
    } else {
      hide(fanLabel);
    }

    // Labels below bar — collision-aware (higher priority labels kept).
    // Plain-English bucket names to match the new track-momentum panel.
    const labelsEl = $('#conv-bar-labels');
    labelsEl.innerHTML = '';
    const MIN_GAP = 8;
    const allLabels = [
      { pos: toPos(p25), name: 'Bottom 25%', val: p25.toFixed(2) + '%', cls: '', priority: 1 },
      { pos: toPos(median), name: 'Median', val: median.toFixed(2) + '%', cls: '', priority: 2 },
      { pos: toPos(p75), name: 'Top 25%', val: p75.toFixed(2) + '%', cls: '', priority: 1 },
    ];
    if (hasUserRate) {
      allLabels.push({ pos: toPos(cr), name: 'You', val: cr.toFixed(2) + '%', cls: 'conv-bar-label-you', priority: 5 });
    }
    if (target > cr) {
      const targetName = atTop ? 'Top 1%' : 'Top 25%';
      allLabels.push({ pos: toPos(target), name: targetName, val: target.toFixed(2) + '%', cls: 'conv-bar-label-target', priority: 4 });
    }
    if (!atTop || target !== p99) {
      allLabels.push({ pos: toPos(p99), name: 'Top 1%', val: p99.toFixed(2) + '%', cls: '', priority: 1 });
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
      const peerCountRaw = comp.peer_count || 0;
      const poolTotalRaw = comp.peer_pool_total || peerCountRaw;
      const peerCount = peerCountRaw.toLocaleString();
      const peerScope = (poolTotalRaw > peerCountRaw)
        ? `${peerCount} of ${poolTotalRaw.toLocaleString()} sonic peers (those with conversion data)`
        : `${peerCount} sonic peers`;

      // Sample-size guard: below ~500 monthly listeners, conversion ratio is
      // statistically noisy (likely reflects friends/family/early supporters,
      // not commercial conversion behavior). Prepend a warning if applicable.
      const SMALL_SAMPLE_THRESHOLD = 500;
      const isSmallSample = listeners > 0 && listeners < SMALL_SAMPLE_THRESHOLD;
      const smallSampleNote = isSmallSample
        ? `<span class="small-sample-note"><strong>Small sample:</strong> with only ${listeners.toLocaleString()} monthly listeners, this ratio reflects a tiny audience (often friends, family, or early supporters). Take the peer comparison with a grain of salt and re-check once your listener count grows.</span> `
        : '';

      // Prefer raw followers/listener ratio with documented bucketing when available.
      // Source: Chartlex / Chartmetric — see memory/reference_conversion_metrics_research.md.
      // Falls back to the legacy "X% conversion" copy for backward compat.
      const fl_ratio = userProfile.fol_listener_ratio;
      const bucket = userProfile.retention_bucket;
      if (fl_ratio != null && bucket) {
        let bucketMsg;
        const ratioStr = `<span class="fan-number">${fl_ratio.toFixed(2)} followers per monthly listener</span>`;
        if (bucket === 'healthy') {
          bucketMsg = `Your ratio of ${ratioStr} sits in the healthy retention band (0.1–1.0 per Chartlex / Chartmetric benchmarks). Above 0.1 correlates with 2–3× more Spotify Release Radar placements — Spotify's algorithm treats your audience as engaged, not just casual.`;
        } else if (bucket === 'marginal') {
          bucketMsg = `Your ratio of ${ratioStr} is in the marginal band — close to the threshold where retention becomes a concern. Healthy is above 0.1; below 0.067 indicates audience width without depth. Tightening follower-conversion CTAs in your release notes and bio is the lever here.`;
        } else if (bucket === 'shallow') {
          bucketMsg = `Your ratio of ${ratioStr} is below the shallow-audience threshold (0.067 per Chartlex). Your monthly listener count is growing faster than fan retention — a "width without depth" pattern. The next milestone is converting more listeners into followers via release calls-to-action and Spotify Profile pinned releases.`;
        } else { // stale_or_superstar
          bucketMsg = `Your ratio of ${ratioStr} is unusually high — typically seen on superstar accounts with massive cumulative followers, or on accounts whose monthly listener count has dropped over time while the follower count carried over. At this ratio your audience math isn't directly comparable to peers at your current listener scale.`;
        }
        oppEl.innerHTML = smallSampleNote + bucketMsg;
      } else if (fans > 0 && atTop) {
        oppEl.innerHTML = smallSampleNote + `You're already in the <span class="fan-number">top 25%</span> of ${peerScope}. Reaching the top 1% (${target.toFixed(1)}%) would convert an estimated <span class="fan-number">${fans.toLocaleString()} additional listeners into followers</span> — each one a direct line to your releases, merch drops, and tour dates via Spotify push notifications.`;
      } else if (fans > 0) {
        const monthlyFans = Math.round(fans / 12);
        oppEl.innerHTML = smallSampleNote + `Across ${peerScope}, the top 25% convert at <span class="fan-number">${target.toFixed(1)}%</span>. Closing that gap means an estimated <span class="fan-number">${fans.toLocaleString()} new followers/year</span> (~${monthlyFans.toLocaleString()}/month) — each one receiving push notifications for your releases, merch drops, and tour dates.`;
      } else {
        // No-additional-fans branch — the "exceptional" message. With small
        // samples this is almost always misleading, so the warning matters most here.
        if (isSmallSample) {
          oppEl.innerHTML = smallSampleNote + `You're converting at <span class="fan-number">${cr.toFixed(1)}%</span> — formally above the top 1% of ${peerScope}, but with this listener count the ratio isn't statistically comparable to peers at scale.`;
        } else {
          oppEl.innerHTML = `You're converting at <span class="fan-number">${cr.toFixed(1)}%</span> — above the top 1% of ${peerScope}. Your listener-to-follower conversion is exceptional.`;
        }
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

        // Show the "with gap closed" projection ONLY when there's no track-momentum
        // gap line already doing this job. The new track-level panel projects gap
        // revenue via empirical peer-typical artist listeners at composite p75 —
        // that's a cleaner, less inflated number than the old artist-level
        // additional_fans projection (which was built on the made-up conversion
        // formula). When the new panel is present, just show current revenue.
        const hasTrackMomentumGap = !!(tm && tm.gap_additional_revenue
                                       && tm.gap_additional_revenue > 0
                                       && tm.gap_target_listeners
                                       && Math.round((tm.composite_percentile || 0) * 100) < 75);
        const suppressOldGap = !!tm;  // any track_momentum present → use the new panel's gap math

        if (fans > 0 && !suppressOldGap) {
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
        } else if (fans > 0 && suppressOldGap) {
          earningsEl.innerHTML = `
            <div class="conv-tier-row">
              <span class="conv-tier-label">Current (${fmtNum(listeners)} listeners)</span>
              <span class="conv-tier-val">${fmtRev(currentRev)}/year</span>
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

        // Estimated save rate range
        const saveLow = userProfile.estimated_save_rate_low;
        const saveHigh = userProfile.estimated_save_rate_high;
        if (saveLow && saveHigh) {
          const saveEl = $('#conv-save-rate');
          if (saveEl) {
            // Use midpoint for threshold classification
            const saveMid = (saveLow + saveHigh) / 2;
            let algoStatus, algoClass;
            if (saveLow >= 12) { algoStatus = 'Algorithmic placement likely within 10-14 days'; algoClass = 'save-excellent'; }
            else if (saveLow >= 8) { algoStatus = 'Strong algorithmic distribution expected'; algoClass = 'save-strong'; }
            else if (saveLow >= 5) { algoStatus = 'Above algorithmic threshold — 40% more Discover Weekly placements'; algoClass = 'save-good'; }
            else if (saveHigh >= 5) { algoStatus = 'Near algorithmic threshold — playlist seeding could push you over'; algoClass = 'save-average'; }
            else { algoStatus = 'Below algorithmic threshold — playlist seeding recommended'; algoClass = 'save-low'; }

            saveEl.innerHTML = `
              <div class="save-rate-display">
                <div class="save-rate-value ${algoClass}">${saveLow}–${saveHigh}%</div>
                <div class="save-rate-label">Estimated Save Rate</div>
                <div class="save-rate-status">${algoStatus}</div>
                <div class="save-rate-note">Your listeners convert to followers at ${userProfile.conversion_rate ? userProfile.conversion_rate.toFixed(2) + '%' : '—'} — artists at this level typically see ${saveLow}–${saveHigh}% of listeners saving their tracks. Above 5% is where Spotify's algorithm starts recommending you to new listeners. (Industry benchmarks: Playlist Push, LoudLab, Chartlex.) Connect Spotify for Artists for your actual save rate.</div>
              </div>
            `;
            show(saveEl);
          }
        }
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

  // Production Recommendations — amalgamated target ranges (cohort edge ↔
  // signature edge). Falls back to the legacy string list if the backend
  // didn't send structured ranges.
  const recRanges = data.recommendation_ranges || [];
  const recRangesWrap = $('#rec-ranges');
  const recList = $('#rec-list');
  if (recRangesWrap && recRanges.length > 0) {
    renderRecRanges(recRanges);
    show(recRangesWrap);
    if (recList) hide(recList);
  } else if (recList) {
    if (recRangesWrap) hide(recRangesWrap);
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
    show(recList);
  }

  // Legacy signature card removed — its data is now the upper edge of each
  // range above. Guard in case a cached page still has the element.
  const sigCard = $('#signature-rec-card');
  if (sigCard) sigCard.classList.add('hidden');
}

// Format a production-feature value for display, by unit kind from the backend.
function fmtFeatVal(kind, v) {
  if (v == null || isNaN(v)) return '–';
  switch (kind) {
    case 'pct':  return (v * 100).toFixed(1) + '%';
    case 'db':   return v.toFixed(1) + ' dB';
    case 'lufs': return v.toFixed(1) + ' LUFS';
    case 'hz':   return Math.round(v) + ' Hz';
    case 'rate': return v.toFixed(1) + ' /s';
    case 'ms':   return v.toFixed(1) + ' ms';
    case 'lu':   return v.toFixed(1) + ' LU';
    default:     return v.toFixed(3);
  }
}

// The real-world move from `you` to `target`, in the feature's natural unit.
// EQ band ratios read as dB (10·log10 ratio), matching how engineers think.
function fmtMove(kind, you, target) {
  if (you == null || target == null || isNaN(you) || isNaN(target)) return '';
  const s = (v, dp, unit) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(dp) + unit;
  switch (kind) {
    case 'pct':
      if (you > 0 && target > 0) return s(10 * Math.log10(target / you), 1, ' dB');
      return s((target - you) * 100, 1, ' pts');
    case 'db':
    case 'lufs': return s(target - you, 1, ' dB');
    case 'lu':   return s(target - you, 1, ' LU');
    case 'hz':   return (target >= you ? '+' : '−') + Math.round(Math.abs(target - you)) + ' Hz';
    case 'rate': return s(target - you, 1, ' /s');
    case 'ms':   return s(target - you, 1, ' ms');
    default:     return you !== 0 ? s(((target - you) / Math.abs(you)) * 100, 0, '%') : '';
  }
}

// Compact range label, unit shown once: "17.1–19.0%", "18.5–21.0 dB".
function fmtRange(kind, a, b) {
  const A = fmtFeatVal(kind, a), B = fmtFeatVal(kind, b);
  const m = A.match(/^([\d.\-−]+)(.*)$/);
  if (m && B.endsWith(m[2])) return m[1] + '–' + B; // strip unit off the low end
  return A + '–' + B;
}

// Resolve the percentile band for a rec (with a synthetic fallback) and whether
// the artist already sits inside the p25–p75 "winners" zone.
// True when the move to the nearest zone edge rounds to ~0 in display units —
// i.e. you're effectively at the edge already. Mirrors fmtMove's units so a row
// that would read "−0.0 /s" or "−1%" counts as nailed, not an Adjustment.
function recMoveNegligible(kind, you, edge) {
  if (you == null || edge == null || isNaN(you) || isNaN(edge)) return false;
  switch (kind) {
    case 'pct':
      return (you > 0 && edge > 0) ? Math.abs(10 * Math.log10(edge / you)) < 0.05
                                   : Math.abs((edge - you) * 100) < 0.05;
    case 'db': case 'lufs': case 'lu': case 'rate': case 'ms':
      return Math.abs(edge - you) < 0.05;
    case 'hz':
      return Math.abs(edge - you) < 0.5;
    default: // raw 0-1 metric shown as % — treat <1.5% off as cooked
      return you !== 0 && Math.abs((edge - you) / Math.abs(you)) * 100 < 1.5;
  }
}

function recBand(r) {
  const you = r.you;
  let p5, p25, p50, p75, p95;
  if (r.percentiles) {
    ({ p5, p25, p50, p75, p95 } = r.percentiles);
  } else {
    const t = r.target_cohort;
    const sp = Math.abs(t - you) || (Math.abs(t) * 0.1 + 1e-6);
    p25 = t - sp * 0.15; p75 = t + sp * 0.15; p50 = t;
    p5 = t - sp * 0.6; p95 = t + sp * 0.6;
  }
  let inZone = you >= p25 && you <= p75;
  // A hair outside the zone (move rounds to ~0) is effectively nailed — fold it
  // into the zone so it shows "✓ in the zone" and lands under "already nailing".
  if (!inZone) {
    const edge = you < p25 ? p25 : p75;
    if (recMoveNegligible(r.unit_kind, you, edge)) inZone = true;
  }
  return { p5, p25, p50, p75, p95, inZone };
}

// One meter row. The bar is anchored to the peer band (p5–p95); if You falls
// outside it, the dot clamps to the edge with an arrow pointing into the zone —
// so a big move reads as "head this way," not "stranded off the chart." The
// headline carries the real magnitude (e.g. "+31% to land in").
function recRangeRow(r, band) {
  const you = r.you, kind = r.unit_kind;
  const { p5, p25, p50, p75, p95, inZone } = band;

  const pad = ((p95 - p5) || (Math.abs(p95) * 0.1 + 1e-6)) * 0.06;
  const sMin = p5 - pad, sMax = p95 + pad;
  const rawPos = v => ((v - sMin) / (sMax - sMin)) * 100;
  const pos = v => Math.max(0, Math.min(100, rawPos(v)));

  const zoneL = pos(p25), zoneW = Math.max(pos(p75) - zoneL, 1.5);
  const offLow = you < p5, offHigh = you > p95;
  const dotPos = Math.max(2, Math.min(98, pos(you)));
  const dotCls = offLow ? ' off-low' : offHigh ? ' off-high' : '';
  const edge = you < p25 ? p25 : p75;
  const moveStr = inZone ? '✓ in the zone' : fmtMove(kind, you, edge) + ' to land in';

  return (
    `<div class="rec-range">` +
      `<div class="rec-range-head">` +
        `<span class="rec-domain">${r.domain}</span>` +
        `<span class="rec-act">${r.action}</span>` +
        `<span class="rec-move${inZone ? ' inrange' : ''}">${moveStr}</span>` +
      `</div>` +
      `<div class="rec-range-bar">` +
        `<div class="rec-range-band" style="left:${zoneL}%;width:${zoneW}%"></div>` +
        `<div class="rec-range-edge median" style="left:${pos(p50)}%"></div>` +
        `<div class="rec-range-dot${dotCls}" style="left:${dotPos}%"></div>` +
      `</div>` +
      `<div class="rec-range-legend">` +
        `<span class="rrl you">You <b>${fmtFeatVal(kind, you)}</b></span>` +
        `<span class="rrl zone">Target zone <b>${fmtRange(kind, p25, p75)}</b></span>` +
        `<span class="rrl agree">${r.agree[0]}/${r.agree[1]} agree</span>` +
      `</div>` +
    `</div>`
  );
}

// Split recs into "Adjustments" (you sit outside the winners' zone) and
// "What you're already nailing" (you're inside it) — an in-zone feature isn't a
// fix, it's a strength, so we celebrate it rather than list it as a to-do.
function renderRecRanges(ranges) {
  const wrap = $('#rec-ranges');
  if (!wrap) return;

  const adjust = [], strengths = [];
  ranges.forEach(r => {
    const band = recBand(r);
    (band.inZone ? strengths : adjust).push(recRangeRow(r, band));
  });

  let html = '';
  if (adjust.length) {
    html += `<div class="rec-group-label">Adjustments to make</div>` + adjust.join('');
  }
  if (strengths.length) {
    html += `<div class="rec-group-label strengths">✓ What you're already nailing</div>` + strengths.join('');
  }
  wrap.innerHTML = html || `<div class="rec-group-label">No strong consensus from your peer cohort.</div>`;
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
      const hidden = Math.max(totalAllMatches - totalCount, 0);
      const hiddenNote = hidden > 0
        ? ` · ${hidden.toLocaleString()} hidden (missing genre data)`
        : '';
      label = `Showing ${Math.min(matchesShown, totalCount)} of ${totalCount.toLocaleString()} matches${hiddenNote}`;
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
    matchRow.addEventListener('click', (e) => {
      // Don't toggle playlists when clicking the Spotify link
      if (e.target.closest('a')) return;
      // Collapse any other open playlist rows first
      document.querySelectorAll('tr.playlist-row:not(.hidden)').forEach(openRow => {
        if (openRow !== plRow) openRow.classList.add('hidden');
      });
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
  if (countEl) countEl.textContent = `${total} unique playlists found — ${recentCount} active in last 90 days. Click to expand.`;

  // Show count badge on collapsed header
  const badge = $('#playlist-count-badge');
  if (badge) badge.textContent = `(${total})`;

  // Wire up collapse toggle
  const toggle = $('#playlists-toggle');
  const collapsible = $('#playlists-collapsible');
  if (toggle && collapsible && !toggle._wired) {
    toggle._wired = true;
    toggle.style.cursor = 'pointer';
    toggle.addEventListener('click', (e) => {
      if (e.target.closest('.csv-download-btn')) return; // don't toggle on CSV btn click
      const arrow = toggle.querySelector('.collapse-arrow');
      if (collapsible.classList.contains('hidden')) {
        collapsible.classList.remove('hidden');
        if (arrow) arrow.innerHTML = '&#9660;';
      } else {
        collapsible.classList.add('hidden');
        if (arrow) arrow.innerHTML = '&#9654;';
      }
    });
  }

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

function renderEditorialPlaylists(allPlaylists) {
  const card = $('#editorial-playlists-card');
  if (!card) return;

  const editorial = allPlaylists.filter(pl => pl.editorial);
  if (!editorial.length) return;

  show(card);
  const sorted = editorial.sort((a, b) => (b.followers || 0) - (a.followers || 0));

  card.innerHTML = `
    <h3>Spotify Editorial Playlists <span class="playlist-count-badge">(${sorted.length})</span></h3>
    <p class="card-sub">Editorial playlists your sonic peers appear on — use these for your Spotify for Artists pitch.</p>
    <div class="editorial-list">
      ${sorted.map(pl => {
        const freshDate = pl.added_at || pl.last_updated || '';
        const isRecent = _isRecentlyActive(freshDate, 90);
        return `
        <div class="editorial-item ${isRecent ? 'recently-active' : ''}">
          <a href="${pl.link}" target="_blank" rel="noopener">${pl.name}</a>
          <span class="playlist-followers">${(pl.followers || 0).toLocaleString()} followers</span>
          <span class="editorial-via">via ${pl.sonic_match || 'matched artist'}</span>
          ${pl.status === 'current' ? '<span class="playlist-badge current-badge">Current</span>' : ''}
          ${isRecent ? '<span class="playlist-badge active-badge">Active</span>' : ''}
          ${freshDate ? `<span class="playlist-freshness">${_formatFreshness(freshDate)}</span>` : ''}
        </div>`;
      }).join('')}
    </div>
  `;
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

function renderLiveForecast() {
  // Build a running forecast from accumulated curators
  if (!curatorRows.length) return;

  let totalReach = 0, totalStreams = 0, totalCost = 0;
  const costByMethod = {};

  // Per-curator stream potential (followers × 3% stream rate)
  const perCuratorStreams = [];
  curatorRows.forEach(c => {
    const followers = c.followers || 0;
    let method = 'Email', cost = 0;
    if (c.groover_url) { method = 'Groover'; cost = 2; }
    else if (c.submithub_url) { method = 'SubmitHub'; cost = 2; }

    totalReach += followers;
    perCuratorStreams.push(followers * 0.03);
    totalCost += cost;

    if (!costByMethod[method]) costByMethod[method] = { count: 0, cost: 0 };
    costByMethod[method].count++;
    costByMethod[method].cost += cost;
  });

  const totalAcceptance = curatorRows.reduce((sum, c) => {
    if (c.groover_url) return sum + 0.20;
    if (c.submithub_url) return sum + 0.12;
    return sum + 0.065;
  }, 0);

  const placementsLow = Math.max(1, Math.round(totalAcceptance * 0.7));
  const placementsHigh = Math.max(1, Math.round(totalAcceptance * 1.3));

  // Best case: top N placements, worst case: bottom N
  const sortedDesc = [...perCuratorStreams].sort((a, b) => b - a);
  const sortedAsc = [...perCuratorStreams].sort((a, b) => a - b);
  let streamsHigh = Math.round(sortedDesc.slice(0, placementsHigh).reduce((a, b) => a + b, 0));
  let streamsLow = Math.round(sortedAsc.slice(0, placementsLow).reduce((a, b) => a + b, 0));

  // Ensure low <= high
  if (streamsLow > streamsHigh) [streamsLow, streamsHigh] = [streamsHigh, streamsLow];

  let algoLow = Math.round(streamsLow * 1.5);
  let algoHigh = Math.round(streamsHigh * 3.0);
  let totalStreamsLow = streamsLow + algoLow;
  let totalStreamsHigh = streamsHigh + algoHigh;
  let followersLow = Math.round(totalStreamsLow * 0.001);
  let followersHigh = Math.round(totalStreamsHigh * 0.001);
  let revLow = totalStreamsLow * 0.004;
  let revHigh = totalStreamsHigh * 0.004;

  // Ensure all low <= high
  if (followersLow > followersHigh) [followersLow, followersHigh] = [followersHigh, followersLow];
  if (revLow > revHigh) [revLow, revHigh] = [revHigh, revLow];

  renderCampaignForecast({
    curator_count: curatorRows.length,
    total_reach: Math.round(totalReach),
    placements_low: placementsLow,
    placements_high: placementsHigh,
    streams_low: streamsLow,
    streams_high: streamsHigh,
    algo_streams_low: algoLow,
    algo_streams_high: algoHigh,
    total_streams_low: totalStreamsLow,
    total_streams_high: totalStreamsHigh,
    new_followers_low: Math.max(0, followersLow),
    new_followers_high: followersHigh,
    revenue_low: revLow,
    revenue_high: revHigh,
    total_cost: totalCost,
    cost_by_method: costByMethod,
    cost_per_stream: (totalCost / Math.max((streamsLow + streamsHigh) / 2, 1)).toFixed(4),
    net_roi_low: revLow - totalCost,
    net_roi_high: revHigh - totalCost,
    top_curators: [],
    _complete: false,
    _pct: enrichmentPct,
  });
}

function renderCampaignForecast(data) {
  const card = $('#campaign-forecast-card');
  if (!card) return;
  show(card);

  const fmtN = n => n.toLocaleString();
  const isComplete = data._complete;
  const pctLabel = !isComplete && data._pct != null ? ` <span class="forecast-pct">${data._pct}% scanned</span>` : '';

  let topCuratorsHtml = '';
  if (data.top_curators && data.top_curators.length > 0) {
    topCuratorsHtml = '<div class="forecast-top-curators"><h4>Highest-Impact Curators</h4><ol>' +
      data.top_curators.map(c =>
        `<li><strong>${c.name}</strong> — ${c.playlist_name} (${fmtN(c.followers)} followers) via ${c.method}<br><span class="forecast-detail">${c.acceptance_rate}% est. acceptance · ~${fmtN(c.expected_streams)} streams</span></li>`
      ).join('') + '</ol></div>';
  }

  // Cost breakdown by method
  let costBreakdownHtml = '';
  if (data.cost_by_method) {
    const methods = Object.entries(data.cost_by_method)
      .sort((a, b) => b[1].cost - a[1].cost);
    costBreakdownHtml = methods.map(([method, info]) =>
      `<span class="cost-line">${info.count} via ${method} — ${info.cost > 0 ? '$' + info.cost.toFixed(0) : 'Free'}</span>`
    ).join('');
  }

  const roiClass = data.net_roi_high > 0 ? 'roi-positive' : 'roi-negative';

  card.innerHTML = `
    <h3>Campaign Forecast${pctLabel}</h3>
    <p class="card-sub">Predicted impact from pitching ${data.curator_count} contactable curators${!isComplete ? ' (updating live...)' : ''}</p>
    <div class="forecast-grid">
      <div class="forecast-stat">
        <div class="forecast-value">${fmtN(data.total_reach)}</div>
        <div class="forecast-label">Combined playlist followers</div>
      </div>
      <div class="forecast-stat">
        <div class="forecast-value">${data.placements_low}–${data.placements_high}</div>
        <div class="forecast-label">Expected playlist placements</div>
      </div>
      <div class="forecast-stat">
        <div class="forecast-value">${fmtN(data.streams_low)}–${fmtN(data.streams_high)}</div>
        <div class="forecast-label">Estimated playlist streams</div>
      </div>
      <div class="forecast-stat">
        <div class="forecast-value">+${fmtN(data.algo_streams_low)}–${fmtN(data.algo_streams_high)}</div>
        <div class="forecast-label">Algorithmic bonus streams (if save rate &gt; 5%)</div>
      </div>
      <div class="forecast-stat">
        <div class="forecast-value">${data.new_followers_low}–${fmtN(data.new_followers_high)}</div>
        <div class="forecast-label">Estimated new followers</div>
      </div>
      <div class="forecast-stat">
        <div class="forecast-value">$${data.revenue_low.toFixed(0)}–$${data.revenue_high.toFixed(0)}</div>
        <div class="forecast-label">Est. streaming revenue</div>
      </div>
      <div class="forecast-stat forecast-stat-total">
        <div class="forecast-value">${fmtN(data.total_streams_low)}–${fmtN(data.total_streams_high)}</div>
        <div class="forecast-label">Total potential streams</div>
      </div>
    </div>
    <div class="forecast-roi-section">
      <div class="forecast-cost-breakdown">
        <h4>Campaign Investment</h4>
        <div class="cost-lines">${costBreakdownHtml}</div>
        <div class="cost-total">Total out-of-pocket: <strong>$${(data.total_cost || 0).toFixed(0)}</strong></div>
      </div>
      <div class="forecast-roi">
        <h4>What You're Paying</h4>
        <div class="roi-metric">You'd spend <strong class="roi-negative">$${(data.total_cost || 0).toFixed(0)}</strong> to reach <strong class="roi-positive">${fmtN(data.total_reach)}</strong> playlist followers</div>
        <div class="roi-metric">That's <strong class="roi-positive">$${data.cost_per_stream || '0.00'}/stream</strong> (total spend ÷ expected streams) — cheaper than Playlist Push (~$0.02) or Spotify Ads (~$0.04)</div>
        <div class="roi-metric">Streaming revenue: <strong class="roi-positive">$${data.revenue_low.toFixed(0)}–$${data.revenue_high.toFixed(0)}</strong> — the real value is algorithmic reach, not stream payouts</div>
      </div>
    </div>
    ${topCuratorsHtml}
    <div class="forecast-insight">
      <h4>Why This Matters</h4>
      <p>Direct streaming ROI from playlist placements is often negative — but that's not the play. Playlist placements seed the <strong>save-rate signal</strong> that triggers Spotify's algorithmic playlists. ~40% of all Spotify streams come from algorithmic playlists (Music Ally, 2026). Discover Weekly alone has generated <strong>100 billion+ streams</strong> since 2015, with 77% going to emerging artists (Spotify Newsroom, June 2025).</p>
      <p>Independent campaign data from 1,200+ artist campaigns suggests the algorithm weights save rate and repeat-listen ratio roughly <strong>3x higher</strong> than raw stream volume (Chartlex, 2025-2026). New releases with &gt;5% save rate see significantly more Discover Weekly placements. New releases with &gt;20% save rate and 2.0+ stream-to-listener ratio have been observed triggering algorithmic placement within 10-14 days.</p>
      <p>Sonically-targeted pitches deliver higher save rates than generic submissions because the playlist listeners already like that sound. That's the multiplier — you're not buying streams, you're buying <strong>ignition</strong> into the algorithmic flywheel.</p>
    </div>
    <div class="forecast-sources">
      <h4>Sources</h4>
      <ul>
        <li>Discover Weekly: 100B+ streams, 77% to emerging artists — <a href="https://newsroom.spotify.com/2025-06-30/discover-weekly-turns-10-celebrating-100-billion-tracks-streamed-and-a-decade-of-personalized-discovery/" target="_blank" rel="noopener">Spotify Newsroom, June 2025</a></li>
        <li>~40% of Spotify streams from algorithmic playlists — <a href="https://musically.com/2026/03/11/how-to-turn-spotify-data-into-algorithmic-growth/" target="_blank" rel="noopener">Music Ally, March 2026</a></li>
        <li>Save rate ~3x weight, 1,200+ campaign dataset — <a href="https://www.chartlex.com/blog/streaming/spotify-algorithm-2026-retention-revolution" target="_blank" rel="noopener">Chartlex, 2025-2026</a></li>
        <li>Acceptance rates: SubmitHub 10-14% (SubmitHub reported), Groover ~20% (Groover reported), cold email estimated 5-8%</li>
        <li>Playlist stream conversion: 3-5% of followers per placement — <a href="https://loopsolitaire.co.uk/blog/spotify-editorial-playlists/" target="_blank" rel="noopener">Loop Solitaire</a>, Playlist Push campaign data</li>
        <li>Algorithmic trigger: 400-600 streams/day for popularity score 30/100 — <a href="https://www.decentmusicpr.com/post/spotify-algorithm-trigger-points" target="_blank" rel="noopener">Decent Music PR</a></li>
        <li>Follower conversion from playlists: ~0.1% — <a href="https://loopsolitaire.co.uk/blog/spotify-editorial-playlists/" target="_blank" rel="noopener">Loop Solitaire editorial playlist study</a></li>
        <li>Spotify payout: $0.003-0.005/stream — <a href="https://www.izotope.com/en/learn/what-are-lufs" target="_blank" rel="noopener">Spotify Loud &amp; Clear 2025</a>, streaming calculator aggregates</li>
        <li>Targeting multiplier: proprietary — based on sonic similarity consensus across matched artist catalog</li>
      </ul>
    </div>
  `;
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
// PDF Export — snapshots the live result sections so the PDF
// looks like the website (charts, bars, A&R table and all).
// Vector cover page + html2canvas snapshot of each visible
// section, one section per page (split across pages if tall).
// -------------------------------------------------------
async function generateAnalysisPDF() {
  const lib = window.jspdf;
  if (!lib || !lib.jsPDF) { alert('PDF library not loaded — please refresh and try again.'); return; }
  if (typeof html2canvas !== 'function') { alert('PDF renderer not loaded — please refresh and try again.'); return; }
  const d = window._lastAnalysisResult;
  if (!d) { alert('Run an analysis first.'); return; }

  const btn = document.getElementById('pdf-download-btn');
  const prevLabel = btn ? btn.textContent : '';
  if (btn) { btn.textContent = 'Building PDF…'; btn.disabled = true; }

  try {
    const doc = new lib.jsPDF({ unit: 'pt', format: 'letter', orientation: 'portrait' });
    const W = doc.internal.pageSize.getWidth();
    const H = doc.internal.pageSize.getHeight();
    const M = 40;
    const INK = [22, 24, 30], MUTED = [120, 124, 134], ACCENT = [34, 197, 94], HAIR = [225, 227, 231];

    const src = d.source || {};
    const up = d.user_profile || {};
    const artist = src.artist_name || up.name || 'Unknown Artist';
    const track = src.track_name || 'Untitled Track';
    const scanDate = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });

    // ---- Cover page (vector) ----
    doc.setFont('helvetica', 'bold'); doc.setFontSize(9); doc.setTextColor.apply(doc, ACCENT);
    doc.text('SONICCONVERTER', M, 60);
    doc.setDrawColor.apply(doc, HAIR); doc.setLineWidth(0.8); doc.line(M, 70, W - M, 70);
    doc.setFont('helvetica', 'normal'); doc.setFontSize(12); doc.setTextColor.apply(doc, MUTED);
    doc.text('TRACK ANALYSIS', M, 250);
    doc.setFont('helvetica', 'bold'); doc.setFontSize(38); doc.setTextColor.apply(doc, INK);
    const tl = doc.splitTextToSize(track, W - 2 * M); doc.text(tl, M, 292);
    doc.setFont('helvetica', 'normal'); doc.setFontSize(20); doc.setTextColor.apply(doc, MUTED);
    doc.text(artist, M, 292 + tl.length * 42);
    doc.setFontSize(11); doc.text('Generated ' + scanDate, M, H - 50);

    // ---- Collect visible result sections in DOM order ----
    const root = document.getElementById('results-section');
    const nodes = Array.from(root.querySelectorAll('.artist-card, .result-card, .conversion-card, .genre-callout'))
      .filter(el => {
        if (el.classList.contains('hidden')) return false;
        if (el.querySelector('#pdf-download-btn')) return false;
        return el.offsetHeight > 24 && el.offsetWidth > 24; // visible + non-empty
      });

    const imgW = W - 2 * M;
    const avail = H - 2 * M;

    for (const el of nodes) {
      const canvas = await html2canvas(el, {
        scale: 2, backgroundColor: '#ffffff', useCORS: true, logging: false,
        scrollX: 0, scrollY: -window.scrollY,
        onclone: (clonedDoc) => {
          // html2canvas can't render repeating-linear-gradient, so the striped
          // "target zone" bands come out invisible. Repaint them solid in the
          // capture clone only (opacity stays → still reads as a translucent zone).
          clonedDoc.querySelectorAll('.rec-range-band, .conv-bar-opportunity').forEach(b => {
            b.style.background = '#D8E166';
          });
        },
      });
      if (!canvas.width || !canvas.height) continue;
      const ratio = imgW / canvas.width;
      const fullH = canvas.height * ratio;

      if (fullH <= avail) {
        doc.addPage();
        doc.addImage(canvas.toDataURL('image/jpeg', 0.92), 'JPEG', M, M, imgW, fullH);
      } else {
        // Section taller than a page — slice it across multiple pages.
        const sliceSrcH = avail / ratio; // source px per page
        for (let sY = 0; sY < canvas.height; sY += sliceSrcH) {
          const hPx = Math.min(sliceSrcH, canvas.height - sY);
          const slice = document.createElement('canvas');
          slice.width = canvas.width; slice.height = hPx;
          slice.getContext('2d').drawImage(canvas, 0, sY, canvas.width, hPx, 0, 0, canvas.width, hPx);
          doc.addPage();
          doc.addImage(slice.toDataURL('image/jpeg', 0.92), 'JPEG', M, M, imgW, hPx * ratio);
        }
      }
    }

    // ---- Footer (brand + page numbers) on every page ----
    const total = doc.getNumberOfPages();
    for (let p = 1; p <= total; p++) {
      doc.setPage(p);
      doc.setFont('helvetica', 'normal'); doc.setFontSize(8); doc.setTextColor.apply(doc, MUTED);
      doc.text(`${track} — ${artist}`, M, H - 22);
      doc.text(`${p} / ${total}`, W - M, H - 22, { align: 'right' });
    }

    const safe = s => String(s).replace(/[\/\\:*?"<>|]/g, '_').trim();
    doc.save(`${safe(artist)} - ${safe(track)} - SonicConverter.pdf`);
  } catch (e) {
    console.error('PDF export failed', e);
    alert('PDF export failed: ' + (e && e.message ? e.message : e));
  } finally {
    if (btn) { btn.textContent = prevLabel || 'Download PDF'; btn.disabled = false; }
  }
}
(function() {
  const btn = document.getElementById('pdf-download-btn');
  if (btn) btn.addEventListener('click', generateAnalysisPDF);
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
