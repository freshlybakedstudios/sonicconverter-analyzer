// SonicConverter Analyzer — Frontend logic
// Handles: registration, file upload, analysis, results rendering

const API_URL = ''; // Same origin when served by FastAPI; set to ngrok URL for remote

let accessToken = null;
let selectedFile = null;

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

// Enable analyze button only when file AND genre are selected
function updateAnalyzeButton() {
  const btn = $('#analyze-btn');
  const hasFile = !!selectedFile;
  const hasGenre = !!$('#genre-select').value;
  btn.disabled = !(hasFile && hasGenre);
}

// Listen for genre changes
$('#genre-select').addEventListener('change', updateAnalyzeButton);

// -------------------------------------------------------
// Analyze
// -------------------------------------------------------
$('#analyze-btn').addEventListener('click', analyzeTrack);

async function analyzeTrack() {
  if (!selectedFile || !accessToken) return;
  const genre = $('#genre-select').value;
  if (!genre) {
    alert('Please select a genre before analyzing.');
    return;
  }

  // Show loading
  hide($('#upload-section'));
  show($('#loading-section'));

  const statuses = [
    'Extracting audio features',
    'Analyzing frequency spectrum',
    'Detecting emotional character',
    'Matching against 140,000+ tracks',
    'Generating recommendations',
  ];
  let statusIdx = 0;
  const statusInterval = setInterval(() => {
    statusIdx = Math.min(statusIdx + 1, statuses.length - 1);
    $('#loader-status').textContent = statuses[statusIdx];
  }, 3000);

  try {
    const form = new FormData();
    form.append('file', selectedFile);
    form.append('token', accessToken);
    const genre = $('#genre-select').value;
    if (genre) form.append('genre', genre);

    const res = await fetch(`${API_URL}/api/analyze`, { method: 'POST', body: form });
    clearInterval(statusInterval);

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Analysis failed');
    }

    const data = await res.json();
    renderResults(data);

    hide($('#loading-section'));
    show($('#results-section'));
    show($('#floating-cta'));
    $('#results-section').scrollIntoView({ behavior: 'smooth' });

  } catch (err) {
    clearInterval(statusInterval);
    hide($('#loading-section'));
    show($('#upload-section'));
    alert('Analysis failed: ' + err.message);
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

  // Conversion comparison card — show if we have peer data
  const convCard = $('#conversion-card');
  const comp = (userProfile && userProfile.conversion_comparison) || {};
  const hasUserRate = userProfile && userProfile.conversion_rate != null;

  if (userProfile && (hasUserRate || comp.peer_median != null)) {
    // Only show "Your Rate" box if we have their data
    const yoursStat = $('#conv-yours-stat');
    if (hasUserRate) {
      $('#conv-yours').textContent = userProfile.conversion_rate.toFixed(1) + '%';
      show(yoursStat);
    } else {
      hide(yoursStat);
    }

    $('#conv-median').textContent = comp.peer_median != null ? comp.peer_median.toFixed(1) + '%' : '-';
    $('#conv-top25').textContent = comp.peer_top_25 != null ? comp.peer_top_25.toFixed(1) + '%' : '-';

    // Fan opportunity — only show if we have their rate
    const oppEl = $('#conv-opportunity');
    if (hasUserRate && userProfile.additional_fans > 0) {
      const fans = userProfile.additional_fans.toLocaleString();
      const rev = '$' + userProfile.additional_revenue.toLocaleString();
      oppEl.innerHTML =
        `Closing the gap to the top 25% of your sonic peers could convert <span class="fan-number">${fans} additional followers</span> from your existing listeners — that's <span class="fan-number">${rev}</span> if each new follower bought a $25 ticket or merch item.`;
      show(oppEl);
    } else {
      hide(oppEl);
    }
    show(convCard);
  } else {
    hide(convCard);
  }

  // Trajectory targets (flattery matches)
  const flatteryCard = $('#flattery-card');
  const flatteryList = $('#flattery-list');
  if (flatteryMatches.length > 0) {
    flatteryList.innerHTML = '';
    flatteryMatches.forEach((m, i) => {
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
    });
    show(flatteryCard);
  } else {
    hide(flatteryCard);
  }

  // Matches table
  const tbody = $('#matches-body');
  tbody.innerHTML = '';
  matches.forEach((m, i) => {
    const tr = document.createElement('tr');
    const linkUrl = m.track_url || m.spotify_url;
    const artistLink = linkUrl
      ? `<a href="${linkUrl}" target="_blank" rel="noopener">${m.name}</a>`
      : m.name;
    const emos = (m.emotions || []).filter(e => e && e !== 'neutral').slice(0, 3);
    const emoTags = emos.map(e => `<span class="mini-tag">${EMOTION_LABELS[e] || e}</span>`).join('');
    const convRate = m.conversion_rate != null ? m.conversion_rate.toFixed(1) + '%' : '-';
    // Combine all genre fields into one display
    const allGenres = new Set();
    if (m.primary_genre && m.primary_genre.toLowerCase() !== 'unknown') allGenres.add(m.primary_genre);
    if (m.secondary_genre && m.secondary_genre.toLowerCase() !== 'unknown') allGenres.add(m.secondary_genre);
    (m.artist_genres || []).forEach(g => { if (g) allGenres.add(g); });
    const genreStr = allGenres.size > 0 ? [...allGenres].join(', ') : '-';
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td class="match-artist">${artistLink}</td>
      <td class="match-sim">${(m.similarity * 100).toFixed(1)}%</td>
      <td class="match-conversion">${convRate}</td>
      <td class="match-tier">${m.tier || '-'}</td>
      <td class="match-genre">${genreStr}</td>
      <td class="match-emotions">${emoTags}</td>
    `;
    tbody.appendChild(tr);
  });

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
