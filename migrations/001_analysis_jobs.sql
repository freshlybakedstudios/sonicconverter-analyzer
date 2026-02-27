-- Analysis jobs table for background enrichment tracking
CREATE TABLE IF NOT EXISTS analysis_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    token TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'enriching',
    spotify_url TEXT,
    features JSONB DEFAULT '{}',
    matches JSONB DEFAULT '[]',
    playlists JSONB DEFAULT '{}',
    related_artists JSONB DEFAULT '[]',
    credits JSONB DEFAULT '{}',
    curator_emails JSONB DEFAULT '{}',
    confidence_map JSONB DEFAULT '{}',
    progress JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Index for polling by status (used by mac_worker)
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status ON analysis_jobs (status, created_at);

-- ISRC to Chartmetric track ID cache
CREATE TABLE IF NOT EXISTS isrc_cm_track_map (
    isrc TEXT PRIMARY KEY,
    cm_track_id INTEGER NOT NULL,
    resolved_at TIMESTAMPTZ DEFAULT now()
);

-- Auto-cleanup old jobs (optional, run manually or via cron)
-- DELETE FROM analysis_jobs WHERE created_at < now() - interval '7 days';
