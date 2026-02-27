-- Cache for track playlist data from Chartmetric
-- Avoids re-fetching playlists for the same track on repeat analyses
CREATE TABLE IF NOT EXISTS track_playlists_cache (
    cm_track_id INTEGER PRIMARY KEY,
    playlists JSONB DEFAULT '[]',
    fetched_at TIMESTAMPTZ DEFAULT now()
);

-- Also add spotify_url column to analysis_jobs if not already there
ALTER TABLE analysis_jobs ADD COLUMN IF NOT EXISTS spotify_url TEXT;
