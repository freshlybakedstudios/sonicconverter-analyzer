"""
Background job state management for async enrichment.

Uses Supabase `analysis_jobs` table to persist job state so SSE clients
can reconnect and pick up where they left off. Falls back to in-memory
dict when Supabase is unavailable.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Optional


class JobManager:
    """Manages analysis job state in Supabase (or in-memory fallback)."""

    def __init__(self, supabase=None):
        self._supabase = supabase
        self._mem: dict = {}  # fallback in-memory store

    def set_supabase(self, client):
        self._supabase = client

    def create_job(self, token: str, features: dict, matches: list,
                   all_matches: list = None) -> str:
        """Create a new analysis job. Returns the job_id (UUID)."""
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        job = {
            'id': job_id,
            'token': token,
            'status': 'enriching',
            'features': features,
            'matches': matches[:20],  # core matches for immediate display
            'all_matches': all_matches or matches,
            'playlists': {},
            'related_artists': [],
            'credits': {},
            'curator_emails': {},
            'confidence_map': {},
            'progress': {
                'playlists': '0/0',
                'related_artists': 'pending',
                'credits': '0/0',
                'curator_emails': '0/0',
            },
            'created_at': now,
            'updated_at': now,
        }

        if self._supabase:
            try:
                row = {
                    'id': job_id,
                    'token': token,
                    'status': 'enriching',
                    'features': json.dumps(features),
                    'matches': json.dumps(matches[:20]),
                    'playlists': '{}',
                    'related_artists': '[]',
                    'credits': '{}',
                    'curator_emails': '{}',
                    'confidence_map': '{}',
                    'progress': json.dumps(job['progress']),
                    'created_at': now,
                    'updated_at': now,
                }
                self._supabase.table('analysis_jobs').insert(row).execute()
            except Exception as e:
                print(f"JobManager: Supabase insert failed, using in-memory: {e}")

        self._mem[job_id] = job
        return job_id

    def update_job(self, job_id: str, **kwargs):
        """
        Update job fields. Accepts any combination of:
          status, playlists, related_artists, credits, curator_emails,
          confidence_map, progress
        """
        now = datetime.now(timezone.utc).isoformat()

        # Update in-memory
        if job_id in self._mem:
            for key, val in kwargs.items():
                if key == 'progress' and isinstance(val, dict):
                    self._mem[job_id].setdefault('progress', {}).update(val)
                elif key == 'playlists' and isinstance(val, dict):
                    self._mem[job_id].setdefault('playlists', {}).update(val)
                elif key == 'credits' and isinstance(val, dict):
                    self._mem[job_id].setdefault('credits', {}).update(val)
                elif key == 'curator_emails' and isinstance(val, dict):
                    self._mem[job_id].setdefault('curator_emails', {}).update(val)
                elif key == 'confidence_map' and isinstance(val, dict):
                    self._mem[job_id].setdefault('confidence_map', {}).update(val)
                else:
                    self._mem[job_id][key] = val
            self._mem[job_id]['updated_at'] = now

        # Update Supabase
        if self._supabase:
            try:
                row = {'updated_at': now}
                for key, val in kwargs.items():
                    if key in ('status',):
                        row[key] = val
                    elif key == 'progress':
                        # Merge progress into existing
                        existing = self._mem.get(job_id, {}).get('progress', {})
                        row['progress'] = json.dumps(existing)
                    elif key in ('playlists', 'credits', 'curator_emails', 'confidence_map'):
                        existing = self._mem.get(job_id, {}).get(key, {})
                        row[key] = json.dumps(existing)
                    elif key == 'related_artists':
                        row[key] = json.dumps(val)
                self._supabase.table('analysis_jobs').update(row).eq('id', job_id).execute()
            except Exception as e:
                print(f"JobManager: Supabase update failed: {e}")

    def get_job_state(self, job_id: str) -> Optional[dict]:
        """Get full job state. Returns None if not found."""
        if job_id in self._mem:
            return self._mem[job_id]

        if self._supabase:
            try:
                resp = self._supabase.table('analysis_jobs').select('*').eq('id', job_id).execute()
                if resp.data:
                    row = resp.data[0]
                    # Parse JSON fields
                    for field in ('features', 'matches', 'playlists', 'related_artists',
                                  'credits', 'curator_emails', 'confidence_map', 'progress'):
                        if isinstance(row.get(field), str):
                            try:
                                row[field] = json.loads(row[field])
                            except (json.JSONDecodeError, TypeError):
                                pass
                    self._mem[job_id] = row
                    return row
            except Exception as e:
                print(f"JobManager: Supabase fetch failed: {e}")

        return None

    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Remove in-memory jobs older than max_age_hours."""
        cutoff = time.time() - (max_age_hours * 3600)
        to_delete = []
        for job_id, job in self._mem.items():
            created = job.get('created_at', '')
            try:
                ts = datetime.fromisoformat(created.replace('Z', '+00:00')).timestamp()
                if ts < cutoff:
                    to_delete.append(job_id)
            except (ValueError, AttributeError):
                pass
        for job_id in to_delete:
            del self._mem[job_id]
