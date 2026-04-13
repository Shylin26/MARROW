CREATE TABLE IF NOT EXISTS sessions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_uid TEXT UNIQUE,-- Unique ID for the session (UUID)
    start_time REAL,
    end_time REAL,
    metadata TEXT
);
CREATE TABLE IF NOT EXISTS observations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    timestamp REAL,
    frame_path TEXT,
    app_name TEXT,
    window_title TEXT,
    keystrokes TEXT,
    mouse_x INTEGER,
    mouse_y INTEGER,
    mouse_events TEXT,
    audio_transcript TEXT,
    -- Nullable filled by Whisper later
    FOREIGN KEY (session_id)
    REFERENCES sessions(session_uid)

);
-- Indices are critical they allow the trainer to fetch sequences
-- from a specific session in chronological order instantly
CREATE INDEX IF NOT EXISTS idx_obs_session ON observations(session_id);
CREATE INDEX IF NOT EXISTS idx_obs_timestamp ON observations(timestamp);
