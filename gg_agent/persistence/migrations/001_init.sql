-- Sessions and their transcripts. Applied by persistence/migrate.py.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS schema_version (version int NOT NULL);

CREATE TABLE sessions (
  id                text PRIMARY KEY,
  parent_session_id text REFERENCES sessions(id) ON DELETE SET NULL,
  source            text NOT NULL DEFAULT 'cli',     -- 'cli' | 'api' | 'subagent'
  provider          text,
  model             text,
  system_prompt     text,
  cwd               text,
  title             text,                            -- first user message, truncated to 80 chars
  metadata          jsonb NOT NULL DEFAULT '{}',
  started_at        timestamptz NOT NULL DEFAULT now(),
  ended_at          timestamptz,
  end_reason        text,
  message_count     int    NOT NULL DEFAULT 0,
  tool_call_count   int    NOT NULL DEFAULT 0,
  input_tokens      bigint NOT NULL DEFAULT 0,
  output_tokens     bigint NOT NULL DEFAULT 0,
  last_activity_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON sessions (parent_session_id);
CREATE INDEX ON sessions (last_activity_at DESC);
CREATE INDEX ON sessions (cwd, last_activity_at DESC);    -- for --continue

CREATE TABLE messages (
  id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,   -- THE ordering key
  session_id   text NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  role         text NOT NULL,                  -- user | assistant | tool
  content      text,
  tool_calls   jsonb,                          -- assistant only, OpenAI shape
  tool_call_id text,                           -- tool only
  tool_name    text,                           -- tool only (the message's "name" key)
  active       boolean NOT NULL DEFAULT true,  -- reserved for future compaction/rewind
  created_at   timestamptz NOT NULL DEFAULT now(),
  -- Capped: to_tsvector errors above ~1MB, and huge tool output is noise anyway.
  search_tsv   tsvector GENERATED ALWAYS AS (
                 to_tsvector('simple'::regconfig,
                   left(coalesce(content, ''), 100000) || ' ' || coalesce(tool_name, ''))) STORED
);
CREATE INDEX ON messages (session_id, id);
CREATE INDEX messages_fts  ON messages USING gin (search_tsv);
CREATE INDEX messages_trgm ON messages USING gin (content gin_trgm_ops)
  WHERE role IN ('user', 'assistant');
