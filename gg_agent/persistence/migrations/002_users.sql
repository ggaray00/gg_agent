-- Users, and an owner for every session.
CREATE TABLE users (
  id            text PRIMARY KEY,                  -- the user_id: uuid4 hex
  email         text NOT NULL UNIQUE,              -- stored trimmed + lowercased
  password_hash text NOT NULL,                     -- scrypt, see persistence/users.py; never plaintext
  created_at    timestamptz NOT NULL DEFAULT now()
);

-- NOT NULL with no default: this fails on a database that already holds
-- ownerless sessions, which is deliberate — assign them an owner first.
ALTER TABLE sessions ADD COLUMN owner_id text NOT NULL REFERENCES users(id) ON DELETE CASCADE;
CREATE INDEX ON sessions (owner_id, last_activity_at DESC);
