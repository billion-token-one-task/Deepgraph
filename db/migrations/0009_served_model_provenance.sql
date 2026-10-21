-- Record which model actually answered, not only which one we asked for.
--
-- The gateway substitutes silently: a request for gemini-3.8-flash-high is
-- served by gemini-3.8-flash, and gemini-3.7-flash-high by
-- gemini-3.7-flash-exp-b. `model` is copied from our own configuration, so it
-- recorded the request and could never reveal the difference. For a ledger
-- whose purpose is that a reader can check what produced a result, that made
-- the one field they would check unreliable.
--
-- Nullable on purpose: rows written before this column existed genuinely do
-- not know what served them, and a backfilled guess would be worse than a
-- null. A provider that reports no model leaves it null too.
ALTER TABLE llm_route_observations
    ADD COLUMN IF NOT EXISTS served_model TEXT;
