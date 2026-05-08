-- ============================================================
--  SecureGuard AI — Supabase Database Schema
--  Run this once in Supabase → SQL Editor
-- ============================================================

-- 1. USER PROFILES
create table if not exists public.user_profiles (
  id                    uuid default gen_random_uuid() primary key,
  user_id               uuid references auth.users(id) on delete cascade unique not null,
  full_name             text,
  card_last4            text,
  registered_location   text default 'India',
  daily_spend_limit     numeric default 80,
  max_transactions_day  integer default 10,
  email                 text,
  updated_at            timestamptz default now()
);

-- 2. SESSION TRANSACTIONS (Panel 2 logs)
create table if not exists public.session_transactions (
  id                uuid default gen_random_uuid() primary key,
  user_id           uuid references auth.users(id) on delete cascade not null,
  session_location  text,
  amount            numeric,
  num_transactions  integer,
  merchant_type     text,
  flagged           boolean default false,
  flags             jsonb default '[]',
  timestamp         timestamptz default now()
);

-- 3. ALERTS LOG
create table if not exists public.alerts (
  id          uuid default gen_random_uuid() primary key,
  user_id     uuid references auth.users(id) on delete cascade not null,
  alert_type  text,
  severity    text,
  title       text,
  description text,
  flags       jsonb default '[]',
  timestamp   timestamptz default now(),
  read        boolean default false
);

-- ── Row Level Security ────────────────────────────────────────────────────────
alter table public.user_profiles       enable row level security;
alter table public.session_transactions enable row level security;
alter table public.alerts              enable row level security;

-- Users can only see/edit their own data
create policy "own profile"        on public.user_profiles
  for all using (auth.uid() = user_id);

create policy "own sessions"       on public.session_transactions
  for all using (auth.uid() = user_id);

create policy "own alerts"         on public.alerts
  for all using (auth.uid() = user_id);
