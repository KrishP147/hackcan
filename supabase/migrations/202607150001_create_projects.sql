-- Project metadata only. Video, frames, masks, and renders remain on the
-- frameshift-projects Modal Volume.
create extension if not exists pgcrypto;

create table if not exists public.projects (
  id uuid primary key default gen_random_uuid(),
  project_id text not null unique,
  user_id text not null,
  name text not null default 'Untitled Project',
  thumbnail_url text,
  status text not null default 'created',
  last_frame integer not null default 0 check (last_frame >= 0),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists projects_user_updated_idx
  on public.projects (user_id, updated_at desc);

create or replace function public.set_projects_updated_at()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists projects_set_updated_at on public.projects;
create trigger projects_set_updated_at
before update on public.projects
for each row execute function public.set_projects_updated_at();

-- The browser never queries this table directly. Next.js verifies the Auth0
-- session and uses the server-only service role, always filtering by user_id.
alter table public.projects enable row level security;
revoke all on table public.projects from anon, authenticated;
grant select, insert, update, delete on table public.projects to service_role;
