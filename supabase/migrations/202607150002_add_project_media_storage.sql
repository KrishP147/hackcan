-- Durable project media. Modal Volume is a disposable working cache.
alter table public.projects
  add column if not exists original_path text,
  add column if not exists current_path text,
  add column if not exists thumbnail_path text,
  add column if not exists checkpoint_path text,
  add column if not exists export_path text,
  add column if not exists storage_status text not null default 'pending',
  add column if not exists frame_count integer not null default 0,
  add column if not exists edit_version integer not null default 0;

insert into storage.buckets (
  id,
  name,
  public,
  file_size_limit,
  allowed_mime_types
)
values (
  'project-media',
  'project-media',
  false,
  52428800,
  array[
    'video/mp4',
    'video/quicktime',
    'video/webm',
    'video/x-m4v',
    'image/jpeg',
    'application/gzip'
  ]
)
on conflict (id) do update set
  public = excluded.public,
  file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;
