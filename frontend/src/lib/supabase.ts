import { createClient, type SupabaseClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SECRET_KEY
  || process.env.SUPABASE_SERVICE_ROLE_KEY;

// Supabase is optional for local editing. Project sync is enabled when
// production credentials are configured.
export const supabaseAdmin: SupabaseClient | null = supabaseUrl && supabaseServiceKey
  ? createClient(supabaseUrl, supabaseServiceKey)
  : null;

export type Project = {
  id: string;
  project_id: string;
  user_id: string;
  name: string;
  thumbnail_url: string | null;
  original_path?: string | null;
  current_path?: string | null;
  thumbnail_path?: string | null;
  checkpoint_path?: string | null;
  export_path?: string | null;
  storage_status?: string;
  frame_count?: number;
  edit_version?: number;
  status: string;
  last_frame: number;
  created_at: string;
  updated_at: string;
};
