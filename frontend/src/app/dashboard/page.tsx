import { auth0, isAuth0Configured } from "@/lib/auth0";
import { supabaseAdmin } from "@/lib/supabase";
import { DashboardGrid } from "@/components/dashboard/DashboardGrid";
import type { Project } from "@/lib/supabase";
import { redirect } from "next/navigation";

export default async function DashboardPage() {
  if (!supabaseAdmin || !isAuth0Configured()) redirect("/");
  const session = await auth0.getSession();
  if (!session) redirect("/auth/login?returnTo=/dashboard");

  const { data: projects } = await supabaseAdmin
    .from("projects")
    .select("*")
    .eq("user_id", session.user.sub)
    .order("updated_at", { ascending: false });

  return (
    <div className="min-h-screen bg-[#F9FAFB]">
      {/* Top bar */}
      <header className="bg-white border-b border-[#E5E7EB] sticky top-0 z-10">
        <div className="max-w-7xl mx-auto h-20 px-6 md:px-12 lg:px-24 flex items-center justify-between">
          <a href="/" className="flex items-center gap-2 group">
            <div className="w-11 h-11 rounded-full bg-[#171717] group-hover:bg-[#F43F5E] flex items-center justify-center transition-colors">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
                <polygon points="5 3 19 12 5 21 5 3" fill="white" />
              </svg>
            </div>
            <span className="font-semibold tracking-tight text-[#171717] text-2xl">FrameShift</span>
          </a>
          <div className="flex items-center gap-4">
            {session.user.picture && (
              <img
                src={session.user.picture}
                alt={session.user.name ?? ""}
                className="w-11 h-11 rounded-full object-cover"
              />
            )}
            <span className="text-sm text-[#6B7280] hidden sm:block">
              {session.user.name ?? session.user.email}
            </span>
            <a
              href="/auth/logout"
              className="rounded-2xl border border-[#171717] px-6 py-2.5 text-base font-semibold text-[#171717] transition-all hover:border-[#F43F5E] hover:text-[#F43F5E]"
            >
              Sign out
            </a>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-6 py-10">
        <DashboardGrid initialProjects={(projects as Project[]) ?? []} />
      </main>
    </div>
  );
}
