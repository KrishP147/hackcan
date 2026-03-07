"use client";

import { useState } from "react";
import {
  Upload,
  ScanSearch,
  Paintbrush,
  Settings,
  type LucideIcon,
} from "lucide-react";

interface SidebarItem {
  id: string;
  icon: LucideIcon;
  label: string;
  action: () => void;
  disabled?: boolean;
  loading?: boolean;
}

interface EditorSidebarProps {
  videoLoaded: boolean;
  isDetecting: boolean;
  onUpload: () => void;
  onDetect: () => void;
  onEditClick: () => void;
}

export function EditorSidebar({
  videoLoaded,
  isDetecting,
  onUpload,
  onDetect,
  onEditClick,
}: EditorSidebarProps) {
  const [activeId, setActiveId] = useState<string | null>(null);

  const items: SidebarItem[] = [
    { id: "upload", icon: Upload, label: "Upload", action: onUpload },
    {
      id: "detect",
      icon: ScanSearch,
      label: "Detect",
      action: onDetect,
      disabled: !videoLoaded,
      loading: isDetecting,
    },
    {
      id: "edit",
      icon: Paintbrush,
      label: "Edit",
      action: onEditClick,
      disabled: !videoLoaded,
    },
    { id: "settings", icon: Settings, label: "Settings", action: () => {} },
  ];

  return (
    <aside
      className="w-[60px] flex flex-col items-center py-4 gap-1.5 border-r shrink-0"
      style={{ background: "var(--ed-surface)", borderColor: "var(--ed-border)" }}
    >
      {items.map((item) => {
        const Icon = item.icon;
        const isActive = activeId === item.id;
        return (
          <button
            key={item.id}
            onClick={() => {
              setActiveId(item.id);
              item.action();
            }}
            disabled={item.disabled}
            aria-label={item.label}
            title={item.label}
            className="relative w-10 h-10 flex items-center justify-center rounded-2xl transition-all duration-200 group"
            style={{
              background: isActive ? "rgba(244,63,94,0.12)" : "transparent",
              color: isActive ? "var(--accent)" : "var(--ed-icon)",
              opacity: item.disabled ? 0.25 : 1,
              cursor: item.disabled ? "not-allowed" : "pointer",
            }}
          >
            {item.loading ? (
              <div
                className="w-[18px] h-[18px] border-2 border-t-transparent rounded-full animate-spin"
                style={{ borderColor: "var(--accent)", borderTopColor: "transparent" }}
              />
            ) : (
              <Icon
                className="w-[18px] h-[18px] transition-all duration-200"
                strokeWidth={isActive ? 2.2 : 1.6}
              />
            )}
            {/* Tooltip */}
            <span
              className="absolute left-full ml-3 px-2 py-1 rounded-lg text-xs font-medium opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-50 shadow-xl border"
              style={{
                background: "var(--ed-surface)",
                color: "var(--ed-text)",
                borderColor: "var(--ed-border)",
              }}
            >
              {item.label}
            </span>
          </button>
        );
      })}
    </aside>
  );
}
