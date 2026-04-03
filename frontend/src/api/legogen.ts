// ── Types matching backend response schema ────────────────────────────

export interface Part {
  part_id: string;
  name: string;
  category: string;
  color: string;
  color_hex: string;
  is_trans?: boolean;
  quantity: number;
}

export interface BuildStep {
  step_number: number;
  title: string;
  instruction: string;
  parts: Part[];
  part_count: number;
}

export interface Subassembly {
  name: string;
  type: string;
  parts: Part[];
  spatial: {
    position: string;
    orientation: string;
    connects_to: string[];
  };
}

export interface LegoDescription {
  set_id: string;
  object: string;
  category: string;
  subcategory: string;
  complexity: string;
  total_parts: number;
  dominant_colors: string[];
  dimensions_estimate: { width: string; height: string; depth: string };
  subassemblies: Subassembly[];
  build_hints: string[];
}

export interface GenerateResponse {
  description: LegoDescription;
  steps: BuildStep[];
  metadata: {
    model_version: string;
    generation_time_ms: number;
    json_valid: boolean;
    errors: string[];
  };
}

// ── API client ────────────────────────────────────────────────────────

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function generateBuild(
  image: File,
  prompt?: string
): Promise<GenerateResponse> {
  const form = new FormData();
  form.append("image", image);
  if (prompt) form.append("prompt", prompt);

  const res = await fetch(`${API_BASE}/api/generate`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }

  return res.json();
}

export async function generateBuildFromText(
  prompt: string
): Promise<GenerateResponse> {
  const form = new FormData();
  form.append("prompt", prompt);

  const res = await fetch(`${API_BASE}/api/generate-from-text`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }

  return res.json();
}
