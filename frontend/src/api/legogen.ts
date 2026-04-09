// ── Types matching backend response schema ────────────────────────────

export interface Part {
  part_id: string;
  name: string;
  category: string;
  color: string;
  color_hex: string;
  is_trans?: boolean;
  quantity: number;
  grid_pos?: [number, number];        // [x, z] first instance (backward compat)
  grid_positions?: [number, number][]; // per-instance [x, z] stud coordinates
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

export interface CheckResult {
  name: string;
  category: "legality" | "stability";
  status: "pass" | "warn" | "fail";
  message: string;
  details?: Record<string, unknown>;
}

export interface ValidationReport {
  score: number;
  checks: CheckResult[];
  summary: string;
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
  validation?: ValidationReport;
}

// ── Brick coordinate types ──────────────────────────────────────────

export interface BrickCoord {
  h: number; w: number;
  x: number; y: number; z: number;
  color: string;
}

export interface BrickResponse {
  bricks: string;
  caption: string;
  brick_count: number;
  stable: boolean;
  metadata: {
    model_version: string;
    generation_time_ms: number;
    rejections: number;
    rollbacks: number;
  };
}

export function parseBrickString(raw: string): BrickCoord[] {
  if (!raw || !raw.trim()) return [];
  const re = /(\d+)x(\d+) \((\d+),(\d+),(\d+)\) #([0-9A-Fa-f]{6})/;
  return raw.trim().split('\n').flatMap(line => {
    const m = line.trim().match(re);
    if (!m) return [];
    return [{ h: +m[1], w: +m[2], x: +m[3], y: +m[4], z: +m[5], color: '#' + m[6] }];
  });
}

const LDRAW_IDS: Record<string, string> = {
  '1x1': '3005', '1x2': '3004', '2x1': '3004',
  '2x2': '3003', '1x4': '3010', '4x1': '3010',
  '2x4': '3001', '4x2': '3001', '1x6': '3009', '6x1': '3009',
  '2x6': '2456', '6x2': '2456', '1x8': '3008', '8x1': '3008',
};

/** Convert brick coordinates into build steps grouped by z-layer. */
export function bricksToSteps(bricks: BrickCoord[]): { steps: BuildStep[]; zLevels: number[] } {
  const byZ = new Map<number, BrickCoord[]>();
  for (const b of bricks) {
    const group = byZ.get(b.z) ?? [];
    group.push(b);
    byZ.set(b.z, group);
  }

  const zLevels = [...byZ.keys()].sort((a, b) => a - b);

  const steps: BuildStep[] = zLevels.map((z, i) => {
    const layerBricks = byZ.get(z)!;

    // Group by (dims, color) for parts list
    const partMap = new Map<string, { dims: string; color: string; count: number }>();
    for (const b of layerBricks) {
      const dims = `${b.h}x${b.w}`;
      const key = `${dims}-${b.color}`;
      const existing = partMap.get(key);
      if (existing) existing.count++;
      else partMap.set(key, { dims, color: b.color, count: 1 });
    }

    const parts: Part[] = [...partMap.values()].map(({ dims, color, count }) => ({
      part_id: LDRAW_IDS[dims] ?? '0000',
      name: `Brick ${dims}`,
      category: 'Bricks',
      color: color.replace('#', ''),
      color_hex: color,
      quantity: count,
    }));

    return {
      step_number: i + 1,
      title: `Layer ${z} — height ${z}`,
      instruction: `Place ${layerBricks.length} brick${layerBricks.length > 1 ? 's' : ''} at height ${z}`,
      parts,
      part_count: layerBricks.length,
    };
  });

  return { steps, zLevels };
}

export async function generateBricks(
  image?: File,
  prompt?: string,
): Promise<BrickResponse> {
  const form = new FormData();
  if (image) form.append("image", image);
  if (prompt) form.append("prompt", prompt);
  const res = await fetch(`${API_BASE}/api/generate-bricks`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Gallery types ────────────────────────────────────────────────────

export interface GalleryBuild {
  id: string;
  title: string;
  category: string;
  complexity: string;
  parts_count: number;
  description_json: string;
  thumbnail_b64: string;
  stars: number;
  star_count: number;
  created_at: string;
}

// ── API client ────────────────────────────────────────────────────────

const API_BASE = import.meta.env.VITE_API_URL ?? '';

// ── SSE Streaming API ──────────────────────────────────────────────

export interface StreamProgress {
  stage: 'stage1' | 'stage2' | 'validating';
  message: string;
}

export async function generateBuildStream(
  onProgress: (progress: StreamProgress) => void,
  image?: File,
  prompt?: string,
): Promise<GenerateResponse> {
  const form = new FormData();
  if (image) form.append("image", image);
  if (prompt) form.append("prompt", prompt);

  const res = await fetch(`${API_BASE}/api/generate-stream`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = '';
  let result: GenerateResponse | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';

    let eventType = '';
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        eventType = line.slice(7).trim();
      } else if (line.startsWith('data: ') && eventType) {
        const data = JSON.parse(line.slice(6));
        if (eventType === 'progress') {
          onProgress(data as StreamProgress);
        } else if (eventType === 'result') {
          result = data as GenerateResponse;
        } else if (eventType === 'error') {
          throw new Error(data.detail ?? 'Generation failed');
        }
        eventType = '';
      }
    }
  }

  if (!result) throw new Error('No result received from stream');
  return result;
}

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

export async function validateBuild(
  description: LegoDescription
): Promise<ValidationReport> {
  const res = await fetch(`${API_BASE}/api/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(description),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Validation failed" }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }

  return res.json();
}

// ── Gallery API ──────────────────────────────────────────────────────

export async function listGalleryBuilds(params?: {
  category?: string;
  sort?: string;
  q?: string;
}): Promise<GalleryBuild[]> {
  const query = new URLSearchParams();
  if (params?.category) query.set('category', params.category);
  if (params?.sort) query.set('sort', params.sort);
  if (params?.q) query.set('q', params.q);
  const qs = query.toString();

  const res = await fetch(`${API_BASE}/api/gallery${qs ? `?${qs}` : ''}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function createGalleryBuild(data: {
  title: string;
  description_json: string;
  thumbnail_b64: string;
}): Promise<GalleryBuild> {
  const res = await fetch(`${API_BASE}/api/gallery`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err2 = await res.json().catch(() => ({ detail: 'Failed to save' }));
    throw new Error(err2.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

export async function getGalleryBuild(id: string): Promise<GalleryBuild> {
  const res = await fetch(`${API_BASE}/api/gallery/${id}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function starGalleryBuild(id: string, stars: number): Promise<GalleryBuild> {
  const res = await fetch(`${API_BASE}/api/gallery/${id}/star`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ stars }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
