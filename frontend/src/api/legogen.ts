// ═══════════════════════════════════════════════════════════════════
//  LEGOGEN API client — two-stage brick pipeline
// ═══════════════════════════════════════════════════════════════════

const API_BASE = import.meta.env.VITE_API_URL ?? '';

// ── Types ───────────────────────────────────────────────────────────

export interface BrickCoord {
  h: number; w: number;
  x: number; y: number; z: number;
  color: string;
}

export interface BrickResponse {
  bricks: string;
  caption?: string;
  brick_count: number;
  stable: boolean;
  metadata: {
    model_version: string;
    generation_time_ms: number;
    rejections: number;
    rollbacks: number;
    termination_reason?: string;
    final_stable?: boolean;
    outlines_enabled?: boolean;
    palette_validation_enabled?: boolean;
    hit_max_rejections?: boolean;
    hit_max_rollbacks?: boolean;
    n?: number;
    picked_index?: number;
    stable_rate?: number;
    selection_strategy?: string;
  };
}

export interface GalleryBuild {
  id: string;
  title: string;
  caption: string;
  bricks: string;
  brick_count: number;
  stable: boolean | number;
  thumbnail_b64: string;
  stars: number;
  star_count: number;
  created_at: string;
}

// ── Brick parsing / layer projection ───────────────────────────────

const BRICK_RE = /(\d+)x(\d+) \((\d+),(\d+),(\d+)\) #([0-9A-Fa-f]{6})/;

export function parseBrickString(raw: string): BrickCoord[] {
  if (!raw || !raw.trim()) return [];
  return raw.trim().split('\n').flatMap(line => {
    const m = line.trim().match(BRICK_RE);
    if (!m) return [];
    return [{ h: +m[1], w: +m[2], x: +m[3], y: +m[4], z: +m[5], color: '#' + m[6] }];
  });
}

export interface LayerStep {
  step_number: number;
  z: number;
  bricks: BrickCoord[];
  brick_count: number;
  /** Dim → count tally, for the parts legend. */
  tally: { dims: string; color: string; count: number }[];
}

/** Group bricks by z-level into ordered build steps. */
export function bricksToLayers(bricks: BrickCoord[]): { steps: LayerStep[]; zLevels: number[] } {
  const byZ = new Map<number, BrickCoord[]>();
  for (const b of bricks) {
    const g = byZ.get(b.z) ?? [];
    g.push(b);
    byZ.set(b.z, g);
  }
  const zLevels = [...byZ.keys()].sort((a, b) => a - b);
  const steps: LayerStep[] = zLevels.map((z, i) => {
    const layer = byZ.get(z)!;
    const tallyMap = new Map<string, { dims: string; color: string; count: number }>();
    for (const b of layer) {
      const dims = `${b.h}x${b.w}`;
      const key = `${dims}-${b.color}`;
      const hit = tallyMap.get(key);
      if (hit) hit.count++;
      else tallyMap.set(key, { dims, color: b.color, count: 1 });
    }
    return {
      step_number: i + 1,
      z,
      bricks: layer,
      brick_count: layer.length,
      tally: [...tallyMap.values()].sort((a, b) => b.count - a.count),
    };
  });
  return { steps, zLevels };
}

// ── Generation API ─────────────────────────────────────────────────

export async function generateBricks(
  image?: File,
  prompt?: string,
  n?: number,
  requireStable?: boolean,
): Promise<BrickResponse> {
  const form = new FormData();
  if (image) form.append('image', image);
  if (prompt) form.append('prompt', prompt);
  if (n !== undefined) form.append('n', String(n));
  if (requireStable !== undefined) form.append('require_stable', String(requireStable));

  const res = await fetch(`${API_BASE}/api/generate-bricks`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Streaming (SSE) ────────────────────────────────────────────────

export type StreamEvent =
  | { type: 'progress'; stage: 'stage1' | 'stage2'; message: string; caption?: string }
  | { type: 'brick'; count: number }
  | { type: 'rollback'; count: number }
  | { type: 'sample'; index: number; of: number; stable: boolean };

/** Wall-clock cap on the whole SSE stream (ms). Matches backend timeout
 * plus a small buffer for network overhead, so we bail out if the server
 * stops emitting events mid-stream. */
const DEFAULT_STREAM_TIMEOUT_MS = 150_000;

export async function generateBricksStream(
  opts: {
    image?: File;
    prompt?: string;
    n?: number;
    requireStable?: boolean;
    onEvent: (evt: StreamEvent) => void;
    signal?: AbortSignal;
    timeoutMs?: number;
  },
): Promise<BrickResponse> {
  const form = new FormData();
  if (opts.image) form.append('image', opts.image);
  if (opts.prompt) form.append('prompt', opts.prompt);
  if (opts.n !== undefined) form.append('n', String(opts.n));
  if (opts.requireStable !== undefined) form.append('require_stable', String(opts.requireStable));

  // Combine the caller's abort signal with our own timeout signal so either
  // one can short-circuit a hung stream. Without this the fetch+reader can
  // block indefinitely when the backend sends headers and then stalls.
  const timeoutCtl = new AbortController();
  const timeoutMs = opts.timeoutMs ?? DEFAULT_STREAM_TIMEOUT_MS;
  const timeoutId = setTimeout(() => timeoutCtl.abort(), timeoutMs);
  const onCallerAbort = () => timeoutCtl.abort();
  // If the caller's signal is already aborted, addEventListener won't fire —
  // propagate the abort up front so fetch never starts.
  if (opts.signal?.aborted) timeoutCtl.abort();
  else opts.signal?.addEventListener('abort', onCallerAbort);

  try {
    const res = await fetch(`${API_BASE}/api/generate-stream`, {
      method: 'POST',
      body: form,
      signal: timeoutCtl.signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(err.detail ?? `HTTP ${res.status}`);
    }
    const reader = res.body?.getReader();
    if (!reader) throw new Error('No response body');

    const decoder = new TextDecoder();
    let buffer = '';
    let result: BrickResponse | null = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split('\n\n');
      // String.split always returns an array of length ≥ 1, so pop() never
      // returns undefined here; the `?? ''` is a belt-and-suspenders fallback.
      /* v8 ignore next -- unreachable: split() minimum length is 1 */
      buffer = chunks.pop() ?? '';

      for (const chunk of chunks) {
        const lines = chunk.split('\n');
        let eventType = '';
        let dataStr = '';
        for (const line of lines) {
          if (line.startsWith('event: ')) eventType = line.slice(7).trim();
          else if (line.startsWith('data: ')) dataStr = line.slice(6);
        }
        if (!eventType || !dataStr) continue;
        let data: any;
        try { data = JSON.parse(dataStr); } catch { continue; }

        if (eventType === 'progress') {
          opts.onEvent({ type: 'progress', stage: data.stage, message: data.message, caption: data.caption });
        } else if (eventType === 'brick') {
          opts.onEvent({ type: 'brick', count: data.count });
        } else if (eventType === 'rollback') {
          opts.onEvent({ type: 'rollback', count: data.count });
        } else if (eventType === 'sample') {
          opts.onEvent({ type: 'sample', index: data.index, of: data.of, stable: data.stable });
        } else if (eventType === 'result') {
          result = data as BrickResponse;
        } else if (eventType === 'error') {
          throw new Error(data.detail ?? 'Generation failed');
        }
      }
    }
    if (!result) throw new Error('Stream ended without result');
    return result;
  } finally {
    clearTimeout(timeoutId);
    opts.signal?.removeEventListener('abort', onCallerAbort);
  }
}

// ── Gallery API ────────────────────────────────────────────────────

export async function listGalleryBuilds(params?: {
  sort?: string;
  q?: string;
}): Promise<GalleryBuild[]> {
  const qs = new URLSearchParams();
  if (params?.sort) qs.set('sort', params.sort);
  if (params?.q) qs.set('q', params.q);
  const suffix = qs.toString();
  const res = await fetch(`${API_BASE}/api/gallery${suffix ? `?${suffix}` : ''}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function createGalleryBuild(data: {
  title: string;
  caption: string;
  bricks: string;
  brick_count: number;
  stable: boolean;
  thumbnail_b64?: string;
}): Promise<GalleryBuild> {
  const res = await fetch(`${API_BASE}/api/gallery`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to save' }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
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

export async function downloadLDraw(title: string, bricks: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/export-ldr`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, bricks }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }

  const text = await res.text();
  const disposition = res.headers.get('Content-Disposition') ?? '';
  const filenameMatch = disposition.match(/filename="([^"]+)"/i);
  const filename = filenameMatch?.[1] ?? 'legogen-build.ldr';

  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
