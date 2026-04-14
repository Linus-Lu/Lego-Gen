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

export interface Part {
  part_id: string;
  name: string;
  category: string;
  color: string;
  color_hex: string;
  quantity: number;
}

export interface BuildStep {
  step_number: number;
  title: string;
  instruction: string;
  parts: Part[];
  part_count: number;
}

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

// ── API client ────────────────────────────────────────────────────────

const API_BASE = import.meta.env.VITE_API_URL ?? '';

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
