// frontend/src/api/legogen.test.ts
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import {
  parseBrickString,
  bricksToLayers,
  generateBricks,
  listGalleryBuilds,
  createGalleryBuild,
  getGalleryBuild,
  starGalleryBuild,
} from './legogen';

describe('parseBrickString', () => {
  it('returns [] for empty input', () => {
    expect(parseBrickString('')).toEqual([]);
    expect(parseBrickString('   \n  ')).toEqual([]);
  });

  it('parses a single brick line', () => {
    const out = parseBrickString('2x4 (0,0,0) #C91A09');
    expect(out).toEqual([
      { h: 2, w: 4, x: 0, y: 0, z: 0, color: '#C91A09' },
    ]);
  });

  it('parses multiple brick lines', () => {
    const raw = '2x4 (0,0,0) #C91A09\n1x1 (5,5,2) #FFFFFF';
    const out = parseBrickString(raw);
    expect(out).toHaveLength(2);
    expect(out[1]).toEqual({ h: 1, w: 1, x: 5, y: 5, z: 2, color: '#FFFFFF' });
  });

  it('skips lines that do not match the grammar', () => {
    const raw = '2x4 (0,0,0) #C91A09\nnonsense\n1x1 (1,1,1) #000000';
    const out = parseBrickString(raw);
    expect(out).toHaveLength(2);
    expect(out.map(b => b.color)).toEqual(['#C91A09', '#000000']);
  });

  it('tolerates surrounding whitespace per line', () => {
    const raw = '   2x4 (0,0,0) #C91A09   ';
    expect(parseBrickString(raw)).toHaveLength(1);
  });
});

describe('bricksToLayers', () => {
  it('groups by z and orders ascending', () => {
    const bricks = [
      { h: 2, w: 4, x: 0, y: 0, z: 1, color: '#A00000' },
      { h: 2, w: 4, x: 0, y: 0, z: 0, color: '#A00000' },
      { h: 1, w: 1, x: 5, y: 5, z: 1, color: '#A00000' },
    ];
    const { steps, zLevels } = bricksToLayers(bricks);
    expect(zLevels).toEqual([0, 1]);
    expect(steps).toHaveLength(2);
    expect(steps[0].step_number).toBe(1);
    expect(steps[0].z).toBe(0);
    expect(steps[0].brick_count).toBe(1);
    expect(steps[1].brick_count).toBe(2);
  });

  it('tallies by (dims, color) within a layer', () => {
    const bricks = [
      { h: 2, w: 4, x: 0, y: 0, z: 0, color: '#A00000' },
      { h: 2, w: 4, x: 2, y: 0, z: 0, color: '#A00000' },
      { h: 1, w: 1, x: 5, y: 5, z: 0, color: '#FFFFFF' },
    ];
    const { steps } = bricksToLayers(bricks);
    const tally = steps[0].tally;
    expect(tally).toHaveLength(2);
    expect(tally[0]).toEqual({ dims: '2x4', color: '#A00000', count: 2 });
    expect(tally[1]).toEqual({ dims: '1x1', color: '#FFFFFF', count: 1 });
  });

  it('returns empty steps/zLevels for empty input', () => {
    const { steps, zLevels } = bricksToLayers([]);
    expect(steps).toEqual([]);
    expect(zLevels).toEqual([]);
  });
});

function mockFetchResponse(body: unknown, init: Partial<ResponseInit> = {}): Response {
  return new Response(JSON.stringify(body), {
    status: init.status ?? 200,
    headers: { 'Content-Type': 'application/json', ...(init.headers ?? {}) },
  });
}

describe('generateBricks', () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal('fetch', fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('posts multipart form with image, prompt, n', async () => {
    fetchMock.mockResolvedValueOnce(mockFetchResponse({
      bricks: '2x4 (0,0,0) #C91A09',
      brick_count: 1,
      stable: true,
      metadata: { model_version: 'mock', generation_time_ms: 1, rejections: 0, rollbacks: 0 },
    }));
    const file = new File([new Uint8Array([1, 2])], 'x.png', { type: 'image/png' });
    const res = await generateBricks(file, 'hi', 2);
    expect(res.brick_count).toBe(1);
    const [, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe('POST');
    const body = init.body as FormData;
    expect(body.get('image')).toBeInstanceOf(File);
    expect(body.get('prompt')).toBe('hi');
    expect(body.get('n')).toBe('2');
  });

  it('throws with server detail on non-ok response', async () => {
    fetchMock.mockResolvedValueOnce(mockFetchResponse({ detail: 'nope' }, { status: 500 }));
    await expect(generateBricks(undefined, 'hi')).rejects.toThrow('nope');
  });

  it('throws with generic fallback message when body lacks detail', async () => {
    fetchMock.mockResolvedValueOnce(new Response('not json', { status: 502 }));
    await expect(generateBricks(undefined, 'hi')).rejects.toThrow('Request failed');
  });

  it('omits n when undefined', async () => {
    fetchMock.mockResolvedValueOnce(mockFetchResponse({
      bricks: '', brick_count: 0, stable: true,
      metadata: { model_version: 'm', generation_time_ms: 0, rejections: 0, rollbacks: 0 },
    }));
    await generateBricks(undefined, 'hi');
    const body = fetchMock.mock.calls[0][1].body as FormData;
    expect(body.get('n')).toBe(null);
  });
});

describe('Gallery client', () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal('fetch', fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('listGalleryBuilds adds query string when params provided', async () => {
    fetchMock.mockResolvedValueOnce(mockFetchResponse([]));
    await listGalleryBuilds({ sort: 'bricks', q: 'cottage' });
    const [url] = fetchMock.mock.calls[0];
    expect(url).toContain('sort=bricks');
    expect(url).toContain('q=cottage');
  });

  it('listGalleryBuilds without params omits query string', async () => {
    fetchMock.mockResolvedValueOnce(mockFetchResponse([]));
    await listGalleryBuilds();
    const [url] = fetchMock.mock.calls[0];
    expect(url).not.toContain('?');
  });

  it('listGalleryBuilds throws on HTTP error', async () => {
    fetchMock.mockResolvedValueOnce(new Response('', { status: 500 }));
    await expect(listGalleryBuilds()).rejects.toThrow('HTTP 500');
  });

  it('createGalleryBuild POSTs JSON body', async () => {
    const build = { id: '1', title: 't', caption: '', bricks: 'x', brick_count: 1, stable: true, thumbnail_b64: '', stars: 0, star_count: 0, created_at: '2026-04-17' };
    fetchMock.mockResolvedValueOnce(mockFetchResponse(build));
    const payload = { title: 't', caption: '', bricks: 'x', brick_count: 1, stable: true };
    const out = await createGalleryBuild(payload);
    expect(out.id).toBe('1');
    const [, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe('POST');
    expect(init.headers['Content-Type']).toBe('application/json');
    expect(JSON.parse(init.body as string)).toEqual(payload);
  });

  it('createGalleryBuild throws with server detail on error', async () => {
    fetchMock.mockResolvedValueOnce(mockFetchResponse({ detail: 'bad title' }, { status: 400 }));
    await expect(createGalleryBuild({
      title: '', caption: '', bricks: 'x', brick_count: 0, stable: true,
    })).rejects.toThrow('bad title');
  });

  it('createGalleryBuild falls back to generic message when body lacks detail', async () => {
    fetchMock.mockResolvedValueOnce(new Response('not json', { status: 418 }));
    await expect(createGalleryBuild({
      title: 't', caption: '', bricks: 'x', brick_count: 0, stable: true,
    })).rejects.toThrow('Failed to save');
  });

  it('getGalleryBuild throws on 404', async () => {
    fetchMock.mockResolvedValueOnce(new Response('', { status: 404 }));
    await expect(getGalleryBuild('missing')).rejects.toThrow('HTTP 404');
  });

  it('starGalleryBuild PATCHes with stars payload', async () => {
    const build = { id: '1', title: 't', caption: '', bricks: '', brick_count: 0, stable: true, thumbnail_b64: '', stars: 4.5, star_count: 2, created_at: '2026-04-17' };
    fetchMock.mockResolvedValueOnce(mockFetchResponse(build));
    const out = await starGalleryBuild('1', 5);
    expect(out.stars).toBe(4.5);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toContain('/api/gallery/1/star');
    expect(init.method).toBe('PATCH');
    expect(JSON.parse(init.body as string)).toEqual({ stars: 5 });
  });

  it('starGalleryBuild throws on HTTP error', async () => {
    fetchMock.mockResolvedValueOnce(new Response('', { status: 500 }));
    await expect(starGalleryBuild('x', 3)).rejects.toThrow('HTTP 500');
  });
});
