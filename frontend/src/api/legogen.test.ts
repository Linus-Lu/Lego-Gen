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

import { generateBricksStream } from './legogen';

function sseResponse(chunks: string[]): Response {
  const stream = new ReadableStream({
    start(controller) {
      const enc = new TextEncoder();
      for (const c of chunks) controller.enqueue(enc.encode(c));
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  });
}

describe('generateBricksStream', () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal('fetch', fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('parses progress, brick, rollback, sample, result events', async () => {
    const result = {
      bricks: '2x4 (0,0,0) #C91A09',
      brick_count: 1,
      stable: true,
      metadata: { model_version: 'mock', generation_time_ms: 1, rejections: 0, rollbacks: 0 },
    };
    fetchMock.mockResolvedValueOnce(sseResponse([
      'event: progress\ndata: {"stage":"stage1","message":"go","caption":"c"}\n\n',
      'event: brick\ndata: {"count":1}\n\n',
      'event: rollback\ndata: {"count":1}\n\n',
      'event: sample\ndata: {"index":1,"of":2,"stable":true}\n\n',
      'event: result\ndata: ' + JSON.stringify(result) + '\n\n',
    ]));
    const events: any[] = [];
    const out = await generateBricksStream({
      prompt: 'x',
      onEvent: e => events.push(e),
    });
    expect(events.map(e => e.type)).toEqual(['progress', 'brick', 'rollback', 'sample']);
    expect(out).toEqual(result);
  });

  it('throws when server emits an error event', async () => {
    fetchMock.mockResolvedValueOnce(sseResponse([
      'event: error\ndata: {"detail":"boom"}\n\n',
    ]));
    await expect(generateBricksStream({
      prompt: 'x',
      onEvent: () => {},
    })).rejects.toThrow('boom');
  });

  it('throws when stream ends without result', async () => {
    fetchMock.mockResolvedValueOnce(sseResponse([
      'event: progress\ndata: {"stage":"stage1","message":"..."}\n\n',
    ]));
    await expect(generateBricksStream({
      prompt: 'x',
      onEvent: () => {},
    })).rejects.toThrow('Stream ended without result');
  });

  it('throws with server detail on non-ok response', async () => {
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'bad' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    }));
    await expect(generateBricksStream({
      prompt: 'x',
      onEvent: () => {},
    })).rejects.toThrow('bad');
  });

  it('throws fallback message when non-ok body lacks JSON', async () => {
    fetchMock.mockResolvedValueOnce(new Response('not json', { status: 502 }));
    // Match legogen.ts behavior: res.json().catch returns { detail: 'Request failed' },
    // so the thrown message is 'Request failed' (not 'HTTP 502').
    await expect(generateBricksStream({
      prompt: 'x',
      onEvent: () => {},
    })).rejects.toThrow('Request failed');
  });

  it('throws when response has no body', async () => {
    const noBody = new Response(null, { status: 200 });
    Object.defineProperty(noBody, 'body', { value: null });
    fetchMock.mockResolvedValueOnce(noBody);
    await expect(generateBricksStream({
      prompt: 'x',
      onEvent: () => {},
    })).rejects.toThrow('No response body');
  });

  it('ignores malformed event chunks', async () => {
    const result = {
      bricks: '', brick_count: 0, stable: true,
      metadata: { model_version: 'm', generation_time_ms: 0, rejections: 0, rollbacks: 0 },
    };
    fetchMock.mockResolvedValueOnce(sseResponse([
      'event: progress\ndata: this-is-not-json\n\n',  // JSON parse error — skipped
      ':\n\n',                                         // comment-only, no event/data
      'event: result\ndata: ' + JSON.stringify(result) + '\n\n',
    ]));
    const events: any[] = [];
    const out = await generateBricksStream({
      prompt: 'x',
      onEvent: e => events.push(e),
    });
    expect(events).toEqual([]);
    expect(out).toEqual(result);
  });

  it('forwards multipart fields (image, prompt, n)', async () => {
    const result = {
      bricks: '', brick_count: 0, stable: true,
      metadata: { model_version: 'm', generation_time_ms: 0, rejections: 0, rollbacks: 0 },
    };
    fetchMock.mockResolvedValueOnce(sseResponse([
      'event: result\ndata: ' + JSON.stringify(result) + '\n\n',
    ]));
    const file = new File([new Uint8Array([1])], 'x.png', { type: 'image/png' });
    await generateBricksStream({
      image: file, prompt: 'hi', n: 3, onEvent: () => {},
    });
    const [, init] = fetchMock.mock.calls[0];
    const body = init.body as FormData;
    expect(body.get('image')).toBeInstanceOf(File);
    expect(body.get('prompt')).toBe('hi');
    expect(body.get('n')).toBe('3');
  });

  it('caller AbortSignal aborts the fetch', async () => {
    const ctrl = new AbortController();
    fetchMock.mockImplementationOnce((_url, init) => {
      return new Promise((_, reject) => {
        init.signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
      });
    });
    const promise = generateBricksStream({
      prompt: 'x', onEvent: () => {}, signal: ctrl.signal,
    });
    ctrl.abort();
    await expect(promise).rejects.toThrow(/abort/i);
  });

  it('pre-aborted caller signal aborts before fetch starts', async () => {
    const ctrl = new AbortController();
    ctrl.abort();  // already aborted before the call
    fetchMock.mockImplementationOnce((_url, init) => {
      // The combined timeoutCtl signal should already be aborted by the time
      // fetch runs, so this either rejects immediately or the function bails
      // before calling fetch at all. Either outcome manifests as an AbortError.
      return new Promise((_, reject) => {
        if (init.signal.aborted) {
          reject(new DOMException('aborted', 'AbortError'));
        } else {
          init.signal.addEventListener('abort', () =>
            reject(new DOMException('aborted', 'AbortError')),
          );
        }
      });
    });
    await expect(generateBricksStream({
      prompt: 'x', onEvent: () => {}, signal: ctrl.signal,
    })).rejects.toThrow(/abort/i);
  });

  it('timeout aborts the fetch', async () => {
    fetchMock.mockImplementationOnce((_url, init) => {
      return new Promise((_, reject) => {
        init.signal.addEventListener('abort', () => reject(new DOMException('timeout', 'AbortError')));
      });
    });
    await expect(generateBricksStream({
      prompt: 'x', onEvent: () => {}, timeoutMs: 5,
    })).rejects.toThrow(/abort|timeout/i);
  });

  it('falls back to HTTP <status> when non-ok body is valid JSON without detail', async () => {
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ other: 'thing' }), {
      status: 502, headers: { 'Content-Type': 'application/json' },
    }));
    // err.detail is undefined here → the ?? `HTTP ${res.status}` fallback fires.
    await expect(generateBricksStream({
      prompt: 'x', onEvent: () => {},
    })).rejects.toThrow('HTTP 502');
  });

  it('falls back to generic message when error event body lacks detail', async () => {
    fetchMock.mockResolvedValueOnce(sseResponse([
      'event: error\ndata: {"code":"x"}\n\n',
    ]));
    // data.detail is undefined → the ?? 'Generation failed' fallback fires.
    await expect(generateBricksStream({
      prompt: 'x', onEvent: () => {},
    })).rejects.toThrow('Generation failed');
  });
});

describe('coverage-closing corner cases', () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal('fetch', fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('generateBricks falls back to HTTP <status> when non-ok JSON lacks detail', async () => {
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ other: 'thing' }), {
      status: 503, headers: { 'Content-Type': 'application/json' },
    }));
    await expect(generateBricks(undefined, 'hi')).rejects.toThrow('HTTP 503');
  });

  it('createGalleryBuild falls back to HTTP <status> when non-ok JSON lacks detail', async () => {
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ other: 'thing' }), {
      status: 418, headers: { 'Content-Type': 'application/json' },
    }));
    await expect(createGalleryBuild({
      title: 't', caption: '', bricks: 'x', brick_count: 0, stable: true,
    })).rejects.toThrow('HTTP 418');
  });

  it('getGalleryBuild returns the build JSON on 200', async () => {
    const build = {
      id: 'abc', title: 't', caption: '', bricks: '', brick_count: 0,
      stable: true, thumbnail_b64: '', stars: 0, star_count: 0,
      created_at: '2026-04-17',
    };
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(build), {
      status: 200, headers: { 'Content-Type': 'application/json' },
    }));
    const out = await getGalleryBuild('abc');
    expect(out).toEqual(build);
  });
});
