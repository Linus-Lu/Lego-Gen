// frontend/src/api/legogen.test.ts
import { describe, it, expect } from 'vitest';
import { parseBrickString, bricksToLayers } from './legogen';

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
