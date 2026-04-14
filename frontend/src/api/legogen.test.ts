import { describe, it, expect } from 'vitest';
import { parseBrickString, bricksToSteps, type BrickCoord } from './legogen';

// ── parseBrickString ────────────────────────────────────────────────

describe('parseBrickString', () => {
  it('returns empty array for empty string', () => {
    expect(parseBrickString('')).toEqual([]);
  });

  it('returns empty array for whitespace only', () => {
    expect(parseBrickString('  \n  ')).toEqual([]);
  });

  it('parses a single brick', () => {
    const bricks = parseBrickString('2x4 (5,3,0) #C91A09');
    expect(bricks).toHaveLength(1);
    expect(bricks[0]).toEqual({
      h: 2, w: 4, x: 5, y: 3, z: 0, color: '#C91A09',
    });
  });

  it('parses multiple bricks', () => {
    const raw = '2x4 (5,3,0) #C91A09\n1x2 (3,7,1) #05131D';
    const bricks = parseBrickString(raw);
    expect(bricks).toHaveLength(2);
    expect(bricks[1].color).toBe('#05131D');
  });

  it('skips malformed lines', () => {
    const raw = 'garbage\n2x4 (1,2,3) #AABBCC\ninvalid';
    const bricks = parseBrickString(raw);
    expect(bricks).toHaveLength(1);
    expect(bricks[0].h).toBe(2);
  });
});

// ── bricksToSteps ───────────────────────────────────────────────────

describe('bricksToSteps', () => {
  it('returns empty for no bricks', () => {
    const { steps, zLevels } = bricksToSteps([]);
    expect(steps).toEqual([]);
    expect(zLevels).toEqual([]);
  });

  it('groups bricks by z-level', () => {
    const bricks: BrickCoord[] = [
      { h: 2, w: 4, x: 0, y: 0, z: 0, color: '#C91A09' },
      { h: 2, w: 4, x: 2, y: 0, z: 0, color: '#C91A09' },
      { h: 2, w: 4, x: 0, y: 0, z: 1, color: '#FFFFFF' },
    ];
    const { steps, zLevels } = bricksToSteps(bricks);
    expect(zLevels).toEqual([0, 1]);
    expect(steps).toHaveLength(2);
    expect(steps[0].part_count).toBe(2);
    expect(steps[1].part_count).toBe(1);
  });

  it('sorts zLevels ascending', () => {
    const bricks: BrickCoord[] = [
      { h: 1, w: 1, x: 0, y: 0, z: 2, color: '#FF0000' },
      { h: 1, w: 1, x: 0, y: 0, z: 0, color: '#FF0000' },
      { h: 1, w: 1, x: 0, y: 0, z: 1, color: '#FF0000' },
    ];
    const { zLevels } = bricksToSteps(bricks);
    expect(zLevels).toEqual([0, 1, 2]);
  });

  it('aggregates same-dimension same-color bricks into quantity', () => {
    const bricks: BrickCoord[] = [
      { h: 2, w: 4, x: 0, y: 0, z: 0, color: '#C91A09' },
      { h: 2, w: 4, x: 4, y: 0, z: 0, color: '#C91A09' },
      { h: 2, w: 4, x: 8, y: 0, z: 0, color: '#C91A09' },
    ];
    const { steps } = bricksToSteps(bricks);
    expect(steps[0].parts).toHaveLength(1);
    expect(steps[0].parts[0].quantity).toBe(3);
  });

  it('maps known dimensions to LDRAW IDs', () => {
    const bricks: BrickCoord[] = [
      { h: 2, w: 4, x: 0, y: 0, z: 0, color: '#C91A09' },
    ];
    const { steps } = bricksToSteps(bricks);
    expect(steps[0].parts[0].part_id).toBe('3001');
  });

  it('uses fallback ID for unknown dimensions', () => {
    const bricks: BrickCoord[] = [
      { h: 3, w: 3, x: 0, y: 0, z: 0, color: '#C91A09' },
    ];
    const { steps } = bricksToSteps(bricks);
    expect(steps[0].parts[0].part_id).toBe('0000');
  });
});
