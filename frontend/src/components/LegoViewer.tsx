import { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import BrickMesh, { getBrickSize } from './BrickMesh';
import BrickCoordViewer from './BrickCoordViewer';
import { parseBrickString } from '../api/legogen';
import type { BuildStep } from '../api/legogen';

// ── Props ─────────────────────────────────────────────────────────────

interface LegoViewerProps {
  steps: BuildStep[];
  currentStep: number;
  brickString?: string;
}

// ── Layout engine ────────────────────────────────────────────────────

const BRICK_H = 1.2;
const GAP = 0.08; // small gap between bricks

interface PlacedBrick {
  key: string;
  position: [number, number, number];
  size: [number, number, number];
  color: string;
  isTrans: boolean;
  stepNum: number;
}

/**
 * Pack bricks into a roughly rectangular footprint for each layer.
 * Fills rows left-to-right, wrapping when a row exceeds the target width.
 */
function packLayer(
  parts: { size: [number, number, number]; color: string; isTrans: boolean; quantity: number; gridPos?: [number, number]; gridPositions?: [number, number][] }[],
  baseY: number,
  stepNum: number,
  startIdx: number,
): { bricks: PlacedBrick[]; layerHeight: number; nextIdx: number } {
  const bricks: PlacedBrick[] = [];
  let idx = startIdx;

  // If parts have per-instance grid_positions, use them for placement
  const hasGridPos = parts.some(p => p.gridPositions != null || p.gridPos != null);
  if (hasGridPos) {
    const positions: [number, number, number][] = [];
    for (const p of parts) {
      const [w, h, d] = p.size;
      for (let q = 0; q < p.quantity; q++) {
        // Prefer per-instance grid_positions, fall back to grid_pos + linear offset
        const gx = p.gridPositions?.[q]?.[0] ?? ((p.gridPos?.[0] ?? 0) + q * w);
        const gz = p.gridPositions?.[q]?.[1] ?? (p.gridPos?.[1] ?? 0);
        positions.push([gx + w / 2, baseY + h / 2, gz + d / 2]);
        bricks.push({
          key: `brick-${idx}`,
          position: [0, 0, 0],
          size: [w, h, d],
          color: p.color,
          isTrans: p.isTrans,
          stepNum,
        });
        idx++;
      }
    }
    // Center the footprint
    if (positions.length > 0) {
      const allX = positions.map(p => p[0]);
      const allZ = positions.map(p => p[2]);
      const cx = (Math.min(...allX) + Math.max(...allX)) / 2;
      const cz = (Math.min(...allZ) + Math.max(...allZ)) / 2;
      for (let i = 0; i < bricks.length; i++) {
        bricks[i].position = [positions[i][0] - cx, positions[i][1], positions[i][2] - cz];
      }
    }
    const maxH = parts.reduce((m, p) => Math.max(m, p.size[1]), BRICK_H);
    return { bricks, layerHeight: maxH, nextIdx: idx };
  }

  // Fallback: expand parts by quantity and auto-pack
  const expanded: { size: [number, number, number]; color: string; isTrans: boolean }[] = [];
  for (const p of parts) {
    for (let q = 0; q < p.quantity; q++) {
      expanded.push({ size: p.size, color: p.color, isTrans: p.isTrans });
    }
  }

  if (expanded.length === 0) return { bricks: [], layerHeight: BRICK_H, nextIdx: idx };

  // Calculate target footprint width from total area
  const totalArea = expanded.reduce((a, b) => a + b.size[0] * b.size[2], 0);
  const targetWidth = Math.max(4, Math.ceil(Math.sqrt(totalArea) * 1.2));

  // Sort bricks: wider bricks first for better packing
  const sorted = [...expanded].sort((a, b) => (b.size[0] * b.size[2]) - (a.size[0] * a.size[2]));

  // Simple row-based packing
  let curX = 0;
  let curZ = 0;
  let rowMaxDepth = 0;
  let maxHeight = 0;

  // Track all placed positions to compute center offset
  const positions: [number, number, number][] = [];

  for (const brick of sorted) {
    const [w, h, d] = brick.size;

    // Wrap to next row if exceeding target width
    if (curX + w > targetWidth && curX > 0) {
      curZ += rowMaxDepth + GAP;
      curX = 0;
      rowMaxDepth = 0;
    }

    positions.push([curX + w / 2, baseY + h / 2, curZ + d / 2]);
    bricks.push({
      key: `brick-${idx}`,
      position: [0, 0, 0], // will be centered below
      size: [w, h, d],
      color: brick.color,
      isTrans: brick.isTrans,
      stepNum,
    });

    curX += w + GAP;
    rowMaxDepth = Math.max(rowMaxDepth, d);
    maxHeight = Math.max(maxHeight, h);
    idx++;
  }

  // Center the footprint around origin
  if (positions.length > 0) {
    const minX = Math.min(...positions.map(p => p[0] - bricks[positions.indexOf(p)]?.size[0] / 2 || 0));
    const maxX = Math.max(...positions.map((p, i) => p[0] + bricks[i].size[0] / 2));
    const minZ = Math.min(...positions.map(p => p[2] - bricks[positions.indexOf(p)]?.size[2] / 2 || 0));
    const maxZ = Math.max(...positions.map((p, i) => p[2] + bricks[i].size[2] / 2));
    const cx = (minX + maxX) / 2;
    const cz = (minZ + maxZ) / 2;

    for (let i = 0; i < bricks.length; i++) {
      bricks[i].position = [
        positions[i][0] - cx,
        positions[i][1],
        positions[i][2] - cz,
      ];
    }
  }

  return { bricks, layerHeight: maxHeight, nextIdx: idx };
}

// ── Scene content ─────────────────────────────────────────────────────

function Scene({ steps, currentStep }: LegoViewerProps) {
  const bricks = useMemo(() => {
    const result: PlacedBrick[] = [];
    let globalIdx = 0;
    let currentY = 0;

    for (const step of steps) {
      // Expand parts with their sizes
      const partsWithSize = step.parts.map(part => ({
        size: getBrickSize(part),
        color: part.color_hex,
        isTrans: part.is_trans ?? part.color.toLowerCase().includes('trans'),
        quantity: part.quantity,
        gridPos: part.grid_pos,
        gridPositions: part.grid_positions,
      }));

      const { bricks: layerBricks, layerHeight, nextIdx } = packLayer(
        partsWithSize,
        currentY,
        step.step_number,
        globalIdx,
      );

      result.push(...layerBricks);
      currentY += layerHeight + GAP;
      globalIdx = nextIdx;
    }

    return result;
  }, [steps]);

  return (
    <>
      <ambientLight intensity={0.5} />
      <directionalLight position={[8, 12, 6]} intensity={1.2} castShadow />
      <directionalLight position={[-4, 8, -6]} intensity={0.3} />

      <Grid
        args={[20, 20]}
        cellSize={1}
        cellThickness={0.5}
        cellColor="#334155"
        sectionSize={4}
        sectionThickness={1}
        sectionColor="#475569"
        fadeDistance={25}
        position={[0, -0.01, 0]}
      />

      {bricks.map((b) => {
        if (b.stepNum > currentStep) return null;
        const isCurrent = b.stepNum === currentStep;
        const opacity = isCurrent ? 1.0 : 0.35;

        return (
          <BrickMesh
            key={b.key}
            position={b.position}
            size={b.size}
            color={b.color}
            isTrans={b.isTrans}
            opacity={opacity}
          />
        );
      })}

      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.1}
        minDistance={3}
        maxDistance={30}
      />
    </>
  );
}

// ── Main component ────────────────────────────────────────────────────

const LegoViewer: React.FC<LegoViewerProps> = ({ steps, currentStep, brickString }) => {
  if (brickString) {
    const bricks = parseBrickString(brickString);
    return <BrickCoordViewer bricks={bricks} />;
  }
  if (!steps.length) {
    return (
      <div className="w-full h-full bg-gray-800 rounded-lg flex items-center justify-center min-h-[300px] border-2 border-dashed border-gray-700">
        <div className="text-center text-gray-500">
          <div className="text-5xl mb-2 opacity-50" aria-hidden="true">🧊</div>
          <p className="font-medium">3D Interactive View</p>
          <p className="text-xs mt-2">Upload an image to see the 3D build</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full rounded-lg overflow-hidden bg-[#1a1a2e]" aria-label={`3D build viewer showing step ${currentStep} of ${steps.length}`}>
      <Canvas
        camera={{ position: [10, 8, 10], fov: 45 }}
        shadows
        gl={{ antialias: true }}
      >
        <color attach="background" args={['#1a1a2e']} />
        <fog attach="fog" args={['#1a1a2e', 20, 40]} />
        <Scene steps={steps} currentStep={currentStep} />
      </Canvas>
    </div>
  );
};

export default LegoViewer;
