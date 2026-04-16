import { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import BrickMesh from './BrickMesh';
import type { BrickCoord } from '../api/legogen';

const STUD = 1.0;
const BRICK_H = 1.0;

interface BrickCoordViewerProps {
  bricks: BrickCoord[];
  /** Ordered z-levels (one per step). Required for step-wise reveal. */
  zLevels?: number[];
  /** 1-indexed step. When set, bricks at z ≤ zLevels[currentStep-1] render solid. */
  currentStep?: number;
  /** If true, ghosts future-layer bricks at low opacity. */
  showFuture?: boolean;
  /** Disable orbit controls (useful for thumbnails). */
  frozen?: boolean;
  /** Force camera position. */
  camera?: [number, number, number];
  /** Enable / disable the grid plane. */
  showGrid?: boolean;
}

function Scene({
  bricks, zLevels, currentStep, showFuture, showGrid,
}: Omit<BrickCoordViewerProps, 'frozen' | 'camera'>) {
  const placed = useMemo(() => {
    if (!bricks.length) return [];

    const atStep = zLevels && currentStep != null ? zLevels[currentStep - 1] : undefined;
    const currentZ = atStep ?? -1;
    const maxSolidZ = atStep ?? Infinity;

    // For framing, use ALL bricks so camera stays stable between steps.
    const all = bricks.map(b => [
      (b.x + b.h / 2) * STUD,
      b.z * BRICK_H + BRICK_H / 2,
      (b.y + b.w / 2) * STUD,
    ]);
    const xs = all.map(p => p[0]);
    const ys = all.map(p => p[1]);
    const zs = all.map(p => p[2]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = Math.min(...ys);
    const cz = (Math.min(...zs) + Math.max(...zs)) / 2;

    return bricks
      .filter(b => (showFuture ? true : b.z <= maxSolidZ))
      .map((b, i) => {
        const isCurrent = b.z === currentZ;
        const isPast = b.z < currentZ;
        const isFuture = b.z > maxSolidZ;
        let opacity = 1;
        if (atStep != null) {
          if (isCurrent) opacity = 1;
          else if (isPast) opacity = 0.6;
          else if (isFuture) opacity = 0.12;
        }
        return {
          key: `brick-${i}`,
          position: [
            (b.x + b.h / 2) * STUD - cx,
            b.z * BRICK_H + BRICK_H / 2 - cy,
            (b.y + b.w / 2) * STUD - cz,
          ] as [number, number, number],
          size: [b.h * STUD, BRICK_H, b.w * STUD] as [number, number, number],
          color: b.color,
          opacity,
          wireframe: isFuture,
        };
      });
  }, [bricks, zLevels, currentStep, showFuture]);

  return (
    <>
      <ambientLight intensity={0.45} />
      <directionalLight position={[10, 14, 8]} intensity={1.25} castShadow />
      <directionalLight position={[-6, 8, -6]} intensity={0.35} color="#6cc2ff" />
      {showGrid !== false && (
        <Grid
          args={[30, 30]}
          cellSize={1}
          cellThickness={0.4}
          cellColor="#1e242e"
          sectionSize={5}
          sectionThickness={0.9}
          sectionColor="#2a313d"
          fadeDistance={28}
          infiniteGrid
          position={[0, -0.01, 0]}
        />
      )}
      {placed.map(b => (
        <BrickMesh {...b} />
      ))}
      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.12}
        minDistance={3}
        maxDistance={40}
      />
    </>
  );
}

export default function BrickCoordViewer({
  bricks, zLevels, currentStep, showFuture, frozen, camera, showGrid,
}: BrickCoordViewerProps) {
  if (!bricks.length) {
    return (
      <div className="relative w-full h-full bp-grid-dense border border-[var(--color-line)] overflow-hidden">
        <div className="absolute inset-0 grid place-items-center">
          <div className="text-center">
            <p className="mono text-[10px] tracking-[0.2em] uppercase text-[var(--color-mute)] mb-2">
              VIEWPORT // EMPTY
            </p>
            <p className="mono text-[13px] text-[var(--color-fg-strong)]">
              awaiting build stream<span className="caret" />
            </p>
          </div>
        </div>
      </div>
    );
  }

  const cam = camera ?? [12, 10, 12];
  return (
    <div className="relative w-full h-full bg-[var(--color-ink-1)] overflow-hidden">
      <Canvas camera={{ position: cam, fov: 42 }} shadows gl={{ antialias: true }}>
        <color attach="background" args={['#0a0c10']} />
        <fog attach="fog" args={['#0a0c10', 22, 46]} />
        <Scene
          bricks={bricks}
          zLevels={zLevels}
          currentStep={currentStep}
          showFuture={showFuture}
          showGrid={showGrid}
        />
        {frozen && null}
      </Canvas>
    </div>
  );
}
