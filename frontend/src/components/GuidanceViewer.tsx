import { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import BrickMesh, { getBrickSize } from './BrickMesh';
import type { BuildStep } from '../api/legogen';

interface GuidanceViewerProps {
  steps: BuildStep[];
  currentStep: number;
  /** Index of the brick within the current step that is being narrated (0-based). -1 = none */
  narratedBrickIdx?: number;
  exploded?: boolean;
}

const STEP_Y_BASE = 0;
const LAYER_HEIGHT = 2.0;
const EXPLODED_SPREAD = 1.5;

interface BrickData {
  key: string;
  position: [number, number, number];
  size: [number, number, number];
  color: string;
  isTrans: boolean;
  stepNum: number;
  localIdx: number; // index within step
}

function GuidanceScene({ steps, currentStep, narratedBrickIdx = -1, exploded = false }: GuidanceViewerProps) {
  const bricks = useMemo(() => {
    const result: BrickData[] = [];
    let globalIdx = 0;

    for (const step of steps) {
      const stepIdx = step.step_number - 1;
      const extraSpread = exploded ? stepIdx * EXPLODED_SPREAD : 0;
      const baseY = STEP_Y_BASE + stepIdx * LAYER_HEIGHT + extraSpread;

      let partIdx = 0;
      for (const part of step.parts) {
        const size = getBrickSize(part);
        for (let q = 0; q < part.quantity; q++) {
          const col = partIdx % 6;
          const row = Math.floor(partIdx / 6);
          const x = (col - 2.5) * (size[0] + 0.15);
          const z = (row - 1) * (size[2] + 0.15);
          const y = baseY + size[1] / 2;

          result.push({
            key: `gbrick-${globalIdx}`,
            position: [x, y, z],
            size,
            color: part.color_hex,
            isTrans: part.is_trans ?? part.color.toLowerCase().includes('trans'),
            stepNum: step.step_number,
            localIdx: partIdx,
          });
          partIdx++;
          globalIdx++;
        }
      }
    }

    return result;
  }, [steps, exploded]);

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
        const isPast = b.stepNum < currentStep;
        const isCurrent = b.stepNum === currentStep;
        const isFuture = b.stepNum > currentStep;
        const isNarrated = isCurrent && b.localIdx === narratedBrickIdx;

        let opacity = 1.0;
        let wireframe = false;
        let glow = false;

        if (isPast) {
          opacity = 0.5;
        } else if (isFuture) {
          opacity = 0.15;
          wireframe = true;
        } else if (isNarrated) {
          glow = true;
        }

        return (
          <BrickMesh
            key={b.key}
            position={b.position}
            size={b.size}
            color={b.color}
            isTrans={b.isTrans}
            opacity={opacity}
            wireframe={wireframe}
            glow={glow}
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

export default function GuidanceViewer(props: GuidanceViewerProps) {
  const { steps, currentStep } = props;

  if (!steps.length) {
    return (
      <div className="w-full h-full bg-gray-800 rounded-lg flex items-center justify-center min-h-[300px] border-2 border-dashed border-gray-700">
        <div className="text-center text-gray-500">
          <p className="font-medium">Guidance Viewer</p>
          <p className="text-xs mt-2">No build data loaded</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full rounded-lg overflow-hidden bg-[#1a1a2e]" aria-label={`Guidance viewer step ${currentStep} of ${steps.length}`}>
      <Canvas
        camera={{ position: [10, 8, 10], fov: 45 }}
        shadows
        gl={{ antialias: true }}
      >
        <color attach="background" args={['#1a1a2e']} />
        <fog attach="fog" args={['#1a1a2e', 20, 40]} />
        <GuidanceScene {...props} />
      </Canvas>
    </div>
  );
}
