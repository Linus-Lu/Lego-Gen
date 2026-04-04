import { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import type { BuildStep, Part } from '../api/legogen';

// ── Props ─────────────────────────────────────────────────────────────

interface LegoViewerProps {
  steps: BuildStep[];
  currentStep: number;
}

// ── Position mapping ──────────────────────────────────────────────────

const STEP_Y_BASE = 0;
const LAYER_HEIGHT = 2.0;

function getBrickSize(part: Part): [number, number, number] {
  const name = part.name.toLowerCase();
  const cat = part.category.toLowerCase();

  if (cat.includes('baseplate') || name.includes('baseplate'))
    return [4, 0.2, 4];
  if (cat.includes('plate') || name.includes('plate'))
    return [1.6, 0.35, 0.8];
  if (cat.includes('slope') || cat.includes('roof') || name.includes('slope'))
    return [1.6, 1.0, 0.8];
  if (cat.includes('tile') || name.includes('tile'))
    return [1.6, 0.25, 0.8];
  if (name.includes('window') || name.includes('door'))
    return [0.8, 1.6, 0.3];
  // Default brick
  return [1.6, 1.0, 0.8];
}

// ── Brick component ───────────────────────────────────────────────────

interface BrickProps {
  position: [number, number, number];
  size: [number, number, number];
  color: string;
  isTrans: boolean;
  opacity: number;
}

function Brick({ position, size, color, isTrans, opacity }: BrickProps) {
  const finalOpacity = isTrans ? Math.min(opacity, 0.45) : opacity;
  return (
    <mesh position={position} castShadow receiveShadow>
      <boxGeometry args={size} />
      <meshStandardMaterial
        color={color}
        transparent={isTrans || opacity < 1}
        opacity={finalOpacity}
        roughness={0.3}
        metalness={0.05}
      />
    </mesh>
  );
}

// ── Stud on top of brick ──────────────────────────────────────────────

function Stud({ position, color, opacity, isTrans }: { position: [number, number, number]; color: string; opacity: number; isTrans: boolean }) {
  const finalOpacity = isTrans ? Math.min(opacity, 0.45) : opacity;
  return (
    <mesh position={position}>
      <cylinderGeometry args={[0.18, 0.18, 0.15, 12]} />
      <meshStandardMaterial
        color={color}
        transparent={isTrans || opacity < 1}
        opacity={finalOpacity}
        roughness={0.3}
        metalness={0.05}
      />
    </mesh>
  );
}

// ── Scene content ─────────────────────────────────────────────────────

interface SceneProps {
  steps: BuildStep[];
  currentStep: number;
}

function Scene({ steps, currentStep }: SceneProps) {
  const bricks = useMemo(() => {
    const result: {
      key: string;
      position: [number, number, number];
      size: [number, number, number];
      color: string;
      isTrans: boolean;
      stepNum: number;
    }[] = [];

    let globalIdx = 0;

    for (const step of steps) {
      const stepIdx = step.step_number - 1;
      const baseY = STEP_Y_BASE + stepIdx * LAYER_HEIGHT;

      let partIdx = 0;
      for (const part of step.parts) {
        const size = getBrickSize(part);
        for (let q = 0; q < part.quantity; q++) {
          // Lay out in a grid within the step layer
          const col = partIdx % 6;
          const row = Math.floor(partIdx / 6);
          const x = (col - 2.5) * (size[0] + 0.15);
          const z = (row - 1) * (size[2] + 0.15);
          const y = baseY + size[1] / 2;

          result.push({
            key: `brick-${globalIdx}`,
            position: [x, y, z],
            size,
            color: part.color_hex,
            isTrans: part.is_trans ?? part.color.toLowerCase().includes('trans'),
            stepNum: step.step_number,
          });
          partIdx++;
          globalIdx++;
        }
      }
    }

    return result;
  }, [steps]);

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[8, 12, 6]} intensity={1.2} castShadow />
      <directionalLight position={[-4, 8, -6]} intensity={0.3} />

      {/* Ground grid */}
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

      {/* Bricks */}
      {bricks.map((b) => {
        if (b.stepNum > currentStep) return null;
        const isCurrent = b.stepNum === currentStep;
        const opacity = isCurrent ? 1.0 : 0.35;

        return (
          <group key={b.key}>
            <Brick
              position={b.position}
              size={b.size}
              color={b.color}
              isTrans={b.isTrans}
              opacity={opacity}
            />
            {/* Add a stud on top for non-flat pieces */}
            {b.size[1] > 0.3 && (
              <Stud
                position={[b.position[0], b.position[1] + b.size[1] / 2 + 0.075, b.position[2]]}
                color={b.color}
                opacity={opacity}
                isTrans={b.isTrans}
              />
            )}
          </group>
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

const LegoViewer: React.FC<LegoViewerProps> = ({ steps, currentStep }) => {
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
