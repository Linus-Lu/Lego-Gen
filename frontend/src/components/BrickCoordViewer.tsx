import { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import BrickMesh from './BrickMesh';
import type { BrickCoord } from '../api/legogen';

interface BrickCoordViewerProps {
  bricks: BrickCoord[];
}

const STUD = 1.0;
const BRICK_H = 1.0;

function Scene({ bricks }: BrickCoordViewerProps) {
  const placed = useMemo(() => {
    if (!bricks.length) return [];
    const positions = bricks.map(b => ({
      position: [
        (b.x + b.h / 2) * STUD,
        b.z * BRICK_H + BRICK_H / 2,
        (b.y + b.w / 2) * STUD,
      ] as [number, number, number],
      size: [b.h * STUD, BRICK_H, b.w * STUD] as [number, number, number],
      color: b.color,
    }));
    const allX = positions.map(p => p.position[0]);
    const allY = positions.map(p => p.position[1]);
    const allZ = positions.map(p => p.position[2]);
    const cx = (Math.min(...allX) + Math.max(...allX)) / 2;
    const cy = Math.min(...allY);
    const cz = (Math.min(...allZ) + Math.max(...allZ)) / 2;
    return positions.map((p, i) => ({
      ...p,
      position: [p.position[0] - cx, p.position[1] - cy, p.position[2] - cz] as [number, number, number],
      key: `brick-${i}`,
    }));
  }, [bricks]);

  return (
    <>
      <ambientLight intensity={0.5} />
      <directionalLight position={[8, 12, 6]} intensity={1.2} castShadow />
      <directionalLight position={[-4, 8, -6]} intensity={0.3} />
      <Grid args={[20, 20]} cellSize={1} cellThickness={0.5} cellColor="#334155"
        sectionSize={4} sectionThickness={1} sectionColor="#475569"
        fadeDistance={25} position={[0, -0.01, 0]} />
      {placed.map(b => (
        <BrickMesh key={b.key} position={b.position} size={b.size}
          color={b.color} isTrans={false} opacity={1} />
      ))}
      <OrbitControls makeDefault enableDamping dampingFactor={0.1}
        minDistance={3} maxDistance={30} />
    </>
  );
}

const BrickCoordViewer: React.FC<BrickCoordViewerProps> = ({ bricks }) => {
  if (!bricks.length) {
    return (
      <div className="w-full h-full bg-gray-800 rounded-lg flex items-center justify-center min-h-[300px] border-2 border-dashed border-gray-700">
        <div className="text-center text-gray-500">
          <p className="font-medium">3D Brick View</p>
          <p className="text-xs mt-2">Generate a model to see the 3D build</p>
        </div>
      </div>
    );
  }
  return (
    <div className="w-full h-full rounded-lg overflow-hidden bg-[#1a1a2e]">
      <Canvas camera={{ position: [12, 10, 12], fov: 45 }} shadows gl={{ antialias: true }}>
        <color attach="background" args={['#1a1a2e']} />
        <fog attach="fog" args={['#1a1a2e', 20, 40]} />
        <Scene bricks={bricks} />
      </Canvas>
    </div>
  );
};

export default BrickCoordViewer;
