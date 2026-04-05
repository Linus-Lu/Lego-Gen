import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import type { Mesh } from 'three';
import type { Part } from '../api/legogen';

// ── Size mapping ─────────────────────────────────────────────────────

export function getBrickSize(part: Part): [number, number, number] {
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
  return [1.6, 1.0, 0.8];
}

// ── Stud ─────────────────────────────────────────────────────────────

function Stud({
  position,
  color,
  opacity,
  isTrans,
  wireframe = false,
}: {
  position: [number, number, number];
  color: string;
  opacity: number;
  isTrans: boolean;
  wireframe?: boolean;
}) {
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
        wireframe={wireframe}
      />
    </mesh>
  );
}

// ── BrickMesh ────────────────────────────────────────────────────────

interface BrickMeshProps {
  position: [number, number, number];
  size: [number, number, number];
  color: string;
  isTrans: boolean;
  opacity?: number;
  wireframe?: boolean;
  glow?: boolean;
}

export default function BrickMesh({
  position,
  size,
  color,
  isTrans,
  opacity = 1,
  wireframe = false,
  glow = false,
}: BrickMeshProps) {
  const meshRef = useRef<Mesh>(null);
  const finalOpacity = isTrans ? Math.min(opacity, 0.45) : opacity;

  useFrame(() => {
    if (!glow || !meshRef.current) return;
    const mat = meshRef.current.material as any;
    if (mat.emissiveIntensity !== undefined) {
      mat.emissiveIntensity = 0.3 + Math.sin(Date.now() * 0.005) * 0.2;
    }
  });

  return (
    <group>
      <mesh ref={meshRef} position={position} castShadow receiveShadow>
        <boxGeometry args={size} />
        <meshStandardMaterial
          color={color}
          transparent={isTrans || opacity < 1}
          opacity={finalOpacity}
          roughness={0.3}
          metalness={0.05}
          wireframe={wireframe}
          emissive={glow ? '#00ff44' : '#000000'}
          emissiveIntensity={glow ? 0.3 : 0}
        />
      </mesh>
      {size[1] > 0.3 && (
        <Stud
          position={[position[0], position[1] + size[1] / 2 + 0.075, position[2]]}
          color={color}
          opacity={opacity}
          isTrans={isTrans}
          wireframe={wireframe}
        />
      )}
    </group>
  );
}
