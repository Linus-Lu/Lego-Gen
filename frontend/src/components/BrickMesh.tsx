import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import type { Mesh } from 'three';
import type { Part } from '../api/legogen';

// ── LEGO unit constants (in Three.js units) ─────────────────────────
// 1 stud = 1.0 unit width/depth
// 1 brick height = 1.2 units (real ratio: 9.6mm tall / 8mm stud pitch)
// 1 plate height = 0.4 units (real ratio: 3.2mm / 8mm)

const STUD = 1.0;
const BRICK_H = 1.2;
const PLATE_H = 0.4;

// ── Size mapping ─────────────────────────────────────────────────────

/** Parse "Brick 2x4" or "Plate 1x6" style names into [width, height, depth]. */
export function getBrickSize(part: Part): [number, number, number] {
  const name = part.name.toLowerCase();
  const cat = part.category.toLowerCase();

  // Try to parse NxM dimensions from the name
  const dimMatch = name.match(/(\d+)\s*x\s*(\d+)/);
  let w = 2, d = 2; // default 2x2
  if (dimMatch) {
    w = parseInt(dimMatch[1], 10);
    d = parseInt(dimMatch[2], 10);
    // LEGO convention: first number is shorter side, but "1x4" means 1-wide 4-long
    // Keep as-is since that matches the name
  }

  // Determine height from category/name
  let h = BRICK_H;
  if (cat.includes('baseplate') || name.includes('baseplate')) {
    return [Math.max(w, 8) * STUD, 0.2, Math.max(d, 8) * STUD];
  }
  if (cat.includes('plate') || name.includes('plate')) {
    h = PLATE_H;
  } else if (cat.includes('tile') || name.includes('tile')) {
    h = PLATE_H * 0.8;
  } else if (cat.includes('slope') || name.includes('slope')) {
    h = BRICK_H;
  } else if (name.includes('window') || name.includes('door')) {
    // Windows/doors are typically taller
    h = BRICK_H * 2;
  }

  return [w * STUD, h, d * STUD];
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

  // Add studs on top — one per stud position
  const studs: [number, number, number][] = [];
  const nStudsX = Math.max(1, Math.round(size[0] / STUD));
  const nStudsZ = Math.max(1, Math.round(size[2] / STUD));
  if (size[1] > 0.3) {
    for (let sx = 0; sx < nStudsX; sx++) {
      for (let sz = 0; sz < nStudsZ; sz++) {
        studs.push([
          position[0] + (sx - (nStudsX - 1) / 2) * STUD,
          position[1] + size[1] / 2 + 0.075,
          position[2] + (sz - (nStudsZ - 1) / 2) * STUD,
        ]);
      }
    }
  }

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
      {studs.map((sp, i) => (
        <Stud
          key={i}
          position={sp}
          color={color}
          opacity={opacity}
          isTrans={isTrans}
          wireframe={wireframe}
        />
      ))}
    </group>
  );
}
