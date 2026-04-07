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

// ── Color fallback map (LEGO color name → hex) ─────────────────────
// Used when model output is missing color_hex (truncation, malformed JSON)
const COLOR_NAME_TO_HEX: Record<string, string> = {
  'black': '#05131D', 'blue': '#0055BF', 'bright green': '#4B9F4A',
  'bright light blue': '#9FC3E9', 'bright light orange': '#F8BB3D',
  'bright light yellow': '#FFF03A', 'bright orange': '#FE8A18',
  'bright pink': '#E4ADC8', 'coral': '#FF698F', 'dark azure': '#078BC9',
  'dark blue': '#143044', 'dark bluish gray': '#6C6E68',
  'dark brown': '#352100', 'dark green': '#184632',
  'dark orange': '#A95500', 'dark pink': '#C870A0',
  'dark red': '#720E0F', 'dark tan': '#958A73',
  'dark turquoise': '#008F9B', 'green': '#237841', 'lavender': '#E1D5ED',
  'light aqua': '#ADC3C0', 'light bluish gray': '#A0A5A9',
  'light gray': '#9BA19D', 'light nougat': '#FCC39E',
  'lime': '#BBE90B', 'magenta': '#923978', 'medium azure': '#36AEBF',
  'medium blue': '#5A93DB', 'medium lavender': '#AC78BA',
  'medium nougat': '#AA7D55', 'nougat': '#D09168',
  'olive green': '#9B9A5A', 'orange': '#FE8A18',
  'pearl gold': '#AA7F2E', 'red': '#C91A09', 'reddish brown': '#582A12',
  'sand blue': '#596072', 'sand green': '#A0BCAC',
  'tan': '#E4CD9E', 'teal': '#008F9B',
  'trans-clear': '#FCFCFC', 'trans-light blue': '#C1DFF0',
  'trans-red': '#C91A09', 'trans-green': '#84B68D',
  'trans-orange': '#F08F1C', 'trans-yellow': '#F5CD2F',
  'white': '#FFFFFF', 'yellow': '#F2CD37',
  'bright orange': '#FE8A18', 'dark azure': '#078BC9',
};

/** Resolve a brick color to a hex string, with fallback from color name. */
export function resolveColorHex(part: Part): string {
  if (part.color_hex) return part.color_hex;
  const name = (part.color ?? '').toLowerCase().trim();
  return COLOR_NAME_TO_HEX[name] ?? '#A0A5A9'; // fallback to light bluish gray
}

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
