const STUD = 1.0;

interface BrickMeshProps {
  /** Center position in scene units. */
  position: [number, number, number];
  /** Box size in scene units. */
  size: [number, number, number];
  color: string;
  opacity?: number;
  /** Render as a wireframe outline (ghost-preview mode). */
  wireframe?: boolean;
}

/** Simple LEGO brick mesh: body + top studs. */
export default function BrickMesh({
  position, size, color, opacity = 1, wireframe = false,
}: BrickMeshProps) {
  const translucent = opacity < 1;

  const studs: [number, number, number][] = [];
  const nx = Math.max(1, Math.round(size[0] / STUD));
  const nz = Math.max(1, Math.round(size[2] / STUD));
  if (size[1] > 0.3) {
    for (let sx = 0; sx < nx; sx++) {
      for (let sz = 0; sz < nz; sz++) {
        studs.push([
          position[0] + (sx - (nx - 1) / 2) * STUD,
          position[1] + size[1] / 2 + 0.08,
          position[2] + (sz - (nz - 1) / 2) * STUD,
        ]);
      }
    }
  }

  return (
    <group>
      <mesh position={position} castShadow receiveShadow>
        <boxGeometry args={size} />
        <meshStandardMaterial
          color={color}
          roughness={0.42}
          metalness={0.04}
          transparent={translucent}
          opacity={opacity}
          wireframe={wireframe}
        />
      </mesh>
      {studs.map((sp, i) => (
        <mesh key={i} position={sp} castShadow>
          <cylinderGeometry args={[0.2, 0.2, 0.16, 14]} />
          <meshStandardMaterial
            color={color}
            roughness={0.42}
            metalness={0.04}
            transparent={translucent}
            opacity={opacity}
            wireframe={wireframe}
          />
        </mesh>
      ))}
    </group>
  );
}
