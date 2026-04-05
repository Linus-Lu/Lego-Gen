import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { starGalleryBuild } from '../api/legogen';
import type { GalleryBuild } from '../api/legogen';

interface GalleryCardProps {
  build: GalleryBuild;
  onStarUpdate?: (build: GalleryBuild) => void;
}

export default function GalleryCard({ build, onStarUpdate }: GalleryCardProps) {
  const navigate = useNavigate();
  const [hoveredStar, setHoveredStar] = useState(0);

  const handleStar = async (stars: number) => {
    try {
      const updated = await starGalleryBuild(build.id, stars);
      onStarUpdate?.(updated);
    } catch {
      // silent fail
    }
  };

  const complexityColor: Record<string, string> = {
    simple: 'bg-green-500/10 border-green-500/20 text-green-300',
    medium: 'bg-yellow-500/10 border-yellow-500/20 text-yellow-300',
    complex: 'bg-red-500/10 border-red-500/20 text-red-300',
  };

  return (
    <div className="glass rounded-xl overflow-hidden group hover:ring-1 hover:ring-white/10 transition-all">
      {/* Thumbnail */}
      <div className="aspect-video bg-gray-800 relative overflow-hidden">
        {build.thumbnail_b64 ? (
          <img
            src={`data:image/png;base64,${build.thumbnail_b64}`}
            alt={build.title}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-700">
            <svg className="w-10 h-10" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
              <path strokeLinecap="round" strokeLinejoin="round" d="m21 7.5-9-5.25L3 7.5m18 0-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25m0-9v9" />
            </svg>
          </div>
        )}
        {/* Category badge */}
        {build.category && (
          <span className="absolute top-2 left-2 px-2 py-0.5 rounded-md bg-black/60 text-[10px] font-medium text-gray-300 backdrop-blur-sm">
            {build.category}
          </span>
        )}
      </div>

      {/* Content */}
      <div className="p-3 space-y-2">
        <h3 className="font-semibold text-sm text-gray-200 truncate">{build.title}</h3>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs">
            <span className="text-gray-500">{build.parts_count} parts</span>
            <span className={`px-1.5 py-0.5 rounded border text-[10px] ${complexityColor[build.complexity] ?? 'bg-gray-500/10 border-gray-500/20 text-gray-400'}`}>
              {build.complexity}
            </span>
          </div>

          {/* Star rating */}
          <div className="flex items-center gap-0.5" onMouseLeave={() => setHoveredStar(0)}>
            {[1, 2, 3, 4, 5].map((s) => (
              <button
                key={s}
                onClick={() => handleStar(s)}
                onMouseEnter={() => setHoveredStar(s)}
                className="text-sm transition-transform hover:scale-110"
                aria-label={`Rate ${s} stars`}
              >
                <span className={s <= (hoveredStar || Math.round(build.stars)) ? 'text-yellow-400' : 'text-gray-700'}>
                  ★
                </span>
              </button>
            ))}
            {build.star_count > 0 && (
              <span className="text-[10px] text-gray-600 ml-1">({build.star_count})</span>
            )}
          </div>
        </div>

        {/* Build This button */}
        <button
          onClick={() => navigate(`/guide/${build.id}`)}
          className="w-full py-1.5 rounded-lg bg-blue-600/80 hover:bg-blue-500 text-xs font-medium transition opacity-0 group-hover:opacity-100"
        >
          Build This
        </button>
      </div>
    </div>
  );
}
