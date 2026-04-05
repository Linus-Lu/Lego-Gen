import { useState, useEffect } from 'react';
import GalleryCard from './GalleryCard';
import { listGalleryBuilds } from '../api/legogen';
import type { GalleryBuild } from '../api/legogen';

export default function GalleryTab() {
  const [builds, setBuilds] = useState<GalleryBuild[]>([]);
  const [loading, setLoading] = useState(true);
  const [category, setCategory] = useState('');
  const [sort, setSort] = useState('newest');
  const [search, setSearch] = useState('');

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const data = await listGalleryBuilds({
          category: category || undefined,
          sort,
          q: search || undefined,
        });
        setBuilds(data);
      } catch {
        setBuilds([]);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [category, sort, search]);

  const handleStarUpdate = (updated: GalleryBuild) => {
    setBuilds((prev) => prev.map((b) => (b.id === updated.id ? updated : b)));
  };

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3">
        <select
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className="bg-gray-800 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-gray-200 focus:outline-none focus:ring-1 focus:ring-blue-500/30"
        >
          <option value="">All Categories</option>
          <option value="vehicle">Vehicle</option>
          <option value="building">Building</option>
          <option value="animal">Animal</option>
          <option value="character">Character</option>
          <option value="nature">Nature</option>
          <option value="space">Space</option>
        </select>

        <select
          value={sort}
          onChange={(e) => setSort(e.target.value)}
          className="bg-gray-800 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-gray-200 focus:outline-none focus:ring-1 focus:ring-blue-500/30"
        >
          <option value="newest">Newest</option>
          <option value="stars">Top Rated</option>
          <option value="parts">Most Parts</option>
        </select>

        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search builds..."
          className="bg-gray-800 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-blue-500/30 flex-grow max-w-xs"
        />
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" />
            <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
            <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
          </div>
        </div>
      ) : builds.length === 0 ? (
        <div className="text-center py-12">
          <div className="text-4xl mb-3 opacity-30">📦</div>
          <p className="text-gray-500 text-sm">No builds in the gallery yet.</p>
          <p className="text-gray-600 text-xs mt-1">Generate a build and save it to get started.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
          {builds.map((build) => (
            <GalleryCard
              key={build.id}
              build={build}
              onStarUpdate={handleStarUpdate}
            />
          ))}
        </div>
      )}
    </div>
  );
}
