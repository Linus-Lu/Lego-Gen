import { useState } from 'react';
import Header from '../components/Header';
import CompareTab from '../components/CompareTab';
import GalleryTab from '../components/GalleryTab';

export default function ExplorePage() {
  const [activeTab, setActiveTab] = useState<'compare' | 'gallery'>('gallery');

  return (
    <div className="flex flex-col min-h-screen bg-gray-950 text-gray-100">
      <Header />

      <main className="flex-grow">
        <div className="absolute inset-0 bg-mesh pointer-events-none" />
        <div className="relative max-w-6xl mx-auto px-4 py-6">
          {/* Page header */}
          <div className="mb-6">
            <h1 className="text-2xl font-bold text-gradient mb-1">Explore</h1>
            <p className="text-sm text-gray-500">Browse the gallery or compare model checkpoints</p>
          </div>

          {/* Tab bar */}
          <div className="flex gap-1 border-b border-white/5 pb-0 mb-6" role="tablist">
            <button
              role="tab"
              aria-selected={activeTab === 'gallery'}
              onClick={() => setActiveTab('gallery')}
              className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-px ${
                activeTab === 'gallery'
                  ? 'text-blue-300 border-blue-500'
                  : 'text-gray-500 border-transparent hover:text-gray-300'
              }`}
            >
              Gallery
            </button>
            <button
              role="tab"
              aria-selected={activeTab === 'compare'}
              onClick={() => setActiveTab('compare')}
              className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-px ${
                activeTab === 'compare'
                  ? 'text-blue-300 border-blue-500'
                  : 'text-gray-500 border-transparent hover:text-gray-300'
              }`}
            >
              Compare
            </button>
          </div>

          {/* Tab content */}
          {activeTab === 'gallery' ? <GalleryTab /> : <CompareTab />}
        </div>
      </main>
    </div>
  );
}
