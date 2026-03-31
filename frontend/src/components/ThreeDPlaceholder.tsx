import React from 'react';

const ThreeDPlaceholder: React.FC = () => {
  return (
    <div className="w-full h-full bg-gray-800 rounded-lg flex items-center justify-center min-h-[300px] border-2 border-dashed border-gray-700">
      <div className="text-center text-gray-500">
        <div className="text-5xl mb-2 opacity-50">🧊</div>
        <p className="font-medium">3D Interactive View</p>
        <p className="text-xs mt-2">Drag to rotate • Scroll to zoom</p>
      </div>
    </div>
  );
};

export default ThreeDPlaceholder;
