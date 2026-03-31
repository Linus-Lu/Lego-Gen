import React from 'react';

const ProgressIndicator: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center h-full">
      <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-blue-600 mb-4"></div>
      <p className="text-lg text-gray-600 font-medium">Analyzing Model...</p>
      <p className="text-sm text-gray-400">Calculating optimal build steps</p>
    </div>
  );
};

export default ProgressIndicator;

