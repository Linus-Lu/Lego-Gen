import React, { useCallback, useRef, useState } from 'react';

interface UploadPanelProps {
  onFileSelected: (file: File) => void;
}

const UploadPanel: React.FC<UploadPanelProps> = ({ onFileSelected }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      setFileName(file.name);
      onFileSelected(file);
      e.dataTransfer.clearData();
    }
  }, [onFileSelected]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setFileName(file.name);
      onFileSelected(file);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div
      className="relative border border-dashed border-gray-600 bg-gray-800 rounded-xl p-3 text-center hover:border-blue-500 hover:bg-gray-700 transition cursor-pointer group h-full flex flex-col items-center justify-center min-w-[80px] md:min-w-[100px]"
      role="button"
      tabIndex={0}
      aria-label="Upload an image by clicking or dragging"
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      onClick={handleClick}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleClick(); } }}
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
        accept="image/*"
        aria-label="Select image file"
      />
      <div className="text-2xl mb-1 group-hover:scale-110 transition-transform">📷</div>
      {fileName ? (
        <div className="w-full">
            <p className="text-[10px] text-blue-400 font-medium truncate max-w-full px-1">{fileName}</p>
        </div>
      ) : (
        <div className="hidden md:block">
            <p className="text-[10px] text-gray-400 font-medium">Upload</p>
        </div>
      )}
    </div>
  );
};

export default UploadPanel;
