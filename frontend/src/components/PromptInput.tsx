import React from 'react';

const PromptInput: React.FC = () => {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">
        Text Description
      </label>
      <textarea
        className="w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition"
        rows={4}
        placeholder="E.g., A small red house with a yellow roof..."
      />
    </div>
  );
};

export default PromptInput;

