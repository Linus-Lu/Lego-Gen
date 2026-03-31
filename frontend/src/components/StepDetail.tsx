import type { BuildStep } from '../api/legogen';

interface StepDetailProps {
  step: BuildStep | null;
}

const StepDetail: React.FC<StepDetailProps> = ({ step }) => {
  if (!step) {
    return (
      <div className="p-3 text-gray-500 text-center text-sm">
        Select a step to view details
      </div>
    );
  }

  return (
    <div className="p-3 space-y-3 overflow-y-auto">
      <div>
        <h4 className="font-bold text-gray-200 text-sm">{step.title}</h4>
        <p className="text-gray-400 text-xs mt-1">{step.instruction}</p>
      </div>

      {/* Parts table */}
      <div className="space-y-1.5">
        <h5 className="text-[10px] uppercase tracking-wider text-gray-500 font-semibold">
          Parts ({step.part_count})
        </h5>
        {step.parts.map((part, i) => (
          <div
            key={`${part.part_id}-${part.color_hex}-${i}`}
            className="flex items-center gap-2 px-2 py-1.5 bg-gray-800/50 rounded text-xs"
          >
            {/* Color swatch */}
            <div
              className={`w-5 h-5 rounded flex-shrink-0 border border-white/15 ${
                part.is_trans ? 'opacity-60' : ''
              }`}
              style={{ backgroundColor: part.color_hex }}
              title={part.color}
            />

            {/* Name + ID */}
            <div className="flex-grow min-w-0">
              <span className="text-gray-200 truncate block">{part.name}</span>
              <span className="text-gray-500 text-[10px]">
                {part.part_id}
                {part.is_trans && (
                  <span className="ml-1 text-blue-400/70">transparent</span>
                )}
              </span>
            </div>

            {/* Quantity */}
            <span className="text-gray-300 font-mono bg-gray-700 px-1.5 py-0.5 rounded text-[10px] flex-shrink-0">
              x{part.quantity}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default StepDetail;
