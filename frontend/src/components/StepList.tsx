import type { BuildStep } from '../api/legogen';

interface StepListProps {
  steps: BuildStep[];
  currentStep: number;
  onStepSelect: (stepNumber: number) => void;
}

const StepList: React.FC<StepListProps> = ({ steps, currentStep, onStepSelect }) => {
  return (
    <div className="space-y-2 h-full overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-gray-700">
      <h3 className="font-bold text-gray-300 mb-2 sticky top-0 bg-gray-900 pb-2 z-10">
        Instructions ({steps.length} steps)
      </h3>
      {steps.map((step) => {
        const isActive = step.step_number === currentStep;
        const isDone = step.step_number < currentStep;

        // Get unique colors for this step (max 5 dots)
        const uniqueColors = [...new Set(step.parts.map((p) => p.color_hex))].slice(0, 5);
        const extraColors = new Set(step.parts.map((p) => p.color_hex)).size - 5;

        return (
          <div
            key={step.step_number}
            onClick={() => onStepSelect(step.step_number)}
            className={`p-3 border rounded-lg cursor-pointer transition flex items-start gap-3 ${
              isActive
                ? 'border-blue-500 bg-blue-900/30 ring-1 ring-blue-500/50'
                : isDone
                ? 'border-gray-600 bg-gray-800/50 opacity-70'
                : 'border-gray-700 bg-gray-800 hover:bg-gray-700'
            }`}
          >
            {/* Step number */}
            <div
              className={`font-bold rounded-full w-8 h-8 flex-shrink-0 flex items-center justify-center border text-sm ${
                isActive
                  ? 'bg-blue-600 text-white border-blue-500'
                  : isDone
                  ? 'bg-green-900/50 text-green-400 border-green-700'
                  : 'bg-gray-700 text-gray-400 border-gray-600'
              }`}
            >
              {isDone ? '✓' : step.step_number}
            </div>

            {/* Content */}
            <div className="flex-grow min-w-0">
              <p className="text-gray-200 text-sm font-medium truncate">{step.title}</p>
              <div className="flex items-center gap-2 mt-1.5">
                {/* Color dots */}
                <div className="flex gap-1">
                  {uniqueColors.map((hex, i) => (
                    <div
                      key={i}
                      className="w-3 h-3 rounded-full border border-white/20"
                      style={{ backgroundColor: hex }}
                      title={step.parts.find((p) => p.color_hex === hex)?.color}
                    />
                  ))}
                  {extraColors > 0 && (
                    <span className="text-[10px] text-gray-500 ml-0.5">+{extraColors}</span>
                  )}
                </div>
                {/* Part count */}
                <span className="text-[10px] text-gray-500 ml-auto">
                  {step.part_count} parts
                </span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default StepList;
