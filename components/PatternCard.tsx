
import React, { useState } from 'react';
import type { Pattern, Problem } from '../types';

interface PatternCardProps {
  pattern: Pattern;
  isActive: boolean;
  onClick: () => void;
}

const ChevronDownIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
  </svg>
);

const ProblemItem: React.FC<{ problem: Problem, index: number }> = ({ problem, index }) => {
  const [isSolutionVisible, setIsSolutionVisible] = useState(false);

  return (
    <div className="border-t border-gray-200 dark:border-gray-700">
      <button
        onClick={() => setIsSolutionVisible(!isSolutionVisible)}
        className="w-full text-left p-4 flex justify-between items-center hover:bg-gray-50 dark:hover:bg-gray-700/50 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-opacity-50"
      >
        <span className="font-semibold text-gray-800 dark:text-gray-200">{`${index + 1}. ${problem.title}`}</span>
        <ChevronDownIcon className={`w-5 h-5 text-gray-500 dark:text-gray-400 transform transition-transform duration-300 ${isSolutionVisible ? 'rotate-180' : ''}`} />
      </button>
      <div className={`transition-all duration-500 ease-in-out overflow-hidden ${isSolutionVisible ? 'max-h-screen' : 'max-h-0'}`}>
        <div className="p-4 bg-gray-100 dark:bg-gray-900/50">
          <pre className="bg-gray-200 dark:bg-gray-800 text-gray-800 dark:text-gray-200 p-4 rounded-md overflow-x-auto text-sm">
            <code>{problem.pseudoCode}</code>
          </pre>
        </div>
      </div>
    </div>
  );
};


export const PatternCard: React.FC<PatternCardProps> = ({ pattern, isActive, onClick }) => {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden transition-all duration-500 ease-in-out transform hover:-translate-y-1 hover:shadow-2xl">
      <div className="p-6 cursor-pointer" onClick={onClick}>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">{pattern.name}</h2>
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">{pattern.example}</p>
        
        <div className="space-y-3 text-gray-700 dark:text-gray-300">
            <div>
                <h3 className="font-semibold text-purple-600 dark:text-purple-400">When to use:</h3>
                <p className="text-sm">{pattern.whenToUse}</p>
            </div>
            <div>
                <h3 className="font-semibold text-purple-600 dark:text-purple-400">Approach:</h3>
                <p className="text-sm">{pattern.approach}</p>
            </div>
        </div>
      </div>

      <div className={`transition-all duration-500 ease-in-out overflow-hidden ${isActive ? 'max-h-screen' : 'max-h-0'}`}>
        <div className="bg-gray-50 dark:bg-gray-800/50">
          {pattern.problems.map((problem, index) => (
            <ProblemItem key={index} problem={problem} index={index} />
          ))}
        </div>
      </div>
    </div>
  );
};
