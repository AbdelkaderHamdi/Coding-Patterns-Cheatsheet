
import React, { useState } from 'react';
import { CODING_PATTERNS } from './constants/patterns';
import { PatternCard } from './components/PatternCard';

const App: React.FC = () => {
  const [activePatternId, setActivePatternId] = useState<number | null>(null);

  const handlePatternClick = (id: number) => {
    setActivePatternId(prevId => (prevId === id ? null : id));
  };

  return (
    <div className="min-h-screen text-gray-800 dark:text-gray-200 font-sans">
      <header className="bg-gradient-to-r from-purple-700 to-indigo-800 text-white p-8 text-center shadow-md">
        <h1 className="text-4xl md:text-5xl font-extrabold mb-2">🎯 Coding Patterns Cheatsheet</h1>
        <p className="text-lg md:text-xl text-indigo-200">Master 20 patterns, solve unlimited problems</p>
      </header>
      
      <main className="container mx-auto p-4 md:p-8">
        <div className="grid grid-cols-[repeat(auto-fit,minmax(350px,1fr))] gap-8">
          {CODING_PATTERNS.map(pattern => (
            <PatternCard 
              key={pattern.id}
              pattern={pattern}
              isActive={activePatternId === pattern.id}
              onClick={() => handlePatternClick(pattern.id)}
            />
          ))}
        </div>
      </main>

      <footer className="text-center p-8 mt-8 border-t border-gray-200 dark:border-gray-700">
        <p className="text-gray-600 dark:text-gray-400 mb-2">"The beautiful thing about learning is that nobody can take it away from you." - B.B. King</p>
        <p className="text-sm text-gray-500 dark:text-gray-500">Focus on the pattern, not just the problem.</p>
      </footer>
    </div>
  );
};

export default App;
