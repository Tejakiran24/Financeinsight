import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Analyzer from './pages/Analyzer';
import Dashboard from './pages/Dashboard';
import About from './pages/About';

import CursorGlow from './components/CursorGlow';
import AnimatedLines from './components/AnimatedLines';

function App() {
  return (
    <Router>
      <div className="bg-blobs">
        <div className="blob-1" />
        <div className="blob-2" />
        <div className="blob-3" />
      </div>

      <div className="min-h-screen text-slate-200 font-sans flex flex-col relative z-10 selection:bg-primary/30 selection:text-white">
        <CursorGlow />
        <AnimatedLines />
        <Navbar />
        <main className="flex-grow pb-12">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/analyzer" element={<Analyzer />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
        
        {/* Footer */}
        <footer className="w-full py-6 px-8 border-t border-white/5 backdrop-blur-md bg-white/5 mt-auto">
          <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center text-sm text-gray-400">
            <div className="mb-4 md:mb-0">
              <span className="font-bold text-white tracking-widest">FinanceInsight MVP</span>
            </div>
            <div className="flex space-x-8">
              <a href="#" className="hover:text-primary hover:glow-text transition-all">Customer Care</a>
              <span>&copy; {new Date().getFullYear()} All rights reserved.</span>
            </div>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;
