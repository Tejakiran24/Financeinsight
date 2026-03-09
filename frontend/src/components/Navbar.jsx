import React from 'react';
import { NavLink } from 'react-router-dom';
import { Network } from 'lucide-react';
import { motion } from 'framer-motion';

const Navbar = () => {
  return (
    <nav className="sticky top-6 z-50 px-4">
      <div className="max-w-4xl mx-auto glass-pill py-3 px-6 flex items-center justify-between">
        
        <NavLink to="/" className="flex items-center space-x-3 group">
          <motion.div whileHover={{ rotate: 180 }} transition={{ duration: 0.6 }}>
            <Network className="text-primary w-8 h-8 group-hover:text-secondary transition-colors" />
          </motion.div>
          <span className="font-bold text-xl tracking-tight text-white hover:text-primary transition-colors">
            Finance<span className="text-slate-400 font-light tracking-wide">Insight</span>
          </span>
        </NavLink>

        <div className="hidden md:flex items-center space-x-1">
          {[
            { name: 'Home', path: '/' },
            { name: 'Analyzer', path: '/analyzer' },
            { name: 'Dashboard', path: '/dashboard' },
            { name: 'About', path: '/about' }
          ].map((link, idx) => (
            <NavLink 
              key={idx} 
              to={link.path} 
              className={({isActive}) => `
                relative px-5 py-2 rounded-full text-sm font-medium transition-all duration-300
                ${isActive ? 'text-white' : 'text-slate-400 hover:text-white hover:bg-white/5'}
              `}
            >
              {({isActive}) => (
                <>
                  {isActive && (
                    <motion.div
                      layoutId="nav-pill"
                      className="absolute inset-0 bg-primary/20 rounded-full -z-10"
                      transition={{ type: "spring", stiffness: 350, damping: 30 }}
                    />
                  )}
                  {link.name}
                </>
              )}
            </NavLink>
          ))}
        </div>
        
      </div>
    </nav>
  );
};

export default Navbar;
