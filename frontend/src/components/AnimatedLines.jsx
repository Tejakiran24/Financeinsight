import React from 'react';
import { motion } from 'framer-motion';

const AnimatedLines = () => {
  return (
    <div className="pointer-events-none fixed top-0 left-0 w-full h-full z-[2] overflow-hidden">
      {/* Left Side Lines */}
      {/* Cyan Line */}
      <motion.div
        className="absolute left-[3%] w-[2px] h-[30vh] bg-gradient-to-b from-transparent via-primary to-transparent opacity-80 shadow-[0_0_15px_rgba(0,240,255,0.9)]"
        animate={{ top: ['-30vh', '110vh'] }}
        transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
      />
      {/* Pink Line */}
      <motion.div
        className="absolute left-[7%] w-[1px] h-[20vh] bg-gradient-to-b from-transparent via-accent to-transparent opacity-50 shadow-[0_0_15px_rgba(255,0,85,0.9)]"
        animate={{ top: ['-20vh', '110vh'] }}
        transition={{ duration: 7, repeat: Infinity, ease: "linear", delay: 2 }}
      />
      {/* Purple Line moving UP */}
      <motion.div
        className="absolute left-[1%] w-[1px] h-[40vh] bg-gradient-to-t from-transparent via-secondary to-transparent opacity-60 shadow-[0_0_15px_rgba(123,97,255,0.9)]"
        animate={{ bottom: ['-40vh', '110vh'] }}
        transition={{ duration: 6, repeat: Infinity, ease: "linear", delay: 1 }}
      />

      {/* Right Side Lines */}
      {/* Purple Line */}
      <motion.div
        className="absolute right-[3%] w-[2px] h-[35vh] bg-gradient-to-b from-transparent via-secondary to-transparent opacity-80 shadow-[0_0_15px_rgba(123,97,255,0.9)]"
        animate={{ top: ['-35vh', '110vh'] }}
        transition={{ duration: 5, repeat: Infinity, ease: "linear", delay: 1.5 }}
      />
      {/* Cyan Line */}
      <motion.div
        className="absolute right-[6%] w-[1px] h-[25vh] bg-gradient-to-b from-transparent via-primary to-transparent opacity-60 shadow-[0_0_15px_rgba(0,240,255,0.9)]"
        animate={{ top: ['-25vh', '110vh'] }}
        transition={{ duration: 8, repeat: Infinity, ease: "linear", delay: 0.5 }}
      />
      {/* Pink Line moving UP */}
      <motion.div
        className="absolute right-[1%] w-[2px] h-[20vh] bg-gradient-to-t from-transparent via-accent to-transparent opacity-70 shadow-[0_0_15px_rgba(255,0,85,0.9)]"
        animate={{ bottom: ['-20vh', '110vh'] }}
        transition={{ duration: 5.5, repeat: Infinity, ease: "linear", delay: 3 }}
      />
    </div>
  );
};

export default AnimatedLines;
