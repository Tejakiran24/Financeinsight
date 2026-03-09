import React, { useEffect } from 'react';
import { motion, useMotionValue, useSpring } from 'framer-motion';

const CursorGlow = () => {
  const cursorX = useMotionValue(-100);
  const cursorY = useMotionValue(-100);

  const springConfig = { damping: 25, stiffness: 150 };
  const cursorXSpring = useSpring(cursorX, springConfig);
  const cursorYSpring = useSpring(cursorY, springConfig);

  useEffect(() => {
    const moveCursor = (e) => {
      cursorX.set(e.clientX - 200); // Offset by half the width to center
      cursorY.set(e.clientY - 200); // Offset by half the height to center
    };
    
    window.addEventListener('mousemove', moveCursor);
    
    return () => {
      window.removeEventListener('mousemove', moveCursor);
    };
  }, [cursorX, cursorY]);

  return (
    <motion.div
      className="pointer-events-none fixed top-0 left-0 w-[400px] h-[400px] rounded-full z-[1] mix-blend-screen opacity-70"
      style={{
        translateX: cursorXSpring,
        translateY: cursorYSpring,
        background: 'radial-gradient(circle, rgba(0, 240, 255, 0.4) 0%, rgba(123, 97, 255, 0.2) 30%, transparent 70%)',
        filter: 'blur(50px)',
      }}
    />
  );
};

export default CursorGlow;
