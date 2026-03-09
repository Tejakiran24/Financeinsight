import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FileSearch, TrendingUp, PieChart, ShieldCheck } from 'lucide-react';

const Home = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 flex flex-col items-center">
      <motion.div 
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center"
      >
        <h1 className="text-5xl md:text-7xl font-extrabold mb-6 tracking-tight">
          Uncover Insights in <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-secondary via-blue-400 to-accent">
            Financial Data
          </span>
        </h1>
        <p className="mt-4 max-w-2xl mx-auto text-xl text-gray-400">
          AI-powered Named Entity Recognition designed specifically for financial documents, SEC filings, earnings reports, and news.
        </p>
        <div className="mt-10 flex justify-center space-x-6">
          <Link to="/analyzer" className="px-8 py-4 bg-gradient-to-r from-primary to-accent hover:from-blue-700 hover:to-purple-500 text-white rounded-xl font-bold text-lg shadow-lg shadow-primary/30 transform hover:-translate-y-1 transition-all duration-300">
            Start Analyzing
          </Link>
          <Link to="/about" className="px-8 py-4 glass text-white rounded-xl font-bold text-lg hover:bg-white/10 transition-all duration-300">
            Learn More
          </Link>
        </div>
      </motion.div>

      <div className="mt-32 grid grid-cols-1 md:grid-cols-3 gap-8 w-full">
        {[
          { icon: FileSearch, title: 'Smart Extraction', desc: 'Automatically identify companies, monetary values, percentages, dates, and financial events.' },
          { icon: TrendingUp, title: 'Event Detection', desc: 'Detect M&A activities, IPO announcements, stock splits, and earnings calls instantly.' },
          { icon: ShieldCheck, title: 'High Accuracy', desc: 'Powered by advanced NLP models like spaCy and Transformers to ensure precise results.' },
        ].map((feature, idx) => (
          <motion.div 
            key={idx}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 * idx, duration: 0.5 }}
            className="glass p-8 rounded-2xl flex flex-col items-center text-center hover:border-secondary/50 transition-colors"
          >
            <div className="p-4 bg-primary/20 rounded-full mb-6">
              <feature.icon className="w-10 h-10 text-secondary" />
            </div>
            <h3 className="text-xl font-bold mb-3">{feature.title}</h3>
            <p className="text-gray-400">{feature.desc}</p>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

export default Home;
