import React from 'react';
import { Server, Layout, Database, Bot } from 'lucide-react';
import { motion } from 'framer-motion';

const About = () => {
    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 flex flex-col items-center">
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center max-w-3xl mb-16">
                <h1 className="text-4xl font-extrabold mb-6">About FinanceInsight</h1>
                <p className="text-xl text-gray-400">
                    FinanceInsight is an AI-powered platform designed to extract structured 
                    financial data from unstructured text such as earnings reports, news, 
                    and SEC filings using advanced Named Entity Recognition (NER).
                </p>
            </motion.div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-12 w-full max-w-5xl">
                <motion.div initial={{ opacity: 0, x: -30 }} animate={{ opacity: 1, x: 0 }} className="glass p-8 rounded-2xl relative overflow-hidden group">
                    <div className="absolute inset-0 bg-gradient-to-br from-primary/10 to-transparent group-hover:from-primary/20 transition-all duration-500" />
                    <h2 className="text-2xl font-bold mb-6 flex items-center relative z-10">
                        <Layout className="w-6 h-6 mr-3 text-secondary" />
                        Modern Frontend
                    </h2>
                    <ul className="space-y-4 text-gray-300 relative z-10">
                        <li className="flex items-center"><div className="w-2 h-2 rounded-full bg-secondary mr-3" /> React.js Engine</li>
                        <li className="flex items-center"><div className="w-2 h-2 rounded-full bg-secondary mr-3" /> Tailwind CSS Styling</li>
                        <li className="flex items-center"><div className="w-2 h-2 rounded-full bg-secondary mr-3" /> Framer Motion Animations</li>
                        <li className="flex items-center"><div className="w-2 h-2 rounded-full bg-secondary mr-3" /> Recharts Data Visualization</li>
                        <li className="flex items-center"><div className="w-2 h-2 rounded-full bg-secondary mr-3" /> Axios Integration</li>
                    </ul>
                </motion.div>

                <motion.div initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} className="glass p-8 rounded-2xl relative overflow-hidden group">
                    <div className="absolute inset-0 bg-gradient-to-bl from-accent/10 to-transparent group-hover:from-accent/20 transition-all duration-500" />
                    <h2 className="text-2xl font-bold mb-6 flex items-center relative z-10">
                        <Server className="w-6 h-6 mr-3 text-accent" />
                        Robust Backend API
                    </h2>
                    <ul className="space-y-4 text-gray-300 relative z-10">
                        <li className="flex items-center"><div className="w-2 h-2 rounded-full bg-accent mr-3" /> Python FastAPI</li>
                        <li className="flex items-center"><div className="w-2 h-2 rounded-full bg-accent mr-3" /> Uvicorn ASGI Server</li>
                        <li className="flex items-center"><div className="w-2 h-2 rounded-full bg-accent mr-3" /> SpaCy NLP Processing</li>
                        <li className="flex items-center"><div className="w-2 h-2 rounded-full bg-accent mr-3" /> HuggingFace Transformers</li>
                        <li className="flex items-center"><div className="w-2 h-2 rounded-full bg-accent mr-3" /> Regex Rules Engine</li>
                    </ul>
                </motion.div>
                
                <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} className="md:col-span-2 glass p-8 rounded-2xl border border-white/10 mt-6 flex flex-col items-center text-center">
                    <Bot className="w-16 h-16 text-primary mb-4" />
                    <h2 className="text-2xl font-bold mb-3">AI Intelligence</h2>
                    <p className="max-w-2xl text-gray-400">
                        This system processes input strings and extracts categories such as COMPANY, FINANCE, DATE, FINANCIAL_EVENT, and FINANCIAL_METRIC seamlessly, presenting actionable insights for data-driven decisions.
                    </p>
                </motion.div>
            </div>
        </div>
    );
};

export default About;
