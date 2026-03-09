import React, { useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import { UploadCloud, FileText, Loader2, Play } from 'lucide-react';

const rawApiUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:8001";
const API_URL = rawApiUrl.startsWith('http') ? rawApiUrl : `https://${rawApiUrl}`;

const Analyzer = () => {
    const [text, setText] = useState('');
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    const handleAnalyze = async () => {
        if (!text && !file) {
            setError('Please provide text or a file to analyze.');
            return;
        }

        setLoading(true);
        setError('');
        setResult(null);

        try {
            const formData = new FormData();
            if (file) {
                formData.append('file', file);
            } else {
                formData.append('text', text);
            }

            const res = await axios.post(`${API_URL}/analyze`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            
            setResult(res.data);
        } catch (err) {
            setError(err.response?.data?.detail || 'An error occurred during analysis. Make sure the backend is running.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center">
                <h1 className="text-5xl font-extrabold mb-4 glow-text text-white tracking-tight">Financial Document Analyzer</h1>
                <p className="text-gray-400 mb-10 text-center max-w-2xl text-lg">
                    Upload your earnings report, SEC filing, or paste raw text below to automatically extract key financial entities like companies, dates, values, and events.
                </p>

                <div className="w-full max-w-7xl grid grid-cols-1 lg:grid-cols-2 gap-10">
                    {/* Input Section */}
                    <motion.div 
                        initial={{ opacity: 0, x: -30 }} 
                        animate={{ opacity: 1, x: 0 }} 
                        transition={{ duration: 0.5, delay: 0.1 }}
                        className="glass p-8 md:p-10 rounded-3xl flex flex-col space-y-6 shadow-2xl hover:shadow-primary/10 transition-shadow duration-500"
                    >
                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-2">Paste Text</label>
                                <textarea 
                                    className="w-full bg-white/5 border border-primary/20 rounded-xl p-4 text-gray-100 focus:ring-primary focus:border-primary focus:outline-none focus:shadow-[0_0_15px_rgba(0,240,255,0.2)] transition-all resize-none custom-scrollbar"
                                rows="6"
                                placeholder="Paste financial text here..."
                                value={text}
                                onChange={(e) => setText(e.target.value)}
                            />
                        </div>
                        
                        <div className="flex items-center space-x-4">
                            <hr className="flex-1 border-white/10" />
                            <span className="text-gray-500 text-sm">OR</span>
                            <hr className="flex-1 border-white/10" />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-2">Upload File (TXT, PDF)</label>
                            <div className="flex items-center justify-center w-full">
                                <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-primary/20 border-dashed rounded-xl cursor-pointer bg-white/5 hover:bg-white/10 hover:border-primary hover:shadow-[0_0_15px_rgba(0,240,255,0.2)] transition-all">
                                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                                        <UploadCloud className="w-8 h-8 mb-2 text-gray-400" />
                                        <p className="text-sm text-gray-400">
                                            {file ? <span className="text-secondary">{file.name}</span> : <span>Click to upload or drag and drop</span>}
                                        </p>
                                    </div>
                                    <input type="file" className="hidden" onChange={handleFileChange} />
                                </label>
                            </div>
                        </div>

                        {error && <p className="text-red-400 text-sm text-center bg-red-400/10 py-2 rounded-lg">{error}</p>}

                        <button 
                            onClick={handleAnalyze} 
                            disabled={loading}
                            className="w-full flex justify-center items-center py-4 px-4 border border-transparent rounded-xl shadow-lg text-lg font-bold text-white bg-gradient-to-r from-primary via-secondary to-accent hover:opacity-90 button-glow focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary focus:ring-offset-darkBg disabled:opacity-50 transition-all duration-300"
                        >
                            {loading ? <Loader2 className="w-6 h-6 animate-spin mr-2" /> : <Play className="w-6 h-6 mr-2" />}
                            {loading ? 'Analyzing Neural Networks...' : 'Analyze Document'}
                        </button>
                    </motion.div>

                    {/* Results Section */}
                    <motion.div 
                        initial={{ opacity: 0, x: 30 }} 
                        animate={{ opacity: 1, x: 0 }} 
                        transition={{ duration: 0.5, delay: 0.2 }}
                        className="glass p-8 md:p-10 rounded-3xl flex flex-col min-h-[600px] shadow-2xl hover:shadow-secondary/10 transition-shadow duration-500"
                    >
                        <h2 className="text-3xl font-bold mb-6 flex items-center">
                            <FileText className="w-8 h-8 mr-3 text-primary animate-glow" />
                            Extraction Results
                        </h2>
                        
                        {!result ? (
                            <div className="flex-1 flex flex-col items-center justify-center text-gray-500">
                                <FileText className="w-16 h-16 mb-4 opacity-50" />
                                <p>No analysis performed yet.</p>
                                <p className="text-sm mt-2">Submit text to see NER results here.</p>
                            </div>
                        ) : (
                            <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
                                <div className="space-y-4">
                                    {result.entities?.length === 0 ? (
                                        <p className="text-center text-gray-400 mt-10">No financial entities detected.</p>
                                    ) : (
                                        result.entities?.map((ent, idx) => (
                                            <motion.div 
                                                initial={{ opacity: 0, y: 20 }}
                                                animate={{ opacity: 1, y: 0 }}
                                                transition={{ delay: idx * 0.05, type: 'spring', stiffness: 100 }}
                                                whileHover={{ scale: 1.02 }}
                                                key={idx} 
                                                className="bg-white/5 backdrop-blur-md border border-white/10 p-5 rounded-2xl flex justify-between items-center hover:bg-white/10 hover:border-primary/30 transition-all duration-300 shadow-lg"
                                            >
                                                <span className="font-semibold text-white text-lg truncate mr-4">{ent.text}</span>
                                                <span className={`px-3 py-1 text-xs font-bold rounded-full border 
                                                    ${ent.type === 'COMPANY' ? 'bg-primary/20 text-primary border-primary/30' : 
                                                    ent.type === 'FINANCE' ? 'bg-secondary/20 text-secondary border-secondary/30' : 
                                                    ent.type === 'FINANCIAL_EVENT' ? 'bg-accent/20 text-accent border-accent/30' :
                                                    ent.type === 'FINANCIAL_METRIC' ? 'bg-blue-500/20 text-blue-300 border-blue-500/30' :
                                                    'bg-gray-500/20 text-gray-300 border-gray-500/30'}`}>
                                                    {ent.type}
                                                </span>
                                            </motion.div>
                                        ))
                                    )}
                                </div>
                            </div>
                        )}
                    </motion.div>
                </div>
            </motion.div>
        </div>
    );
};

export default Analyzer;
