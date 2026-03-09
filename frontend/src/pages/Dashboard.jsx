import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend, BarChart, Bar, XAxis, YAxis } from 'recharts';
import { Loader2, LayoutDashboard, Database } from 'lucide-react';

const API_URL = "http://127.0.0.1:8000";
const COLORS = ['#06B6D4', '#7C3AED', '#1E3A8A', '#3B82F6', '#8B5CF6', '#10B981'];

const Dashboard = () => {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const res = await axios.get(`${API_URL}/entities`);
                setHistory(res.data);
            } catch (err) {
                console.error("Failed to fetch entity history", err);
            } finally {
                setLoading(false);
            }
        };

        fetchHistory();
    }, []);

    // Aggregate data for charts
    const getStats = () => {
        const typeCount = {};
        let totalEntities = 0;
        
        history.forEach(doc => {
            doc.entities?.forEach(ent => {
                const t = ent.type || 'UNKNOWN';
                typeCount[t] = (typeCount[t] || 0) + 1;
                totalEntities++;
            });
        });

        const pieData = Object.keys(typeCount).map(key => ({
            name: key,
            value: typeCount[key]
        }));

        return { pieData, totalEntities, totalDocs: history.length };
    };

    const stats = getStats();

    if (loading) {
        return (
            <div className="flex h-[80vh] items-center justify-center">
                <Loader2 className="w-12 h-12 animate-spin text-secondary" />
            </div>
        );
    }

    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
            <h1 className="text-4xl font-bold mb-8 flex items-center">
                <LayoutDashboard className="w-10 h-10 mr-4 text-accent" />
                Intelligence Dashboard
            </h1>

            {history.length === 0 ? (
                <div className="glass p-12 rounded-2xl text-center flex flex-col items-center">
                    <Database className="w-16 h-16 text-gray-500 mb-4" />
                    <h2 className="text-2xl font-semibold mb-2">No Data Available</h2>
                    <p className="text-gray-400 mb-6">Analyze some documents first to populate the dashboard.</p>
                </div>
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Metrics Cards */}
                    <div className="lg:col-span-1 space-y-8">
                        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass p-6 rounded-2xl shadow-xl shadow-secondary/5 border-t-4 border-t-secondary">
                            <h3 className="text-gray-400 font-medium">Documents Analyzed</h3>
                            <p className="text-5xl font-bold mt-2 text-white">{stats.totalDocs}</p>
                        </motion.div>
                        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="glass p-6 rounded-2xl shadow-xl shadow-accent/5 border-t-4 border-t-accent">
                            <h3 className="text-gray-400 font-medium">Total Entities Extracted</h3>
                            <p className="text-5xl font-bold mt-2 text-white">{stats.totalEntities}</p>
                        </motion.div>
                        
                        <div className="glass p-6 rounded-2xl shadow-xl">
                            <h3 className="font-bold text-lg mb-4">Recent Entities</h3>
                            <div className="space-y-3">
                                {history.flatMap(h => h.entities).slice(-5).reverse().map((ent, idx) => (
                                    <div key={idx} className="flex justify-between items-center bg-white/5 p-3 rounded-lg text-sm">
                                        <span className="truncate max-w-[150px] font-medium">{ent.text}</span>
                                        <span className="text-xs text-gray-400 px-2 py-1 bg-white/10 rounded">{ent.type}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Charts */}
                    <div className="lg:col-span-2 space-y-8">
                        <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="glass p-6 rounded-2xl shadow-xl h-[400px]">
                            <h3 className="font-bold text-xl mb-6">Entity Distribution</h3>
                            <ResponsiveContainer width="100%" height="90%">
                                <PieChart>
                                    <Pie
                                        data={stats.pieData}
                                        cx="50%"
                                        cy="50%"
                                        innerRadius={80}
                                        outerRadius={120}
                                        paddingAngle={5}
                                        dataKey="value"
                                    >
                                        {stats.pieData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                        ))}
                                    </Pie>
                                    <Tooltip 
                                        contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', borderRadius: '8px' }}
                                        itemStyle={{ color: '#fff' }}
                                    />
                                    <Legend />
                                </PieChart>
                            </ResponsiveContainer>
                        </motion.div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Dashboard;
