import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import { PersonalityCard, PersonalityModal } from './components/PersonalityCard';
import VoiceInterface from './components/VoiceInterface';
import { personalities } from './data/personalities';

function App() {
  const [selectedPersonality, setSelectedPersonality] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [showVoiceInterface, setShowVoiceInterface] = useState(false);
  const [activePersonality, setActivePersonality] = useState(null);

  const handleCardClick = (personality) => {
    setSelectedPersonality(personality);
    setShowModal(true);
  };

  const handleCloseModal = () => {
    setShowModal(false);
    setTimeout(() => setSelectedPersonality(null), 300);
  };

  const handleSelectPersonality = (personality) => {
    console.log('✅ Selected personality:', personality);
    setActivePersonality(personality);
    setShowModal(false);
    setShowVoiceInterface(true);
    console.log('🎤 Opening voice interface...');
  };

  const handleCloseVoiceInterface = () => {
    setShowVoiceInterface(false);
    setTimeout(() => setActivePersonality(null), 300);
  };

  return (
    <div className="min-h-screen relative">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-72 h-72 bg-blue-500/10 rounded-full blur-3xl animate-float"></div>
        <div className="absolute top-40 right-20 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-float" style={{ animationDelay: '1s' }}></div>
        <div className="absolute bottom-20 left-1/3 w-80 h-80 bg-teal-500/10 rounded-full blur-3xl animate-float" style={{ animationDelay: '2s' }}></div>
      </div>

      <Navbar />

      {/* Main Content */}
      <main className="relative z-10 pt-32 pb-16 px-6">
        <div className="max-w-7xl mx-auto">
          {/* Hero Section */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <motion.h1
              className="text-5xl md:text-7xl font-bold mb-6 gradient-text"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.8 }}
            >
              Choose Your AI Personality
            </motion.h1>
            <motion.p
              className="text-gray-400 text-lg md:text-xl max-w-2xl mx-auto leading-relaxed"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.8 }}
            >
              Select from our premium collection of AI personalities, each crafted
              with unique characteristics and conversational styles.
            </motion.p>
          </motion.div>

          {/* Personality Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-12">
            {personalities.map((personality, index) => (
              <PersonalityCard
                key={personality.id}
                personality={personality}
                onClick={handleCardClick}
                index={index}
              />
            ))}
          </div>

          {/* Stats Section */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6, duration: 0.8 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto"
          >
            {[
              { number: '32', label: 'AI Personalities' },
              { number: '100%', label: 'Premium Quality' },
              { number: '24/7', label: 'Always Available' },
            ].map((stat, index) => (
              <motion.div
                key={index}
                whileHover={{ scale: 1.05, y: -5 }}
                className="glass rounded-2xl p-6 text-center"
              >
                <h3 className="text-4xl font-bold gradient-text mb-2">
                  {stat.number}
                </h3>
                <p className="text-gray-400">{stat.label}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </main>

      <Footer />

      {/* Personality Modal */}
      {showModal && (
        <PersonalityModal
          personality={selectedPersonality}
          onClose={handleCloseModal}
          onSelect={handleSelectPersonality}
        />
      )}

      {/* Voice Interface */}
      {showVoiceInterface && activePersonality && (
        <VoiceInterface
          personality={activePersonality}
          onClose={handleCloseVoiceInterface}
        />
      )}
    </div>
  );
}

export default App;
