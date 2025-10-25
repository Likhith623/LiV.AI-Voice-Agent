import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiX, FiCheck } from 'react-icons/fi';

const PersonalityCard = ({ personality, onClick, index }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1, duration: 0.5 }}
      whileHover={{ scale: 1.05, y: -10 }}
      onClick={() => onClick(personality)}
      className="group cursor-pointer relative overflow-hidden rounded-3xl glass glow-effect-hover"
    >
      {/* Background Image with Overlay */}
      <div className="relative h-72 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-gray-900/95 z-10"></div>
        <img
          src={personality.image}
          alt={personality.name}
          className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
          onError={(e) => {
            e.target.src = '/photos/default_bg.png';
          }}
        />
        
        {/* Animated gradient overlay */}
        <div className={`absolute inset-0 bg-gradient-to-br ${personality.color} opacity-0 group-hover:opacity-20 transition-opacity duration-500`}></div>
      </div>

      {/* Content */}
      <div className="absolute bottom-0 left-0 right-0 p-6 z-20">
        <motion.h3
          className="text-2xl font-bold text-white mb-2 group-hover:gradient-text transition-all duration-300"
        >
          {personality.name}
        </motion.h3>
        <p className="text-gray-300 text-sm leading-relaxed">
          {personality.tagline}
        </p>

        {/* Hover indicator */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          whileHover={{ opacity: 1, x: 0 }}
          className="mt-4 flex items-center space-x-2 text-accent-teal opacity-0 group-hover:opacity-100 transition-all duration-300"
        >
          <span className="text-sm font-medium">Learn more</span>
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </motion.div>
      </div>

      {/* Corner accent */}
      <div className="absolute top-4 right-4 w-2 h-2 rounded-full bg-accent-teal animate-pulse z-20"></div>
    </motion.div>
  );
};

const PersonalityModal = ({ personality, onClose, onSelect }) => {
  if (!personality) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
        className="fixed inset-0 bg-black/70 backdrop-blur-md z-50 flex items-center justify-center p-4"
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0, y: 50 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.9, opacity: 0, y: 50 }}
          transition={{ type: "spring", damping: 25, stiffness: 300 }}
          onClick={(e) => e.stopPropagation()}
          className="relative max-w-4xl w-full glass rounded-3xl overflow-hidden max-h-[90vh] overflow-y-auto"
        >
          {/* Close Button */}
          <motion.button
            whileHover={{ scale: 1.1, rotate: 90 }}
            whileTap={{ scale: 0.9 }}
            onClick={onClose}
            className="absolute top-6 right-6 z-30 w-10 h-10 rounded-full glass flex items-center justify-center hover:bg-white/20 transition-all"
          >
            <FiX className="text-white text-xl" />
          </motion.button>

          <div className="grid md:grid-cols-2 gap-0">
            {/* Image Side */}
            <div className="relative h-96 md:h-auto">
              <div className={`absolute inset-0 bg-gradient-to-br ${personality.color} opacity-30`}></div>
              <img
                src={personality.image}
                alt={personality.name}
                className="w-full h-full object-cover"
                onError={(e) => {
                  e.target.src = '/photos/default_bg.png';
                }}
              />
            </div>

            {/* Content Side */}
            <div className="p-8 md:p-12">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
              >
                <h2 className="text-4xl font-bold gradient-text mb-3">
                  {personality.name}
                </h2>
                <p className="text-gray-400 text-lg mb-6">
                  {personality.tagline}
                </p>

                <div className="mb-8">
                  <h3 className="text-white font-semibold mb-3 text-sm uppercase tracking-wider">
                    Description
                  </h3>
                  <p className="text-gray-300 leading-relaxed">
                    {personality.description}
                  </p>
                </div>

                <div className="mb-8">
                  <h3 className="text-white font-semibold mb-4 text-sm uppercase tracking-wider">
                    Key Traits
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {personality.traits.map((trait, index) => (
                      <motion.span
                        key={index}
                        initial={{ opacity: 0, scale: 0 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.3 + index * 0.1 }}
                        className="px-4 py-2 rounded-full glass text-sm text-gray-200 border border-white/20 hover:border-accent-teal/50 transition-all"
                      >
                        {trait}
                      </motion.span>
                    ))}
                  </div>
                </div>

                {/* Select Button */}
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => onSelect(personality)}
                  className={`w-full py-4 rounded-2xl bg-gradient-to-r ${personality.color} text-white font-semibold text-lg flex items-center justify-center space-x-2 hover:shadow-2xl transition-all duration-300 glow-effect`}
                >
                  <FiCheck className="text-xl" />
                  <span>Select {personality.name}</span>
                </motion.button>
              </motion.div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export { PersonalityCard, PersonalityModal };
