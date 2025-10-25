import React from 'react';
import { motion } from 'framer-motion';
import { FiUser, FiSettings } from 'react-icons/fi';

const Navbar = () => {
  return (
    <motion.nav
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="fixed top-0 left-0 right-0 z-50 px-6 py-4"
    >
      <div className="max-w-7xl mx-auto glass rounded-2xl px-8 py-4 flex items-center justify-between">
        {/* Logo */}
        <motion.div
          className="flex items-center space-x-3"
          whileHover={{ scale: 1.05 }}
          transition={{ type: "spring", stiffness: 300 }}
        >
          <div className="relative">
            <div className="w-10 h-10 rounded-xl animated-gradient flex items-center justify-center">
              <svg
                className="w-6 h-6 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                />
              </svg>
            </div>
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-accent-teal rounded-full animate-pulse"></div>
          </div>
          <div>
            <h1 className="text-xl font-bold gradient-text">LiV.AI</h1>
            <p className="text-xs text-gray-400">Voice Agent</p>
          </div>
        </motion.div>

        {/* Right side - User Profile */}
        <div className="flex items-center space-x-4">
          <motion.button
            whileHover={{ scale: 1.1, rotate: 90 }}
            whileTap={{ scale: 0.95 }}
            transition={{ type: "spring", stiffness: 300 }}
            className="w-10 h-10 rounded-full glass flex items-center justify-center hover:bg-white/10 transition-all"
          >
            <FiSettings className="text-gray-300 text-lg" />
          </motion.button>
          
          <motion.div
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
            transition={{ type: "spring", stiffness: 300 }}
            className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center cursor-pointer ring-2 ring-purple-500/50 hover:ring-purple-400 transition-all"
          >
            <FiUser className="text-white text-lg" />
          </motion.div>
        </div>
      </div>
    </motion.nav>
  );
};

export default Navbar;
