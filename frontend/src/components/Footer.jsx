import React from 'react';
import { motion } from 'framer-motion';
import { FiGithub, FiTwitter, FiLinkedin, FiMail } from 'react-icons/fi';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <motion.footer
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.5, duration: 0.5 }}
      className="mt-20 pb-8"
    >
      <div className="max-w-7xl mx-auto px-6">
        <div className="glass rounded-2xl px-8 py-6">
          <div className="flex flex-col md:flex-row items-center justify-between space-y-4 md:space-y-0">
            {/* Copyright */}
            <div className="text-center md:text-left">
              <p className="text-gray-400 text-sm">
                © {currentYear} <span className="gradient-text font-semibold">LiV.AI</span>. All rights reserved.
              </p>
              <p className="text-gray-500 text-xs mt-1">
                Crafted with precision and elegance
              </p>
            </div>

            {/* Social Links */}
            <div className="flex items-center space-x-4">
              {[
                { icon: FiGithub, href: '#', label: 'GitHub' },
                { icon: FiTwitter, href: '#', label: 'Twitter' },
                { icon: FiLinkedin, href: '#', label: 'LinkedIn' },
                { icon: FiMail, href: '#', label: 'Email' },
              ].map((social, index) => (
                <motion.a
                  key={index}
                  href={social.href}
                  whileHover={{ scale: 1.2, y: -2 }}
                  whileTap={{ scale: 0.9 }}
                  transition={{ type: "spring", stiffness: 300 }}
                  className="w-9 h-9 rounded-full glass flex items-center justify-center hover:bg-white/10 transition-all group"
                  aria-label={social.label}
                >
                  <social.icon className="text-gray-400 text-base group-hover:text-white transition-colors" />
                </motion.a>
              ))}
            </div>
          </div>
        </div>
      </div>
    </motion.footer>
  );
};

export default Footer;
