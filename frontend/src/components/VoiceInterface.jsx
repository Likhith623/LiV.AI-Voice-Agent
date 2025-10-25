// VoiceInterface.jsx - Ultra-Optimized Voice Call Component
// Adapted for LiV.AI Voice Agent project structure

import React, { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FiX, FiMic } from "react-icons/fi";
import { API_CONFIG } from '../config/api';

// Audio configuration constants
const AUDIO_CONTEXT_CONFIG = {
  latencyHint: 'interactive',
  sampleRate: 8000,
  echoCancellation: false,
  noiseSuppression: false,
  autoGainControl: false,
  channelCount: 1
};

const SILENCE_DURATION = 200;
const CHECK_INTERVAL = 4;

// Voice Interface Component
const VoiceInterface = ({ personality, onClose }) => {
  console.log('🎯 VoiceInterface component rendered with personality:', personality);
  
  // States
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [error, setError] = useState(null);
  const [isCallActive, setIsCallActive] = useState(false);

  // Refs
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const audioChunksRef = useRef([]);
  const currentAudioRef = useRef(null);
  const silenceDetectionIntervalRef = useRef(null);
  const requestInProgress = useRef(false);
  const callEndedRef = useRef(false);
  
  const voiceActivityRef = useRef({
    isDetected: false,
    silenceTimer: null,
    isRecording: false
  });

  const userEmail = 'user@example.com';

  // Audio playback function
  const playAudioResponse = useCallback(async (audioBase64) => {
    if (!audioBase64 || callEndedRef.current) return;

    try {
      console.log('🎵 Playing audio response...');
      setIsSpeaking(true);
      
      const base64Data = audioBase64.includes(',') ? audioBase64.split(',')[1] : audioBase64;
      const audio = new Audio();
      audio.volume = 1.0;
      currentAudioRef.current = audio;
      
      await new Promise((resolve) => {
        let resolved = false;
        
        const resolveOnce = () => {
          if (!resolved) {
            resolved = true;
            resolve();
          }
        };
        
        audio.addEventListener('canplaythrough', () => {
          if (!callEndedRef.current) {
            audio.play().catch(console.error);
          }
        }, { once: true });
        
        audio.addEventListener('ended', resolveOnce, { once: true });
        audio.addEventListener('error', resolveOnce, { once: true });
        
        setTimeout(resolveOnce, 10000);
        
        audio.src = `data:audio/wav;base64,${base64Data}`;
      });
      
    } catch (error) {
      console.error('❌ Audio playback failed:', error);
    } finally {
      setIsSpeaking(false);
      setAudioLevel(0);
      currentAudioRef.current = null;
    }
  }, []);

  // Backend processing function
  const processWithBackend = useCallback(async (audioBlob) => {
    if (requestInProgress.current || callEndedRef.current) return;
    
    console.log('🚀 Processing audio with backend...');
    requestInProgress.current = true;
    setIsProcessing(true);
    
    try {
      const formData = new FormData();
      formData.append('audio_file', audioBlob, `voice_${Date.now()}.webm`);
      formData.append('bot_id', personality.id);
      formData.append('email', userEmail);
      formData.append('platform', 'web_voice_ultra');
      
      console.log('📤 Sending to backend:', {
        bot_id: personality.id,
        email: userEmail,
        platform: 'web_voice_ultra',
        audioSize: audioBlob.size
      });
      
      const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.VOICE_CALL}`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ Backend error response:', errorText);
        throw new Error(`Backend error: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('✅ Backend response:', data);
      
      // Play audio response
      if (!callEndedRef.current && data.audio_base64) {
        await playAudioResponse(data.audio_base64);
      }
      
    } catch (error) {
      console.error('❌ Processing failed:', error);
      setError(`Error: ${error.message}`);
      setTimeout(() => setError(null), 3000);
    } finally {
      requestInProgress.current = false;
      setIsProcessing(false);
    }
  }, [personality.id, userEmail, playAudioResponse]);

  // Microphone setup
  const setupMicrophone = useCallback(async () => {
    try {
      console.log('🎤 Setting up microphone...');
      
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
          sampleRate: 48000,
        }
      });

      streamRef.current = stream;

      const options = { 
        mimeType: 'audio/webm;codecs=opus',
        audioBitsPerSecond: 32000
      };
      
      if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options.mimeType = 'audio/webm';
      }

      mediaRecorderRef.current = new MediaRecorder(stream, options);
      audioChunksRef.current = [];

      // Setup audio analysis
      if (!audioContextRef.current) {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        audioContextRef.current = new AudioContext();
      }
      
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }

      const source = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 512;
      analyserRef.current.smoothingTimeConstant = 0.3;
      source.connect(analyserRef.current);

      // MediaRecorder event handlers
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        if (audioChunksRef.current.length > 0) {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          audioChunksRef.current = [];
          
          if (audioBlob.size > 1000) {
            await processWithBackend(audioBlob);
          }
        }
      };
      
      return true;
    } catch (error) {
      console.error('❌ Microphone setup failed:', error);
      setError('Microphone access denied');
      setTimeout(() => setError(null), 3000);
      return false;
    }
  }, [processWithBackend]);

  // Voice activity detection
  const startVoiceActivityDetection = useCallback(() => {
    const frequencyBuffer = new Uint8Array(128);

    const checkVoiceActivity = () => {
      if (!isCallActive || callEndedRef.current) return;
      
      if (analyserRef.current) {
        try {
          analyserRef.current.getByteFrequencyData(frequencyBuffer);
          
          let sum = 0;
          let max = 0;
          let activeFrequencies = 0;
          
          for (let i = 0; i < frequencyBuffer.length; i++) {
            const value = frequencyBuffer[i];
            sum += value;
            max = Math.max(max, value);
            if (value > 2) activeFrequencies++;
          }
          
          const average = sum / frequencyBuffer.length;
          const rawLevel = Math.max(average / 3, max / 8, activeFrequencies / 10);
          const amplifiedLevel = Math.min(Math.pow(rawLevel, 0.2) * 5, 1);
          
          setAudioLevel(amplifiedLevel);
          
          const voiceDetected = average > 5 || max > 15 || activeFrequencies > 8;
          
          if (voiceDetected && !isSpeaking && !isProcessing) {
            if (!voiceActivityRef.current.isDetected) {
              voiceActivityRef.current.isDetected = true;
              
              if (!voiceActivityRef.current.isRecording && 
                  mediaRecorderRef.current && 
                  mediaRecorderRef.current.state === 'inactive') {
                mediaRecorderRef.current.start();
                voiceActivityRef.current.isRecording = true;
                setIsListening(true);
              }
            }
            
            if (voiceActivityRef.current.silenceTimer) {
              clearTimeout(voiceActivityRef.current.silenceTimer);
              voiceActivityRef.current.silenceTimer = null;
            }
          } else if (voiceActivityRef.current.isDetected && !voiceDetected && !isSpeaking) {
            if (!voiceActivityRef.current.silenceTimer) {
              voiceActivityRef.current.silenceTimer = setTimeout(() => {
                voiceActivityRef.current.isDetected = false;
                
                if (voiceActivityRef.current.isRecording && 
                    mediaRecorderRef.current && 
                    mediaRecorderRef.current.state === 'recording') {
                  mediaRecorderRef.current.stop();
                  voiceActivityRef.current.isRecording = false;
                  setIsListening(false);
                }
              }, SILENCE_DURATION);
            }
          }
        } catch (error) {
          console.error('❌ Audio analysis error:', error);
        }
      }
    };
    
    const interval = setInterval(checkVoiceActivity, CHECK_INTERVAL);
    silenceDetectionIntervalRef.current = interval;
  }, [isCallActive, isSpeaking, isProcessing]);

  // Start call
  const startCall = useCallback(async () => {
    callEndedRef.current = false;
    
    const micSetup = await setupMicrophone();
    if (!micSetup) return;
    
    setIsCallActive(true);
    startVoiceActivityDetection();
  }, [setupMicrophone, startVoiceActivityDetection]);

  // End call
  const endCall = useCallback(() => {
    console.log('🔴 Ending call...');
    
    callEndedRef.current = true;
    
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current.currentTime = 0;
      currentAudioRef.current = null;
    }
    
    setIsCallActive(false);
    setIsListening(false);
    setIsSpeaking(false);
    setIsProcessing(false);
    setAudioLevel(0);
    
    if (silenceDetectionIntervalRef.current) {
      clearInterval(silenceDetectionIntervalRef.current);
      silenceDetectionIntervalRef.current = null;
    }
    
    if (voiceActivityRef.current.silenceTimer) {
      clearTimeout(voiceActivityRef.current.silenceTimer);
    }
    
    voiceActivityRef.current = {
      isDetected: false,
      silenceTimer: null,
      isRecording: false
    };
    
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    requestInProgress.current = false;
    
    onClose();
  }, [onClose]);

  // Manual toggle recording
  const toggleRecording = useCallback(() => {
    if (isSpeaking || isProcessing) return;
    
    if (!voiceActivityRef.current.isRecording && 
        mediaRecorderRef.current && 
        mediaRecorderRef.current.state === 'inactive') {
      mediaRecorderRef.current.start();
      voiceActivityRef.current.isRecording = true;
      voiceActivityRef.current.isDetected = true;
      setIsListening(true);
    } else if (voiceActivityRef.current.isRecording && 
               mediaRecorderRef.current && 
               mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      voiceActivityRef.current.isRecording = false;
      voiceActivityRef.current.isDetected = false;
      setIsListening(false);
    }
  }, [isSpeaking, isProcessing]);

  // Effects
  useEffect(() => {
    console.log('🎬 VoiceInterface mounted, starting call...');
    startCall();
    
    return () => {
      console.log('🛑 VoiceInterface unmounting, ending call...');
      // Call endCall directly to cleanup
      callEndedRef.current = true;
      
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
        currentAudioRef.current.currentTime = 0;
        currentAudioRef.current = null;
      }
      
      if (silenceDetectionIntervalRef.current) {
        clearInterval(silenceDetectionIntervalRef.current);
        silenceDetectionIntervalRef.current = null;
      }
      
      if (voiceActivityRef.current.silenceTimer) {
        clearTimeout(voiceActivityRef.current.silenceTimer);
      }
      
      if (mediaRecorderRef.current?.state === 'recording') {
        mediaRecorderRef.current.stop();
      }
      
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
      
      requestInProgress.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Render
  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex flex-col bg-[#f8fafc]"
      >
        {/* Header - Centered */}
        <motion.div
          className="absolute top-6 w-full z-10 flex justify-center"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="text-center">
            <h1 className="text-gray-700 text-2xl font-bold tracking-wide">
              {personality.name}
            </h1>
            <div className="flex items-center justify-center mt-1 space-x-2">
              <div className="w-2 h-2 rounded-full bg-green-400" />
              <span className="text-xs text-gray-500">Connected</span>
            </div>
          </div>
        </motion.div>

        {/* Processing Indicator */}
        <AnimatePresence>
          {isProcessing && (
            <motion.div
              className="absolute top-24 w-full z-10 flex justify-center"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
            >
              <div className="flex items-center space-x-3 px-5 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-full shadow-xl">
                <div className="flex space-x-1">
                  {[...Array(3)].map((_, i) => (
                    <motion.div
                      key={i}
                      className="w-2 h-2 bg-white rounded-full"
                      animate={{ scale: [1, 1.4, 1], opacity: [0.5, 1, 0.5] }}
                      transition={{ 
                        duration: 0.6, 
                        repeat: Infinity, 
                        delay: i * 0.1 
                      }}
                    />
                  ))}
                </div>
                <span className="text-sm font-medium">Processing...</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Main Avatar - Centered */}
        <div className="flex-1 flex items-center justify-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.7 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 0.8, type: "spring" }}
            className="relative flex items-center justify-center"
          >
            {/* Avatar Circle */}
            <motion.div
              onClick={toggleRecording}
              className="relative w-48 h-48 rounded-full overflow-hidden border-4 cursor-pointer"
              style={{
                borderColor: isSpeaking 
                  ? '#22c55e' 
                  : isProcessing 
                    ? '#f59e0b' 
                    : isListening
                      ? '#3b82f6'
                      : '#e5e7eb',
                boxShadow: isSpeaking 
                  ? '0 0 40px rgba(34, 197, 94, 0.7)'
                  : isProcessing 
                    ? '0 0 40px rgba(245, 158, 11, 0.7)'
                    : isListening
                      ? '0 0 30px rgba(59, 130, 246, 0.6)'
                      : '0 0 20px rgba(0, 0, 0, 0.15)'
              }}
              animate={{
                scale: isSpeaking
                  ? [1, 1.08, 1]
                  : isListening
                    ? [1, 1 + audioLevel * 0.4, 1]
                    : [1, 1.01, 1]
              }}
              transition={{
                duration: 0.4,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            >
              <img 
                src={personality.image} 
                alt={personality.name} 
                className="w-full h-full object-cover"
              />
              
              {/* Mic Icon Overlay when listening */}
              {isListening && !isSpeaking && !isProcessing && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/20">
                  <motion.div
                    animate={{
                      scale: [0.9, 1.3, 0.9],
                      opacity: [0.8, 1, 0.8]
                    }}
                    transition={{
                      duration: 1.2,
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                  >
                    <FiMic className="w-12 h-12 text-white drop-shadow-2xl" />
                  </motion.div>
                </div>
              )}
            </motion.div>
          </motion.div>
        </div>

        {/* Bottom Controls */}
        <div className="absolute bottom-12 left-1/2 transform -translate-x-1/2">
          <motion.div
            className="flex items-center space-x-12"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            {/* Manual Record Button */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={toggleRecording}
              className={`w-16 h-16 rounded-full ${
                isListening ? 'bg-red-600 hover:bg-red-700' : 'bg-purple-600 hover:bg-purple-700'
              } text-white flex items-center justify-center transition-all duration-200 shadow-lg`}
              title={isListening ? "Stop Recording" : "Start Recording"}
            >
              <FiMic className="w-8 h-8" />
            </motion.button>

            {/* End Call Button */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={endCall}
              className="w-16 h-16 rounded-full bg-red-500 hover:bg-red-600 text-white flex items-center justify-center transition-all duration-200 shadow-lg"
            >
              <FiX className="w-7 h-7" />
            </motion.button>
          </motion.div>
        </div>

        {/* Status Indicator */}
        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2">
          <motion.div
            className="flex items-center space-x-3"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.7 }}
          >
            <div className="w-40 h-2 bg-gray-200 rounded-full overflow-hidden">
              <motion.div
                className={`h-full rounded-full transition-colors duration-200 ${
                  isSpeaking ? 'bg-green-500' :
                  isProcessing ? 'bg-yellow-500' :
                  isListening ? 'bg-blue-500' : 'bg-gray-400'
                }`}
                style={{
                  width: `${Math.min(audioLevel * 100, 100)}%`
                }}
                transition={{ duration: 0.1 }}
              />
            </div>
          </motion.div>
        </div>

        {/* Error Message */}
        <AnimatePresence>
          {error && (
            <motion.div
              className="absolute bottom-32 left-1/2 transform -translate-x-1/2 px-6 py-3 bg-red-500 text-white rounded-xl shadow-xl max-w-sm"
              initial={{ opacity: 0, y: 10, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.9 }}
            >
              <p className="text-sm text-center font-medium">{error}</p>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </AnimatePresence>
  );
};

export default VoiceInterface;
