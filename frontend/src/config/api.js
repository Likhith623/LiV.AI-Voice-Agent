// Backend API Configuration
export const API_CONFIG = {
  // Change this to your backend URL
  BASE_URL: 'http://127.0.0.1:8000',
  
  // API Endpoints
  ENDPOINTS: {
    VOICE_CALL: '/voice-call-ultra-fast',
  },
  
  // Request timeout (ms)
  TIMEOUT: 30000,
};

// Helper function to get full endpoint URL
export const getEndpointUrl = (endpoint) => {
  return `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS[endpoint]}`;
};
