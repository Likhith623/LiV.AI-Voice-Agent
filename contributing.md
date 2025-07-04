# 🤝 Contributing to LiV.AI

Thank you for your interest in contributing to LiV.AI! Here's how you can help make the world's most advanced voice AI backend even better.

## 🚀 Quick Start

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/yourusername/LiV.AI.git
   cd LiV.AI
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment**
   ```bash
   cp .env.example .env
   # Add your API keys
   ```
5. **Start development server**
   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

## 🎯 Areas for Contribution

### High Priority
- 🎤 **New STT Providers** - Add Azure Speech, AWS Transcribe
- 🔊 **TTS Optimizations** - Streaming improvements, new voices
- 🌍 **Language Support** - Additional language models
- ⚡ **Performance** - Caching, parallel processing
- 📊 **Monitoring** - Advanced analytics, alerting

### Medium Priority  
- 🎭 **New Personalities** - Cultural voices, specialized roles
- 🔧 **API Features** - Webhooks, real-time subscriptions
- 🛡️ **Security** - Rate limiting, authentication
- 📱 **Client SDKs** - JavaScript, Python, Go libraries

## 📋 Development Guidelines

### Code Style
- **Type hints** required for all functions
- **Docstrings** for all public functions
- **Error handling** with proper logging
- **Performance** considerations documented

### Testing Requirements
- Unit tests for new functions
- Integration tests for API endpoints
- Performance benchmarks for optimizations
- Audio quality validation for TTS changes

### Pull Request Process
1. Create feature branch from `main`
2. Make changes with tests
3. Update documentation
4. Submit PR with detailed description
5. Address review feedback

## 🧪 Testing Your Changes

### Voice Call Pipeline Testing
```bash
# Test ultra-fast voice call endpoint
curl -X POST http://127.0.0.1:8000/voice-call-ultra-fast \
  -F "audio_file=@test_audio.wav" \
  -F "bot_id=indian_old_male" \
  -F "email=test@example.com" \
  -F "platform=testing"

# Test standard voice call with memory
curl -X POST http://127.0.0.1:8000/voice-call \
  -F "audio_file=@test_audio.wav" \
  -F "bot_id=japanese_mid_female" \
  -F "email=test@example.com"
```

### STT Performance Testing
```bash
# Run comprehensive STT performance test
curl -X POST http://127.0.0.1:8000/test-stt-performance \
  -F "audio_file=@test_audio.wav" \
  -F "iterations=10"

# Check STT performance stats
curl http://127.0.0.1:8000/stt-performance/stats
```

### TTS Testing
```bash
# Test audio generation
curl -X POST http://127.0.0.1:8000/generate-audio-optimized \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Testing TTS generation",
    "bot_id": "parisian_old_male",
    "output_format": {
      "container": "wav",
      "encoding": "pcm_s16le",
      "sample_rate": 8000
    }
  }'

# Check TTS cache performance
curl http://127.0.0.1:8000/tts-cache/stats
```

### Chat Endpoint Testing
```bash
# Test enhanced chat endpoint
curl -X POST http://127.0.0.1:8000/cv/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, how are you today?",
    "bot_id": "berlin_rom_female",
    "user_name": "TestUser",
    "email": "test@example.com",
    "language": "English"
  }'

# Test advanced chat with memory
curl -X POST http://127.0.0.1:8000/v2/cv/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about machine learning",
    "bot_id": "indian_mid_male",
    "user_name": "Developer",
    "email": "dev@example.com"
  }'
```

### System Health Testing
```bash
# Check Redis cache health
curl http://127.0.0.1:8000/redis/health

# Test cache functionality
curl http://127.0.0.1:8000/test-cache-now

# Clear caches for testing (admin)
curl -X DELETE http://127.0.0.1:8000/tts-cache/clear
curl -X DELETE http://127.0.0.1:8000/redis/cache
```

### Performance Monitoring
```bash
# Monitor system performance
curl http://127.0.0.1:8000/stt-performance/stats | jq '.'
curl http://127.0.0.1:8000/tts-cache/stats | jq '.'

# Test with different personalities
for bot in "indian_old_male" "japanese_rom_female" "parisian_mid_male" "berlin_old_female"; do
  echo "Testing with $bot"
  curl -X POST http://127.0.0.1:8000/generate-audio-optimized \
    -H "Content-Type: application/json" \
    -d "{\"transcript\": \"Testing voice $bot\", \"bot_id\": \"$bot\"}"
done
```

## 🔧 Development Environment Setup

### Local Development Server
```bash
# Start with auto-reload for development
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Access the automatic API docs at:
# http://127.0.0.1:8000/docs
# http://127.0.0.1:8000/redoc
```

### Environment Variables Testing
```bash
# Verify all required environment variables are set
python -c "
import os
from dotenv import load_dotenv
load_dotenv()

required_vars = [
    'CARTESIA_API_KEY', 'DEEPGRAM_API_KEY', 'ASSEMBLYAI_API_KEY',
    'GEMINI_API_KEY', 'OPENAI_API_KEY', 'XAI_API_KEY',
    'REDIS_HOST', 'REDIS_PASSWORD'
]

for var in required_vars:
    if os.getenv(var):
        print(f'✅ {var} is set')
    else:
        print(f'❌ {var} is missing')
"
```

## 📊 Performance Benchmarks

### Target Performance Metrics
```bash
# Voice call pipeline targets
echo "Testing voice call performance targets:"
echo "- Ultra-fast endpoint: 2.5-3.5s target"
echo "- STT processing: 1.5-2.5s target"
echo "- LLM response: 2.0-3.0s target"
echo "- TTS generation: 0.5-2.0s target"
echo "- Cache hit rate: >80% target"

# Run performance validation
time curl -X POST http://127.0.0.1:8000/voice-call-ultra-fast \
  -F "audio_file=@test_audio.wav" \
  -F "bot_id=indian_old_male" \
  -F "email=perf-test@example.com"
```

## 🧪 Advanced Testing Scenarios

### Multi-Bot Testing
```bash
# Test all personality types
personalities=(
  "indian_old_male" "indian_old_female" "indian_mid_male" "indian_mid_female"
  "japanese_old_male" "japanese_old_female" "japanese_mid_male" "japanese_mid_female"
  "parisian_old_male" "parisian_old_female" "parisian_mid_male" "parisian_mid_female"
  "berlin_old_male" "berlin_old_female" "berlin_mid_male" "berlin_mid_female"
  "Krishn" "Ram" "Hanuma" "Shiv" "Trimurthi"
)

for bot in "${personalities[@]}"; do
  echo "Testing personality: $bot"
  curl -s -X POST http://127.0.0.1:8000/cv/chat \
    -H "Content-Type: application/json" \
    -d "{\"message\": \"Hello!\", \"bot_id\": \"$bot\", \"email\": \"test@example.com\"}" \
    | jq -r '.response' | head -c 50
  echo "..."
done
```

### Error Handling Testing
```bash
# Test invalid audio file
echo "Testing error handling..."
curl -X POST http://127.0.0.1:8000/voice-call-ultra-fast \
  -F "audio_file=@invalid_file.txt" \
  -F "bot_id=indian_old_male"

# Test invalid bot ID
curl -X POST http://127.0.0.1:8000/cv/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "bot_id": "invalid_bot", "email": "test@example.com"}'
```

## 📝 Documentation Standards

### API Documentation Updates
- All new endpoints must include OpenAPI/Swagger documentation
- Examples required for request/response formats
- Error codes and handling documented
- Performance characteristics noted

### Code Documentation
```python
# Example of required documentation style
async def new_voice_feature(audio_data: bytes, bot_id: str) -> dict:
    """
    Process voice data with new optimized pipeline.
    
    Args:
        audio_data: Raw audio bytes in WAV format
        bot_id: Personality identifier from supported bot list
        
    Returns:
        dict: Processing result with audio and performance metrics
        
    Raises:
        HTTPException: If audio processing fails
        
    Performance:
        Target: <2.0s for standard audio (5-10 seconds)
        Optimizations: Direct buffer processing, smart caching
    """
```

## 🏆 Recognition

Contributors will be:
- ✅ Added to project README
- ✅ Credited in release notes  
- ✅ Invited to contributor discussions
- ✅ Given priority for collaboration opportunities
- ✅ Featured in project showcase

## 📞 Development Support

### Quick Help
- 🐛 **Bugs**: Open GitHub issue with reproduction steps
- 💡 **Features**: Start GitHub discussion for design feedback
- ❓ **Questions**: Comment on existing issues or discussions
- 🔧 **Architecture**: Contact [Likhith Vasireddy](https://github.com/Likhith623) for major changes

### Development Chat
Join our development discussions:
- **GitHub Discussions**: Architecture and feature planning
- **Issue Comments**: Bug reports and feature requests
- **Pull Request Reviews**: Code feedback and improvements

### Testing Checklist Before PR
```bash
# ✅ Run full test suite
pytest tests/ -v

# ✅ Check code formatting
black . && isort . && flake8

# ✅ Validate type hints
mypy main.py --strict

# ✅ Test all voice endpoints
curl http://127.0.0.1:8000/voice-call-ultra-fast
curl http://127.0.0.1:8000/cv/chat
curl http://127.0.0.1:8000/generate-audio-optimized

# ✅ Verify performance targets
curl http://127.0.0.1:8000/stt-performance/stats
```

---

<div align="center">

**🎙️ Let's Build the Future of Voice AI Together!**

*Your contributions help developers worldwide create amazing voice-enabled applications*

[📚 API Docs](http://127.0.0.1:8000/docs) | [🧪 Test Suite](http://127.0.0.1:8000/redoc) | [💬 Discussions](https://github.com/Likhith623/LiV.AI/discussions)

</div>