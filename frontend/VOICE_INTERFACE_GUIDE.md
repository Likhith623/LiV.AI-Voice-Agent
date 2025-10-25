# 🎙️ Voice Interface Integration Guide

## ✨ What's New

Your AI Personality Selector now includes a **premium ChatGPT-style voice interface** that allows users to have real-time voice conversations with any of the 32 AI personalities!

## 🎨 Features

### Visual Design
- **Central Animated Circle** - ChatGPT-inspired voice orb with smooth animations
- **Pulsing Rings** - Three animated rings that expand when listening
- **Audio Visualization** - Circle scales based on microphone input levels
- **Gradient Effects** - Dynamic color transitions (blue → purple → teal)
- **Status Indicators** - Clear visual feedback for listening/processing states

### User Experience
1. **Click to Speak** - Tap the circle to start recording
2. **Real-time Feedback** - Visual audio level detection while speaking
3. **Auto Processing** - Stops recording and sends to backend
4. **Voice Response** - Receives and plays AI-generated audio
5. **Conversation Display** - Shows transcript and response text

### Backend Integration
- ✅ Connects to `/voice-call` endpoint
- ✅ Sends audio with personality ID
- ✅ Receives transcript + AI response
- ✅ Plays audio response automatically
- ✅ Error handling and retry logic

## 🔧 Configuration

### Backend URL
Edit `src/config/api.js` to change backend URL:

```javascript
export const API_CONFIG = {
  BASE_URL: 'http://localhost:8000', // Change this
  ENDPOINTS: {
    VOICE_CALL: '/voice-call',
  },
};
```

### User Email
In `VoiceInterface.jsx`, update the default email:

```javascript
const userEmail = 'user@example.com'; // Replace with actual user
```

## 🎯 How It Works

### 1. User Flow
```
Select Personality → Click Card → View Details → 
Click "Select" Button → Voice Interface Opens → 
Click Circle → Speak → AI Responds with Voice
```

### 2. Technical Flow
```
User speaks → MediaRecorder captures audio → 
Send to /voice-call → Backend processes → 
Returns {transcript, response, audio_base64} → 
Display text + Play audio response
```

### 3. Backend Expected Response
```json
{
  "transcript": "User's spoken text",
  "response": "AI's text response",
  "audio_base64": "base64 encoded audio"
}
```

## 🎨 Design Features

### Animations
- **Idle State**: Gray gradient with mic-off icon
- **Listening State**: Blue-purple gradient with pulsing rings
- **Processing State**: Purple-pink gradient with spinning loader
- **Audio Reactive**: Scales with voice volume (0-30% larger)

### Color Scheme
- **Listening**: Blue (#3b82f6) to Purple (#8b5cf6)
- **Processing**: Purple (#8b5cf6) to Pink (#ec4899)
- **Idle**: Slate (#64748b) to Gray (#374151)

## 📱 Responsive Design

Works perfectly on:
- ✅ Desktop (optimal experience)
- ✅ Tablet (touch-friendly)
- ✅ Mobile (full-screen mode)

## 🔊 Audio Handling

### Recording
- Format: WebM audio
- Sample Rate: Browser default
- Channels: Mono (converted)

### Playback
- Format: WAV (base64 encoded)
- Auto-play on response
- Error handling for unsupported formats

## 🛠️ Required Permissions

The app will request:
- 🎤 **Microphone Access** - Required for voice recording

Users will see a browser permission prompt on first use.

## 🚀 Usage Example

```javascript
// In your backend (main.py), ensure this endpoint exists:
@app.post("/voice-call")
async def voice_call(
    audio: UploadFile = File(...),
    email: str = Form(...),
    bot_id: str = Form(...)
):
    # Process audio
    transcript = await speech_to_text(audio)
    response = await generate_response(transcript, bot_id)
    audio_base64 = await text_to_speech(response, bot_id)
    
    return {
        "transcript": transcript,
        "response": response,
        "audio_base64": audio_base64
    }
```

## 🎭 Personality Integration

Each personality uses its own:
- **Voice ID** - From `personality.voiceId`
- **Bot ID** - From `personality.id`
- **Visual Theme** - From `personality.color`

Voice mapping matches the backend `VOICE_MAPPING` dictionary.

## 🐛 Troubleshooting

### Issue: "Could not access microphone"
**Solution**: Check browser permissions and ensure HTTPS (for production)

### Issue: "Failed to process audio"
**Solution**: Verify backend is running on `http://localhost:8000`

### Issue: No audio playback
**Solution**: Check browser console for audio format errors

### Issue: Long processing time
**Solution**: Backend may be slow - check API response times

## 🔐 Security Notes

- 🔒 Microphone access requires user permission
- 🔒 Audio is sent via FormData (not stored)
- 🔒 Use HTTPS in production for mic access
- 🔒 Implement user authentication for `userEmail`

## 📊 Performance

- **Recording**: ~50KB per second
- **Processing**: 2-5 seconds (depends on backend)
- **Playback**: Instant after loading
- **Memory**: Cleaned up on unmount

## 🎉 Next Steps

1. ✅ **Test Backend** - Ensure `/voice-call` endpoint works
2. ✅ **Add Authentication** - Replace hardcoded email
3. ✅ **Add Conversation History** - Store previous messages
4. ✅ **Add Voice Settings** - Allow voice speed/pitch control
5. ✅ **Add Streaming** - Real-time audio streaming

---

**Made with ❤️ for LiV.AI Voice Agent**
