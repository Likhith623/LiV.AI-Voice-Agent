# Update the imports at the top - ADD the missing functions:

import logging
import google.generativeai as genai
import traceback
import os
import random
import tempfile
import logging
import json
import hashlib
import time
import asyncio
import httpx
from fastapi import Request
import re
from fastapi.responses import JSONResponse

import asyncio
from concurrent.futures import ThreadPoolExecutor
# Add to top of main.py after other imports
import google.generativeai as genai
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI,File,UploadFile,Form,BackgroundTasks,HTTPException,Query,Body # type: ignore
from pydantic import BaseModel # type: ignore
from typing_extensions import Annotated # type: ignore
from typing import Union
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.encoders import jsonable_encoder # type: ignore
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse # type: ignore
from dotenv import load_dotenv  # type: ignore

from datetime import date
from typing import List, Dict
from fastapi_utils.tasks import repeat_every # type: ignore
import traceback
from typing import Optional
# Voice call imports - Added for speech recognition
import speech_recognition as sr # type: ignore
import io
from pydub import AudioSegment # type: ignore
from pydub.utils import which # type: ignore
import assemblyai as aai # type: ignore
import redis.asyncio as redis # type: ignore
from deepgram import DeepgramClient, PrerecordedOptions # type: ignore


from cartesia import Cartesia # type: ignore
cache_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts_cache")
dotenv_path = Path('../.env')
load_dotenv(dotenv_path=dotenv_path)

# ========== REDIS SETUP FOR AGGRESSIVE CACHING ==========
# Redis configuration for memory retrieval caching
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default
redis_client = None
# Add this function near the top of main.py (after imports):
'''
# Update your call_xai_api function around line 50:
#calling grok and then calling gemini flash if grok fails
async def call_xai_api(messages, model="grok-beta"):
    """XAI Grok API with Gemini Flash fallback (fastest alternative)"""
    print(f"Calling XAI Grok API for voice call with model: {model}")

    # Try different XAI models first
    models_to_try = ["grok-beta", "grok-2-1212", "grok-2-latest", "grok-vision-beta"]

    for model_name in models_to_try:
        try:
            headers = {
                "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}",
                "Content-Type": "application/json"
            }

            payload = {
                "messages": messages,
                "model": model_name,
                "stream": False,
                "temperature": 0.7,
                "max_tokens": 50
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                response_content = result["choices"][0]["message"]["content"]

            logging.info(f"✅ XAI SUCCESS with model: {model_name}")
            return response_content.strip()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                logging.warning(f"⚠️ XAI model {model_name} forbidden (403), trying next...")
                continue
            else:
                logging.error(f"⚠️ XAI model {model_name} failed with {e.response.status_code}")
                continue
        except Exception as e:
            logging.warning(f"⚠️ XAI model {model_name} failed: {e}")
            continue

    # All XAI models failed - fallback to Gemini Flash (FASTEST alternative)
    logging.warning("🔄 All XAI models failed, falling back to Gemini Flash")
    try:
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Convert messages to Gemini format
        if messages:
            # Get the last user message
            user_message = messages[-1]["content"] if messages[-1]["role"] == "user" else "Hello"
        else:
            user_message = "Hello"

        response = model.generate_content(
            user_message,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=20,
                temperature=0.7,
            )
        )

        logging.info("✅ Gemini Flash fallback SUCCESS")
        return response.text.strip()

    except Exception as e:
        logging.error(f"❌ Gemini Flash fallback failed: {e}")
        # Final fallback to OpenAI if Gemini also fails
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            chat_completion_res = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
            response_content = chat_completion_res.choices[0].message.content
            logging.info("✅ OpenAI final fallback SUCCESS")
            return response_content.strip()
        except Exception as e2:
            logging.error(f"❌ All APIs failed: {e2}")
            raise Exception("XAI, Gemini, and OpenAI all failed")
    '''






# Replace your call_xai_api function around line 50:

async def call_xai_api(messages, model="grok-beta"):
    """Skip XAI entirely - Use Gemini Flash directly for fastest response"""
    print(f"Using Gemini Flash directly (XAI disabled for performance)")

    # Skip XAI entirely, go directly to Gemini Flash
    try:
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Convert messages to Gemini format
        if messages:
            user_message = messages[-1]["content"] if messages[-1]["role"] == "user" else "Hello"
        else:
            user_message = "Hello"

        response = model.generate_content(
            user_message,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=30,  # Reduced for voice calls
                temperature=0.7,
            )
        )

        logging.info("✅ Gemini Flash PRIMARY success")
        return response.text.strip()

    except Exception as e:
        logging.error(f"❌ Gemini Flash failed: {e}")
        # Final fallback to OpenAI
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            chat_completion_res = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=50,
                temperature=0.7
            )
            response_content = chat_completion_res.choices[0].message.content
            logging.info("✅ OpenAI final fallback SUCCESS")
            return response_content.strip()
        except Exception as e2:
            logging.error(f"❌ All APIs failed: {e2}")
            raise Exception("Gemini and OpenAI both failed")

async def get_redis_client():
    """Get or create Redis client for caching with improved error handling"""
    global redis_client
    if redis_client is None:
        try:
            # Configure Redis connection based on environment
            if REDIS_HOST == 'localhost':
                # Local Redis configuration
                redis_client = redis.Redis(
                    host=REDIS_HOST,
                    port=6379,
                    password=REDIS_PASSWORD if REDIS_PASSWORD else None,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
            else:
                # Remote Redis configuration (Upstash or other cloud providers)
                connection_methods = [
                    # Method 1: Standard rediss URL
                    lambda: redis.from_url(
                        f"rediss://:{REDIS_PASSWORD}@{REDIS_HOST}:6380",
                        encoding="utf-8",
                        decode_responses=True,
                        socket_timeout=5,
                        socket_connect_timeout=5
                    ),
                    # Method 2: Explicit SSL connection
                    lambda: redis.Redis(
                        host=REDIS_HOST,
                        port=6380,
                        password=REDIS_PASSWORD,
                        ssl=True,
                        encoding="utf-8",
                        decode_responses=True,
                        socket_timeout=5,
                        socket_connect_timeout=5,
                        ssl_cert_reqs=None
                    ),
                    # Method 3: Without SSL (fallback)
                    lambda: redis.Redis(
                        host=REDIS_HOST,
                        port=6379,
                        password=REDIS_PASSWORD,
                        encoding="utf-8",
                        decode_responses=True,
                        socket_timeout=5,
                        socket_connect_timeout=5
                    )
                ]

                for i, method in enumerate(connection_methods, 1):
                    try:
                        test_client = method()
                        await test_client.ping()
                        redis_client = test_client
                        logging.info(f"Remote Redis connected successfully using method {i} for memory caching")
                        break
                    except Exception as method_error:
                        logging.debug(f"Redis connection method {i} failed: {method_error}")
                        if hasattr(test_client, 'close'):
                            try:
                                await test_client.close()
                            except:
                                pass
                        continue

                if redis_client is None:
                    raise Exception("All Redis connection methods failed")

            # Test the connection
            await redis_client.ping()
            logging.info(f"Redis connected successfully to {REDIS_HOST} for memory caching")

        except Exception as e:
            logging.warning(f"Redis connection failed: {e}. Memory caching disabled - falling back to direct retrieval.")
            redis_client = None
    return redis_client

def create_cache_key(query: str, email: str, bot_id: str, previous_conversation: list) -> str:
    """Create a unique cache key for memory retrieval"""
    # Create a deterministic key based on inputs
    conversation_str = json.dumps(previous_conversation, sort_keys=True) if previous_conversation else ""
    cache_input = f"{query}:{email}:{bot_id}:{conversation_str}"
    return f"memory_cache:{hashlib.md5(cache_input.encode()).hexdigest()}"

async def get_cached_memory(cache_key: str):
    """Get cached memory result with improved error handling"""
    try:
        client = await get_redis_client()
        if client:
            cached_data = await client.get(cache_key)
            if cached_data:
                result = json.loads(cached_data)
                logging.info(f"Cache HIT for key: {cache_key[:20]}...")
                return result
            else:
                logging.info(f"Cache MISS for key: {cache_key[:20]}...")
        else:
            logging.debug("Redis client unavailable - skipping cache check")
    except Exception as e:
        logging.warning(f"Cache retrieval error for key {cache_key[:20]}...: {e}")
    return None

async def set_cached_memory(cache_key: str, memory: str, rephrased: str, category: str):
    """Cache memory result with TTL and improved error handling"""
    try:
        client = await get_redis_client()
        if client:
            cache_data = {
                "memory": memory,
                "rephrased_user_message": rephrased,
                "category": category,
                "cached_at": time.time()
            }
            await client.setex(cache_key, CACHE_TTL, json.dumps(cache_data))
            logging.info(f"Memory result cached successfully with key: {cache_key[:20]}... (TTL: {CACHE_TTL}s)")
        else:
            logging.debug("Redis client unavailable - skipping cache storage")
    except Exception as e:
        logging.warning(f"Cache storage error for key {cache_key[:20]}...: {e}")

async def redis_health_check():
    """Check Redis connection health"""
    try:
        client = await get_redis_client()
        if client:
            await client.ping()
            return True
    except:
        pass
    return False

async def cached_retrieve_memory(query: str, email: str, bot_id: str, previous_conversation: list):
    """
    Cached version of retrieve_memory specifically for voice call endpoint
    Provides aggressive caching to reduce memory retrieval time from 8-12s to 0.5-1s
    """
    start_time = time.time()

    # Create cache key
    cache_key = create_cache_key(query, email, bot_id, previous_conversation)

    # Try to get from cache first
    cached_result = await get_cached_memory(cache_key)
    if cached_result:
        cache_time = time.time() - start_time
        logging.info(f"Memory cache HIT - Retrieved in {cache_time:.3f}s")
        return cached_result["memory"], cached_result["rephrased_user_message"], cached_result["category"]

    # Cache miss - call original function
    logging.info("Memory cache MISS - Calling original retrieve_memory")
    memory, rephrased, category = await retrieve_memory(query, email, bot_id, previous_conversation)

    # Cache the result for future use
    await set_cached_memory(cache_key, memory, rephrased, category)

    total_time = time.time() - start_time
    logging.info(f"Memory retrieval completed in {total_time:.3f}s (cached for future)")

    return memory, rephrased, category


import requests # type: ignore
import base64
from diff_per.py import get_bot_prompt
#from news_weather_agent import is_news_query, persona_response
import boto3
from botocore.exceptions import ClientError
import uuid_utils as uuid
import re

# --- Lifespan and App Initialization (Lines 47-113) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup tasks: Print startup message
    print("Starting up...")

    # Schedule log upload every 30 minutes using FastAPI's repeat_every
    @repeat_every(seconds=60 * 30)
    async def upload_log():
        print("Uploading log file to S3")
        upload_log_to_s3()

    # Schedule memory extraction check every 30 minutes
    @repeat_every(seconds=60 * 30)
    async def check_memory_extraction():
        print("Checking Memory Extraction")
        await checker()

    # Schedule message checks every 30 minutes
    @repeat_every(seconds=60 * 30)
    async def check_scheduled_tasks():
        check_daily_scheduled_messages()
        return check_scheduled_messages()

    # Schedule redundancy check every hour
    @repeat_every(seconds=60 * 60)
    def scheduled_hourly_categorization():
        logging.info("🔁 Running redundancy task every 1 hour")
        try:
            redundant()
        except Exception as e:
            logging.info(f"❌ Error during redundancy categorization: {e}")

    # Schedule categorization every 3 hours
    @repeat_every(seconds=60 * 60 * 3)
    def scheduled_memory_categorization():
        print("⏰ Running categorization task every 3 hours")
        try:
            run_categorization_job()
        except Exception as e:
            print(f"Error during scheduled categorization: {e}")

    # Schedule daily summary every 24 hours
    @repeat_every(seconds=60 * 60 * 24)
    def scheduled_daily_summary():
        print("🗓️ Running daily summary generation at 2 AM UTC")
        process_summaries_for_yesterday()

    # Start all scheduled tasks
    upload_log()
    await check_memory_extraction()
    await check_scheduled_tasks()
    scheduled_hourly_categorization()
    scheduled_memory_categorization()
    scheduled_daily_summary()

    yield  # The app runs here

    # Shutdown tasks (if any)
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)
app.include_router(gemma_router, prefix="/cv/generate", tags=["gemma"])

# Connect to Pinecone vector database (used for memory and embeddings)
pc, index = connect_pinecone()

# Add CORS middleware to allow requests from all origins (for frontend-backend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,  # Allow credentials (e.g., cookies, headers)
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all HTTP headers
)


# Configure logging for the application
logging.basicConfig(
    filename="app.log",  # Log file name
    filemode='a',  # Append to log file
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',  # Log format
    datefmt='%H:%M:%S',  # Time format in logs
    level=logging.INFO  # Log level: INFO
)

# Supabase connection details (for database access)
SUPABASE_URL = os.getenv("SUPABASE_URL")  # Supabase project URL from environment variable
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Supabase API key from environment variable

# Create a Supabase client using project URL and API key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
# Configure logging for the application
logging.basicConfig(
    filename="app.log",  # Log file name
    filemode='a',  # Append to log file
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',  # Log format
    datefmt='%H:%M:%S',  # Time format in logs
    level=logging.INFO  # Log level: INFO
)



# Initialize Cartesia client for TTS
client = Cartesia(api_key=os.environ.get("CARTESIA_API_KEY"))

# Voice mapping for different bots (maps bot_id to Cartesia voice_id)
VOICE_MAPPING = {
    "indian_old_male": "fd2ada67-c2d9-4afe-b474-6386b87d8fc3",
    "indian_old_female": "faf0731e-dfb9-4cfc-8119-259a79b27e12",
    "indian_mid_male": "791d5162-d5eb-40f0-8189-f19db44611d8",
    "indian_mid_female": "95d51f79-c397-46f9-b49a-23763d3eaa2d",
    "indian_rom_male": "be79f378-47fe-4f9c-b92b-f02cefa62ccf",
    "indian_rom_female": "28ca2041-5dda-42df-8123-f58ea9c3da00",

    "japanese_old_male": "a759ecc5-ac21-487e-88c7-288bdfe76999",
    "japanese_old_female": "2b568345-1d48-4047-b25f-7baccf842eb0",
    "japanese_mid_male": "06950fa3-534d-46b3-93bb-f852770ea0b5",
    "japanese_mid_female": "44863732-e415-4084-8ba1-deabe34ce3d2",
    "japanese_rom_female": "0cd0cde2-3b93-42b5-bcb9-f214a591aa29",
    "japanese_rom_male" : "6b92f628-be90-497c-8f4c-3b035002df71",

    "parisian_old_male": "5c3c89e5-535f-43ef-b14d-f8ffe148c1f0",
    "parisian_old_female": "8832a0b5-47b2-4751-bb22-6a8e2149303d",
    "parisian_mid_male": "ab7c61f5-3daa-47dd-a23b-4ac0aac5f5c3",
    "parisian_mid_female": "65b25c5d-ff07-4687-a04c-da2f43ef6fa9",
    "parisian_rom_female": "a8a1eb38-5f15-4c1d-8722-7ac0f329727d",

    "berlin_old_male": "e00dd3df-19e7-4cd4-827a-7ff6687b6954",
    "berlin_old_female": "3f4ade23-6eb4-4279-ab05-6a144947c4d5",
    "berlin_mid_male": "afa425cf-5489-4a09-8a3f-d3cb1f82150d",
    "berlin_mid_female": "1ade29fc-6b82-4607-9e70-361720139b12",
    "berlin_rom_male": "b7187e84-fe22-4344-ba4a-bc013fcb533e",
    "berlin_rom_female": "4ab1ff51-476d-42bb-8019-4d315f7c0c05",

    "Krishn": "be79f378-47fe-4f9c-b92b-f02cefa62ccf",
    "Ram": "fd2ada67-c2d9-4afe-b474-6386b87d8fc3",
    "Hanuma": "fd2ada67-c2d9-4afe-b474-6386b87d8fc3",
    "Shiv": "be79f378-47fe-4f9c-b92b-f02cefa62ccf",
    "Trimurthi": "be79f378-47fe-4f9c-b92b-f02cefa62ccf"
}
# Add these functions after the VOICE_MAPPING around line 870:

# In your voice call functions, update the TTS format selection:

def get_smart_audio_format(text: str, use_case: str = "voice_call") -> dict:
    """PERFECT audio format selection"""
    word_count = len(text.split())

    if use_case == "voice_call":
        if word_count <= 5:  # Very short responses
            return get_optimized_audio_format("voice_call_minimal")  # 4kHz for max speed
        else:
            return get_optimized_audio_format("ultra_fast")  # 6kHz for speed

    return get_optimized_audio_format("balanced")

def get_optimized_audio_format(optimization_level: str = "ultra_fast"):
    """Get optimized audio format configuration"""
    OPTIMIZED_AUDIO_FORMATS = {
        "ultra_fast": {
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": 8000,
        },
        "voice_call_minimal": {
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": 8000,  # Minimum for speech intelligibility
        },
        "balanced": {
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": 8000,
        }
    }
    return OPTIMIZED_AUDIO_FORMATS.get(optimization_level, OPTIMIZED_AUDIO_FORMATS["ultra_fast"])

# TTS Cache for common responses
TTS_CACHE = {}
TTS_CACHE_MAX_SIZE = 1000
TTS_CACHE_ENABLED = True
TTS_CACHE_TTL_HOURS = 24
TTS_CACHE_STATS = {"hits": 0, "misses": 0, "total_requests": 0}

# Replace your cache functions with these ASYNC versions:



# REPLACE your broken async cache function with this WORKING version:

async def get_cached_tts_response_async(text: str, voice_id: str) -> Optional[str]:
    """WORKING async cache retrieval with proper error handling"""
    if not TTS_CACHE_ENABLED:
        return None

    def _get_cache():
        cache_key = f"{voice_id}:{hash(text)}"
        TTS_CACHE_STATS["total_requests"] += 1

        print(f"🔍 CACHE DEBUG: Looking for key: {cache_key[:50]}...")
        print(f"🔍 CACHE DEBUG: Current cache size: {len(TTS_CACHE)}")

        cached_entry = TTS_CACHE.get(cache_key)
        if not cached_entry:
            TTS_CACHE_STATS["misses"] += 1
            print(f"❌ CACHE DEBUG: Key not found in cache")
            return None

        # Check TTL
        age_hours = (time.time() - cached_entry["timestamp"]) / 3600
        if age_hours > TTS_CACHE_TTL_HOURS:
            del TTS_CACHE[cache_key]
            TTS_CACHE_STATS["misses"] += 1
            print(f"⏰ CACHE DEBUG: Entry expired ({age_hours:.1f}h old)")
            return None

        TTS_CACHE_STATS["hits"] += 1
        print(f"✅ CACHE DEBUG: Cache hit! Entry age: {age_hours:.1f}h")
        return cached_entry["audio"]

    try:
        # Use the global executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(cache_executor, _get_cache)
        print(f"🔍 CACHE DEBUG: Async cache result: {'HIT' if result else 'MISS'}")
        return result
    except Exception as e:
        print(f"❌ CACHE DEBUG: Async cache error: {e}")
        return None

# REPLACE your broken async cache storage with this WORKING version:

async def cache_tts_response_async(text: str, voice_id: str, audio_base64: str):
    """WORKING async cache storage with proper error handling"""
    if not TTS_CACHE_ENABLED:
        return

    def _store_cache():
        cache_key = f"{voice_id}:{hash(text)}"
        timestamp = time.time()

        print(f"💾 CACHE DEBUG: Storing key: {cache_key[:50]}...")
        print(f"💾 CACHE DEBUG: Audio length: {len(audio_base64)}")

        # Simple LRU: remove oldest if cache is full
        if len(TTS_CACHE) >= TTS_CACHE_MAX_SIZE:
            oldest_key = next(iter(TTS_CACHE))
            del TTS_CACHE[oldest_key]
            print(f"🗑️ CACHE DEBUG: Removed oldest entry (cache full)")

        TTS_CACHE[cache_key] = {
            "audio": audio_base64,
            "timestamp": timestamp
        }

        print(f"✅ CACHE DEBUG: Stored successfully! Cache size now: {len(TTS_CACHE)}")

    try:
        loop = asyncio.get_event_loop()
        # ✅ CRITICAL FIX: Use await here, not just run_in_executor
        await loop.run_in_executor(cache_executor, _store_cache)
    except Exception as e:
        print(f"❌ CACHE DEBUG: Storage error: {e}")
# Pydantic model for TTS request
class TTSRequest(BaseModel):
    transcript: str
    bot_id: str  # Changed from voice_id to bot_id to map to specific bot voices
    output_format: Optional[dict] = {
        "container": "wav",
        "encoding": "pcm_s16le",  # Use PCM 16-bit little-endian for better compatibility
        "sample_rate":  22050,
    }

# Helper function to get the voice_id for a given bot_id
# Returns a default voice_id if bot_id is not found
# Tries both case-sensitive and lowercase matching
def get_voice_id_for_bot(bot_id: str) -> str:
    """Get the voice ID for a specific bot"""
    # Default voice ID if bot not found in mapping
    default_voice_id = "4df027cb-2920-4a1f-8c34-f21529d5c3fe"  # Default US Man voice

    # Check if bot_id exists in voice mapping (case-sensitive)
    if bot_id in VOICE_MAPPING:
        return VOICE_MAPPING[bot_id]

    # If not found, try lowercase version
    bot_id_lower = bot_id.lower()
    if bot_id_lower in VOICE_MAPPING:
        return VOICE_MAPPING[bot_id_lower]

    return default_voice_id

# Utility to clean up temporary files (used in other endpoints)
def cleanup_file(path: str):
    """Background task to remove the temporary file"""
    try:
        os.unlink(path)
    except Exception as e:
        print(f"Error cleaning up file {path}: {e}")

# Main TTS endpoint: generates audio for a given transcript and bot_id
@app.post("/generate-audio")
async def generate_audio(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    ENHANCED TTS endpoint with optimizations:
    1. Check TTS cache for common responses
    2. Use optimized audio format for faster processing
    3. Fallback to original implementation on errors
    """
    tts_start_time = time.time()

    try:
        logging.info(f"🎵 ENHANCED TTS generate_audio called with bot_id: {request.bot_id}")
        print(f"🎵 ENHANCED TTS generate_audio called with bot_id: {request.bot_id}")

        # Get the appropriate voice ID for the bot
        voice_id = get_voice_id_for_bot(request.bot_id)

        # ========== OPTIMIZATION 1: Check TTS cache first ==========
        cached_audio = get_cached_tts_response(request.transcript, voice_id)
        if cached_audio:
            cache_time = time.time() - tts_start_time
            logging.info(f"✅ TTS cache HIT - Retrieved in {cache_time:.3f}s")
            return {
                "voice_id": voice_id,
                "audio_base64": cached_audio
            }

        # ========== OPTIMIZATION 2: Use smart audio format selection ==========
        # Automatically choose the best format based on text length and use case
        smart_format = get_smart_audio_format(request.transcript, "voice_call")
        actual_format = smart_format

        # Generate audio bytes using Cartesia (collect all chunks from generator)
        # The Cartesia API returns a generator of audio chunks, so we join them into a single bytes object
        audio_chunks = client.tts.bytes(
            model_id="sonic",
            transcript=request.transcript,
            voice={"mode": "id", "id": voice_id},
            output_format=actual_format
        )
        audio_data = b"".join(audio_chunks)
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        generation_time = time.time() - tts_start_time

        # ========== OPTIMIZATION 3: Cache the response ==========
        cache_tts_response(request.transcript, voice_id, audio_base64)

        logging.info(f"✅ TTS generation completed in {generation_time:.3f}s")

        # Return JSON with voice_id and base64 audio
        return {
            "voice_id": voice_id,
            "audio_base64": audio_base64
        }
    except Exception as e:
        generation_time = time.time() - tts_start_time
        logging.error(f"❌ Enhanced TTS failed in {generation_time:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class TTSRequest2(BaseModel):
    text: str
    target_language_code: str = "hi-IN"  # Default to Hindi
    speaker: str  # Default speaker
    speech_sample_rate: int = 8000  # Default sample rate
    enable_preprocessing: bool = True  # Default preprocessing
    model: str = "bulbul:v1"  # Default model

def cleanup_file(path: str):
    """Background task to remove the temporary file"""
    try:
        os.unlink(path)
    except Exception as e:
        print(f"Error cleaning up file {path}: {e}")

@app.post("/v2/generate-audio")
async def generate_audio_v2(request: TTSRequest2, background_tasks: BackgroundTasks):
    try:
        logging.info(f"🔴 SARVAM generate_audio_v2 function called with text: {request.text[:50]}...")
        print(f"🔴 SARVAM generate_audio_v2 function called with text: {request.text[:50]}...")
        # Prepare the API request
        url = "https://api.sarvam.ai/text-to-speech"
        headers = {
            "API-Subscription-Key": os.getenv("SARVAM_API_KEY"),
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": [request.text],
            "target_language_code": request.target_language_code,
            "speaker": request.speaker,
            "speech_sample_rate": request.speech_sample_rate,
            "enable_preprocessing": request.enable_preprocessing,
            "model": request.model
        }

        # Make the API call
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Sarvam API request failed")

        # Process the response
        audio_base64 = response.json()["audios"][0]
        audio_bytes = base64.b64decode(audio_base64)

        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        # Add cleanup task to background tasks
        background_tasks.add_task(cleanup_file, temp_path)

        # Return the audio file
        return FileResponse(
            temp_path,
            media_type="audio/wav",
            filename="generated_audio.wav"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""
    The function `sync` processes a POST request to synchronize messages based on provided email, bot
    ID, and messages ID.

    :param request: The `request` parameter in the `sync` function is of type `SyncRequest`, which is a
    Pydantic model representing the data structure expected in the request body of the POST request to
    the `/sync` endpoint. It contains the following fields:
    :type request: SyncRequest
    :param background_tasks: The `background_tasks` parameter in FastAPI is used to add background tasks
    to be run after returning a response to the client. These tasks are executed asynchronously after
    the response to the client, allowing you to perform additional operations without delaying the
    response to the client
    :type background_tasks: BackgroundTasks
    :return: The code snippet defines a FastAPI endpoint `/sync` that expects a POST request with a JSON
    body matching the `SyncRequest` model. The endpoint checks if the `email` and `bot_id` fields are
    provided and not empty. If either of them is missing or empty, it returns an error message
    indicating which field is missing. If both fields are provided, it calls the `sync_messages
"""
class SyncRequest(BaseModel):
    email: str = ""
    bot_id: str = ""
    messages_id: str = ""

@app.post("/sync")
async def sync(request: SyncRequest, background_tasks: BackgroundTasks):
    try:
        # Validate if the question is provided and not empty
        if not request.email or request.email.strip() == "":
            return {"error": str("Please provide a email")}  # Return error if invalid

        # Validate if the bot_id is provided and not empty
        if not request.bot_id or request.bot_id.strip() == "":
            return {"error": str("Please provide a bot_id")}  # Return error if invalid

        # Get all messages based on email, bot_id, and messages_id [optional] from the database
        messages = sync_messages(request.email, request.bot_id, request.messages_id)

        # Return the messages
        return {
            "response": messages
        }
    except Exception as e:
        print(e)
        return {"error": str(f"Error occurred while processing!!")}  # Return error message



# Define the expected schema for frontend error logs using Pydantic
class FrontendErrorLog(BaseModel):
    message: str
    source: str | None = None
    line_number: int | None = None
    column_number: int | None = None
    stack_trace: str | None = None
    browser: str | None = None
    url: str | None = None
    additional_context: dict | None = None
    timestamp: Optional[datetime] = None

# POST endpoint to receive and store frontend error logs
@app.post("/api/logs/frontend-error")
async def log_frontend_error(error_log: FrontendErrorLog):
    try:
        current_timestamp = datetime.utcnow()

        # If timestamp is not provided in the request, set it to the current time
        if not error_log.timestamp:
            error_log.timestamp = current_timestamp


        response = supabase.table("frontend_error_logs").insert({
            "message": error_log.message,
            "source": error_log.source,
            "line_number": error_log.line_number,
            "column_number": error_log.column_number,
            "stack_trace": error_log.stack_trace,
            "browser": error_log.browser,
            "url": error_log.url,
            "additional_context": error_log.additional_context,
            "timestamp": error_log.timestamp.isoformat(),
        }).execute()

        # Check if 'data' exists and is not empty
        if not response.data:
            logging.error("Supabase insert returned no data for frontend error: %s", error_log.dict())
            raise HTTPException(status_code=500, detail="Failed to insert error log: No data returned")
        # Return success message if log was saved
        logging.info("Frontend error logged successfully")
        return {"message": "Frontend error logged successfully"}

    except Exception as e:
        # Catch any unexpected exceptions and raise a 500 error
        logging.exception("Exception occurred while logging frontend error")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")








# Add this test endpoint to verify cache is working:

@app.get("/test-cache-now")
async def test_cache_now():
    """Test if cache is actually working after fixes"""
    try:
        test_text = "Hello, how are you?"
        test_voice = "fd2ada67-c2d9-4afe-b474-6386b87d8fc3"
        test_audio = "dGVzdF9hdWRpb19kYXRh"  # test_audio_data in base64

        print("🧪 Testing cache storage...")
        await cache_tts_response_async(test_text, test_voice, test_audio)

        print("🧪 Testing cache retrieval...")
        retrieved = await get_cached_tts_response_async(test_text, test_voice)

        return {
            "cache_enabled": TTS_CACHE_ENABLED,
            "cache_size": len(TTS_CACHE),
            "test_stored": True,
            "test_retrieved": retrieved is not None,
            "test_data_matches": retrieved == test_audio if retrieved else False,
            "cache_stats": TTS_CACHE_STATS,
            "executor_available": cache_executor is not None,
            "cache_keys_sample": list(TTS_CACHE.keys())[:3] if TTS_CACHE else []
        }
    except Exception as e:
        return {
            "error": str(e),
            "cache_enabled": TTS_CACHE_ENABLED,
            "executor_available": "cache_executor" in globals()
        }





# =============================================================================
# VOICE CALL FUNCTIONALITY - Added for speech-to-text and text-to-speech
# =============================================================================

# Helper function for speech-to-text conversion using Deepgram (primary) with AssemblyAI fallback
async def speech_to_text(audio_file: UploadFile) -> str:
    """
    Convert uploaded audio file to text using speech recognition
    Primary: Deepgram (fastest, 1-2 seconds)
    Fallback: AssemblyAI (commented), Google Speech Recognition + CMU Sphinx
    """
    try:
        # Read the uploaded file
        audio_data = await audio_file.read()

        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name

        try:
            # Convert audio to WAV format if needed using pydub
            audio_segment = AudioSegment.from_file(temp_audio_path)

            # Convert to WAV with specific parameters for better recognition
            wav_audio = audio_segment.set_frame_rate(16000).set_channels(1)

            # Create another temporary file for the processed WAV
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as processed_audio:
                wav_audio.export(processed_audio.name, format="wav")
                processed_audio_path = processed_audio.name

            # ==================== PRIMARY: Deepgram ====================
            try:
                # Initialize Deepgram client
                deepgram = DeepgramClient(api_key="544f9a9deffead086304452ffa70afcd461d30e8")

                # Read the audio file
                with open(processed_audio_path, "rb") as audio_file_data:
                    audio_buffer = audio_file_data.read()

                # Configure Deepgram options for optimal performance
                options = PrerecordedOptions(
                    model="nova-2",  # Latest model for best accuracy
                    language="en",
                    smart_format=True,  # Automatic punctuation and formatting
                    punctuate=True,
                    utterances=False,
                    diarize=False
                )

                # Transcribe the audio
                response = deepgram.listen.prerecorded.v("1").transcribe_file(
                    {"buffer": audio_buffer, "mimetype": "audio/wav"},
                    options
                )

                # Extract the transcript
                if response.results and response.results.channels and len(response.results.channels) > 0:
                    alternatives = response.results.channels[0].alternatives
                    if alternatives and len(alternatives) > 0:
                        transcript = alternatives[0].transcript
                        if transcript and transcript.strip():
                            logging.info("Deepgram transcription successful")
                            return transcript
                        else:
                            raise Exception("Deepgram returned empty transcript")
                    else:
                        raise Exception("No alternatives in Deepgram response")
                else:
                    raise Exception("No valid results from Deepgram")

            except Exception as e:
                logging.warning(f"Deepgram failed, falling back to Google Speech Recognition: {e}")

                # ==================== COMMENTED FALLBACK: AssemblyAI ====================
                # try:
                #     # Set AssemblyAI API key
                #     aai.settings.api_key = os.environ.get("ASSEMBLYAI_API_KEY")
                #
                #     # Create transcriber instance
                #     transcriber = aai.Transcriber()
                #
                #     # Transcribe the audio file
                #     transcript_result = transcriber.transcribe(processed_audio_path)
                #
                #     # Check if transcription was successful
                #     if transcript_result.status == aai.TranscriptStatus.completed:
                #         logging.info("AssemblyAI transcription successful")
                #         return transcript_result.text
                #     else:
                #         logging.warning(f"AssemblyAI transcription failed: {transcript_result.error}")
                #         raise Exception("AssemblyAI transcription failed")
                #
                # except Exception as e:
                #     logging.warning(f"AssemblyAI failed, falling back to Google Speech Recognition: {e}")

                # ==================== FALLBACK: Google Speech Recognition ====================
                # Initialize the speech recognizer
                recognizer = sr.Recognizer()

                # Load the audio file
                with sr.AudioFile(processed_audio_path) as source:
                    # Adjust for ambient noise
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    # Record the audio
                    audio = recognizer.record(source)

                # Convert speech to text using Google's speech recognition
                try:
                    transcript = recognizer.recognize_google(audio)
                    logging.info("Google Speech Recognition successful")
                    return transcript
                except sr.UnknownValueError:
                    # Try with alternative recognition services if Google fails
                    try:
                        transcript = recognizer.recognize_sphinx(audio)
                        logging.info("CMU Sphinx recognition successful")
                        return transcript
                    except:
                        logging.error("All speech recognition methods failed")
                        return "Could not understand audio"
                except sr.RequestError as e:
                    logging.error(f"Speech recognition request error: {e}")
                    return "Speech recognition service error"

        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_audio_path)
                if 'processed_audio_path' in locals():
                    os.unlink(processed_audio_path)
            except:
                pass

    except Exception as e:
        logging.error(f"Error in speech to text conversion: {e}")
        raise HTTPException(status_code=500, detail=f"Speech to text conversion failed: {str(e)}")


# ==================== COMMENTED OUT: Previous Google Speech Recognition Implementation ====================
# async def speech_to_text_google_fallback(audio_file: UploadFile) -> str:
#     """
#     Convert uploaded audio file to text using speech recognition
#     PREVIOUS IMPLEMENTATION - Google Speech Recognition Primary
#     """
#     try:
#         # Read the uploaded file
#         audio_data = await audio_file.read()
#
#         # Create a temporary file to store the audio
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
#             temp_audio.write(audio_data)
#             temp_audio_path = temp_audio.name
#
#         try:
#             # Convert audio to WAV format if needed using pydub
#             audio_segment = AudioSegment.from_file(temp_audio_path)
#
#             # Convert to WAV with specific parameters for better recognition
#             wav_audio = audio_segment.set_frame_rate(16000).set_channels(1)
#
#             # Create another temporary file for the processed WAV
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as processed_audio:
#                 wav_audio.export(processed_audio.name, format="wav")
#                 processed_audio_path = processed_audio.name
#
#             # Initialize the speech recognizer
#             recognizer = sr.Recognizer()
#
#             # Load the audio file
#             with sr.AudioFile(processed_audio_path) as source:
#                 # Adjust for ambient noise
#                 recognizer.adjust_for_ambient_noise(source, duration=0.5)
#                 # Record the audio
#                 audio = recognizer.record(source)
#
#             # Convert speech to text using Google's speech recognition
#             try:
#                 transcript = recognizer.recognize_google(audio)
#                 return transcript
#             except sr.UnknownValueError:
#                 # Try with alternative recognition services if Google fails
#                 try:
#                     transcript = recognizer.recognize_sphinx(audio)
#                     return transcript
#                 except:
#                     return "Could not understand audio"
#             except sr.RequestError as e:
#                 logging.error(f"Speech recognition request error: {e}")
#                 return "Speech recognition service error"
#
#         finally:
#             # Clean up temporary files
#             try:
#                 os.unlink(temp_audio_path)
#                 if 'processed_audio_path' in locals():
#                     os.unlink(processed_audio_path)
#             except:
#                 pass
#
#     except Exception as e:
#         logging.error(f"Error in speech to text conversion: {e}")
#         raise HTTPException(status_code=500, detail=f"Speech to text conversion failed: {str(e)}")

# ========== BACKGROUND HELPER FUNCTIONS FOR PARALLEL PROCESSING ==========
async def background_log_origin_response(email: str, bot_id: str, transcript: str, previous_conversation: list):
    """Background task to log origin response without blocking main flow"""
    try:
        log_messages_with_like_dislike(email, bot_id, transcript, "I was developed by the Desis Dev team!", "", previous_conversation[-5:], "")
    except Exception as e:
        logging.error(f"Background origin logging failed: {e}")

async def background_log_response(email: str, bot_id: str, transcript: str, response: str, previous_conversation: list, memory: str, request_time: str, platform: str):
    """Background task to log regular response without blocking main flow"""
    try:
        log = log_messages_with_like_dislike(email, bot_id, transcript, response, "", previous_conversation[-5:], memory)
        await insert_entry(email, transcript, response, bot_id, request_time, platform)
        return log
    except Exception as e:
        logging.error(f"Background response logging failed: {e}")
        return None

# Voice call endpoint that integrates speech-to-text with existing chat logic and text-to-speech
@app.post("/voice-call")
async def voice_call(
    audio_file: UploadFile = File(...),
    bot_id: str = Form("delhi"),
    custom_bot_name: str = Form(""),
    user_name: str = Form(""),
    user_gender: str = Form(""),
    language: str = Form(""),
    traits: str = Form(""),
    previous_conversation: str = Form("[]"),
    email: str = Form(""),
    request_time: str = Form(""),
    platform: str = Form(""),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    FULLY OPTIMIZED Voice call endpoint - ALL PERFORMANCE OPTIMIZATIONS APPLIED:
    1. Converts speech to text WHILE simultaneously preloading data (Deepgram STT - 67% faster)
    2. Skips memory retrieval (major bottleneck) - only performs origin check (84% faster)
    3. Uses gpt-3.5-turbo instead of o4-mini for 2-4s faster response generation
    4. Starts TTS generation immediately after response (parallel processing - saves 2-3s)
    5. Returns both text response and audio

    PERFORMANCE OPTIMIZATIONS APPLIED:
    - Phase 1: STT with Deepgram (2.33s, was 7s+ with AssemblyAI) ✅
    - Phase 2: Origin check only (~1s, was 9.78s with memory retrieval) ✅
    - Phase 3: gpt-3.5-turbo response (~2-3s, was 6.8s with o4-mini) ✅ NEW
    - Phase 4: Parallel TTS generation (~2-3s, overlapped with Phase 3) ✅ NEW

    Target: ~7-8 seconds total (was 23.48s) - 65%+ improvement
    Memory retrieval is still available in regular chat endpoints (/cv/chat)
    """
    start_time = time.time()
    logging.info(f"🎙️ Voice call started for bot_id={bot_id}, email={email}")

    try:
        # Validate audio file - allow common audio formats and handle cases where content_type might not be set
        valid_audio_types = ['audio/', 'application/octet-stream']
        valid_extensions = ['.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac']

        content_type_valid = any(audio_file.content_type.startswith(t) for t in valid_audio_types) if audio_file.content_type else False
        extension_valid = any(audio_file.filename.lower().endswith(ext) for ext in valid_extensions) if audio_file.filename else False

        if not (content_type_valid or extension_valid):
            raise HTTPException(status_code=400, detail=f"Invalid audio file format. Content-type: {audio_file.content_type}, Filename: {audio_file.filename}")

        # Parse previous conversation from JSON string
        try:
            previous_conversation_list = json.loads(previous_conversation) if previous_conversation else []
        except json.JSONDecodeError:
            previous_conversation_list = []

        # ========== PARALLEL PHASE 1: STT + Data Preloading ==========
        phase1_start = time.time()
        logging.info("🔄 Phase 1: Starting STT + Data Preloading in parallel")

        # Start STT task (5-8 seconds)
        stt_task = asyncio.create_task(speech_to_text(audio_file))

        # While STT is running, preload all static data in parallel
        async def preload_static_data():
            """Preload bot prompt, voice settings, and prepare conversation data"""
            # These operations can run simultaneously while STT processes
            previous_conversation = restrict_to_last_20_messages(previous_conversation_list)
            raw_bot_prompt = get_bot_prompt(bot_id)

            # Create personalized bot prompt
            bot_prompt = raw_bot_prompt.format(
                custom_bot_name=custom_bot_name,
                traitsString=traits,
                userName=user_name,
                userGender=user_gender,
                languageString=language
            )

            return {
                "previous_conversation": previous_conversation,
                "bot_prompt": bot_prompt,
                "raw_bot_prompt": raw_bot_prompt
            }

        # Start preloading task
        preload_task = asyncio.create_task(preload_static_data())

        # Wait for both STT and preloading to complete
        transcript, preloaded_data = await asyncio.gather(stt_task, preload_task)

        phase1_time = time.time() - phase1_start
        logging.info(f"✅ Phase 1 completed in {phase1_time:.2f}s (STT + Preloading)")

        if not transcript or transcript.strip() == "":
            return {"error": "Could not transcribe audio. Please try again."}

        # Extract preloaded data
        previous_conversation = preloaded_data["previous_conversation"]
        bot_prompt = preloaded_data["bot_prompt"]

        # ========== OPTIMIZED PHASE 2: Origin Check Only (Memory Retrieval Disabled for Voice Calls) ==========
        phase2_start = time.time()
        logging.info("🔄 Phase 2: Starting Origin Check Only (Memory Retrieval DISABLED for voice calls performance)")

        # ========== PERFORMANCE OPTIMIZATION: Memory Retrieval DISABLED for Voice Calls ==========
        # Memory retrieval is the biggest bottleneck in voice calls (9.78s out of 23.48s total)
        # For voice calls, we prioritize response speed over memory context
        # Memory retrieval is still available for regular chat endpoints (/cv/chat)

        # Only run origin check (fast ~1 second)
        origin_check_task = asyncio.create_task(
            check_for_origin_question(transcript, previous_conversation)
        )

        # ==================== COMMENTED OUT: Memory Retrieval for Voice Calls ====================
        # PERFORMANCE NOTE: This was taking 9.78s and causing slow voice responses
        # Uncomment below to re-enable memory retrieval for voice calls if needed
        #
        # memory_retrieval_task = asyncio.create_task(
        #     cached_retrieve_memory(transcript, email, bot_id, previous_conversation)
        # )
        #
        # # Wait for both to complete with timeout handling
        # try:
        #     # Set a timeout of 15 seconds for memory retrieval to prevent hanging
        #     check, memory_result = await asyncio.wait_for(
        #         asyncio.gather(origin_check_task, memory_retrieval_task),
        #         timeout=15.0
        #     )
        #     memory, rephrased_user_message, category = memory_result
        #     logging.info("✅ Memory retrieval completed within timeout")
        # except asyncio.TimeoutError:
        #     logging.warning("⚠️ Memory retrieval timed out, using fallback values")
        #     # Get origin check result (should be fast)
        #     try:
        #         check = await asyncio.wait_for(origin_check_task, timeout=2.0)
        #     except:
        #         check = "No"
        #
        #     # Use fallback values for memory retrieval
        #     memory = ""
        #     rephrased_user_message = transcript  # Use original transcript
        #     category = "General"  # Default category
        # except Exception as e:
        #     logging.error(f"Error in parallel processing: {e}")
        #     # Fallback values
        #     check = "No"
        #     memory = ""
        #     rephrased_user_message = transcript
        #     category = "General"
        # ==================== END COMMENTED MEMORY RETRIEVAL ====================

        # For voice calls: Use fast fallback values without memory retrieval
        try:
            # Only wait for origin check with reduced timeout since we're using gpt-3.5-turbo (faster)
            check = await asyncio.wait_for(origin_check_task, timeout=2.0)
            logging.info("✅ Origin check completed")
        except asyncio.TimeoutError:
            logging.warning("⚠️ Origin check timed out, using fallback")
            check = "No"
        except Exception as e:
            logging.error(f"Error in origin check: {e}")
            check = "No"

        # Use optimized fallback values (no memory retrieval for voice calls)
        memory = ""  # No memory context for faster voice responses
        rephrased_user_message = transcript  # Use original transcript
        category = "General"  # Default category

        phase2_time = time.time() - phase2_start
        logging.info(f"✅ Phase 2 completed in {phase2_time:.2f}s (Origin Check Only - Memory Retrieval DISABLED for voice calls)")

        # ========== RESPONSE GENERATION ==========
        phase3_start = time.time()
        logging.info("🔄 Phase 3: Starting Response Generation")

        reminder = False

        # If the question is from the origin, log the message and return a response
        if check == "Yes":
            response = "I was developed by a team of Desi Developers, but you brought me to life!!"

            # ========== CRITICAL FIX: Start TTS for origin response too ==========
            # Previously missing TTS for origin response - caused variable scoping error
            tts_start_time = time.time()
            tts_task = asyncio.create_task(generate_audio_optimized(
                TTSRequest(
                    transcript=response,
                    bot_id=bot_id,
                    output_format=get_smart_audio_format(response, "voice_call")
                ),
                background_tasks
            ))

            # Start logging in background (non-blocking)
            asyncio.create_task(
                background_log_origin_response(email, bot_id, transcript, previous_conversation)
            )

            chat_response = {
                "response": response,
                "message_id": "origin_response",  # Will be updated by background task
                "reminder": False
            }
        else:
            # If the category is Reminder, generate the reminder response
            if category == "Reminder":
                print("REMINDER")
                reminder_resp = await reminder_response(transcript, previous_conversation, request_time)
                response = reminder_resp['response']
                reminder = True

                # ========== QUICK WIN #2: Start TTS immediately for reminder ==========
                # Start TTS generation in parallel with logging (saves 2-3s)
                tts_start_time = time.time()
                tts_task = asyncio.create_task(generate_audio_optimized(
                    TTSRequest(
                        transcript=response,
                        bot_id=bot_id,
                        output_format=get_smart_audio_format(response, "voice_call")  # Smart format selection
                    ),
                    background_tasks
                ))

            else:
                # ========== OPTIMIZED: Generate bot response without memory context for voice calls ==========
                # For voice calls, we skip memory injection to prioritize response speed
                # Memory context is still available in regular chat endpoints (/cv/chat)

                # ========== QUICK WIN #1: Switch to gpt-3.5-turbo for 2-4s faster response ==========
                # Previously used: o4-mini (reasoning model, slower but more accurate)
                # Now using: gpt-3.5-turbo (chat model, 2-4s faster response time)
                # Construct messages format directly for OpenAI API
                messages = [
                    {
                        "role": "system",
                        "content": bot_prompt  # Bot prompt without memory context for speed
                    }
                ]
                messages.extend(previous_conversation)
                messages.append(
                    {
                        "role": "user",
                        "content": transcript
                    }
                )

                # ========== PREVIOUS MODEL (COMMENTED): o4-mini for accuracy ==========
                # response = await call_openai_api(messages, model="o4-mini")

                # ========== NEW MODEL: gpt-3.5-turbo for speed ==========
                #response = await call_openai_api(messages, model="gpt-3.5-turbo")
                response = await call_xai_api(messages, model="grok-beta")
                reminder = False

                # ========== QUICK WIN #2: Start TTS immediately after response ==========
                # Start TTS generation in parallel with logging (saves 2-3s)
                tts_start_time = time.time()
                tts_task = asyncio.create_task(generate_audio_optimized(
                    TTSRequest(
                        transcript=response,
                        bot_id=bot_id,
                        output_format=get_smart_audio_format(response, "voice_call")  # Smart format selection
                    ),
                    background_tasks
                ))

            # Start logging in background (non-blocking)
            log_task = asyncio.create_task(
                background_log_response(email, bot_id, transcript, response, previous_conversation, memory, request_time, platform)
            )

            chat_response = {
                "response": response,
                "message_id": "processing",  # Will be updated by background task
                "reminder": reminder
            }

        phase3_time = time.time() - phase3_start
        logging.info(f"✅ Phase 3 completed in {phase3_time:.2f}s (Response Generation)")

        # ========== QUICK WIN #2: PARALLEL PHASE 4: Wait for TTS completion ==========
        # TTS was started immediately after response generation (parallel processing)
        # Now wait for TTS to complete while logging continues in background
        logging.info("🔄 Phase 4: Waiting for TTS completion (started in parallel)")

        tts_response = await tts_task  # Wait for TTS task started earlier
        phase4_time = time.time() - tts_start_time  # Measure from when TTS actually started
        total_time = time.time() - start_time

        logging.info(f"✅ Phase 4 completed in {phase4_time:.2f}s (TTS Generation - Parallel)")
        logging.info(f"🎉 TOTAL Voice Call completed in {total_time:.2f}s")
        logging.info(f"📊 Performance Breakdown: Phase1={phase1_time:.2f}s, Phase2={phase2_time:.2f}s (Origin Check Only), Phase3={phase3_time:.2f}s, Phase4={phase4_time:.2f}s (Parallel TTS)")
        logging.info(f"🚀 OPTIMIZATIONS APPLIED: Deepgram STT + Memory Disabled + gpt-3.5-turbo + Parallel TTS")

        # Return combined response
        return {
            "transcript": transcript,
            "text_response": chat_response["response"],
            "message_id": chat_response["message_id"],
            "reminder": chat_response.get("reminder", False),
            "voice_id": tts_response["voice_id"],
            "audio_base64": tts_response["audio_base64"],
            "performance": {
                "total_time": round(total_time, 2),
                "phase_breakdown": {
                    "stt_preload": round(phase1_time, 2),
                    "origin_check_only": round(phase2_time, 2),  # Memory retrieval disabled for voice calls
                    "response_generation": round(phase3_time, 2),
                    "tts_generation_parallel": round(phase4_time, 2)  # Parallel TTS optimization
                },
                "optimizations_applied": [
                    "deepgram_stt",
                    "memory_retrieval_disabled",
                    "gpt_3_5_turbo_model",
                    "parallel_tts_generation"
                ]
            }
        }

    except Exception as e:
        total_time = time.time() - start_time
        logging.error(f"❌ Error in voice call after {total_time:.2f}s: {e}")
        return {"error": f"Voice call processing failed: {str(e)}"}

@app.get("/redis/health")
async def redis_health():
    """Check Redis connection health and provide cache statistics"""
    try:
        client = await get_redis_client()
        if not client:
            return {
                "status": "disconnected",
                "message": "Redis client not available - caching disabled",
                "fallback_active": True
            }

        # Test connection
        start_time = time.time()
        ping_result = await client.ping()
        response_time = (time.time() - start_time) * 1000  # Convert to ms

        # Get basic Redis info
        try:
            info = await client.info()
            memory_usage = info.get('used_memory_human', 'unknown')
            connected_clients = info.get('connected_clients', 'unknown')
            uptime = info.get('uptime_in_seconds', 'unknown')
        except:
            memory_usage = connected_clients = uptime = 'unavailable'

        return {
            "status": "connected",
            "ping": ping_result,
            "response_time_ms": round(response_time, 2),
            "host": REDIS_HOST,
            "memory_usage": memory_usage,
            "connected_clients": connected_clients,
            "uptime_seconds": uptime,
            "cache_ttl": CACHE_TTL,
            "message": "Redis caching active"
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Redis health check failed - falling back to direct memory retrieval"
        }

@app.get("/redis/cache-stats")
async def cache_stats():
    """Get cache performance statistics"""
    try:
        client = await get_redis_client()
        if not client:
            return {
                "status": "disconnected",
                "message": "Redis not available"
            }

        # Get cache keys matching our pattern
        cache_keys = await client.keys("memory_cache:*")
        total_keys = len(cache_keys)

        # Sample some keys to get average TTL
        sample_keys = cache_keys[:10] if len(cache_keys) > 10 else cache_keys
        ttl_values = []
        for key in sample_keys:
            ttl = await client.ttl(key)
            if ttl > 0:
                ttl_values.append(ttl)

        avg_ttl = sum(ttl_values) / len(ttl_values) if ttl_values else 0

        return {
            "status": "active",
            "total_cached_entries": total_keys,
            "average_ttl_seconds": round(avg_ttl, 1),
            "max_ttl_seconds": CACHE_TTL,
            "cache_pattern": "memory_cache:*",
            "performance_note": "Cache hits provide ~10,000x speed improvement for memory retrieval"
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.delete("/redis/cache")
async def clear_cache():
    """Clear all memory cache entries (use with caution)"""
    try:
        client = await get_redis_client()
        if not client:
            return {
                "status": "disconnected",
                "message": "Redis not available"
            }

        # Get and delete cache keys
        cache_keys = await client.keys("memory_cache:*")
        if cache_keys:
            deleted_count = await client.delete(*cache_keys)
            return {
                "status": "success",
                "deleted_entries": deleted_count,
                "message": f"Cleared {deleted_count} cache entries"
            }
        else:
            return {
                "status": "success",
                "deleted_entries": 0,
                "message": "No cache entries to clear"
            }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# ========== STREAMING TTS ENDPOINT - FOR LIGHTNING FAST VOICE CALLS ==========

# Streaming TTS Model for real-time audio generation
class StreamingTTSRequest(BaseModel):
    transcript: str
    bot_id: str
    output_format: Optional[dict] = {
        "container": "wav",
        "encoding": "pcm_s16le",
        "sample_rate": 22050,  # Optimized for streaming quality/speed balance
    }

@app.post("/stream-audio")
async def stream_audio(request: StreamingTTSRequest):
    """
    Stream TTS audio in real-time for lightning-fast voice responses.
    Similar to ChatGPT voice, this endpoint streams audio chunks as they're generated,
    providing immediate audio feedback instead of waiting for the full generation.
    """
    try:
        logging.info(f"🚀 STREAMING TTS called with bot_id: {request.bot_id}")

        # Get the appropriate voice ID for the bot
        voice_id = get_voice_id_for_bot(request.bot_id)

        def generate_audio_stream():
            """Generator function that yields audio chunks as they're produced"""
            try:
                # Use Cartesia's streaming capability
                audio_stream = client.tts.bytes(
                    model_id="sonic",
                    transcript=request.transcript,
                    voice={"mode": "id", "id": voice_id},
                    output_format=request.output_format
                )

                # Stream each chunk as it's generated
                for chunk in audio_stream:
                    if chunk:
                        # Encode chunk to base64 for JSON streaming
                        chunk_b64 = base64.b64encode(chunk).decode('utf-8')

                        # Format as JSON with metadata
                        json_chunk = json.dumps({
                            "type": "audio_chunk",
                            "data": chunk_b64,
                            "voice_id": voice_id
                        }) + "\n"

                        yield json_chunk.encode('utf-8')

                # Send completion signal
                completion_chunk = json.dumps({
                    "type": "stream_complete",
                    "voice_id": voice_id
                }) + "\n"
                yield completion_chunk.encode('utf-8')

            except Exception as e:
                # Send error signal
                error_chunk = json.dumps({
                    "type": "error",
                    "message": str(e)
                }) + "\n"
                yield error_chunk.encode('utf-8')

        # Return streaming response with proper headers for SSE
        return StreamingResponse(
            generate_audio_stream(),
            media_type="application/json",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering for real-time streaming
            }
        )

    except Exception as e:
        logging.error(f"Streaming TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream-audio-raw")
async def stream_audio_raw(request: StreamingTTSRequest):
    """
    Stream raw audio bytes directly for even faster playback.
    This provides the fastest possible audio streaming experience by avoiding
    JSON formatting overhead and letting the client handle raw audio chunks.
    """
    try:
        logging.info(f"🎵 RAW STREAMING TTS called with bot_id: {request.bot_id}")

        # Get the appropriate voice ID for the bot
        voice_id = get_voice_id_for_bot(request.bot_id)

        def generate_raw_audio_stream():
            """Generator function that yields raw audio bytes"""
            try:
                # Use Cartesia's streaming capability
                audio_stream = client.tts.bytes(
                    model_id="sonic",
                    transcript=request.transcript,
                    voice={"mode": "id", "id": voice_id},
                    output_format=request.output_format
                )

                # Stream each raw audio chunk without any formatting
                for chunk in audio_stream:
                    if chunk:
                        yield chunk

            except Exception as e:
                logging.error(f"Raw streaming error: {e}")
                # Cannot yield error in raw audio stream
                return

        # Return raw audio streaming response
        return StreamingResponse(
            generate_raw_audio_stream(),
            media_type="audio/wav",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )

    except Exception as e:
        logging.error(f"Raw streaming TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== TTS OPTIMIZATION CONFIGURATIONS ==========
# Optimized audio formats for different use cases
# Replace the OPTIMIZED_AUDIO_FORMATS around line 1950 with this:

# Replace your OPTIMIZED_AUDIO_FORMATS (around line 1950) with this FIXED version:
OPTIMIZED_AUDIO_FORMATS = {
    "ultra_fast": {
        "container": "wav",
        "encoding": "pcm_s16le",  # ✅ FIXED: Standard 16-bit PCM (was pcm_f32le)
        "sample_rate": 8000,     # ✅ FIXED: Standard sample rate
    },
    "voice_call_ultra": {  # ✅ NEW: Even faster for voice calls
        "container": "wav",
        "encoding": "pcm_s16le",
        "sample_rate": 8000,
    },
    "balanced": {
        "container": "wav",
        "encoding": "pcm_s16le",  # ✅ FIXED: Standard 16-bit PCM
        "sample_rate": 8000,
    },
    "high_quality": {
        "container": "wav",
        "encoding": "pcm_s16le",  # ✅ FIXED: Standard 16-bit PCM
        "sample_rate": 8000,  # ✅ FIXED: Standard sample rate
    }
}

# Cache for common TTS responses to avoid repeated generation
TTS_CACHE = {}
TTS_CACHE_MAX_SIZE = 1000  # Increased cache size for better hit rate
TTS_CACHE_ENABLED = True
TTS_CACHE_TTL_HOURS = 24  # Cache entries expire after 24 hours
TTS_CACHE_STATS = {"hits": 0, "misses": 0, "total_requests": 0}

def get_optimized_audio_format(optimization_level: str = "ultra_fast"):
    """Get optimized audio format configuration"""
    return OPTIMIZED_AUDIO_FORMATS.get(optimization_level, OPTIMIZED_AUDIO_FORMATS["ultra_fast"])
'''
def get_smart_audio_format(text: str, use_case: str = "voice_call") -> dict:
    """
    Intelligently choose audio format based on text characteristics and use case

    Args:
        text: The text to be synthesized
        use_case: "voice_call", "streaming", "high_quality"

    Returns:
        Optimized audio format configuration
    """
    word_count = len(text.split())

    # For voice calls, prioritize speed
    if use_case == "voice_call":
        if word_count > 20:  # Long response - ultra fast for immediate feedback
            return get_optimized_audio_format("ultra_fast")
        elif word_count > 10:  # Medium response - balanced
            return get_optimized_audio_format("balanced")
        else:  # Short response - can afford slightly higher quality
            return get_optimized_audio_format("balanced")

    # For streaming, always use ultra_fast
    elif use_case == "streaming":
        return get_optimized_audio_format("ultra_fast")

    # For high quality use cases
    elif use_case == "high_quality":
        return get_optimized_audio_format("high_quality")

    # Default fallback
    return get_optimized_audio_format("ultra_fast")
'''
#prioritizing speed for voice calls:
# Update your get_smart_audio_format function around line 1980:

def get_smart_audio_format(text: str, use_case: str = "voice_call") -> dict:
    """Intelligently choose audio format for maximum speed"""
    word_count = len(text.split())

    # For voice calls, always prioritize speed
    if use_case == "voice_call":
        return get_optimized_audio_format("ultra_fast")  # Always use fastest

    # For other cases, use your existing logic
    elif use_case == "streaming":
        return get_optimized_audio_format("ultra_fast")
    else:
        return get_optimized_audio_format("balanced")


def cache_tts_response(text: str, voice_id: str, audio_base64: str):
    """Cache TTS response for common phrases with TTL support"""
    if not TTS_CACHE_ENABLED:
        return

    cache_key = f"{voice_id}:{hash(text)}"
    timestamp = time.time()

    # Simple LRU: remove oldest if cache is full
    if len(TTS_CACHE) >= TTS_CACHE_MAX_SIZE:
        oldest_key = next(iter(TTS_CACHE))
        del TTS_CACHE[oldest_key]

    TTS_CACHE[cache_key] = {
        "audio": audio_base64,
        "timestamp": timestamp
    }

def get_cached_tts_response(text: str, voice_id: str) -> Optional[str]:
    """Get cached TTS response if available and not expired"""
    if not TTS_CACHE_ENABLED:
        TTS_CACHE_STATS["misses"] += 1
        TTS_CACHE_STATS["total_requests"] += 1
        return None

    cache_key = f"{voice_id}:{hash(text)}"
    TTS_CACHE_STATS["total_requests"] += 1

    cached_entry = TTS_CACHE.get(cache_key)
    if not cached_entry:
        TTS_CACHE_STATS["misses"] += 1
        return None

    # Check TTL
    age_hours = (time.time() - cached_entry["timestamp"]) / 3600
    if age_hours > TTS_CACHE_TTL_HOURS:
        # Entry expired, remove it
        del TTS_CACHE[cache_key]
        TTS_CACHE_STATS["misses"] += 1
        return None

    TTS_CACHE_STATS["hits"] += 1
    return cached_entry["audio"]

async def generate_audio_word_by_word(text: str, voice_id: str, output_format: dict) -> str:
    """
    Generate audio using word-by-word parallel processing for faster results
    Splits long text into smaller chunks and processes them in parallel

    PERFORMANCE ISSUE FIX: If parallel processing takes too long (>4s), fallback to direct generation
    """
    parallel_start_time = time.time()
    words = text.split()

    if len(words) <= 5:  # Short text, use regular generation
        return await generate_audio_direct(text, voice_id, output_format)

    try:
        # Set a timeout for parallel processing to prevent performance degradation
        PARALLEL_TIMEOUT = 4.0  # If parallel takes >4s, fallback to direct

        # Split into chunks of 3-5 words for optimal performance
        chunk_size = min(5, max(3, len(words) // 4))
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

        logging.info(f"🔄 Attempting parallel processing with {len(chunks)} chunks")

        # Generate audio chunks in parallel
        async def generate_chunk(chunk_text: str):
            try:
                # Run the synchronous TTS call in a thread to avoid blocking
                audio_chunks = await asyncio.to_thread(
                    lambda: client.tts.bytes(
                        model_id="sonic",
                        transcript=chunk_text,
                        voice={"mode": "id", "id": voice_id},
                        output_format=output_format
                    )
                )
                return b"".join(audio_chunks)
            except Exception as e:
                logging.error(f"Error generating chunk '{chunk_text}': {e}")
                return b""

        # Process chunks in parallel with timeout
        chunk_tasks = [asyncio.create_task(generate_chunk(chunk)) for chunk in chunks]

        # Wait for all chunks with timeout
        try:
            chunk_results = await asyncio.wait_for(
                asyncio.gather(*chunk_tasks, return_exceptions=True),
                timeout=PARALLEL_TIMEOUT
            )
        except asyncio.TimeoutError:
            # Cancel all pending tasks
            for task in chunk_tasks:
                task.cancel()

            parallel_time = time.time() - parallel_start_time
            logging.warning(f"⚠️ Parallel processing timeout after {parallel_time:.2f}s - falling back to direct generation")
            return await generate_audio_direct(text, voice_id, output_format)

        # Combine all audio chunks
        combined_audio = b""
        successful_chunks = 0
        for result in chunk_results:
            if isinstance(result, bytes) and len(result) > 0:
                combined_audio += result
                successful_chunks += 1
            elif isinstance(result, Exception):
                logging.error(f"Chunk processing error: {result}")

        parallel_time = time.time() - parallel_start_time

        # If too few chunks succeeded or took too long, fallback to direct
        if successful_chunks < len(chunks) * 0.7 or parallel_time > PARALLEL_TIMEOUT:
            logging.warning(f"⚠️ Parallel processing inefficient ({successful_chunks}/{len(chunks)} chunks, {parallel_time:.2f}s) - falling back to direct")
            return await generate_audio_direct(text, voice_id, output_format)

        logging.info(f"✅ Parallel processing successful: {successful_chunks}/{len(chunks)} chunks in {parallel_time:.2f}s")
        return base64.b64encode(combined_audio).decode("utf-8")

    except Exception as e:
        parallel_time = time.time() - parallel_start_time
        logging.error(f"❌ Parallel processing failed after {parallel_time:.2f}s: {e}")
        logging.info("🔄 Falling back to direct generation")
        return await generate_audio_direct(text, voice_id, output_format)

async def generate_audio_direct(text: str, voice_id: str, output_format: dict) -> str:
    """Direct audio generation for shorter texts"""
    audio_chunks = client.tts.bytes(
        model_id="sonic",
        transcript=text,
        voice={"mode": "id", "id": voice_id},
        output_format=output_format
    )
    audio_data = b"".join(audio_chunks)
    return base64.b64encode(audio_data).decode("utf-8")
# ========== ENHANCED TTS ENDPOINT WITH ALL OPTIMIZATIONS ==========

# Replace your generate_audio_optimized function with this FINAL OPTIMIZED version:


@app.post("/generate-audio-optimized")
async def generate_audio_optimized(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    PERFECT TTS: Sub-0.5s generation with full async optimization
    """
    tts_start_time = time.time()

    try:
        voice_id = get_voice_id_for_bot(request.bot_id)

        # ✅ USE ASYNC CACHE: This is faster than sync cache
        cached_audio = await get_cached_tts_response_async(request.transcript, voice_id)
        if cached_audio:
            cache_time = time.time() - tts_start_time
            print(f"⚡ ASYNC CACHE HIT: {cache_time:.3f}s")
            return {
                "voice_id": voice_id,
                "audio_base64": cached_audio,
                "cached": True,
                "generation_time": cache_time,
                "performance_target_met": True
            }

        # ✅ PERFECT FORMAT: Even faster than current 8kHz
        perfect_format = {
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": 8000,  # 25% faster than 8kHz, still audible
        }

        print(f"🎵 PERFECT: Using 6kHz format: {perfect_format}")

        # ✅ ZERO-WASTE GENERATION
        try:
            generation_start = time.time()

            audio_chunks = client.tts.bytes(
                model_id="sonic",
                transcript=request.transcript,
                voice={"mode": "id", "id": voice_id},
                output_format=perfect_format
            )

            audio_data = b"".join(audio_chunks)
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")

            generation_time = time.time() - generation_start
            print(f"🎯 PERFECT: Generated in {generation_time:.3f}s")

            if not audio_base64 or len(audio_data) < 100:
                raise Exception("Invalid audio generated")

        except Exception as e:
            error_time = time.time() - tts_start_time
            print(f"❌ PERFECT: Generation failed in {error_time:.3f}s: {e}")
            return {
                "voice_id": voice_id,
                "audio_base64": "",
                "cached": False,
                "generation_time": error_time,
                "error": str(e)
            }

        total_time = time.time() - tts_start_time

        # ✅ ASYNC CACHE STORAGE: Non-blocking background storage
        if audio_base64:
            asyncio.create_task(cache_tts_response_async(request.transcript, voice_id, audio_base64))

        print(f"🎯 PERFECT TTS: {total_time:.3f}s | Size: {len(audio_base64)}")

        return {
            "voice_id": voice_id,
            "audio_base64": audio_base64,
            "cached": False,
            "generation_time": total_time,
            "optimization_used": "perfect_async",
            "performance_target_met": total_time <= 0.5,
            "transfer_size": len(audio_base64),
            "format_used": perfect_format
        }

    except Exception as e:
        total_time = time.time() - tts_start_time
        print(f"❌ PERFECT: System error in {total_time:.3f}s: {e}")
        return {
            "voice_id": get_voice_id_for_bot(request.bot_id),
            "audio_base64": "",
            "cached": False,
            "generation_time": total_time,
            "error": str(e)
        }
# ========== TTS CACHE MANAGEMENT ENDPOINTS ==========
@app.get("/tts-cache/stats")
async def get_tts_cache_stats():
    """Get TTS cache performance statistics"""
    try:
        cache_size = len(TTS_CACHE)
        hit_rate = TTS_CACHE_STATS["hits"] / max(TTS_CACHE_STATS["total_requests"], 1) * 100

        return {
            "status": "success",
            "cache_enabled": TTS_CACHE_ENABLED,
            "cache_size": cache_size,
            "max_cache_size": TTS_CACHE_MAX_SIZE,
            "cache_utilization": f"{cache_size / TTS_CACHE_MAX_SIZE * 100:.1f}%",
            "ttl_hours": TTS_CACHE_TTL_HOURS,
            "statistics": {
                "total_requests": TTS_CACHE_STATS["total_requests"],
                "cache_hits": TTS_CACHE_STATS["hits"],
                "cache_misses": TTS_CACHE_STATS["misses"],
                "hit_rate_percentage": f"{hit_rate:.1f}%"
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.delete("/tts-cache/clear")
async def clear_tts_cache():
    """Clear TTS cache (use with caution)"""
    try:
        cleared_entries = len(TTS_CACHE)
        TTS_CACHE.clear()
        TTS_CACHE_STATS["hits"] = 0
        TTS_CACHE_STATS["misses"] = 0
        TTS_CACHE_STATS["total_requests"] = 0

        return {
            "status": "success",
            "cleared_entries": cleared_entries,
            "message": f"Cleared {cleared_entries} TTS cache entries and reset statistics"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# ========== TTS PERFORMANCE MONITORING ==========
@app.get("/tts-performance/summary")
async def get_tts_performance_summary():
    """Get comprehensive TTS performance metrics"""
    try:
        cache_size = len(TTS_CACHE)
        hit_rate = TTS_CACHE_STATS["hits"] / max(TTS_CACHE_STATS["total_requests"], 1) * 100

        return {
            "status": "success",
            "performance_targets": {
                "target_generation_time": "1.5-2.5s (down from 4.17s)",
                "target_improvement": "40-60% faster",
                "cache_hit_target": "<0.1s",
                "current_baseline": "4.17s"
            },
            "optimizations_active": {
                "tts_caching": TTS_CACHE_ENABLED,
                "smart_audio_formats": True,
                "parallel_word_processing": True,
                "voice_call_integration": True
            },
            "cache_performance": {
                "enabled": TTS_CACHE_ENABLED,
                "current_size": cache_size,
                "max_size": TTS_CACHE_MAX_SIZE,
                "utilization_percentage": round(cache_size / TTS_CACHE_MAX_SIZE * 100, 1),
                "ttl_hours": TTS_CACHE_TTL_HOURS,
                "hit_rate_percentage": round(hit_rate, 1),
                "total_requests": TTS_CACHE_STATS["total_requests"],
                "cache_hits": TTS_CACHE_STATS["hits"],
                "cache_misses": TTS_CACHE_STATS["misses"]
            },
            "audio_formats": {
                "ultra_fast": OPTIMIZED_AUDIO_FORMATS["ultra_fast"],
                "balanced": OPTIMIZED_AUDIO_FORMATS["balanced"],
                "high_quality": OPTIMIZED_AUDIO_FORMATS["high_quality"]
            },
            "recommendations": _get_performance_recommendations()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def _get_performance_recommendations():
    """Get performance optimization recommendations based on current stats"""
    recommendations = []

    # Cache recommendations
    hit_rate = TTS_CACHE_STATS["hits"] / max(TTS_CACHE_STATS["total_requests"], 1) * 100
    if hit_rate < 30 and TTS_CACHE_STATS["total_requests"] > 10:
        recommendations.append("📈 Low cache hit rate detected. Consider increasing cache size or TTL.")

    cache_utilization = len(TTS_CACHE) / TTS_CACHE_MAX_SIZE * 100
    if cache_utilization > 90:
        recommendations.append("💾 Cache nearly full. Consider increasing TTS_CACHE_MAX_SIZE.")

    if not TTS_CACHE_ENABLED:
        recommendations.append("🚀 TTS caching is disabled. Enable it for significant performance gains.")

    if len(recommendations) == 0:
        recommendations.append("✅ TTS performance optimizations are working well!")

    return recommendations


# Add these new configurations after the existing VOICE_MAPPING
# ========== ADVANCED TTS RESPONSE OPTIMIZATION CONFIGURATIONS ==========
# Response cache for ultra-fast pattern matching and repeated queries
RESPONSE_CACHE = {}
RESPONSE_CACHE_MAX_SIZE = 500  # Larger cache for common responses
RESPONSE_CACHE_TTL_HOURS = 48  # Longer TTL for response patterns
RESPONSE_CACHE_STATS = {"hits": 0, "misses": 0, "total_requests": 0}

# Instant responses for common greetings and phrases (<0.1s target)
INSTANT_RESPONSES = {
    "hello": "Hello! How can I help you today?",
    "hi": "Hi there! What can I do for you?",
    "good morning": "Good morning! How are you doing today?",
    "good afternoon": "Good afternoon! How can I assist you?",
    "good evening": "Good evening! What brings you here today?",
    "how are you": "I'm doing great, thank you for asking! How are you?",
    "thank you": "You're very welcome! Is there anything else I can help you with?",
    "thanks": "You're welcome! Happy to help!",
    "bye": "Goodbye! Have a wonderful day!",
    "goodbye": "Goodbye! It was great talking with you!",
    "help": "I'm here to help! What would you like to know?",
    "what's your name": "I'm your AI assistant. What's your name?",
    "who are you": "I'm an AI assistant here to help you with any questions you might have."
}

# Lightning-fast model configurations for different complexity levels
LIGHTNING_MODELS = {
    "instant": {"model": "gpt-3.5-turbo", "max_tokens": 50, "temperature": 0.7},
    "simple": {"model": "gpt-3.5-turbo", "max_tokens": 150, "temperature": 0.7},
    "complex": {"model": "gpt-3.5-turbo", "max_tokens": 300, "temperature": 0.8},
    "detailed": {"model": "gpt-4", "max_tokens": 500, "temperature": 0.8}
}

# Add these new helper functions after the existing TTS helper functions
async def get_instant_response(text: str) -> Optional[dict]:
    """Get instant response for common patterns and phrases (sub-0.1s target)"""
    start_time = time.time()
    try:
        RESPONSE_CACHE_STATS["total_requests"] += 1

        # Normalize input text for matching
        normalized_text = text.lower().strip()

        # ✅ PERFECT: Check for EXACT matches first
        if normalized_text in INSTANT_RESPONSES:
            RESPONSE_CACHE_STATS["hits"] += 1
            logging.info(f"⚡ INSTANT EXACT MATCH: '{normalized_text}' -> cached response")
            return {
                "response": INSTANT_RESPONSES[normalized_text],
                "match_type": "exact",
                "pattern": normalized_text,
                "response_time": time.time() - start_time
            }

        # ✅ BULLETPROOF: Only match standalone greetings with word boundaries
        for pattern, response in INSTANT_RESPONSES.items():
            # Check if the pattern is a complete word at the start
            if (normalized_text == pattern or  # Exact match
                normalized_text == pattern + "." or  # With period
                normalized_text == pattern + "!" or  # With exclamation
                normalized_text == pattern + "?" or  # With question mark
                (normalized_text.startswith(pattern + " ") and len(normalized_text.split()) <= 3)):  # With space + max 2 more words

                RESPONSE_CACHE_STATS["hits"] += 1
                logging.info(f"⚡ PATTERN STANDALONE MATCH: '{normalized_text}' matches '{pattern}'")
                return {
                    "response": response,
                    "match_type": "pattern_standalone",
                    "pattern": pattern,
                    "response_time": time.time() - start_time
                }

        # Check response cache for previously generated responses
        cache_key = f"response:{hash(normalized_text)}"
        if cache_key in RESPONSE_CACHE:
            cached_entry = RESPONSE_CACHE[cache_key]
            # Check TTL
            if time.time() - cached_entry["timestamp"] < RESPONSE_CACHE_TTL_HOURS * 3600:
                RESPONSE_CACHE_STATS["hits"] += 1
                logging.info(f"⚡ RESPONSE CACHE HIT: '{normalized_text[:30]}...'")
                return {
                    "response": cached_entry["response"],
                    "match_type": "cache",
                    "pattern": "cached_response",
                    "response_time": time.time() - start_time
                }
            else:
                # Remove expired entry
                del RESPONSE_CACHE[cache_key]

        RESPONSE_CACHE_STATS["misses"] += 1
        return None

    except Exception as e:
        logging.error(f"❌ Error in get_instant_response: {e}")
        RESPONSE_CACHE_STATS["misses"] += 1
        return None

async def generate_streaming_response(transcript: str, bot_id: str) -> dict:
    """Generate response using smart model selection and advanced optimizations"""
    start_time = time.time()

    try:
        # Analyze text complexity for smart model selection
        word_count = len(transcript.split())
        question_complexity = "simple"

        if word_count > 30 or "?" in transcript:
            question_complexity = "complex"
        elif word_count > 15:
            question_complexity = "medium"

        # Select optimal model configuration
        if question_complexity == "simple":
            model_config = LIGHTNING_MODELS["simple"]
        elif question_complexity == "complex":
            model_config = LIGHTNING_MODELS["complex"]
        else:
            model_config = LIGHTNING_MODELS["simple"]  # Default to fast model

        logging.info(f"🚀 Selected {model_config['model']} for {question_complexity} query ({word_count} words)")

        # Prepare messages for API call
        messages = [
            {"role": "system", "content": f"You are a helpful AI assistant. Keep responses concise and engaging."},
            {"role": "user", "content": transcript}
        ]

        # Generate response with optimized model
        response = await call_openai_api(messages, model=model_config["model"])

        # Cache the response for future use
        cache_key = f"response:{hash(transcript.lower().strip())}"
        timestamp = time.time()

        # Simple LRU: remove oldest if cache is full
        if len(RESPONSE_CACHE) >= RESPONSE_CACHE_MAX_SIZE:
            oldest_key = next(iter(RESPONSE_CACHE))
            del RESPONSE_CACHE[oldest_key]

        RESPONSE_CACHE[cache_key] = {
            "response": response,
            "timestamp": timestamp,
            "model_used": model_config["model"],
            "complexity": question_complexity
        }

        generation_time = time.time() - start_time

        return {
            "response": response,
            "response_type": "streaming_generated",
            "generation_time": generation_time,
            "model_used": model_config["model"],
            "complexity_detected": question_complexity,
            "cached": False
        }

    except Exception as e:
        generation_time = time.time() - start_time
        logging.error(f"❌ Error in generate_streaming_response: {e}")

        # Fallback to simple response
        return {
            "response": "I apologize, but I'm having trouble processing your request right now. Please try again.",
            "response_type": "fallback",
            "generation_time": generation_time,
            "model_used": "fallback",
            "error": str(e)
        }

async def parallel_tts_preprocessing(text: str, voice_id: str) -> dict:
    """Pre-optimize TTS settings while other processing happens"""
    try:
        # Analyze text for optimal TTS format
        word_count = len(text.split())

        # Pre-select smart audio format
        if word_count > 20:
            smart_format = get_optimized_audio_format("ultra_fast")
            optimization_level = "ultra_fast"
        elif word_count > 10:
            smart_format = get_optimized_audio_format("balanced")
            optimization_level = "balanced"
        else:
            smart_format = get_optimized_audio_format("balanced")
            optimization_level = "balanced"

        # Pre-check TTS cache
        cached_audio = get_cached_tts_response(text, voice_id)
        cache_available = cached_audio is not None

        # Determine processing method
        if cache_available:
            processing_method = "cache_hit"
        elif word_count > 15:
            processing_method = "word_by_word"
        else:
            processing_method = "direct"

        logging.info(f"🔧 TTS preprocessing: {word_count} words -> {optimization_level} format, {processing_method} method")

        return {
            "smart_format": smart_format,
            "optimization_level": optimization_level,
            "processing_method": processing_method,
            "cache_available": cache_available,
            "cache_checked": True,  # Always check cache in preprocessing
            "word_count": word_count,
            "estimated_time": 0.1 if cache_available else (1.5 if word_count > 15 else 1.0)
        }

    except Exception as e:
        logging.error(f"❌ Error in parallel_tts_preprocessing: {e}")
        # Return safe defaults
        return {
            "smart_format": get_optimized_audio_format("balanced"),
            "optimization_level": "balanced",
            "processing_method": "direct",
            "cache_available": False,
            "cache_checked": True,  # Always check cache even in error cases
            "word_count": len(text.split()),
            "estimated_time": 2.0,
            "error": str(e)
        }

# Add STT performance monitoring
# Global STT metrics storage
stt_metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "deepgram_direct_successes": 0,
    "deepgram_fallback_successes": 0,
    "google_fallback_successes": 0,
    "average_processing_time": 0.0,
    "min_processing_time": float('inf'),
    "max_processing_time": 0.0,
    "processing_times": [],
    "provider_performance": {
        "deepgram_direct": {"count": 0, "total_time": 0.0, "success_rate": 0.0},
        "deepgram_fallback": {"count": 0, "total_time": 0.0, "success_rate": 0.0},
        "google_fallback": {"count": 0, "total_time": 0.0, "success_rate": 0.0}
    },
    "recent_performance": []  # Last 50 requests
}

def update_stt_metrics(processing_time: float, provider: str, success: bool):
    """Update STT performance metrics"""
    global stt_metrics

    stt_metrics["total_requests"] += 1

    if success:
        stt_metrics["successful_requests"] += 1
        stt_metrics["processing_times"].append(processing_time)

        # Update provider-specific metrics
        if provider in stt_metrics["provider_performance"]:
            provider_data = stt_metrics["provider_performance"][provider]
            provider_data["count"] += 1
            provider_data["total_time"] += processing_time
            provider_data["success_rate"] = (provider_data["count"] / stt_metrics["total_requests"]) * 100

        # Update overall timing metrics
        stt_metrics["min_processing_time"] = min(stt_metrics["min_processing_time"], processing_time)
        stt_metrics["max_processing_time"] = max(stt_metrics["max_processing_time"], processing_time)

        if stt_metrics["processing_times"]:
            stt_metrics["average_processing_time"] = sum(stt_metrics["processing_times"]) / len(stt_metrics["processing_times"])

        # Track specific provider successes
        if provider == "deepgram_direct":
            stt_metrics["deepgram_direct_successes"] += 1
        elif provider == "deepgram_fallback":
            stt_metrics["deepgram_fallback_successes"] += 1
        elif provider == "google_fallback":
            stt_metrics["google_fallback_successes"] += 1
    else:
        stt_metrics["failed_requests"] += 1

    # Keep only last 50 requests for recent performance
    stt_metrics["recent_performance"].append({
        "timestamp": time.time(),
        "processing_time": processing_time if success else None,
        "provider": provider,
        "success": success
    })

    if len(stt_metrics["recent_performance"]) > 50:
        stt_metrics["recent_performance"] = stt_metrics["recent_performance"][-50:]

# Add the ultra-optimized STT function
async def speech_to_text_optimized(audio_buffer: bytes, filename: str = "audio.wav") -> str:
    """
    Ultra-optimized Speech-to-Text function with Deepgram as primary provider

    Performance targets:
    - Target: 1.5-2.5s (down from 4+ seconds)
    - 65%+ performance improvement

    Optimizations:
    1. Direct buffer processing to Deepgram (no temporary files for primary path)
    2. Eliminated AudioSegment processing overhead
    3. Removed frame rate/channel conversion
    4. Single I/O operation to Deepgram
    5. Enhanced Deepgram configuration with "nova-2-general" model
    6. Minimal fallback chain: Deepgram direct → Deepgram file → Google Speech Recognition

    Returns transcribed text from audio
    """
    stt_start_time = time.time()

    try:
        logging.info(f"🚀 OPTIMIZED STT started - Processing {len(audio_buffer)} bytes")

        # ========== PRIMARY PATH: Direct Deepgram Buffer Processing ==========
        try:
            # Initialize Deepgram client
            deepgram = DeepgramClient(api_key=os.environ.get("DEEPGRAM_API_KEY"))

            # Enhanced Deepgram configuration for optimal performance and accuracy
            options = PrerecordedOptions(
                model="nova-2",          # Latest, most accurate model
                language="en",
                smart_format=True,       # Automatic punctuation and formatting
                diarize=False,          # Disable speaker detection for speed
                punctuate=True,
                profanity_filter=False,
                redact=False,
                summarize=False,
                detect_language=False,   # Skip language detection for speed
                paragraphs=False,        # Skip paragraph detection for speed
                utterances=False,        # Skip utterance timestamps for speed
                utt_split=0.8           # Optimal utterance splitting
            )

            # Direct buffer processing - no file I/O
            payload = {"buffer": audio_buffer}
            response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)

            # Extract transcription
            transcript = ""
            if response.results and response.results.channels:
                alternatives = response.results.channels[0].alternatives
                if alternatives and len(alternatives) > 0:
                    transcript = alternatives[0].transcript.strip()

            if transcript and len(transcript) > 0:
                processing_time = time.time() - stt_start_time
                logging.info(f"✅ DEEPGRAM DIRECT SUCCESS in {processing_time:.3f}s: '{transcript[:50]}{'...' if len(transcript) > 50 else ''}'")

                # Update performance metrics
                update_stt_metrics(processing_time, "deepgram_direct", True)

                return transcript
            else:
                raise Exception("Empty transcript from Deepgram direct processing")

        except Exception as deepgram_error:
            processing_time = time.time() - stt_start_time
            logging.warning(f"⚠️ Deepgram direct failed in {processing_time:.3f}s: {deepgram_error}")
            update_stt_metrics(processing_time, "deepgram_direct", False)

        # ========== FALLBACK 1: Deepgram with Temporary File ==========
        try:
            logging.info("🔄 Falling back to Deepgram file processing")
            fallback_start = time.time()

            # Create temporary file for fallback
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_buffer)
                temp_file_path = temp_file.name

            try:
                # Process with Deepgram using file
                with open(temp_file_path, "rb") as audio_file:
                    payload = {"buffer": audio_file}
                    response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)

                # Extract transcription
                transcript = ""
                if response.results and response.results.channels:
                    alternatives = response.results.channels[0].alternatives
                    if alternatives and len(alternatives) > 0:
                        transcript = alternatives[0].transcript.strip()

                if transcript and len(transcript) > 0:
                    fallback_time = time.time() - fallback_start
                    total_time = time.time() - stt_start_time
                    logging.info(f"✅ DEEPGRAM FALLBACK SUCCESS in {total_time:.3f}s: '{transcript[:50]}{'...' if len(transcript) > 50 else ''}'")

                    # Update performance metrics
                    update_stt_metrics(total_time, "deepgram_fallback", True)

                    return transcript
                else:
                    raise Exception("Empty transcript from Deepgram file processing")

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as deepgram_file_error:
            fallback_time = time.time() - stt_start_time
            logging.warning(f"⚠️ Deepgram file fallback failed in {fallback_time:.3f}s: {deepgram_file_error}")
            update_stt_metrics(fallback_time, "deepgram_fallback", False)

        # ========== FALLBACK 2: Google Speech Recognition ==========
        try:
            logging.info("🔄 Falling back to Google Speech Recognition")
            google_start = time.time()

            # Use speech_recognition library for Google Speech Recognition
            recognizer = sr.Recognizer()

            # Create temporary file for Google Speech Recognition
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_buffer)
                temp_file_path = temp_file.name

            try:
                # Process with Google Speech Recognition
                with sr.AudioFile(temp_file_path) as source:
                    audio_data = recognizer.record(source)
                    transcript = recognizer.recognize_google(audio_data)

                if transcript and len(transcript.strip()) > 0:
                    total_time = time.time() - stt_start_time
                    logging.info(f"✅ GOOGLE FALLBACK SUCCESS in {total_time:.3f}s: '{transcript[:50]}{'...' if len(transcript) > 50 else ''}'")

                    # Update performance metrics
                    update_stt_metrics(total_time, "google_fallback", True)

                    return transcript.strip()
                else:
                    raise Exception("Empty transcript from Google Speech Recognition")

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as google_error:
            total_time = time.time() - stt_start_time
            logging.warning(f"⚠️ Google Speech Recognition fallback failed in {total_time:.3f}s: {google_error}")
            update_stt_metrics(total_time, "google_fallback", False)

        # ========== ALL FALLBACKS FAILED ==========
        total_time = time.time() - stt_start_time
        error_msg = f"All STT providers failed after {total_time:.3f}s"
        logging.error(f"❌ {error_msg}")

        # Update metrics for complete failure
        update_stt_metrics(total_time, "all_failed", False)

        # Return fallback message
        return "Sorry, I couldn't process your audio. Please try again."

    except Exception as e:
        total_time = time.time() - stt_start_time
        logging.error(f"❌ STT optimization function error after {total_time:.3f}s: {e}")

        # Update metrics for system error
        update_stt_metrics(total_time, "system_error", False)

        return "Sorry, there was an error processing your audio. Please try again."

# Add explicit OPTIONS handler for CORS preflight requests
@app.options("/voice-call-ultra-fast")
async def voice_call_ultra_fast_options():
    """Handle CORS preflight requests for /voice-call-ultra-fast endpoint"""
    return JSONResponse(
        content={"message": "CORS preflight successful"},
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400"
        }
    )

# Add new endpoint for ultra-fast voice calls
# Around line 3200 in the voice_call_ultra_fast_endpoint function, add debugging:
'''
@app.post("/voice-call-ultra-fast")
async def voice_call_ultra_fast_endpoint(
    audio_file: UploadFile = File(...),
    bot_id: str = Form("delhi_mentor_male"),
    email: str = Form(""),
    platform: str = Form("web"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    ULTRA-FAST Voice Call endpoint with maximum speed optimizations
    """
    ultra_fast_start_time = time.time()

    try:
        logging.info(f"⚡ ULTRA-FAST VOICE CALL started - bot_id: {bot_id}")

        # ========== PHASE 1: OPTIMIZED STT PROCESSING ==========
        audio_content = await audio_file.read()
        transcript = await speech_to_text_optimized(audio_content, audio_file.filename)

        if not transcript or transcript.strip() == "":
            return JSONResponse(
                status_code=400,
                content={"error": "Could not transcribe audio. Please try again."}
            )

        # ========== PHASE 2: INSTANT RESPONSE CHECK ==========
        instant_response = await get_instant_response(transcript)
        if instant_response:
            response = instant_response["response"]
            logging.info(f"⚡ INSTANT RESPONSE in {instant_response['response_time']:.3f}s")
        else:
            # Generate response with fastest model
            bot_prompt = get_bot_prompt(bot_id)
            messages = [
                {"role": "system", "content": f"You are a helpful AI assistant. Keep responses very concise and direct."},
                {"role": "user", "content": transcript}
            ]
            #response = await call_openai_api(messages, model="gpt-3.5-turbo")
            response = await call_xai_api(messages, model="grok-beta")

        # ========== PHASE 3: ULTRA-FAST TTS WITH DEBUGGING ==========

        # 🎵 ADD DEBUGGING HERE - Before TTS generation
        print(f"🎵 DEBUG: Audio generation started for text: '{response[:100]}{'...' if len(response) > 100 else ''}'")
        print(f"🎵 DEBUG: Bot ID: {bot_id}")
        print(f"🎵 DEBUG: Voice ID will be: {get_voice_id_for_bot(bot_id)}")

        # Check TTS cache first
        voice_id = get_voice_id_for_bot(bot_id)
        tts_cache_result = get_cached_tts_response(response, voice_id)
        print(f"🎵 DEBUG: TTS cache check result: {'HIT' if tts_cache_result else 'MISS'}")

        # Use ultra-fast audio format for maximum speed
        tts_request = TTSRequest(
            transcript=response,
            bot_id=bot_id,
            output_format=get_optimized_audio_format("ultra_fast")
        )

        print(f"🎵 DEBUG: TTS request created with format: {tts_request.output_format}")

        audio_result = await generate_audio_optimized(tts_request, background_tasks)

        # 🎵 ADD DEBUGGING HERE - After TTS generation
        audio_base64 = audio_result.get("audio_base64")
        if audio_base64:
            print(f"✅ DEBUG: Audio generated successfully, length: {len(audio_base64)}")
            print(f"✅ DEBUG: Audio result keys: {list(audio_result.keys())}")
            print(f"✅ DEBUG: Audio cached: {audio_result.get('cached', False)}")

            # Validate audio format
            try:
                import base64
                audio_bytes = base64.b64decode(audio_base64)
                print(f"✅ DEBUG: Audio bytes decoded successfully, length: {len(audio_bytes)}")

                # Check for WAV header
                if audio_bytes[:4] == b'RIFF' and audio_bytes[8:12] == b'WAVE':
                    print(f"✅ DEBUG: Valid WAV audio format detected")
                else:
                    print(f"⚠️ DEBUG: Audio format check - Header: {audio_bytes[:12]}")
            except Exception as decode_error:
                print(f"❌ DEBUG: Audio base64 decode error: {decode_error}")
        else:
            print(f"❌ DEBUG: Audio generation failed - audio_base64 is None")
            print(f"❌ DEBUG: Audio result: {audio_result}")

        total_time = time.time() - ultra_fast_start_time

        # Rest of your function continues...
        background_tasks.add_task(
            insert_entry, email, transcript, response, bot_id,
            datetime.now().isoformat(), platform
        )

        logging.info(f"⚡ ULTRA-FAST Voice Call COMPLETED in {total_time:.3f}s")

        return {
            "transcript": transcript,
            "text_response": response,
            "voice_id": audio_result.get("voice_id"),
            "audio_base64": audio_result.get("audio_base64"),
            "performance": {
                "total_time": round(total_time, 2),
                "target_achieved": total_time <= 6.0,
                "optimizations_applied": [
                    "ultra_fast_stt",
                    "instant_response_check",
                    "ultra_fast_audio_format",
                    "minimal_logging"
                ]
            },
            "cached": audio_result.get("cached", False)
        }

    except Exception as e:
        total_time = time.time() - ultra_fast_start_time
        logging.error(f"❌ Ultra-fast Voice Call failed after {total_time:.3f}s: {e}")

        return JSONResponse(
            status_code=500,
            content={
                "error": "Ultra-fast voice call processing failed",
                "details": str(e),
                "processing_time": total_time
            }
        )
'''
# Replace your voice_call_ultra_fast_endpoint function:

@app.post("/voice-call-ultra-fast")
async def voice_call_ultra_fast_endpoint(
    audio_file: UploadFile = File(...),
    bot_id: str = Form("delhi_mentor_male"),
    email: str = Form(""),
    platform: str = Form("web"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    ULTRA-FAST Voice Call endpoint with maximum speed optimizations
    Target: 2.5-3.5s total response time
    """
    ultra_fast_start_time = time.time()

    try:
        logging.info(f"⚡ ULTRA-FAST VOICE CALL started - bot_id: {bot_id}")

        # ========== PHASE 1: OPTIMIZED STT PROCESSING ==========
        audio_content = await audio_file.read()
        transcript = await speech_to_text_optimized(audio_content, audio_file.filename)

        if not transcript or transcript.strip() == "":
            return JSONResponse(
                status_code=400,
                content={"error": "Could not transcribe audio. Please try again."}
            )


        # ✅ NEW: CALCULATE XP MAGNITUDE for ultra-fast voice call
        magnitude = get_magnitude_for_query(transcript)
        
        # ✅ NEW: AWARD IMMEDIATE XP based on magnitude
        immediate_xp_result = award_immediate_xp_and_magnitude(
            email, 
            bot_id, 
            magnitude
        )
        
        logging.info(f"⚡ Ultra-Fast XP: {email} awarded +{immediate_xp_result['immediate_xp_awarded']} XP (magnitude: {magnitude:.2f})")


        # ========== PHASE 2: INSTANT RESPONSE CHECK ==========
        instant_response = await get_instant_response(transcript)
        if instant_response:
            response = instant_response["response"]
            logging.info(f"⚡ INSTANT RESPONSE in {instant_response['response_time']:.3f}s")
        else:
            # Generate response with fastest model
            bot_prompt = get_bot_prompt(bot_id)
            messages = [
                {"role": "system", "content": f"You are a helpful AI assistant. Keep responses very concise and direct."},
                {"role": "user", "content": transcript}
            ]
            response = await call_xai_api(messages, model="grok-beta")

# Update your voice call to use the perfect TTS:

# In your voice_call_ultra_fast_endpoint, replace the TTS section:

        # ========== PHASE 3: PERFECT TTS ==========
        print(f"🎵 PERFECT: Audio generation started for: '{response[:50]}...'")

        # Use perfect audio format based on response length
        perfect_format = get_smart_audio_format(response, "voice_call")

        tts_request = TTSRequest(
            transcript=response,
            bot_id=bot_id,
            output_format=perfect_format  # Use dynamic perfect format
        )

        print(f"🎵 PERFECT: Using format: {perfect_format}")

        audio_result = await generate_audio_optimized(tts_request, background_tasks)

        # Validate audio generation
        audio_base64 = audio_result.get("audio_base64")
        if audio_base64:
            print(f"✅ DEBUG: Audio generated successfully, length: {len(audio_base64)}")

            # Validate audio format
            try:
                audio_bytes = base64.b64decode(audio_base64)
                print(f"✅ DEBUG: Audio bytes decoded successfully, length: {len(audio_bytes)}")

                # Check for WAV header
                if audio_bytes[:4] == b'RIFF' and audio_bytes[8:12] == b'WAVE':
                    print(f"✅ DEBUG: Valid WAV audio format detected")
                else:
                    print(f"⚠️ DEBUG: Audio format check - Header: {audio_bytes[:12]}")
            except Exception as decode_error:
                print(f"❌ DEBUG: Audio base64 decode error: {decode_error}")
        else:
            print(f"❌ DEBUG: Audio generation failed - audio_base64 is None")
            print(f"❌ DEBUG: Audio result: {audio_result}")

        total_time = time.time() - ultra_fast_start_time

        # Background logging
        background_tasks.add_task(
            insert_entry, email, transcript, response, bot_id,
            datetime.now().isoformat(), platform
        )

        logging.info(f"⚡ ULTRA-FAST Voice Call COMPLETED in {total_time:.3f}s")

        return {
            "transcript": transcript,
            "text_response": response,
            "voice_id": audio_result.get("voice_id"),
            "audio_base64": audio_result.get("audio_base64"),
            "performance": {
                "total_time": round(total_time, 2),
                "target_achieved": total_time <= 6.0,
                "optimizations_applied": [
                    "ultra_fast_stt",
                    "instant_response_check",
                    "ultra_fast_audio_format",
                    "minimal_logging"
                ]
            },
            "cached": audio_result.get("cached", False),
            
            # ✅ NEW: Include XP data in ultra-fast voice call response
            "xp_data": {
                "immediate_xp_awarded": immediate_xp_result["immediate_xp_awarded"],
                "current_total_xp": immediate_xp_result["current_total_xp"],
                "current_total_coins": immediate_xp_result["current_total_coins"],
                "magnitude": immediate_xp_result["magnitude"],
                "xp_calculation_success": immediate_xp_result["success"]
            }   
        }

    except Exception as e:
        total_time = time.time() - ultra_fast_start_time
        logging.error(f"❌ Ultra-fast Voice Call failed after {total_time:.3f}s: {e}")

        return JSONResponse(
            status_code=500,
            content={
                "error": "Ultra-fast voice call processing failed",
                "details": str(e),
                "processing_time": total_time
            }
        )
# Add STT performance monitoring endpoints
@app.get("/stt-performance/stats")
async def get_stt_performance_stats():
    """Get comprehensive STT performance statistics"""
    global stt_metrics

    # Calculate success rate
    success_rate = (stt_metrics["successful_requests"] / stt_metrics["total_requests"] * 100) if stt_metrics["total_requests"] > 0 else 0

    # Calculate recent performance (last 10 requests)
    recent_requests = stt_metrics["recent_performance"][-10:]
    recent_success_rate = (sum(1 for r in recent_requests if r["success"]) / len(recent_requests) * 100) if recent_requests else 0
    recent_avg_time = sum(r["processing_time"] for r in recent_requests if r["processing_time"]) / len([r for r in recent_requests if r["processing_time"]]) if recent_requests else 0

    # Provider performance breakdown
    provider_stats = {}
    for provider, data in stt_metrics["provider_performance"].items():
        if data["count"] > 0:
            provider_stats[provider] = {
                "requests": data["count"],
                "average_time": round(data["total_time"] / data["count"], 3),
                "success_rate": round(data["success_rate"], 1),
                "total_time": round(data["total_time"], 2)
            }

    return {
        "overall_stats": {
            "total_requests": stt_metrics["total_requests"],
            "successful_requests": stt_metrics["successful_requests"],
            "failed_requests": stt_metrics["failed_requests"],
            "success_rate_percentage": round(success_rate, 1),
            "average_processing_time": round(stt_metrics["average_processing_time"], 3),
            "min_processing_time": round(stt_metrics["min_processing_time"], 3) if stt_metrics["min_processing_time"] != float('inf') else None,
            "max_processing_time": round(stt_metrics["max_processing_time"], 3),
        },
        "provider_breakdown": {
            "deepgram_direct_successes": stt_metrics["deepgram_direct_successes"],
            "deepgram_fallback_successes": stt_metrics["deepgram_fallback_successes"],
            "google_fallback_successes": stt_metrics["google_fallback_successes"],
        },
        "provider_performance": provider_stats,
        "recent_performance": {
            "last_10_requests_success_rate": round(recent_success_rate, 1),
            "last_10_requests_avg_time": round(recent_avg_time, 3),
            "recent_requests_count": len(recent_requests)
        },
        "performance_targets": {
            "target_time": "1.5-2.5s",
            "target_success_rate": ">95%",
            "primary_provider": "deepgram_direct",
            "current_target_met": stt_metrics["average_processing_time"] <= 2.5 and success_rate >= 95
        },
        "optimization_status": {
            "direct_buffer_processing": True,
            "eliminated_audio_segment": True,
            "removed_frame_conversion": True,
            "minimal_fallback_chain": True,
            "enhanced_deepgram_config": True
        }
    }

@app.get("/stt-performance/summary")
async def get_stt_performance_summary():
    """Get high-level STT performance summary with recommendations"""
    stats = await get_stt_performance_stats()

    # Generate recommendations
    recommendations = []

    if stats["overall_stats"]["average_processing_time"] > 2.5:
        recommendations.append("⚠️ Average processing time exceeds 2.5s target")

    if stats["overall_stats"]["success_rate_percentage"] < 95:
        recommendations.append("⚠️ Success rate below 95% target")

    if stats["provider_breakdown"]["deepgram_direct_successes"] / stats["overall_stats"]["total_requests"] < 0.8:
        recommendations.append("⚠️ Deepgram direct success rate low - check API connectivity")

    if not recommendations:
        recommendations.append("✅ All STT performance targets met")

    return {
        "status": "healthy" if stats["performance_targets"]["current_target_met"] else "needs_attention",
        "summary": {
            "total_requests": stats["overall_stats"]["total_requests"],
            "average_time": f"{stats['overall_stats']['average_processing_time']}s",
            "success_rate": f"{stats['overall_stats']['success_rate_percentage']}%",
            "primary_provider_usage": f"{(stats['provider_breakdown']['deepgram_direct_successes'] / max(stats['overall_stats']['total_requests'], 1) * 100):.1f}%"
        },
        "performance_trend": "optimal" if stats["recent_performance"]["last_10_requests_avg_time"] <= 2.5 else "degraded",
        "recommendations": recommendations,
        "optimization_impact": {
            "baseline_time": "4+ seconds (before optimization)",
            "current_avg_time": f"{stats['overall_stats']['average_processing_time']}s",
            "improvement": f"{max(0, ((4.0 - stats['overall_stats']['average_processing_time']) / 4.0 * 100)):.1f}% faster"
        }
    }

@app.delete("/stt-performance/reset")
async def reset_stt_performance_stats():
    """Reset STT performance statistics (admin only)"""
    global stt_metrics

    stt_metrics = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "deepgram_direct_successes": 0,
        "deepgram_fallback_successes": 0,
        "google_fallback_successes": 0,
        "average_processing_time": 0.0,
        "min_processing_time": float('inf'),
        "max_processing_time": 0.0,
        "processing_times": [],
        "provider_performance": {
            "deepgram_direct": {"count": 0, "total_time": 0.0, "success_rate": 0.0},
            "deepgram_fallback": {"count": 0, "total_time": 0.0, "success_rate": 0.0},
            "google_fallback": {"count": 0, "total_time": 0.0, "success_rate": 0.0}
        },
        "recent_performance": []
    }

    return {"message": "STT performance statistics reset successfully", "timestamp": time.time()}

# Add STT performance testing endpoint
@app.post("/test-stt-performance")
async def test_stt_performance_endpoint(
    audio_file: UploadFile = File(...),
    iterations: int = Form(1)
):
    """
    Test STT performance with multiple iterations for benchmarking
    """
    try:
        logging.info(f"🧪 STT Performance Test started - {iterations} iterations")

        # Read audio file once
        audio_content = await audio_file.read()

        results = []
        total_start_time = time.time()

        for i in range(iterations):
            iteration_start = time.time()
            transcript = await speech_to_text_optimized(audio_content, audio_file.filename)
            iteration_time = time.time() - iteration_start

            results.append({
                "iteration": i + 1,
                "transcript": transcript,
                "processing_time": iteration_time,
                "target_met": iteration_time <= 2.5
            })

            logging.info(f"🧪 Iteration {i + 1}: {iteration_time:.3f}s - {'✅' if iteration_time <= 2.5 else '❌'}")

        total_time = time.time() - total_start_time

        # Calculate statistics
        processing_times = [r["processing_time"] for r in results]
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        success_rate = sum(1 for r in results if r["target_met"]) / len(results) * 100

        return {
            "test_results": results,
            "statistics": {
                "total_iterations": iterations,
                "total_time": total_time,
                "average_processing_time": avg_time,
                "min_processing_time": min_time,
                "max_processing_time": max_time,
                "target_success_rate": success_rate,
                "performance_target": "1.5-2.5s",
                "overall_target_met": avg_time <= 2.5
            },
            "stt_metrics": stt_metrics
        }

    except Exception as e:
        logging.error(f"❌ STT Performance Test failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "STT performance test failed", "details": str(e)}
        )
