import streamlit as st
import asyncio
import edge_tts
from google import genai
from google.genai import types

# Page styling & Title
st.set_page_config(page_title="Ranesh Boss AI", page_icon="⚡", layout="centered")
st.title("⚡ Ranesh Boss Turbo AI")
st.caption("Serving Ranesh Boss • Powered by Neural Fast Voice")

# =========================================================================
# 1. FIXED API CLIENT SETUP FOR STREAMLIT CLOUD (Bypasses the 401 bug)
# =========================================================================
API_KEY = st.secrets["AQ.Ab8RN6KWkotFOr8HuXAB75323XgDEmKJnERd_VBBNKFK50i1hQ"]

# We explicitly pass the key into the client options to override header formatting
client = genai.Client(
    api_key=API_KEY,
    http_options={"headers": {"x-goog-api-key": API_KEY}}
)
# =========================================================================

# ... Keep the rest of your system prompt and chat history code completely the same ...

# 2. System Instructions
system_prompt = (
    "You are a helpful AI assistant serving your Boss, Ranesh. "
    "Rule 1: Always respond in the EXACT same language the user uses "
    "(English for English, Hindi script for Hindi, Hinglish for Hinglish). "
    "Rule 2: Address the user respectfully as 'Ranesh' or 'Ranesh Boss'. "
    "Rule 3: Keep responses direct, expressive, and conversational. "
    "Rule 4: Do not repeat previous questions. Answer directly without looping or giving unprompted city history."
)


# 3. Super Fluent & Fast Neural TTS Function
async def generate_neural_speech(text_to_speak):
    try:
        clean_text = text_to_speak.replace("*", "").replace("#", "")
        # MadhurNeural: Most natural and fluent Hindi/Hinglish/Indian English voice
        # rate="+15%" makes the speech fast and energetic
        voice = "hi-IN-MadhurNeural"
        communicate = edge_tts.Communicate(clean_text, voice, rate="+15%")
        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        return bytes(audio_data)
    except Exception:
        return None

def get_voice_audio(text):
    return asyncio.run(generate_neural_speech(text))

# 4. Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "audio" in message and message["audio"]:
            st.audio(message["audio"], format="audio/mp3")

# 5. Input Mode Selector
input_mode = st.radio(
    "👉 Kaise baat karna chahenge, Ranesh Boss?",
    ["⌨️ Likhna (Type Karein)", "🎙️ Bolna (Mic Use Karein)"],
    horizontal=True
)

user_prompt = None
audio_bytes_payload = None

if input_mode == "⌨️ Likhna (Type Karein)":
    chat_input = st.chat_input("Yahan apna sawaal type kijiye, Boss...")
    if chat_input:
        user_prompt = chat_input
elif input_mode == "🎙️ Bolna (Mic Use Karein)":
    audio_data = st.audio_input("Mic dabakar boliye, Boss:")
    if audio_data is not None:
        audio_bytes = audio_data.read()
        if "last_audio_hash" not in st.session_state or st.session_state.last_audio_hash != hash(audio_bytes):
            st.session_state.last_audio_hash = hash(audio_bytes)
            user_prompt = "🎤 [Aapka Voice Message]"
            audio_bytes_payload = types.Part.from_bytes(
                data=audio_bytes,
                mime_type="audio/wav"
            )

# 6. Process & Speak
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            if audio_bytes_payload:
                contents_to_send = [
                    audio_bytes_payload,
                    "Listen to this audio from Ranesh Boss and reply directly."
                ]
            else:
                contents_to_send = user_prompt

            response = client.models.generate_content_stream(
                model="gemini-3.5-flash-lite",
                contents=contents_to_send,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt
                )
            )
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
            
            # Fast neural audio generate karna
            audio_bytes = get_voice_audio(full_response)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "audio": audio_bytes
            })
            
        except Exception as e:
            st.error(f"Error: {e}")
