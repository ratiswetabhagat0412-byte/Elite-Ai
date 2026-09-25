import streamlit as st
from google import genai
from google.genai import types
from gtts import gTTS
import io

# Page styling & Title
st.set_page_config(page_title="Ranesh Boss AI", page_icon="⚡", layout="centered")
st.title("⚡ Ranesh Boss Turbo AI")
st.caption("Serving Ranesh Boss • 100% Reliable Native Voice")

# 1. API Client Setup
API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

# 2. System Instructions for Boss
system_prompt = (
    "You are a helpful AI assistant serving your Boss, Ranesh. "
    "Rule 1: Always respond in the EXACT same language the user uses "
    "(English for English, Hindi script for Hindi, Hinglish for Hinglish). "
    "Rule 2: Address the user respectfully as 'Ranesh' or 'Ranesh Boss'. "
    "Rule 3: Keep responses direct, expressive, and conversational. "
    "Rule 4: Do not repeat previous questions. Answer directly without looping."
)

# 3. 100% Working Native Voice Function (gTTS Hindi-India Male Tone)
def get_voice_audio(text_to_speak):
    try:
        clean_text = text_to_speak.replace("*", "").replace("#", "")
        # Using Indian Accent for natural flow
        tts = gTTS(text=clean_text, lang='hi', tld='co.in', slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as e:
        st.error(f"Voice Error: {e}")
        return None

# 4. Chat History Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation
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

# 6. Process Input, Stream Response & Play Audio
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
            
            # Generate stable native audio track
            audio_stream = get_voice_audio(full_response)
            if audio_stream:
                st.audio(audio_stream, format="audio/mp3", autoplay=True)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "audio": audio_stream
            })
            
        except Exception as e:
            st.error(f"Error: {e}")
