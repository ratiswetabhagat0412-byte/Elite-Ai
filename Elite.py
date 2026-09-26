import asyncio
import edge_tts
from google import genai
from google.genai import types
import streamlit as st

# Page styling & Title
st.set_page_config(page_title="Ranesh Boss AI", page_icon="⚡", layout="centered")
st.title("⚡ Ranesh Boss Turbo AI")
st.caption("Serving Ranesh Boss • Pure Neural Fast Voice & Live Google Search Restored")

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
    "Rule 4: Do not repeat previous questions. Answer directly without looping. "
    "Rule 5: Use Google Search automatically whenever up-to-date, factual, or detailed real-world information is required."
)

# 3. Restored Fluent & Fast Madhur Neural TTS Function
async def generate_neural_speech(text_to_speak):
    try:
        clean_text = text_to_speak.replace("*", "").replace("#", "")
        voice = "hi-IN-MadhurNeural"
        communicate = edge_tts.Communicate(clean_text, voice, rate="+15%")
        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        return bytes(audio_data)
    except Exception:
        return None

# 4. Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("🔍 Search Sources"):
                for title, url in message["sources"]:
                    st.markdown(f"- [{title}]({url})")
        if message.get("audio"):
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

# 6. Process & Stream Output
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        sources = []
        
        try:
            if audio_bytes_payload:
                contents_to_send = [
                    audio_bytes_payload,
                    "Listen to this audio from Ranesh Boss and reply directly."
                ]
            else:
                contents_to_send = user_prompt

            # Enable Google Search grounding tool
            response = client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=contents_to_send,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
            )

            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    response_placeholder.markdown(full_response + "▌")
                
                # Extract grounding metadata / source URLs if Google Search was triggered
                if chunk.candidates and chunk.candidates[0].grounding_metadata:
                    metadata = chunk.candidates[0].grounding_metadata
                    if metadata.grounding_chunks:
                        for gc in metadata.grounding_chunks:
                            if gc.web and gc.web.uri:
                                title = gc.web.title or gc.web.uri
                                if (title, gc.web.uri) not in sources:
                                    sources.append((title, gc.web.uri))

            response_placeholder.markdown(full_response)

            # Display web sources under an expander if search was used
            if sources:
                with st.expander("🔍 Search Sources"):
                    for title, url in sources:
                        st.markdown(f"- [{title}]({url})")
            
            # Generate Audio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            audio_bytes = loop.run_until_complete(generate_neural_speech(full_response))
            loop.close()
            
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "sources": sources,
                "audio": audio_bytes
            })
            
        except Exception as e:
            st.error(f"Error: {e}")
