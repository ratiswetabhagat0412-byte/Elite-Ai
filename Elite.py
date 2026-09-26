import asyncio
import edge_tts
from google import genai
from google.genai import types
import streamlit as st

# Page Configuration
st.set_page_config(page_title="Boss AI", page_icon="⚡", layout="centered")
st.title("⚡ Boss Turbo AI")
st.caption("Serving Boss • Pure Neural Fast Voice & Live Google Search")

# 1. API Client Setup
API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

# 2. System Instructions
system_prompt = (
    "You are a helpful AI assistant serving your user, whom you must always address simply as 'Boss'. "
    "Rule 1: Always respond in the EXACT same language the user uses "
    "(English for English, Hindi script for Hindi, Hinglish for Hinglish). "
    "Rule 2: Address the user respectfully as 'Boss'. Never use any other personal name. "
    "Rule 3: Keep responses direct, expressive, crisp, and conversational. "
    "Rule 4: Do not repeat previous questions. Answer directly without looping. "
    "Rule 5: Use Google Search automatically whenever up-to-date, factual, or real-world information is required."
)

# 3. Sidebar Controls
with st.sidebar:
    st.header("⚙️ Boss Settings")
    enable_voice = st.toggle("🔊 Voice Response", value=True)
    voice_speed = st.slider("⚡ Voice Speed", min_value=0, max_value=30, value=15, step=5, format="+%d%%")
    selected_voice = st.selectbox(
        "🎙️ Neural Voice",
        options=["hi-IN-MadhurNeural", "hi-IN-SwaraNeural", "en-IN-PrabhatNeural", "en-IN-NeerjaNeural"],
        index=0
    )
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.session_state.gemini_history = []
        st.rerun()

# 4. Neural Speech Generator
async def generate_neural_speech(text_to_speak, voice, speed):
    try:
        clean_text = text_to_speak.replace("*", "").replace("#", "")
        communicate = edge_tts.Communicate(clean_text, voice, rate=f"+{speed}%")
        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        return bytes(audio_data)
    except Exception:
        return None

# 5. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "gemini_history" not in st.session_state:
    st.session_state.gemini_history = []

# Display Existing Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("image"):
            st.image(message["image"], caption="Uploaded Image", use_container_width=True)
        if message.get("sources"):
            with st.expander("🔍 Search Sources"):
                for title, url in message["sources"]:
                    st.markdown(f"- [{title}]({url})")
        if message.get("audio"):
            st.audio(message["audio"], format="audio/mp3")

# 6. Inputs (Image / Text / Mic)
uploaded_file = st.sidebar.file_uploader("📷 Share an image with Boss AI", type=["png", "jpg", "jpeg", "webp"])

input_mode = st.radio(
    "👉 Kaise baat karna chahenge, Boss?",
    ["⌨️ Likhna (Type Karein)", "🎙️ Bolna (Mic Use Karein)"],
    horizontal=True
)

user_prompt_display = None
user_parts = []
uploaded_image_bytes = None

if uploaded_file:
    uploaded_image_bytes = uploaded_file.read()
    user_parts.append(
        types.Part.from_bytes(data=uploaded_image_bytes, mime_type=uploaded_file.type)
    )

if input_mode == "⌨️ Likhna (Type Karein)":
    chat_input = st.chat_input("Yahan apna sawaal type kijiye, Boss...")
    if chat_input:
        user_prompt_display = chat_input
        user_parts.append(types.Part.from_text(text=chat_input))

elif input_mode == "🎙️ Bolna (Mic Use Karein)":
    audio_data = st.audio_input("Mic dabakar boliye, Boss:")
    if audio_data is not None:
        audio_bytes = audio_data.read()
        if "last_audio_hash" not in st.session_state or st.session_state.last_audio_hash != hash(audio_bytes):
            st.session_state.last_audio_hash = hash(audio_bytes)
            user_prompt_display = "🎤 [Aapka Voice Message]"
            user_parts.append(
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
            )
            user_parts.append(
                types.Part.from_text(text="Listen carefully to this audio message from Boss and reply directly.")
            )

# 7. Process & Stream Output
if user_prompt_display and user_parts:
    user_msg_entry = {"role": "user", "content": user_prompt_display}
    if uploaded_image_bytes:
        user_msg_entry["image"] = uploaded_image_bytes
    st.session_state.messages.append(user_msg_entry)

    with st.chat_message("user"):
        st.markdown(user_prompt_display)
        if uploaded_image_bytes:
            st.image(uploaded_image_bytes, caption="Uploaded Image", use_container_width=True)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        sources = []

        try:
            current_user_content = types.Content(role="user", parts=user_parts)
            payload_contents = st.session_state.gemini_history + [current_user_content]

            response = client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=payload_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
            )

            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    response_placeholder.markdown(full_response + "▌")

                if chunk.candidates and chunk.candidates[0].grounding_metadata:
                    metadata = chunk.candidates[0].grounding_metadata
                    if metadata.grounding_chunks:
                        for gc in metadata.grounding_chunks:
                            if gc.web and gc.web.uri:
                                title = gc.web.title or gc.web.uri
                                if (title, gc.web.uri) not in sources:
                                    sources.append((title, gc.web.uri))

            response_placeholder.markdown(full_response)

            if sources:
                with st.expander("🔍 Search Sources"):
                    for title, url in sources:
                        st.markdown(f"- [{title}]({url})")

            st.session_state.gemini_history.append(current_user_content)
            st.session_state.gemini_history.append(
                types.Content(role="model", parts=[types.Part.from_text(text=full_response)])
            )

            audio_bytes = None
            if enable_voice and full_response.strip():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                audio_bytes = loop.run_until_complete(
                    generate_neural_speech(full_response, selected_voice, voice_speed)
                )
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
