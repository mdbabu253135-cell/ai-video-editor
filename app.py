import streamlit as st
import whisper
import tempfile
import os
from PIL import Image
from moviepy.editor import *

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Video Editor", layout="wide")
st.title("🎬 AI Scene Video Editor (Optimized Render Version)")

# ---------------- INPUT ----------------
images = st.file_uploader(
    "Upload Images (Sequential)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

audio_file = st.file_uploader(
    "Upload Voiceover (MP3/WAV)",
    type=["mp3", "wav"]
)

script_file = st.file_uploader(
    "Upload Script (.txt with ###SCENE###)",
    type=["txt"]
)

aspect = st.selectbox("Aspect Ratio", ["16:9", "9:16"])
mix_transition = st.toggle("Mix Transition (0.15s)", value=False)

# ---------------- HELPERS ----------------

def size(r):
    # ✅ 720p for stability (IMPORTANT)
    return (1280, 720) if r == "16:9" else (720, 1280)


def crop_resize(clip, target):
    return clip.resize(height=target[1]).crop(
        width=target[0],
        height=target[1],
        x_center=clip.w / 2,
        y_center=clip.h / 2
    )


def split_scenes(text):
    return [s.strip() for s in text.split("###SCENE###") if s.strip()]

# ---------------- MAIN ----------------

if st.button("🚀 Generate Video", use_container_width=True):

    if not images or not audio_file or not script_file:
        st.error("Missing files!")
        st.stop()

    with st.spinner("Processing..."):

        temp = tempfile.mkdtemp()

        # ---------------- Save audio ----------------
        audio_path = os.path.join(temp, "audio.mp3")
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())

        # ---------------- Script ----------------
        script = script_file.read().decode("utf-8")
        scenes = split_scenes(script)

        # ---------------- Whisper (SAFE) ----------------
        try:
            model = whisper.load_model("base")  # ✅ safer than small
        except:
            st.warning("Whisper failed, switching to tiny...")
            model = whisper.load_model("tiny")

        st.info("Transcribing audio...")

        result = model.transcribe(audio_path)

        audio_duration = AudioFileClip(audio_path).duration
        total_scenes = len(scenes)

        # ---------------- SAFE TIMELINE ----------------
        base_split = audio_duration / total_scenes

        scene_times = []
        for i in range(total_scenes):
            start = i * base_split
            end = (i + 1) * base_split
            scene_times.append((start, end))

        # ---------------- BUILD VIDEO ----------------
        target_size = size(aspect)
        clips = []

        for i in range(min(len(images), total_scenes)):

            img_path = os.path.join(temp, f"img_{i}.jpg")

            with open(img_path, "wb") as f:
                f.write(images[i].read())

            start, end = scene_times[i]
            duration = max(end - start, 1)

            clip = ImageClip(img_path).set_duration(duration)
            clip = crop_resize(clip, target_size)

            if mix_transition:
                clip = clip.crossfadein(0.15)

            clips.append(clip)

        # ---------------- FINAL VIDEO ----------------
        final = concatenate_videoclips(
            clips,
            method="compose",
            padding=-0.15 if mix_transition else 0
        )

        audio = AudioFileClip(audio_path)
        final = final.set_audio(audio)

        output = os.path.join(temp, "output.mp4")

        st.info("Rendering video...")

        final.write_videofile(
            output,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast"
        )

    st.success("🎉 Successful!")

    with open(output, "rb") as f:
        st.download_button(
            "⬇ Download Video",
            f,
            file_name="final_video.mp4",
            mime="video/mp4"
        )
