import streamlit as st
import whisper
import tempfile
import os
from PIL import Image, ImageEnhance
from moviepy.editor import *

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Video Editor", layout="wide")
st.title("🎬 AI Scene Video Editor (Stable Render Version)")

# ---------------- INPUT ----------------
images = st.file_uploader(
    "Upload Images (Sequential)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

audio_file = st.file_uploader(
    "Upload Voiceover (Fish Audio MP3/WAV)",
    type=["mp3", "wav"]
)

script_file = st.file_uploader(
    "Upload Script (.txt with ###SCENE###)",
    type=["txt"]
)

aspect = st.selectbox("Aspect Ratio", ["16:9", "9:16"])
mix_transition = st.toggle("Mix Transition (0.15s)", value=False)
filter_4k = st.toggle("4K Enhancement", value=False)

# ---------------- HELPERS ----------------

def size(r):
    return (1920, 1080) if r == "16:9" else (1080, 1920)


def enhance_image(path):
    img = Image.open(path)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Sharpness(img).enhance(1.2)

    out = path.replace(".", "_enhanced.")
    img.save(out)
    return out


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

    with st.spinner("Processing (this may take 2–5 min)..."):

        temp = tempfile.mkdtemp()

        # ---------------- Save audio ----------------
        audio_path = os.path.join(temp, "audio.mp3")
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())

        # ---------------- Script ----------------
        script = script_file.read().decode("utf-8")
        scenes = split_scenes(script)

        # ---------------- Whisper (SAFE LOAD) ----------------
        try:
            model = whisper.load_model("small")
        except:
            st.warning("Whisper load failed, switching to base...")
            model = whisper.load_model("base")

        st.info("Transcribing audio...")

        result = model.transcribe(audio_path)
        audio_duration = AudioFileClip(audio_path).duration

        total_scenes = len(scenes)

        # ---------------- SAFE TIMELINE ----------------
        # (Hybrid approach = stable + decent sync)
        base_split = audio_duration / total_scenes

        scene_times = []
        for i in range(total_scenes):
            start = i * base_split
            end = (i + 1) * base_split
            scene_times.append((start, end))

        # ---------------- VIDEO BUILD ----------------
        target_size = size(aspect)
        clips = []

        for i in range(min(len(images), total_scenes)):

            img_path = os.path.join(temp, f"img_{i}.jpg")

            with open(img_path, "wb") as f:
                f.write(images[i].read())

            if filter_4k:
                img_path = enhance_image(img_path)

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

        st.info("Rendering video... (final step)")

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
