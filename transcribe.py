import sys
import os
import time
from faster_whisper import WhisperModel

def setup_cuda_paths():
    venv_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "env", "Lib", "site-packages")
    potential_paths = [
        os.path.join(venv_base, "nvidia", "cublas", "bin"),
        os.path.join(venv_base, "nvidia", "cudnn", "bin"),
    ]
    for path in potential_paths:
        if os.path.exists(path):
            os.add_dll_directory(path)

setup_cuda_paths()

def run_transcription(file_path):
    # Load model (Stay with 'small' to keep it under 1GB)
    model = WhisperModel("small", device="cuda", compute_type="float16")

    print(f"Starting Hinglish transcription for: {file_path}")
    start_time = time.time()

    hinglish_prompt = (
        "Hello CodeArmy, kaise hain aap sab log? Aaj main aapko bataunga "
        "ki kaise aap free mein AI models use kar sakte hain. "
        "Nvidia GPU use karke fast transcription karein."
    )

    segments, info = model.transcribe(
        file_path,
        beam_size=5,
        initial_prompt=hinglish_prompt, 
        vad_filter=True
    )

    print("\n--- Transcription Result ---")
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}", flush=True)

    end_time = time.time()
    print(f"\n--- Processed in {end_time - start_time:.2f} seconds ---")

if __name__ == "__main__":
    audio_input = "test.mp3" 
    if os.path.exists(audio_input):
        run_transcription(audio_input)
    else:
        print(f"Error: {audio_input} not found.")