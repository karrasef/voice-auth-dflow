import numpy as np
import whisper
import re
import scipy.io.wavfile as wavfile
import pyaudio
import wave
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import pyttsx3
from dotenv import load_dotenv
import os
import requests
import json
import uuid

model = whisper.load_model("small.en")  

engine =pyttsx3.init()

load_dotenv()

#Parameters for VAD
CHUNK = 1024  # Number of audio samples per frame
FORMAT = pyaudio.paInt16
CHANNELS = 1
SILENCE_THRESHOLD = int(os.getenv("SILENCE_THRESHOLD"))
SILENCE_LIMIT = int(os.getenv("SILENCE_LIMIT"))

SAMPLERATE = int(os.getenv("SAMPLERATE"))
DURATION = int(os.getenv("DURATION"))
PHRASES = [os.getenv("PHRASES")]
PROFILE_DIR = "voice_profiles" 
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD"))
WAKE_WORD_REGEX = os.getenv("WAKE_WORD_REGEX")

Path(PROFILE_DIR).mkdir(exist_ok=True)

def is_silent(data):
    """Check if audio data is below the silence threshold."""
    return np.max(np.frombuffer(data, dtype=np.int16)) < SILENCE_THRESHOLD
def record_audio(filename, samplerate):
    """Records audio until a period of silence is detected and saves it as a WAV file."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=samplerate, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording... Speak now.")
    frames = []
    silent_chunks = 0

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        if is_silent(data):
            silent_chunks += 1
        else:
            silent_chunks = 0

        if silent_chunks > SILENCE_LIMIT:
            print("Silence detected. Stopping recording.")
            break

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio to a WAV file
    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(samplerate)
    wf.writeframes(b"".join(frames))
    wf.close()

    print(f"Recording saved to {filename}")

def transcribe_audio_whisper(audio_data, samplerate):
    try:
        wavfile.write("temp.wav", samplerate, audio_data)
        result = model.transcribe("temp.wav") 
        return result['text']
    except Exception as e:
        print(f"An error occured during transcription: {e}")
        return ""    

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen_for_wake_word():
    """Continuously listens for the wake word using dynamic audio recording and transcription."""
    print("Listening for wake word...")
    while True:
        # Use the `record_audio` function to record until silence is detected
        filename = "wake_word_attempt.wav"
        record_audio(filename, SAMPLERATE)  # Record audio dynamically
        audio_data = process_audio(filename)

        if audio_data is None or audio_data.size == 0:
            print("Failed to process audio. Retrying...")
            continue

        # Transcribe the audio using Whisper
        transcribed_text = transcribe_audio_whisper(audio_data, SAMPLERATE)
        if not transcribed_text:
            print("No transcription was returned. Retrying...")
            continue

        print(f"Transcribed Text: {transcribed_text}")

        # Check if the wake word is detected in the transcribed text
        if re.search(WAKE_WORD_REGEX, transcribed_text, re.IGNORECASE):
            print("Wake word 'Kira' detected!")
            speak("Hello there!")
            print("Listening for user command...")
            return

def process_audio(filepath):
    if not Path(filepath).is_file():
        print(f"File {filepath} does not exist!")
        return None
    try:
        wav = preprocess_wav(Path(filepath))
        return wav
    except Exception as e:
        print(f"An error occurred while processing the file {filepath}: {e}")
        return None

def compare_embeddings(embed1, embed2):
    return np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))

def authenticate(similarity, threshold):
    return similarity >= threshold


def enroll_user(encoder, username):
    """Enrolls a new user and assigns a unique ID."""
    user_id = str(uuid.uuid4())  
    print(f"Generated ID for user '{username}': {user_id}")

    embeddings = []
    for idx, phrase in enumerate(PHRASES, 1):
        print(f"\nPlease say the following phrase (repeat it if necessary): '{phrase}'")
        filename = f"enroll_{username}_phrase_{idx}.wav"
        record_audio(filename, SAMPLERATE)
        wav_data = process_audio(filename)
        
        if wav_data is None or wav_data.size == 0:
            print(f"Failed to process audio for phrase {idx}. Please try again.")
            continue
            
        try:
            embed = encoder.embed_utterance(wav_data)
            embeddings.append(embed)
            print(f"Phrase {idx} processed.")
        except Exception as e:
            print(f"Failed to generate embedding for phrase {idx}: {e}")

    if len(embeddings) == 0:
        print("Enrollment failed: No valid embeddings were created.")
        return

    avg_embedding = np.mean(embeddings, axis=0)
    profile_path = Path(PROFILE_DIR) / f"{user_id}_profile.npy"
    np.save(profile_path, avg_embedding)
    print(f"\nVoice profile for '{username}' (ID: {user_id}) created and saved.")

    #Save user details (username and ID)
    user_data_path = Path(PROFILE_DIR) / f"{user_id}_data.json"
    with open(user_data_path, "w") as f:
        json.dump({"id": user_id, "username": username}, f)

    print(f"User data for '{username}' saved with ID '{user_id}'.")        

def load_voice_profiles():
    profiles = {}
    user_data = {}

    for profile_path in Path(PROFILE_DIR).glob("*.npy"):
        try:
            user_id = profile_path.stem.replace("_profile", "")
            profile_embedding = np.load(profile_path)
            profiles[user_id] = profile_embedding

            user_data_path = Path(PROFILE_DIR) / f"{user_id}_data.json"
            if user_data_path.exists():
                with open(user_data_path, "r") as f:
                    user_data[user_id] = json.load(f)
        except Exception as e:
            print(f"Failed to load profile for user ID {user_id}: {e}")
            continue
    return profiles, user_data

def send_to_dflow(transcribed_text, user_id):
    """Send the transcribed text to dFlow (Rasa endpoint)."""
    rasa_endpoint = "http://localhost:5005/webhooks/rest/webhook"  # dFlow's REST endpoint
    payload = {
        "sender": user_id,
        "message": transcribed_text
    }

    print(f"Sending to dFlow: {payload}") 

    try:
        response = requests.post(rasa_endpoint, json=payload)
        if response.status_code == 200:
            print("Response from dFlow:")
            for reply in response.json():
                bot_response = reply.get('text', '')
                if bot_response:
                    print(f"Bot: {bot_response}")
                    speak(bot_response)
        else:
            print(f"Failed to send text to dFlow. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending text to dFlow: {e}")

def authenticate_user(encoder, profiles, user_data, threshold):
    """Authenticates a user by comparing a recorded phrase with all saved voice profiles."""
    
    print("\nPlease say a phrase to authenticate:")
    record_audio('auth_attempt.wav', SAMPLERATE)
    auth_wav = process_audio('auth_attempt.wav')
    
    if auth_wav is None or auth_wav.size == 0:
        print("Authentication failed: Unable to process the audio.")
        return
    
    try:
        auth_embed = encoder.embed_utterance(auth_wav)
    except Exception as e:
        print(f"Failed to generate embedding for authentication: {e}")
        return
    
    best_match_id = None
    best_match_username = None
    highest_similarity = -1

    for user_id, profile_embed in profiles.items():
        try:
            similarity = compare_embeddings(auth_embed, profile_embed)
            username = user_data[user_id]["username"] # Retrieve the username
            print(f"Similarity with {username}(ID: {user_id}): {similarity:.3f}")
        except Exception as e:
            print(f"Failed to compare embeddings for user ID {user_id}: {e}")
            continue
        
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match_id = user_id
            best_match_username = username
    
    if best_match_id and authenticate(highest_similarity, threshold):
        print(f"User '{best_match_username}' (ID: {best_match_id}) successfully authenticated.")

        # Transcribe the audio using Whisper
        transcribed_text = transcribe_audio_whisper(auth_wav, SAMPLERATE)
        if not transcribed_text:
            print("No transcription was returned. Retrying...")
        print(f"User's command: {transcribed_text}")
        send_to_dflow(transcribed_text, best_match_id)     # Send transcribed text and userID to dflow
        listen_for_wake_word()  
            
        profiles, user_data = load_voice_profiles()
        if profiles:
            authenticate_user(encoder, profiles, user_data, DEFAULT_THRESHOLD)
        else:
            print("No profiles available. Please enroll users first.")
    else:
        print("Authentication failed: No matching profile found.")
        listen_for_wake_word()  
            
        profiles, user_data = load_voice_profiles()
        if profiles:
            authenticate_user(encoder, profiles, user_data, DEFAULT_THRESHOLD)
        else:
            print("No profiles available. Please enroll users first.")

def delete_user_profile(username):
    profile_path = Path(PROFILE_DIR) / f"{username}_profile.npy"
    if profile_path.exists():
        os.remove(profile_path)
        print(f"Profile for '{username}' deleted.")
    else:
        print(f"No profile found for '{username}'.")

    enrollment_files = Path(".").glob(f"enroll_{username}_phrase_*.wav")
    for file in enrollment_files:
        os.remove(file)
        print(f"Deleted file: {file}")

def list_voice_profiles():
    """Lists all the saved voice profiles."""
    profiles = [path.stem.replace("_profile", "") for path in Path(PROFILE_DIR).glob("*.npy")]
    return profiles

def main():
    encoder = VoiceEncoder()

    while True:
        print("\nOptions:")
        print("1. Enroll new user")
        print("2. Authenticate user")
        print("3. Delete a user profile")
        print("4. Exit")
        
        choice = input("Enter your choice: ")

        if choice == '1':
            username = input("Enter a username for enrollment: ").strip()
            enroll_user(encoder, username)
        elif choice == '2':
            listen_for_wake_word()  
            
            profiles, user_data = load_voice_profiles()
            if profiles:
                authenticate_user(encoder, profiles, user_data, DEFAULT_THRESHOLD)
            else:
                print("No profiles available. Please enroll users first.")
        elif choice == '3':
            listen_for_wake_word()  
            
            profiles = list_voice_profiles()
            if profiles:
                print("\nAvailable profiles:")
                for idx, profile in enumerate(profiles, 1):
                    print(f"{idx}. {profile}")
                
                profile_idx = int(input("Enter the number of the profile to delete: "))
                if 1 <= profile_idx <= len(profiles):
                    username = profiles[profile_idx - 1]
                    delete_user_profile(username)
                else:
                    print("Invalid selection.")
            else:
                print("No voice profiles available.")
        elif choice == '4':
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
