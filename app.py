from flask import Flask, request, jsonify, render_template
import whisper
from pydub import AudioSegment
import os
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import math
from pytube import YouTube
import validators
import logging
from functools import lru_cache
from dotenv import load_dotenv
import time

load_dotenv()

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

MODEL_NAME = "google/pegasus-multi_news"

def convert_to_wav(audio_file_path):
    try:
        logging.info(f"Converting {audio_file_path} to WAV...")
        file_name, file_extension = os.path.splitext(audio_file_path)
        if file_extension.lower() != '.wav':
            audio = AudioSegment.from_file(audio_file_path)
            wav_file_path = f"{file_name}.wav"
            audio.export(wav_file_path, format="wav")
            logging.info(f"Conversion successful. WAV file saved at {wav_file_path}")
            return wav_file_path
        return audio_file_path
    except Exception as e:
        logging.error(f"Error converting audio to WAV: {e}")
        raise ValueError(f"Error converting audio to WAV: {e}")

@lru_cache(maxsize=1)
def load_whisper_model():
    return whisper.load_model("base")

@lru_cache(maxsize=1)
def load_pegasus_model():
    tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
    model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME)
    return tokenizer, model

def transcribe_audio_with_whisper(audio_file_path):
    try:
        logging.info(f"Transcribing audio file: {audio_file_path}")
        model = load_whisper_model()
        wav_file_path = convert_to_wav(audio_file_path)
        if wav_file_path:
            result = model.transcribe(wav_file_path)
            return result["text"]
        return None
    except Exception as e:
        logging.error(f"Error in audio transcription: {e}")
        raise ValueError(f"Error in audio transcription: {e}")

def summarize_text_with_pegasus(text, tokenizer, model):
    try:
        inputs = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
        total_tokens = len(inputs["input_ids"][0])
        min_summary_length = max(math.ceil(total_tokens / 5), 30)
        max_summary_length = min(math.ceil(total_tokens / 3), 120)

        if min_summary_length >= max_summary_length:
            min_summary_length = max_summary_length - 1

        summary_ids = model.generate(
            inputs.input_ids,
            num_beams=4,
            min_length=min_summary_length,
            max_length=max_summary_length,
            early_stopping=True
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        logging.error(f"Error in text summarization: {e}")
        raise ValueError(f"Error in text summarization: {e}")

def download_audio_from_youtube(url, retries=3, backoff_factor=0.3):
    attempt = 0
    while attempt < retries:
        try:
            if validators.url(url):
                yt = YouTube(url)
                audio_stream = yt.streams.filter(only_audio=True).first()
                if audio_stream:
                    output_file_path = audio_stream.download(filename="downloaded_audio")
                    return output_file_path
                else:
                    raise ValueError("No audio streams found for the YouTube video.")
            else:
                raise ValueError("Invalid YouTube URL provided.")
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            attempt += 1
            time.sleep(backoff_factor * (2 ** attempt))
            if attempt == retries:
                raise ValueError(f"Error downloading audio from YouTube: {e}")
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'aac', 'flac', 'm4a'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'url' in request.form and request.form['url']:
            youtube_url = request.form['url']
            audio_file_path = download_audio_from_youtube(youtube_url)
        elif 'file' in request.files:
            audio_file = request.files['file']
            if not audio_file.filename:
                return jsonify({"error": "No file selected."}), 400
            if not allowed_file(audio_file.filename):
                return jsonify({"error": "Invalid file type. Please upload an audio file."}), 400
            audio_file_path = f"uploaded_audio.{audio_file.filename.split('.')[-1]}"
            audio_file.save(audio_file_path)
        else:
            return jsonify({"error": "No audio file or URL provided."}), 400
        
        transcription = transcribe_audio_with_whisper(audio_file_path)
        if transcription:
            tokenizer, model = load_pegasus_model()
            summary = summarize_text_with_pegasus(transcription, tokenizer, model)
            return jsonify({"transcription": transcription, "summary": summary})
        else:
            return jsonify({"error": "Transcription failed."}), 500
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred."}), 500


if __name__ == "__main__":
    app.run(debug=True)
