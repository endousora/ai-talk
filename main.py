import os
import openai
import warnings
import numpy as np
import sounddevice as sd
import soundfile as sf
import queue
import torch
from pathlib import Path
from faster_whisper import WhisperModel
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel
from style_bert_vits2.logging import logger

# 警告を無視する設定
warnings.filterwarnings("ignore", message=".*torchaudio._backend.set_audio_backend.*")
logger.remove()

# OpenAI APIキーを設定
openai.api_key = "api keyを記入"

# デバイスとモデルの設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
BEAM_SIZE = 3 if DEVICE == "cuda" else 2
MODEL_TYPE = "medium" if DEVICE == "cuda" else "medium"

model = WhisperModel(MODEL_TYPE, device=DEVICE, compute_type=COMPUTE_TYPE)

bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

model_file = "siori/satou_e36_s1000.safetensors"
config_file = "siori/config.json"
style_file = "siori/style_vectors.npy"

assets_root = Path("model_assets")
model_TTS = TTSModel(
    model_path=assets_root / model_file,
    config_path=assets_root / config_file,
    style_vec_path=assets_root / style_file,
    device=DEVICE
)

audio_que = queue.Queue()

# プロンプトの初期ファイル読み込み
PROMPT_FILE = "./prompt/ai.txt"
FIRST_MESSAGE_FILE = "./prompt/ai-first.txt"

with open(PROMPT_FILE, encoding="utf-8") as f:
    sys_prompt = f.read()

with open(FIRST_MESSAGE_FILE, encoding="utf-8") as f:
    first_message = f.read()

def call_first_message():
    sr, audio = model_TTS.infer(text=first_message)
    sd.play(audio, sr)
    sd.wait()

def speech2audio(fs=16000, silence_threshold=0.5, min_duration=0.1, amplitude_threshold=0.025):
    record_Flag = False
    non_recorded_data = []
    recorded_audio = []
    silent_time = 0
    input_time = 0
    start_threshold = 0.3
    all_time = 0

    with sd.InputStream(samplerate=fs, channels=1) as stream:
        while True:
            data, overflowed = stream.read(int(fs * min_duration))
            all_time += 1
            if all_time == 10:
                print("stand by ready OK")
            elif all_time >= 10:
                if np.max(np.abs(data)) > amplitude_threshold and not record_Flag:
                    input_time += min_duration
                    if input_time >= start_threshold:
                        record_Flag = True
                        print("recording...")
                        recorded_audio = non_recorded_data[int(-1 * start_threshold * 10) - 2:]

                else:
                    input_time = 0

                if overflowed:
                    print("Overflow occurred. Some samples might have been lost.")
                if record_Flag:
                    recorded_audio.append(data)
                else:
                    non_recorded_data.append(data)

                if np.all(np.abs(data) < amplitude_threshold):
                    silent_time += min_duration
                    if silent_time >= silence_threshold and record_Flag:
                        print("finished")
                        record_Flag = False
                        break
                else:
                    silent_time = 0

    audio_data = np.concatenate(recorded_audio, axis=0)
    return audio_data

def audio2text(data, model):
    result = ""
    data = data.flatten().astype(np.float32)

    segments, _ = model.transcribe(data, beam_size=BEAM_SIZE)
    for segment in segments:
        result += segment.text

    return result

def text2text2speech(user_prompt, cnt):
    """ユーザーのプロンプトをもとにテキスト生成と音声生成を行う"""
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages
        )
        generated_text = response['choices'][0]['message']['content']
        print("AI response:", generated_text)

        sr, audio = model_TTS.infer(text=generated_text)
        sd.play(audio, sr)
        sd.wait()

    except Exception as e:
        print(f"Error during OpenAI chat completion: {e}")

def process_roleai(audio_data, model, cnt):
    user_prompt = audio2text(audio_data, model)
    print("user: ", user_prompt)

    text2text2speech(user_prompt, cnt)

def main():
    cnt = 0
    call_first_message()
    while True:
        audio_data = speech2audio()
        process_roleai(audio_data, model, cnt)
        cnt += 1

if __name__ == "__main__":
    main()
