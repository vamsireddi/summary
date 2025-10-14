import os
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

SUMMARY_MODEL = "gpt-4-turbo-preview"
SUMMARY_PROMPT = (
    "You are an expert meeting minutes generator. "
    "Analyze the following meeting transcript. "
    "Output structured: Summary, Key Decisions, Action Items."
)

def transcribe_audio(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="text")
    return transcript

def generate_summary(transcript):
    full_prompt = SUMMARY_PROMPT + "\n\n" + transcript
    response = client.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=[{"role":"system","content":"You are a professional meeting summarizer."},
                  {"role":"user","content":full_prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status":"ok","message":"Meeting Summarizer API running"})

@app.route('/summarize', methods=['POST'])
def summarize_meeting():
    if 'audio_file' not in request.files:
        return jsonify({"error":"No 'audio_file' part"}), 400
    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return jsonify({"error":"No selected file"}), 400
    temp_path = f"/tmp/{audio_file.filename}"
    audio_file.save(temp_path)
    transcript = transcribe_audio(temp_path)
    summary = generate_summary(transcript)
    os.remove(temp_path)
    return jsonify({"transcript": transcript, "meeting_summary": summary})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
