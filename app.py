#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 18:57:18 2026

@authors: Elias Oliveira Serqueira & Lislei Jorge Patrizzi Martins
@e-mail: elias.serqueira@uftm.edu.br
"""

import os
import io
import base64
import requests
from flask import Flask, request, jsonify, send_file
import google.generativeai as genai
from pydub import AudioSegment

app = Flask(__name__)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
API_KEY = os.environ.get("GEMINI_API_KEY")

def pcm_to_wav(pcm_bytes, sample_rate=16000, channels=1, sample_width=2):
    import wave
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return wav_io.getvalue()

@app.route('/transcrever', methods=['POST'])
def transcrever_audio():
    pcm_data = request.data
    if not pcm_data:
        return jsonify({"erro": "Nenhum áudio recebido"}), 400

    try:
        # 1. Transcrição e Resposta
        wav_data = pcm_to_wav(pcm_data)
        audio_b64 = base64.b64encode(wav_data).decode('utf-8')

        model_chat = genai.GenerativeModel('gemini-3.5-flash-lite')
        prompt = (
            "Transcreva o áudio e responda em português de forma curta. "
            "No final da sua resposta, adicione uma barra vertical '|' seguida por UMA das seguintes emoções: "
            "ALEGRIA, TRISTEZA, ESPANTO, NERVOSO, EXALTADO ou NEUTRO."
        )
        response = model_chat.generate_content([prompt, {"mime_type": "audio/wav", "data": audio_b64}])
        texto_completo = response.text.strip()
        
        if '|' in texto_completo:
            texto_para_falar, emocao = texto_completo.rsplit('|', 1)
        else:
            texto_para_falar = texto_completo
            emocao = "NEUTRO"

        # 2. TTS via REST
        tts_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash-tts:generateContent?key={API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": texto_para_falar}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}}
            }
        }
        
        tts_resp = requests.post(tts_url, json=payload, timeout=30)
        if tts_resp.status_code != 200:
            return jsonify({"status": "erro", "detalhes": tts_resp.text}), 500

        # Extrai o áudio base64 (que o Gemini retorna como WAV)
        audio_b64_24k = tts_resp.json()['candidates'][0]['content']['parts'][0]['inlineData']['data']
        audio_bytes_24k = base64.b64decode(audio_b64_24k)

        # 3. Conversão PERFEITA usando pydub
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes_24k), format="wav")
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        
        # Exporta como PCM puro (sem cabeçalho)
        pcm_16k = audio.raw_data

        # 4. Envia o PCM puro de volta
        resp = send_file(io.BytesIO(pcm_16k), mimetype='application/octet-stream')
        resp.headers['X-Emocao'] = emocao
        return resp

    except Exception as e:
        print(f"Erro: {e}")
        return jsonify({"status": "erro", "detalhes": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
