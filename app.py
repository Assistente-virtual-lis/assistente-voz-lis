#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 18:57:18 2026

@author: elias
"""

import os
import io
import wave
import base64
from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def pcm_to_wav(pcm_bytes, sample_rate=16000, channels=1, sample_width=2):
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
        wav_data = pcm_to_wav(pcm_data)
        audio_b64 = base64.b64encode(wav_data).decode('utf-8')

        model = genai.GenerativeModel('gemini-3.5-flash-lite')
        
        # O prompt agora obriga o robô a escolher uma emoção da nossa lista
        prompt = (
            "Transcreva o áudio e responda em português de forma curta. "
            "No final da sua resposta, adicione uma barra vertical '|' seguida por UMA das seguintes emoções "
            "que melhor representa o tom da sua resposta: ALEGRIA, TRISTEZA, ESPANTO, NERVOSO, EXALTADO ou NEUTRO. "
            "Exemplo: 'Olá, como vai?|ALEGRIA'"
        )
        
        response = model.generate_content([
            prompt,
            {"mime_type": "audio/wav", "data": audio_b64}
        ])
        
        texto_resposta = response.text.strip()
        print(f"Resposta: {texto_resposta}")

        return jsonify({"status": "sucesso", "texto": texto_resposta}), 200

    except Exception as e:
        print(f"Erro: {e}")
        return jsonify({"status": "erro", "detalhes": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
