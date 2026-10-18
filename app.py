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

# A chave será configurada no painel do Render depois (Environment Variables)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Dicionário para manter o histórico de cada dispositivo (Multi-assistentes)
historico_dispositivos = {}

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
    # Identifica qual ESP32 está falando
    device_id = request.headers.get('X-Device-ID', 'default_device')
    
    pcm_data = request.data
    if not pcm_data:
        return jsonify({"erro": "Nenhum áudio recebido"}), 400

    try:
        wav_data = pcm_to_wav(pcm_data)
        audio_b64 = base64.b64encode(wav_data).decode('utf-8')

        # Cria ou recupera a sessão de chat específica desse dispositivo
        if device_id not in historico_dispositivos:
            model = genai.GenerativeModel('gemini-3.6-flash')
            historico_dispositivos[device_id] = model.start_chat(history=[])
        
        chat = historico_dispositivos[device_id]

        # O Gemini transcreve e já responde
        prompt = "Transcreva o que é dito neste áudio em português e responda de forma amigável."
        response = chat.send_message([
            prompt,
            {"mime_type": "audio/wav", "data": audio_b64}
        ])
        
        texto_resposta = response.text.strip()
        print(f"[{device_id}] Resposta: {texto_resposta}")

        # Mantém o histórico curto para não estourar a memória
        if len(chat.history) > 10:
            historico_dispositivos[device_id] = genai.GenerativeModel('gemini-3.5-flash').start_chat(history=chat.history[-6:])

        return jsonify({"status": "sucesso", "texto": texto_resposta}), 200

    except Exception as e:
        print(f"Erro no dispositivo {device_id}: {e}")
        return jsonify({"erro": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
