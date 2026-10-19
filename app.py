#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 18:57:18 2026

@authors: Elias Oliveira Serqueira & Lislei Jorge Patrizzi Martins
@e-mail: elias.serqueira@uftm.edu.br
"""

import os
import io
import wave
import base64
import asyncio
import requests
from flask import Flask, request, jsonify, send_file
import google.generativeai as genai
from pydub import AudioSegment
import edge_tts

app = Flask(__name__)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
API_KEY = os.environ.get("GEMINI_API_KEY")

def pcm_to_wav(pcm_bytes, sample_rate=16000, channels=1, sample_width=2):
    """Converte PCM bruto do ESP32 para WAV para o Gemini entender"""
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
        # 1. TRANSCRIÇÃO E RESPOSTA (Gemini)
        wav_data = pcm_to_wav(pcm_data)
        audio_b64 = base64.b64encode(wav_data).decode('utf-8')

        model_chat = genai.GenerativeModel('gemini-3.5-flash-lite')
        prompt = (
            "Você é a LIS (Local Intelligent System), uma assistente virtual simpática. "
            "Se o usuário disser o seu nome 'lis' ou 'liz' seguido de uma pergunta, responda à pergunta dele em português de forma direta e ultra curta. "
            "Se o usuário não disser 'lis' ou 'liz', responda APENAS a palavra 'IGNORAR'. "
            "No final da sua resposta, adicione uma barra vertical '|' seguida por UMA destas emoções: "
            "ALEGRIA, TRISTEZA, ESPANTO, NERVOSO, EXALTADO ou NEUTRO. "
            "Exemplo: 'O dia esta lindamente ensolarado.|ALEGRIA'"
        )
        
        response = model_chat.generate_content([
            prompt,
            {"mime_type": "audio/wav", "data": audio_b64}
        ])
        
        texto_completo = response.text.strip()
        print(f"Resposta bruta do Gemini: {texto_completo}")

        # Verifica se o Gemini decidiu ignorar
        if "IGNORAR" in texto_completo:
            resp = send_file(io.BytesIO(b""), mimetype='application/octet-stream')
            resp.headers['X-Emocao'] = "NEUTRO"
            return resp

        # Separa o texto da emoção
        if '|' in texto_completo:
            texto_para_falar, emocao = texto_completo.rsplit('|', 1)
        else:
            texto_para_falar = texto_completo
            emocao = "NEUTRO"

        texto_para_falar = texto_para_falar.replace("IGNORAR", "").strip()

        if not texto_para_falar:
             resp = send_file(io.BytesIO(b""), mimetype='application/octet-stream')
             resp.headers['X-Emocao'] = "NEUTRO"
             return resp

        # 2. TTS VIA EDGE (Microsoft) - Gratuito e sem limite baixo
        voice = "pt-BR-FranciscaNeural"
        
        async def gerar_audio():
            communicate = edge_tts.Communicate(texto_para_falar, voice)
            audio_bytes = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]
            return audio_bytes

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_mp3 = loop.run_until_complete(gerar_audio())
        loop.close()

        if not audio_mp3:
            return jsonify({"status": "erro", "detalhes": "TTS nao retornou audio"}), 500

        # 3. CONVERSÃO E TRATAMENTO (Pydub)
        # Converte MP3 (Edge) para PCM 16kHz 16bits
        audio = AudioSegment.from_file(io.BytesIO(audio_mp3), format="mp3")
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        
        # Filtro passa-baixa para suavizar e atenuação para não estourar o MAX98357
        audio = audio.low_pass_filter(3000)
        audio = audio - 18  # Ajuste de volume (o seu ajuste perfeito)
        
        pcm_16k = audio.raw_data

        # 4. ENVIA O PCM DE VOLTA PARA O ESP32
        resp = send_file(io.BytesIO(pcm_16k), mimetype='application/octet-stream')
        resp.headers['X-Emocao'] = emocao
        return resp

    except Exception as e:
        print(f"Erro: {e}")
        return jsonify({"status": "erro", "detalhes": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
