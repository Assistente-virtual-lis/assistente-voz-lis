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
from flask import Flask, request, jsonify, send_file
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

def resample_wav_24k_to_16k(wav_bytes_24k):
    """Converte WAV de 24kHz para 16kHz (simples decimação para evitar chiado)"""
    wav_in = wave.open(io.BytesIO(wav_bytes_24k), 'rb')
    frames = wav_in.readframes(wav_in.getnframes())
    
    # Converte bytes para array de int16
    import array
    samples_24k = array.array('h', frames)
    
    # Decimação: pega 2 de cada 3 amostras (24k -> 16k)
    samples_16k = array.array('h', [samples_24k[i] for i in range(0, len(samples_24k), 3)][:int(len(samples_24k)/1.5)])
    
    # Cria novo WAV
    wav_out = io.BytesIO()
    with wave.open(wav_out, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(samples_16k.tobytes())
    
    return wav_out.getvalue()

@app.route('/transcrever', methods=['POST'])
def transcrever_audio():
    pcm_data = request.data
    if not pcm_data:
        return jsonify({"erro": "Nenhum áudio recebido"}), 400

    try:
        # 1. Transcrição e resposta do Gemini (como já funciona)
        wav_data = pcm_to_wav(pcm_data)
        audio_b64 = base64.b64encode(wav_data).decode('utf-8')

        model = genai.GenerativeModel('gemini-3.5-flash-lite')
        
        prompt = (
            "Transcreva o áudio e responda em português de forma curta. "
            "No final da sua resposta, adicione uma barra vertical '|' seguida por UMA das seguintes emoções: "
            "ALEGRIA, TRISTEZA, ESPANTO, NERVOSO, EXALTADO ou NEUTRO."
        )
        
        response = model.generate_content([
            prompt,
            {"mime_type": "audio/wav", "data": audio_b64}
        ])
        
        texto_completo = response.text.strip()
        
        # Separa o texto da emoção para o TTS (o TTS não deve falar a emoção)
        if '|' in texto_completo:
            texto_para_falar, emocao = texto_completo.rsplit('|', 1)
        else:
            texto_para_falar = texto_completo
            emocao = "NEUTRO"

        # 2. Chama o TTS do Gemini
        tts_response = model.generate_content(
            f"Fale de forma natural: {texto_para_falar}",
            generation_config={
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}},
                    "language_code": "pt-BR"
                }
            }
        )
        
        # 3. Extrai o áudio e converte para 16kHz
        audio_bytes_24k = None
        for part in tts_response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                audio_bytes_24k = part.inline_data.data
                break
        
        if not audio_bytes_24k:
            return jsonify({"status": "erro", "detalhes": "TTS nao retornou audio"}), 500

        # Converte 24kHz para 16kHz
        audio_16k = resample_wav_24k_to_16k(audio_bytes_24k)
        
        # 4. Retorna o WAV + a emoção no header
        resp = send_file(io.BytesIO(audio_16k), mimetype='audio/wav')
        resp.headers['X-Emocao'] = emocao
        return resp

    except Exception as e:
        print(f"Erro: {e}")
        return jsonify({"status": "erro", "detalhes": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
