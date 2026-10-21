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
import psycopg2

app = Flask(__name__)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
API_KEY = os.environ.get("GEMINI_API_KEY")

# Pega a URL do banco que você configurou no Render
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db_connection():
    # O Render exige SSL para conexões externas, mas para internas é opcional.
    # O parâmetro sslmode='require' garante que funcione em ambos.
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    return conn

def init_db():
    """Cria a tabela de histórico se ela não existir."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS historico_chat (
            id SERIAL PRIMARY KEY,
            role VARCHAR(20) NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

# Chame a inicialização quando o app começar
# (Coloque isso logo após configurar o app = Flask(__name__))
init_db()

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
    global chat_session # Permite modificar a sessão global
    
    pcm_data = request.data
    if not pcm_data:
        return jsonify({"erro": "Nenhum áudio recebido"}), 400

    try:
        # 1. TRANSCRIÇÃO E RESPOSTA (Gemini)
        wav_data = pcm_to_wav(pcm_data)
        audio_b64 = base64.b64encode(wav_data).decode('utf-8')

        # 1. BUSCAR AS ÚLTIMAS MENSAGENS DO BANCO
        conn = get_db_connection()
        cur = conn.cursor()
        # Pega as últimas 10 interações (5 do usuário, 5 da LIS) para dar contexto
        cur.execute("SELECT role, content FROM historico_chat ORDER BY id DESC LIMIT 10")
        linhas = cur.fetchall()
        # Inverte para ficar na ordem cronológica correta (antigas primeiro)
        historico_recente = list(reversed(linhas))
        cur.close()
        conn.close()

        # 2. MONTAR O PROMPT COM MEMÓRIA
        contexto = ""
        for role, content in historico_recente:
            if role == "user":
                contexto += f"Usuário: {content}\n"
            else:
                contexto += f"LIS: {content}\n"
                
        prompt = (
            "Você é a LIS (Local Intelligent System), uma assistente virtual MUITO expressiva. "
            "Se o usuário não disser 'LIS', fique em silêncio seguido da emoção ESPANTO . "
            "Se disser 'LIS', responda de forma ultra curta. "
            "REGRA DE OURO: Você DEVE terminar TODA resposta com uma barra vertical '|' e uma emoção. "
            "Escolha a emoção que combina com o contexto: ALEGRIA, TRISTEZA, ESPANTO, NERVOSO, EXALTADO ou NEUTRO. "
            "Se o usuário te cumprimentar (bom dia, boa tarde), seja muito feliz e use ALEGRIA ou EXALTADO. "
            "Usuário: LIS, bom dia! -> Sua resposta: 'Bom dia! Como posso ajudar?|ALEGRIA' "
            "Abaixo está o histórico recente da conversa. Use-o para manter a coerência, "
            "lembrar do que foi falado e não se repetir.\n\n"
            f"HISTÓRICO:\n{contexto}\n"
            "Agora, responda ao novo áudio do usuário."
            "Agora responda à pergunta do usuário seguindo EXATAMENTE esse formato."
        )
        
        # 3. CHAMAR O GEMINI (agora com o texto do histórico + áudio)
        model_chat = genai.GenerativeModel('gemini-3.5-flash-lite')
        response = model_chat.generate_content([
            prompt_sistema,
            {"mime_type": "audio/wav", "data": audio_b64}
        ])
        
        texto_completo = response.text.strip()

        print(f"--- O Gemini respondeu: {texto_completo} ---")
        
        # Separa o texto da emoção
        if '|' in texto_completo:
            texto_para_falar, emocao = texto_completo.rsplit('|', 1)
        else:
            texto_para_falar = texto_completo
            emocao = "NEUTRO"

        # 2. TTS VIA EDGE (Microsoft)
        # Usando a voz neural brasileira Francisca
        voice = "pt-BR-FranciscaNeural"
        
        async def gerar_audio():
            communicate = edge_tts.Communicate(texto_para_falar, voice)
            audio_bytes = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]
            return audio_bytes

        # Executa a função assíncrona do edge-tts
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
        
        # Filtro passa-baixa para suavizar (opcional)
        audio = audio.low_pass_filter(3000)
        
        # ATENUAÇÃO: Reduz o volume para não estourar o MAX98357
        audio = audio - 18  # Se ficar baixo, mude para -6. Se estourar, mude para -15.
        
        pcm_16k = audio.raw_data

        # 4. SALVAR NO BANCO (User e LIS)
        # Salva o que o usuário disse (você pode transcrever o áudio antes ou salvar um placeholder)
        # Para simplificar, vamos salvar a resposta da LIS e o texto que o Gemini entendeu.
        # Se quiser salvar o que o usuário falou, você precisaria de uma chamada de ASR extra.
        # Por enquanto, vamos salvar apenas a resposta da LIS para ela não se repetir.
        
        conn = get_db_connection()
        cur = conn.cursor()
        # Salva a resposta da LIS
        cur.execute(
            "INSERT INTO historico_chat (role, content) VALUES (%s, %s)",
            ("assistant", texto_para_falar) # texto_para_falar é a variável que você já usa
        )
        # Se você tiver o texto do usuário (ex: via transcrição), salve também como 'user'
        conn.commit()
        cur.close()
        conn.close()
        
        # 5. ENVIA O PCM DE VOLTA PARA O ESP32
        # Formato: [10 bytes de texto da emoção][Resto do áudio PCM]
        emocao_padded = emocao.ljust(10, ' ').encode('utf-8')[:10]
        
        # Cria o arquivo final juntando a emoção + áudio
        payload_final = emocao_padded + pcm_16k

        # Envia como um arquivo octet-stream simples
        return app.response_class(payload_final, mimetype='application/octet-stream')

    except Exception as e:
        print(f"Erro: {e}")
        return jsonify({"status": "erro", "detalhes": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
