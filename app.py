#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 6:14 2026

@authors: Elias Oliveira Serqueira & Lislei Jorge Patrizzi Martins
@e-mail: elias.serqueira@uftm.edu.br
"""

import os
import io
import wave
import base64
import asyncio
import gc
from flask import Flask, request, jsonify
import psycopg2
from groq import Groq
from pydub import AudioSegment
import edge_tts

app = Flask(__name__)

# ====== CONFIGURAÇÕES ======
# Groq (Substitui o Gemini para ASR e Chat)
client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Banco de Dados (PostgreSQL no Render)
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db_connection():
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

# Inicializa o banco ao iniciar o app
init_db()

def pcm_to_wav(pcm_bytes, sample_rate=16000, channels=1, sample_width=2):
    """Converte PCM bruto do ESP32 para WAV para o Groq entender"""
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

    # Lê os sensores enviados pelo ESP32 nos headers
    temp_atual = request.headers.get("X-Temperatura", "Desconhecida")
    umid_atual = request.headers.get("X-Umidade", "Desconhecida")
    hora_atual = request.headers.get("X-Hora", "Desconhecida")
    data_atual = request.headers.get("X-Data", "Desconhecida")
    is_checkin = request.headers.get("X-Checkin-Automatico") == "true"

    try:
        # 1. TRANSCRIÇÃO (Groq Whisper)
        if is_checkin:
            print("--- Check-in automático disparado ---")
            texto_usuario = "[SISTEMA]: Faça um check-in proativo agora."
        else:
            wav_data = pcm_to_wav(pcm_data)
            
            # Chama o Whisper da Groq
            transcription = client_groq.audio.transcriptions.create(
                file=("audio.wav", wav_data),
                model="whisper-large-v3",
                language="pt",
                response_format="json"
            )
            texto_usuario = transcription.text.strip()
            print(f"--- Usuário disse: {texto_usuario} ---")

        # 2. BUSCAR AS ÚLTIMAS MENSAGENS DO BANCO
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT role, content FROM historico_chat ORDER BY id DESC LIMIT 10")
        linhas = cur.fetchall()
        historico_recente = list(reversed(linhas))
        
        # Salva o que o usuário falou
        if texto_usuario != "..." and texto_usuario != "" and not is_checkin:
            cur.execute(
                "INSERT INTO historico_chat (role, content) VALUES (%s, %s)",
                ("user", texto_usuario)
            )
            conn.commit()
        cur.close()
        conn.close()

        # 3. MONTAR O PROMPT COM MEMÓRIA E SENSORES
        contexto = ""
        for role, content in historico_recente:
            if role == "user":
                contexto += f"Usuário: {content}\n"
            else:
                contexto += f"LIS: {content}\n"

        # Prompt do Sistema (Persona e Regras)
        prompt_sistema = (
            "Você é a LIS (Local Intelligent System), uma assistente virtual acolhedora e cuidadora, projetada para ajudar idosos. "
            "Fale de forma clara, pausada e sempre demonstre empatia e cuidado com a saúde do usuário. "
            "Se o usuário falar algo que claramente não é para você (ruído), fique em silêncio seguido da emoção |ESPANTO. "
            "REGRA DE OURO: Você DEVE terminar TODA resposta com uma barra vertical '|' e uma emoção. "
            "IMPORTANTE: Seja extremamente concisa. Limite suas respostas a no máximo 2 ou 3 frases curtas. "
            "Vá direto ao ponto para não cansar o usuário.\n\n"
            "Emoções: ALEGRIA, TRISTEZA, ESPANTO, NERVOSO, EXALTADO ou NEUTRO.\n\n"
            f"INFORMAÇÃO DO AMBIENTE FÍSICO AGORA: Temperatura = {temp_atual}°C, Umidade = {umid_atual}%.\n"
            f"DATA E HORA ATUAIS: Hoje é dia {data_atual}, e agora são exatamente {hora_atual}.\n"
            "REGRAS DE SEGURANÇA PARA A SAÚDE DO IDOSO:\n"
            "1. Se a temperatura estiver acima de 30°C, alerte sobre o calor, recomende evitar exercícios físicos ao ar livre e beber água.\n"
            "2. Considere umidade abaixo de 60% como baixa e abaixo de 25% como alerta. Se estiver baixa ou em alerta, recomende beber muita água e usar umidificador.\n"
            "3. Se o clima estiver bom, apenas converse normalmente.\n\n"
            f"HISTÓRICO DA CONVERSA:\n{contexto}\n"
            f"Usuário: {texto_usuario}\n"
            "LIS: "
        )

        # 4. CHAMAR O GROQ PARA RESPONDER
        chat_completion = client_groq.chat.completions.create(
            messages=[
                {"role": "system", "content": "Você é a LIS, uma assistente para idosos."},
                {"role": "user", "content": prompt_sistema}
            ],
            model="openai/gpt-oss-120b", # Modelo rápido e com cota de 1000/dia
            temperature=0.7,
            max_tokens=150 # Limita o tamanho para respostas curtas
        )
        
        texto_completo = chat_completion.choices[0].message.content
        print(f"--- O Groq respondeu: {texto_completo} ---")
        
        # Separa o texto da emoção
        if '|' in texto_completo:
            texto_para_falar, emocao = texto_completo.rsplit('|', 1)
        else:
            texto_para_falar = texto_completo
            emocao = "NEUTRO"

        # 5. TTS VIA EDGE (Microsoft)
        if texto_para_falar.strip() == "":
            print("Sem texto para falar, retornando apenas a emoção.")
            pcm_16k = b"" 
        else:
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
            try:
                audio_mp3 = loop.run_until_complete(gerar_audio())
            finally:
                loop.close()
                asyncio.set_event_loop(None)

            if not audio_mp3:
                return jsonify({"status": "erro", "detalhes": "TTS nao retornou audio"}), 500

            # 6. CONVERSÃO E TRATAMENTO
            audio = AudioSegment.from_file(io.BytesIO(audio_mp3), format="mp3")
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio = audio.low_pass_filter(3000)
            audio = audio - 18
            pcm_16k = audio.raw_data
            
            del audio_mp3
            del audio

        # 7. SALVAR RESPOSTA DA LIS NO BANCO
        if texto_para_falar.strip() != "":
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO historico_chat (role, content) VALUES (%s, %s)",
                ("assistant", texto_para_falar)
            )
            conn.commit()
            cur.close()
            conn.close()
        
        # 8. ENVIA O PCM DE VOLTA PARA O ESP32
        emocao_padded = emocao.ljust(10, ' ').encode('utf-8')[:10]
        payload_final = emocao_padded + pcm_16k

        # Força a limpeza de memória
        gc.collect()

        return app.response_class(payload_final, mimetype='application/octet-stream')

    except Exception as e:
        print(f"ERRO NO SERVIDOR: {e}")
        erro_str = str(e)
        
        # Tratamento específico para erro de cota (429) da Groq
        if "429" in erro_str or "rate limit" in erro_str.lower():
            return jsonify({
                "status": "erro", 
                "detalhes": "Cota de uso da IA excedida. Tente novamente mais tarde."
            }), 429
            
        return jsonify({"status": "erro", "detalhes": erro_str}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
