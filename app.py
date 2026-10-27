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
    pcm_data = request.data
    if not pcm_data:
        return jsonify({"erro": "Nenhum áudio recebido"}), 400

    temp_atual = request.headers.get("X-Temperatura", "Desconhecida")
    umid_atual = request.headers.get("X-Umidade", "Desconhecida")
    hora_atual = request.headers.get("X-Hora", "Desconhecida")
    data_atual = request.headers.get("X-Data", "Desconhecida")    
    is_checkin = request.headers.get("X-Checkin-Automatico") == "true"

    try:
        # 1. TRANSCRIÇÃO (ou Check-in)
        if is_checkin:
            print("--- Check-in automático disparado ---")
            texto_usuario = "[SISTEMA]: Faça um check-in proativo agora."
        else:
            wav_data = pcm_to_wav(pcm_data)
            audio_b64 = base64.b64encode(wav_data).decode('utf-8')
            model_asr = genai.GenerativeModel('gemini-3.5-flash-lite')
            asr_prompt = "Transcreva exatamente o que o usuário disse neste áudio em português. Se não houver fala, retorne apenas '...'."
            
            response_asr = model_asr.generate_content([
                asr_prompt,
                {"mime_type": "audio/wav", "data": audio_b64}
            ])
            texto_usuario = response_asr.text.strip()
            print(f"--- Usuário disse: {texto_usuario} ---")

        # 2. BUSCAR AS ÚLTIMAS MENSAGENS DO BANCO
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT role, content FROM historico_chat ORDER BY id DESC LIMIT 10")
        linhas = cur.fetchall()
        historico_recente = list(reversed(linhas))
        
        if texto_usuario != "..." and texto_usuario != "":
            cur.execute(
                "INSERT INTO historico_chat (role, content) VALUES (%s, %s)",
                ("user", texto_usuario)
            )
            conn.commit()
        cur.close()
        conn.close()

        # 3. MONTAR O PROMPT
        contexto = ""
        for role, content in historico_recente:
            if role == "user":
                contexto += f"Usuário: {content}\n"
            else:
                contexto += f"LIS: {content}\n"

        prompt_completo = (
            "Você é a LIS (Local Intelligent System), uma assistente virtual acolhedora e cuidadora, projetada para ajudar idosos. "
            "Fale de forma clara, pausada e sempre demonstre empatia e cuidado com a saúde do usuário. "
            "Se o usuário falar algo que claramente não é para você (ruído), fique em silêncio seguido da emoção |ESPANTO. "
            "REGRA DE OURO: Você DEVE terminar TODA resposta com uma barra vertical '|' e uma emoção.\n"
            "IMPORTANTE: Seja extremamente concisa. Limite suas respostas a no máximo 2 ou 3 frases curtas. "
            "Vá direto ao ponto para não cansar o usuário.\n\n"
            "Emoções: ALEGRIA, TRISTEZA, ESPANTO, NERVOSO, EXALTADO ou NEUTRO.\n\n"
            f"INFORMAÇÃO DO AMBIENTE FÍSICO AGORA: Temperatura = {temp_atual}°C, Umidade = {umid_atual}%.\n"
            f"DATA E HORA ATUAIS: Hoje é dia {data_atual}, e agora são exatamente {hora_atual}.\n"
            "Use essa informação caso o usuário pergunte as horas ou a data.\n"
            "REGRAS DE SEGURANÇA PARA A SAÚDE DO IDOSO:\n"
            f"1. Se a temperatura estiver acima de 30°C, alerte sobre o calor, recomende evitar exercícios físicos ao ar livre e beber água.\n"
            f"2. Considere umidade abaixo de 60% como baixa e abaixo de 25% como alerta. Se estiver baixa ou em alerta, recomende beber muita água e usar umidificador.\n"
            "3. Se o clima estiver bom, apenas converse normalmente.\n\n"
            f"HISTÓRICO DA CONVERSA:\n{contexto}\n"
            f"Usuário: {texto_usuario}\n"
            "LIS: "
        )

        # 4. CHAMAR O GEMINI
        model_chat = genai.GenerativeModel('gemini-3.5-flash-lite')
        response = model_chat.generate_content(prompt_completo)
        texto_completo = response.text.strip()
        print(f"--- A LIS respondeu: {texto_completo} ---")
        
        if '|' in texto_completo:
            texto_para_falar, emocao = texto_completo.rsplit('|', 1)
        else:
            texto_para_falar = texto_completo
            emocao = "NEUTRO"

        # 5. TTS VIA EDGE (Microsoft)
        if texto_para_falar.strip() == "":
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
            audio_mp3 = loop.run_until_complete(gerar_audio())
            loop.close()

            if not audio_mp3:
                return jsonify({"status": "erro", "detalhes": "TTS nao retornou audio"}), 500

            audio = AudioSegment.from_file(io.BytesIO(audio_mp3), format="mp3")
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio = audio.low_pass_filter(3000)
            audio = audio - 18
            pcm_16k = audio.raw_data

        # 6. SALVAR NO BANCO
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
        
        # 7. ENVIA PARA O ESP32
        emocao_padded = emocao.ljust(10, ' ').encode('utf-8')[:10]
        payload_final = emocao_padded + pcm_16k

        return app.response_class(payload_final, mimetype='application/octet-stream')
    
    except Exception as e:
        print(f"ERRO NO SERVIDOR: {e}")
        erro_str = str(e)
        
        # Se o erro for de cota do Gemini (429)
        if "429" in erro_str or "quota" in erro_str.lower():
            return jsonify({
                "status": "erro", 
                "detalhes": "Cota de uso da IA excedida. Tente novamente mais tarde."
            }), 429
            
        # Se for qualquer outro erro (500)
        return jsonify({"status": "erro", "detalhes": erro_str}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
