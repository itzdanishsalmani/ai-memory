from dotenv import load_dotenv
import logging
from mem0 import Memory
import os
from openai import OpenAI
import json

logging.basicConfig(level=logging.WARNING)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

config = {
    "version":"v1.1",
    "embedder":{
        "provider":"gemini",
        "config":{
            "api_key":GOOGLE_API_KEY, "model":"models/gemini-embedding-001"
        }
    },
    "llm":{
        "provider":"openai",
        "config":{
            "api_key":GOOGLE_API_KEY,
            "model":"gemini-2.5-flash",
            "openai_base_url":"https://generativelanguage.googleapis.com/v1beta/openai/",
            "max_tokens":8192,
        }
    },
    "vector_store":{
        "provider":"qdrant",
        "config":{
            "host":"localhost",
            "port":6333,       
                        "collection_name": "mem0_gemini",
                 "embedding_model_dims": 768,
        }
    }
}

mem_client = Memory.from_config(config)



while True:
    user_query = input(">")

    search_memory = mem_client.search(query=user_query, filters={"user_id": "user_1"})

    memories = [
        f"{memory.get('id')} {memory.get('memory')}"
        for memory in search_memory.get("results", [])
    ]

    print("memories", memories)

    SYSTEM_PROMPT = f"""
    Here is the context about user,
    {json.dumps(memories)}
    """

    response = client.chat.completions.create(
            model="gemini-2.5-flash",
        messages =[
            { "role" : "system", "content" : SYSTEM_PROMPT },
            { "role" : "user", "content" : user_query }
    ]
)

    ai_response = response.choices[0].message.content

    print("AI response", ai_response)
    result = mem_client.add(
        [
            { "role" : "user", "content" : user_query },
            { "role" : "assistant", "content" : ai_response }
        ],
        user_id="user_1",
    )

    print("Saved:", result)