from character_chatbot import CharacterChatBot  # Asegúrate de que este nombre coincida con tu archivo

# Ruta al modelo entrenado o publicado en Hugging Face (o carpeta local)
model_path = "final_ckpt"  # o el nombre en Hugging Face si ya lo subiste

# Instancia del chatbot
chatbot = CharacterChatBot(model_path=model_path)

# Historial de ejemplo (puede estar vacío)
history = [
    {"user": "Hola Naruto, ¿cómo estás?", "naruto": "¡Estoy listo para entrenar, dattebayo!"},
    {"user": "¿Qué piensas de Sasuke?", "naruto": "Es mi amigo... aunque también mi rival."}
]

# Mensaje nuevo del usuario
user_input = "¿Cuál es tu jutsu favorito?"

# Obtener la respuesta del chatbot
response = chatbot.chat(user_input, history)

print("🗨️ Naruto responde:")
print(response)
