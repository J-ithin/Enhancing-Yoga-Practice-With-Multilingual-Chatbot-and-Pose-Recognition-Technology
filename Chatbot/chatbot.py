import streamlit as st
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Set page configuration
st.set_page_config(page_title="Yoga Chatbot", page_icon="🧘", layout="wide")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('intents.json', 'r', errors="ignore") as json_data:
    intents = json.load(json_data)

FILE = "yoga.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"


def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I do not understand..."


# Streamlit UI
st.title("🧘 Yoga Chatbot")
st.write("Chat with Sam, your yoga assistant! Type 'quit' to exit.")

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(f"You: {chat['user']}")
    with st.chat_message("assistant"):
        st.write(f"{bot_name}: {chat['bot']}")

# Input form
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("Your message:", key="user_input")
    submit_button = st.form_submit_button(label="Send")

# Process user input
if submit_button and user_input:
    if user_input.lower() == "quit":
        st.session_state.chat_history.append({"user": user_input, "bot": "Goodbye!"})
        st.rerun()
    else:
        response = get_response(user_input)
        st.session_state.chat_history.append({"user": user_input, "bot": response})
        st.rerun()