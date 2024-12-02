import requests
import gradio as gr
from flask import Flask, request, jsonify
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model and tokenizer
with open('classification_model', 'rb') as f:
    classified_model = pickle.load(f)

with open('tokenized_data', 'rb') as f:
    tokenizer = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Define the Gradio function
def detect_cyberbullying(comment):
    try:
        # Tokenize the input text using the loaded tokenizer
        sequences = tokenizer.texts_to_sequences([comment])  # Convert text to sequences of integers

        # Pad the sequences to ensure uniform input size
        padded_data = pad_sequences(sequences, maxlen=150, padding='pre')  # Ensure max_len is consistent with your model's training

        # Predict using the model
        prediction = classified_model.predict(padded_data)

        # Return a user-friendly result based on the model's prediction
        if prediction[0][0] > 0.5:  # Access the first element of the prediction array
            return "<strong style='color: red;'>🚨 Warning: Cyberbullying Detected!\n" + "</strong>"
        else:
            return "<strong style='color: green;'>✅ Safe: No Cyberbullying Detected.\n" + "</strong>"
    except Exception as e:
        return f"<strong style='color: orange;'>Error: {str(e)}</strong>"

# Gradio Interface (with Enhanced Styles)
with gr.Blocks(css="""
    body {
        background: linear-gradient(135deg, #1B0030, #3A0078);
        font-family: 'Roboto', sans-serif;
        color: #E0E0E0;
    }
    .gradio-container {
        text-align: center;
        padding: 40px;
        max-width: 800px;
        margin: auto;
        box-shadow: 0 0 15px rgba(128, 0, 255, 0.8); /* Glowing purple shadow */
        border-radius: 20px;
        background-color: rgba(20, 20, 60, 0.8);
        position: relative;
        overflow: hidden;
    }
    .gradio-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        box-shadow: 0 0 15px rgba(128, 0, 255, 0.8); /* Glowing purple shadow */
        opacity: 0.1;
        z-index: -1;
    }
    .gr-row {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .gr-column {
        padding: 20px;
    }
    .comment-box {
        font-size: 16px;
        padding: 12px;
        border-radius: 12px;
        border: 2px solid #8000FF;
        background-color: rgba(20, 20, 60, 0.8);
        color: #FFFFFF;
        box-shadow: 0 0 15px rgba(128, 0, 255, 0.8); /* Glowing purple shadow */
        animation: fadeIn 1.5s ease-in;
        text-align: center;
    }
    h1 {
        font-size: 42px;
        color: #A566FF;
        font-weight: bold;
        text-align: center;
        text-shadow: 0 0 30px rgba(0, 255, 255, 0.8), 0 0 60px rgba(0, 255, 255, 0.6); /* Brighter cyan glow */
        animation: glow 2s infinite alternate;
    }
    @keyframes glow {
        from {
            text-shadow: 0 0 30px rgba(0, 255, 255, 0.8), 0 0 60px rgba(0, 255, 255, 0.6);
        }
        to {
            text-shadow: 0 0 40px rgba(0, 255, 255, 2), 0 0 80px rgba(0, 255, 255, 0.8);
        }
    }
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    h3 {
        color: #B889FF;
        font-size: 20px;
        text-align: center;
    }
    .result-text {
        font-size: 18px;
        font-weight: bold;
        color: #D9B3FF;
    }
    .footer-text {
        font-size: 14px;
        color: #B889FF;
        margin-top: 20px;
    }
    .gr-button {
        background: linear-gradient(135deg, #8000FF, #4600FF);
        color: white;
        border-radius: 10px;
        padding: 12px;
        font-size: 18px;
        box-shadow: 0 0 20px rgba(128, 0, 255, 0.5);
        transition: all 0.3s;
        animation: fadeIn 1.5s ease-in;
    }
    .gr-button:hover {
        background: linear-gradient(135deg, #4600FF, #8000FF);
        transform: scale(1.05);
    }
""") as interface:
    # Header Section
    gr.Markdown(
        """
        # ⚖️ Cyberbullying Detection System  
        ### Analyze comments for harmful language and prevent online abuse.
        """
    )
    
    # Two-column Layout
    with gr.Row():
        # Input Column
        with gr.Column():
            gr.Markdown("### 💬 Enter the Comment:")
            comment_input = gr.Textbox(
                label="Comment",
                placeholder="Type a comment to analyze...",
                lines=4,
                elem_classes="comment-box"
            )
            analyze_button = gr.Button("🚀 Analyze Comment")
        
        # Output Column
        with gr.Column():
            gr.Markdown("### 🧲 Detection Result:")
            result_output = gr.HTML(
                label="Analysis Result",
                
                elem_classes="comment-box"
            )
    
    # Footer Section
    gr.Markdown(
        """
        ---
        **🔒 Note**: This tool is designed to assist in detecting harmful language.  
        """
    )

    # Link button, input, and output
    analyze_button.click(detect_cyberbullying, inputs=comment_input, outputs=result_output)

# Run Flask alongside Gradio
@app.route("/")
def home():
    return "Flask & Gradio deployed!"

interface.launch(share=True, server_name="0.0.0.0", server_port=7860)  # Use a different port, e.g., 7860

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
