import gradio as gr
import requests

FLASK_BACKEND_URL = "http://127.0.0.1:5000/classify"  # Update with your Flask backend URL

def classify(comment):
    try:
        response = requests.post(FLASK_BACKEND_URL, json={"comment": comment})
        data = response.json()
        if "result" in data:
            result = data["result"]
            if result == "Cyberbullying":
                return f"🚫 {result}"
            else:
                return f"✅ {result}"
        return "Error processing comment", "gray"
    except Exception as e:
        return f"Error: {str(e)}", "gray"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🌐 Cyberbullying Detection System
        **Promoting Positive Online Interactions**
        > Enter a comment below to check if it's cyberbullying or not.

        <style>
            .result-box {
                color: white;
                background-color: gray;
                padding: 10px;
                border-radius: 5px;
            }
        </style>
        """
    )

    with gr.Row():
        with gr.Column():
            comment_input = gr.Textbox(
                lines=3,
                placeholder="💬 Type your comment here...",
                label="Your Comment",
            )
            submit_button = gr.Button("🔍 Detect", variant="primary")

        with gr.Column():
            result_output = gr.Textbox(label="Classification Result", elem_id="result-box")  # Apply custom ID for styling

    gr.Markdown("### 💡 Examples")
    examples = gr.Examples(
        examples=[
            "You're amazing, keep it up! 😊",
            "You're the worst person I've ever seen! 😠",
        ],
        inputs=[comment_input],
        outputs=[result_output],
        fn=classify,
        cache_examples=True,
    )

    submit_button.click(classify, inputs=[comment_input], outputs=[result_output])

    gr.Markdown(
        """
        ---
        **💖 Promoting Kindness**
        """
    )

if __name__ == "__main__":
    demo.launch()
