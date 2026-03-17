from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub import InferenceClient

# ===============================
# HARDCODED HUGGING FACE TOKEN
# ===============================
HUGGINGFACE_API_TOKEN = "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"  # <-- replace with your token
MODEL = "tiiuae/falcon-7b-instruct"          # <-- replace with your model

client = InferenceClient(token=HUGGINGFACE_API_TOKEN)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

chat_history = []

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "history": chat_history})

@app.post("/chat", response_class=HTMLResponse)
def chat(request: Request, message: str = Form(...)):
    global chat_history
    # Combine history + new message
    prompt = "\n".join([f"User: {h[0]}\nAI: {h[1]}" for h in chat_history])
    prompt += f"\nUser: {message}\nAI:"

    output = client.text_generation(model=MODEL, inputs=prompt, max_new_tokens=200)
    response = output[0].generated_text.strip()

    chat_history.append((message, response))
    return templates.TemplateResponse("index.html", {"request": request, "history": chat_history})