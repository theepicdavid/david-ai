from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub import InferenceClient
from starlette.middleware.sessions import SessionMiddleware

# ===============================
# HARDCODED HUGGING FACE TOKEN
# ===============================
HUGGINGFACE_API_TOKEN = "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"
MODEL = "tiiuae/falcon-7b-instruct"

client = InferenceClient(token=HUGGINGFACE_API_TOKEN)

app = FastAPI()

# ✅ add session middleware (THIS FIXES "PUBLIC CHAT")
app.add_middleware(SessionMiddleware, secret_key="super-secret-key")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    history = request.session.get("chat_history", [])
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "history": history}
    )


@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, message: str = Form(...)):
    history = request.session.get("chat_history", [])

    # build prompt
    prompt = "\n".join(
        [f"User: {h[0]}\nAI: {h[1]}" for h in history]
    )
    prompt += f"\nUser: {message}\nAI:"

    # ✅ FIXED HuggingFace call
    generated = client.text_generation(
        model=MODEL,
        inputs=prompt,
        max_new_tokens=200
    )

    # ✅ FIXED response parsing
    if isinstance(generated, str):
        response = generated[len(prompt):].strip()
    else:
        response = str(generated)

    # save per-user history
    history.append((message, response))
    request.session["chat_history"] = history

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "history": history}
    )
