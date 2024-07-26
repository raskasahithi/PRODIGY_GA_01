from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Initialize FastAPI app
app = FastAPI()

# Load model and tokenizer
model_path = './fine-tuned-gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")


class TextRequest(BaseModel):
    prompt: str
    max_length: int = 50


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate")
async def generate(request: TextRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors='pt', truncation=True, padding=True)
        outputs = model.generate(inputs['input_ids'], max_length=request.max_length, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
