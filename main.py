from fastapi import FastAPI
from transformers import MarianTokenizer, MarianMTModel

app = FastAPI()

model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

@app.get("/")
def read_root():
    return {"message": "Translate API Working"}

@app.get("/translate")
def translate(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    result = tokenizer.decode(translated[0], skip_special_tokens=True)
    return {"translation": result}


# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
# import asyncio
# import torch

# app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"message": "Hello from Translate API!"}

# model_name = "facebook/m2m100_418M"
# tokenizer = M2M100Tokenizer.from_pretrained(model_name)
# model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# if torch.cuda.is_available():
#     model = model.to("cuda")

# class TranslateRequest(BaseModel):
#     data: dict
#     toLanguages: list[str]

# @app.post("/translate")
# async def translate(request: TranslateRequest):
#     text = request.data.get("message", "")
#     if not text:
#         raise HTTPException(status_code=400, detail="No message provided")
#     to_languages = request.toLanguages

#     async def translate_to_lang(lang):
#         tokenizer.src_lang = "en"
#         tokenizer.tgt_lang = lang
#         inputs = tokenizer(text, return_tensors="pt")
#         if torch.cuda.is_available():
#             inputs = {k: v.to("cuda") for k, v in inputs.items()}
#         outputs = model.generate(**inputs)
#         translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return lang, translated

#     tasks = [asyncio.create_task(translate_to_lang(lang)) for lang in to_languages]
#     results = await asyncio.gather(*tasks)

#     return {lang: translation for lang, translation in results}
