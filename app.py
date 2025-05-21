# 📦 Çoklu LLM Karşılaştırma Arayüzü
# Modeller: LLaMA 3, Zephyr, Nous Hermes, DeepSeek, OpenAI GPT-3.5

import gradio as gr
import asyncio
import time
import logging
from huggingface_hub import AsyncInferenceClient
from openai import AsyncOpenAI

# 🌐 Konsol loglama
logging.basicConfig(level=logging.INFO)

# 💬 Ortak sistem mesajı (her modele gönderilen bağlamsal giriş)
system_prompt = (
    "Yapay zekanın ve büyük dil modellerinin (LLM) gelecekteki potansiyel kullanım alanlarını keşfeden bir makale yazın. "
    "Bu teknolojilerin toplumsal etkilerini, etik sorunlarını ve olası yenilikçi uygulamalarını tartışın. "
    "Ayrıca, bu teknolojilerin insan yaşamını nasıl dönüştürebileceği hakkında öngörülerde bulunun."
)

# 🧠 Model listesi (başlık, model_id)
ALL_MODELS = [
    ("LLaMA 3 - 70B", "meta-llama/Meta-Llama-3-70B-Instruct"),
    ("Nous Hermes 2 - Mixtral", "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"),
    ("Zephyr ORPO", "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1"),
    ("DeepSeek R1", "deepseek-ai/DeepSeek-R1"),
    ("OpenAI GPT-3.5", "openai/gpt-3.5-turbo"),  # Özel işleniyor
]

# 🔁 OpenAI modelini çağır
async def call_openai(prompt, model_name, openai_api_key):
    try:
        client = AsyncOpenAI(api_key=openai_api_key)
        start_time = time.time()
        logging.info(f"OpenAI çağrılıyor: {model_name}")

        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=800,
        )
        result = response.choices[0].message.content
        duration = time.time() - start_time
        return f"### {model_name}\n\n{result}\n\n⌛ Yanıt süresi: {duration:.1f} saniye"
    except Exception as e:
        logging.error(f"{model_name} hatası: {str(e)}")
        return f"❌ **{model_name} hatası:** {str(e)}"

# 🔁 Hugging Face modelleri çağır
async def call_hf(model_id, model_name, prompt, hf_token):
    try:
        client = AsyncInferenceClient(model=model_id, token=hf_token)
        start_time = time.time()
        logging.info(f"Hugging Face modeli çağrılıyor: {model_name}")

        response = await client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_new_tokens=500,
            temperature=0.7,
        )
        duration = time.time() - start_time
        result = response.choices[0].message["content"]
        return f"### {model_name}\n\n{result}\n\n⌛ Yanıt süresi: {duration:.1f} saniye"
    except Exception as e:
        logging.error(f"{model_name} hatası: {str(e)}")
        return f"❌ **{model_name} hatası:** {str(e)}"

# 🚀 Tüm seçilen modelleri aynı anda çalıştır
async def run_selected(prompt, hf_token, openai_token, selected_models):
    if not prompt or (not hf_token and not openai_token):
        return ["⚠️ Prompt ve tokenlar gerekli."]

    model_map = dict(ALL_MODELS)
    tasks = []

    for name in selected_models:
        if name == "OpenAI GPT-3.5":
            tasks.append(call_openai(prompt, name, openai_token))
        else:
            model_id = model_map[name]
            tasks.append(call_hf(model_id, name, prompt, hf_token))

    results = await asyncio.gather(*tasks)
    return results

# 🎨 Gradio Arayüzü
with gr.Blocks(theme="NoCrypt/miku") as demo:
    gr.Markdown("## 🧠 Çoklu LLM Karşılaştırıcı (Zephyr, LLaMA, DeepSeek, Nous, OpenAI)")

    with gr.Row():
        prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Bir soru ya da görev girin...")

    with gr.Row():
        hf_token = gr.Textbox(label="Hugging Face Token", type="password", placeholder="HF Token (LLaMA, Zephyr, DeepSeek)")
        openai_token = gr.Textbox(label="OpenAI API Key", type="password", placeholder="OpenAI API Key")

    model_selector = gr.CheckboxGroup(
        choices=[name for name, _ in ALL_MODELS],
        label="Kullanılacak Modelleri Seçin",
        value=[name for name, _ in ALL_MODELS],
    )

    generate_btn = gr.Button("🚀 Modelleri Çalıştır")
    outputs = [gr.Markdown() for _ in ALL_MODELS]

    generate_btn.click(
        fn=run_selected,
        inputs=[prompt, hf_token, openai_token, model_selector],
        outputs=outputs,
    )

# Uygulamayı başlat
demo.launch()
