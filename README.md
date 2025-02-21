🦙 LLM - Meta Llama 3.1 (8B) 🚀



🔍 Sobre o Projeto
Este repositório fornece um ambiente otimizado para executar o Meta-Llama-3.1-8B-Instruct, um modelo de linguagem de alto desempenho. Com suporte a CUDA, bitsandbytes, accelerate e otimização de memória, você pode rodar inferência de maneira eficiente em sua GPU. 💡🔥

🚀 Características

✅ Modelo: Meta-Llama-3.1-8B-Instruct 🦙
✅ Frameworks: PyTorch + Transformers + Accelerate + bitsandbytes
✅ Otimizações: Redução de memória com 4-bit quantization
✅ Compatibilidade: GPUs NVIDIA com suporte a CUDA
✅ Inferência rápida e eficiente

🛠️ Configuração e Instalação

🔹 1. Clone o Repositório

 git clone https://github.com/IsaiasSaraiva/LLM---meta-llama
 cd LLM-meta-llama

🔹 2. Crie um Ambiente Virtual (Recomendado)

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

🔹 3. Instale as Dependências

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes

🔹 4. Baixe o Modelo

from transformers import AutoModelForCausalLM, AutoTokenizer

id_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(id_model, device_map="auto", load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(id_model)

🔹 5. Rode um Exemplo

input_text = "Explique a teoria da relatividade de forma simples."
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_length=200)
print(tokenizer.decode(output[0], skip_special_tokens=True))

