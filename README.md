# Leitor de Imagens

Aplicação em Streamlit que recebe uma imagem, gera uma legenda com BLIP,
detecta objetos com DETR e traduz a legenda e os rótulos para português usando
modelos da biblioteca Transformers.

## Requisitos

- Python 3.10 ou superior
- Conexão com a internet na primeira execução para baixar os modelos
- Memória suficiente para carregar os modelos de visão e tradução

## Instalação e execução

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

No Windows, ative o ambiente virtual com `.venv\Scripts\activate`.

Selecione uma imagem PNG, JPG, JPEG ou WEBP e clique em **Descrever imagem**.
Os modelos são baixados na primeira solicitação e mantidos em cache durante a
execução do aplicativo.
