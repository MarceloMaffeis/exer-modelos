from __future__ import annotations

from typing import Any

import streamlit as st
from PIL import Image, UnidentifiedImageError
from transformers import pipeline


st.set_page_config(
	page_title="Leitor de Imagens",
	page_icon="📷",
	layout="centered",
)

st.markdown(
	"""
	<style>
	:root {
		--ink: #172a2a;
		--muted: #58716b;
		--paper: #f4f7f2;
		--accent: #c94f35;
		--line: #d7e1d8;
	}
	.stApp { background: var(--paper); color: var(--ink); }
	.block-container { max-width: 860px; padding-top: 3rem; }
	h1 { color: var(--ink); letter-spacing: 0; font-size: 2.7rem; }
	.intro { color: var(--muted); font-size: 1.08rem; margin-bottom: 1.8rem; }
	[data-testid="stFileUploader"] {
		background: #fff;
		border: 1px solid var(--line);
		border-radius: 8px;
		padding: 1rem;
	}
	.stButton > button[kind="primary"] {
		background: var(--accent);
		border: 0;
		border-radius: 6px;
		color: #fff;
		font-weight: 650;
		min-height: 2.8rem;
	}
	.stButton > button[kind="primary"]:hover { background: #a93d29; }
	[data-testid="stAlert"] { border-radius: 6px; }
	</style>
	""",
	unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[Any, Any]:
	"""Carrega os modelos uma vez e os reutiliza entre interações."""
	captioner = pipeline(
		"image-to-text",
		model="Salesforce/blip-image-captioning-base",
	)
	translator = pipeline(
		"translation",
		model="Helsinki-NLP/opus-mt-en-ROMANCE",
	)
	return captioner, translator


st.title("O que há nesta imagem?")
st.markdown(
	'<p class="intro">Envie uma imagem e receba uma descrição gerada por inteligência artificial.</p>',
	unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
	"Escolha uma imagem",
	type=["png", "jpg", "jpeg", "webp"],
	help="Formatos aceitos: PNG, JPG, JPEG e WEBP.",
)

if uploaded_file is not None:
	try:
		image = Image.open(uploaded_file).convert("RGB")
	except (UnidentifiedImageError, OSError):
		st.error("Não foi possível abrir esse arquivo como imagem. Tente outro arquivo.")
	else:
		st.image(image, caption=uploaded_file.name, use_container_width=True)

		if st.button("Descrever imagem", type="primary", use_container_width=True):
			try:
				with st.spinner(
					"Carregando os modelos. Na primeira execução, isso pode levar alguns minutos..."
				):
					captioner, translator = load_models()
			except Exception as error:
				st.error(
					"Não foi possível carregar os modelos. Na primeira execução, eles são "
					"baixados do Hugging Face; a rede pode estar lenta ou bloquear os arquivos."
				)
				st.code(f"{type(error).__name__}: {error}", language="text")
			else:
				try:
					with st.spinner("Analisando a imagem..."):
						caption_result = captioner(image, max_new_tokens=60)
						english_caption = caption_result[0]["generated_text"].strip()
						translation_result = translator(
							f">>pt<< {english_caption}",
							max_new_tokens=100,
						)
						portuguese_caption = translation_result[0]["translation_text"].strip()

					st.subheader("Descrição")
					st.success(portuguese_caption)
				except Exception as error:
					st.error("Os modelos foram carregados, mas a análise da imagem falhou.")
					st.code(f"{type(error).__name__}: {error}", language="text")
