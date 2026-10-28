from __future__ import annotations

from typing import Any

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from transformers import pipeline


def annotate_detections(
	image: Image.Image,
	detections: list[dict[str, Any]],
	label_names: dict[str, str],
) -> Image.Image:
	annotated = image.copy()
	draw = ImageDraw.Draw(annotated)
	font_size = max(12, min(24, image.width // 45))
	try:
		font = ImageFont.truetype("DejaVuSans.ttf", font_size)
	except OSError:
		font = ImageFont.load_default()

	colors = ["#c94f35", "#147d77", "#315ca8", "#a34886", "#8a6518"]
	for index, detection in enumerate(detections):
		box = detection["box"]
		x_min = max(0, min(image.width - 1, int(box["xmin"])))
		y_min = max(0, min(image.height - 1, int(box["ymin"])))
		x_max = max(x_min + 1, min(image.width - 1, int(box["xmax"])))
		y_max = max(y_min + 1, min(image.height - 1, int(box["ymax"])))
		color = colors[index % len(colors)]
		label = f"{label_names[detection['label']]} {detection['score']:.0%}"
		text_box = draw.textbbox((0, 0), label, font=font)
		text_width = text_box[2] - text_box[0]
		text_height = text_box[3] - text_box[1]
		padding = 4
		label_top = max(0, y_min - text_height - padding * 2)

		draw.rectangle((x_min, y_min, x_max, y_max), outline=color, width=3)
		draw.rectangle(
			(x_min, label_top, x_min + text_width + padding * 2, label_top + text_height + padding * 2),
			fill=color,
		)
		draw.text((x_min + padding, label_top + padding), label, fill="white", font=font)
	return annotated


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
def load_models() -> tuple[Any, Any, Any]:
	"""Carrega os modelos uma vez e os reutiliza entre interações."""
	captioner = pipeline(
		"image-to-text",
		model="Salesforce/blip-image-captioning-base",
	)
	object_detector = pipeline(
		"object-detection",
		model="facebook/detr-resnet-50",
	)
	translator = pipeline(
		"translation",
		model="Helsinki-NLP/opus-mt-en-ROMANCE",
	)
	return captioner, object_detector, translator


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
					captioner, object_detector, translator = load_models()
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
						detections = object_detector(image, threshold=0.5)

						label_names: dict[str, str] = {}
						labels = list(dict.fromkeys(item["label"] for item in detections))
						if labels:
							translated_labels = translator(
								[f">>pt<< {label}" for label in labels],
								max_new_tokens=40,
							)
							label_names = {
								label: result["translation_text"].strip()
								for label, result in zip(labels, translated_labels)
							}

					annotated_image = annotate_detections(image, detections, label_names)
					st.image(
						annotated_image,
						caption="Objetos identificados",
						use_container_width=True,
					)
					st.subheader("Descrição")
					st.success(portuguese_caption)
					st.subheader("Objetos encontrados")
					if detections:
						for item in detections:
							st.write(
								f"{label_names[item['label']]} "
								f"({item['score']:.0%} de confiança)"
							)
					else:
						st.info("Nenhum objeto foi identificado com confiança acima de 50%.")
				except Exception as error:
					st.error("Os modelos foram carregados, mas a análise da imagem falhou.")
					st.code(f"{type(error).__name__}: {error}", language="text")
