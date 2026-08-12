import sys
import subprocess
import os
subprocess.check_call([sys.executable, "-m", "pip", "install", "newspaper3k"])
import requests
import json
import re
import spacy
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from collections import Counter
from newspaper import Article, Config as NewspaperConfig
from bs4 import BeautifulSoup as bf
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types

nlp_sm = spacy.load("es_core_news_sm")
nlp_lg = spacy.load("es_core_news_lg")

googleaistudio_api_key = os.environ.get("GOOGLE_AI_API_KEY")
gnews_api_key = os.environ.get("GNEWS_API_KEY")
client = genai.Client(api_key=googleaistudio_api_key)

NEWSPAPER_CONFIG = NewspaperConfig()
NEWSPAPER_CONFIG.request_timeout = 6

PAGINAS_CONTRASTE = [
    "https://www.bbc.com/mundo",
    "https://www.tvn.cl",
    "https://www.df.cl",
    "https://www.biobiochile.cl",
    "https://www.emol.com",
]

# --- Parámetros de optimización (tokens / tiempo) ---
MAX_FUENTES_CONTRASTE = 3        # tope de fuentes a contrastar por noticia
CARACTERES_CONTRASTE_IA = 1200   # recorte de texto enviado a la IA por fuente
MAX_WORKERS_DESCARGA = 6         # descargas de artículos en paralelo

SYSTEM_INSTRUCTION = """
Actúa como un experto en Fact-Checking y análisis de medios de comunicación.
Se te proporcionará una 'noticia_base' y una lista de 'otras_versiones' de la misma noticia.
Compara la noticia base contra TODAS las otras versiones en conjunto, siguiendo estas reglas:
1) Identifica contradicciones fácticas: cifras, fechas, nombres propios, lugares, cargos o declaraciones textuales que difieran entre fuentes.
2) Clasifica cada contradicción según su severidad: CRÍTICA si altera el significado central del hecho (ej: número de víctimas, fecha del evento, protagonista equivocado), o MENOR si es un detalle periférico que no cambia el hecho central (ej: hora exacta, nombre de una calle secundaria).
3) Considera una omisión significativa únicamente cuando la ausencia del dato pueda modificar la interpretación de los hechos principales. Ignora omisiones de contexto secundario, detalles periféricos o estilo.
4) Registra únicamente coincidencias verificables presentes en al menos dos fuentes que respalden la noticia base.
5) No evalúes opiniones, juicios editoriales, tono, estilo narrativo ni interpretaciones.
6) No infieras información que no aparezca explícitamente en los textos. Si un dato no puede contrastarse con ninguna otra fuente, márcalo como no_verificable.
7) Si no encuentras contradicciones, omisiones o coincidencias, devuelve un arreglo vacío [].
8) La conclusión debe basarse exclusivamente en los hallazgos detectados, no en suposiciones.
9) Calcula un puntaje de credibilidad del 0.00 al 100.00 con hasta dos decimales, donde 100.00 significa total consistencia con las otras fuentes y 0.00 significa contradicciones críticas en todos los hechos principales. Este puntaje debe ser realista y gradual, no punitivo: una noticia bien contrastada y sin contradicciones importantes debe quedar entre 85 y 99; una noticia con contraste parcial o alguna omisión menor debe quedar entre 65 y 85; solo múltiples contradicciones CRÍTICAS deben llevarla por debajo de 65.
El puntaje parte de 100.0 y se calcula así: descuenta 15 puntos por cada contradicción CRÍTICA, 4 puntos por cada contradicción MENOR, y 5 puntos por cada omisión significativa.
Suma 3 puntos por cada coincidencia verificada, con un máximo de 15 puntos por coincidencias. El puntaje final no puede ser menor a 0.0 ni mayor a 100.0.
Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional, sin bloques de código markdown, usando esta estructura exacta:
{"contradicciones": [{"tipo": "CRÍTICA o MENOR", "campo": "cifra, fecha, nombre, lugar, declaración u otro", "noticia_base": "<valor en la noticia base>", "otras_fuentes": "<valor en las otras fuentes>", "descripcion": "<explicación breve y objetiva>"}], "omisiones_significativas": [{"dato_omitido": "<qué falta en la noticia base>", "presente_en": "<fuente donde aparece>", "impacto": "<por qué su ausencia modifica la interpretación de los hechos principales>"}], "coincidencias": [{"dato": "<dato confirmado>", "fuentes_que_coinciden": 0}], "resumen": {"total_contradicciones_criticas": 0, "total_contradicciones_menores": 0, "total_omisiones": 0, "total_coincidencias": 0, "nivel_consistencia": "ALTO, MEDIO o BAJO", "puntaje_veracidad": 0.00, "conclusion": "<2 o 3 oraciones objetivas basadas únicamente en los hallazgos detectados>"}}
"""


def keyword_extraction(contenido, titulo, top_n=5):
    def obtener_frecuencia(texto):
        doc = nlp_sm(texto)
        clean_palabras = [
            token.text.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha
        ]
        return Counter(clean_palabras)

    conteo_contenido = obtener_frecuencia(contenido)
    conteo_titulo = obtener_frecuencia(titulo)
    palabras_comunes = set(conteo_contenido.keys()) & set(conteo_titulo.keys())

    puntuacion_comun = {
        palabra: conteo_contenido[palabra] + conteo_titulo[palabra]
        for palabra in palabras_comunes
    }
    resultado = sorted(puntuacion_comun.items(), key=lambda x: x[1], reverse=True)
    return [palabra for palabra, _ in resultado[:top_n]]


def _descargar_articulo(url):
    """Descarga y parsea un artículo UNA sola vez. Antes se descargaba
    hasta 3 veces la misma URL en distintas partes de compute_score."""
    try:
        articulo = Article(url, config=NEWSPAPER_CONFIG)
        articulo.download()
        articulo.parse()
        if articulo.text:
            return url, articulo.text
    except Exception:
        pass
    return url, None


def _buscar_urls_candidatas(keywords):
    if not keywords:
        return []
    pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)

    def revisar_pagina(url):
        encontrados = []
        try:
            html_pag = requests.get(url, timeout=5).text
            soup = bf(html_pag, "html.parser")
            for link in soup.find_all("a", href=True):
                href = link.get("href")
                texto = link.get_text()
                if pattern.search(texto) or pattern.search(href):
                    if href.startswith("/"):
                        href = url.rstrip("/") + href
                    encontrados.append(href)
        except Exception:
            pass
        return encontrados

    saved_urls = []
    with ThreadPoolExecutor(max_workers=len(PAGINAS_CONTRASTE)) as ex:
        for encontrados in ex.map(revisar_pagina, PAGINAS_CONTRASTE):
            for href in encontrados:
                if href not in saved_urls:
                    saved_urls.append(href)
    return saved_urls


def compute_score(noticia):
    keywords = noticia["keywords"]
    doc_original = nlp_lg(noticia["content"])  # un solo parseo, reutilizado abajo

    candidatas = _buscar_urls_candidatas(keywords)

    # Descarga en paralelo, una sola vez por URL (antes: hasta 3 veces c/u)
    contenidos = {}
    if candidatas:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_DESCARGA) as ex:
            futuros = [ex.submit(_descargar_articulo, u) for u in candidatas]
            for fut in as_completed(futuros):
                url, texto = fut.result()
                if texto:
                    contenidos[url] = texto

    # Filtro de similitud semántica
    urls_validas = []
    for url, texto in contenidos.items():
        similitud = doc_original.similarity(nlp_lg(texto))
        if similitud > 0.75:
            urls_validas.append(url)

    # Tope de fuentes -> menos tokens/tiempo en el paso de IA
    urls_validas = urls_validas[:MAX_FUENTES_CONTRASTE]
    noticias_validas = len(urls_validas)

    # --- Análisis matemático: entidades y cifras ---
    entidades_mias = {ent.text.lower() for ent in doc_original.ents if ent.label_ in ("PER", "LOC", "ORG")}
    numeros_mios = {t.text for t in doc_original if t.pos_ == "NUM" or t.like_num}

    entidades_contraste_total = set()
    numeros_contraste_total = set()
    for url in urls_validas:
        doc_v = nlp_lg(contenidos[url])
        entidades_contraste_total |= {ent.text.lower() for ent in doc_v.ents if ent.label_ in ("PER", "LOC", "ORG")}
        numeros_contraste_total |= {t.text for t in doc_v if t.pos_ == "NUM" or t.like_num}

    omisiones_claves = list(entidades_mias - entidades_contraste_total)
    cifras_nuevas = list(numeros_contraste_total - numeros_mios)

    tasa_validacion = (
        len(entidades_mias & entidades_contraste_total) / len(entidades_mias)
        if entidades_mias else 1.0
    )
    total_numeros = len(numeros_mios) + len(cifras_nuevas)
    tasa_numeros_correctos = (len(numeros_mios) / total_numeros) if total_numeros else 1.0

    promedio_coincidencia = (tasa_validacion + tasa_numeros_correctos) / 2
    if noticias_validas == 0:
        veracidad = 68.0
    else:
        veracidad = 50.0 + (promedio_coincidencia * 50.0)
        veracidad = max(0.0, min(100.0, round(veracidad, 2)))

    # --- Contraste con IA: UNA sola llamada por noticia con TODAS las
    # fuentes juntas (antes: una llamada por cada fuente -> N veces el
    # system_instruction y N round-trips) ---
    conclusion_gis = ""
    veracidad_total_gis = None

    if urls_validas:
        otras_versiones = [
            {"fuente": url, "texto": contenidos[url][:CARACTERES_CONTRASTE_IA]}
            for url in urls_validas
        ]
        payload = {
            "noticia_base": noticia["content"][:CARACTERES_CONTRASTE_IA],
            "otras_versiones": otras_versiones,
        }
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.1,
            response_mime_type="application/json",
            max_output_tokens=2000,
        )
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=json.dumps(payload, ensure_ascii=False),
                config=config,
            )
            analisis_data = json.loads(response.text)
            veracidad_total_gis = analisis_data["resumen"]["puntaje_veracidad"]
            conclusion_gis = analisis_data["resumen"]["conclusion"]
        except Exception:
            veracidad_total_gis = None

    if veracidad_total_gis is None:
        # Sin evaluación de IA disponible (no había fuentes o falló la
        # llamada): en vez de castigar con 0.00, se usa el puntaje
        # matemático para que el promedio ponderado no se desplome
        # artificialmente por un problema ajeno a la veracidad real.
        veracidad_total_gis = veracidad
        conclusion_gis = conclusion_gis or (
            "No se encontraron suficientes fuentes externas para contrastar esta "
            "noticia; el puntaje se basa principalmente en el análisis interno "
            "de entidades y cifras, no implica que la noticia sea falsa."
        )

    # Puntaje final en escala 0-100 (porcentaje real de credibilidad):
    # 45% ponderación IA + 55% ponderación matemática.
    porcentaje_tot_veracidad = (veracidad_total_gis * 0.45) + (veracidad * 0.55)
    porcentaje_tot_veracidad = round(max(0.0, min(100.0, porcentaje_tot_veracidad)), 2)

    return porcentaje_tot_veracidad, conclusion_gis


app = Flask(__name__)
CORS(app)
app.json.ensure_ascii = False

categorias_validas = {
    "salud": "health",
    "deportes": "sports",
    "negocios": "business",
    "general": "general",
    "tecnologia": "technology",
    "entretenimiento": "entertainment",
    "ciencia": "science",
    "mundo": "world",
}


def _procesar_articulo(i, art):
    """Arma la noticia 'liviana' para el feed: NO llama a compute_score
    (ni scraping ni IA). Eso se difiere a /verify, y solo se ejecuta si el
    usuario realmente abre esa noticia -> evita gastar tokens de IA y
    tiempo de scraping en las noticias que nadie termina leyendo."""
    titulo = art.get("title", "")
    contenido = art.get("content", "") or ""
    imagen = art.get("image")

    contenido = contenido.replace("\n", " ")
    contenido = re.sub(r"ver también\s*", " ", contenido, flags=re.IGNORECASE)

    try:
        titulo = titulo.encode("latin1").decode("utf-8")
    except Exception:
        pass
    try:
        contenido = contenido.encode("latin1").decode("utf-8")
    except Exception:
        pass

    keywords = keyword_extraction(contenido, titulo)

    return {
        "id": i + 1,
        "title": titulo,
        "image": imagen,
        "content": contenido,
        "keywords": keywords,
    }


@app.route("/process", methods=["GET"])
def obtener_noticias():
    """Endpoint rápido para el feed/listado. No corre scraping ni IA,
    así que no gasta tokens: solo arma id/title/image/content/keywords
    para las 15 noticias. El score se calcula después, por noticia,
    en /verify."""
    category_user = request.args.get("category", "general")
    category = categorias_validas.get(category_user.lower(), "general")
    url = f"https://gnews.io/api/v4/top-headlines?category={category}&lang=es&country=cl&max=15&apikey={gnews_api_key}"

    try:
        response = requests.get(url)
        response.encoding = "utf-8"
        data = response.json()
        articles = data["articles"]

        global_news = [_procesar_articulo(i, art) for i, art in enumerate(articles)]

        return Response(
            json.dumps({"status": "ok", "datos": global_news}, ensure_ascii=False),
            content_type="application/json; charset=utf-8",
        )
    except Exception as e:
        return jsonify({"status": "error", "mensaje": str(e)}), 500


@app.route("/verify", methods=["POST"])
def verificar_noticia():
    """Endpoint 'bajo demanda': la app (App Inventor) lo llama SOLO cuando
    el usuario abre una noticia puntual. Recibe el content y keywords que
    ya devolvió /process (no requiere que el servidor guarde nada en
    memoria) y ahí sí corre el scraping + análisis matemático + IA.

    Body JSON esperado:
    {
        "content": "<texto de la noticia>",
        "keywords": ["palabra1", "palabra2", ...]
    }
    """
    body = request.get_json(silent=True) or {}
    contenido = (body.get("content") or "").strip()
    keywords = body.get("keywords") or []

    if not contenido:
        return jsonify({"status": "error", "mensaje": "Falta 'content' en el body"}), 400

    try:
        noticia = {"content": contenido, "keywords": keywords}
        score, conclusion = compute_score(noticia)
        return jsonify({"status": "ok", "score": score, "conclusion": conclusion})
    except Exception as e:
        return jsonify({"status": "error", "mensaje": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
