import os
import requests
import json
import re
import time
import spacy 
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from collections import Counter
from newspaper import Article  
from bs4 import BeautifulSoup as bf
from google import genai
from google.genai import types

googleaistudio_api_key = os.environ.get("GOOGLE_AI_API_KEY")
gnews_api_key = os.environ.get("GNEWS_API_KEY")

nlp = spacy.load("es_core_news_sm")
def keyword_extraction (contenido, titulo, top_n=5):
    def obtener_frecuencia (texto):
        doc = nlp(texto)
        clean_palabras = []
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha:
                 clean_palabras.append(token.text.lower())
            
        return Counter(clean_palabras)
    conteo_contenido = obtener_frecuencia(contenido)
    conteo_titulo = obtener_frecuencia(titulo)

    palabras_comunes = set(conteo_contenido.keys()) & set(conteo_titulo.keys())

    puntuacion_comun = {}
    for palabra in palabras_comunes:
        puntuacion_comun[palabra] = conteo_contenido[palabra] + conteo_titulo[palabra]
    
    resultado = sorted(puntuacion_comun.items(), key=lambda x: x[1], reverse =True)

    return [palabra for palabra, puntaje in resultado [:top_n]]

def verificar_simlitud(noticia_base, noticia_contraste):
    client = genai.Client(api_key=googleaistudio_api_key)

    config_filtro = types.GenerateContentConfig(
        system_instruction="""
            Estas dos noticias ya fueron preseleccionadas por un filtro de similitud semántica (coincidencia de keywords y embeddings). Tu tarea es la verificación final: confirmar si realmente describen el mismo acontecimiento específico, descartando falsos positivos típicos de similitud semántica (mismo tema pero distinto evento, o distintas etapas/reacciones del mismo proceso). Criterios obligatorios: 
            1. Deben coincidir el hecho principal, los actores involucrados y la acción central del acontecimiento. 
            2. Diferencias de redacción, cifras menores o nivel de detalle no afectan la coincidencia si el hecho es el mismo. 
            3. Si las noticias describen etapas distintas de un mismo proceso (anuncio, debate, aprobación, implementación o reacción pública), responde NO. 
            4. Si una noticia es una actualización, consecuencia o reacción de un tercero a la otra, responde NO. 
            5. Si los hechos ocurrieron en momentos o instancias diferentes, aunque pertenezcan al mismo tema general, responde NO.
            6. Si existe cualquier duda razonable sobre si se trata del mismo acontecimiento exacto, responde NO.
            7. Evalúa únicamente identidad del evento concreto, no similitud temática. 
            Responde únicamente con la palabra "SI" o la palabra "NO", en mayúsculas, sin puntuación ni texto adicional.
        """,
        temperature= 0.1
    )
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=f"noticia base: {noticia_base}\n\nnoticia contraste {noticia_contraste}",
            config=config_filtro
        )

        resultado = response.text.strip().upper()

        return resultado
    except Exception:
        return False

def compute_score(noticia) :
    noticia_base = noticia["content"]
    paginas_contraste = ["https://www.bbc.com/mundo",
                             "https://www.tvn.cl",
                             "https://www.df.cl",
                             "https://www.biobiochile.cl",
                             "https://www.emol.com"]
    
    saved_urls = []
    keywords = noticia["keywords"]
    text_noticia = nlp(noticia["content"])
    for url  in paginas_contraste:
        try:
            html_pag = requests.get(url, timeout=5).text
            soup = bf(html_pag, "html.parser")
            pattern = re.compile("|".join(keywords), re.IGNORECASE)
            response = soup.find_all("a", href=True)

            for link in response: 
                 href = link.get("href")
                 link_texto = link.get_text()
             
                 if pattern.search(link_texto) or pattern.search(href):
                   if href.startswith("/"):
                         href = url.rstrip("/") + href
                   if href not in saved_urls:
                        try:
                             html_candidato = requests.get(href, timeout=5).text
                             soup_candidato = bf(html_candidato, "html.parser")
                             texto_candidato = soup_candidato.get_text()

                             doc_candidato = nlp(texto_candidato)
                             puntaje_simlitud = text_noticia.similarity(doc_candidato)

                             if puntaje_simlitud >= 0.75:
                                 saved_urls.append(href)
                        except Exception:
                            pass
                             

        except Exception:
            continue
    noticias_validas = 0
    for url in saved_urls[:]:
            try:
                articulo = Article(url)
                articulo.download()
                articulo.parse()

                frnew_content = articulo.text
                frnew_title = articulo.title

                doc_nuevo = nlp(frnew_content)
                similitud = text_noticia.similarity(doc_nuevo)

                if similitud > 0.75:

                    time.sleep(12) 

                    mismo_evento = verificar_simlitud(noticia_base, frnew_content)

                    if mismo_evento == "SI":
                        noticias_validas += 1
                else:
                    saved_urls.remove(url) 

            except Exception:
                if url in saved_urls:
                    saved_urls.remove(url)
                continue

    doc_sm_original = text_noticia
    
    entidades_mias = set([ent.text.lower() for ent in doc_sm_original.ents if ent.label_ in ["PER", "LOC", "ORG"]])
    numeros_mios = set([token.text for token in doc_sm_original if token.pos_ == "NUM" or token.like_num])
    
    entidades_contraste_total = set()
    numeros_contraste_total = set()
    
    for url_valida in saved_urls:
        try:
            articulo_v = Article(url_valida)
            articulo_v.download()
            articulo_v.parse()

            doc_v = nlp(articulo_v.text)
            for ent in doc_v.ents:
                if ent.label_ in ["PER", "LOC", "ORG"]:
                    entidades_contraste_total.add(ent.text.lower())
            
            for token in doc_v:
                if token.pos_ == "NUM" or token.like_num:
                    numeros_contraste_total.add(token.text)
        except Exception:
            continue

    omisiones_claves = list(entidades_mias - entidades_contraste_total)
    cifras_nuevas = list(numeros_contraste_total - numeros_mios)

    def hay_coincidencia_parcial(entidad, set_contraste):
        palabras_entidad = entidad.split()
        for otra in set_contraste:
            if any(palabra in otra for palabra in palabras_entidad):
                return True
        return False

    entidades_compartidas = [e for e in entidades_mias if hay_coincidencia_parcial(e, entidades_contraste_total)]
    tasa_validacion = len(entidades_compartidas) / len(entidades_mias) if entidades_mias else 1.0

    total_numeros = len(numeros_mios) + len(cifras_nuevas)
    if total_numeros > 0:
        tasa_numeros_correctos = len(numeros_mios) / total_numeros
    else:
        tasa_numeros_correctos = 1.0

    promedio_coincidencia = (tasa_validacion * 0.4 + tasa_numeros_correctos * 0.6)

    veracidad = 3.0 + (promedio_coincidencia * 7.0)

    if noticias_validas == 0:
        veracidad = 1.0
    veracidad = max(0.00, min(10.00, round(veracidad,2)))

    noticia["reporte_auditoria"] = {
        "score_veracidad": veracidad,
        "omisiones_detectadas": omisiones_claves,
        "cifras_no_verificadas": cifras_nuevas,
     }
    

    client = genai.Client(api_key= googleaistudio_api_key)
 
    versiones_contraste = []
    for url in saved_urls:
        try:
            articulo_c = Article(url)
            articulo_c.download()
            articulo_c.parse()
            versiones_contraste.append({
                "fuente": url,
                "texto": articulo_c.text
            })
            noticia_contraste = articulo_c.text
        except Exception:
            continue
  
    conclusion_gis = " "
    veracidad_gis = 0.00

    if len(versiones_contraste) > 0 :
        config = types.GenerateContentConfig(
            system_instruction="""
                    Actúa como un experto en Fact-Checking y análisis de medios de comunicación.
                    Se te proporcionará una 'noticia_base' y una lista de 'otras_versiones' de la misma noticia.
                    Compara la noticia base contra cada una de las otras versiones siguiendo estas reglas:
                    1) Identifica contradicciones fácticas: cifras, fechas, nombres propios, lugares, cargos o declaraciones textuales que difieran entre fuentes.
                    2) Clasifica cada contradicción según su severidad: CRÍTICA si altera el significado central del hecho (ej: número de víctimas, fecha del evento, protagonista equivocado), o MENOR si es un detalle periférico que no cambia el hecho central (ej: hora exacta, nombre de una calle secundaria).
                    3) Considera una omisión significativa únicamente cuando la ausencia del dato pueda modificar la interpretación de los hechos principales. Ignora omisiones de contexto secundario, detalles periféricos o estilo.
                    4) Registra únicamente coincidencias verificables presentes en al menos dos fuentes que respalden la noticia base.
                    5) No evalúes opiniones, juicios editoriales, tono, estilo narrativo ni interpretaciones.
                    6) No infieras información que no aparezca explícitamente en los textos. Solo trabaja con información verificable dentro de los textos proporcionados. Si un dato no puede contrastarse con ninguna otra fuente, márcalo como no_verificable.
                    7) Si no encuentras contradicciones, omisiones o coincidencias para una categoría, devuelve un arreglo vacío [].
                    8) La conclusión debe basarse exclusivamente en los hallazgos detectados y no en suposiciones ni inferencias.
                    9) Calcula un puntaje de veracidad del 0.00 al 10.00 con hasta dos decimales, donde 10.00 significa total consistencia con las otras fuentes y 0.00 significa contradicciones críticas en todos los hechos principales.
                    El puntaje parte de 10.0 y se calcula así: descuenta 0.8 punto por cada contradicción CRÍTICA, 0.12 puntos por cada contradicción MENOR, y 0.25 puntos por cada omisión significativa.
                    Suma 0.35 puntos por cada coincidencia verificada, con un máximo de 3.0 puntos por coincidencias. El puntaje final no puede ser menor a 0.0 ni mayor a 10.0.
                    Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional, sin bloques de código markdown, usando esta estructura exacta: {"contradicciones": [{"tipo": "CRÍTICA o MENOR", "campo": "cifra, fecha, nombre, lugar, declaración u otro", "noticia_base": "<valor en la noticia base>", "otras_fuentes": "<valor en las otras fuentes>", "descripcion": "<explicación breve y objetiva>"}], "omisiones_significativas": [{"dato_omitido": "<qué falta en la noticia base>", "presente_en": "<fuente donde aparece>", "impacto": "<por qué su ausencia modifica la interpretación de los hechos principales>"}], "coincidencias": [{"dato": "<dato confirmado>", "fuentes_que_coinciden": 0}], "resumen": {"total_contradicciones_criticas": 0, "total_contradicciones_menores": 0, "total_omisiones": 0, "total_coincidencias": 0, "nivel_consistencia": "ALTO, MEDIO o BAJO", "puntaje_veracidad": 0.00, "conclusion": "<2 o 3 oraciones objetivas basadas únicamente en los hallazgos detectados>"}}
            """,
            temperature=0.1,
            response_mime_type= "application/json"
        )

        bloque_versiones = "\n\n".join(
            [f"Fuente: {v['fuente']}\nTexto: {v['texto']}" for v in versiones_contraste]
        )
        try:
            time.sleep(12)
            response = client.models.generate_content(
                model = "gemini-2.5-flash",
                contents=f"noticia base:{noticia_base}\n\n noticia contaste:{bloque_versiones}",
                config= config
            )
            analisis_data = json.loads(response.text)
            resumen_obj = analisis_data.get("resumen", {})
            veracidad_gis = resumen_obj.get("puntaje_veracidad", 0.0)
            conclusion_gis = resumen_obj.get("conclusion", "No se generó ninguna conclusión.")
        except Exception as e:
            veracidad_total_gis = 0.00
            conclusion_gis = f"No se pudo completar el análisis: {e}"

    porcentaje_tot_veracidad = ((veracidad_gis * 0.45) + (veracidad * 0.55))
    
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
    "mundo": "world"
}

@app.route("/process", methods=["GET"])
def obtener_noticias():   
   url = request.args.get("url")

   if not url:
       return jsonify({"error": "No se ha proporcionado URL"}), 400
   try:
        response = requests.get(url) 
        response.encoding = "utf-8"
        data = response.json()
        articles = data.get ("articles", [])
        if not articles:
            return jsonify({"error": "No hay articulos"}), 404

        global_news = []

        for i, article in enumerate(articles):
            titulo = article.get("title", "Sin título")
            contenido = article.get("content", "")
            if not contenido:
                continue
            contenido = contenido.replace("\n", " ")
            contenido = re.sub(r"ver también\s*", " ", contenido, flags=re.IGNORECASE)

            try:
                titulo = titulo.encode("latin1").decode("utf-8")
            except(UnicodeDecodeError, UnicodeEncodeError):
                pass

            try:
                contenido = contenido.encode("latin1").decode("utf-8")
            except:
                pass
            
            keyword_extraction_result = (keyword_extraction(contenido, titulo))
            
            noticia = {                          
                "title": titulo,                 
                "content": contenido,
                "keywords": keyword_extraction_result,
                "score" : 0,   
                "conclusion": " "          
                }
            
            noticia["score"], noticia["conclusion"] = compute_score(noticia)
            global_news.append(noticia)
            
        return Response(
            json.dumps(
                {"status": "ok", "datos": global_news},
                ensure_ascii=False   
            ),
            content_type="application/json; charset=utf-8",
        )
   except Exception as e:
       return jsonify({"status": "error", "mensaje": str(e)}), 500
   

def process() :
    news = obtener_noticias()

if __name__ == "__main__":
    app.run(debug=True)