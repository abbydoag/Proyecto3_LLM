# 🌱 STARDEW BOT 🌱

Asistente basado en RAG (Retrieval Augmented Generation) que busca ayudarte con consultas sobre el juego Stardew Valley.
https://stardewvalleywiki.com/Stardew_Valley_Wiki es la fuente de los datos utilizados en este proyecto.

*Nota: Los datos que se usaron están en inglés por lo que este bot solo responderá preguntas en dicho idioma, por favor usar el mismo idioma para hacer consultas.*

---
## 🪏 Requisitos 🪏
**Mínimos:**
- Python 3.10+
- 8 GB de RAM libre (para cargar el modelo y ChromaDB)
- 2 GB de disco (modelo Qwen, ChromaDB y Stardew_data.json)

    *Nota*: Sin GPU la indexación puede a llegar a tomar más tiempo
---

## 🐔 **Instalación** 🐔
1. Clonar repositorio y accederlo
```bash
git clone https://github.com/abbydoag/Proyecto3_LLM
```
2. Instalar requerimientos
```bash
pip install -r requirements.txt
```
3. Construir el corpus (solo la primera vez/hay un XML nuevo)
```bash
python cleanup.py      #genera 'Stardew_data.json'
python index_build.py  #genera 'data/chroma_db/'
```
4. (OPCIONAL) En caso se quiera verificar las paginas
```bash
python verificacion_paginas.py
```
Hay una lista de paginas a verificar, se puede cambiar a lo que se necesite

## 🐮**Levantar aplicacion**🐮
```bash
streamlit run app.py
```
La aplicacion se abrirá en el navegador, esperar un poco a que cargue y se puede usar

## 🐷**Obtener métricas**🐷
Para obtener las métricas se debe correr
```bash
python metricas.py
```

## **En caso se actualiza la wiki**
1. Obtener nuevo dump y colocar en ```data/raw/StardewDump.xml```
2. Regenerar chunks e índice
```bash
python cleanup.py
python index_build.py
```
3. Reevaluar
```bash
python verificacion_paginas.py
```

---
## 🌳 Estructura 🌳
---

```
├── data/
│   ├── raw/
│   │   └── StardewDump.xml          #dump wiki
│   ├── processed/
│   │   └── Stardew_data.json        #chunks (generados por cleanup.py)
│   └── chroma_db/                   #indice vectorial (generado por index_build)
├── eval/
│   └── eval_set.json                #casos evaluacion
├── cleanup.py                       #wikitexto - chunks JSON
├── index_build.py                   #chunks - ChromaDB
├── test.py                          #busqueda hibrida BM25 + embeddings
├── answer.py                        #Qwen2.5-0.5B + prompt de refusal
├── evaluate.py                      #recall@3 y MRR (retrieval)
├── metricas.py                      #fidelity, refusal, hallucination (end-to-end)
├── baseline.py                      #BM25 puro + LLM sin RAG
├── verificacion_paginas.py          #diagnostico corpus e indice
├── app.py                           #aplicación streamlit
├── requirements.txt
└── README.md
```