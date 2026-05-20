# Máster en Inteligencia Artificial, Cloud Computing y DevOps

## Machine Learning y Deep Learning: Guión Entrega Final

[cite_start]**Profesor:** Sergio Benito Martín [cite: 5]  
[cite_start]**Institución:** PontIA Tech [cite: 1]  
[cite_start]**Fecha/Versión:** 2026-02 [cite: 1]

---

## Índice

1. [Introducción](#introducción)
2. [Objetivo General](#objetivo-general)
3. [Requisitos Mínimos](#requisitos-mínimos)
4. [Criterios de Evaluación](#criterios-de-evaluación)
5. [Entregables Obligatorios](#entregables-obligatorios)
6. [Conclusión](#conclusión)
7. [Anexo: Criterios de Evaluación Detallados](#anexo-criterios-de-evaluación)

---

## Introducción

[cite_start]El módulo de Machine Learning y Deep Learning debe ofrecer un conocimiento base sobre los fundamentos del uso de algoritmos de aprendizaje automático que permitirán la adquisición de conocimientos más específicos sobre la Inteligencia Artificial Generativa[cite: 13].

[cite_start]Con el objetivo de poder preparar a los alumnos adecuadamente para adquirir dichos conocimientos y ser capaces de resolver casos de uso reales en su futuro profesional, la evaluación final constará de un ejercicio a resolver centrado en el uso de algoritmos de aprendizaje automático a la vez que se utilizan herramientas y tecnologías ya vistas en módulos previos, de tal manera que el aprendizaje se vaya realizando de manera iterativa e incremental[cite: 14].

---

## Objetivo General

[cite_start]Uno de los grandes avances en la industria ha sido el nacimiento de muchas tecnologías de _low-code_ que permiten entrenar modelos de manera muy rápida y eficiente, probando diferentes algoritmos al mismo tiempo y ofreciendo el resultado al usuario, denominadas librerías de **AutoML** (algunos ejemplos son PyCaret, MLJar, H2O, TPOT, etc.)[cite: 16]. [cite_start]Gracias a estas tecnologías, el proceso de prototipado de un caso de uso es muy rápido e, incluso, permite construir una solución útil para el destinatario final[cite: 16].

[cite_start]Para ello, es necesario que dichas librerías estén construidas de manera muy modular y con una orientación a objetos que permita hacer transparente el proceso de entrenamiento y comparación de dichos modelos[cite: 17].

[cite_start]El objetivo de la práctica es **diseñar e implementar un sistema automático** que[cite: 18]:

- [cite_start]Entrene, evalúe y compare distintos modelos de clasificación binaria[cite: 19].
- [cite_start]Seleccione el mejor modelo según una métrica principal, además de ofrecer una visión de otras secundarias[cite: 20].
- [cite_start]Automatice el flujo completo desde los datos hasta la inferencia[cite: 21].

> [cite_start]_Nota: Se propone adjunto el dataset que se utilizará para llevar a cabo el proyecto[cite: 22]._

---

## Requisitos Mínimos

### 1. Problema de Clasificación Binaria

- [cite_start]**Dataset:** Dataset real proporcionado, con una variable objetivo binaria (0/1)[cite: 25].
- [cite_start]**Justificación:** Se debe incluir una justificación del problema y del conjunto de datos elegido[cite: 26].

### 2. Modelos a Implementar y Comparar

[cite_start]Se deben entrenar al menos los siguientes algoritmos, además de otros que puedan ser de interés[cite: 31]:

- [cite_start]Regresión logística [cite: 32]
- [cite_start]Árbol de decisión [cite: 34]
- [cite_start]Random Forest [cite: 35]
- [cite_start]Gradient Boosting (XGBoost, LightGBM o CatBoost) [cite: 36]
- [cite_start]Red neuronal multicapa usando Keras de TensorFlow [cite: 37]

### 3. Evaluación de Modelos

- [cite_start]**Métrica Principal:** Utilizar al menos una de las siguientes métricas como principal: _accuracy_, _precision_, _recall_, _F1-score_, o _AUC-ROC_[cite: 40, 41]. [cite_start]Se debe justificar el porqué de su elección[cite: 42].
- [cite_start]**Visualizaciones requeridas:** Mostrar la matriz de confusión y la curva ROC[cite: 43].

### 4. Automatización del Flujo

- [cite_start]Implementar un **pipeline estructurado** para: carga de datos, preprocesamiento, entrenamiento, evaluación y selección del mejor modelo[cite: 46, 47].

---

## Criterios de Evaluación

[cite_start]La práctica se realizará **por parejas**, pudiéndose formar por elección propia (en caso de no encontrar pareja, comunicarlo al profesor)[cite: 49, 50]. [cite_start]El objetivo es promover una situación real de trabajo en equipo[cite: 51].

[cite_start]Se valorará la entrega en su conjunto, pero también la contribución individual mediante[cite: 52]:

1. [cite_start]El historial de _commits_ en GitHub para entender la colaboración conjunta[cite: 54].
2. [cite_start]La autonomía, aportaciones técnicas y visibilidad en el repositorio[cite: 55].
3. [cite_start]La definición explícita de roles en la documentación (quién se hace cargo de qué parte)[cite: 56]. [cite_start]_Si no se hace distinción de roles, se asignará la misma nota a ambos integrantes[cite: 57]._

### Defensa del Proyecto

[cite_start]Se realizará una defensa de la práctica a posteriori de la entrega con el profesor (máximo 30 minutos)[cite: 58, 62, 67].

- [cite_start]**Formato:** Los alumnos ejecutarán el código en directo y responderán preguntas para justificar las decisiones de implementación tomadas[cite: 62, 65].
- [cite_start]**Reserva:** Se agendará a través del calendario facilitado por el profesor[cite: 68].

### Desglose de Pesos de Evaluación

| Criterio                                         |   Peso   |
| :----------------------------------------------- | :------: |
| Implementación y comparación de modelos          |   30%    |
| Evaluación técnica y visualización               |   25%    |
| Automatización y estructura del sistema          |   25%    |
| Calidad del código y documentación               |   20%    |
| **Bonus técnicos (objetos, API, embeddings...)** | **+20%** |

[cite_start][cite: 71]

---

## Entregables Obligatorios

[cite_start]Cada pareja debe subir a la plataforma de PontIA el mismo informe en PDF (que incluya la URL del repositorio de GitHub)[cite: 73, 95].

### 1. Repositorio de Código (GitHub)

- [cite_start]Código bien estructurado, separando scripts, notebooks y archivos auxiliares[cite: 75].
- [cite_start]Archivo `.gitignore` adecuado[cite: 85].
- [cite_start]Archivo `requirements.txt` con todas las dependencias y versiones de las librerías[cite: 86, 94].
- **Documentación completa en el README.md:**
  - [cite_start]Autores de la práctica [cite: 77]
  - [cite_start]Descripción del problema y de los datos [cite: 78]
  - [cite_start]Instrucciones para configurar el entorno virtual y ejecutar el proyecto [cite: 79, 94]
  - [cite_start]Resultados y conclusiones [cite: 80]

### 2. Informe Final (PDF o Markdown)

[cite_start]Debe contener obligatoriamente[cite: 87]:

- [cite_start]Definición de los roles de la pareja[cite: 88].
- [cite_start]Justificación del problema[cite: 89].
- [cite_start]Análisis Exploratorio de Datos (EDA)[cite: 90].
- [cite_start]Diseño del sistema[cite: 91].
- [cite_start]Resultados obtenidos y elección del modelo final[cite: 92].
- [cite_start]Reflexión crítica sobre limitaciones y propuestas de mejora[cite: 93].

---

## Anexo: Criterios de Evaluación Detallados

### 1. Implementación y Comparación de Modelos (30%)

[cite_start]Se evalúa la correcta implementación de al menos los 5 modelos exigidos, entrenados bajo un enfoque metodológico coherente y comparados con métricas comunes[cite: 104, 106, 107]. [cite_start]Se valorará positivamente la justificación técnica de por qué un modelo supera a otro y la presentación de los resultados en una tabla comparativa como la siguiente[cite: 109, 110]:

| Modelo                      | Accuracy | F1-score | ROC-AUC |
| :-------------------------- | :------: | :------: | :-----: |
| Logistic Regression         |   0.88   |   0.85   |  0.91   |
| Decision Tree               |   0.89   |   0.83   |  0.92   |
| Random Forest               |   0.89   |   0.91   |  0.94   |
| XGBoost                     |   0.92   |   0.90   |  0.95   |
| Deep Neural Network (Keras) |   0.89   |   0.87   |  0.93   |

[cite_start][cite: 112]

### 2. Evaluación Técnica y Visualización (25%)

[cite_start]Se evalúa la aplicación de métricas relevantes, la inclusión de visualizaciones (curvas ROC comparativas, matrices de confusión legibles, gráficos de importancia de variables) y el uso del Análisis Exploratorio de Datos (EDA) para tomar decisiones de diseño[cite: 113, 115, 116, 117].

### 3. Automatización y Estructura del Sistema (25%)

[cite_start]El proceso debe estar empaquetado en un pipeline claro (no únicamente en un notebook suelto), separando las funcionalidades en distintos módulos o scripts (`data_loader.py`, `trainer.py`, etc.), utilizando funciones o clases reutilizables y un pipeline de Scikit-learn estructurado[cite: 125, 127, 128].

### 4. Calidad del Código y Documentación (20%)

[cite_start]Uso de buenas prácticas de programación (cumplimiento de **PEP8**, nombres de variables descriptivos, comentarios), documentación interna mediante _docstrings_ y organización limpia de carpetas junto con el `.gitignore`[cite: 133, 135].

### 5. Bonus Técnicos (Hasta +20% / +2 puntos adicionales)

[cite_start]Para sumar estos puntos el sistema base debe funcionar correctamente[cite: 140]. Se pueden implementar opciones como:

| Bonus Técnico                       | Descripción                                                                                  |
| :---------------------------------- | :------------------------------------------------------------------------------------------- |
| **API con FastAPI**                 | Crear una API REST para exponer endpoints como `/train`, `/predict` o `/evaluate`.           |
| **Embeddings personalizados**       | Uso de Word2Vec, TF-IDF o embeddings categóricos para representar variables complejas.       |
| **Optimización de hiperparámetros** | Aplicación de `GridSearchCV` o `RandomizedSearchCV`.                                         |
| **Registro de experimentos**        | Utilizar MLFlow para registrar los experimentos con diferentes algoritmos y el modelo final. |
| **Interfaz visual**                 | Presentación de la solución vía Streamlit, Gradio o una interfaz web UI simple.              |

[cite_start][cite: 144]

---

## Estructura Recomendada del Repositorio

[cite_start]El formato final es libre siempre y cuando se separe el código fuente de los notebooks de exploración[cite: 160]. Se recomienda la siguiente estructura:

```text
proyecto-final-ml/
├── .gitignore               # Archivos que no se suben al repo (modelos, datos temporales...)
├── data/                    # Datos usados en el proyecto
│   ├── raw/                 # Datos originales sin procesar (CSV descargado, Parquet...)
│   └── processed/           # Datos tras limpieza y transformación (listos para modelar)
├── docs/                    # Documentación adicional
├── models/                  # Modelos entrenados (guardados con joblib o pickle)
│   ├── tests/               # (Opcional) Modelos intermedios de prueba
│   ├── logistic_regression.pkl
│   ├── tree.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── lightgbm.pkl
│   ├── neural_network.pkl
│   └── best_model.pkl       # El mejor modelo seleccionado para producción
├── notebooks/               # Todos los notebooks del proyecto
│   ├── exploracion/         # Notebooks de pruebas, EDA inicial, prototipos
│   │   ├── eda_inicial.ipynb
│   │   └── pruebas_modelos.ipynb
│   └── finales/             # Notebooks finales con resultados o presentación
│       ├── eda_final.ipynb
│       └── comparativa_modelos.ipynb
├── outputs/                 # Gráficos, reportes y resultados generados (PNGs, HTMLs...)
│   ├── confusion_matrix.png
│   └── feature_importance.html
├── src/                     # Código fuente del proyecto
│   ├── __init__.py          # Inicializador del paquete src
│   ├── config.py            # Parámetros y configuración del proyecto
│   ├── data_loader.py       # Funciones para cargar y transformar datos
│   ├── model_trainer.py     # Clase o funciones para entrenar modelos
│   ├── evaluator.py         # Métricas y visualización de resultados
│   └── predictor.py         # Funciones para hacer predicciones con modelos entrenados
└── README.md                # Documentación principal del proyecto con comandos de ejecución
```
