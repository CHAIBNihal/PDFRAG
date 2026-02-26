Projet ChatBot

Description
------------
un petit exemple demo sur les etapes necessaire pour extraire le text depuis un fichier pdf et le manipuler 
Les etapes Suivant sont les necessaire en RAG procedure : 

1 - Extraction du texte depuis le PDF
Utilisation de PdfReader pour récupérer le contenu textuel du fichier.

2 - Découpage du texte en chunks
Le texte est segmenté en morceaux (chunks) afin de préparer la génération d’embeddings.

3 - Génération des embeddings et création du vectorestore
Les chunks sont convertis en embeddings via Hugginface lib, puis indexés et stockés dans un vectorestore FAISS.

4 - Analyse et recherche contextuelle (RAG)
Les embeddings permettent de récupérer les passages les plus pertinents en réponse à une question.

5 - Génération de la réponse
Un modèle LLM (ici Llama 3.1) génère la réponse finale basée sur le contexte extrait des PDFs.

Prérequis
---------
- Python 3.8+ installé
- Git (optionnel)

Installation (Windows)
----------------------
1. Créer un environnement virtuel:

```
py -m venv venv
```

2. Activer l'environnement virtuel:

```
venv\Scripts\activate
```

3. Installer les dépendances:

```
pip install -r requirements.txt
```

Lancer l'application
--------------------
Depuis l'environnement activé, lancer Streamlit:

```
streamlit run app.py
```

Fichiers importants
-------------------
- `app.py` : interface principale Streamlit
- `pdfProccess.py` : traitement des PDFs
- `TemplateHtml.py` : gabarit HTML utilisé
- `faiss_index/index.faiss` : index FAISS enregistré

Notes
-----
- Ajustez les variables d'environnement dans un fichier `.env` si nécessaire.
- Si vous rencontrez des erreurs d'installation liées à FAISS, installez `faiss-cpu` séparément selon votre plateforme.

Contribuer
----------
1. Créez une branche feature.
2. Soumettez une PR.

Licence
-------
Voir les mentions dans le dépôt.
