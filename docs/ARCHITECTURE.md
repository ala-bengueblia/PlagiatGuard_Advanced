# Architecture PlagiatGuard

## Présentation
Streamlit offre une interface pour importer et analyser les documents.

## API
FastAPI expose des routes d'analyse de texte et de fichiers.
Les jetons JWT identifient les utilisateurs.

## Moteur
Le texte est extrait, normalisé et découpé en passages.
Le moteur TF-IDF calcule des similarités lexicales.
Les embeddings multilingues sont facultatifs.

## Recherche
FAISS indexe les passages du corpus pour rechercher des candidats.
Tavily permet une recherche Web lorsqu'une clé valide est configurée.

## Stockage
SQLite conserve les comptes, les rapports, les quotas et les événements d'audit.

## Évaluation
Le corpus annoté sert à calculer la précision, le rappel, le F1
et le taux de faux positifs.

## Sécurité
La production nécessite HTTPS, une politique de rétention,
une protection SSRF, des contrôles de rôles et une gestion sécurisée
des secrets.