# P7-Openclassrooms-Datascientist
Projet 7 OpenClassrooms : Implémentez un modèle de scoring

Vous êtes Data Scientist au sein d'une société financière, nommée "Prêt à dépenser", qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.

L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).
De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner.
Prêt à dépenser décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement. 

## Les données
Les données utilisées sont disponibles à cette adresse : https://www.kaggle.com/c/home-credit-default-risk/data

## La mission 
1. Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
2. Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client des chargés de relation client.
3. Mettre en production le modèle de scoring de prédiction à l’aide d’une API, ainsi que le dashboard interactif qui appelle l’API pour les prédictions.
4. Evaluer le data drift entre les données initiales et les données qui permettent de tester le modèle de scoring.

## Les différents files présents 

1. api_scoring_prediction.py : code Python de l'API (Flask)
2. dashboard_scoring_prediction.py : code Python du dashboard (Streamlit)
3. notebook_eda_modelisation.zip : Notebook Jupyter qui regroupe l'EDA & la modélisation pour la classification binaire 
4. Le reste sont les données retenues pour le dashboard, le modèle sous format pickle, des dossiers supports pour Heroku...



## Le dashboard
Le dashboard a été réalisé avec Streamlit et à l'aide d'une API elle-même réalisée avec FastAPI. L'API a été déployé sur le cloud (Heroku) à l'adresse suivante :

https://predictionscoringmk-e22bc31f0151.herokuapp.com/ 

et le dashboard à l'adresse suivante : 

https://predictiondashboardmk-bec0e3ebd521.herokuapp.com/
