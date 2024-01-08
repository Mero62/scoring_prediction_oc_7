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


## Le dashboard
Le dashboard a été réalisé avec Streamlit et à l'aide d'une API elle-même réalisée avec FastAPI. Le déploiement sur une plateforme Cloud n'étant plus nécessaire (et parce que je n'ai pas d'argent pour le hosting), la démonstration du fonctionnement de l'API et du Dashboard sont réalisées en local. 

Le dashboard présenté est celui pour un client dont le prêt a été accordé.
