# ------------------------------------------
# Projet : Implémenter un modèle de scoring
# Données: https://www.kaggle.com/c/home-credit-default-risk/data
# Auteur : Méric Manuel Kucukbas
# Date: Décembre 2023
# ------------------------------------------


from fastapi import status
import requests
import json

# URL local
API_URL = "10.12.245.49"




def test_welcome():
    """Teste la fonction welcome() de l'API."""
    response = requests.get(API_URL)
    assert response.status_code == status.HTTP_200_OK
    assert json.loads(response.content) == 'Welcome to the scoring prediction API'


def test_check_client_id():
    """Teste la fonction check_client_id() de l'API avec un client faisant partie de la base de données."""
    url = API_URL + str(192535)
    response = requests.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert json.loads(response.content) == True


def test_check_client_id_2():
    """Test la fonction check_client_id() de l'API avec un client ne faisant pas partie de la base de données."""
    url = API_URL + str(100000)
    response = requests.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert json.loads(response.content) == False


def test_get_prediction():
    """Test la fonction get_prediction() de l'API. Avec un client ID particulier présent dans le jeu de données final."""
    url = API_URL + "prediction/" + str(192535)
    response = requests.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert json.loads(response.content) == 0.4805479971101088


if __name__ == '__main__':
    test_welcome()
    print("test_welcome passed!")
    test_check_client_id()
    print("check_client_id passed!")
    test_check_client_id_2()
    print("check_client_id_2 passed!")
    test_get_prediction()
    print("test_get_prediction passed!")
    print("All tests passed!")