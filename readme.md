progetto


-aefed:
   contiene gli autoencoder, convoluzionale funzionante, lstm no. 
    funzionamento: entrare nel venv poi nella cartella ed avvia tramite python in due terminali separati prima il server ''python server_app.py '' e poi i client ''python run_experiment.py''  richiede python 3.10.12

-xgboost:
    contiene due cartelle al momento nominate quickstart-xgboost( booster bagging ) e
    quickstart-xgboost0 (booster cyclic) al momento solo il cyclic funziona correttamente, le cartelle di dati presenti sono rimasugli di vecchie prove, suggerisco di eliminarle ed aggiungere una cartella ''data'' contente i dati nel formato corretto per essere trainati

    funzionamento: entrare nelle cartelle contenti i file .toml ed eseguire ''flwr .'' eventualmente ''flwr . --stream'' per una versione verbose (suggerita) 
