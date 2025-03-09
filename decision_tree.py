import numpy as np
import pandas as pd

# Impostazione del seed per la riproducibilità
np.random.seed(42)

# Generazione di un dataset sintetico per l'approvazione di prestiti bancari
def generate_sample_data(n_samples=1000):
    # Creiamo tre variabili esplicative realistiche per l'approvazione di prestiti
    # 1. Reddito annuale (in migliaia di euro)
    reddito = np.random.normal(35, 15, n_samples)
    reddito = np.clip(reddito, 15, 100)  # Limitiamo tra 15k e 100k euro
    
    # 2. Punteggio di credito (score da 300 a 850)
    punteggio_credito = np.random.normal(650, 100, n_samples)
    punteggio_credito = np.clip(punteggio_credito, 300, 850).astype(int)
    
    # 3. Anni di impiego attuale (da 0 a 30)
    anni_impiego = np.random.exponential(5, n_samples)
    anni_impiego = np.clip(anni_impiego, 0, 30).astype(int)
    
    # Funzione di decisione per generare l'approvazione del prestito
    # Creiamo alcune regole realistiche per determinare l'approvazione
    approvazione = []
    for i in range(n_samples):
        # Regole di approvazione prestito
        if punteggio_credito[i] >= 700 and reddito[i] >= 40:
            # Ottimo punteggio di credito e buon reddito: approvato
            approvazione.append('approvato')
        elif punteggio_credito[i] >= 650 and reddito[i] >= 30 and anni_impiego[i] >= 5:
            # Buon punteggio, reddito nella media e stabilità lavorativa: approvato
            approvazione.append('approvato')
        elif punteggio_credito[i] < 580 or (reddito[i] < 25 and anni_impiego[i] < 2):
            # Punteggio basso o reddito basso con bassa stabilità lavorativa: non approvato
            approvazione.append('non_approvato')
        else:
            # Casi intermedi: applichiamo una regola probabilistica
            # Più alto è il punteggio, maggiore è la probabilità di approvazione
            prob_approvazione = (punteggio_credito[i] / 850) * 0.7 + (reddito[i] / 100) * 0.2 + (min(anni_impiego[i], 10) / 10) * 0.1
            if np.random.random() < prob_approvazione:
                approvazione.append('approvato')
            else:
                approvazione.append('non_approvato')
    
    # Creiamo un DataFrame
    data = pd.DataFrame({
        'reddito_annuale': reddito,
        'punteggio_credito': punteggio_credito,
        'anni_impiego': anni_impiego,
        'target': approvazione
    })
    
    return data

# Tracciamento del percorso decisionale per un esempio
def trace_decision_path(tree, features, example):
    feature_names = features.columns
    node_indicator = tree.decision_path([example])
    leaf_id = tree.apply([example])[0]
    
    # Otteniamo la lista dei nodi attraversati
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    
    path_info = []
    for i, node_id in enumerate(node_index):
        # Controlliamo se siamo in un nodo foglia
        if node_id == leaf_id:
            prediction = tree.predict([example])[0]
            result = "approvato" if prediction == "approvato" else "non approvato"
            path_info.append(f"Passo {i+1}: Nodo {node_id} è un nodo foglia. Prestito: {result}")
        else:
            # Altrimenti è un nodo di decisione
            feature_id = tree.tree_.feature[node_id]
            threshold = tree.tree_.threshold[node_id]
            feature_name = feature_names[feature_id]
            
            # Controlliamo se l'esempio va a sinistra o a destra
            if example[feature_id] <= threshold:
                path_info.append(f"Passo {i+1}: Nodo {node_id}, {feature_name} = {example[feature_id]:.1f} <= {threshold:.1f}, vai a sinistra.")
            else:
                path_info.append(f"Passo {i+1}: Nodo {node_id}, {feature_name} = {example[feature_id]:.1f} > {threshold:.1f}, vai a destra.")
    
    return path_info

# Analisi dei nodi dell'albero
def analyze_tree_nodes(clf, X, num_nodes=10):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    values = clf.tree_.value

    # Limita il numero di nodi al massimo disponibile nell'albero
    num_nodes_to_print = min(num_nodes, n_nodes)
    # Mostra un avviso se ci sono meno nodi del richiesto
    if num_nodes > n_nodes:
        print(f"Attenzione: l'albero ha solo {n_nodes} nodi totali, verranno mostrati tutti quelli disponibili.")
    nodes_info = []
    
    for i in range(num_nodes_to_print):
        # Otteniamo l'indice della caratteristica di divisione
        if feature[i] != -2:  # -2 indica un nodo foglia
            feature_name = X.columns[feature[i]]
            split_condition = f"{feature_name} <= {threshold[i]:.1f}"
        else:
            split_condition = "Nodo foglia"
        
        # Calcolo della distribuzione delle classi nel nodo
        node_samples = values[i].sum()
        approvati_count = values[i][0][0]
        non_approvati_count = values[i][0][1]
        
        # Calcolo delle percentuali
        approvati_percent = (approvati_count / node_samples) * 100
        non_approvati_percent = (non_approvati_count / node_samples) * 100
        
        class_distribution = f"approvato: {approvati_percent:.1f}% ({int(approvati_count)}), non_approvato: {non_approvati_percent:.1f}% ({int(non_approvati_count)})"
        
        # Raccolta delle informazioni sul nodo
        node_info = {
            "nodo": i,
            "condizione": split_condition,
            "campioni": node_samples,
            "distribuzione": class_distribution
        }
        
        if feature[i] != -2:  # Se non è un nodo foglia
            node_info["figlio_sx"] = children_left[i]
            node_info["figlio_dx"] = children_right[i]
        
        nodes_info.append(node_info)
    
    return nodes_info