# Albero Decisionale per l'Approvazione di Prestiti Bancari

Questo progetto implementa un'applicazione Streamlit interattiva che illustra il concetto di decision tree, utilizzando come caso pratico l'approvazione di prestiti bancari basata su tre variabili esplicative per classificare una variabile target binaria (approvato/non approvato).

## Obiettivo del Progetto

Come richiesto dalle istruzioni del professore, il progetto mira a:

> Costruire un albero decisionale con tre variabili esplicative e una variabile classificatoria (0 o 1). Il progetto deve essere originale e sensato, non troppo astratto.

Le metriche implementate per valutare la bontà dei risultati includono:
- Sensitivity (recall): Percentuale di true positives (TP) correttamente previsti come positivi TP / (TP +FN)
- Specificity: Percentuale di true negatives (TN) correttamente previsti come negativi TN / (TN +FP)
- Precision: Percentuale di true positives TP / (TP + FP)
- Accuracy (CA): percentuale di TP + TN su tutte le osservazioni
- ROC: per ciascuna soglia, rappresenta graficamente le coppie (sensitivity, 1 - specificity)
- AUC: area sottostante alla curva ROC (0.5, 1)
- F1: media armonica della Precision e della Recall = 2 * (precision * recall) / (precision + recall)

## Contenuto del Repository

- `streamlit_app.py`: Applicazione Streamlit interattiva per visualizzare e interagire con l'albero decisionale.
- `decision_tree.py`: Contiene le funzioni di utilità per generare il dataset e analizzare l'albero decisionale.
- `.gitignore`: Configurazione per escludere file non necessari dal repository.
- `.streamlit`: Contiene la configurazione per Streamlit.

## Struttura del Progetto

Il progetto è organizzato in modo semplice:
- Generazione di dati sintetici con tre variabili esplicative: reddito annuale, punteggio di credito e anni di impiego
- Addestramento di un albero decisionale configurabile
- Analisi dettagliata dei nodi dell'albero
- Visualizzazione dell'albero e dell'importanza delle caratteristiche
- Interfaccia interattiva per simulare nuove richieste di prestito

## Requisiti

Per eseguire il codice:

```bash
# Crea un ambiente virtuale (opzionale ma raccomandato)
python -m venv venv
source venv/bin/activate  # Per Linux/Mac
# oppure
venv\Scripts\activate  # Per Windows

# Installa le dipendenze
pip install -r requirements.txt
```

## Esecuzione

Per avviare l'applicazione Streamlit:

```bash
streamlit run streamlit_app.py
```

L'applicazione sarà accessibile all'indirizzo [http://localhost:8501](http://localhost:8501) nel tuo browser.

## Come Funziona

L'applicazione simula un dataset di richieste di prestito con tre variabili esplicative:
1. **Reddito annuale**: Variabile numerica che rappresenta il reddito del richiedente
2. **Punteggio di credito**: Variabile numerica che rappresenta l'affidabilità creditizia
3. **Anni di impiego**: Variabile categorica che rappresenta la stabilità lavorativa

Queste variabili vengono utilizzate per predire se un prestito verrà approvato o meno (classe target binaria).

L'albero decisionale viene addestrato su questi dati con una profondità massima configurabile per evitare overfitting.

## Struttura dell'Applicazione

L'applicazione è organizzata in 5 sezioni principali:

### 📊 Tab Dataset
- Mostra le prime 10 righe del dataset
- Visualizza la distribuzione delle classi (approvato/non approvato)
- Fornisce statistiche descrittive sulle variabili
- Presenta grafici di distribuzione per ogni caratteristica

### 📈 Tab Prestazioni
- Visualizza le metriche di performance del modello:
  - Accuracy, Precision, Recall/Sensitivity
  - Specificity, F1 Score, AUC
- Mostra la matrice di confusione
- Presenta la curva ROC

### 🌲 Tab Struttura dell'Albero
- Visualizza graficamente l'albero decisionale
- Mostra l'importanza delle diverse caratteristiche

### 🔍 Tab Simulazione
- Permette di simulare nuove richieste di prestito
- Mostra la decisione del modello (approvato/non approvato)
- Visualizza la probabilità di approvazione
- Traccia il percorso decisionale seguito dall'albero

### ❓ Tab FAQ
- Fornisce una spiegazione dell'applicazione e delle sue funzionalità
- Descrive la struttura e l'organizzazione dell'interfaccia utente
- Elenca le tecnologie utilizzate e i dettagli implementativi

## Funzionalità dell'Interfaccia

### Nella barra laterale:
- **Controllo della numerosità del dataset**: Modifica la dimensione del dataset
- **Controllo della profondità dell'albero**: Modifica la complessità del modello
- **Rigenerazione del dataset**: Crea un nuovo dataset casuale

### Nella tab Simulazione:
- **Simulazione interattiva**: Regola i parametri per vedere come cambiano le previsioni
- **Visualizzazione del percorso decisionale**: Segui passo-passo il processo decisionale dell'albero

## Tecnologie Utilizzate

Per lo sviluppo di questa applicazione sono state utilizzate:
- **Streamlit**: Per l'interfaccia web interattiva
- **Scikit-learn**: Per l'implementazione dell'albero decisionale
- **Pandas**: Per la gestione e manipolazione dei dati
- **Matplotlib e Seaborn**: Per le visualizzazioni grafiche
- **NumPy**: Per le operazioni numeriche
- **Plotly**: Per alcuni grafici interattivi

## Dettagli Implementativi

Aspetti tecnici notevoli dell'implementazione:
- **Generazione dati simulati**: Creazione di un dataset realistico con correlazioni tra variabili
- **Visualizzazione avanzata dell'albero**: Rappresentazione grafica personalizzata con colori e dettagli
- **Tracciamento interattivo delle decisioni**: Mostra come l'albero arriva alla decisione finale
- **Metriche di performance complete**: Implementazione di tutte le metriche richieste nelle istruzioni
- **Design UI/UX migliorato**: Uso di container, colori e icone per una migliore esperienza utente

## Crediti

Progetto per il corso di AI e Machine Learning per il Marketing - IULM
Creato da: Christian Centi
