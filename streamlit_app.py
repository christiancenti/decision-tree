import streamlit as st
from decision_tree import generate_sample_data, trace_decision_path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import seaborn as sns
import plotly.graph_objects as go

# Configurazioni di stile
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# App Streamlit
def main():
    # Configurazione pagina e tema
    st.set_page_config(
        page_title="Decision Tree per Prestiti", 
        page_icon=":deciduous_tree:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "Applicazione per l'apprendimento degli alberi decisionali"
        }
    )
    
    # Header
    st.title("🌳 Albero Decisionale per l'Approvazione di Prestiti Bancari")
    
    with st.container(border=True):
        st.info("""
        Questa applicazione dimostra come un albero decisionale può essere utilizzato per prevedere se un prestito bancario 
        sarà approvato o meno, in base a tre fattori principali: **reddito annuale**, **punteggio di credito** e **anni di impiego**.
        
        Usa i controlli nella barra laterale per modificare i parametri e vedere come cambia il modello.
        """)
    
    # Sidebar per parametri con migliore organizzazione    
    with st.sidebar:
        st.title("📊 Parametri del Modello")
        
        # Definisco prima il valore predefinito per num_samples
        default_num_samples = 1000
        
        # Opzioni Dataset (spostiamo questa sezione prima della configurazione albero)
        st.subheader("🔄 Opzioni Dataset")
        
        # Otteniamo il valore di num_samples con un valore predefinito
        num_samples = st.slider("Numero di campioni", 100, 2000, default_num_samples, 100,
                          help="Numero di osservazioni nel dataset")
        
        # Tracciamento dei cambiamenti con session_state
        if 'num_samples_previous' not in st.session_state:
            st.session_state.num_samples_previous = default_num_samples
            
        # Controlliamo se il valore è cambiato
        if num_samples != st.session_state.num_samples_previous:
            # Rigeneriamo il dataset
            st.session_state.data = generate_sample_data(num_samples)
            st.success(f"Dataset rigenerato con {num_samples} campioni!")
            # Aggiorniamo il valore precedente
            st.session_state.num_samples_previous = num_samples
        
        # Pulsante per rigenerazione manuale
        if st.button("🔁 Rigenera Dataset", use_container_width=True):
            st.session_state.data = generate_sample_data(num_samples)
            st.success(f"Dataset rigenerato con {num_samples} campioni!")
        
        # Configurazione Albero (spostiamo questa sezione dopo le opzioni del dataset)
        with st.container(border=True):
            st.subheader("🎚️ Configurazione Albero")
            # Parametri albero
            max_depth = st.slider("Profondità massima dell'albero", 1, 10, 3, 
                          help="Controlla quanto può crescere l'albero in profondità. Valori più alti consentono modelli più complessi.")
            
        # Aggiungiamo il QR code alla fine della sidebar
        st.divider()
        st.subheader("📱 Accedi all'app")
        st.image("qr_code.svg", use_container_width=True)
        
    
    # Inizializza il dataset alla prima esecuzione
    if 'data' not in st.session_state:
        st.session_state.data = generate_sample_data(num_samples)
    
    data = st.session_state.data
    
    # Crea un sistema di navigazione con tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dataset", "📈 Prestazioni", "🌲 Struttura dell'Albero", "🔍 Simulazione", "❓ FAQ"])
    
    with tab1:
        st.header("📊 Esplorazione Dataset")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            with st.container(border=True):
                st.subheader("🔢 Prime 10 righe del dataset")
                st.dataframe(data.head(10), use_container_width=True)
        
        with col2:
            with st.container(border=True):
                st.subheader("📊 Distribuzione delle classi")
                fig, ax = plt.subplots()
                sns.countplot(x='target', data=data, ax=ax, palette=['#4CAF50', '#F44336'])
                ax.set_xlabel('Approvazione Prestito')
                ax.set_xticklabels(['Approvato', 'Non Approvato'])
                ax.set_ylabel('Numero di clienti')
                ax.set_title('Distribuzione delle approvazioni')
                st.pyplot(fig)
            
        # Statistiche descrittive
        with st.container(border=True):
            st.subheader("📝 Statistiche descrittive")
            st.write("Panoramica statistica delle variabili nel dataset")
            st.dataframe(data.describe(), use_container_width=True)
            
        # Grafici di distribuzione delle variabili
        st.subheader("📊 Distribuzione delle variabili")
        
        with st.container(border=True):
            dist_cols = st.columns(3)
            with dist_cols[0]:
                fig, ax = plt.subplots()
                sns.histplot(data['reddito_annuale'], kde=True, ax=ax, color='#2196F3')
                ax.set_title('Distribuzione Reddito Annuale')
                ax.set_xlabel('Reddito (k€)')
                st.pyplot(fig)
                
            with dist_cols[1]:
                fig, ax = plt.subplots()
                sns.histplot(data['punteggio_credito'], kde=True, ax=ax, color='#FF9800')
                ax.set_title('Distribuzione Punteggio di Credito')
                ax.set_xlabel('Punteggio')
                st.pyplot(fig)
                
            with dist_cols[2]:
                fig, ax = plt.subplots()
                sns.histplot(data['anni_impiego'], kde=True, ax=ax, color='#9C27B0')
                ax.set_title('Distribuzione Anni di Impiego')
                ax.set_xlabel('Anni')
                st.pyplot(fig)
    
    # Divisione in training e test set
    X = data[['reddito_annuale', 'punteggio_credito', 'anni_impiego']]
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Creazione e addestramento dell'albero decisionale
    clf = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    clf.fit(X_train, y_train)
    
    # Predizione sul test set
    y_pred = clf.predict(X_test)
    
    # Calcolo delle probabilità per ROC e AUC
    y_pred_proba = clf.predict_proba(X_test)
    
    # Calcolo confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    if len(cm) == 2:  # Se abbiamo due classi
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = f1_score(y_test, y_pred, pos_label='approvato')
        
        # Calcolo ROC e AUC (solo per problemi binari)
        # Troviamo l'indice della classe positiva 'approvato'
        if clf.classes_[0] == 'approvato':
            pos_class_idx = 0
        else:
            pos_class_idx = 1
            
        fpr, tpr, _ = roc_curve(y_test == 'approvato', y_pred_proba[:, pos_class_idx])
        roc_auc = auc(fpr, tpr)
    else:
        # Per problemi multiclasse (anche se in questo caso abbiamo solo 'approvato' e 'non_approvato')
        sensitivity = recall_score(y_test, y_pred, average='weighted')
        # Non c'è un valore diretto per specificity in multiclasse
        specificity = 0
        precision = precision_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = 0  # Non calcoliamo AUC per multiclasse in questo esempio
    
    with tab2:
        st.header("📈 Metriche di Performance")
        
        # Funzione per mostrare una metrica con spiegazione
        def metric_with_explanation(label, value, explanation):
            st.metric(label, f"{value:.4f}", help=explanation)
        
        with st.container(border=True):
            st.subheader("🎯 Indicatori di Performance")
            
            # Organizziamo tutte le metriche in 3 righe con 3 metriche ciascuna (3x3 grid)
            row1_cols = st.columns(3)
            row2_cols = st.columns(3)
            
            # Prima riga: Accuracy, Precision, Recall
            with row1_cols[0]:
                metric_with_explanation(
                    "✅ Accuracy", 
                    accuracy, 
                    "Percentuale di predizioni corrette sul totale.\n\n"
                    "Formula: (TP + TN) / (TP + TN + FP + FN)\n\n"
                    "Misura la percentuale di predizioni corrette del modello."
                )
                
            with row1_cols[1]:
                metric_with_explanation(
                    "🎯 Precision", 
                    precision, 
                    "Percentuale di true positives sul totale dei positivi previsti.\n\n"
                    "Formula: TP / (TP + FP)\n\n"
                    "Misura quanto sono affidabili le predizioni positive del modello."
                )
                
            with row1_cols[2]:
                metric_with_explanation(
                    "🔍 Recall/Sensitivity", 
                    sensitivity, 
                    "Percentuale di true positives (TP) correttamente previsti come positivi.\n\n"
                    "Formula: TP / (TP + FN)\n\n"
                    "Misura la capacità del modello di identificare correttamente i casi positivi."
                )
            
            # Seconda riga: Specificity, F1 Score, AUC
            with row2_cols[0]:
                metric_with_explanation(
                    "🛡️ Specificity", 
                    specificity, 
                    "Percentuale di true negatives (TN) correttamente previsti come negativi.\n\n"
                    "Formula: TN / (TN + FP)\n\n"
                    "Misura la capacità del modello di identificare correttamente i casi negativi."
                )
                
            with row2_cols[1]:
                metric_with_explanation(
                    "⚖️ F1 Score", 
                    f1, 
                    "Media armonica di precision e recall.\n\n"
                    "Formula: 2 * (precision * recall) / (precision + recall)\n\n"
                    "Fornisce un bilanciamento tra precision e recall."
                )
            
            with row2_cols[2]:
                if len(cm) == 2:  # Solo per classificazione binaria
                    metric_with_explanation(
                        "📈 AUC", 
                        roc_auc, 
                        "Area sotto la curva ROC.\n\n"
                        "Range: [0.5, 1]\n\n"
                        "Misura la capacità del modello di distinguere tra le classi. Un valore di 0.5 indica un modello casuale, 1.0 indica un modello perfetto."
                    )
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                st.subheader("🧩 Matrice di Confusione")
                
                # Semplificazione della matrice di confusione
                # Mostriamo solo i valori assoluti con percentuali sul totale
                
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Prepariamo una matrice con valori assoluti e percentuali
                annot = np.empty_like(cm, dtype=object)
                total = np.sum(cm)
                
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        percentage = 100 * cm[i, j] / total
                        annot[i, j] = f"{cm[i, j]}\n({percentage:.1f}%)"
                
                # Etichette dei casi
                labels = ['Approvato', 'Non Approvato']
                
                # Visualizziamo la matrice
                sns.heatmap(cm, annot=annot, fmt="", cmap='Blues',
                          xticklabels=labels, yticklabels=labels, ax=ax)
                
                plt.title("Matrice di Confusione")
                plt.ylabel('Classe reale')
                plt.xlabel('Classe prevista')
                st.pyplot(fig)
                
                st.write("""
                    La matrice di confusione mostra:
                - **Veri Positivi (in alto a sinistra)**: Prestiti approvati correttamente classificati come approvati
                - **Falsi Negativi (in alto a destra)**: Prestiti approvati erroneamente classificati come non approvati
                - **Falsi Positivi (in basso a sinistra)**: Prestiti non approvati erroneamente classificati come approvati
                - **Veri Negativi (in basso a destra)**: Prestiti non approvati correttamente classificati come non approvati
                
                In ciascuna cella viene mostrato:
                - Numero assoluto di casi
                - Percentuale rispetto al totale complessivo (in parentesi)
                """)
        
        with col2:
            if len(cm) == 2:  # Solo per problemi binari
                with st.container(border=True):
                    st.subheader("📉 Curva ROC")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('1 - Specificity (False Positive Rate)')
                    ax.set_ylabel('Sensitivity (True Positive Rate)')
                    ax.set_title('Receiver Operating Characteristic (ROC)')
                    ax.legend(loc="lower right")
                    st.pyplot(fig)
                    
                    st.write("""
                        La curva ROC rappresenta il compromesso tra:
                    - **Sensibilità (True Positive Rate)**: Capacità di rilevare correttamente i prestiti da approvare
                    - **1-Specificità (False Positive Rate)**: Errore nel classificare prestiti che non dovrebbero essere approvati
                    
                    Più la curva si avvicina all'angolo superiore sinistro, migliore è il modello.
                    """)
    
    with tab3:
        st.header("🌲 Struttura dell'Albero Decisionale")
        
        # Visualizzazione dell'albero decisionale
        with st.container(border=True):
            st.subheader("🔍 Visualizzazione dell'Albero")
            
            # Creazione della legenda personalizzata
            st.write("**Legenda:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                - 🟦 **Blu**: Tendenza verso "Approvato"
                - 🟧 **Arancione**: Tendenza verso "Non Approvato"
                - 🔍 **Intensità del colore**: Più intenso = maggiore certezza
                """)
            with col2:
                st.markdown("""
                - ⬅️ **Sinistra**: Se la condizione è VERA
                - ➡️ **Destra**: Se la condizione è FALSA
                - ⬇️ **Dall'alto verso il basso**
                """)
            with col3:
                st.markdown("""
                - 🎯 **Condizione**: es. "reddito ≤ 30.5"
                - ✅ Se VERO: vai a SINISTRA
                - ❌ Se FALSO: vai a DESTRA
                """)
            
            # Plot dell'albero con parametri semplificati
            fig, ax = plt.subplots(figsize=(20, 10))
            plot_tree(clf, 
                    feature_names=X.columns,
                    class_names=['approvato', 'non_approvato'],
                    filled=True,
                    rounded=True,
                    fontsize=10,
                    proportion=True,  # Visualizza solo proporzioni invece di conteggi
                    precision=1,      # Riduce il numero di decimali
                    impurity=False,
                    ax=ax)
            
            # Aggiunta di un titolo esplicativo
            plt.title("Albero Decisionale per l'Approvazione dei Prestiti", fontsize=16)
            
            st.pyplot(fig)
            
            # Aggiungo un esempio pratico
            with st.expander("💡 Esempio di Lettura", expanded=True):
                st.markdown("""
                **Come seguire una decisione:**
                1. Parti dal nodo in alto (radice)
                2. Leggi la condizione (es. "reddito_annuale ≤ 30.5")
                3. Se la condizione è VERA ✅ → vai a SINISTRA ⬅️
                4. Se la condizione è FALSA ❌ → vai a DESTRA ➡️
                5. Ripeti finché non raggiungi un nodo finale
                """)
        
        # Importanza delle caratteristiche
        with st.container(border=True):
            st.subheader("⭐ Importanza delle Caratteristiche")
            
            feature_importance = pd.DataFrame(
                {'feature': X.columns, 'importance': clf.feature_importances_}
            ).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax, palette=['#2196F3', '#FF9800', '#4CAF50'])
            
            # Aggiungi le percentuali alle barre
            for i, (importance, feature) in enumerate(zip(feature_importance['importance'], feature_importance['feature'])):
                ax.text(importance + 0.01, i, f'{importance:.2%}', va='center')
                
            plt.title('Importanza delle caratteristiche')
            plt.tight_layout()
            st.pyplot(fig)
            
            with st.expander("📚 Cosa significa l'importanza delle caratteristiche?", expanded=True):
                st.write("""
                    L'importanza delle caratteristiche indica quanto ciascuna variabile contribuisce alla capacità predittiva del modello:
                    
                    - **Valori più alti** indicano che la caratteristica è più influente nel determinare l'approvazione del prestito
                    - **Valori più bassi** indicano che la caratteristica ha meno impatto sulla decisione finale
                    
                    L'importanza è calcolata in base a quanto ciascuna caratteristica riduce l'impurità quando viene utilizzata per dividere i dati.
                    """)
    
    with tab4:
        st.header("🔍 Simulazione di Approvazione Prestito")
        
        with st.container(border=True):
            st.info("""
            Utilizza gli slider sottostanti per simulare una richiesta di prestito con diversi valori 
            per reddito, punteggio di credito e anni di impiego. Il modello predirà se il prestito sarà
            approvato o meno.
            """)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                reddito_val = st.slider("💰 Reddito Annuale (k€)", 
                                       float(X['reddito_annuale'].min()), 
                                       float(X['reddito_annuale'].max()), 
                                       35.0,
                                       help="Il reddito annuale del richiedente in migliaia di euro")
            
            with col2:
                punteggio_val = st.slider("📊 Punteggio di Credito", 
                                         int(X['punteggio_credito'].min()), 
                                         int(X['punteggio_credito'].max()), 
                                         650,
                                         help="Il punteggio creditizio del richiedente (300-850)")
            
            with col3:
                anni_val = st.slider("⏱️ Anni di Impiego", 
                                    int(X['anni_impiego'].min()), 
                                    int(X['anni_impiego'].max()), 
                                    5,
                                    help="Da quanti anni il richiedente lavora presso il datore di lavoro attuale")
        
        # Predizione con l'albero decisionale
        input_data = np.array([[reddito_val, punteggio_val, anni_val]])
        
        # Predizione
        prediction = clf.predict(input_data)[0]
        
        # Visualizzazione della predizione
        with st.container(border=True):
            st.subheader("📝 Risultato della Valutazione")
            
            # Decisione e probabilità
            decision_cols = st.columns([1, 1])
            
            # Colonna per la decisione
            with decision_cols[0]:
                st.write("### Decisione:")
                if prediction == 'approvato':
                    st.success("### ✅ PRESTITO APPROVATO")
                    st.markdown("""
                    <div style='text-align: center; margin-top: 10px; font-size: 14px;'>
                        Le caratteristiche del cliente soddisfano i criteri di approvazione
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("### ❌ PRESTITO NON APPROVATO")
                    st.markdown("""
                    <div style='text-align: center; margin-top: 10px; font-size: 14px;'>
                        Le caratteristiche del cliente non soddisfano i criteri minimi richiesti
                    </div>
                    """, unsafe_allow_html=True)
                    
            # Colonna per la probabilità
            with decision_cols[1]:
                st.write("### Probabilità di approvazione:")
                
                # Visualizzazione grafica della probabilità
                proba = clf.predict_proba(input_data)[0]
                proba_class_idx = 0 if clf.classes_[0] == 'approvato' else 1
                approvazione_prob = proba[proba_class_idx] * 100
                
                # Utilizziamo un colore che cambia in base alla probabilità
                if approvazione_prob >= 75:
                    color = "green"
                    description = "Alta probabilità di approvazione"
                    emoji = "🟢"
                elif approvazione_prob >= 50:
                    color = "orange" 
                    description = "Media probabilità di approvazione"
                    emoji = "🟠"
                else:
                    color = "red"
                    description = "Bassa probabilità di approvazione"
                    emoji = "🔴"
                
                # Scriviamo il valore di probabilità con formattazione grande
                st.markdown(f"<h1 style='text-align: center; color: {color};'>{approvazione_prob:.2f}%</h1>", unsafe_allow_html=True)
                
                # Progress bar
                st.progress(approvazione_prob / 100)
                
                # Descrizione della probabilità
                st.markdown(f"<div style='text-align: center;'>{emoji} {description}</div>", unsafe_allow_html=True)
        
        # Visualizzazione dell'albero decisionale
        with st.container(border=True):
            st.subheader("🌲 Albero Decisionale")
            fig, ax = plt.subplots(figsize=(20, 10))
            plot_tree(clf, 
                    feature_names=X.columns,
                    class_names=['approvato', 'non_approvato'],
                    filled=True,
                    rounded=True,
                    fontsize=10,
                    proportion=True,  # Visualizza solo proporzioni invece di conteggi
                    precision=1,      # Riduce il numero di decimali
                    impurity=False,
                    ax=ax)
            
            plt.title("Albero Decisionale per l'Approvazione dei Prestiti", fontsize=16)
            st.pyplot(fig)

        # Tracciamento del percorso decisionale
        with st.container(border=True):
            st.subheader("🛣️ Percorso Decisionale")
            st.write("Ecco come l'albero decisionale ha raggiunto questa conclusione:")
            
            path_info = trace_decision_path(clf, X, input_data[0])
            
            for i, step in enumerate(path_info):
                if i == len(path_info) - 1:  # L'ultimo passo (nodo foglia)
                    st.success(f"✅ **Decisione finale:** {step}")
                else:
                    st.info(f"🔍 **Passo {i+1}:** {step}")
    
    with tab5:
        st.header("❓ Guida all'Applicazione")
        
        with st.container(border=True):
            st.subheader("🎯 Cosa fa questa applicazione?")
            st.write("""
            Questa applicazione dimostra come un **albero decisionale** può essere utilizzato per analizzare l'approvazione di prestiti bancari.
            
            Abbiamo sviluppato un'interfaccia interattiva che permette di:
            1. Visualizzare e analizzare un dataset simulato di richieste di prestito
            2. Esplorare le prestazioni di un modello di classificazione basato su albero decisionale
            3. Esaminare la struttura dell'albero e come prende decisioni
            4. Simulare nuove richieste di prestito e vedere l'esito previsto
            """)
            
        with st.container(border=True):
            st.subheader("🧭 Struttura dell'applicazione")
            
            st.write("L'applicazione è organizzata in 5 sezioni principali:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **📊 Tab Dataset**
                - Mostra le prime 10 righe del dataset
                - Visualizza la distribuzione delle classi (approvato/non approvato)
                - Fornisce statistiche descrittive sulle variabili
                - Presenta grafici di distribuzione per ogni caratteristica
                """)
                
                st.markdown("""
                **📈 Tab Prestazioni**
                - Visualizza le metriche di performance del modello:
                  - Accuracy, Precision, Recall/Sensitivity
                  - Specificity, F1 Score, AUC
                - Mostra la matrice di confusione
                - Presenta la curva ROC (per classificazione binaria)
                """)
            
            with col2:
                st.markdown("""
                **🌲 Tab Struttura dell'Albero**
                - Visualizza graficamente l'albero decisionale
                - Mostra l'importanza delle diverse caratteristiche
                - Permette di esplorare nel dettaglio i vari nodi dell'albero
                """)
                
                st.markdown("""
                **🔍 Tab Simulazione**
                - Permette di simulare nuove richieste di prestito
                - Mostra la decisione del modello (approvato/non approvato)
                - Visualizza la probabilità di approvazione
                - Traccia il percorso decisionale seguito dall'albero
                """)
        
        with st.container(border=True):
            st.subheader("⚙️ Funzionalità Principali")
            
            st.markdown("""
            **Nella barra laterale:**
            - **Controllo della numerosità del dataset**: Modifica la dimensione del dataset
            - **Controllo della profondità dell'albero**: Modifica la complessità del modello
            - **Rigenerazione del dataset**: Crea un nuovo dataset casuale
            
            **Nella tab Simulazione:**
            - **Simulazione interattiva**: Regola i parametri per vedere come cambiano le previsioni
            - **Visualizzazione del percorso decisionale**: Segui passo-passo il processo decisionale dell'albero
            """)
        
        with st.container(border=True):
            st.subheader("🧪 Tecnologie Utilizzate")
            
            st.markdown("""
            Per lo sviluppo di questa applicazione abbiamo utilizzato:
            
            - **Streamlit**: Per l'interfaccia web interattiva
            - **Scikit-learn**: Per l'implementazione dell'albero decisionale
            - **Pandas**: Per la gestione e manipolazione dei dati
            - **Matplotlib e Seaborn**: Per le visualizzazioni grafiche
            - **NumPy**: Per le operazioni numeriche
            - **Plotly**: Per alcuni grafici interattivi
            """)
            
        with st.container(border=True):
            st.subheader("📋 Dettagli Implementativi")
            
            st.markdown("""
            Aspetti tecnici notevoli della nostra implementazione:
            
            - **Generazione dati simulati**: Creazione di un dataset realistico con correlazioni tra variabili
            - **Visualizzazione avanzata dell'albero**: Rappresentazione grafica personalizzata con colori e dettagli
            - **Tracciamento interattivo delle decisioni**: Mostra come l'albero arriva alla decisione finale
            - **Metriche di performance complete**: Implementazione di tutte le metriche richieste nelle istruzioni
            - **Design UI/UX migliorato**: Uso di container, colori e icone per una migliore esperienza utente
            """)

    # Aggiungi un footer
    st.divider()
    st.caption("🏫 Progetto AI e Machine Learning per il Marketing - IULM")
    st.caption("👨‍💻 Creato da: Christian Centi")

if __name__ == "__main__":
    main()