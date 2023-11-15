import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.metrics import confusion_matrix, auc, roc_curve, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def data_exploration(df):
    """Funzione chiamata dal main.py che stampa le informazioni relative al dataset """
    print("\n------------------------------ DATA EXPLORATION ------------------------------")
    # stampa informazioni relative al dataset
    print("\n--- Numero di righe e colonne del dataset: ")
    print(df.shape)

    print("\n--- Preview del dataset: ")
    print(df.head())

    print("\n--- Informazioni generali: ")
    print(df.info())

    print("\n--- Indicatori Statistici di base: ")
    print(df.describe().transpose())

    print("\n--- controllo dei valori nulli: ")
    print(df.isnull().sum())
    # Verifichiamo se il dataset contiene valori nulli e le stampiamo
    percent_missing = df.isnull().sum() * 100 / len(df)
    print("\n--- percentuale valori nulli: \n", percent_missing)

    print("\n--- ricerca valori unique: ")
    print(df.nunique())


def data_trasformation(df):
    """Funzione chiamata dal main.py che effettua la data transformation degli attributi Required car parking space, Repeated guest, Booking status
    sostituendo eventuali valori no, yes con 0 e 1 rispettivamente.
    Inooltre elimina dal dataset l'attributo Booking_ID """

    # Required car parking space
    # sostituisce i valori No, Yes con i valori 0, 1 rispettivamente
    df['required_car_parking_space'] = df['required_car_parking_space'].replace(0, "No")
    df['required_car_parking_space'] = df['required_car_parking_space'].replace(1, "Yes")

    # Repeated guest
    # sostituisce i valori No, Yes con i valori 0, 1 rispettivamente
    df['repeated_guest'] = df.repeated_guest.replace(0, "No")
    df['repeated_guest'] = df.repeated_guest.replace(1, "Yes")

    # Booking status
    # sostituisce i valori No, Yes con i valori 0, 1 rispettivamente
    booking_status_mapping = {'Canceled': 1, 'Not_Canceled': 0}
    df = df.replace({"booking_status": booking_status_mapping})

    # crea la lista degli attributi categorici e numerici
    cat_cols = get_categorical_columns(df)
    cat_cols.remove("Booking_ID")
    # usato per aggiungere gli elementi alla lista cat_cols
    # cat_cols.extend(["required_car_parking_space","repeated_guest"])

    cat_cols = list(set(cat_cols))
    cont_cols = [x for x in df.columns if x not in cat_cols]
    cont_cols.remove("Booking_ID")

    print("Categorical Columns :", cat_cols, "\n")
    print("Numeric Columns :", cont_cols, "\n")

    # Rimuove la colonnaBooking_ID
    df = df.drop(['Booking_ID'], axis=1)
    return df


def one_hot_encoding(dataset, lista_attributi_categorici):
    dataset = pd.get_dummies(dataset, columns=['room_type_reserved', 'repeated_guest', 'required_car_parking_space',
                                               'market_segment_type', 'type_of_meal_plan'])
    print("\n--- eseguito one-hot encoding degli attributi categorici")
    return dataset


def feature_selection(df, do_feature_importance, title):
    """Prende in ingresso il dataset, nella variabile Y salva la colonna target e va suddividere il dataset in train e test set
    inoltre permette di stampare i grafici relativi alla mutual information e feature importance degli attributi del dataset"""
    print("\n---------- " + title + " ----------")
    y = df['booking_status']  # la colonna del target Booking_status
    X = df.drop(['booking_status'], axis=1)  # tutte le colonne del dataset senza target

    # X_train e X_test sono train e test set formati da tutte le colonne del dataset senza quella target
    # y_train e y_test sono train e test set formati solo dalla colonna target
    # divisione 30% test set e 70% train set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("\n--- " + title + " diviso in train e test set")

    if do_feature_importance:
        feature_importance(X_train, y_train, title)
        print("--- Generato grafico feature importance attributi " + title)

    X_train_original = X_train
    X_test_original = X_test

    # in ingresso al modello dell'albero decisionale vengono forniti tutti gli attributi, nei commenti è presente una lista di attributi che si potrebbero utilizzare basandosi sui grafici della mutual information e feature importance
    # Features are selected based on feature importance and mutual information
    selected_features = ['lead_time', 'avg_price_per_room', 'market_segment_type_Online', 'arrival_date',
                         'no_of_special_requests', 'arrival_month', 'no_of_week_nights', 'no_of_weekend_nights',
                         'no_of_adults', 'arrival_year', 'repeated_guest_No', 'no_of_children']
    '''X_train.columns'''
    # Features are selected based on feature importance and mutual information

    X_train = X_train_original[selected_features]
    X_test = X_test_original[selected_features]

    return X_train, X_test, y_train, y_test


def modelling(X_train, X_test, y_train, y_test, title):
    print("\n------------------------------ MODELLING ------------------------------")

    # Apri il file in modalità scrittura
    with open("Performance " + title + ".txt", "w") as file:

        print("\n---------- DECISION TREE " + title + " ----------")
        print("\n---------- DECISION TREE " + title + " ----------", file=file)

        '''
        Quando si definisce un modello di classificazione Decision Tree in Python utilizzando DecisionTreeClassifier da scikit-learn, il parametro random_state specifica il generatore di numeri casuali utilizzato dal modello. 
        Il suo valore determina la riproducibilità dell'addestramento del modello.
        Il parametro random_state è opzionale e può assumere diversi valori, se viene fornito un valore specifico ad esempio random_state = 0, il modello utilizzerà sempre lo stesso generatore di numeri casuali per generare gli stessi risultati ogni volta che viene addestrato con gli stessi dati.
        Questo può essere utile per scopi di riproducibilità e per garantire che i risultati siano coerenti durante lo sviluppo e la valutazione del modello. 
        Se non viene specificato alcun valore per random_state, il modello utilizzerà un generatore di numeri casuali diverso ad ogni esecuzione, il che potrebbe portare a risultati diversi ad ogni addestramento.
        '''

        decision_tree_model = DecisionTreeClassifier(max_depth=4, random_state=0)
        decision_tree_model.fit(X_train, y_train)

        '''
        calcola e stampa un report di classificazione per valutare le prestazioni del modello Decision Tree addestrato utilizzando il set di addestramento (X_train, y_train). 
        Questo report include diverse metriche di valutazione per ogni classe nel dataset.
        Stampando il report di classificazione per il set di addestramento, si può ottenere una panoramica delle prestazioni del modello sui dati che sono stati utilizzati per addestrarlo. 
        Questo può dare un'idea di come il modello si comporta sulla stessa distribuzione di dati con cui è stato addestrato
        '''

        print("\n--- Prestazioni del modello Decision Tree applicato al set di Test: \n")
        print("\n--- Prestazioni del modello Decision Tree applicato al set di Test: \n", file=file)
        y_pred = decision_tree_model.predict(X_test)

        print_confusion_matrix(y_test, y_pred)
        print(classification_report(y_test, y_pred))
        print(classification_report(y_test, y_pred), file=file)

        # Accuratezza modello
        decision_tree_accuracy = accuracy_score(y_test, y_pred)
        print("--- Accuratezza del modello Decision Tree:", round(decision_tree_accuracy, 5))
        print("--- Accuratezza del modello Decision Tree:", round(decision_tree_accuracy, 5), file=file)

        # genera l'immagine dell'albero binario
        plt.figure(figsize=(45, 30))
        tree.plot_tree(decision_tree_model.fit(X_train, y_train), feature_names=X_train.columns,
                       class_names=['Canceled', 'Not Canceled'], filled=True, fontsize=20) # , rounded=True, fontsize=12
        #plt.title("Grafo dell'Albero Decisionale " + title, fontsize=20)  # Aggiungi il titolo sopra l'albero
        plt.show()

        # #ROC Curve
        # Calcolo delle probabilità predette per le classi positive
        y_prob = decision_tree_model.predict_proba(X_test)[:, 1]

        # Calcolo della curva ROC
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        # Calcolo dell'AUC
        roc_auc = auc(fpr, tpr)

        # Plot della curva ROC
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasso Falsi Positivi')
        plt.ylabel('Tasso Veri Positivi')
        plt.title('Curva ROC Albero Binario - ' + title)
        plt.legend(loc="lower right")
        plt.show()

        print("--- AUC del modello Decision Tree:", round(roc_auc, 5))
        print("--- AUC del modello Decision Tree:", round(roc_auc, 5), file=file)


        print("\n---------- SUPPORT VECTOR MACHINE " + title + " ----------\n")
        print("\n---------- SUPPORT VECTOR MACHINE " + title + " ----------\n", file=file)
        '''Il kernel lineare è una scelta comune per i modelli SVM e rappresenta una funzione di similarità lineare tra le istanze nel dataset. 
        Questo significa che il modello SVM cercherà di trovare un iperpiano lineare ottimale 
        per separare le diverse classi nel problema di classificazione.'''

        svm_model = SVC(kernel='linear', random_state=0)
        svm_model.fit(X_train, y_train)

        y_pred = svm_model.predict(X_test)
        print_confusion_matrix(y_test, y_pred)

        print(classification_report(y_test, y_pred))
        print(classification_report(y_test, y_pred), file=file)

        # Accuratezza modello
        svm_accuracy = accuracy_score(y_test, y_pred)
        print("--- Accuratezza del modello SVM:", round(svm_accuracy, 5))
        print("--- Accuratezza del modello SVM:", round(svm_accuracy, 5), file=file)

        # #ROC Curve
        # Calcolo delle probabilità predette per le classi positive
        y_prob = svm_model.decision_function(X_test)

        # Calcolo della curva ROC
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        # Calcolo dell'AUC
        roc_auc = auc(fpr, tpr)

        # Plot della curva ROC
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasso Falsi Positivi')
        plt.ylabel('Tasso Veri Positivi')
        plt.title('Curva ROC SVM - ' + title)
        plt.legend(loc="lower right")
        plt.show()

        print("--- AUC del modello SVM:", round(roc_auc, 5))
        print("--- AUC del modello SVM:", round(roc_auc, 5), file=file)


# Restituisce le variabili categoriche
def get_categorical_columns(df):
    cols = df.columns
    num_cols = df._get_numeric_data().columns
    return list(set(cols) - set(num_cols))


# Restituisce la distribuzione delle variabili categoriche
def get_distribution(df, cat_cols):
    for i in cat_cols:
        print("**********\n")
        print(i + ": \n")
        print(df[i].value_counts())
        print("**********\n")


# Stampa la matrice di confusione
def print_confusion_matrix(y_true, y_pred):
    cf_matrix = confusion_matrix(y_true, y_pred)

    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')
    plt.xlabel("0 - Non Cancellata , 1 - Cancellata")
    plt.ylabel("1 - Cancellata , 0 - Non Cancellata")
    plt.title("Matrice di confusione - Actual vs Predicted ")
    plt.show()


def print_value_counts(df, column_name):
    counts = df[column_name].value_counts()
    for value, count in counts.items():
        print(f"Valore {value}: {count} volte")


def print_correlation_matrix(df, title):
    correlation_matrix = df.corr().round(2)
    sns.set(rc={'figure.figsize': (23, 13)})
    ax = sns.heatmap(data=correlation_matrix, annot=True)
    ax.set_title("matrice di correlazione " + title, pad=20, y=1.05,
                 fontsize=22)  # Aggiungi pad e y per distanziare il titolo
    plt.show()


# ------------------------------ADD NULL VALUES------------------------------#
def introduce_null_values(column, percentage, df):
    if column.dtype == 'int64':
        num_null_values = int(len(column) * (percentage / 100))
        null_indices = np.random.choice(len(column), num_null_values, replace=False)
        column.loc[null_indices] = 999
    elif column.dtype == 'object':
        num_null_values = int(len(column) * (percentage / 100))
        null_indices = np.random.choice(len(column), num_null_values, replace=False)
        column.loc[null_indices] = ""
    elif column.dtype == 'float':
        num_null_values = int(len(column) * (percentage / 100))
        null_indices = np.random.choice(len(column), num_null_values, replace=False)
        column.loc[null_indices] = 999.99

    df[column.name] = column

    return df


# ------------------------------ ADD OUTLIERS INT------------------------------#
def add_outliers_int(column, percentage, df):
    # Calcola la media e la deviazione standard dei dati
    mean = df[column].mean()
    std = df[column].std()

    # Calcola i limiti degli outlier basati sul punteggio z
    z_score = 8  # Z-score per il limite degli outlier (puoi cambiare questo valore se necessario)
    lower_limit = mean - (z_score * std)
    upper_limit = mean + (z_score * std)

    # Se vuoi limitare gli outlier all'interno dei valori massimi e minimi presenti nella colonna,
    # puoi utilizzare anche i seguenti limiti:
    # lower_limit = max(df[column].min(), mean - (z_score * std))
    # upper_limit = min(df[column].max(), mean + (z_score * std))

    percentage = percentage / 100

    # Calcola il numero di righe da modificare
    n_rows = int(len(df) * percentage)

    # Inserisci gli outlier casualmente nel dataset
    outlier_indices = np.random.choice(df.index, size=n_rows, replace=False)
    outliers = np.random.uniform(lower_limit, upper_limit, size=n_rows)
    outliers = np.abs(outliers)  # Modulo per evitare valori nulli
    outliers = np.round(outliers).astype(int)  # Approssima i valori a numeri interi
    df.loc[outlier_indices, column] = outliers

    return df


# ------------------------------ ADD OUTLIERS OBJ------------------------------#
def add_categorical_outliers(column, percentage, dataset):
    # Calcola il valore outlier per la colonna specificata
    outlier_value = dataset[column].value_counts().idxmin()
    print(outlier_value)

    # Determina il numero di righe da modificare
    n_rows = int(len(dataset) * (percentage / 100))

    # Inserisci il valore outlier nella percentuale di righe specificata
    outlier_indices = dataset.sample(n_rows).index
    dataset.loc[outlier_indices, column] = outlier_value

    return dataset


# ------------------------------ ADD INCONSISTENCY------------------------------#
def introduce_inconsistencies(column_array, values_array, dataset, percentage):
    modified_dataset = dataset.copy()

    num_rows = len(modified_dataset)
    num_rows_to_modify = int(num_rows * (percentage / 100))
    rows_to_modify = random.sample(range(num_rows), num_rows_to_modify)

    for row_index in rows_to_modify:
        row = modified_dataset.loc[row_index, column_array]
        modified_dataset.loc[row_index, column_array[0]] = values_array[0]
        modified_dataset.loc[row_index, column_array[1]] = values_array[1]

    print(modified_dataset)

    return modified_dataset


# ------------------------------ ADD DUPLICATE------------------------------#
def duplicate_rows(dataset, percent):
    num_duplicates = int(len(dataset) * percent / 100)
    duplicated_rows = np.random.choice(dataset.index, size=num_duplicates, replace=True)
    duplicated_data = dataset.loc[duplicated_rows]
    dataset = pd.concat([dataset, duplicated_data], ignore_index=True)
    return dataset


# -------------------------------ADD INACCURACY------------------------------
def remove_first_letter(column, dataset, percentage):
    num_rows = int(len(dataset) * (percentage / 100))
    rows_to_modify = random.sample(range(len(dataset)), num_rows)
    rows_modified = 0

    if column.dtype == 'object':
        for i, value in enumerate(column):
            if i not in rows_to_modify:
                continue
            if value is not None and len(value) > 0:
                index = random.randint(0, len(value) - 1)
                modified_value = value[:index] + value[index + 1:]
                dataset.loc[i, column.name] = modified_value
                rows_modified += 1

    elif pd.api.types.is_integer_dtype(column):
        for i, value in enumerate(column):
            if i not in rows_to_modify:
                continue
            if pd.notnull(value):
                value_str = str(value)
                modified_value = value_str[:-1]
                if modified_value:
                    dataset.loc[i, column.name] = int(modified_value)
                    rows_modified += 1

    elif pd.api.types.is_float_dtype(column):
        for i, value in enumerate(column):
            if i not in rows_to_modify:
                continue
            if pd.notnull(value):
                value_str = str(value)
                decimal_index = value_str.index('.')
                modified_value = value_str[:decimal_index] + value_str[decimal_index + 1:]
                if modified_value:
                    dataset.loc[i, column.name] = float(modified_value)
                    rows_modified += 1

    return dataset


# ------------------------------ FEATURE IMPORTANCE------------------------------#
def feature_importance(x_train, y_train, title):
    """Feature Importance is the list of features that the model considers being important.
    It gives an importance score for each feature, depicting the importance of that feature for the prediction.
    Feature Importance is an inbuilt function in the Scikit-Learn implementation of many machine learning models.
    These feature importance scores can be used to identify the best subset of features, and then proceed with training a robust model with that subset of features."""
    
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    sort = dt.feature_importances_.argsort()
    features_sorted = np.array(x_train.columns.to_list())
    features_sorted = features_sorted[sort]
    feature_importances = pd.DataFrame({'features': features_sorted, 'importance': dt.feature_importances_[sort]})

    plt.figure(figsize=(26, 20))
    sns.barplot(data=feature_importances, y="features", x="importance",
                order=feature_importances.sort_values('importance').features)
    plt.xlabel("Importanza delle feature " + title)
    plt.title("Importanza di ciascuna feature " + title)
    plt.show()
