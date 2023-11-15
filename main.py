# pip install pandas
# pip install scikit-learn
# pip install graphviz
# pip install seaborn

import pandas as pd

import dataQualityEvaluation
import graphs_gen
import utils

''' Queste variabili definiscono le varie opzioni di esecuzione del codice potendo comprendere la creazione della matrice di correlazione, feature importance, generatore dei grafici e metodi per sporcare il dataset'''
# definizione variabili per esecuzione codice
correlationMatrix = False

# Applicare solo a dataset pulito, se True genera grafici mutal informatione e fessture importance
featureImportance = True
graphs_generator = False

# Variabili per sporcare il dataset
add_null_values = False
add_outlier_int = False
add_outlier_obj = False
add_inconsistency = False
add_duplicate = False
add_inaccuracy = False

file_path = 'Hotel_Reservations.csv'
df = pd.read_csv(file_path)

dataQualityEvaluation.dataQuality_evaluation(df, "Data Quality")


percentage_change = 10


print("\n------------------------------ ESECUZIONE CON PERCENTAGE_CHANGE = " + str(percentage_change) + " ------------------------------\n")
# Specifica il percorso del file CSV
file_path = 'Hotel_Reservations.csv'

# importa il dataset CSV
df = pd.read_csv(file_path)

# Visualizzazione grafici
if graphs_generator:
    dataset = df.copy()
    graphs_gen.show_graphs(dataset)

# ############################################DATA EXPLORATION####################################################

# utils.data_exploration(df)

# ############################################ DATA TRANSFORMATION ####################################################
print("\n------------------------------ DATA TRANSFORMATIONS ------------------------------\n")

df = utils.data_trasformation(df)

# ################### CREAZIONE DUPLICATI DATASET ###################
''' vengono creati alcuni duplicati del dataset originale, una copia usata come dataset pulito e gli altri verranno successivamente sporcati'''
dataset_pulito = df.copy()
dataset_sporco = df.copy()

# ############################################ CODICE PER SPORCARE DATASET ############################################
if add_null_values:
    print("\n------------------------------ADD NULL VALUES------------------------------")

    # Aggiunge il 35% di valori nulli a ciascuna colonna, richiamando la funzione
    dataset_sporco = utils.introduce_null_values(dataset_sporco["lead_time"], percentage_change, dataset_sporco)
    dataset_sporco = utils.introduce_null_values(dataset_sporco["arrival_month"], percentage_change, dataset_sporco)
    dataset_sporco = utils.introduce_null_values(dataset_sporco["market_segment_type"], percentage_change, dataset_sporco)

if add_outlier_int:
    print("\n-------------------------------ADD OUTLIERS INT------------------------------")
    # Aggiunge gli outlier nella colonna specificata per parametro
    dataset_sporco = utils.add_outliers_int('avg_price_per_room', percentage_change, dataset_sporco)
    dataset_sporco = utils.add_outliers_int('no_of_week_nights', percentage_change, dataset_sporco)

if add_outlier_obj:
    print("\n-------------------------------ADD OUTLIERS OBJ------------------------------")
    # Aggiunge gli outlier nella colonna specificata per parametro
    dataset_sporco = utils.add_categorical_outliers('market_segment_type', percentage_change, dataset_sporco)

if add_inconsistency:
    print("\n-------------------------------ADD INCONSISTENCY------------------------------")
    column_array, values_array = dataQualityEvaluation.check_adults_children_consistency(df)

    dataset_sporco = utils.introduce_inconsistencies(column_array, values_array, dataset_sporco, percentage_change)

if add_duplicate:
    print("\n-------------------------------ADD DUPLICATE------------------------------")
    dataset_sporco = utils.duplicate_rows(dataset_sporco, percentage_change)

if add_inaccuracy:
    print("\n-------------------------------ADD INACCURACY------------------------------")
    dataset_sporco = utils.remove_first_letter(df["type_of_meal_plan"], dataset_sporco, percentage_change)
    dataset_sporco = utils.remove_first_letter(df["no_of_special_requests"], dataset_sporco, percentage_change)
    dataset_sporco = utils.remove_first_letter(df["arrival_month"], dataset_sporco, percentage_change)

# Trasformazione attributi categorici con one-hot encoding
dataset_pulito = utils.one_hot_encoding(dataset_pulito,
                                        ['room_type_reserved', 'repeated_guest', 'required_car_parking_space',
                                         'market_segment_type', 'type_of_meal_plan'])
dataset_sporco = utils.one_hot_encoding(dataset_sporco,
                                        ['room_type_reserved', 'repeated_guest', 'required_car_parking_space',
                                         'market_segment_type', 'type_of_meal_plan'])

if correlationMatrix:
    utils.print_correlation_matrix(dataset_pulito, "Dataset Pulito")
    utils.print_correlation_matrix(dataset_sporco, "Dataset Sporco")

# ############################################ FEATURES SELECTION & MODELLING ####################################################
print("\n------------------------------ FEATURES SELECTION ------------------------------")
'''
# Feature selection using Logistic regression model
# https://towardsdatascience.com/feature-selection-using-logistic-regression-model-efc949569f58
# https://towardsdatascience.com/top-7-feature-selection-techniques-in-machine-learning-94e08730cd09
# https://www.analyticsvidhya.com/blog/2020/10/feature-selection-techniques-in-machine-learning/'''

X_train_pulito, X_test_pulito, y_train_pulito, y_test_pulito = utils.feature_selection(dataset_pulito, featureImportance, "Dataset Pulito")
utils.modelling(X_train_pulito, X_test_pulito, y_train_pulito, y_test_pulito, "Dataset Pulito")

X_train_sporco, X_test_sporco, y_train_sporco, y_test_sporco = utils.feature_selection(dataset_sporco, featureImportance, "Dataset_Sporco_Tutti_Metodi_" + str(percentage_change))
utils.modelling(X_train_sporco, X_test_pulito, y_train_sporco, y_test_pulito, "Dataset_Sporco_LeadTime_valoriNulli_" + str(percentage_change))

