import datetime
import os

import pandas as pd

file_path = 'Hotel_Reservations.csv'
df = pd.read_csv(file_path)


def dataQuality_evaluation(df, cartella):
    pulisci_cartella(cartella)

    print_to_console_and_file(
        "--------------------------------------------- METRICHE DATA-QUALITY DATASET PULITO ---------------------------------------------")
    print_to_console_and_file(
        "\n----------------------------------------------------------CONSISTENZA---------------------------------------------------------\n")

    check_adults_children_consistency(df)
    check_nights_consistency(df)
    check_car_parking_space_consistency(df)
    check_lead_time_consistency(df)
    check_special_requests_consistency(df)
    check_booking_status_consistency(df)
    check_avg_price_consistency(df)
    check_arrival_date_consistency(df)

    print_to_console_and_file(
        "\n----------------------------------------------------------COERENZA-------------------------------------------------------------\n")
    unique_value(df)

    print_to_console_and_file(
        "\n--------------------------------------------------------ACCURATEZZA-------------------------------------------------------------\n")
    check_column_types(df)

    print_to_console_and_file(
        "\n-------------------------------------------------------- INTEGRITA'--------------------------------------------------------------\n")
    check_null_values(df)
    check_duplicates(df)
    check_integrity_attributes(df)


def pulisci_cartella(cartella):
    # Verifica se la cartella esiste
    if not os.path.exists(cartella):
        print(f"La cartella {cartella} non esiste.")
        return

    # Verifica se il file "DataQuality_DatasetPulito" esiste nella cartella
    file_path = os.path.join(cartella, "DataQuality_DatasetPulito.txt")
    if not os.path.isfile(file_path):
        print(f"Il file {file_path} non esiste.")
        return

    # Elimina il file "DataQuality_DatasetPulito"
    os.remove(file_path)

    print(f"Il file {file_path} è stato eliminato.")


# Funzione per scrivere i messaggi sia sulla console che nel file
def print_to_console_and_file(message):
    path_file = "Data Quality/DataQuality_DatasetPulito"
    print(message)
    with open(path_file + ".txt", "a") as file:
        file.write(str(message) + "\n")


# ##################### PRESTAZIONI DATASET ##########################


# Viene verificata la coerenza degli attributi no_of_adults e no_di_bambini
def check_adults_children_consistency(data):
    array1 = ["no_of_adults", "no_of_children"]
    # Verifica che i valori di "no_of_adults" e "no_di_bambini" siano non negativi
    if (data['no_of_adults'] >= 0).any() or (data['no_of_children'] >= 0).any():
        print_to_console_and_file("I valori di no_of_adults e no_of_children sono non negativi.")
    else:
        print_to_console_and_file("Errore: I valori di no_of_adults e no_of_children devono essere non negativi.")

    column_array = ["no_of_adults", "no_of_children"]

    # Instanziamo in caso che non vengano trovati valori di inconsistenza
    values_array = [0, 2]

    # Verifica la relazione logica tra il numero di adulti e il numero di bambini
    invalid_relation = (data['no_of_adults'] == 0) & (data['no_of_children'] > 0)
    if invalid_relation.any():
        print_to_console_and_file("Errore: Ci sono righe con no_of_adults pari a 0 e no_of_children maggiore di 0.")
        # DEBUG: print("Righe non valide:")
        first_invalid_row = (data[invalid_relation].iloc[0])
        # DEBUG: print(data[invalid_relation])

        # Estrai i valori delle colonne "no_of_adults" e "no_of_children"
        no_of_adults_value = first_invalid_row['no_of_adults']
        no_of_children_value = first_invalid_row['no_of_children']

        # Crea un array con i valori estratti
        values_array = [no_of_adults_value, no_of_children_value]

    return column_array, values_array


# Verifica la coerenza degli attributi "no_of_weekend_nights" e "no_of_week_nights"
def check_nights_consistency(data):
    # Verifica che i valori di "no_of_weekend_nights" e "no_of_week_nights" siano non negativi
    if (data['no_of_weekend_nights'] < 0).any() or (data['no_of_week_nights'] < 0).any():
        print_to_console_and_file(
            "Errore: I valori di 'no_of_weekend_nights' e 'no_of_week_nights' devono essere non negativi.")


# Verifica la coerenza dell'attributo "required_car_parking_space"
def check_car_parking_space_consistency(data):
    # Verifica che i valori di "required_car_parking_space" siano 0 o 1
    invalid_values = (data['required_car_parking_space'] != 0) & (data['required_car_parking_space'] != 1)
    if invalid_values.any():
        print_to_console_and_file("Errore: I valori di 'required_car_parking_space' devono essere 0 o 1.")
        print_to_console_and_file("Righe non valide:")
        print_to_console_and_file(data[invalid_values])
    else:
        print_to_console_and_file("[ATTRIBUTO REQUIRED_PARKING_SPACE]: Non e' presente inconsistenza.")


# Verifica la coerenza dell'attributo "lead_time"
def check_lead_time_consistency(data):
    # Verifica che i valori di "lead_time" siano non negativi
    if (data['lead_time'] < 0).any():
        print_to_console_and_file("Errore: I valori di 'lead_time' devono essere non negativi.")
    else:
        print_to_console_and_file("[ATTRIBUTO LEAD_TIME]: Non e' presente inconsistenza.")


# Verifica la coerenza dell'attributo "no_of_special_requests"
def check_special_requests_consistency(data):
    # Verifica che i valori di "no_of_special_requests" siano non negativi
    if (data['no_of_special_requests'] < 0).any():
        print_to_console_and_file("Errore: I valori di 'no_of_special_requests' devono essere non negativi.")
    else:
        print_to_console_and_file("[ATTRIBUTO SPECIAL_REQUESTS]: Non e' presente inconsistenza.")


# Verifica la coerenza dell'attributo "booking_status"
def check_booking_status_consistency(data):
    # Verifica che i valori di "booking_status" siano "Not_Cancelled" o "Cancelled"
    invalid_values = (data['booking_status'] != "Canceled") & (data['booking_status'] != "Not_Canceled")
    if invalid_values.any():
        print_to_console_and_file("Errore: I valori di 'booking_status' devono essere 'Annullato' o 'Confermato'.")
        print_to_console_and_file("Righe non valide:")
        print_to_console_and_file(data[invalid_values])
    else:
        print_to_console_and_file("[ATTRIBUTO BOOKING_STATUS]: Non e' presente inconsistenza.")


# Verifica la coerenza dell'attributo "avg_price_per_room"
def check_avg_price_consistency(data):
    # Verifica che i valori di "avg_price_per_room" siano non negativi
    if (data['avg_price_per_room'] < 0).any():
        print_to_console_and_file("Errore: I valori di 'avg_price_per_room' devono essere non negativi.")
    else:
        print_to_console_and_file("[ATTRIBUTO AVG_PER_ROOM]: Non e' presente inconsistenza.")


# Verifica la coerenza degli attributi "arrival_year", "arrival_month" e "arrival_date"
def check_arrival_date_consistency(data):
    # Verifica il formato corretto degli attributi "arrival_anno", "arrival_month" e "arrival_date"
    invalid_format = ~data['arrival_year'].astype(str).str.match(r'^\d{4}$') | \
                     ~data['arrival_month'].astype(str).str.match(r'^\d{1,2}$') | \
                     ~data['arrival_date'].astype(str).str.match(r'^\d{1,2}$')
    if invalid_format.any():
        print_to_console_and_file(
            "Errore: I valori di 'arrival_year', 'arrival_month' e 'arrival_date' devono essere nel formato corretto.")
        print_to_console_and_file("Righe non valide:")
        print_to_console_and_file(data[invalid_format])

    # Verifica se i valori corrispondono a date valide
    invalid_dates = []
    for idx, row in data.iterrows():
        try:
            year = int(row['arrival_year'])
            month = int(row['arrival_month'])
            day = int(row['arrival_date'])
            datetime.datetime(year, month, day)
        except ValueError:
            invalid_dates.append(idx)
    if len(invalid_dates) > 0:
        print_to_console_and_file(
            "Errore: Alcuni valori di 'arrival_year', 'arrival_month' e 'arrival_date' non corrispondono a date valide.")
        print_to_console_and_file("Righe non valide:")
        print_to_console_and_file(data.loc[invalid_dates])


# ######################################## COERENZA ##########################################


# Vengono stampati i valori unici
def unique_value(df):
    # Verifica i valori nelle colonne no_of_adults e no_of_children
    print_to_console_and_file("VALORI NELL'ATTRIBUTO no_of_adults:")
    print_to_console_and_file(df['no_of_adults'].unique())

    print_to_console_and_file("VALORI NELL'ATTRIBUTO no_of_children:")
    print_to_console_and_file(df['no_of_children'].unique())

    # Verifica i valori nelle colonne arrival_year, arrival_month e arrival_date
    print_to_console_and_file("VALORI NELL'ATTRIBUTO arrival_year:")
    print_to_console_and_file(df['arrival_year'].unique())

    print_to_console_and_file("VALORI NELL'ATTRIBUTO arrival_month:")
    print_to_console_and_file(df['arrival_month'].unique())

    print_to_console_and_file("VALORI NELL'ATTRIBUTO arrival_date:")
    print_to_console_and_file(df['arrival_date'].unique())

    # Verifica i valori nella colonna booking_status
    print_to_console_and_file("VALORI NELL'ATTRIBUTO booking_status:")
    print_to_console_and_file(df['booking_status'].unique())


# ######################################## ACCURATEZZA #########################################

def check_column_types(df):
    for column_name, column_data in df.items():

        if column_data.dtype == 'object':
            if column_data.apply(lambda x: isinstance(x, str)).all():
                print_to_console_and_file(f"Tutti i valori nella colonna '{column_name}' sono di tipo stringa.")
            else:
                print_to_console_and_file(f"Errore: La colonna '{column_name}' contiene diversi tipi di dati.")

        elif column_data.dtype == 'int64':
            if column_data.apply(lambda x: isinstance(x, int)).all():
                print_to_console_and_file(f"Tutti i valori nella colonna '{column_name}' sono di tipo intero.")
            else:
                print_to_console_and_file(f"Errore: La colonna '{column_name}' contiene diversi tipi di dati.")

        elif column_data.dtype == 'float64':
            if column_data.apply(lambda x: isinstance(x, float)).all():
                print_to_console_and_file(f"Tutti i valori nella colonna '{column_name}' sono di tipo float.")
            else:
                print_to_console_and_file(f"Errore: La colonna '{column_name}' contiene diversi tipi di dati.")

        else:
            print_to_console_and_file(f"Errore: Tipo di dati non gestito per la colonna '{column_name}'.")

    print_to_console_and_file("Tutte le colonne contengono solo valori del tipo corretto.")
    return True


# ############################### INTEGRITÁ ##################################################


# VALORI NULLI
def check_null_values(df):
    missing_values = df.isnull().sum()

    # Calcola la percentuale di valori mancanti per ciascuna colonna
    missing_percentage = (missing_values / len(df)) * 100

    # Visualizza il numero e la percentuale di valori mancanti per ciascuna colonna
    print_to_console_and_file("\n--- Valori mancanti per colonna:")
    print_to_console_and_file(missing_values)
    print_to_console_and_file("\n--- Percentuale di valori mancanti per colonna:")
    print_to_console_and_file(missing_percentage)


# DUPLICATI
def check_duplicates(df):
    # Conta il numero di righe duplicate nel DataFrame
    duplicated_rows = df.duplicated().sum()

    # Visualizza il numero di righe duplicate
    print_to_console_and_file("\n--- Numero di righe duplicate:")
    print_to_console_and_file(duplicated_rows)


def check_integrity_attributes(df):
    # INTEGRITÁ COLONNA AVG PRICE PER ROOM
    # Calcola la deviazione standard per la colonna 'avg_price_per_room'
    std_dev = df['avg_price_per_room'].std()

    # Calcola la soglia per i valori outlier come 3 volte la deviazione standard sopra e sotto la media
    upper_threshold = df['avg_price_per_room'].mean() + (3 * std_dev)
    mean = df['no_of_children'].mean()
    lower_threshold = max(mean - (3 * std_dev), 0)

    # Trova i valori anomali nella colonna 'avg_price_per_room'
    anomalous_values = df[(df['avg_price_per_room'] > upper_threshold) | (df['avg_price_per_room'] < lower_threshold)]

    # Visualizza i valori anomali
    print_to_console_and_file("\n--- Valori outlier per l'attributo avg_price_per_room:")
    print_to_console_and_file(anomalous_values)

    # INTEGRITÁ COLONNA MARKET SEGMENT TYPE
    # Calcola la frequenza dei valori nell'attributo 'market_segment_type'
    segment_counts = df['market_segment_type'].value_counts()

    # Definisci la soglia per considerare un valore come anomalo (ad esempio, meno del 1% delle occorrenze)
    threshold = len(df) * 0.01
    print_to_console_and_file(threshold)

    # Seleziona solo i valori anomali nell'attributo 'market_segment_type'
    anomalous_values = segment_counts[segment_counts < threshold]

    # Seleziona solo le righe con valori anomali nell'attributo 'market_segment_type'
    anomalous_rows = df[df['market_segment_type'].isin(anomalous_values.index)]

    # Visualizza le righe con valori anomali nell'attributo 'market_segment_type'
    print_to_console_and_file("Righe con valori anomali nell'attributo 'market_segment_type':")
    print_to_console_and_file(anomalous_rows['market_segment_type'])

    # INTEGRITÁ N OF CHILDREN
    std_dev = df['no_of_children'].std()
    upper_threshold = df['no_of_children'].mean() + (
            5 * std_dev)  # SOGLIA MOLTIPLICATA PER 5: DISTRIBUZIONE DI VALORI > 2 MA NON SEMBRA UN OUTLIER -> SCRIVI NELLA RELAZIONE
    mean = df['no_of_children'].mean()
    lower_threshold = max(mean - (5 * std_dev), 0)

    anomalous_values = df[(df['no_of_children'] > upper_threshold) | (df['no_of_children'] < lower_threshold)]

    # Visualizza i valori anomali
    print_to_console_and_file("\n--- Valori outlier per l'attributo no_of_children:")
    print_to_console_and_file(anomalous_values)

    # INTEGRITÁ PER ATTRIBUTO N OF ADULTS STD
    std_dev = df['no_of_adults'].std()
    upper_threshold = df['no_of_adults'].mean() + (
            3 * std_dev)  # SOGLIA MOLTIPLICATA PER 5: DISTRIBUZIONE DI VALORI > 2 MA NON SEMBRA UN OUTLIER -> SCRIVI NELLA RELAZIONE
    mean = df['no_of_adults'].mean()
    lower_threshold = max(mean - (3 * std_dev), 0)

    anomalous_values = df[(df['no_of_adults'] > upper_threshold) | (df['no_of_adults'] < lower_threshold)]

    # Visualizza i valori anomali
    print_to_console_and_file("\n--- Valori outlier per l'attributo no_of_adults:")
    print_to_console_and_file(anomalous_values)

    # INTEGRITÁ RICHIESTE SPECIALI
    # Definisci il range tipico per il numero di richieste speciali
    std_dev = df['no_of_special_requests'].std()
    upper_threshold = df['no_of_special_requests'].mean() + (
            3 * std_dev)  # SOGLIA MOLTIPLICATA PER 5: DISTRIBUZIONE DI VALORI > 2 MA NON SEMBRA UN OUTLIER -> SCRIVI NELLA RELAZIONE
    mean = df['no_of_special_requests'].mean()
    lower_threshold = max(mean - (3 * std_dev), 0)

    anomalous_values = df[
        (df['no_of_special_requests'] > upper_threshold) | (df['no_of_special_requests'] < lower_threshold)]

    # Visualizza i valori anomali
    print_to_console_and_file("\n--- Valori outlier per l'attributo no_of_special_requests:")
    print_to_console_and_file(anomalous_values)
