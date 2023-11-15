import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def show_graphs(df):
    # #1. PRIMO GRAFICO: Percentuale di prenotazioni cancellate/non cancellate

    # Get the total number of reservations
    total_reservations = df["booking_status"].count()

    # Get the number of canceled reservations
    canceled_reservations = df["booking_status"].value_counts()["Canceled"]

    # Get the number of non-canceled reservations
    non_canceled_reservations = total_reservations - canceled_reservations

    # Create a list of labels for the pie chart
    labels = ["Prenotazioni Cancellate", "Prenotazioni Mantenute"]

    # Create a list of sizes for the pie chart
    sizes = [canceled_reservations, non_canceled_reservations]
    # Create a pie chart with 2D effect
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%.1f%%", shadow=False, colors=["#90e0ef", "#0077b6"], textprops={'fontsize': 25})

    ax.set_title("Distribuzione Prenotazioni")

    # Show the pie chart
    plt.show()

    # #2. SECONDO GRAFICO: Numero di adulti presenti nelle prenotazioni e numero di bambini

    # Apply the default theme
    sns.set_theme()
    # todo vedere che cambia
    # sns.set(rc={'figure.figsize': (10.7, 7.27)})

    # Definiamo la palette di default 
    palette_default = sns.color_palette("viridis")
    sns.set_palette("viridis")

    # Get the number of adults on each reservation
    adults_per_reservation = df["no_of_adults"].value_counts()

    # Get the number of childrens on each reservation
    childrens_per_reservation = df["no_of_children"].value_counts()

    # Create a figure with two axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the adults graph on the first axis
    sns.barplot(x=adults_per_reservation.index, y=adults_per_reservation.values, ax=ax1)

    # Add labels to the adults graph
    ax1.set_title("Numero di Adulti nella Prenotazione", fontsize = 20)
    ax1.set_xlabel("Numero di Adulti", fontsize = 20)
    ax1.set_ylabel("Numero di Prenotazioni", fontsize = 20)

    # Add the counter above each bar on the adults graph
    for i, v in enumerate(adults_per_reservation.values):
        ax1.annotate(str(v), xy=(adults_per_reservation.index[i], v), xytext=(0, 10), textcoords="offset points",
                     ha="center", va="bottom", fontsize=18)

    # Change the background color of the adults graph
    ax1.set_facecolor("#E0E0E0")

    # Add a border around the adults graph
    ax1.set_frame_on(True)

    # Add a legend to the adults graph
    ax1.legend(loc="upper right")

    # Plot the childrens graph on the second axis
    sns.barplot(x=childrens_per_reservation.index, y=childrens_per_reservation.values, ax=ax2)

    # Add labels to the childrens graph
    ax2.set_title("Numero di Bambini nella Prenotazione", fontsize = 20)
    ax2.set_xlabel("Numero di Bambini", fontsize = 20)
    ax2.set_ylabel("Numero di Prenotazioni", fontsize = 20)

    # Add the counter above each bar on the childrens graph
    for i, v in enumerate(childrens_per_reservation.values):
        ax2.annotate(str(v), xy=(childrens_per_reservation.index[i], v), xytext=(0, 10), textcoords="offset points",
                     ha="center", va="bottom", fontsize=18)

    # Change the background color of the adults graph
    ax2.set_facecolor("#E0E0E0")

    # Add a border around the adults graph
    ax2.set_frame_on(True)

    # Add a legend to the adults graph
    ax2.legend(loc="upper right")

    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=20)
    ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=20)

    ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=20)
    ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=20)

    # Show the graph
    plt.show()

    # #3. TERZO GRAFICO: visualizzazione cancellazioni per anno (2017)

    # Filtra il DataFrame per le prenotazioni con "arrival_year" uguale a 2017
    filtered_df = df[df["arrival_year"] == 2017]

    # Calcola il numero totale di prenotazioni
    total_reservations = filtered_df["booking_status"].count()

    # Calcola il numero di prenotazioni cancellate
    canceled_reservations = filtered_df["booking_status"].value_counts()["Canceled"]

    # Calcola il numero di prenotazioni non cancellate
    non_canceled_reservations = total_reservations - canceled_reservations

    # Crea una lista di etichette per il grafico a torta
    labels = ["Prenotazioni Cancellate", "Prenotazioni Mantenute"]

    # Crea una lista di dimensioni per il grafico a torta
    sizes = [canceled_reservations, non_canceled_reservations]

    # Crea un grafico a torta con effetto 2D
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%.1f%%", shadow=False, colors=["#90e0ef", "#0077b6"], textprops={'fontsize': 25})

    ax.set_title("Distribuzione prenotazioni 2017")

    # Mostra il grafico a torta
    plt.show()

    # #3Bis. TERZO GRAFICO: visualizzazione cancellazioni per anno (2018)

    # Filtra il DataFrame per le prenotazioni con "arrival_year" uguale a 2018
    filtered_df = df[df["arrival_year"] == 2018]

    # Calcola il numero totale di prenotazioni
    total_reservations = filtered_df["booking_status"].count()

    # Calcola il numero di prenotazioni cancellate
    canceled_reservations = filtered_df["booking_status"].value_counts()["Canceled"]

    # Calcola il numero di prenotazioni non cancellate
    non_canceled_reservations = total_reservations - canceled_reservations

    # Crea una lista di etichette per il grafico a torta
    labels = ["Prenotazioni Cancellate", "Prenotazioni Mantenute"]

    # Crea una lista di dimensioni per il grafico a torta
    sizes = [canceled_reservations, non_canceled_reservations]

    # Crea un grafico a torta con effetto 2D
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%.1f%%", shadow=False, colors=["#90e0ef", "#0077b6"], textprops={'fontsize': 25})

    ax.set_title("Distribuzione prenotazioni 2018")

    # Mostra il grafico a torta
    plt.show()


    # #4. QUARTO GRAFICO: visualizzazione di percentuale di prenotazioni cancellate in confronto con il numero di bambini
    # presenti nella prenotazioni

    # Conteggio delle prenotazioni cancellate con bambini e senza bambini
    canceled_with_children = len(df[(df["booking_status"] == "Canceled") & (df["no_of_children"] > 0)])
    canceled_with_no_children = len(df[(df["booking_status"] == "Canceled") & (df["no_of_children"] == 0)])

    # Conteggio delle prenotazioni non cancellate con bambini e senza bambini
    not_canceled_with_children = len(df[(df["booking_status"] == "Not_Canceled") & (df["no_of_children"] > 0)])
    not_canceled_with_no_children = len(df[(df["booking_status"] == "Not_Canceled") & (df["no_of_children"] == 0)])

    # Verifichiamo le percentuali
    total_reservations_with_children = len(df[df["no_of_children"] > 0])
    total_reservations_with_no_children = len(df[df["no_of_children"] == 0])

    no_reservations_canceled_with_children = (canceled_with_children / total_reservations_with_children) * 100
    no_reservations_canceled_without_children = (canceled_with_no_children / total_reservations_with_no_children) * 100

    print("percentuale di prenotazioni cancellate con bambini: ", no_reservations_canceled_with_children)
    print("percentuale di prenotazioni cancellate senza bambini: ", no_reservations_canceled_without_children)

    # Etichette per le categorie
    categories = ["Senza Bambini", "Con Bambini"]

    # Dati da visualizzare sul grafico
    canceled_data = [canceled_with_no_children, canceled_with_children]
    not_canceled_data = [not_canceled_with_no_children, not_canceled_with_children]

    # Posizioni delle barre
    bar_positions = [0, 1]

    # Larghezza delle barre
    bar_width = 0.4

    # Creazione del grafico a barre con colori diversi per le categorie
    plt.bar(bar_positions, canceled_data, label="Canceled", color="red", width=bar_width)
    plt.bar(bar_positions, not_canceled_data, label="Not Canceled", color="green", width=bar_width,
            bottom=canceled_data)

    plt.xlabel("Tipologia Prenotazione", fontsize=20)
    plt.ylabel("Occorrenze", fontsize=20)
    plt.title("Stato prenotazioni con bambini", fontsize=20)
    plt.xticks(bar_positions, categories, fontsize=25)
    plt.legend()

    # Aggiunta dei valori all'interno delle barre rosse
    plt.text(bar_positions[0], canceled_data[0],
             str(canceled_data[0]) + "\n(" + str(round(no_reservations_canceled_without_children, 2)) + "%)",
             ha='center',
             va='bottom', color='white', fontsize=18)
    plt.text(bar_positions[1], canceled_data[1],
             str(canceled_data[1]) + "\n(" + str(round(no_reservations_canceled_with_children, 2)) + "%)", ha='center',
             va='bottom', color='white', fontsize=18)

    # Visualizzazione del grafico
    plt.show()

    # 5. QUINTO GRAFICO: visualizzazione della permanenza degli ospiti nell'hotel

    # Impostazione della palette viridis
    sns.set_palette("viridis")

    # Calcolo del numero di prenotazioni per ciascun numero di notti di permanenza
    weekend_nights_count = df["no_of_weekend_nights"].value_counts()
    weekend_nights_count = weekend_nights_count.sort_index()

    # Calcolo del numero di prenotazioni per ciascun numero di notti della settimana
    week_nights_count = df["no_of_week_nights"].value_counts()
    week_nights_count = week_nights_count.sort_index()

    # Creazione dei subplot per disporre i grafici affiancati
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Grafico per il numero di notti di permanenza nel weekend
    sns.barplot(x=weekend_nights_count.index, y=weekend_nights_count.values, ax=ax1)
    ax1.set_xlabel("Numero di Notti di Permanenza nel Weekend")
    ax1.set_ylabel("Numero di Prenotazioni")
    ax1.set_title("Distribuzione Prenotazioni per Numero di Notti nel Weekend")

    # Aggiunta del numero delle prenotazioni sopra ogni barra nel primo grafico
    for i, v in enumerate(weekend_nights_count.values):
        ax1.text(i, v, str(v), ha='center', va='bottom')

    # Grafico per il numero di notti di permanenza durante la settimana
    sns.barplot(x=week_nights_count.index, y=week_nights_count.values, ax=ax2)
    ax2.set_xlabel("Numero di Notti di Permanenza durante la Settimana")
    ax2.set_ylabel("Numero di Prenotazioni")
    ax2.set_title("Distribuzione Prenotazioni per Numero di Notti settimanali")

    # Aggiunta del numero delle prenotazioni sopra ogni barra nel secondo grafico
    for i, v in enumerate(week_nights_count.values):
        ax2.text(i, v, str(v), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # 6. SESTO GRAFICO: visualizzazione della richiesta del parking space

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Grafico 1: Distribuzione delle Prenotazioni in base alla richiesta di parcheggio
    parking_space_labels = {0: "No", 1: "Si"}
    df["required_car_parking_space_label"] = df["required_car_parking_space"].map(parking_space_labels)
    parking_space_count = df["required_car_parking_space_label"].value_counts().sort_index()

    sns.barplot(x=parking_space_count.index, y=parking_space_count.values, ax=axes[0])

    axes[0].set_xticklabels(parking_space_labels.values(), fontsize=20)
    axes[0].set_xlabel("Richiesta parcheggio", fontsize = 20)
    axes[0].set_ylabel("Numero di prenotazioni", fontsize = 20)
    axes[0].set_title("Distribuzione Prenotazioni in base alla richiesta di parcheggio", fontsize = 20)

    for i, v in enumerate(parking_space_count.values):
        axes[0].text(i, v, str(v), ha='center', va='bottom', fontsize = 20)

    # Grafico 2: Relazione tra lo stato di cancellazione delle prenotazioni e la richiesta di parcheggio
    canceled_with_parking = len(df[(df["booking_status"] == "Canceled") & (df["required_car_parking_space"] == 1)])
    canceled_without_parking = len(df[(df["booking_status"] == "Canceled") & (df["required_car_parking_space"] == 0)])
    not_canceled_with_parking = len(
        df[(df["booking_status"] == "Not_Canceled") & (df["required_car_parking_space"] == 1)])
    not_canceled_without_parking = len(
        df[(df["booking_status"] == "Not_Canceled") & (df["required_car_parking_space"] == 0)])

    data = pd.DataFrame({
        "Parking": ["Without Parking Space", "With Parking Space"],
        "Canceled": [canceled_without_parking, canceled_with_parking],
        "Not Canceled": [not_canceled_without_parking, not_canceled_with_parking]
    })

    melted_data = data.melt(id_vars="Parking", var_name="Reservation Status", value_name="Count")

    sns.barplot(x="Parking", y="Count", hue="Reservation Status", data=melted_data, ax=axes[1])

    axes[1].tick_params(axis='x', labelsize=20)
    axes[1].set_ylabel("Numero di prenotazioni", fontsize = 20)
    axes[1].set_title("Stato prenotazioni in base alla richiesta del parcheggio", fontsize = 20)

    for p in axes[1].patches:
        axes[1].annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2, p.get_height()), ha='center',
                         va='bottom', fontsize = 25)

    # Regolare il layout dei grafici
    plt.tight_layout()

    # Mostrare l'immagine con i due grafici affiancati
    plt.show()
    df = df.drop(['required_car_parking_space_label'], axis=1)

    # 7. SETTIMO GRAFICO: Tipi di camera prenotate

    # Calcolo del numero di prenotazioni per ciascun tipo di camera
    room_type_count = df["room_type_reserved"].value_counts().sort_index()

    # Creazione del grafico utilizzando Seaborn
    sns.barplot(x=room_type_count.index, y=room_type_count.values)

    plt.xlabel("Tipo di Camera")
    plt.ylabel("Numero di Prenotazioni")
    plt.title("Distribuzione dei Tipi di Camera")

    # Aggiunta del numero delle prenotazioni sopra ogni barra
    for i, v in enumerate(room_type_count.values):
        plt.text(i, v, str(v), ha='center', va='bottom')

    plt.xticks(rotation=45)  # Rotazione delle etichette sull'asse x per una migliore leggibilità

    plt.show()

    # 8. OTTAVO GRAIFICO: Giorni che intercorrono tra la prenotazione e il giorno di arrivo

    # Calcolo del numero di prenotazioni per ciascun lead time
    lead_time_count = df["lead_time"].value_counts().sort_index()

    # Creazione del grafico utilizzando Seaborn
    sns.barplot(x=lead_time_count.index, y=lead_time_count.values)

    plt.xlabel("Lead Time")
    plt.ylabel("Numero di Prenotazioni")
    plt.title("Distribuzione del Lead Time")

    plt.xticks(rotation=45)  # Rotazione delle etichette sull'asse x per una migliore leggibilità
    plt.xticks(range(0, len(lead_time_count.index), 100), lead_time_count.index[::100])

    plt.show()

    # 9. NONO GRAFICO: visualizzazione delle prenotazioni in base al mese di prenotazione

    # Mappatura dei valori numerici dell'attributo "arrival_month" ai nomi dei mesi corrispondenti
    month_labels = {
        1: "Gennaio",
        2: "Febbraio",
        3: "Marzo",
        4: "Aprile",
        5: "Maggio",
        6: "Giugno",
        7: "Luglio",
        8: "Agosto",
        9: "Settembre",
        10: "Ottobre",
        11: "Novembre",
        12: "Dicembre"
    }
    df["arrival_month_label"] = df["arrival_month"].map(month_labels)

    # Creazione della colonna "arrival_month_label" come categoria ordinata
    df["arrival_month_label"] = pd.Categorical(df["arrival_month_label"], categories=list(month_labels.values()),
                                               ordered=True)

    # Calcolo del numero di prenotazioni per ciascun mese di arrivo
    monthly_arrival_count = df["arrival_month_label"].value_counts().sort_index()

    # Creazione del grafico utilizzando Seaborn
    sns.barplot(x=monthly_arrival_count.index, y=monthly_arrival_count.values)

    plt.xlabel("Mese di Arrivo")
    plt.ylabel("Numero di Prenotazioni")
    plt.title("Distribuzione delle Prenotazioni per Mese di Arrivo")

    plt.xticks(rotation=45)  # Rotazione delle etichette sull'asse x per una migliore leggibilità

    plt.show()
    df = df.drop(['arrival_month_label'], axis=1)

    # 10 DECIMO GRAFICO: Distinzione tra alta e bassa stagione e relazione tra cancellazione e non cancellazione

    # Creazione di un nuovo dataframe con le prenotazioni in alta stagione
    high_season_df = df[(df["arrival_month"] >= 5) & (df["arrival_month"] <= 10)]
    high_season_count = len(high_season_df)

    # Creazione di un nuovo dataframe con le prenotazioni in bassa stagione
    low_season_df = df[~((df["arrival_month"] >= 5) & (df["arrival_month"] <= 10))]
    low_season_count = len(low_season_df)

    # Calcolo del numero di prenotazioni cancellate e non cancellate per alta stagione
    high_season_canceled = len(high_season_df[high_season_df["booking_status"] == "Canceled"])
    high_season_not_canceled = len(high_season_df[high_season_df["booking_status"] == "Not_Canceled"])

    # Calcolo del numero di prenotazioni cancellate e non cancellate per bassa stagione
    low_season_canceled = len(low_season_df[low_season_df["booking_status"] == "Canceled"])
    low_season_not_canceled = len(low_season_df[low_season_df["booking_status"] == "Not_Canceled"])

    no_reservations_canceled_high_season = (high_season_canceled / high_season_count) * 100
    no_reservations_canceled_low_season = (low_season_canceled / low_season_count) * 100

    # Dati da visualizzare sul grafico
    seasons = ["Alta stagione", "Bassa stagione"]
    canceled_data = [high_season_canceled, low_season_canceled]
    not_canceled_data = [high_season_not_canceled, low_season_not_canceled]

    # Posizioni delle barre
    bar_positions = [0, 1]

    # Larghezza delle barre
    bar_width = 0.4

    # Creazione del grafico a barre affiancate utilizzando Seaborn
    plt.bar(bar_positions, canceled_data, label="Canceled", color="red", width=bar_width)
    plt.bar(bar_positions, not_canceled_data, label="Not Canceled", color="green", width=bar_width,
            bottom=canceled_data)

    # Aggiunta delle etichette delle percentuali sopra le barre rosse
    plt.text(bar_positions[0], canceled_data[0], f"{no_reservations_canceled_high_season:.2f}%", ha="center",
             va="bottom", color="white")
    plt.text(bar_positions[1], canceled_data[1], f"{no_reservations_canceled_low_season:.2f}%", ha="center",
             va="bottom", color="white")

    plt.xlabel("Stagione di arrivo")
    plt.ylabel("Count")
    plt.title("Reservation Status per Stagione di Arrivo")
    plt.xticks(bar_positions, seasons)
    plt.legend()

    plt.show()

    # #11. UNDICESIMO GRAFICO: visualizzazione delle prenotazioni cancellate o meno in base al mese di arrivo

    # Calcolo del numero di prenotazioni cancellate per ogni mese
    canceled_counts = df[df["booking_status"] == "Canceled"]["arrival_month"].value_counts().sort_index()

    # Calcolo del numero di prenotazioni non cancellate per ogni mese
    not_canceled_counts = df[df["booking_status"] == "Not_Canceled"]["arrival_month"].value_counts().sort_index()

    # Posizioni delle barre
    bar_positions = list(range(len(month_labels)))

    # Larghezza delle barre
    bar_width = 0.35

    # Creazione del grafico a barre affiancate
    plt.bar(bar_positions, canceled_counts, label="Canceled", color="#ff4d4d", width=bar_width)
    plt.bar([p + bar_width for p in bar_positions], not_canceled_counts, label="Not Canceled", color="#5cd65c",
            width=bar_width)

    plt.xlabel("Mese di Arrivo")
    plt.ylabel("Count")
    plt.title("Prenotazioni Cancellate e Non Cancellate per Mese di Arrivo")
    plt.xticks(bar_positions, list(month_labels.values()), rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 12. DODICESIMO GRAFICO: visualizzazioni delle prenotazioni suddivise in base al segment type

    # Calcolo del numero di prenotazioni per ogni tipo di segmento di mercato
    segment_counts = df["market_segment_type"].value_counts().sort_values()

    # Creazione del grafico con Seaborn
    ax = sns.countplot(x="market_segment_type", data=df, order=segment_counts.index)

    plt.xlabel("Tipo di Segmento di Mercato")
    plt.ylabel("Numero di Prenotazioni")
    plt.title("Suddivisione delle Prenotazioni per Segmento di Mercato")

    plt.xticks(rotation=45)  # Rotazione delle etichette sull'asse x per una migliore leggibilità

    # Aggiunta del numero sopra ogni barra
    for i, count in enumerate(segment_counts):
        ax.annotate(count, (i, count), ha='center', va='bottom')

    plt.show()

    # 13. TREDICESIMO GRAFICO: Percentuale di prenotazioni cancellate in base al segmento di mercato

    booking_counts = df["market_segment_type"].value_counts()
    print(booking_counts)
    canceled_counts = df[df["booking_status"] == "Canceled"]["market_segment_type"].value_counts()
    print(canceled_counts)
    not_canceled_counts = df[df["booking_status"] == "Not_Canceled"]["market_segment_type"].value_counts()
    print(not_canceled_counts)

    # Creazione del dataframe con i dati
    data = pd.DataFrame({'Canceled': canceled_counts, 'Not Canceled': not_canceled_counts})

    bar_positions = np.arange(len(data.index))

    # Larghezza delle barre
    bar_width = 0.35

    # Creazione del grafico a barre
    plt.bar(bar_positions, data["Not Canceled"], color="green", label="Not Canceled")
    plt.bar(bar_positions, data["Canceled"], color="red", label="Canceled", bottom=data["Not Canceled"])

    # Aggiunta delle etichette sopra le barre
    for i, count in enumerate(data["Canceled"]):
        if count > 0:
            plt.text(i, count, str(count), ha='center', va='bottom', fontsize=20)
        else:
            plt.text(i, count, str(count), ha='center', va='bottom', fontsize=20, color='black')

    # Personalizzazione del grafico
    plt.xlabel("Market Segment Type", fontsize = 20)
    plt.ylabel("Numero di prenotazioni", fontsize = 20)
    plt.title("Suddivisione delle Prenotazioni per Segmento di Mercato", fontsize = 20)
    plt.xticks(bar_positions, data.index, rotation=45, fontsize = 17)  # Etichette sull'asse x
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 14. QUATTORDICESIMO GRAFICO: prenotazioni effettuate in base a nuovo/vecchio cliente

    # Creazione della figura e degli assi
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

    # PRIMO GRAFICO: Prenotazioni effettuate in base a nuovo/vecchio cliente
    guest_counts = df["repeated_guest"].value_counts()
    data1 = pd.DataFrame({'Count': guest_counts, 'Repeated Guest': guest_counts.index})
    sns.barplot(data=data1, x='Repeated Guest', y='Count', ax=ax1)
    ax1.set_xlabel("Ospite ripetuto")
    ax1.set_ylabel("Numero di prenotazioni")
    ax1.set_title("Distribuzioni delle prenotazioni di un ospite ripetuto")

    # Aggiungi le etichette sopra le barre
    for i, count in enumerate(guest_counts):
        ax1.text(i, count, str(count), ha='center', va='bottom')

    # SECONDO GRAFICO: Prenotazioni cancellate/non cancellate in base a nuovo/vecchio cliente
    canceled_bookings = df[df["booking_status"] == "Canceled"]
    not_canceled_bookings = df[df["booking_status"] == "Not_Canceled"]

    canceled_counts = canceled_bookings["repeated_guest"].value_counts()
    not_canceled_counts = not_canceled_bookings["repeated_guest"].value_counts()

    data2 = pd.DataFrame(
        {'Canceled': canceled_counts, 'Not Canceled': not_canceled_counts, 'Repeated Guest': canceled_counts.index})
    melted_data = data2.melt(id_vars='Repeated Guest', var_name='Booking Status', value_name='Count')
    sns.barplot(data=melted_data, x='Repeated Guest', y='Count', hue='Booking Status', ax=ax2)
    ax2.set_xlabel("Ospite ripetuto")
    ax2.set_ylabel("Numero di prenotazioni")
    ax2.set_title("Prenotazioni da ospiti ripetuti e stato delle prenotazioni")

    # Aggiungi le etichette sopra le colonne
    for p in ax2.patches:
        height = p.get_height()
        if height > 0:
            ax2.annotate(str(int(height)), (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')

    # Mostra la figura con entrambi i grafici
    plt.tight_layout()
    plt.show()

    # 15. QUINDICESIMO GRAFICO: Verifica delle prenotazioni in base al numero di previous cancellations

    # Calcola il conteggio delle prenotazioni per ogni valore di "previus_cancellations"
    cancellation_counts = df["no_of_previous_cancellations"].value_counts()

    # Ordina i valori per il numero di cancellazioni in ordine crescente
    cancellation_counts = cancellation_counts.sort_index()

    # Crea il grafico utilizzando Seaborn
    sns.barplot(x=cancellation_counts.index, y=cancellation_counts)

    # Aggiungi le etichette sopra le barre
    for i, count in enumerate(cancellation_counts):
        plt.text(i, count, str(count), ha='center', va='bottom')

    # Personalizza il grafico
    plt.xlabel("Numero di prenotazioni precedentemente cancellate")
    plt.ylabel("Numero di prenotazioni")
    plt.title("Distribuzione delle prenotazioni in base al numero di prenotazioni precedentemente cancellate")

    # Mostra il grafico
    plt.tight_layout()
    plt.show()

    # 16. SEDICESIMO GRAFICO: prezzo medio per camera

    # Crea il grafico utilizzando Seaborn
    sns.boxplot(x=df["avg_price_per_room"])

    # Personalizza il grafico
    plt.xlabel("Distribuzione del prezzo medioper stanza")
    plt.ylabel("Prezzo")
    plt.title("Distribuzione del prezzo medio per stanze")

    # Mostra il grafico
    plt.tight_layout()
    plt.show()

    # 17. DICIASSETTESIMO GRAFICO: cancellazioni in base al prezzo
    # Crea le fasce di prezzo desiderate
    price_ranges = pd.interval_range(start=0, end=550, freq=50, closed='left')

    # Aggiungi la colonna della fascia di prezzo al DataFrame
    df['price_range_category'] = pd.cut(df['avg_price_per_room'], bins=price_ranges)

    # Raggruppa per la colonna della fascia di prezzo e calcola il conteggio delle cancellazioni
    canceled_counts = df[df["booking_status"] == "Canceled"]["price_range_category"].value_counts().sort_index()

    # Creazione del grafico a barre
    sns.barplot(x=canceled_counts.index, y=canceled_counts.values)

    # Personalizzazione delle etichette dell'asse x
    plt.xticks(range(len(canceled_counts.index)), canceled_counts.index.astype(str))

    # Personalizzazione del grafico
    plt.xlabel("Range dei prezzi")
    plt.ylabel("Conteggio prenotazioni cancellate")
    plt.title("Prenotazioni cancellate in base al range di prezzi")

    # Mostra il grafico
    plt.tight_layout()
    plt.show()

    df = df.drop(['price_range_category'], axis=1)

    # 18. DIOCIOTTESIMO GRAFICO: Numero di richieste speciali
    sns.boxplot(x=df["no_of_special_requests"])

    # Personalizza il grafico
    plt.xlabel("Distribuzione delle richieste speciali")
    plt.ylabel("Numero di richieste speciali")
    plt.title("Distribuzione della media del numero di richieste speciali")

    # Mostra il grafico
    plt.tight_layout()
    plt.show()

    # 19. DICIANNOVESIMO GRAFICO: Visualizzazione delle prenotazioni cancellate/non cancellate in base al numero di richieste speciali
    # Creazione del grafico a barre
    sns.countplot(data=df, x="no_of_special_requests", hue="booking_status")

    # Personalizzazione del grafico
    plt.xlabel("Numero di richieste speciali", fontsize = 20)
    plt.ylabel("Numero di prenotazioni", fontsize = 20)
    plt.title("Richieste speciali e prenotazioni", fontsize = 20)
    plt.legend(title="Booking Status")

    # Aggiunta delle etichette sopra le barre
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height}", (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', fontsize = 20)

    # Mostra il grafico
    plt.tight_layout()
    plt.show()
