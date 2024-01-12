#READ ALL CVS FROM ENERGIES DIRECTORY AND UNION IN ONE DATASET
import pandas as pd, os, re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import timedelta as dt

def calcolaEnergiaPerOgniSensore(dfMain):
    listaDataframe=[]
    listaNomiSensori = []
    everyTarget_df = dfMain.groupby("target")

    for group_name, group_data in everyTarget_df:
        print(f"Group: {group_name}")
        listaNomiSensori.append(group_name)
        #raggruppo per ogni gruppo sommo tutti quelli della run 0 e ottengo potenza del target
        everyRun = group_data.groupby("directory")
        group_data.insert(3, "numero_istanze", 0, True)
        group_data.insert(4, "energia", 0, True)
        result_df = everyRun.agg({'power': 'sum','numero_istanze': 'count'}).reset_index()
        result_df["energia"] = (result_df["power"] * result_df["numero_istanze"] * 0.00027) #calcolo dell'energia
        print(result_df.to_string())
        print("--------------------")
        listaDataframe.append(result_df)
        result_df.to_csv("./separatedDatasets/"+group_name+".csv")

    return listaDataframe, listaNomiSensori

def calcolaEnergiaPerOgniRun(lista,listaSensori):
    datasetTotal = pd.DataFrame(columns=list(range(51)))
    i=0

    for dataset in lista:
        dataset=dataset.drop('power', axis=1)
        dataset=dataset.drop('numero_istanze', axis=1)
        datasetTotal.loc[listaSensori[i]] = dataset.T.loc["energia"]
        i+=1

    datasetTotal.to_csv("./separateRun/uniqueDataset.csv")

    return datasetTotal


def createUniqueDataset(path):
    os.chdir("..")
    dfMain = pd.DataFrame({'name': [], 'time': [], 'power': [], 'sensor': [], 'target': [], 'directory': []})
    os.chdir(path)

    for root, dirs, files in os.walk("./Energies"):
        for filename in files:
            #print(os.path.join(root, filename))
            path=os.path.join(root, filename)
            df = pd.read_csv(path, index_col=False)
            m = re.search("[\d]+", path.split("/")[2]).group()
            #print(m)
            df.insert(5,"directory",int(m),True)
            dfMain=pd.concat([df,dfMain], axis=0, sort=False, ignore_index=True)

    dfMain['time'] = pd.to_datetime(dfMain['time'])
    dfMain=dfMain.reset_index(drop=True)
    dfMain=dfMain.sort_values(by=['directory','time'],ascending=True,ignore_index=True )
    os.chdir("..")
    os.chdir("..")
    os.chdir("./datasetUnion/datasets")

    dfMain.to_csv("mainDataset.csv")
    return dfMain

def EnergyEvolutionForEveryRun(df,dfAF):
    #NON VEDO CONSIDERAZIONI INTERESSANTI DA FARE POST REFACTORING
    fig, axes = plt.subplots(nrows=2, ncols=1)
    box = dict(facecolor='yellow', pad=5, alpha=0.2)

    ax1 = axes[0]
    ax1.plot(df)
    ax1.set_title("Andamento dell'energia per ogni Run (PreRefactoring)", fontweight='bold')

    ax1.set_ylabel("Energia in J", bbox=box)

    ax2 = axes[1]
    ax2.plot(df)
    ax2.set_title("Andamento dell'energia per ogni Run (PostRefactoring)", fontweight='bold')

    ax2.set_ylabel("Energia in J", bbox=box)

    plt.suptitle("Evoluzione dell'energia per ogni run pre e post refactoring", fontsize=16)
    plt.show()



def legend():
    plt.xlabel('Numero Run')
    plt.ylabel('Energia Media per Run')
    plt.title('Multiple Lines Plot')
    plt.legend()
def plotForEverySensor(df,dfA):
    # blu pre refactoring e arancio post
    fig, axes = plt.subplots(nrows=4, ncols=2)
    box = dict(facecolor='yellow', pad=5, alpha=0.2)

    ax1 = axes[0, 0]
    ax1.plot(df.loc["crate-container-cd"], color="blue")
    ax1.plot(dfA.loc["crate-container-cd"], color="orange")
    ax1.set_title('crate-container-cd', fontweight='bold')
    ax1.set_ylabel('Energia Media per Run', bbox=box)
    ax1.set_xlabel('Numero della Run', bbox=box)

    ax2 = axes[0,1]
    ax2.plot(df.loc["global"], color="blue")
    ax2.plot(dfA.loc["global"], color="orange")
    ax2.set_title('global', fontweight='bold')
    ax2.set_ylabel('Energia Media per Run', bbox=box)
    ax2.set_xlabel('Numero della Run', bbox=box)


    ax3 = axes[1, 0]
    ax3.plot(df.loc["hwpc-sensor-container"], color="blue")
    ax3.plot(dfA.loc["hwpc-sensor-container"], color="orange")
    ax3.set_title('hwpc-sensor-container', fontweight='bold')
    ax3.set_ylabel('Energia Media per Run', bbox=box)
    ax3.set_xlabel('Numero della Run', bbox=box)


    ax4 = axes[1, 1]
    ax4.plot(df.loc["influx_dest"], color="blue")
    ax4.plot(dfA.loc["influx_dest"], color="orange")
    ax4.set_title('influx_dest Pre Refactoring', fontweight='bold')
    ax4.set_ylabel('Energia Media per Run', bbox=box)
    ax4.set_xlabel('Numero della Run', bbox=box)

    ax5 = axes[2, 0]
    ax5.plot(df.loc["mongo_source"], color="blue")
    ax5.plot(dfA.loc["mongo_source"], color="orange")
    ax5.set_title('mongo_source', fontweight='bold')
    ax5.set_ylabel('Energia Media per Run', bbox=box)
    ax5.set_xlabel('Numero della Run', bbox=box)

    ax6 = axes[2, 1]
    ax6.plot(df.loc["rapl"], color="blue")
    ax6.plot(dfA.loc["rapl"], color="orange")
    ax6.set_title('rapl', fontweight='bold')
    ax6.set_ylabel('Energia Media per Run', bbox=box)
    ax6.set_xlabel('Numero della Run', bbox=box)

    ax7 = axes[3, 0]
    ax7.plot(df.loc["smartwatts-formula"], color="blue")
    ax7.plot(dfA.loc["smartwatts-formula"], color="orange")
    ax7.set_title('smartwatts-formula', fontweight='bold')
    ax7.set_ylabel('Energia Media per Run', bbox=box)
    ax7.set_xlabel('Numero della Run', bbox=box)

    fig.subplots_adjust(hspace=0.7)
    plt.suptitle('Evoluzione del consumo di energia per ogni sensore Pre e Post Refactoring', fontsize=16)

    plt.show()
def piechartEnergyTot(df):
    df['avg'] = df.sum(axis=1, numeric_only=True)
    df["avg"]=df["avg"]/51
    new_order = ["crate-container-cd", "hwpc-sensor-container", "global", "influx_dest", "rapl", "mongo_source",
                 "smartwatts-formula"]

    df_new = df["avg"].reindex(new_order)
    labels = ["crate-container-cd \nE=2492.102652J", "hwpc-sensor-container \nE=3.485958J", "global \nE=2546.906348J", "influx_dest \nE=2.118533J", "rapl \nE=3463.177497J", "                                    mongo_source \n                                     E=18.695846J",
                 "smartwatts-formula \nE=30.503360J"]
    sizes = df_new
    print(sizes)

    '''
    explode = (0, 0, 0, 0, 0, 0, 0.7)
    fig, ax = plt.subplots()
    wedges, texts, pcts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        colors = ['orange', 'blue', 'brown', 'grey', 'purple', 'red', 'green'],
        pctdistance=0.5,
        explode=explode,
        radius=0.5
    )
    for i, patch in enumerate(wedges):
        texts[i].set_color(patch.get_facecolor())
    plt.setp(pcts, color='black', fontweight='bold')
    plt.setp(texts, fontweight=600)
    plt.legend(title = "Sensori:")
    plt.tight_layout()
    # Add a title
    plt.title('Consumo energetico medio per sensore')
    # Show the chart
    plt.show()
    '''

    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    wedges, texts = ax.pie(sizes, wedgeprops=dict(width=0.5), startangle=-40)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(labels[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)

    ax.set_title("Consumo energetico medio per sensore")

    plt.show()

def histogramChart(df,dfAR):
    df1 = (df.T)
    df1 = df1.sum()/51
    df2 = (dfAR.T)
    df2 = df2.sum() / 51
    sensors = df1.index.to_list()
    print(df1)
    print(df2)
    doubleDataset = {
        'Pre refactoring': df1,
        'Post Refactoring': df2,
    }
    x = np.arange(len(sensors))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in doubleDataset.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Energia in Joule/ora')

    ax.set_title('Istogramma dei Valori di Energia per Sensori Pre e Post Refactoring')
    ax.set_xticks(x + width, sensors)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, max(df2) + 1)
    #plt.ylim(0, max(df1) + 1)
    plt.show()
def drawSingleHist(ax, lista,listaAR):
    doubleDataset = {
        'Pre refactoring': lista,
        'Post Refactoring': listaAR,
    }
    x = np.arange(len(list(range(51))))  # the label locations
    width = 0.4  # the width of the bars
    multiplier = 0

    for attribute, measurement in doubleDataset.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        #ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Energia in Joule/ora')

    ax.set_title('Istogramma dei Valori di Energia per Run Pre e Post Refactoring')
    #ax.set_xticks(x + width, list(range(51)))
    ax.legend(loc='upper left', ncols=3)
def histogramChartEverySensor(df,dfA):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    box = dict(facecolor='yellow', pad=5, alpha=0.2)

    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]
    fig, axes1 = plt.subplots(nrows=2, ncols=2)
    box = dict(facecolor='yellow', pad=5, alpha=0.2)

    ax5 = axes1[0, 0]
    ax6 = axes1[0, 1]
    ax7 = axes1[1, 0]

    drawSingleHist(ax1, df.loc["crate-container-cd"],dfA.loc["crate-container-cd"])
    drawSingleHist(ax2, df.loc["global"],dfA.loc["global"])
    drawSingleHist(ax3, df.loc["hwpc-sensor-container"],dfA.loc["hwpc-sensor-container"])
    drawSingleHist(ax4, df.loc["influx_dest"],dfA.loc["influx_dest"])
    drawSingleHist(ax5, df.loc["mongo_source"],dfA.loc["mongo_source"])
    drawSingleHist(ax6, df.loc["rapl"],dfA.loc["rapl"])
    drawSingleHist(ax7, df.loc["smartwatts-formula"],dfA.loc["smartwatts-formula"])

    fig.subplots_adjust(hspace=0.7)
    plt.suptitle('Evoluzione del consumo di energia per ogni sensore', fontsize=16)
    plt.show()

def custom_agg(group):
    total_power = group['power'].sum()
    time_difference = group['time'].iloc[-1] - group['time'].iloc[0]
    return pd.Series({'power': total_power, 'time': time_difference})
def calcolaEnergiaeTempoMedio(df):
    result = df.groupby('directory').apply(custom_agg).reset_index()
    result["time"]=result['time'].dt.total_seconds()/ 60
    return result

def plotTimePerRun(df,dfAF):
    # I RISULTATI DEL GRAFICO SONO IN LINEA CON CIO CHE CI ASPETTIAMO, ALL'AUMENTARE DELL'ENERGIA AUMENTA IL TEMPO PER L'ESECUZIONE
    fig, axes = plt.subplots(nrows=1, ncols=2)
    box = dict(facecolor='yellow', pad=5, alpha=0.2)
    ax1 = axes[0]
    ax1.scatter(df['time'], df["power"], label='Energia Media per Run', color='blue')
    ax1.set_title('Relazione tra Tempo e Energia Media per Run')
    ax1.set_xlabel('Tempo in minuti', bbox=box)
    ax1.set_ylabel('Energia Media', bbox=box)

    ax2 = axes[1]
    ax2.scatter(dfAF['time'], dfAF["power"], label='Energia Media per Run', color='red')
    ax2.set_title('Relazione tra Tempo e Energia Media per Run')
    ax2.set_xlabel('Tempo in minuti', bbox=box)
    ax2.set_ylabel('Energia Media', bbox=box)

    plt.suptitle('Confronto relazione tempo - energia media per run pre e post refactoring', fontsize=16)
    plt.show()

    '''plt.figure(figsize=(10, 6))
    plt.scatter(df['time'], df["power"], label='Energia Media per Run', color='blue')
    plt.title('Relazione tra Tempo e Energia Media per Run')
    plt.xlabel('Tempo')
    plt.ylabel('Energia Media')
    plt.legend()
    plt.show()'''
def boxPlot(df):
    plt.figure(figsize=(8, 6))
    sns.boxplot(df['power'])
    plt.title('Boxplot of Power')
    plt.ylabel('Power')
    plt.show()
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='time', data=df)
    plt.title('Boxplot of time')
    plt.xlabel('Time')
    plt.show()




print(os.getcwd())
dfMain = createUniqueDataset("./rawData/crate-master")
os.chdir("..")
print(os.getcwd())
dfAfterRefactor = createUniqueDataset("./rawData/crate-cd-out")

#CALCOLA ENERGIA PER OGNI SENSORE (BEFORE E AFTER REFACTOR)
listaDatasets, listaSensori = calcolaEnergiaPerOgniSensore(dfMain)
listaDatasetsAR, listaSensoriAR = calcolaEnergiaPerOgniSensore(dfAfterRefactor)

#CALCOLA ENERGIA PER RUN PER OGNI SENSORE (BEFORE E AFTER REFACTOR)
dfMediaEnergiaPerRunPerSensore = calcolaEnergiaPerOgniRun(listaDatasets,listaSensori)
dfMediaEnergiaPerRunPerSensoreAF = calcolaEnergiaPerOgniRun(listaDatasetsAR,listaSensoriAR)

#CALCOLA ENERGIA MEDIA E TEMPO MEDIO PER OGNI RUN (BEFORE E AFTER REFACTOR)
dfEnergiaMediaTempoPerRun = calcolaEnergiaeTempoMedio(dfMain)
dfEnergiaMediaTempoPerRunAF = calcolaEnergiaeTempoMedio(dfAfterRefactor)

#PLOT CHE METTE IN RELAZIONE TEMPO MEDIO E ENERGIA MEDIA PER OGNI RUN
#plotTimePerRun(dfEnergiaMediaTempoPerRun,dfEnergiaMediaTempoPerRunAF)

#2 BOX PLOT ENERGIA MEDIA E TEMPO MEDIO
#boxPlot(dfEnergiaMediaTempoPerRun)

#EVOLUZIONE ENERGIA MEDIA PER OGNI RUN (BEFORE E AFTER REFACTOR) GRAFICO A DISPERSIONE
#EnergyEvolutionForEveryRun(dfMediaEnergiaPerRunPerSensore, dfMediaEnergiaPerRunPerSensoreAF)

#EVOLUZIONE ENERGIA MEDIA PER OGNI RUN, 1 GRAFO PER OGNI SENSORE (BEFORE E AFTER REFACTOR)
#plotForEverySensor(dfMediaEnergiaPerRunPerSensore,dfMediaEnergiaPerRunPerSensoreAF)

#DIAGRAMMA A TORTA CHE MOSTRA L'ENERGIA MEDIA SUDDIVISA PER OGNI SENSORE
#piechartEnergyTot(dfMediaEnergiaPerRunPerSensore)
#piechartEnergyTot(dfMediaEnergiaPerRunPerSensoreAF)

#ISTOGRAMMA CHE MOSTRA L'ENERGIA MEDIA PER RUN DI OGNI SENSORE (BEFORE E AFTER REFACTOR)
#histogramChart(dfMediaEnergiaPerRunPerSensore,dfMediaEnergiaPerRunPerSensoreAF)

#ISTOGRAMMA CHE MOSTRA L'ENERGIA MEDIA PER RUN, 1 GRAFO PER SENSORE (BEFORE E AFTER REFACTOR)
histogramChartEverySensor(dfMediaEnergiaPerRunPerSensore,dfMediaEnergiaPerRunPerSensoreAF)







