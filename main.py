from PyQt6 import QtCore, QtGui, QtWidgets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(353, 453)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.submitButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.submitButton.setGeometry(QtCore.QRect(10, 370, 93, 28))
        self.submitButton.setObjectName("submitButton")
        self.submitButton.clicked.connect(self.kMeansClustering)
        self.label = QtWidgets.QLabel(parent=self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 100, 55, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 170, 55, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 240, 75, 16))
        self.label_3.setObjectName("label_3")
        self.label_5 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(10, 310, 100, 16))
        self.label_5.setObjectName("label_5")
        self.comboBox_2 = QtWidgets.QComboBox(parent=self.centralwidget)
        self.comboBox_2.setGeometry(QtCore.QRect(10, 190, 191, 22))
        self.comboBox_2.setEditable(False)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_3 = QtWidgets.QComboBox(parent=self.centralwidget)
        self.comboBox_3.setGeometry(QtCore.QRect(10, 260, 191, 22))
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.horizontalSlider = QtWidgets.QSlider(parent=self.centralwidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(10, 330, 191, 22))
        self.horizontalSlider.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.horizontalSlider.setAutoFillBackground(False)
        self.horizontalSlider.setMinimum(16)
        self.horizontalSlider.setMaximum(56)
        self.horizontalSlider.setProperty("value", 56)
        self.horizontalSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.horizontalSlider.setInvertedAppearance(False)
        self.horizontalSlider.setInvertedControls(False)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.comboBox_4 = QtWidgets.QComboBox(parent=self.centralwidget)
        self.comboBox_4.setGeometry(QtCore.QRect(10, 120, 191, 22))
        self.comboBox_4.setEditable(False)
        self.comboBox_4.setObjectName("comboBox_4")
        self.label_4 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(70, 10, 211, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_4.setFont(font)
        self.label_4.setScaledContents(False)
        self.label_4.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_4.setObjectName("label_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 353, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.horizontalSlider.valueChanged.connect(self.updateMaxLabel)
        self.comboBox_2.currentIndexChanged.connect(self.checkCombination)
        self.checkCombination(0)
        initial_slider_value = self.horizontalSlider.value()
        self.updateMaxLabel(initial_slider_value)
        self.comboBox_4.addItems(teamList)

    def updateMaxLabel(self, value):
        self.label_5.setText(f"Maximum Age: {value}")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.submitButton.setText(_translate("MainWindow", "Submit"))
        self.label.setText(_translate("MainWindow", "Team"))
        self.label_2.setText(_translate("MainWindow", "Position"))
        self.label_3.setText(_translate("MainWindow", "Playing Style"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "CB"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "LB/RB"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "CDM"))
        self.comboBox_2.setItemText(3, _translate("MainWindow", "CM"))
        self.comboBox_2.setItemText(4, _translate("MainWindow", "CAM/CF"))
        self.comboBox_2.setItemText(5, _translate("MainWindow", "LW/RW"))
        self.comboBox_2.setItemText(6, _translate("MainWindow", "ST"))
        self.comboBox_3.setItemText(0, _translate("MainWindow", "Ball Playing"))
        self.comboBox_3.setItemText(1, _translate("MainWindow", "Central Defender"))
        self.comboBox_3.setItemText(2, _translate("MainWindow", "Fullback"))
        self.comboBox_3.setItemText(3, _translate("MainWindow", "Wingback"))
        self.comboBox_3.setItemText(4, _translate("MainWindow", "Deep Lying Playmaker"))
        self.comboBox_3.setItemText(5, _translate("MainWindow", "Anchor"))
        self.comboBox_3.setItemText(6, _translate("MainWindow", "Box to Box"))
        self.comboBox_3.setItemText(7, _translate("MainWindow", "Playmaker"))
        self.comboBox_3.setItemText(8, _translate("MainWindow", "Advanced Playmaker"))
        self.comboBox_3.setItemText(9, _translate("MainWindow", "Shadow Striker"))
        self.comboBox_3.setItemText(10, _translate("MainWindow", "Inside Forward"))
        self.comboBox_3.setItemText(11, _translate("MainWindow", "Wide Forward"))
        self.comboBox_3.setItemText(12, _translate("MainWindow", "Target"))
        self.comboBox_3.setItemText(13, _translate("MainWindow", "Poacher"))
        self.label_4.setText(_translate("MainWindow", "Player Parameters"))

    def checkCombination(self, index: int):
        allowed_combinations = {
            "CB": ["Central Defender", "Ball Playing"],
            "LB/RB": ["Fullback", "Wingback"],
            "CDM": ["Deep Lying Playmaker", "Anchor"],
            "CM": ["Box to Box", "Playmaker"],
            "CAM/CF": ["Advanced Playmaker", "Shadow Striker"],
            "LW/RW": ["Inside Forward", "Wide Forward"],
            "ST": ["Target", "Poacher"]
        }
        selected_position = self.comboBox_2.currentText()
        current_index = self.comboBox_3.currentIndex()
        self.comboBox_3.clear()
        self.comboBox_3.addItems(allowed_combinations[selected_position])
        self.comboBox_3.setCurrentIndex(current_index)

    def kMeansClustering (self):
        max_age = self.horizontalSlider.value()
        selected_position = self.comboBox_2.currentText()
        selected_playstyle = self.comboBox_3.currentText()
        defender = ["Player", "Pos", "Squad", "Goals", "SoT%", "PasTotCmp%", "Assists",
            "TI", "SCA", "GCA", "Blocks", "Int", "Clr", "Crs", "AerWon%"]
        midfielder = ["Player", "Pos", "Squad", "Goals", "SoT%", "ShoDist", "PasTotCmp%",
            "PasShoCmp%", "PasLonCmp%", "Assists", "PasAss", "SCA", "GCA", "TklWon", "CPA"]
        forward = ["Player", "Pos", "Squad", "Goals", "SoT%", "G/Sh", "ShoDist", "PasTotCmp%",
            "Assists", "PasAss", "SCA", "GCA", "TouAttPen", "CPA", "Crs"]
        df = pd.read_csv("Preprocessed 2021-2022 Football Player Stats.csv", encoding='latin1')

        if(selected_position == "CB" or selected_position == "LB/RB"):
            filtered_df = df[(df['Age'] <= max_age) & df['Pos'].str.contains("DF")]
            x = filtered_df[defender]
        elif(selected_position == "ST" or selected_position == "LW/RW"):
            filtered_df = df[(df['Age'] <= max_age) & df['Pos'].str.contains("FW")]
            x = filtered_df[forward]
        else:
            filtered_df = df[(df['Age'] <= max_age) & (df['Pos'].str.contains("MF"))]
            x = filtered_df[midfielder]

        if selected_position in ["CDM", "CM", "CAM/CF"]:
            kmeans = KMeans(n_clusters=6, random_state=42)
        else:
            kmeans = KMeans(n_clusters=4, random_state=42)

        player_info = x[['Player', 'Pos', 'Squad']]
        x = x.drop(['Player', 'Pos', 'Squad'], axis=1)
        pca = PCA(n_components=2)
        x = pca.fit_transform(x)
        kmeans.fit(x)
        df['Cluster'] = kmeans.labels_
        cluster_stats = df.groupby('Cluster').mean()
        print(cluster_stats)

        #DF -> cluster 1: cluster 2: cluster 3: cluster 4:
        #MF -> cluster 1: cluster 2: cluster 3: cluster 4: cluster 5: cluster 6:
        #FW -> cluster 1: cluster 2: cluster 3: cluster 4:

        #clustered_data = pd.concat([pd.DataFrame(kmeans.labels_, columns=['Cluster']), player_info], axis=1)
        #CLUSTER NUMBER BASED ON PLAY STYLES
        #desired_cluster = 0  # Change this to the cluster number you want to analyze

        # Filter the DataFrame for the desired cluster
        #cluster_df = clustered_data[clustered_data['Cluster'] == desired_cluster]

        # CRITERIA BASED ON PLAY STYLES -> EACH HAS ITS OWN CRITERIA
        #criteria = ['Goals', 'Assists']  # Add more criteria if needed

        # Sort the players within the cluster based on the criteria
        #sorted_cluster_df = cluster_df.sort_values(by=criteria, ascending=False)

        # Pick the top 5 players based on the criteria
        #top_players = sorted_cluster_df.head(5)

        #print(top_players)
        #msgBox = QtWidgets.QMessageBox
        #prsent on pyQT -> Open new small window with ui: Top 5 player 1. Name Team Age
        #SETUP TEXT
        #msgBox.setText("Submit successful!")
        #msgBox.setStandardButtons(QtWidgets.QMessageBox.OK)
        #msgBox.exec_()
        silhouette = silhouette_score(x, kmeans.labels_)
        print("Silhouette Score:", silhouette)

        team = self.comboBox_4.currentText()
        #RANKS LIST = TEAM RANK FROM DICTIONARY, LIST OF FIVE "RANKS" WITH THAT NUMBER REPEATED
        #mlp(PLAYERS INFO VARIABLE HERE, RANKS LIST HERE)
        #CALL SECOND MODEL AT THE END... SO CAN TEST OVERALL

    #RUN KNN MODEL, SEND TOP 5, TEST 25-50 TIMES PER POS
    #RUN SECOND MODEL TO PREDICT DATA (AFTER TESTING SECOND MODEL WITH 50 - 100 TRANSFERS)

def handling_missing_data(df):
    missing_data = df.isnull().sum()
    print("Columns with missing data in the first database:")
    print(missing_data[missing_data > 0])
    print("\nRows with missing data:")
    rows_with_missing_data = df[df.isnull().any(axis=1)]
    print(rows_with_missing_data)

def preprocessing(df, name):
    handling_missing_data(df)

    columns = ["Player", "Pos", "Squad", "Age", "Min", "Goals", "SoT%", "G/Sh", "ShoDist",
    "PasTotCmp%", "PasShoCmp%", "PasMedCmp%", "PasLonCmp%", "Assists", "PasAss", "Pas3rd", "PPA", "CrsPA",
    "PasCrs", "TI", "SCA", "GCA", "Tkl", "TklWon", "TklDri%", "Blocks", "Int", "Clr",
    "Err", "Touches", "TouAtt3rd", "TouAttPen", "CPA", "Crs", "PKwon",
    "PKcon", "Recov", "AerWon%"]
    non_converted = ['Player', 'Pos', 'Squad', 'Age', 'Min', "G/Sh", "ShoDist"]
    df = df[columns]
    df = df[(df['Pos'] != 'GK') & (df['Min'] >= 450) & df['Squad'].isin(teamList)]
    for col in columns:
        if col not in non_converted and '%' not in col:
            df[col] = df[col] / df['Min'] * 90
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
        elif '%' in col:
            df[col] = df[col] / 100
        elif col == "G/Sh" or col == "ShoDist":
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
    df['Rankings'] = df['Squad'].map(dict(zip(teamList, rankings)))
    df = df[columns + ['Rankings']]
    df.reset_index(drop=True, inplace=True)
    name = "Preprocessed " + name
    df.to_csv(name, index=False, encoding='latin1')

def mlp(testedPlayers, newRanks):
    df1 = pd.read_csv("Preprocessed 2021-2022 Football Player Stats.csv", encoding='latin1')
    df2 = pd.read_csv("Preprocessed 2022-2023 Football Player Stats.csv", encoding='latin1')

    common_players = set(df1['Player']).intersection(set(df2['Player']))
    common_players_df1 = df1[df1['Player'].isin(common_players)]
    common_players_df2 = df2[df2['Player'].isin(common_players)]

    x_train = common_players_df1.iloc[:, 5:].values
    x_train = np.concatenate((x_train, common_players_df1['Rankings'].values.reshape(-1, 1)),
                       axis=1)
    y_train = common_players_df2['Rankings'].tolist()

    x_test = testedPlayers.iloc[:, 5:].values
    x_test = np.concatenate((x_test, testedPlayers['Rankings'].values.reshape(-1, 1)), axis=1)

    y_test = newRanks

    model = Sequential([
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, epochs=50, batch_size=32)

    mse = model.evaluate(x_test, y_test)
    print("Mean Squared Error:", mse)

    predictions = model.predict(x_test)
    print("Predictions:", predictions)

if __name__ == "__main__":
    import sys
    import pandas as pd

    df1 = pd.read_csv('2021-2022 Football Player Stats.csv', encoding='latin1')
    df2 = pd.read_csv('2022-2023 Football Player Stats.csv', encoding='latin1')

    uefa_rankings = {
        'Arsenal': 17,
        'Atalanta': 24,
        'Athletic Club': 86,
        'Atlético Madrid': 9,
        'Barcelona': 6,
        'Bayern Munich': 1,
        'Betis': 78,
        'Bordeaux': 119,
        'Burnley': 76,
        'Chelsea': 5,
        'Dortmund': 19,
        'Eint Frankfurt': 26,
        'Espanyol': 85,
        'Everton': 77,
        'Freiburg': 103,
        'Getafe': 84,
        'Granada': 83,
        'Hertha BSC': 101,
        'Hoffenheim': 71,
        'Inter': 23,
        'Juventus': 8,
        'Köln': 101,
        'Lazio': 31,
        'Leicester City': 69,
        'Leverkusen': 30,
        'Lille': 56,
        'Liverpool': 2,
        'Lyon': 20,
        "M'Gladbach": 79,
        'Manchester City': 3,
        'Manchester Utd': 10,
        'Marseille': 38,
        'Milan': 45,
        'Monaco': 61,
        'Napoli': 25,
        'Nice': 115,
        'Paris S-G': 7,
        'RB Leipzig': 13,
        'Real Madrid': 4,
        'Real Sociedad': 62,
        'Reims': 116,
        'Rennes': 49,
        'Roma': 11,
        'Saint-Étienne': 118,
        'Sevilla': 12,
        'Strasbourg': 98,
        'Torino': 99,
        'Tottenham': 14,
        'Union Berlin': 100,
        'Valencia': 43,
        'Villarreal': 18,
        'West Ham': 74,
        'Wolfsburg': 73,
        'Wolves': 75
    }

    teamList = list(uefa_rankings.keys())
    rankings = list(uefa_rankings.values())

    preprocessing(df1, "2021-2022 Football Player Stats.csv")
    preprocessing(df2, "2022-2023 Football Player Stats.csv")

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())