from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QMessageBox
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neural_network import MLPRegressor


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
        team = self.comboBox_4.currentText()
        defender = ["Player", "Pos", "Squad", "Age", "Rankings", "Goals", "SoT%", "PasTotCmp%", "Assists", "ShoDist",
            "PasShoCmp%", "PasMedCmp%", "PasLonCmp%", "TI", "SCA", "GCA", "Blocks", "Int", "Clr", "Crs",
            "PasAss", "CrsPA", "PasCrs", "Tkl", "TklWon", "TklDri%", "Err", "Recov", "PKcon", "AerWon%"]
        midfielder = ["Player", "Pos", "Squad", "Age", "Rankings", "Goals", "G/Sh", "SoT%", "PasTotCmp%", "Assists", "ShoDist",
            "PasShoCmp%", "PasMedCmp%", "PasLonCmp%", "SCA", "GCA", "Blocks", "Int",
            "PasAss", "Pas3rd", "PPA", "Tkl", "TklWon", "TklDri%", "Recov", "PKcon", "CPA",
            "PKwon","Touches", "TouAtt3rd", "TouAttPen", "AerWon%"]
        forward = ["Player", "Pos", "Squad", "Age", "Rankings", "Goals", "G/Sh", "SoT%", "PasTotCmp%", "Assists", "ShoDist",
            "PasShoCmp%", "PasMedCmp%", "PasLonCmp%", "SCA", "GCA",  "Pas3rd", "PPA", "Crs", "PasAss",
            "CrsPA", "PasCrs", "PKwon", "CPA", "Touches", "TouAtt3rd", "TouAttPen", "AerWon%"]
        df = pd.read_csv("Preprocessed 2021-2022 Football Player Stats.csv", encoding='latin1')
        df = df[(df['Squad'] != team)]

        if(selected_position == "CB" or selected_position == "LB/RB"):
            filtered_df = df[(df['Age'] <= max_age) & df['Pos'].str.contains("DF")]
            x = filtered_df[defender]
            pos = "DF"
        elif(selected_position == "ST" or selected_position == "LW/RW"):
            filtered_df = df[(df['Age'] <= max_age) & df['Pos'].str.contains("FW")]
            x = filtered_df[forward]
            pos = "FW"
        else:
            filtered_df = df[(df['Age'] <= max_age) & (df['Pos'].str.contains("MF"))]
            x = filtered_df[midfielder]
            pos = "MF"

        if selected_position in ["CDM", "CM", "CAM/CF"]:
            kmeans = KMeans(n_clusters=6, random_state=35)
        elif selected_position in ["CB", "LB/RB"]:
            kmeans = KMeans(n_clusters=4, random_state=15)
        else:
            kmeans = KMeans(n_clusters=4, random_state=1)

        player_info = x[['Player', 'Pos', 'Squad', "Age", "Rankings"]]
        x = x.drop(['Player', 'Pos', 'Squad', "Age", "Rankings"], axis=1)
        kmeans.fit(x)
        labels = kmeans.labels_
        x['Cluster'] = labels
        cluster_stats = x.groupby('Cluster').mean()
        pd.set_option('display.max_columns', None)
        print(cluster_stats)
        silhouette = silhouette_score(x, kmeans.labels_)
        print("Silhouette Score:", silhouette)

        combined_data = pd.concat([player_info, x], axis=1)

        if selected_playstyle == "Ball Playing":
            cluster = 0
            criteria = ["PasTotCmp%", "PasShoCmp%", "PasMedCmp%", "PasLonCmp%",
                "Blocks", "Int", "Clr", "TklWon", "AerWon%"]
        elif selected_playstyle == "Central Defender":
            cluster = 2
            criteria = ["Blocks", "Int", "Clr", "TklWon", "AerWon%",
                "PasTotCmp%", "PasShoCmp%", "PasMedCmp%", "PasLonCmp%"]
        elif selected_playstyle == "Fullback":
            cluster = 1
            criteria = ["Crs", "PasAss", "CrsPA", "PasCrs", "SCA", "GCA",
                "Blocks", "Int", "Clr", "TklWon", "AerWon%", "TI"]
        elif selected_playstyle == "Wingback":
            cluster = 3
            criteria = ["Goals", "Assists", "SCA", "GCA", "Crs", "PasAss", "CrsPA", "PasCrs",
                "Blocks", "Int", "Clr", "TklWon", "TI"]
        elif selected_playstyle == "Deep Lying Playmaker":
            cluster = 0
            criteria = ["PasTotCmp%", "PasShoCmp%", "PasMedCmp%", "PasLonCmp%",
                "Blocks", "Int", "TklWon", "AerWon%"]
        elif selected_playstyle == "Anchor":
            cluster = 5
            criteria = ["Blocks", "Int", "TklWon", "AerWon%",
                "PasTotCmp%", "PasShoCmp%", "PasMedCmp%", "PasLonCmp%"]
        elif selected_playstyle == "Box to Box":
            cluster = 4
            criteria = ["Goals", "PasTotCmp%", "Assists", "PasShoCmp%", "PasMedCmp%", "PasLonCmp%",
                "SCA", "GCA", "Blocks", "Int", "PasAss", "PPA", "Tkl", "TklWon", "TklDri%", "Recov", "CPA"]
        elif selected_playstyle == "Playmaker":
            cluster = 1
            criteria = ["Goals", "PasTotCmp%", "Assists", "SCA", "GCA", "PasAss", "Pas3rd",
                "PasShoCmp%", "PasMedCmp%", "PasLonCmp%"]
        elif selected_playstyle == "Advanced Playmaker":
            cluster = 2
            criteria = ["Assists", "SCA", "GCA", "PasAss", "Pas3rd", "Goals",
                "PasShoCmp%", "PasMedCmp%", "PasLonCmp%", "PPA", "CPA"]
        elif selected_playstyle == "Shadow Striker":
            cluster = 3
            criteria = ["Goals", "SCA", "GCA", "Assists", "PasAss", "Pas3rd",
                "PasShoCmp%", "PasMedCmp%", "PasLonCmp%", "PPA", "CPA"]
        elif selected_playstyle == "Inside Forward":
            cluster = 0
            criteria = ["PPA", "CPA", "SCA", "GCA", "Goals", "Assists", "PasAss", "Pas3rd"]
        elif selected_playstyle == "Wide Forward":
            cluster = 1
            criteria = ["CrsPA", "PasCrs", "PPA", "Goals", "Assists", "SCA", "GCA", "Crs", "PasAss", "TouAtt3rd"]
        elif selected_playstyle == "Target":
            cluster = 3
            criteria = ["Goals", "Assists", "PasAss", "AerWon%", "SCA", "GCA", "TouAtt3rd", "TouAttPen"]
        else:
            cluster = 2
            criteria = ["Goals", "G/Sh", "SoT%", "TouAtt3rd", "TouAttPen", "SCA", "GCA", "Assists", "PasAss"]

        cluster_df = combined_data[combined_data['Cluster'] == cluster]
        sorted_cluster_df = cluster_df.sort_values(by=criteria, ascending=False)
        team_rank = uefa_rankings.get(team)
        sorted_cluster_df['NewRanking'] = team_rank
        sorted_cluster_df.rename(columns={'Rankings': 'OldRanking'}, inplace=True)
        top_players = sorted_cluster_df.head(5)
        top_players_info = ""
        counter = 1
        for index, player in top_players.iterrows():
            top_players_info += f"{counter}. {player['Player']} {player['Squad']} {player['Age']}\n"
            counter += 1

        msgBox = QMessageBox()
        msgBox.setWindowTitle("Top Players")
        msgBox.setText("Top Players (Name, Team, Age):\n" + top_players_info)
        msgBox.exec()

        col = ['Player', 'Pos', 'Squad', "Age", "Rankings"] + criteria
        #mlp(top_players, team, pos, col)

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

def mlp(testedPlayers, team, pos, col):
    df1 = pd.read_csv("Preprocessed 2021-2022 Football Player Stats.csv", encoding='latin1')
    df2 = pd.read_csv("Preprocessed 2022-2023 Football Player Stats.csv", encoding='latin1')

    common_players = set(df1['Player']).intersection(set(df2['Player']))
    common_players_df1 = df1[(df1['Player'].isin(common_players)) & (df1['Pos'].str.contains(pos))][col].drop_duplicates(subset=['Player'])
    common_players_df2 = df2[(df2['Player'].isin(common_players)) & (df2['Pos'].str.contains(pos))][col].drop_duplicates(subset=['Player'])
    common_players_df1 = common_players_df1.sort_values(by='Player')
    common_players_df2 = common_players_df2.sort_values(by='Player')
    common_players_df1 = common_players_df1.reset_index(drop=True)
    common_players_df2 = common_players_df2.reset_index(drop=True)
    common_players_df1.rename(columns={'Rankings': 'OldRanking'}, inplace=True)
    common_players_df2.rename(columns={'Rankings': 'NewRanking'}, inplace=True)
    oldTestedRanks = common_players_df1['OldRanking']
    newTestedRanks = common_players_df2['NewRanking']
    common_players_df1['NewRanking'] = newTestedRanks
    common_players_df2['OldRanking'] = oldTestedRanks

    col.remove('Pos')
    col.remove('Player')
    col.remove('Squad')
    col.remove('Age')
    col.remove('Rankings')
    col.append('OldRanking')
    col.append('NewRanking')


    x_train = common_players_df1[col]
    y_train = common_players_df2[col]

    x_test = testedPlayers[col]

    model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)
    model.fit(x_train, y_train)
    print("Model Fitted")

    y_pred = model.predict(x_test)
    print("Y Predicted")

    for i, player in enumerate(testedPlayers):
        print("Player:", player)
        print("Predicted New Statistics:", y_pred[i])

    team_stats = df1[(df1['Squad'] == team) & (df1['Pos'] == pos)]
    print("Current Statistics for Players in {} playing {}:".format(team, pos))
    print(team_stats)


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