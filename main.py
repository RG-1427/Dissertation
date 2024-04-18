#Importing Packages for GUI
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QMessageBox

#Importing Packages for kmeans clustering and MLP
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor

#Main GUI window
class Ui_MainWindow(object):

    #Setup
    def setupUi(self, MainWindow):

        #Main Window
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(353, 453)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        #Submit Button
        self.submitButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.submitButton.setGeometry(QtCore.QRect(10, 370, 93, 28))
        self.submitButton.setObjectName("submitButton")
        self.submitButton.clicked.connect(self.kMeansClustering)

        #Labels for the GUI
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

        #Dropdown menus for the GUI
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

        #Age horizontal slider
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

        #More dropdown menus
        self.comboBox_4 = QtWidgets.QComboBox(parent=self.centralwidget)
        self.comboBox_4.setGeometry(QtCore.QRect(10, 120, 191, 22))
        self.comboBox_4.setEditable(False)
        self.comboBox_4.setObjectName("comboBox_4")

        #Another label for setup
        self.label_4 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(70, 10, 211, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_4.setFont(font)
        self.label_4.setScaledContents(False)
        self.label_4.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_4.setObjectName("label_4")
        MainWindow.setCentralWidget(self.centralwidget)

        #Menubar
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 353, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        #Setup GUI variables
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        #Setup horizontal slider, position and playstyle dropdown menus, and teams dropdown menu
        self.horizontalSlider.valueChanged.connect(self.updateMaxLabel)
        self.comboBox_2.currentIndexChanged.connect(self.checkCombination)
        self.checkCombination(0)
        initial_slider_value = self.horizontalSlider.value()
        self.updateMaxLabel(initial_slider_value)
        self.comboBox_4.addItems(teams_list)

    #Updating maximum age value displayed
    def updateMaxLabel(self, value):

        #When user adjusts the horizontal slider, edit the text to show the maximum age value
        self.label_5.setText(f"Maximum Age: {value}")

    #Setting up text in the GUI
    def retranslateUi(self, MainWindow):

        #Setting up text in the labels
        _translate = QtCore.QCoreApplication.translate

        #Application window and submit button text
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.submitButton.setText(_translate("MainWindow", "Submit"))

        #Setting up labels
        self.label.setText(_translate("MainWindow", "Team"))
        self.label_2.setText(_translate("MainWindow", "Position"))
        self.label_3.setText(_translate("MainWindow", "Playing Style"))\

        #Position and play style labels
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

        #Title setup
        self.label_4.setText(_translate("MainWindow", "Player Parameters"))

    #Creating combinations for positions and play styles
    def checkCombination(self, index: int):

        #Setting up the allowed combinations list
        allowed_combinations = {
            "CB": ["Central Defender", "Ball Playing"],
            "LB/RB": ["Fullback", "Wingback"],
            "CDM": ["Deep Lying Playmaker", "Anchor"],
            "CM": ["Box to Box", "Playmaker"],
            "CAM/CF": ["Advanced Playmaker", "Shadow Striker"],
            "LW/RW": ["Inside Forward", "Wide Forward"],
            "ST": ["Target", "Poacher"]
        }

        #Based on the positions dropdown, only allow to choose from specific play styles
        selected_position = self.comboBox_2.currentText()
        current_index = self.comboBox_3.currentIndex()
        self.comboBox_3.clear()
        self.comboBox_3.addItems(allowed_combinations[selected_position])
        self.comboBox_3.setCurrentIndex(current_index)

    #Kmeans clustering model
    def kMeansClustering (self):

        #Gathering user parameters that were selected
        max_age = self.horizontalSlider.value()
        selected_position = self.comboBox_2.currentText()
        selected_playstyle = self.comboBox_3.currentText()
        team = self.comboBox_4.currentText()

        #Setting up columns that will be used for each position cluster
        defender = ["Player", "Pos", "Squad", "Age", "Rankings", "Goals", "SoT%", "PasTotCmp%", "Assists", "ShoDist",
            "PasShoCmp%", "PasMedCmp%", "PasLonCmp%", "TI", "SCA", "GCA", "Blocks", "Int", "Clr", "Crs",
            "PasAss", "CrsPA", "PasCrs", "Tkl", "TklWon", "TklDri%", "Err", "Recov", "PKcon", "AerWon%"]
        midfielder = ["Player", "Pos", "Squad", "Age", "Rankings", "Goals", "G/Sh", "SoT%", "PasTotCmp%", "Assists",
            "ShoDist", "PasShoCmp%", "PasMedCmp%", "PasLonCmp%", "SCA", "GCA", "Blocks", "Int",
            "PasAss", "Pas3rd", "PPA", "Tkl", "TklWon", "TklDri%", "Recov", "PKcon", "CPA",
            "PKwon","Touches", "TouAtt3rd", "TouAttPen", "AerWon%"]
        forward = ["Player", "Pos", "Squad", "Age", "Rankings", "Goals", "G/Sh", "SoT%", "PasTotCmp%", "Assists",
            "ShoDist", "PasShoCmp%", "PasMedCmp%", "PasLonCmp%", "SCA", "GCA",  "Pas3rd", "PPA", "Crs", "PasAss",
            "CrsPA", "PasCrs", "PKwon", "CPA", "Touches", "TouAtt3rd", "TouAttPen", "AerWon%"]

        #Reading the preprocessed data, removing players in the same squad as the team of the user
        df = pd.read_csv("Preprocessed 2021-2022 Football Player Stats.csv", encoding='latin1')
        df = df[(df['Squad'] != team)]

        #Filtering the dataset based on the maximum age and position
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

        #Creating the clustering model for the positions
        if selected_position in ["CDM", "CM", "CAM/CF"]:
            kmeans = KMeans(n_clusters=6, random_state=35)
        elif selected_position in ["CB", "LB/RB"]:
            kmeans = KMeans(n_clusters=4, random_state=15)
        else:
            kmeans = KMeans(n_clusters=4, random_state=1)

        #Removing player personal information from the model
        player_info = x[['Player', 'Pos', 'Squad', "Age", "Rankings"]]
        x = x.drop(['Player', 'Pos', 'Squad', "Age", "Rankings"], axis=1)

        #Kmeans clustering, displaying the clusters average stats to determine which is what play style
        kmeans.fit(x)
        labels = kmeans.labels_
        x['Cluster'] = labels
        cluster_stats = x.groupby('Cluster').mean()
        pd.set_option('display.max_columns', None)
        print(cluster_stats)

        #Testing the accuracy of the kmeans clustering model
        silhouette = silhouette_score(x, kmeans.labels_)
        print("Silhouette Score:", silhouette)

        #Adding back the players personal info to the data
        combined_data = pd.concat([player_info, x], axis=1)

        #Based on selected play style, pick the correct cluster and the criteria for the players
        #who best suit the play style
        if selected_playstyle == "Ball Playing":
            cluster = 3
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
            cluster = 0
            criteria = ["Goals", "Assists", "SCA", "GCA", "Crs", "PasAss", "CrsPA", "PasCrs",
                "Blocks", "Int", "Clr", "TklWon", "TI"]
        elif selected_playstyle == "Deep Lying Playmaker":
            cluster = 0
            criteria = ["PasTotCmp%", "PasShoCmp%", "PasMedCmp%", "PasLonCmp%",
                "Blocks", "Int", "TklWon", "AerWon%"]
        elif selected_playstyle == "Anchor":
            cluster = 4
            criteria = ["TklWon", "Int", "Blocks", "AerWon%",
                "PasTotCmp%", "PasShoCmp%", "PasMedCmp%", "PasLonCmp%"]
        elif selected_playstyle == "Box to Box":
            cluster = 5
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

        #Sorting the players by criteria
        cluster_df = combined_data[combined_data['Cluster'] == cluster]
        sorted_cluster_df = cluster_df.sort_values(by=criteria, ascending=False)

        #Adding old and new team ranking for the MLP model
        team_rank = uefa_rankings.get(team)
        sorted_cluster_df['NewRanking'] = team_rank
        sorted_cluster_df.rename(columns={'Rankings': 'OldRanking'}, inplace=True)

        #Generating the top players by text
        top_players = sorted_cluster_df.head(5)
        top_players_info = ""
        counter = 1
        for index, player in top_players.iterrows():
            top_players_info += f"{counter}. {player['Player']} {player['Squad']} {player['Age']}\n"
            counter += 1

        #Displaying the 5 top players on a message box
        msgBox = QMessageBox()
        msgBox.setWindowTitle("Top Players")
        msgBox.setText("Top Players (Name, Team, Age):\n" + top_players_info)
        msgBox.exec()

        #Selecting the columns for the MLP model, and calling git
        col = ['Player', 'Pos', 'Squad', "Age", "Rankings"] + criteria
        mlp(top_players, team, pos, col)

#MLP Model
def mlp(tested_players, team, pos, col):

    #Reading the preprocessed data
    df1 = pd.read_csv("Preprocessed 2021-2022 Football Player Stats.csv", encoding='latin1')
    df2 = pd.read_csv("Preprocessed 2022-2023 Football Player Stats.csv", encoding='latin1')

    #Creating training data using players that are in both datasets and dropping duplicate players
    common_players = set(df1['Player']).intersection(set(df2['Player']))
    common_players_df1 = df1[(df1['Player'].isin(common_players))][col].drop_duplicates(subset=['Player'])
    common_players_df2 = df2[(df2['Player'].isin(common_players))][col].drop_duplicates(subset=['Player'])

    #Sorting the values by player so they are in the same order, and re-setting the index
    common_players_df1 = common_players_df1.sort_values(by='Player')
    common_players_df2 = common_players_df2.sort_values(by='Player')
    common_players_df1 = common_players_df1.reset_index(drop=True)
    common_players_df2 = common_players_df2.reset_index(drop=True)

    #Adding the old and new ranks to players so the model takes that into account when predicting statistics
    common_players_df1.rename(columns={'Rankings': 'OldRanking'}, inplace=True)
    common_players_df2.rename(columns={'Rankings': 'NewRanking'}, inplace=True)
    old_tested_ranks = common_players_df1['OldRanking']
    new_tested_ranks = common_players_df2['NewRanking']
    common_players_df1['NewRanking'] = new_tested_ranks
    common_players_df2['OldRanking'] = old_tested_ranks

    #Removing columns not needed for MLP model
    col.remove('Player')
    col.remove('Pos')
    col.remove('Squad')
    col.remove('Age')
    col.remove('Rankings')
    col.append('OldRanking')
    col.append('NewRanking')

    #Creating the training and testing data
    x_train = common_players_df1[col]
    y_train = common_players_df2[col]
    x_test = tested_players[col]

    #Creating and running the model
    model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)
    model.fit(x_train, y_train)

    #Predicting the stats of the 5 players
    y_pred = model.predict(x_test)

    #Adding back the columns needed for the player information display
    col.insert(0, 'Player')
    col.insert(1, 'Pos')
    col.insert(2, 'Age')
    col.remove('OldRanking')
    col.remove('NewRanking')

    #Reading original data to remove data standardization
    df = pd.read_csv("2021-2022 Football Player Stats.csv", encoding='latin1')

    #Creating the text of the 5 top players and their statistics
    text = "\nStatistics Predicted for top 5 Players\n"
    counter = 0
    for i in range(0, 5):
        for column in col:
            if column in ['Player', 'Pos', 'Age']:
                text += f"{column}: {tested_players[column].iloc[i]}, "
            else:
                value = y_pred[i][counter]
                value = value * df[column].std() + df[column].mean()
                text += f"{column}: {value}, "
                counter += 1
        counter = 0
        text += "\n"

    #Adding the text of the current players in that team for comparison
    team_stats = df1[(df1['Squad'] == team) & (df1['Pos'] == pos)]
    text += "\nCurrent Statistics for Players in {} playing {}:\n".format(team, pos)
    for i in range(len(team_stats)):
        for column in col:
            if column in ['Player', 'Pos', 'Age']:
                text += f"{column}: {team_stats[column].iloc[i]}, "
            elif '%' in column:
                value = team_stats[column].iloc[i]
                value = value * 100
                text += f"{column}: {value}, "
            else:
                value = team_stats[column].iloc[i]
                value = value * df[column].std() + df[column].mean()
                text += f"{column}: {value}, "
        text += "\n"

    #Displaying the results in the console
    print(text)

    #Displaying accuracy of the model based on training data
    mae = mean_absolute_error(x_train, y_train)
    print(f"Mean Absolute Error (MAE): {mae}")

#Handling missing data
def handling_missing_data(df):

    #Finding out which columns and rows have missing data and printing them out
    missing_data = df.isnull().sum()
    print("Columns with missing data in the first dataset:")
    print(missing_data[missing_data > 0])
    print("\nRows with missing data:")
    rows_with_missing_data = df[df.isnull().any(axis=1)]
    print(rows_with_missing_data)

#Data preprocessing
def preprocessing(df, name):

    #Handling missing data
    handling_missing_data(df)

    #List of columns required for the kmeans analysis
    columns = ["Player", "Pos", "Squad", "Age", "Min", "Goals", "SoT%", "G/Sh", "ShoDist",
    "PasTotCmp%", "PasShoCmp%", "PasMedCmp%", "PasLonCmp%", "Assists", "PasAss", "Pas3rd", "PPA", "CrsPA",
    "PasCrs", "TI", "SCA", "GCA", "Tkl", "TklWon", "TklDri%", "Blocks", "Int", "Clr",
    "Err", "Touches", "TouAtt3rd", "TouAttPen", "CPA", "Crs", "PKwon",
    "PKcon", "Recov", "AerWon%"]

    #columns that should not be converted to per 90 minutes as they are measurements
    non_converted = ['Player', 'Pos', 'Squad', 'Age', 'Min', "G/Sh", "ShoDist"]

    #Removing unnecessary columns and
    df = df[columns]
    df = df[(df['Pos'] != 'GK') & (df['Min'] >= 450) & df['Squad'].isin(teams_list)]

    #For each column, standardize it
    for col in columns:
        #If it needs to be converted to per 90, convert it and then standardize it
        if col not in non_converted and '%' not in col:
            df[col] = df[col] / df['Min'] * 90
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
        #Set percentages to be on a scale from 0-1
        elif '%' in col:
            df[col] = df[col] / 100
        #Standartzide non converted columns that will be used in analysis
        elif col == "G/Sh" or col == "ShoDist":
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std

    #Add rankings column and save the new files
    df['Rankings'] = df['Squad'].map(dict(zip(teams_list, rankings)))
    df = df[columns + ['Rankings']]
    df.reset_index(drop=True, inplace=True)
    name = "Preprocessed " + name
    df.to_csv(name, index=False, encoding='latin1')

#Main class
if __name__ == "__main__":

    #Pandas and system packages needed
    import sys
    import pandas as pd

    #Reading the datasets
    df1 = pd.read_csv('2021-2022 Football Player Stats.csv', encoding='latin1')
    df2 = pd.read_csv('2022-2023 Football Player Stats.csv', encoding='latin1')

    #List of teams and their EUFA ranking
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

    #Creating a list of rankings and teams
    teams_list = list(uefa_rankings.keys())
    rankings = list(uefa_rankings.values())

    #Preprocess datasets
    preprocessing(df1, "2021-2022 Football Player Stats.csv")
    preprocessing(df2, "2022-2023 Football Player Stats.csv")

    #Call the GUI
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())