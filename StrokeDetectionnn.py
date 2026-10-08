from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ================= GUI =================
main = tkinter.Tk()
main.title("Stroke Risk Identification Using Supervised Machine Learning Models")
main.geometry("1000x650")
main.config(bg='light coral')

# ================= TITLE (FIXED & VISIBLE) =================
title = Label(
    main,
    text="Stroke Risk Identification Using Supervised Machine Learning Models",
    font=('Times New Roman', 18, 'bold'),
    bg='light coral',
    fg='dark blue'
)
title.place(x=80, y=20)

# ================= GLOBALS =================
global filename, dataset, rf
global le1, le2, le3, le4, le5
global X, Y, X_train, X_test, y_train, y_test

accuracy, precision, recall, fscore = [], [], [], []

# ================= FUNCTIONS =================
def loadDataset():
    global filename, dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    text.insert(END, str(filename) + " loaded\n\n")
    text.insert(END, str(dataset.head()))

def preprocessDataset():
    global X, Y, X_train, X_test, y_train, y_test
    global dataset, le1, le2, le3, le4, le5

    text.delete('1.0', END)

    le1, le2, le3, le4, le5 = LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder()

    dataset.fillna(0, inplace=True)
    dataset['gender'] = le1.fit_transform(dataset['gender'].astype(str))
    dataset['ever_married'] = le2.fit_transform(dataset['ever_married'].astype(str))
    dataset['work_type'] = le3.fit_transform(dataset['work_type'].astype(str))
    dataset['Residence_type'] = le4.fit_transform(dataset['Residence_type'].astype(str))
    dataset['smoking_status'] = le5.fit_transform(dataset['smoking_status'].astype(str))

    label = dataset.groupby('stroke').size()
    data = dataset.values

    X = data[:, 1:-1]
    Y = data[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    text.insert(END, "Dataset Preprocessed Successfully\n\n")
    text.insert(END, f"Total Records : {X.shape[0]}\n")
    text.insert(END, f"Training Records : {X_train.shape[0]}\n")
    text.insert(END, f"Testing Records : {X_test.shape[0]}\n\n")

    label.plot(kind="bar")
    plt.title("Normal vs Stroke Cases")
    plt.show()

def calculateMetrics(predict, testY, algorithm):
    a = accuracy_score(testY, predict) * 100
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    text.insert(END, f"{algorithm} Accuracy  : {a:.2f}\n")
    text.insert(END, f"{algorithm} Precision : {p:.2f}\n")
    text.insert(END, f"{algorithm} Recall    : {r:.2f}\n")
    text.insert(END, f"{algorithm} FScore    : {f:.2f}\n\n")

    cm = confusion_matrix(testY, predict)
    sns.heatmap(cm, annot=True, fmt='g', cmap='viridis',
                xticklabels=['Normal', 'Stroke'],
                yticklabels=['Normal', 'Stroke'])
    plt.title(algorithm + " Confusion Matrix")
    plt.show()

def trainNaiveBayes():
    cls = GaussianNB()
    cls.fit(X_train, y_train)
    calculateMetrics(cls.predict(X_test), y_test, "Naive Bayes")

def trainDT():
    cls = DecisionTreeClassifier()
    cls.fit(X_train, y_train)
    calculateMetrics(cls.predict(X_test), y_test, "J48")

def trainKNN():
    cls = KNeighborsClassifier(n_neighbors=2)
    cls.fit(X_train, y_train)
    calculateMetrics(cls.predict(X_test), y_test, "KNN")

def trainRanfomForest():
    global rf
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    calculateMetrics(rf.predict(X_test), y_test, "Random Forest")

def trainANN():
    Y_cat = to_categorical(Y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, Y_cat, test_size=0.2)

    model = Sequential([
        Dense(256, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_tr, y_tr, epochs=50, verbose=0)

    preds = np.argmax(model.predict(X_te), axis=1)
    calculateMetrics(preds, np.argmax(y_te, axis=1), "ANN")

def graph():
    if len(accuracy) < 5:
        text.insert(END, "Please train all algorithms before generating graph\n")
        return

    df = pd.DataFrame({
        "Algorithm": ["Naive Bayes", "J48", "KNN", "Random Forest", "ANN"],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": fscore
    })

    df.set_index("Algorithm").plot(kind='bar', figsize=(10,6))
    plt.title("Comparison of Machine Learning Algorithms")
    plt.ylabel("Performance (%)")
    plt.xlabel("Algorithms")
    plt.grid(axis='y')
    plt.show()

def predict():
    text.delete('1.0', END)

    testfile = filedialog.askopenfilename(initialdir="Dataset")
    testdata = pd.read_csv(testfile)
    testdata.fillna(0, inplace=True)

    testdata['gender'] = le1.transform(testdata['gender'].astype(str))
    testdata['ever_married'] = le2.transform(testdata['ever_married'].astype(str))
    testdata['work_type'] = le3.transform(testdata['work_type'].astype(str))
    testdata['Residence_type'] = le4.transform(testdata['Residence_type'].astype(str))
    testdata['smoking_status'] = le5.transform(testdata['smoking_status'].astype(str))

    testX = testdata.values[:, 1:]
    preds = rf.predict(testX)

    # ---- TITLE INSIDE OUTPUT ----
    text.insert(END, "PREDICTION RESULTS\n")
    text.insert(END, "-" * 60 + "\n\n")

    for i in range(len(preds)):
        if preds[i] == 1:
            text.insert(
                END,
                f"Test Data = {testX[i]}  PREDICTED AS ====> STROKE\n\n"
            )
        else:
            text.insert(
                END,
                f"Test Data = {testX[i]}  PREDICTED AS ====> NO STROKE\n\n"
            )


# ================= UI BUTTONS =================
font = ('times', 12, 'bold')

Button(main, text="Upload Dataset", command=loadDataset, font=font).place(x=10,y=100)
Button(main, text="Preprocess Dataset Algorithm", command=preprocessDataset, font=font).place(x=300,y=100)
Button(main, text="Train Naive Bayes Algorithm", command=trainNaiveBayes, font=font).place(x=730,y=100)
Button(main, text="Train J48 Algorithm", command=trainDT, font=font).place(x=10,y=150)
Button(main, text="Train KNN Algorithm", command=trainKNN, font=font).place(x=300,y=150)
Button(main, text="Train Random Forest Algorithm", command=trainRanfomForest, font=font).place(x=730,y=150)
Button(main, text="Train ANN Algorithm", command=trainANN, font=font).place(x=10,y=200)
Button(main, text="Comparison Graph", command=graph, font=font).place(x=300,y=200)
Button(main, text="Predict Test data", command=predict, font=font).place(x=730,y=200)

text = Text(main, height=20, width=160, font=font)
text.place(x=10,y=250)

main.mainloop()
