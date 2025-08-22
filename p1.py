from tkinter import *
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tkinter import messagebox
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = Tk()
root.title("Customer Segmentation App by Purvi Kapadia")
root.geometry("650x600+50+50")
root.configure(background='lightblue')
f = ("Century", 20, "bold")
f2 = ("Century", 17, "bold")
f3 = ("Century", 17, "bold")

# Safely set icon
try:
    root.iconbitmap("customer.ico")
except Exception as e:
    print("Using default icon:", e)

lab_header = Label(root, text="Customer Type Segmentation", font=f, fg="darkblue")
lab_header.pack(pady=10)

# load your dataset
data = pd.read_csv("E:\kamal sir\machine learning sept 2023\internship\customer_type_project\Mall_Customers.csv")

# features
features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# create and fit the K-Means model
optimal_k = 5  # you can adjust this based on the Elbow Method graph
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
data['Cluster'] = kmeans.fit_predict(scaled_features)

lab_age = Label(root, text="Enter Age:", font=f2)
lab_age.pack(pady=10)
ent_age = Entry(root, font=f2)
ent_age.pack(pady=10)

lab_income = Label(root, text="Enter Annual Income (k$):", font=f2)
lab_income.pack(pady=10)
ent_income = Entry(root, font=f2)
ent_income.pack(pady=10)

lab_spending_score = Label(root, text="Enter Spending Score (1-100):", font=f2)
lab_spending_score.pack(pady=10)
ent_spending_score = Entry(root, font=f2)
ent_spending_score.pack(pady=10)

# set focus on the first input field
ent_age.focus_set()

def find():
    try:
        # retrieve input values
        age_str = ent_age.get()
        income_str = ent_income.get()
        spending_score_str = ent_spending_score.get()

        # validation: check for null or empty values
        if not all([age_str, income_str, spending_score_str]):
            raise ValueError("Invalid input values. Please enter numbers without spaces.")

        # validation: check if the input contains only valid numeric values
        if not all(re.match(r'^-?\d+\.?\d*$', value) for value in [age_str, income_str, spending_score_str]):
            raise ValueError("Invalid input values. Please enter valid numbers.")

        # convert input values to float
        age = float(age_str)
        income = float(income_str)
        spending_score = float(spending_score_str)

        # validation: check for negative numbers
        if any(val < 0 for val in [age, income, spending_score]):
            raise ValueError("Invalid input values. Please enter positive numbers only.")

        # validation: check if Spending Score is in the valid range
        if not (0 <= spending_score <= 100):
            raise ValueError("Invalid input values. Spending Score should be between 0 and 100.")

        # make a prediction using the trained K-Means model
        input_data = pd.DataFrame([[age, income, spending_score]],
                                  columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
        scaled_input_data = scaler.transform(input_data)
        cluster_prediction = kmeans.predict(scaled_input_data)[0]

        # display the prediction
        lab_ans.config(text=f"Predicted Customer Segment: {cluster_prediction}")

        # clear input fields
        ent_age.delete(0, 'end')
        ent_income.delete(0, 'end')
        ent_spending_score.delete(0, 'end')

        # set focus on the first input field
        ent_age.focus_set()

        # plot clusters
        plot_clusters()

    except ValueError as e:
        messagebox.showerror("Error", str(e))

        # clear input fields
        ent_age.delete(0, 'end')
        ent_income.delete(0, 'end')
        ent_spending_score.delete(0, 'end')

        # set focus on the first input field
        ent_age.focus_set()
        # clear output field
        lab_ans.config(text="")

def plot_clusters():
    plt.figure(figsize=(6, 6))
    plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='rainbow')
    plt.title('Customer Segmentation')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    
    # display the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

btn_predict = Button(root, text="Predict Customer Segment", font=f3, command=find)
lab_ans = Label(root, font=f3,fg="purple")
btn_predict.pack(pady=10)
lab_ans.pack(pady=10)

root.mainloop()



