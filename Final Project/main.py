#Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
#import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from PIL import Image

#Reading in File to create Dataframe
file_path = 'C:/Users/rayde/PycharmProjects/Dog_Breed_App/new_breed_df.csv'
breed_df = pd.read_csv(file_path)

#Dropping Column/s to create a new Dataframe
breed_df = breed_df.drop(columns=breed_df.columns[0:1])
print(breed_df.head())

#Creating the target and encode variables for algorithim
target = 'Breed'
encode = ['average height', 'average weight', 'average lifespan']

target_map = {
    'German Longhaired Pointer': 0,
    'German Pinscher': 1,
    'German Shepherd Dog': 2,
    'German Shorthaired Pointer': 3,
    'German Wirehaired Pointer': 4
}

#Function to assign a value to the breed of dog for algorittim
def target_encode(val):
    return target_map[val]

breed_df['Breed'] = breed_df['Breed'].apply(target_encode)

print(breed_df.head())

#Splitting the dataframe to fix the RandomForestClassifier
X = breed_df.drop('Breed', axis=1)
Y = breed_df['Breed']

clf = RandomForestClassifier()
clf.fit(X, Y)

#Predicting X values
y_pred = clf.predict(X)

#from sklearn import metrics
#print()

#Accuracy Model
pred = "Accuracy of the Model:", metrics.accuracy_score(Y, y_pred)
#________________________________________________________________________________________________________

#Streamlit App
st.write(
    """
    # Dog Breed App
    
    This app will predict the breed of a dog!
    """
)

#Code for Sidebar of the Streamlit App
st.sidebar.header("Please Input Some Data")

#If else statement for user to upload a .CSV file or use the default values provided
upload_file = st.sidebar.file_uploader("Upload you input CSV file", type=["csv"])
if upload_file is not None:
    input_df = pd.read_csv(upload_file)
else:
    #Function for default breed values
    def user_action():
        average_height = st.sidebar.slider('average height', 18.5, 23.5)
        average_weight = st.sidebar.slider('average weight', 35.1, 67.5)
        average_lifespan = st.sidebar.slider('average lifespan', 11, 13)
        data = {
            'average height': average_height,
            'average weight': average_weight,
            'average lifespan': average_lifespan
        }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_action()

#Creating a new datafile inorder to concatenate the input_df and new dogs Dataframe
file_path = 'C:/Users/rayde/PycharmProjects/Dog_Breed_App/new_breed_df.csv'
raw_file = pd.read_csv(file_path)
dogs = raw_file.drop(columns=['Breed'])
df = pd.concat([input_df, dogs], axis=0)

#View the values in the dataframe table in streamlit app
df = df[:1]

st.subheader("User Input features")

#if/else statement to download a .CSV file if values are wanted for a report
if upload_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded.  Currently using example input parameters')
    st.write(df)

#NP Array that will match the values in the RFC to match the prediction.
dog_breed = np.array([
    'German Longhaired Pointer',
    'German Pinscher',
    'German Shepherd Dog',
    'German Shorthaired Pointer',
    'German Wirehaired Pointer'
])

#Dropping random null column
df = df.drop(columns = df.columns[3])

#Geteting the prediction results
prediction = clf.predict(df)
prediction_proba = pred

#Writing the prediction results to Streamlit app
st.write('Breed Prediction')
st.write(dog_breed[prediction])

#Retrieving images using the PIL Library
img0 = Image.open('C:/Users/rayde/PycharmProjects/Dog_Breed_App/DogImages/German-Longhaired-Pointer.jpeg')
img1 = Image.open('C:/Users/rayde/PycharmProjects/Dog_Breed_App/DogImages/German-Pinscher.jpeg')
img2 = Image.open('C:/Users/rayde/PycharmProjects/Dog_Breed_App/DogImages/German-Shepherd.jpeg')
img3 = Image.open('C:/Users/rayde/PycharmProjects/Dog_Breed_App/DogImages/German-Shorthaired-Pointer.jpeg')
img4 = Image.open('C:/Users/rayde/PycharmProjects/Dog_Breed_App/DogImages/German-Wirehaired-Pointer.jpeg')

#If/Else statement to match image with prediction results and display the correct image.
if prediction == 0:
    st.image(img0, use_column_width = "auto")
elif prediction == 1:
    st.image(img1, use_column_width = "auto")
elif prediction == 2:
    st.image(img2, use_column_width = "auto")
elif prediction == 3:
    st.image(img3, use_column_width = "auto")
elif prediction == 4:
    st.image(img4, use_column_width = "auto")
else:
    st.write('No Image Viewable')

#Display Accuracy Score
st.subheader('Prediction Probability')
st.write(prediction_proba)


