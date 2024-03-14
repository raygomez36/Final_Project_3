import streamlit as st
import pandas as pd
import numpy as np
#import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from PIL import Image

file_path = 'C:/Users/rayde/PycharmProjects/Dog_Breed_App/new_breed_df.csv'
breed_df = pd.read_csv(file_path)

breed_df = breed_df.drop(columns=breed_df.columns[0:1])
print(breed_df.head())

target = 'Breed'
encode = ['average height', 'average weight', 'average lifespan']

target_map = {
    'German Longhaired Pointer': 0,
    'German Pinscher': 1,
    'German Shepherd Dog': 2,
    'German Shorthaired Pointer': 3,
    'German Wirehaired Pointer': 4
}

def target_encode(val):
    return target_map[val]

breed_df['Breed'] = breed_df['Breed'].apply(target_encode)

print(breed_df.head())

X = breed_df.drop('Breed', axis=1)
Y = breed_df['Breed']

clf = RandomForestClassifier()
clf.fit(X, Y)

y_pred = clf.predict(X)

#from sklearn import metrics
#print()

pred = "Accuracy of the Model:", metrics.accuracy_score(Y, y_pred)
#________________________________________________________________________________________________________

st.write(
    """
    # Dog Breed App
    
    This app will predict the breed of a dog!
    """
)

st.sidebar.header("Please Input Some Data")

upload_file = st.sidebar.file_uploader("Upload you input CSV file", type=["csv"])
if upload_file is not None:
    input_df = pd.read_csv(upload_file)
else:
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

file_path = 'C:/Users/rayde/PycharmProjects/Dog_Breed_App/new_breed_df.csv'
raw_file = pd.read_csv(file_path)
dogs = raw_file.drop(columns=['Breed'])
df = pd.concat([input_df, dogs], axis=0)


df = df[:1]

st.subheader("User Input features")

if upload_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded.  Currently using example input parameters')
    st.write(df)

dog_breed = np.array([
    'German Longhaired Pointer',
    'German Pinscher',
    'German Shepherd Dog',
    'German Shorthaired Pointer',
    'German Wirehaired Pointer'
])

df = df.drop(columns = df.columns[3])

prediction = clf.predict(df)
prediction_proba = pred

st.write('Breed Prediction')
st.write(dog_breed[prediction])

img0 = Image.open('C:/Users/rayde/PycharmProjects/Dog_Breed_App/DogImages/German-Longhaired-Pointer.jpeg')
img1 = Image.open('C:/Users/rayde/PycharmProjects/Dog_Breed_App/DogImages/German-Pinscher.jpeg')
img2 = Image.open('C:/Users/rayde/PycharmProjects/Dog_Breed_App/DogImages/German-Shepherd.jpeg')
img3 = Image.open('C:/Users/rayde/PycharmProjects/Dog_Breed_App/DogImages/German-Shorthaired-Pointer.jpeg')
img4 = Image.open('C:/Users/rayde/PycharmProjects/Dog_Breed_App/DogImages/German-Wirehaired-Pointer.jpeg')

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


st.subheader('Prediction Probability')
st.write(prediction_proba)


