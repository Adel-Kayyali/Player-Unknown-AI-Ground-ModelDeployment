import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the model, scaler and other necessary files
model = joblib.load(open('PUBG_model.sav','rb'))
scaler = joblib.load(open('scaler.sav','rb'))
columns = joblib.load(open('columns.sav','rb'))

# Define the function to make predictions
def predict(df):
    # Scale the input data using the same scaler used during training
    scaled_df = scaler.transform(df)
    # Make the prediction
    prediction = model.predict(scaled_df)
    return prediction

def convert_placement(prediction):
    """
    Convert the predicted placement from the model into a more understandable format.
    """

    if prediction > 0.4:
        return '1st-5th' 
    elif prediction > 0.3:
        return '6th-10th'
    elif prediction > 0.2:
        return '11th-20th' 
    elif prediction > 0.1:
        return '21st-30th' 
    else:
        return '31th-100th'


# Define the Streamlit app
def main():
    
    page_bg_img="""
    <style>
    [data-testid="stAppViewContainer"]{
    background-image: url("https://www.pixel4k.com/wp-content/uploads/2020/01/pubg-helmet-2019-w7-3840x2160-1-2048x1152.jpg.webp");
    background-size:cover;
    }

    [data-testid="stToolbar"]{
    right: 2rem;
    }

    [data-testid="stSidebar"]{
    background-image: url("https://i.pinimg.com/564x/62/d3/f3/62d3f30fd9273728c51afbeb25515957.jpg");
    background-size: cover;
    }
    
    [data-testid="manage-app-button"] {
    display: none;
    }

    </style>
    """

    
    st.set_page_config(page_title="PUBG PLACEMENT PREDICTOR", page_icon=":video_game:", layout="wide")
    st.title('PUBG PLACEMENT PREDICTOR')
    st.sidebar.title('Enter the following details to predict your Placement:')
    st.markdown(page_bg_img, unsafe_allow_html=True)

    
    #image_url = "https://wallpaperaccess.com/full/190978.jpg"
    #st.image(image_url, caption="", use_column_width=True)

    #options for the SelectBox
    options = ['1-35 Groups', '36-66 Groups', '67-100 Groups']
    options_match = ['Solo', 'Duo', 'Squad']

    # Collect input from the user
    input_data = {
        'selected_match_type': st.sidebar.selectbox('Choose the Match Type', options_match),
        'selected_option': st.sidebar.selectbox('Choose the Number of Groups in your game', options),
        'DBNOs': st.sidebar.slider('Knock-Downs', 0, 100, 5),
        'headshotKills': st.sidebar.slider('Headshot Kills', 0, 100, 5),
        'killStreaks': st.sidebar.slider('Kill Streaks', 0, 100, 5),
        'longestKill': st.sidebar.slider('Longest Distance Kill', 0, 5000, 50),
        'totalDistance': st.sidebar.slider('Total Distance Moved', 0, 30000, 200),
        'totalItemsPicked': st.sidebar.slider('Total Items Picked', 0, 200, 5),
        'totalDamageByTeam': st.sidebar.slider('Total Damage By Your Team', 0, 10000, 100),
        'totalTeamBuffs': st.sidebar.slider('Total Heals & Boosts Your Team Used', 0, 100, 5),
        'teamwork': st.sidebar.slider('Total Assists & Revives You did for the Team', 0, 100, 2)
        }
    
    # Set the values for matchType_solo and matchType_squad based on user input
    if input_data['selected_match_type'] == 'Solo':
        input_data['matchType_solo'] = 1
        input_data['matchType_squad'] = 0
    elif input_data['selected_match_type'] == 'Duo':
        input_data['matchType_solo'] = 0
        input_data['matchType_squad'] = 0
    else:
        input_data['matchType_solo'] = 0
        input_data['matchType_squad'] = 1

    # Set the values for numGroups_bins_67-100 and numGroups_bins_36-66 based on user input
    if input_data['selected_option'] == '1-35 Groups':
        input_data['numGroups_bins_67-100'] = 0
        input_data['numGroups_bins_36-66'] = 0
    elif input_data['selected_option'] == '36-66 Groups':
        input_data['numGroups_bins_67-100'] = 0
        input_data['numGroups_bins_36-66'] = 1
    else:
        input_data['numGroups_bins_67-100'] = 1
        input_data['numGroups_bins_36-66'] = 0
    
    # Convert the input into a dataframe
    input_df = pd.DataFrame([input_data], columns=columns)

    # Scale the input data using the same scaler used during training
    scaled_input_df = scaler.transform(input_df)

    # Make the prediction and display the result
    result = predict(scaled_input_df)
    predicted_placement = convert_placement(result[0])
    #st.write('<h1>Your predicted Placement is:</h1>', unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; font-size: 100px;'>{predicted_placement}</h1>", unsafe_allow_html=True)


    # Add footer with link to your repo
    footer = st.container()
    st.markdown('---')

    

    col1, col2,col3 = st.columns(3)

    github_link = 'https://github.com/Adel-Kayyali/PUBG-Predictor'
    github_image = 'https://cdn.iconscout.com/icon/free/png-512/github-1521500-1288242.png?f=avif&w=256'

    linkedin_link = 'https://www.linkedin.com/in/adel-kayyali-96b884240/'
    linkedin_image = 'https://cdn.iconscout.com/icon/free/png-512/linkedin-1464529-1239440.png?f=avif&w=256'

    with col1:
        st.markdown(f"<div style='text-align: center;'><a href='{github_link}' target='_blank'><img src='{github_image}' width='100'></a>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div style='text-align: center;'><a href='{linkedin_link}' target='_blank'><img src='{linkedin_image}' width='100'></a>", unsafe_allow_html=True)


    with col3:
        st.markdown(f"<div style='text-align: center;'><a href='mailto:adelkayyali@outlook.com' target='_blank'><img src='https://cdn.iconscout.com/icon/free/png-512/mail-808-475025.png?f=avif&w=256' width='100'></a>", unsafe_allow_html=True)

    
    
# Run the app
if __name__ == '__main__':
    main()
