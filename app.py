from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import pickle
import pandas as pd
from pydantic import BaseModel

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API input schema
class UserInput(BaseModel):
    Age: int
    Gender: str
    Platform: str
    Daily_Usage: float
    Posts_Per_Day: float
    Likes_Per_Day: float
    Comments_Per_Day: float
    Messages_Per_Day: float

def preprocess_input(Age, Gender, Platform, Daily_Usage, Posts, Likes, Comments, Messages):
    feature_names = [
        "Age",
        "Daily_Usage_Time (minutes)",
        "Posts_Per_Day",
        "Likes_Received_Per_Day",
        "Comments_Received_Per_Day",
        "Messages_Sent_Per_Day",
        "Gender_Female",
        "Gender_Male",
        "Gender_Non-binary",
        "Platform_Facebook",
        "Platform_Instagram",
        "Platform_LinkedIn",
        "Platform_Snapchat",
        "Platform_Telegram",
        "Platform_Twitter",
        "Platform_Whatsapp"
    ]
    data = dict.fromkeys(feature_names, 0)
    data["Age"] = Age
    data["Daily_Usage_Time (minutes)"] = Daily_Usage
    data["Posts_Per_Day"] = Posts
    data["Likes_Received_Per_Day"] = Likes
    data["Comments_Received_Per_Day"] = Comments
    data["Messages_Sent_Per_Day"] = Messages
    # One-hot encode Gender
    if Gender.lower() == "female":
        data["Gender_Female"] = 1
    elif Gender.lower() == "male":
        data["Gender_Male"] = 1
    elif Gender.lower() == "non-binary":
        data["Gender_Non-binary"] = 1
    # One-hot encode Platform
    platform_key = f"Platform_{Platform.capitalize()}"
    if platform_key in data:
        data[platform_key] = 1
    # Construct DataFrame in the correct order
    df = pd.DataFrame([[data[feature] for feature in feature_names]], columns=feature_names)
    # Scale numeric columns
    numeric_cols = [
        "Age",
        "Daily_Usage_Time (minutes)",
        "Posts_Per_Day",
        "Likes_Received_Per_Day",
        "Comments_Received_Per_Day",
        "Messages_Sent_Per_Day"
    ]
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df

def predict_gradio(Age, Gender, Platform, Daily_Usage, Posts, Likes, Comments, Messages):
    df = preprocess_input(Age, Gender, Platform, Daily_Usage, Posts, Likes, Comments, Messages)
    pred = model.predict(df)[0]
    emotion = label_encoder.inverse_transform([pred])[0]
    return emotion

@app.post("/predict")
def predict_api(data: UserInput):
    df = preprocess_input(
        data.Age, data.Gender, data.Platform, data.Daily_Usage,
        data.Posts_Per_Day, data.Likes_Per_Day, data.Comments_Per_Day, data.Messages_Per_Day
    )
    pred = model.predict(df)[0]
    emotion = label_encoder.inverse_transform([pred])[0]
    return {"Predicted Emotion": emotion}

# Mount Gradio UI at /gradio
gr_interface = gr.Interface(
    fn=predict_gradio,
    inputs=[
        gr.Number(label="Age (years)", info="Enter your age (e.g., 25)", precision=0),
        gr.Dropdown(
            choices=["Female", "Male", "Non-binary"],
            label="Gender",
            value="Female"
        ),
        gr.Dropdown(
            choices=[
                "Facebook", "Instagram", "LinkedIn", "Snapchat",
                "Telegram", "Twitter", "Whatsapp"
            ],
            label="Social Media Platform",
            value="Instagram"
        ),
        gr.Number(label="Daily Usage Time (minutes)", info="Average daily app usage time", precision=0),
        gr.Number(label="Posts Per Day", info="Average number of posts per day", precision=0),
        gr.Number(label="Likes Received Per Day", info="Average likes received daily", precision=0),
        gr.Number(label="Comments Received Per Day", info="Average comments received daily", precision=0),
        gr.Number(label="Messages Sent Per Day", info="Average messages sent daily", precision=0)
    ],
    outputs=gr.Textbox(
        label="Predicted Dominant Emotion",
        show_copy_button=True
    ),
    title="Social Media Emotion Classifier",
    description=(
        "<div style='font-size: 16px; line-height: 1.6;'>"
        "Predict your <b>dominant emotion</b> expressed on social media based on your usage patterns.<br>"
        "Fill in your activity details below and click <b>Submit</b> to see your result."
        "</div>"
    ),
    theme=gr.themes.Base(),
    allow_flagging="never",
    css="""
        .gradio-container { max-width: 600px !important; margin: auto; }
        .gr-input label { font-weight: 600; }
        .gr-input input, .gr-input select { border-radius: 6px; }
        .gr-button { background: #2563eb; color: white; font-weight: 600; }
        .gr-output label { font-weight: 600; }
    """
)


app = gr.mount_gradio_app(app, gr_interface, path="/gradio")