from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import pickle
import pandas as pd
from pydantic import BaseModel
from fastapi.responses import RedirectResponse

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
        gr.Number(label="Age (years)", info="Enter your age (e.g., 25)", precision=0, elem_id="age-input"),
        gr.Dropdown(
            choices=["Female", "Male", "Non-binary"],
            label="Gender",
            value="Female",
            elem_id="gender-input"
        ),
        gr.Dropdown(
            choices=[
                "Facebook", "Instagram", "LinkedIn", "Snapchat",
                "Telegram", "Twitter", "Whatsapp"
            ],
            label="Social Media Platform",
            value="Instagram",
            elem_id="platform-input"
        ),
        gr.Number(label="Daily Usage Time (minutes)", info="Average daily app usage time", precision=0, elem_id="usage-input"),
        gr.Number(label="Posts Per Day", info="Average number of posts per day", precision=0, elem_id="posts-input"),
        gr.Number(label="Likes Received Per Day", info="Average likes received daily", precision=0, elem_id="likes-input"),
        gr.Number(label="Comments Received Per Day", info="Average comments received daily", precision=0, elem_id="comments-input"),
        gr.Number(label="Messages Sent Per Day", info="Average messages sent daily", precision=0, elem_id="messages-input")
    ],
    outputs=gr.Textbox(
        label="Predicted Dominant Emotion",
        show_copy_button=True,
        elem_id="emotion-output"
    ),
    title = "🎭 <span class='gr-title'>Social Media Emotion Classifier</span>",
    description = (
        "<div class='gr-description' style='font-size: 18px; line-height: 1.7; margin-bottom: 18px;'>"
        "<b>Discover your dominant emotion on social media!</b><br>"
        "Fill in your activity details below and click <b>Submit</b> to see your result."
        "</div>"
    ),
    theme=gr.themes.Soft(),
    allow_flagging="never",
    css="""
        .gradio-container {
            max-width: 540px !important;
            margin: 40px auto 0 auto;
            background: #f9fafb;
            border-radius: 18px;
            box-shadow: 0 6px 32px 0 rgba(30,41,59,0.10);
            padding: 32px 32px 24px 32px;
        }
        .gr-title {
            font-size: 1.5rem !important;
            font-weight: 800 !important;
            color: #2563eb !important;
            letter-spacing: -1px;
            margin-bottom: 10px;
        }
        .gr-description {
            font-size: 1 rem !important;
            color: #374151 !important;
            margin-bottom: 18px;
        }
        .gr-input label, .gr-output label {
            font-weight: 700 !important;
            color: #1e293b !important;
            font-size: 1.05rem !important;
        }
        .gr-input input, .gr-input select {
            border-radius: 8px !important;
            border: 1.5px solid #cbd5e1 !important;
            background: #fff !important;
            font-size: 1rem !important;
            padding: 10px 12px !important;
            margin-bottom: 8px !important;
            color: #0f172a !important;
            opacity: 1 !important;
        }
        .gr-input input::placeholder {
            color: #64748b !important;
            opacity: 1 !important;
        }
        .gr-button {
            background: linear-gradient(90deg, #2563eb 0%, #60a5fa 100%) !important;
            color: #fff !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
            border-radius: 8px !important;
            padding: 12px 0 !important;
            margin-top: 18px !important;
            box-shadow: 0 2px 8px 0 rgba(37,99,235,0.08);
            transition: background 0.2s;
        }
        .gr-button:hover {
            background: linear-gradient(90deg, #1d4ed8 0%, #38bdf8 100%) !important;
        }
        #emotion-output textarea {
            background: #f1f5f9 !important;
            color: #0f172a !important;
            font-size: 1.2rem !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
            border: 1.5px solid #cbd5e1 !important;
            padding: 14px !important;
            min-height: 60px !important;
            opacity: 1 !important;
        }
        #emotion-output textarea::placeholder {
            color: #64748b !important;
            opacity: 1 !important;
        }
        @media (max-width: 700px) {
            .gradio-container {
                padding: 18px 6vw 18px 6vw !important;
            }
        }

        /* Dark theme overrides */
        @media (prefers-color-scheme: dark) {

            .gr-title {
                color: #60a5fa !important;
            }
            .gr-description {
                color: #d1d5db !important;
            }

            .gradio-container {
                background: #18181b !important;
                box-shadow: 0 6px 32px 0 rgba(0,0,0,0.25);
            }
            .gr-title {
                color: #60a5fa !important;
            }
            .gr-description {
                color: #d1d5db !important;
            }
            .gr-input label, .gr-output label {
                color: #f1f5f9 !important;
            }
            .gr-input input, .gr-input select {
                background: #27272a !important;
                border: 1.5px solid #334155 !important;
                color: #f1f5f9 !important;
                opacity: 1 !important;
            }
            .gr-input input::placeholder {
                color: #94a3b8 !important;
                opacity: 1 !important;
            }
            .gr-button {
                background: linear-gradient(90deg, #2563eb 0%, #60a5fa 100%) !important;
                color: #fff !important;
            }
            .gr-button:hover {
                background: linear-gradient(90deg, #38bdf8 0%, #2563eb 100%) !important;
            }
            #emotion-output textarea {
                background: #27272a !important;
                color: #f1f5f9 !important;
                border: 1.5px solid #334155 !important;
                opacity: 1 !important;
            }
            #emotion-output textarea::placeholder {
                color: #94a3b8 !important;
                opacity: 1 !important;
            }
        }
    """
)

@app.get("/")
def redirect_to_gradio():
    return RedirectResponse(url="/gradio")

# Mount Gradio at /gradio
app = gr.mount_gradio_app(app, gr_interface, path="/gradio")
