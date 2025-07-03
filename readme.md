# Emotion Classifier - Social Media Usage

This project is a web-based application that predicts the **dominant emotion** expressed on social media based on user activity patterns. It uses a machine learning model trained on social media usage data and is served via a FastAPI backend with a Gradio-powered user interface.

---

## Features

- **Predicts dominant emotion** (e.g., Anxiety, Happiness, etc.) from social media usage.
- User-friendly web interface (Gradio).
- REST API endpoint for programmatic access.
- Handles categorical and numerical features with proper preprocessing and scaling.

---

## Requirements

- Python 3.8+
- pip

### Python Packages

- fastapi
- gradio
- pandas
- scikit-learn
- pydantic
- uvicorn
- pickle (standard library)

Install all requirements:
```sh
pip install fastapi gradio pandas scikit-learn pydantic uvicorn
```

---

## Usage

### 1. Place Model Files

Ensure the following files are in the project directory:
- `model.pkl` (trained ML model)
- `scaler.pkl` (StandardScaler used during training)
- `label_encoder.pkl` (LabelEncoder for emotion labels)

### 2. Run the Server

```sh
uvicorn app:app --reload
```

- The Gradio UI will be available at: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- The API endpoint is at: [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)

---

## API Usage

Send a POST request to `/predict` with JSON body:
```json
{
  "Age": 25,
  "Gender": "Female",
  "Platform": "Instagram",
  "Daily_Usage": 120,
  "Posts_Per_Day": 2,
  "Likes_Per_Day": 50,
  "Comments_Per_Day": 10,
  "Messages_Per_Day": 20
}
```

**Response:**
```json
{
  "Predicted Emotion": "Happiness"
}
```

---

## Gradio Web UI

- Fill in your age, gender, platform, and daily activity stats.
- Click **Submit** to see your predicted dominant emotion.

---

## Project Structure

```
emotion-classifier/
│
├── app.py                # Main FastAPI + Gradio application
├── model.pkl             # Trained ML model
├── scaler.pkl            # StandardScaler used for numeric features
├── label_encoder.pkl     # LabelEncoder for emotion labels
├── README.md             # This file
```

---

## Notes

- The model expects the following features (in order):  
  `Age`, `Daily_Usage_Time (minutes)`, `Posts_Per_Day`, `Likes_Received_Per_Day`, `Comments_Received_Per_Day`, `Messages_Sent_Per_Day`, `Gender_Female`, `Gender_Male`, `Gender_Non-binary`, `Platform_Facebook`, `Platform_Instagram`, `Platform_LinkedIn`, `Platform_Snapchat`, `Platform_Telegram`, `Platform_Twitter`, `Platform_Whatsapp`
- All preprocessing (scaling, one-hot encoding) is handled automatically.

---

## License

This project is for academic and research purposes.

---
