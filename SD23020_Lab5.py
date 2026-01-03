import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# -----------------------------
# Step 1: Page configuration
# -----------------------------
st.set_page_config(
    page_title="Image Classification using ResNet18",
    layout="centered"
)

st.title("Computer Vision Image Classification")
st.write("Upload an image to classify it using a pre-trained ResNet18 model.")

# -----------------------------
# Step 2 & 3: Load model (CPU only)
# -----------------------------
device = torch.device("cpu")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()
model.to(device)

# Load ImageNet labels
labels = models.ResNet18_Weights.DEFAULT.meta["categories"]

# -----------------------------
# Step 5: Image preprocessing
# -----------------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Step 6: File uploader
# -----------------------------
uploaded_file = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")


    # -----------------------------
    # Step 7: Model inference
    # -----------------------------
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    # -----------------------------
    # Step 8: Softmax & top-5
    # -----------------------------
    probabilities = F.softmax(output, dim=1)[0]
    top5_prob, top5_idx = torch.topk(probabilities, 5)

    results = []
    for i in range(5):
        results.append({
            "Class": labels[top5_idx[i]],
            "Probability": float(top5_prob[i])
        })

    df = pd.DataFrame(results)

    st.subheader("Top-5 Predictions")
    st.dataframe(df)

    # -----------------------------
    # Step 9: Bar chart
    # -----------------------------
    st.subheader("Prediction Probabilities")
    st.bar_chart(df.set_index("Class"))
