import torch
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
import os

# Script'in bulunduğu dizin
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model sözlüğü 
MODELS = {
    "DenseNet_SE_LSTM (0.9785)": {
        "path": os.path.join(BASE_DIR, "models", "DenseNet_SE_LSTM_0.9785", "DenseNet_SE_LSTM_jit.pt"),
        "accuracy": 0.9785
    },
    "MARK0 (0.9486)": {
        "path": os.path.join(BASE_DIR, "models", "MARK0_0.9486", "MARK0_jit.pt"),
        "accuracy": 0.9486
    },
    "MARK1 (0.9573)": {
        "path": os.path.join(BASE_DIR, "models", "MARK1_0.9573", "MARK1_jit.pt"),
        "accuracy": 0.9573
    }
}

# Hastalık sınıfları
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Görüntü ön işleme
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Yüklü model cache
loaded_models = {}

def load_model(model_name):
    """Seçilen modeli yükler"""
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    model_path = MODELS[model_name]["path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    loaded_models[model_name] = model
    return model

def predict(image, model_name):
    """Görüntü için tahmin yapar"""
    if image is None:
        return "Lütfen bir görüntü yükleyin.", {}
    
    try:
        # Modeli yükle
        model = load_model(model_name)
        
        # Görüntüyü ön işle
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = Image.fromarray(image).convert("RGB")
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Tahmin yap
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Sonuçları hazırla
        result_text = f"Tahmin Edilen Hastalık: {CLASS_NAMES[predicted_class]}\nGüven: {confidence:.4f}"
        
        # Tüm hastalıklar için olasılıklar
        prob_dict = {CLASS_NAMES[i]: float(probabilities[0][i]) for i in range(len(CLASS_NAMES))}
        
        return result_text, prob_dict
    
    except Exception as e:
        return f"Hata: {str(e)}", {}

# Gradio arayüzü
with gr.Blocks(title="Model Test Arayüzü") as demo:
    gr.Markdown("Model Test Arayüzü")
    gr.Markdown("Eğitilmiş modellerle görüntü sınıflandırma yapın.")
    
    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()),
                value=list(MODELS.keys())[0],
                label="Model Seçin",
                info="Test etmek istediğiniz modeli seçin"
            )
            image_input = gr.Image(label="Görüntü Yükle", type="numpy")
            predict_btn = gr.Button("Tahmin Yap", variant="primary")
        
        with gr.Column():
            result_text = gr.Textbox(label="Tahmin Sonucu", lines=3)
            result_probs = gr.Label(label="Hastalık Olasılıkları", num_top_classes=4)
    
    # Model bilgilerini göster
    gr.Markdown("### Model Bilgileri")
    model_info = gr.Dataframe(
        headers=["Model", "Doğruluk"],
        datatype=["str", "number"],
        value=[[name, info["accuracy"]] for name, info in MODELS.items()],
        interactive=False
    )
    
    predict_btn.click(
        fn=predict,
        inputs=[image_input, model_dropdown],
        outputs=[result_text, result_probs]
    )

if __name__ == "__main__":
    demo.launch(share=False)
