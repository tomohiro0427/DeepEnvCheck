import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def check_environment():
    print("=== Environment Check ===")
    print(f"PyTorch version : {torch.__version__}")
    print(f"Transformers version : {__import__('transformers').__version__}")
    print(f"Datasets version : {__import__('datasets').__version__}")
    print(f"CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device name : {torch.cuda.get_device_name(0)}")
    print("=========================\n")

def main():
    # 0. 動作環境の確認
    check_environment()

    # 1. データセットを読み込み（CIFAR-10）
    print("Loading CIFAR-10 dataset...")
    dataset = load_dataset("cifar10")
    image = dataset["test"][0]["img"]  # CIFAR-10のPIL画像

    # 2. モデルと前処理器を読み込み（Hugging Face）
    print("Loading pretrained model from Hugging Face...")
    model_name = "microsoft/resnet-18"  # 軽量で推論が速いモデル
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)

    # 3. 前処理
    transform = Compose([
        Resize((224, 224)),  # ResNet用にリサイズ
        ToTensor(),
        Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    inputs = transform(image).unsqueeze(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = inputs.to(device)

    # 4. 推論
    print("Running inference...")
    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs.logits
        pred = logits.argmax(-1).item()

    # 5. 結果出力
    print("\n=== Inference Result ===")
    print(f"Predicted class id: {pred}")
    print(f"Label name: {model.config.id2label.get(pred, 'unknown')}")
    print("=========================")

if __name__ == "__main__":
    main()
