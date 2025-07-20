import torch
import torch.nn as nn
from mcunet.model_zoo import build_model
from PIL import Image
from torchvision import transforms

class_names = ['background', 'person']

def load_model(checkpoint_path):
    model, image_size, _ = build_model(net_id="mcunet-10fps-vww", pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, image_size

if __name__ == "__main__":
    model, image_size = load_model("/home/verse/Python/GD_Net/mcunet_model/mcunet-10fps_vww.pth")
    print("Model image size:", image_size)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img = transform(Image.open("/home/verse/Pictures/mine.jpg").convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        logits = model(img)  # logits: [1, 2]
        probs = torch.nn.functional.softmax(logits[0], dim=0)
        pred = torch.argmax(probs).item()
        print(f"Top prediction: Class {pred} ({class_names[pred]}) - {probs[pred]:.2%}")
