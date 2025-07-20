import torch
from PIL import Image
from torchvision import transforms
from mcunet.model_zoo import build_model

def load_model(checkpoint_path):
    """加载模型和权重"""
    # 注意：net_id必须与检查点匹配
    net, image_size, _ = build_model(net_id="mcunet-512kB", pretrained=False)
    # net, image_size, _ = build_model(net_id="mcunet-10fps-vww", pretrained=False)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}  # 处理可能的分布式训练前缀

    net.load_state_dict(state_dict)
    net.eval()
    return net, image_size

def preprocess_image(image_path, image_size):
    """图像预处理"""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(Image.open(image_path).convert('RGB')).unsqueeze(0)

def predict(model, input_tensor):
    """执行推理"""
    with torch.no_grad():
        if torch.cuda.is_available():
            model = model.to('cuda')
            input_tensor = input_tensor.to('cuda')
        return model(input_tensor)

# 使用示例
if __name__ == "__main__":
    # 1. 初始化
    model, image_size = load_model("/home/verse/Python/GD_Net/mcunet_model/mcunet-512kb-2mb_imagenet.pth")
    # model, image_size = load_model("/home/verse/Python/GD_Net/mcunet_model/mcunet-10fps_vww.pth")

    # 2. 预处理
    input_tensor = preprocess_image("/home/verse/Pictures/drone_a.jpg", image_size)

    # 3. 推理
    output = predict(model, input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # 4. 解析结果
    predicted_idx = torch.argmax(probabilities).item()
    print(f"Top prediction: Class {predicted_idx} ({probabilities[predicted_idx]:.2%})")