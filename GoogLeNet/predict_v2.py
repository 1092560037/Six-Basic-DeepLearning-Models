import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model_v2 import GoogLeNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img_path = "tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)

    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    model = GoogLeNet(num_classes=5, aux_logits=False, init_weights=False, replace_5x5=True).to(device)

    weights_path = "./googleNet_v2.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    with torch.no_grad():
        output = model(img.to(device)).squeeze()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).item()

    print_res = "class: {}   prob: {:.3f}".format(class_indict[str(predict_cla)],
                                                  predict[predict_cla].item())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3f}".format(class_indict[str(i)],
                                                   predict[i].item()))
    plt.show()


if __name__ == '__main__':
    main()
