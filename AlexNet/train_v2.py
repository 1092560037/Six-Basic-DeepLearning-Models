import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

def main():
    # 使用GPU训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),  # ImageNet标准化
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])}

    data_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    image_path = os.path.join(data_root, "Flowers")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=True,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # 加载预训练的AlexNet
    net = models.alexnet(pretrained=True)
    # 修改最后的全连接层为5类
    net.classifier[6] = nn.Linear(4096, 5)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()

    # Phase 1: 冻结卷积特征提取层，仅训练分类器
    print("Phase 1: Training classifier only")
    for param in net.features.parameters():
        param.requires_grad = False
    for param in net.classifier.parameters():
        param.requires_grad = True

    optimizer_phase1 = optim.Adam(net.classifier.parameters(), lr=0.001)
    epochs_phase1 = 5
    best_acc = 0.0
    save_path = './AlexNet_v2.pth'

    # 初始化训练日志
    training_log = []

    for epoch in range(epochs_phase1):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        train_steps = len(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer_phase1.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer_phase1.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs_phase1, loss)

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        print('[phase1 epoch %d] train_loss: %.3f  val_accuracy: %.3f' % (epoch + 1, running_loss / train_steps, val_accurate))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

        # 记录日志
        training_log.append({
            'epoch': epoch + 1,
            'phase': 'Phase 1',
            'train_loss': running_loss / train_steps,
            'val_accuracy': val_accurate
        })

    # Phase 2: 解冻后半段卷积层并以更小学习率进行微调
    print("Phase 2: Fine-tuning last convolutional layers")
    # 解冻后半段卷积层：假设后半段是features的最后两个Conv层 (features[10] 和 features[12])
    # AlexNet features: 0:Conv,1:ReLU,2:MaxPool,3:Conv,4:ReLU,5:MaxPool,6:Conv,7:ReLU,8:Conv,9:ReLU,10:Conv,11:ReLU,12:MaxPool
    # 后半段：从features[6]开始解冻 (第三个Conv)
    for i in range(6, len(net.features)):
        for param in net.features[i].parameters():
            param.requires_grad = True

    # 使用更小的学习率
    optimizer_phase2 = optim.Adam([
        {'params': net.features.parameters(), 'lr': 1e-5},
        {'params': net.classifier.parameters(), 'lr': 1e-4}
    ])

    epochs_phase2 = 5
    for epoch in range(epochs_phase2):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer_phase2.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer_phase2.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs_phase2, loss)

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        print('[phase2 epoch %d] train_loss: %.3f  val_accuracy: %.3f' % (epoch + 1, running_loss / train_steps, val_accurate))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

        # 记录日志
        training_log.append({
            'epoch': epoch + 1 + epochs_phase1,  # 继续epoch编号
            'phase': 'Phase 2',
            'train_loss': running_loss / train_steps,
            'val_accuracy': val_accurate
        })

    print('Finished Training')

    # 保存训练日志到Excel
    df = pd.DataFrame(training_log)
    df.to_excel('alexnet_training_log.xlsx', index=False)
    print('Training log saved to alexnet_training_log.xlsx')

if __name__ == '__main__':
    main()