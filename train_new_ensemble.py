"""

    Defect Verification AI Classifier 개발 Project
    해성DS - 부산대학교 시각 지능 및 인지 연구실
    2022.08

    EfficientNet 기반의 Defect Verification Classifier Training / Evaluation
    Training 된 Model의 ONNX Export

    Example commands:

    # Training
    python train.py --path_hds_cls PATH_TO_DATASET --ckpt1 PATH_TO_FIRST_MODEL_FILE --ckpt2 PATH_TO_SECOND_MODEL_FILE --ckpt3 PATH_TO_THIRD_MODEL_FILE --batch_size BATCH_SIZE --epochs EPOCHES

    # Testing
    python train.py --path_hds_cls PATH_TO_DATASET --ckpt PATH_TO_ENSEMBLE_MODEL_FILE --test_only

    # ONNX Export
    python train.py --path_hds_cls PATH_TO_DATASET --ckpt PATH_TO_ENSEMBLE_MODEL_FILE --test_only --onnx

"""


## Import libraries
# pip install onnx
# pip install onnxruntime-gpu
import onnx
import onnxruntime as ort

import numpy as np
import torch
import torch.nn as nn
import torchsummary
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import copy
import random
import argparse
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import torchvision.transforms.functional as F
from tqdm import tqdm

# import pandas as pd
# import json
# from PIL import Image
# import cv2
# from torch.optim import lr_scheduler
# from torchvision import transforms
# import os

## Import EfficientNet module
# 설치: pip install efficientnet_pytorch
from efficientnet_pytorch import EfficientNet


class_names = {
    "0": "DEF",   # "0": "DEF"
    "1": "OIL",   # "1": "OIL"
    "2": "REF",   # "2": "REF"
}

num_show_img = 5

# 데이터 체크 함수
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


parser = argparse.ArgumentParser(description="Defect Verification Classifier")

parser.add_argument('--path_hds_cls',
                    type=str,
                    default='training_set',
                    help='HDS Classification Dataset 경로')
parser.add_argument('--model1', type=str, default='efficientnet-b0',
                    choices=('efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2'),
                    help='EfficientNet 모델 종류')

parser.add_argument('--model2', type=str, default='efficientnet-b1',
                    choices=('efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2'),
                    help='EfficientNet 모델 종류')

parser.add_argument('--model3', type=str, default='efficientnet-b2',
                    choices=('efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2'),
                    help='EfficientNet 모델 종류')

parser.add_argument('--test_only',
                    action='store_true',
                    help='Test만 실시하기 위한 옵션')
parser.add_argument('--ckpt',
                    type=str,
                    default=None,
                    help='Pretrained checkpoint 경로')
parser.add_argument('--ckpt1',
                    type=str,
                    default=None,
                    help='Pretrained checkpoint 경로')
parser.add_argument('--ckpt2',
                    type=str,
                    default=None,
                    help='Pretrained checkpoint 경로')
parser.add_argument('--ckpt3',
                    type=str,
                    default=None,
                    help='Pretrained checkpoint 경로')
parser.add_argument('--onnx',
                    action='store_true',
                    help='Test 후 ONNX format으로 저장하기 위한 옵션')

parser.add_argument('--batch_size',
                    type=int,
                    default=32,
                    help='훈련을 위한 Batch size')
parser.add_argument('--epochs',
                    type=int,
                    default=40,
                    help='훈련 Epoch 수')

parser.add_argument('--patch_height',
                    type=int,
                    default=240,
                    help='height of a patch to crop')
parser.add_argument('--patch_width',
                    type=int,
                    default=240,
                    help='width of a patch to crop')

parser.add_argument('--seed',
                    type=int,
                    default=555,
                    help='random seed point')

args = parser.parse_args()

## Print input arguments
print('\n\n=== Arguments ===')
for key in sorted(vars(args)):
    print(key, ':',  getattr(args, key))
print('\n')


random_seed = args.seed
random.seed(random_seed)
torch.manual_seed(random_seed)

# image_size = EfficientNet.get_image_size(model_name)
# print(image_size)

batch_size = args.batch_size


## 커스텀 데이터셋 생성
# 데이터셋 구조
# DATA_ROOT = args.path_hds_cls
# DATA_ROOT/DEF : DEF 샘플 이미지
# DATA_ROOT/OIL : OIL 샘플 이미지
# DATA_ROOT/REF : 정상 샘플 이미지
# 
# ImageNet normalization을 적용

def_dataset = datasets.ImageFolder(
    args.path_hds_cls, transforms.Compose([
        transforms.Resize((args.patch_height, args.patch_width)),
        # transforms.RandomRotation(50),
        # transforms.RandomHorizontalFlip(p = 0.5),
        transforms.Grayscale(), 
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=[0.456], std=[0.224]),
    ])
)

## 훈련 / 검증 / 시험 데이터 분할 (train:valid:test = 8:1:1)
datasets = dict()

train_idx, tmp_idx = train_test_split(
    list(range(len(def_dataset))), test_size=0.2, random_state=random_seed
)
datasets['train'] = Subset(def_dataset, train_idx)
tmp_dataset = Subset(def_dataset, tmp_idx)

val_idx, test_idx = train_test_split(
    list(range(len(tmp_dataset))), test_size=0.5, random_state=random_seed
)
datasets['valid'] = Subset(tmp_dataset, val_idx)
datasets['test']  = Subset(tmp_dataset, test_idx)

## Data loader 선언
dataloaders = dict()
data_num = dict()
batch_num = dict()
for split in ['train', 'valid', 'test']:
    dataloaders[split] = DataLoader(
        datasets[split], batch_size=batch_size, shuffle=True, num_workers=4
    )
    data_num[split] = len(datasets[split])
    batch_num[split] = len(dataloaders[split])

    print('{} samples : {}'.format(split, data_num[split]))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set gpu

criterion = nn.CrossEntropyLoss()


# 모델 학습 코드
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
    
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            pbar = tqdm(total=data_num[phase], dynamic_ncols=True)
            pbar.set_description("{:<10s}| Epoch {:5d} / {:5d}".format(
                phase, epoch, num_epochs-1))

            running_loss, running_corrects, num_cnt = 0.0, 0, 0
            
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_cnt += len(labels)

                # Log strings
                err_str = "{:<10s}| Epoch {:5d} / {:5d} | Loss = {:9.4f} | Acc = {:7.4f}".format(
                    phase, epoch, num_epochs-1, running_loss / num_cnt, running_corrects / num_cnt
                )

                pbar.set_description(err_str)
                pbar.update(len(labels))

            pbar.close()

            if phase == 'train':
                scheduler.step()
            
            epoch_loss = float(running_loss / num_cnt)
            epoch_acc  = float((running_corrects.double() / num_cnt).cpu()*100)
            
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)

            # print("{:<10s}| Epoch {:5d} / {:5d} | Loss = {:9.4f} | Acc = {:7.4f}".format(
            #     phase, epoch, num_epochs, epoch_loss, epoch_acc
            # ))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
#                 best_model_wts = copy.deepcopy(model.module.state_dict())
                print('==> best model saved - %d / %.1f'%(best_idx, best_acc))
                
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: %d - %.1f' %(best_idx, best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # torch.save(model.state_dict(), 'model.pt')
    torch.save(model, '{}_e{}_acc_{:08.4f}.pt'.format("efficientnet-ensemble", args.epochs, best_acc))
    print('model saved to: {}_e{}_acc_{:08.4f}.pt'.format("efficientnet-ensemble", args.epochs, best_acc))
    return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc


# 모델 테스트 및 시각화
def test_and_visualize_model(model, phase = 'test', num_images=16):
    # phase = 'train', 'valid', 'test'
    if args.test_only:
        model = torch.load(args.ckpt)

    was_training = model.training
    model.eval()
    fig = plt.figure(figsize=(16,16))

    running_loss, running_corrects, num_cnt = 0.0, 0, 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)  # batch의 평균 loss 출력

            running_loss    += loss.item() * inputs.size(0)
            running_corrects+= torch.sum(preds == labels.data)
            num_cnt += inputs.size(0)  # batch size

        since = time.time()
        outputs = model(inputs)
        time_elapsed = time.time() - since
        test_loss = running_loss / num_cnt
        test_acc  = running_corrects.double() / num_cnt

        print('test complete in {:}s'.format(time_elapsed % 60))
        print('test done : loss/acc : %.2f / %.5f' % (test_loss, test_acc*100))

    # 예시 그림 plot
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # 예시 그림 plot
            for j in range(1, num_images+1):
                ax = plt.subplot(num_images//4, 4, j)
                ax.axis('off')
                ax.set_title('%s : %s -> %s'%(
                    'True' if class_names[str(labels[j-1].cpu().numpy())]==class_names[str(preds[j-1].cpu().numpy())] else 'False',
                    class_names[str(labels[j-1].cpu().numpy())], class_names[str(preds[j-1].cpu().numpy())]))
                imshow(inputs.cpu().data[j-1])
            plt.savefig('result2.png', dpi=300)
            plt.show(block=False)
            break

    model.train(mode=was_training);  # 다시 train모드로

    return test_acc


## EfficientNet pre-trained model 불러오기
model_name1 = args.model1
model_name2 = args.model2
model_name3 = args.model3

if args.ckpt1 is not None:
    model1 = torch.load(args.ckpt1)
else:
    # 1-channel input, 3-channel output으로 설정
    model1 = EfficientNet.from_pretrained(model_name1, in_channels=1, num_classes=3)
    model1 = model1.to(device)

if args.ckpt2 is not None:
    model2 = torch.load(args.ckpt2)
else:
    # 1-channel input, 3-channel output으로 설정
    model2 = EfficientNet.from_pretrained(model_name2, in_channels=1, num_classes=3)
    model2 = model2.to(device)
    
if args.ckpt3 is not None:
    model3 = torch.load(args.ckpt3)
else:
    # 1-channel input, 3-channel output으로 설정
    model3 = EfficientNet.from_pretrained(model_name3, in_channels=1, num_classes=3)
    model3 = model3.to(device)

class MyEnsemble(nn.Module):

    def __init__(self, modelA, modelB, modelC, input):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.modelA.trainable = False
        self.modelB.trainable = False
        self.modelC.trainable = False

        self.fc1 = nn.Linear(input, 3)
        # self.fc1.trainable = True

    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)
        out3 = self.modelC(x)

        out = out1 + out2 + out3

        x = self.fc1(out)
        return x
    
    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self.modelA.set_swish(memory_efficient=False)
        self.modelB.set_swish(memory_efficient=False)
        self.modelC.set_swish(memory_efficient=False)
    
model = MyEnsemble(model1, model2, model3, 3)
model.to(device)

if not args.test_only:
    optimizer_ft = optim.SGD(model.parameters(),
                             lr = 1e-7,
                             # lr = 0.05,
                             momentum=0.9,
                             weight_decay=1e-4)

    lmbda = lambda epoch: 0.98739
    exp_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer_ft, lr_lambda=lmbda)

    print('')
    # torchsummary.summary(model, (3, args.patch_height, args.patch_width))
    model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc = \
        train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=args.epochs)

    ## 결과 그래프
    print('best model : %d - %1.f / %.1f'%(best_idx, valid_acc[best_idx], valid_loss[best_idx]))
    fig, ax1 = plt.subplots()

    ax1.plot(train_acc, 'b-')
    ax1.plot(valid_acc, 'r-')
    plt.plot(best_idx, valid_acc[best_idx], 'ro')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('acc', color='k')
    ax1.tick_params('y', colors='k')

    ax2 = ax1.twinx()
    ax2.plot(train_loss, 'g-')
    ax2.plot(valid_loss, 'k-')
    plt.plot(best_idx, valid_loss[best_idx], 'ro')
    ax2.set_ylabel('loss', color='k')
    ax2.tick_params('y', colors='k')

    plt.legend(['train acc', 'valid_acc', 'train loss', 'valid loss'])

    fig.tight_layout()
    plt.show(block=False)

test_acc = test_and_visualize_model(model, phase = 'test')

## ONNX 를 이용한 최종 모델 저장
if args.onnx:
    print("Save the final model (ACC : {:07.5f}) using ONNX".format(test_acc))

    import torch.onnx
    # ONNX export를 위한 swish 교체
    model.set_swish(memory_efficient=False)
    model.eval()
    B, C, H, W = batch_size, 1, args.patch_height, args.patch_width

    # Model Graph 계산을 위한 Dummy 이용 Forward Operation
    print("Input data size : [{}, {}, {}, {}]".format(B, C, H, W))

    x = torch.randn(B, C, H, W, requires_grad=True).to(device)
    y = model(x)

    # Model 변환
    # 첫 번째 Batch 차원은 Dynamic 차원으로 설정
    path_save = "{}_testacc_{:07.5f}.onnx".format("efficient-ensemble", test_acc)
    torch.onnx.export(
        model, x, path_save,
        # verbose=True,
        input_names=["image"], output_names=["cls"],
        export_params=True, training=torch.onnx.TrainingMode.EVAL,
        # operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        dynamic_axes={"image": {0 : "batch_size"}, "cls": {0 : "batch_size"}}
    )

    # 변환 된 Model의 유효성을 확인
    model_onnx = onnx.load(path_save)
    onnx.checker.check_model(model_onnx)
    print(onnx.helper.printable_graph(model_onnx.graph))

    # ONNX 를 이용한 실행 테스트
    ort_session = ort.InferenceSession(
        path_save,  providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_outs = ort_session.run(
        None,
        {"image": np.random.randn(B, C, H, W).astype(np.float32)},
    )

    # ONNX 런타임에서 계산된 결과값
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    
    print("ONNXRuntime을 이용한 Exported Model 실행 Test 완료")
    
    print("ONNXRuntime을 이용한 Exported Model Consistency Test 시작")

    # 실제 데이터를 이용한 Test
    for i, (inputs, labels) in enumerate(dataloaders['test']):
        inputs = inputs.to(device)

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
        ort_outs = ort_session.run(None, ort_inputs)

        model_outs = model(inputs)

        # np.testing.assert_allclose(to_numpy(model_outs), ort_outs[0],
        #                            rtol=0.01, atol=0.01)

        ort_preds = np.argmax(ort_outs[0], axis=1)
        model_preds = np.argmax(to_numpy(model_outs), axis=1)

        np.testing.assert_equal(model_preds, ort_preds)

    print("ONNXRuntime을 이용한 Exported Model Consistency Test 완료")
    print("저장된 ONNX Model : {}".format(path_save))
