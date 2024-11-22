import numpy as np
import torchvision
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from src.network_modified import ContraNetwork, Cut, View, Unpack, Channel

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ======================================================================================= Prepare ImageNet training data
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

imagenet_data = torchvision.datasets.ImageFolder(
    '/scratch/download/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train', transform=preprocess)

dataset_subset = torch.utils.data.Subset(imagenet_data, np.random.choice(len(imagenet_data), 63_000, replace=False))
train_data, test_data = torch.utils.data.random_split(dataset_subset, [60_000, 3_000])

data_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=100,
                                          shuffle=True)

test_data = torch.utils.data.DataLoader(test_data,
                                        batch_size=100,
                                        shuffle=True)


# =========================================================================================== Load test image from paper

img = Image.open("catdog.jpeg")
data = preprocess(img).unsqueeze(0)

# ======================================================================================================= Hijack network
os.environ['TORCH_HOME'] = './torchhub'
net = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
net.eval()

layers_ = list(list(net.modules())[0].features)
classifier = list(list(net.modules())[0].classifier)

# hijack layer from vgg16 to analyze them
seqs = [[0, 1], [2, 3], [4], [5, 6], [7, 8], [9], [10, 11], [12, 13], [14, 15], [16], [17, 18], [19, 20], [21, 22],
        [23], [24, 25], [26, 27], [28, 29], [30]]
cseqs = [[0, 1], [3, 4], [6]]

# for convolution part
layers = []
for seq in seqs:
    temp = []
    for l in seq:
        if isinstance(layers_[l], nn.MaxPool2d):
            # layers_[l].return_indices = True
            pass
        temp.append(layers_[l])
    # temp.append(View("\nForward"))
    layers.append(nn.Sequential(*temp))

# for classifier part
layers.append(nn.Sequential(nn.Flatten()))  # , View("\nFlatten")
for seq in cseqs:
    temp = []
    for l in seq:
        temp.append(classifier[l])
    # temp.append(View("\nForward"))
    layers.append(nn.Sequential(*temp))

# ======================================================================================================== Prepare conet
colayers = []
for j, layer in enumerate(layers):
    if isinstance(layer[0], nn.MaxPool2d):
        # colayers.append(nn.Sequential(Unpack(nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)),
        #                              # View(" Reco")
        #                              ))
        colayers.append(nn.Sequential(
            nn.ConvTranspose2d(kernel_size=2, stride=2, padding=0, in_channels=layers[j - 1][0].in_channels,
                               out_channels=layers[j - 1][0].out_channels),
            # View("Reco")
        ))
    elif isinstance(layer[0], nn.Flatten):
        colayers.append(nn.Sequential(Channel([512, 7, 7])))  # , View("Reco")

    elif isinstance(layer[0], nn.Linear):
        colayers.append(nn.Sequential(
            nn.Linear(layer[0].out_features, layer[0].in_features),
            nn.ReLU(),
            # View("Reco")
        ))
    else:
        colayers.append(
            nn.Sequential(
                nn.ConvTranspose2d(layer[0].out_channels, layer[0].in_channels, kernel_size=(3, 3), stride=1),
                Cut(),
                nn.ReLU(),
                # View("Reco")
            )
        )

# ================================================================================================= Train reconstruction
colayers = nn.Sequential(*colayers)
layers = nn.Sequential(*layers)
conet = ContraNetwork(layers, colayers, device=device)
train_generator = data_loader  # NoiseLoader(n=1000)

conet.load("Imagenet_conet_200")
# conet.load("Imagenet_conet_double")
# conet.eval(test_data, verbose=True)
#conet.train(train_generator, its=1)
#conet.eval(test_data, verbose=True)
# conet.save("Imagenet_conet_double")

# ============================================================================================ Run cascade on test image
cascasdes = conet.cascade(data)
#img = cascasdes[0].detach().numpy().reshape((3, 224, 224))
#img -= np.min(img)
#img /= np.max(img)
#img = np.swapaxes(img, 0, 2)
#img = np.swapaxes(img, 0, 1)
#plt.imshow(img)
#plt.show()

fig, axs = plt.subplots(1, 2)
axs = axs.flatten()
c = 0
for i, cascade in enumerate(cascasdes):
    if i in [0, 17]:  # 17
        img = cascade.detach().numpy().reshape((3, 224, 224))
        img -= np.min(img)
        img /= np.max(img)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 0, 1)
        axs[c].imshow(img)
        c += 1
plt.show()

# ======================================================================================================= Channel images
# modify output and see how it influences the reconstruction
img = cascasdes[0].detach().numpy().reshape((3, 224, 224))
fig, axs = plt.subplots(2, 2)
axs = axs.flatten()

img -= np.min(img)
img /= np.max(img)
img = np.swapaxes(img, 0, 2)
img = np.swapaxes(img, 0, 1)
axs[0].imshow(img)
axs[0].title.set_text(f"Reference image")

images = [(3, 26), (6, 3), (10, 112)]
gen = conet.cascade_to(data, 10)
img = next(gen)

for k, (level, channel) in enumerate(images):
    gen = conet.cascade_to(data, level)
    img = next(gen)
    for j in range(img.shape[1]):
        if j == channel:
            continue

        img[:, j] = 0

    result = gen.send(img)
    img = result.detach().numpy().reshape((3, 224, 224))
    img -= np.min(img)
    img /= np.max(img)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    axs[k + 1].imshow(img)
    axs[k + 1].title.set_text(f"Layer {level} channel {channel}")

plt.show()

"""
gen = conet.cascade_to(data, 10)
img = next(gen)
for i in range(img.shape[1]):
    if not (i in [112, 169, 180, 216]):
        continue
    img_ = img.detach().clone()
    # deactivate channels and check image
    for j in range(img.shape[1]):
        if i == j:
            continue
        img_[:, j] = 0
    result = gen.send(img_)
    img_ = result.detach().numpy().reshape((3, 224, 224))
    img_ -= np.min(img_)
    img_ /= np.max(img_)
    img_ = np.swapaxes(img_, 0, 2)
    img_ = np.swapaxes(img_, 0, 1)
    plt.title(f"Reconstruction Channel {i}")
    plt.imshow(img_)
    plt.savefig(f"/home/ythurn/Desktop/images_recon/10_{i}.png")
    plt.close()
    # plt.show()
exit()"""



for i, cascade in enumerate(cascasdes):
    img = cascade.detach().numpy().reshape((3, 224, 224))
    if i > 0:
        print(conet.conet[i - 1])
    img -= np.min(img)
    img /= np.max(img)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    plt.imshow(img)
    plt.show()

# ================================================================================================= Run network normally
# deactivate return indices for proper runs
for seq in seqs:
    for l in seq:
        if isinstance(layers_[l], nn.MaxPool2d):
            layers_[l].return_indices = False

net = net.to(device)
result = net(data.to(device)).cpu().detach().numpy().flatten()
print(np.argmax(result))
print(np.argsort(result)[::-1])
plt.plot(result)
plt.show()
