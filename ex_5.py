import torch
import torch.nn as nn
import os
import os.path
import soundfile as sf
import librosa
import numpy as np
import torch.utils.data as data
import re

# global variables
EPOCHS = 20
learning_rate = 0.001
first_channels = 32
second_channels = 62
filter_size = 4
classes = 30
fcnn_2nd_conv = 1200
fcnn_3rd_conv = 320

AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    spects = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    spects.append(item)
    return spects


def spect_loader(path, window_size, window_stride, window, normalize, max_len=101):
    y, sr = sf.read(path)
    # n_fft = 4096
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)

    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)

    # S = log(S+1)
    spect = np.log1p(spect)

    # make all spects with the same dims
    # TODO: change that in the future
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:, :max_len]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)

    return spect


class GCommandLoader(data.Dataset):
    """A google command data set loader where the wavs are arranged in this way: ::
        root/one/xxx.wav
        root/one/xxy.wav
        root/one/xxz.wav
        root/head/123.wav
        root/head/nsdf3.wav
        root/head/asd932_.wav
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the spect to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        spects (list): List of (spects path, class_index) tuples
        STFT parameter: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101):
        classes, class_to_idx = find_classes(root)
        spects = make_dataset(root, class_to_idx)

        if len(spects) == 0:
            raise (RuntimeError(
                "Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(
                    AUDIO_EXTENSIONS)))

        self.root = root
        self.spects = spects
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len
        self.len = len(self.spects)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        # print(index)
        path, target = self.spects[index]
        spect = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)
        # print (path)
        if self.transform is not None:
            spect = self.transform(spect)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return spect, target

    def __len__(self):
        return self.len

    def set_spects(self, sp):
        self.spects = sp


def load_train_and_valid(batch_size):
    train_dataset = GCommandLoader('files/train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, sampler=None)

    valid_dataset = GCommandLoader('files/valid')
    validation_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)
    return [train_loader, validation_loader]


def load_test(path, batch_size):
    dataset = GCommandLoader(path)
    # sorting the wav files
    new_spects = sort_test(dataset)
    # set the datset to use the sorted spects
    dataset.set_spects(new_spects)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)
    return [test_loader, dataset]


class ConvolutionNet(nn.Module):
    def __init__(self):
        super(ConvolutionNet, self).__init__()
        # The network is a convolution Network that includes 4 layers using Relu activation function and max pooling.
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, first_channels, kernel_size=filter_size, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(first_channels, second_channels, kernel_size=filter_size, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(second_channels, 25, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(25, 30, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # performing drop out in order to avoid over fitting
        self.drop_out = nn.Dropout()
        # fully connected network with 2 hidden layers
        self.fc1 = nn.Linear(1800, fcnn_2nd_conv)
        self.fc2 = nn.Linear(fcnn_2nd_conv, fcnn_3rd_conv)
        self.fc3 = nn.Linear(fcnn_3rd_conv, classes)

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = output.reshape(output.size(0), -1)
        output = self.drop_out(output)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output


def train(train_loader, epoch, EPOCHS, loss_function, optimizer, model):
    model.train()
    loader_size = len(train_loader)
    for i, (images, labels) in enumerate(train_loader):
        # images = images.to("cuda")
        # labels = labels.to("cuda")
        outputs = model(images)
        loss = loss_function(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        if (i + 1) % 50 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, EPOCHS, i + 1, loader_size, loss.item(), (correct / total) * 100))


def test_model(valid_loader, model):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            # images = images.to("cuda")
            # labels = labels.to("cuda")
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model: {} %'.format((correct / total) * 100))


def write_to_file(test_loader, model, dataset):
    spects = dataset.spects
    wav_files = []
    predictions = []
    print_list = []
    classes_dict = {0: 'bed', 1: "bird", 2: "cat", 3: "dog", 4: "down", 5: "eight", 6: "five", 7: "four", 8: "go",
                    9: "happy", 10: "house", 11: "left", 12: "marvin",
                    13: "nine", 14: "no", 15: "off", 16: "on", 17: "one", 18: "right", 19: "seven",
                    20: "sheila", 21: "six", 22: "stop", 23: "three", 24: "tree", 25: "two", 26: "up",
                    27: "wow", 28: "yes", 29: "zero"}
    for spect in spects:
        file_name = spect[0].split('/test')
        file_name = file_name[len(file_name) - 1]
        file_name = file_name.replace("\\", "")
        wav_files.append(file_name)
    model.eval()
    with torch.no_grad():
        # get the predictions of the model on the test loader
        for images, labels in test_loader:
            # images = images.to("cuda")
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.tolist())
        # create the needed format of the output file test_y
        for wav_file, prediction in zip(wav_files, predictions):
            line = wav_file + "," + str(classes_dict[prediction])
            print_list.append(line)
        with open('test_y', 'w') as f:
            for line in print_list:
                f.write("%s\n" % line)
    f.close()


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def sort_test(dataset):
    path = 'files/sub_dir/test'
    spects = dataset.spects
    wav_files = []
    new_spects = []
    for spect in spects:
        file_name = spect[0].split('/test')
        file_name = file_name[len(file_name) - 1]
        wav_files.append(file_name)
    wav_files.sort(key=natural_keys)
    for wav in wav_files:
        tup = (path + wav, 1)
        new_spects.append(tup)
    return new_spects


def Network():
    # load the data: train, valid and test.
    train_loader, validation_loader = load_train_and_valid(batch_size=200)
    test_loader, test_dataset = load_test('files/sub_dir/', batch_size=200)
    model = ConvolutionNet()
    # model = ConvolutionNet().to("cuda")

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(EPOCHS):
        train(train_loader, epoch, EPOCHS, loss_function, optimizer, model)
    # Check the model accuracy with the validation loader
    test_model(validation_loader, model)
    write_to_file(test_loader, model, test_dataset)


if __name__ == '__main__':
    Network()