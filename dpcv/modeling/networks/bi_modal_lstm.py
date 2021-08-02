import torch
import torch.nn as nn


class BiModelLSTM(nn.Module):
    def __init__(self):
        super(BiModelLSTM, self).__init__()
        self.audio_branch = nn.Linear(in_features=68, out_features=32)
        self.image_branch_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=9, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.image_branch_linear = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=16 * 8 * 8, out_features=1024),
            nn.Linear(in_features=1024, out_features=128),
            nn.Dropout(0.2)
        )
        # the original paper set hidden_size to 128
        # self.lstm = nn.LSTM(input_size=160, hidden_size=128)
        # self.out_linear = nn.Linear(in_features=128, out_features=5)
        self.lstm = nn.LSTM(input_size=160, hidden_size=512)
        self.out_linear = nn.Linear(in_features=512, out_features=5)

    def forward(self, audio_feature, img_feature):
        x_audio = self.audio_branch(audio_feature)  # (bs * 6, 32)
        x_img = self.image_branch_conv(img_feature)  # (bs * 6, 16 * 8 * 8)
        x_img = self.image_branch_linear(x_img)  # (bs * 6, 128)
        x = torch.cat([x_audio, x_img], dim=-1)
        x = x.view(6, -1, 160)  # x_shape = (6, bs, 160)
        x, _ = self.lstm(x)  # x_shape = (6, bs, 128)
        x = self.out_linear(x)  # x_shape = (6, bs, 5)
        x = x.permute(1, 0, 2)  # x_shape = (bs, 6, 5)
        y = torch.sigmoid(x).mean(dim=1)  # y_shape = (bs, 5)
        return y


class ImgLSTM(nn.Module):
    def __init__(self):
        super(ImgLSTM, self).__init__()
        self.image_branch_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=9, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.image_branch_linear = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=16 * 8 * 8, out_features=1024),
            nn.Linear(in_features=1024, out_features=128),
            nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=128)
        self.out_linear = nn.Linear(in_features=128, out_features=5)

    def forward(self, img_feature):
        x_img = self.image_branch_conv(img_feature)  # (bs * 6, 16 * 8 * 8)
        x_img = self.image_branch_linear(x_img)  # (bs * 6, 128)
        x = x_img.view(6, -1, 128)  # x_shape = (6, bs, 160)
        x, _ = self.lstm(x)  # x_shape = (6, bs, 128)
        x = self.out_linear(x)  # x_shape = (6, bs, 5)
        x = x.permute(1, 0, 2)  # x_shape = (bs, 6, 5)
        y = torch.sigmoid(x).mean(dim=1)  # y_shape = (bs, 5)
        return y


class AudioLSTM(nn.Module):

    def __init__(self):
        super(AudioLSTM, self).__init__()
        self.audio_branch = nn.Linear(in_features=68, out_features=32)
        self.lstm = nn.LSTM(input_size=32, hidden_size=128)
        self.out_linear = nn.Linear(in_features=128, out_features=5)

    def forward(self, audio_feature):
        x_audio = self.audio_branch(audio_feature)  # (bs * 6, 32)
        x = x_audio.view(6, -1, 32)  # x_shape = (6, bs, 160)
        x, _ = self.lstm(x)  # x_shape = (6, bs, 128)
        x = self.out_linear(x)  # x_shape = (6, bs, 5)
        x = x.permute(1, 0, 2)  # x_shape = (bs, 6, 5)
        y = torch.sigmoid(x).mean(dim=1)  # y_shape = (bs, 5)
        return y


def get_bi_modal_lstm_model():
    bi_modal_lstm = BiModelLSTM()
    bi_modal_lstm.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return bi_modal_lstm


def get_img_modal_lstm_model():
    img_lstm = ImgLSTM()
    img_lstm.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return img_lstm


def get_aud_modal_lstm_model():
    aud_lstm = AudioLSTM()
    aud_lstm.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return aud_lstm


if __name__ == "__main__":
    """
    basically batch size is the number of video
    """
    bs = 2
    au_ft = torch.randn((bs * 6, 68))
    im_ft = torch.randn((bs * 6, 3, 112, 112))
    # bi_model = BiModelLSTM()
    # out = bi_model(au_ft, im_ft)

    # img_model = ImgLSTM()
    # out = img_model(im_ft)

    aud_model = AudioLSTM()
    out = aud_model(au_ft)
    print(out.shape)



