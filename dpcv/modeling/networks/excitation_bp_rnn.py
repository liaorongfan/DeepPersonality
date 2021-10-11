import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models.utils import load_state_dict_from_url

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class MultiLSTMCellRelu(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers=1, use_dropout=None, dropout=0.5):
        """"Constructor of the class"""
        super(LSTMCellRelu, self).__init__()

        self.nlayers = nlayers
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=dropout)

        ih, hh = [], []
        for i in range(nlayers):
            ih.append(nn.Linear(input_size, 4 * hidden_size))
            hh.append(nn.Linear(hidden_size, 4 * hidden_size))
        self.w_ih = nn.ModuleList(ih)
        self.w_hh = nn.ModuleList(hh)

    def forward(self, input, hidden):
        """"Defines the forward computation of the LSTMCell"""
        hy, cy = [], []
        for i in range(self.nlayers):
            hx, cx = hidden[0][i], hidden[1][i]
            gates = self.w_ih[i](input) + self.w_hh[i](hx)
            i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)

            i_gate = torch.sigmoid(i_gate)
            f_gate = torch.sigmoid(f_gate)
            c_gate = torch.relu(c_gate)
            o_gate = torch.sigmoid(o_gate)

            ncx = (f_gate * cx) + (i_gate * c_gate)
            nhx = o_gate * torch.relu(ncx)
            cy.append(ncx)
            hy.append(nhx)
            if self.use_dropout:
                input = self.dropout(nhx)

        hy, cy = torch.stack(hy, 0), torch.stack(cy, 0)
        return hy, cy


class LSTMCellRelu(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCellRelu, self).__init__()

        self.w_ih = nn.Linear(input_size, 4 * hidden_size)
        self.w_hh = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, input, hidden):
        """"Defines the forward computation of the LSTMCell"""
        hx, cx = hidden[0], hidden[1]
        gates = self.w_ih(input) + self.w_hh(hx)
        i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        c_gate = torch.relu(c_gate)
        o_gate = torch.sigmoid(o_gate)

        cy = (f_gate * cx) + (i_gate * c_gate)
        hy = o_gate * torch.relu(cy)

        return hy, cy


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['alexnet'], progress=progress)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


class AlexNetLSTM(nn.Module):

    def __init__(self):
        super(AlexNetLSTM, self).__init__()
        self.extractor = alexnet(pretrained=True)
        self.lstm_cell = LSTMCellRelu(4096, 2048)
        self.classifier = nn.Linear(2048, 2)
        self.hx = Variable(torch.randn(1, 2048)).cuda()
        self.cx = Variable(torch.randn(1, 2048)).cuda()

    def forward(self, x):
        x = self.extractor(x)
        # (tim_step=1, batch_size=1, input_dim=4096, hidden_dim=2048)
        self.hx, self.cx = self.lstm_cell(x, (self.hx, self.cx))
        y = self.classifier(self.hx)
        return y

