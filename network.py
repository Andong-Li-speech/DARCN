import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class darcn(nn.Module):
    def __init__(self, causal_flag, stage_number):
        super(darcn, self).__init__()
        self.causal_flag = causal_flag
        self.Iter = stage_number
        self.aunet = AUnet()
        self.mnet = MNet(self.causal_flag)
        self.sgru = Stage_GRU()

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        ori_x = x
        curr_x = x
        h = None
        out_list = []
        for i in range(self.Iter):
            x = torch.cat((ori_x, curr_x), dim = 1)
            h = self.sgru(x, h)
            att_list = self.aunet(x)
            curr_x = self.mnet(h, att_list)
            out_list.append(curr_x.squeeze())
        return out_list

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            'aunet': model.aunet,
            'mnet': model.mnet,
            'sgru': model.sgru,
            'causal_flag': model.causal_flag,
            'Iter': model.Iter,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }

        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package


class AUnet(nn.Module):
    def __init__(self):
        super(AUnet, self).__init__()
        self.en = AEncoder()
        self.de = ADecoder()
    def forward(self, x):
        x, en_list = self.en(x)
        de_list = self.de(x, en_list)
        return de_list

class MNet(nn.Module):
    def __init__(self, causal_flag):
        super(MNet, self).__init__()
        self.causal_flag = causal_flag
        self.en = MEncoder()
        self.de = MDecoder()
        self.glu_list = nn.ModuleList([GLU(dilation=2 ** i, in_channel=256, causal_flag=causal_flag) for i in range(6)])

    def forward(self, x, att_list):
        x, en_list = self.en(x, att_list)
        batch_size, _, seq_len, _ = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, -1, seq_len)
        x_skip = Variable(torch.zeros(x.shape), requires_grad = True).to(x.device)
        for i in range(6):
            x = self.glu_list[i](x)
            x_skip = x_skip + x
        x = x_skip
        x = x.view(batch_size, 64, 4, seq_len)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = self.de(x, en_list)
        del x_skip, en_list
        return x


class AEncoder(nn.Module):
    def __init__(self):
        super(AEncoder, self).__init__()
        self.pad1 = nn.ConstantPad2d((0, 0, 1, 0), value=0.)
        self.pad2 = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.en1 = nn.Sequential(
            self.pad1,
            nn.Conv2d(2, 16, kernel_size=(2, 5), stride=(1, 2)),
            nn.BatchNorm2d(16),
            nn.ELU())
        self.en2 = nn.Sequential(
            self.pad2,
            nn.Conv2d(16, 32, kernel_size=(2, 5), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ELU())
        self.en3 = nn.Sequential(
            self.pad2,
            nn.Conv2d(32, 32, kernel_size=(2, 5), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ELU())
        self.en4 = nn.Sequential(
            self.pad2,
            nn.Conv2d(32, 64, kernel_size=(2, 5), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ELU())
        self.en5 = nn.Sequential(
            self.pad2,
            nn.Conv2d(64, 64, kernel_size=(2, 5), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ELU())

    def forward(self, x):
        x_list = []
        x = self.en1(x)
        x_list.append(x)
        x = self.en2(x)
        x_list.append(x)
        x = self.en3(x)
        x_list.append(x)
        x = self.en4(x)
        x_list.append(x)
        x = self.en5(x)
        return x, x_list

class ADecoder(nn.Module):
    def __init__(self):
        super(ADecoder, self).__init__()
        self.up_f = up_Chomp_F(1)
        self.down_f = down_Chomp_F(1)
        self.chomp_t = Chomp_T(1)
        self.de1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(64),
            nn.ELU())
        self.de2 = nn.Sequential(
            nn.ConvTranspose2d(64*2, 32, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(32),
            nn.ELU())
        self.de3 = nn.Sequential(
            nn.ConvTranspose2d(32*2, 32, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(32),
            nn.ELU())
        self.de4 = nn.Sequential(
            nn.ConvTranspose2d(32*2, 16, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(16),
            nn.ELU())
        self.de5 = nn.Sequential(
            nn.ConvTranspose2d(16*2, 16, kernel_size=(2, 5), stride=(1, 2)),
            self.chomp_t,
            nn.BatchNorm2d(16),
            nn.ELU())

    def forward(self, x, x_list):
        de_list = []
        x = self.de1(x)
        de_list.append(x)
        x = self.de2(torch.cat((x, x_list[-1]), dim= 1))
        de_list.append(x)
        x = self.de3(torch.cat((x, x_list[-2]), dim= 1))
        de_list.append(x)
        x = self.de4(torch.cat((x, x_list[-3]), dim = 1))
        de_list.append(x)
        x = self.de5(torch.cat((x, x_list[-4]), dim = 1))
        de_list.append(x)
        return de_list


class MEncoder(nn.Module):
    def __init__(self):
        super(MEncoder, self).__init__()
        self.pad1 = nn.ConstantPad2d((0, 0, 1, 0), value=0.)
        self.pad2 = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.pad3 = nn.ConstantPad2d((2, 2, 1, 0), value=0.)
        self.fen1 = nn.Sequential(
            self.pad3,
            nn.Conv2d(16, 16, kernel_size=(2, 5), stride=(1, 1)))
        self.ben1 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ELU())
        self.fen2 = nn.Sequential(
            self.pad1,
            nn.Conv2d(16, 16, kernel_size=(2, 5), stride=(1, 2)))
        self.ben2 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ELU())
        self.fen3 = nn.Sequential(
            self.pad2,
            nn.Conv2d(16, 32, kernel_size=(2, 5), stride=(1, 2)))
        self.ben3 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ELU())
        self.fen4 = nn.Sequential(
            self.pad2,
            nn.Conv2d(32, 32, kernel_size=(2, 5), stride=(1, 2)))
        self.ben4 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ELU())
        self.fen5 = nn.Sequential(
            self.pad2,
            nn.Conv2d(32, 64, kernel_size=(2, 5), stride=(1, 2)))
        self.ben5 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ELU())
        self.en6 = nn.Sequential(
            self.pad2,
            nn.Conv2d(64, 64, kernel_size=(2, 5), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ELU())
        self.point_conv1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size= 1),
            nn.Sigmoid())
        self.point_conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1),
            nn.Sigmoid())
        self.point_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.Sigmoid())
        self.point_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.Sigmoid())
        self.point_conv5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x, att_list):
        x_list = []
        x = self.fen1(x)
        x = self.ben1(x * self.point_conv1(att_list[-1]))
        x = self.fen2(x)
        x = self.ben2(x * self.point_conv2(att_list[-2]))
        x_list.append(x)
        x = self.fen3(x)
        x = self.ben3(x * self.point_conv3(att_list[-3]))
        x_list.append(x)
        x = self.fen4(x)
        x = self.ben4(x * self.point_conv4(att_list[-4]))
        x_list.append(x)
        x = self.fen5(x)
        x = self.ben5(x * self.point_conv5(att_list[-5]))
        x_list.append(x)
        x = self.en6(x)
        x_list.append(x)
        return x, x_list


class MDecoder(nn.Module):
    def __init__(self):
        super(MDecoder, self).__init__()
        self.up_f = up_Chomp_F(1)
        self.down_f = down_Chomp_F(1)
        self.chomp_t = Chomp_T(1)
        self.att1 = Attention_Block(64, 64, 64)
        self.att2 = Attention_Block(64, 64, 64)
        self.att3 = Attention_Block(32, 32, 32)
        self.att4 = Attention_Block(32, 32, 32)
        self.att5 = Attention_Block(16, 16, 16)
        self.de1 = nn.Sequential(
            nn.ConvTranspose2d(64*2, out_channels=64, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(64),
            nn.ELU())
        self.de2 = nn.Sequential(
            nn.ConvTranspose2d(64*2, 32, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(32),
            nn.ELU())
        self.de3 = nn.Sequential(
            nn.ConvTranspose2d(32*2, 32, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(32),
            nn.ELU())
        self.de4 = nn.Sequential(
            nn.ConvTranspose2d(32*2, 16, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(16),
            nn.ELU())
        self.de5 = nn.Sequential(
            nn.ConvTranspose2d(16*2, 16, kernel_size=(2, 5), stride=(1, 2)),
            self.chomp_t,
            nn.BatchNorm2d(16),
            nn.ELU())
        self.de6 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=(1,1), stride=(1, 1)),
            nn.Softplus())

    def forward(self, x, en_list):
        en_list[-1] = self.att1(x, en_list[-1])
        x = self.de1(torch.cat((x, en_list[-1]), dim=1))
        en_list[-2] = self.att2(x, en_list[-2])
        x = self.de2(torch.cat((x, en_list[-2]), dim=1))
        en_list[-3] = self.att3(x, en_list[-3])
        x = self.de3(torch.cat((x, en_list[-3]), dim=1))
        en_list[-4] = self.att4(x, en_list[-4])
        x = self.de4(torch.cat((x, en_list[-4]), dim=1))
        en_list[-5] = self.att5(x, en_list[-5])
        x = self.de5(torch.cat((x, en_list[-5]), dim=1))
        x = self.de6(x)
        return x


class Stage_GRU(nn.Module):
    def __init__(self):
        super(Stage_GRU, self).__init__()
        self.pad = nn.ConstantPad2d((2, 2, 1, 0), value=0.)
        self.pre_conv = nn.Sequential(
            self.pad,
            nn.Conv2d(2, 16, kernel_size=(2, 5), stride=(1, 1)),
            nn.ELU())
        self.conv_xz = nn.Sequential(
            self.pad,
            nn.Conv2d(16, 16, kernel_size=(2, 5)))
        self.conv_xr = nn.Sequential(
            self.pad,
            nn.Conv2d(16, 16, kernel_size=(2, 5)))
        self.conv_xn = nn.Sequential(
            self.pad,
            nn.Conv2d(16, 16, kernel_size=(2, 5)))
        self.conv_hz = nn.Sequential(
            self.pad,
            nn.Conv2d(16, 16, kernel_size=(2, 5)))
        self.conv_hr = nn.Sequential(
            self.pad,
            nn.Conv2d(16, 16, kernel_size=(2, 5)))
        self.conv_hn = nn.Sequential(
            self.pad,
            nn.Conv2d(16, 16, kernel_size=(2, 5)))

    def forward(self, x, h=None):
        x = self.pre_conv(x)
        if h is None:
            z = F.sigmoid(self.conv_xz(x))
            f = F.tanh(self.conv_xn(x))
            h = z * f
        else:
            z = F.sigmoid(self.conv_xz(x) + self.conv_hz(h))
            r = F.sigmoid(self.conv_xr(x) + self.conv_hr(h))
            n = F.tanh(self.conv_xn(x) + self.conv_hn(r * h))
            h = (1 - z) * h + z * n
        return h


class Attention_Block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_Block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x*psi

class GLU(nn.Module):
    def __init__(self, dilation, in_channel, causal_flag):
        super(GLU, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_channel, 64, kernel_size=1),
            nn.BatchNorm1d(64))
        if causal_flag is True:
            self.pad = nn.ConstantPad1d((int(dilation * 10), 0), value=0.)
        else:
            self.pad = nn.ConstantPad1d((int(dilation * 5), int(dilation * 5)), value=0.)

        self.left_conv = nn.Sequential(
            nn.ELU(),
            self.pad,
            nn.Conv1d(64, 64, kernel_size=11, dilation=dilation),
            nn.BatchNorm1d(64))
        self.right_conv = nn.Sequential(
            nn.ELU(),
            self.pad,
            nn.Conv1d(64, 64, kernel_size=11, dilation=dilation),
            nn.BatchNorm1d(num_features=64),
            nn.Sigmoid())
        self.out_conv = nn.Sequential(
            nn.Conv1d(64, 256, kernel_size=1),
            nn.BatchNorm1d(256))
        self.out_elu = nn.ELU()

    def forward(self, inpt):
        x = inpt
        x = self.in_conv(x)
        x1 = self.left_conv(x)
        x2 = self.right_conv(x)
        x = x1 * x2
        x = self.out_conv(x)
        x = x + inpt
        x = self.out_elu(x)
        return x

class up_Chomp_F(nn.Module):
    def __init__(self, chomp_f):
        super(up_Chomp_F, self).__init__()
        self.chomp_f = chomp_f
    def forward(self, x):
        return x[:, :, :, self.chomp_f:]

class down_Chomp_F(nn.Module):
    def __init__(self, chomp_f):
        super(down_Chomp_F, self).__init__()
        self.chomp_f = chomp_f
    def forward(self, x):
        return x[:, :, :, :-self.chomp_f]

class Chomp_T(nn.Module):
    def __init__(self, chomp_t):
        super(Chomp_T, self).__init__()
        self.chomp_t = chomp_t
    def forward(self, x):
        return x[:, :, :-self.chomp_t, :]