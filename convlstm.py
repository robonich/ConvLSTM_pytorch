import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, device):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias

        self.device      = device
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              # ここで 4 * self.hidden_dim にしているのは後にこれを4等分してcombined_convに分けるから
                              # 理解した
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        #print("combined", combined.requires_grad)
        
        # combined_conv の self.hidden_dim * 4 のサイズを持っている
        combined_conv = self.conv(combined)

        # ここの split で sigmoid や tanh に入力する前の畳み込んである状態の値にすることができた
        # cc は combined_conv
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        #print("cc_i", cc_i.requires_grad)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        # h_next, c_next の size は self.hidden_dim
        return h_next, c_next

    def init_hidden(self, batch_size):
        return (torch.zeros([batch_size, self.hidden_dim, self.height, self.width], requires_grad=True).to(self.device),
                torch.zeros([batch_size, self.hidden_dim, self.height, self.width], requires_grad=True).to(self.device))


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, linear_dim, device,
                 batch_first=False, bias=True, return_all_layers=False, use_all_t=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        # 最後の cell 以外のの output もすべて出すかどうか
        self.return_all_layers = return_all_layers

        # 最後の linear 層をどうするか
        self.linear_dim = linear_dim
        # すべての t　を利用するか
        self.use_all_t = use_all_t
        
        self.device = device

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          device=self.device))

        self.cell_list = nn.ModuleList(cell_list)
        
        linear_list = []
        
        for i in range(len(self.linear_dim)):
            _t = 5 # これは予め決めておかないと（固定値）
            if not self.use_all_t:
                _t = 1
            cur_input_dim = self.hidden_dim[-1] * _t * self.height * self.width if i == 0 else self.linear_dim[i-1]
            linear_list.append(nn.Linear(cur_input_dim, self.linear_dim[i]))
            
        self.linear_list = nn.ModuleList(linear_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        sigmoid で 0~1 の値にしたもの sig_valueを返す。
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        # seq_len は time
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            # まずは h, c を tensor でちゃんと初期化してあげる（backpropするために）
            h, c = hidden_state[layer_idx]
            #print("h", h.requires_grad)
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            # 時間方向でスタックさせている
            # ここで時間方向でスタックさせているのは何故？
            # この時間をすべてを使うかどうかは自分次第、 use_all_t というフラグをおいてみよう
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]
            
        if not self.use_all_t:
            # LSTM 最後の層の、最後の出力 h を次の Linear に入力する
            # x は (batch_size, last_hidden_dim, height, width)
            x = last_state_list[-1][0].view(-1, self.hidden_dim[-1] * self.height * self.width)
        
        # ここから linear 層
        for layer_idx in range(len(self.linear_list)):
            if layer_idx == len(self.linear_list)-1:
                break
            x = F.relu(self.linear_list[layer_idx](x))
            
        # softmax に変更
        x = F.softmax(self.linear_list[-1](x))

        return layer_output_list, last_state_list, x

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
