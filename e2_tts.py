from __future__ import annotations
from pathlib import Path
from random import random
from functools import partial
from itertools import zip_longest
from collections import namedtuple
from typing import Literal, Callable
import jaxtyping
from beartype import beartype

import torch
import torch.nn.functional as F
from torch import nn, tensor, Tensor, from_numpy
from torch.nn import Module, ModuleList, Sequential, Linear
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from torchaudio.functional import DB_to_amplitude
from torchdiffeq import odeint

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, einsum, pack, unpack

from x_transformers import Attention, FeedForward, RMSNorm, AdaptiveRMSNorm
from x_transformers.x_transformers import RotaryEmbedding

from vocos import Vocos

from g2p_en import G2p


# 使用 partial 函数创建一个带默认参数的 pad_sequence 函数
"""
pad_sequence 函数用于将一批不等长的张量填充为相同长度。
这里使用 functools.partial 为 pad_sequence 函数设置默认参数 batch_first=True，
表示返回的张量形状的第一个维度是批量大小。
"""
pad_sequence = partial(pad_sequence, batch_first = True)


class TorchTyping:
    """
    TorchTyping 类用于创建与 PyTorch 张量兼容的类型提示。
    通过指定抽象的数据类型（如 Float、Int、Bool）和形状，可以生成具体的类型提示。

    参数说明:
        abstract_dtype: 抽象的数据类型，如 jaxtyping.Float、jaxtyping.Int、jaxtyping.Bool。
    """
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        """
        根据给定的形状字符串，返回一个类型提示函数。

        参数:
            shapes (str): 形状字符串，例如 "batch 64 64"。

        返回:
            一个类型提示函数，接受一个值并返回带有指定形状和抽象数据类型的类型提示。
        """
        return self.abstract_dtype[Tensor, shapes]


# 定义常用的类型提示
Float = TorchTyping(jaxtyping.Float)
Int = TorchTyping(jaxtyping.Int)
Bool = TorchTyping(jaxtyping.Bool)


# 定义一个命名元组，用于表示损失分解
"""
LossBreakdown 是一个命名元组，用于表示损失分解。
包含两个字段：
    - flow: 流损失。
    - velocity_consistency: 速度一致性损失。
"""
LossBreakdown = namedtuple('LossBreakdown', ['flow', 'velocity_consistency'])


# 定义一个命名元组，用于表示 E2TTS 模型的返回结果
"""
E2TTSReturn 是一个命名元组，用于表示 E2TTS 模型的返回结果。
包含五个字段：
    - loss: 总损失。
    - cond: 条件输入。
    - pred_flow: 预测的流。
    - pred_data: 预测的数据。
    - loss_breakdown: 损失分解。
"""
E2TTSReturn = namedtuple('E2TTS', ['loss', 'cond', 'pred_flow', 'pred_data', 'loss_breakdown'])


def exists(v):
    """
    检查一个值是否存在（即不为 None）。

    参数:
        v: 输入值。

    返回:
        如果 v 不为 None，则返回 True；否则返回 False。
    """
    return v is not None


def default(v, d):
    """
    如果值 v 存在（即不为 None），则返回 v；否则返回默认值 d。

    参数:
        v: 输入值。
        d: 默认值。

    返回:
        如果 v 存在，则返回 v；否则返回 d。
    """
    return v if exists(v) else d


def l2norm(t):
    """
    对输入张量 t 进行 L2 归一化。

    参数:
        t (Tensor): 输入张量。

    返回:
        Tensor: 归一化后的张量。
    """
    return F.normalize(t, dim = -1)


def divisible_by(num, den):
    """
    判断 num 是否能被 den 整除。

    参数:
        num (int): 被除数。
        den (int): 除数。

    返回:
        bool: 如果 num 能被 den 整除，则返回 True；否则返回 False。
    """
    return (num % den) == 0


def pack_one_with_inverse(x, pattern):
    """
    将单个张量 x 按照指定的模式进行打包，并返回一个逆函数用于解包。

    参数:
        x (Tensor): 输入张量。
        pattern (str): 打包模式，例如 'b *' 表示批量大小和其他维度。

    返回:
        Tuple[Tensor, Callable[[Tensor, Optional[str]], Tensor]]: 返回一个元组，包含打包后的张量和逆函数。
            逆函数接受一个打包后的张量和可选的逆模式，返回解包后的张量。
    """
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern = None):
        """
        逆函数，用于解包打包后的张量。

        参数:
            x (Tensor): 打包后的张量。
            inverse_pattern (Optional[str], 可选): 逆模式，如果未指定，则使用默认的打包模式。

        返回:
            Tensor: 解包后的张量。
        """
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse


class Identity(Module):
    """
    Identity 类实现了一个恒等映射模块。
    该模块在输入数据上不做任何变换，直接返回输入数据。

    参数:
        None
    """
    def forward(self, x, **kwargs):
        """
        前向传播方法，执行恒等映射。

        参数:
            x (Tensor): 输入张量。

        返回:
            Tensor: 输出张量，与输入相同。
        """
        return x


def project(x, y):
    """
    将张量 x 投影到张量 y 上，并返回平行和正交分量。

    参数:
        x (Tensor): 输入张量。
        y (Tensor): 目标张量，用于投影。

    返回:
        Tuple[Tensor, Tensor]: 返回一个元组，包含平行分量和正交分量。
    """
    # 使用 pack_one_with_inverse 函数对 x 和 y 进行打包，并获取逆函数
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')
    
    # 将数据类型转换为 double 以提高数值精度
    dtype = x.dtype
    x, y = x.double(), y.double()

    # 对 y 进行 L2 归一化，得到单位向量
    unit = F.normalize(y, dim = -1)

    # 计算平行分量
    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    # 计算正交分量
    orthogonal = x - parallel

    # 使用逆函数将平行和正交分量转换回原始数据类型
    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)


# simple utf-8 tokenizer, since paper went character based
# 基于 UTF-8 的字符级分词器

def list_str_to_tensor(text: list[str], padding_value = -1):
    """
    将字符串列表转换为张量，并对序列进行填充。

    参数:
        text (List[str]): 输入的文本列表。
        padding_value (int, 可选): 填充值，默认为 -1。

    返回:
        Int['b nt']: 填充后的张量，形状为 (batch_size, sequence_length)。
    """
    # 将每个字符串转换为 UTF-8 字节，并转换为对应的整数张量
    list_tensors = [tensor([*bytes(t, 'UTF-8')]) for t in text]
    # 对张量列表进行填充，使其长度一致
    padded_tensor = pad_sequence(list_tensors, padding_value = -1)
    return padded_tensor


# simple english phoneme-based tokenizer
# 基于英文音素的分词器

def get_g2p_en_encode():
    """
    获取一个基于英文音素的编码函数，并返回音素总数。

    返回:
        Tuple[Callable[[List[str], int], Int['b nt']], int]: 返回一个元组，包含编码函数和音素总数。
    """
    # 初始化 G2p 对象，用于将英文文本转换为音素
    g2p = G2p()
    # 获取音素到索引的映射字典
    phoneme_to_index = g2p.p2idx
    # 计算音素总数
    num_phonemes = len(phoneme_to_index)
    # 定义需要扩展的特殊字符及其对应的索引
    extended_chars = [' ', ',', '.', '-', '!', '?', '\'', '"', '...', '..', '. .', '. . .', '. . . .', '. . . . .', '. ...', '... .', '.. ..']
    num_extended_chars = len(extended_chars)

    # 为每个扩展字符分配一个唯一的索引
    extended_chars_dict = {p: (num_phonemes + i) for i, p in enumerate(extended_chars)}
    # 合并音素和扩展字符的映射字典
    phoneme_to_index = {**phoneme_to_index, **extended_chars_dict}

    def encode(text: list[str], padding_value = -1):
        """
        将英文文本列表转换为音素索引张量，并对序列进行填充。

        参数:
            text (List[str]): 输入的英文文本列表。
            padding_value (int, 可选): 填充值，默认为 -1。

        返回:
            Int['b nt']: 填充后的音素索引张量，形状为 (batch_size, sequence_length)。
        """
        # 将每个文本字符串转换为音素列表
        phonemes = [g2p(t) for t in text]
        # 将音素列表转换为对应的索引张量
        list_tensors = [tensor([phoneme_to_index[p] for p in one_phoneme]) for one_phoneme in phonemes]
        # 对张量列表进行填充，使其长度一致
        padded_tensor = pad_sequence(list_tensors, padding_value = -1)
        return padded_tensor

    return encode, (num_phonemes + num_extended_chars)


def log(t, eps = 1e-5):
    """
    对输入张量 t 取对数，并添加一个小的常数以防止数值下溢。

    参数:
        t (Tensor): 输入张量。
        eps (float, 可选): 防止数值下溢的小常数，默认为 1e-5。

    返回:
        Tensor: 对数变换后的张量。
    """
    return t.clamp(min = eps).log()


def lens_to_mask(t, length):
    """
    将长度张量转换为掩码张量。

    参数:
        t (Int['b']): 输入的长度张量，形状为 (batch_size,)。
        length (int, 可选): 可选的最大长度。如果未指定，则使用 t 中的最大值。

    返回:
        Bool['b n']: 掩码张量，形状为 (batch_size, sequence_length)。
    """
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device = t.device)
    return einx.less('n, b -> b n', seq, t)


def mask_from_start_end_indices(seq_len, start, end):
    """
    根据开始和结束索引生成掩码张量。

    参数:
        seq_len (Tensor): 序列长度张量，形状为 (batch_size,)。
        start (Tensor): 开始索引张量，形状为 (batch_size,)。
        end (Tensor): 结束索引张量，形状为 (batch_size,)。

    返回:
        Tensor: 生成的掩码张量，形状为 (batch_size, max_seq_len)。
    """
    # 获取批处理中最大的序列长度
    max_seq_len = seq_len.max().item()  
    # 生成一个从0到max_seq_len-1的序列张量，形状为 (max_seq_len,)
    seq = torch.arange(max_seq_len, device = start.device).long()
    # 生成掩码：seq >= start 并且 seq < end
    return einx.greater_equal('n, b -> b n', seq, start) & einx.less('n, b -> b n', seq, end)


def mask_from_frac_lengths(seq_len, frac_lengths ,max_length):
    """
    根据分数长度生成掩码张量。

    参数:
        seq_len (Tensor): 序列长度张量，形状为 (batch_size,)。
        frac_lengths (Tensor): 分数长度张量，形状为 (batch_size,)。
        max_length (int, 可选): 可选的最大长度。

    返回:
        Tensor: 生成的掩码张量，形状为 (batch_size, max_seq_len)。
    """
    # 计算实际长度：分数长度 * 序列长度
    lengths = (frac_lengths * seq_len).long()
    # 计算最大可能的开始位置：序列长度 - 实际长度
    max_start = seq_len - lengths

    # 生成随机数张量，形状与 frac_lengths 相同
    rand = torch.rand_like(frac_lengths)
    # 生成开始位置：max_start * rand，并限制最小值为0
    start = (max_start * rand).long().clamp(min = 0)
    # 生成结束位置：开始位置 + 实际长度
    end = start + lengths

    # 根据开始和结束位置生成掩码
    out = mask_from_start_end_indices(seq_len, start, end)

    # 如果指定了最大长度，则对掩码进行填充
    if exists(max_length):
        out = pad_to_length(out, max_length)

    return out


def maybe_masked_mean(t, mask):
    """
    对输入张量进行掩码平均。

    参数:
        t (Tensor): 输入张量，形状为 (batch_size, sequence_length, dim)。
        mask (Tensor, 可选): 可选的掩码张量，形状为 (batch_size, sequence_length)。

    返回:
        Tensor: 掩码平均后的张量，形状为 (batch_size, dim)。
    """
    if not exists(mask):
        return t.mean(dim = 1)
    # 使用掩码对输入张量进行掩码处理，将掩码为0的位置设为0
    t = einx.where('b n, b n d, -> b n d', mask, t, 0.)
    # 对掩码后的张量进行求和
    num = reduce(t, 'b n d -> b d', 'sum')
    # 对掩码进行求和，得到非零元素的数量
    den = reduce(mask.float(), 'b n -> b', 'sum')
    # 计算平均：求和 / (非零元素数量 + 1e-5)，以防止除以零
    return einx.divide('b d, b -> b d', num, den.clamp(min = 1.))


def pad_to_length(
    t: Tensor,
    length: int,
    value = None
):
    """
    对输入张量进行填充，使其长度达到指定值。

    参数:
        t (Tensor): 输入张量。
        length (int): 目标长度。
        value (float, 可选): 填充值。

    返回:
        Tensor: 填充后的张量。
    """
    seq_len = t.shape[-1]
    if length > seq_len:
        # 如果目标长度大于当前长度，则进行填充
        t = F.pad(t, (0, length - seq_len), value = value)

    return t[..., :length]


def interpolate_1d(
    x: Tensor,
    length: int,
    mode = 'bilinear'
):
    """
    对输入张量进行1D插值，使其长度达到指定值。

    参数:
        x (Tensor): 输入张量，形状为 (n, d)。
        length (int): 目标长度。
        mode (str, 可选): 插值模式，默认为 'bilinear'。

    返回:
        Tensor: 插值后的张量，形状为 (n, d)。
    """
    # 重塑张量形状以适应插值函数
    x = rearrange(x, 'n d -> 1 d n 1')
    # 进行插值操作
    x = F.interpolate(x, (length, 1), mode = mode)
    # 重塑回原始形状
    return rearrange(x, '1 d n 1 -> n d')


class MelSpec(Module):
    """
    MelSpec 类实现了一个梅尔频谱（Mel-spectrogram）转换模块。
    该模块将输入的音频信号转换为梅尔频谱表示，常用于音频处理和音乐信息检索任务。

    参数说明:
        filter_length (int, 可选): FFT 的窗口长度，默认为1024。
        hop_length (int, 可选): 帧移长度，默认为256。
        win_length (int, 可选): 窗口长度，默认为1024。
        n_mel_channels (int, 可选): 梅尔频带的数量，默认为100。
        sampling_rate (int, 可选): 采样率，默认为24,000 Hz。
        normalize (bool, 可选): 是否对梅尔频谱进行归一化，默认为 False。
        power (float, 可选): 功率，默认为1。
        norm (str, 可选): 归一化方法，默认为 None。
        center (bool, 可选): 是否对信号进行中心化，默认为 True。
    """
    def __init__(
        self,
        filter_length = 1024,
        hop_length = 256,
        win_length = 1024,
        n_mel_channels = 100,
        sampling_rate = 24_000,
        normalize = False,
        power = 1,
        norm = None,
        center = True,
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate

        # 定义梅尔频谱转换模块
        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,        # 采样率
            n_fft=filter_length,              # FFT 的窗口长度
            win_length=win_length,            # 窗口长度
            hop_length=hop_length,            # 帧移长度
            n_mels=n_mel_channels,            # 梅尔频带的数量
            power=power,                      # 功率
            center=center,                    # 是否对信号进行中心化
            normalized=normalize,             # 是否对梅尔频谱进行归一化
            norm=norm                         # 归一化方法
        )

        self.register_buffer('dummy', tensor(0), persistent = False)

    def forward(self, inp):
        """
        前向传播方法，执行梅尔频谱转换。

        参数:
            inp (Tensor): 输入音频信号，形状为 (batch_size, channels, samples)。

        返回:
            Tensor: 梅尔频谱，形状为 (batch_size, n_mel_channels, time_steps)。
        """
        if len(inp.shape) == 3:
            # 如果输入张量的维度为3，则重塑为 (batch_size, samples)
            inp = rearrange(inp, 'b 1 nw -> b nw')

        assert len(inp.shape) == 2

        if self.dummy.device != inp.device:
            self.to(inp.device)

        # 计算梅尔频谱
        mel = self.mel_stft(inp)
        # 对梅尔频谱取对数
        mel = log(mel)
        return mel


class DepthwiseConv(Module):
    """
    DepthwiseConv 类实现了一个深度可分离卷积（Depthwise Separable Convolution）模块。
    该模块首先对每个通道进行独立的卷积操作，然后进行逐点卷积以混合通道信息。
    这里只实现了深度卷积部分（即每个通道的独立卷积）。

    参数说明:
        dim (int): 输入和输出的通道数。
        kernel_size (int): 卷积核的大小。
        groups (int, 可选): 分组卷积的组数，默认为 dim（实现深度卷积）。
    """
    def __init__(
        self,
        dim,
        *,
        kernel_size,
        groups = None
    ):
        super().__init__()
        assert not divisible_by(kernel_size, 2)
        # 默认情况下，实现完整的深度卷积
        groups = default(groups, dim) # full depthwise conv by default

        # 定义深度卷积层
        self.dw_conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups = groups, padding = kernel_size // 2), # 深度卷积
            nn.SiLU() # 使用 SiLU 作为激活函数
        )

    def forward(
        self,
        x,
        mask = None
    ):
        """
        前向传播方法，执行深度卷积。

        参数:
            x (Tensor): 输入张量。
            mask (Tensor, 可选): 可选的掩码张量，用于掩码输入。

        返回:
            Tensor: 输出张量，形状与输入相同。
        """
        if exists(mask):
            # 如果提供了掩码，则根据掩码对输入进行掩码处理
            x = einx.where('b n, b n d, -> b n d', mask, x, 0.)
        # 重塑张量形状以适应卷积层
        x = rearrange(x, 'b n c -> b c n')
        # 应用深度卷积
        x = self.dw_conv1d(x)
        # 重塑回原始形状
        out = rearrange(x, 'b c n -> b n c')

        if exists(mask):
            # 如果提供了掩码，则根据掩码对输出进行掩码处理
            out = einx.where('b n, b n d, -> b n d', mask, out, 0.)

        return out


# adaln zero from DiT paper

class AdaLNZero(Module):
    """
    AdaLN-Zero 类实现了一个自适应层归一化（Adaptive Layer Normalization）模块。
    该模块根据条件输入动态调整归一化参数（gamma），并将其应用于输入张量。
    初始时，gamma 的权重被初始化为零，偏差被初始化为一个固定值（默认为 -2）。

    参数说明:
        dim (int): 输入和输出的特征维度。
        dim_condition (int, 可选): 条件输入的特征维度。如果未指定，则默认为 dim。
        init_bias_value (float, 可选): gamma 偏差的初始值，默认为 -2。
    """
    def __init__(
        self,
        dim,
        dim_condition = None,
        init_bias_value = -2.
    ):
        super().__init__()
        # 如果未指定 dim_condition，则默认为 dim
        dim_condition = default(dim_condition, dim)
        # 定义一个线性层，将条件输入映射到 gamma
        self.to_gamma = nn.Linear(dim_condition, dim)

        # 初始化 gamma 的权重为零
        nn.init.zeros_(self.to_gamma.weight)
        # 初始化 gamma 的偏差为 init_bias_value
        nn.init.constant_(self.to_gamma.bias, init_bias_value)

    def forward(self, x, *, condition):
        """
        前向传播方法，执行自适应层归一化。

        参数:
            x (Tensor): 输入张量。
            condition (Tensor): 条件输入张量。

        返回:
            Tensor: 应用自适应层归一化后的输出张量。
        """
        if condition.ndim == 2:
            # 如果条件输入的维度为2，则重塑为 (batch_size, 1, dim_condition)
            condition = rearrange(condition, 'b d -> b 1 d')

        # 将条件输入映射到 gamma，并应用 sigmoid 函数进行归一化
        gamma = self.to_gamma(condition).sigmoid()
        # 将输入张量乘以 gamma，实现自适应层归一化
        return x * gamma


# random projection fourier embedding

class RandomFourierEmbed(Module):
    """
    RandomFourierEmbed 类实现了一个随机投影傅里叶嵌入模块。
    该模块通过随机投影将输入映射到傅里叶域，并生成傅里叶嵌入表示。

    参数说明:
        dim (int): 输入和输出的特征维度。
    """
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        # 初始化傅里叶嵌入的权重，形状为 (dim // 2,)
        self.register_buffer('weights', torch.randn(dim // 2))

    def forward(self, x):
        """
        前向传播方法，执行随机投影傅里叶嵌入。

        参数:
            x (Tensor): 输入张量，形状为 (batch_size, dim)。

        返回:
            Tensor: 傅里叶嵌入后的张量，形状为 (batch_size, dim * 3)。
        """
        # 计算傅里叶频率：输入张量乘以权重，并乘以 2π
        freqs = einx.multiply('i, j -> i j', x, self.weights) * 2 * torch.pi
        # 打包输入张量和傅里叶嵌入（正弦和余弦），形状为 (batch_size, dim * 3)
        fourier_embed, _ = pack((x, freqs.sin(), freqs.cos()), 'b *')
        return fourier_embed


# character embedding

class CharacterEmbed(Module):
    """
    CharacterEmbed 类实现了一个字符嵌入模块。
    该模块将输入的字符序列转换为嵌入表示，适用于文本处理任务。

    参数说明:
        dim (int): 嵌入的维度。
        num_embeds (int, 可选): 字符嵌入的数量，默认为256。
    """
    def __init__(
        self,
        dim,
        num_embeds = 256,
    ):
        super().__init__()
        self.dim = dim
        # 定义字符嵌入层，使用 num_embeds + 1 个嵌入，因为索引从0开始
        self.embed = nn.Embedding(num_embeds + 1, dim) # 将0作为填充token

    def forward(
        self,
        text,
        max_seq_len: int,
        **kwargs
    ):
        """
        前向传播方法，执行字符嵌入。

        参数:
            text (Tensor): 输入的字符序列。
            max_seq_len (int): 最大序列长度。
            **kwargs: 其他关键字参数。

        返回:
            Tensor: 嵌入后的字符序列，形状为 (batch_size, max_seq_len, dim)。
        """
        # 将所有字符ID加1，将0作为填充token
        text = text + 1 # shift all other token ids up by 1 and use 0 as filler token

        # 如果字符token数量超过梅尔频谱token数量，则截断
        text = text[:, :max_seq_len] # just curtail if character tokens are more than the mel spec tokens, one of the edge cases the paper did not address
        # 使用填充值0进行填充，以确保序列长度达到 max_seq_len
        text = pad_to_length(text, max_seq_len, value = 0)
        # 返回嵌入后的字符序列
        return self.embed(text)


class InterpolatedCharacterEmbed(Module):
    """
    InterpolatedCharacterEmbed 类实现了一个插值字符嵌入模块。
    该模块首先对字符序列进行嵌入，然后根据音频序列的长度对字符嵌入进行插值。
    最后，通过一个多层感知机（MLP）处理绝对位置信息，生成隐式位置嵌入。

    参数说明:
        dim (int): 嵌入的维度。
        num_embeds (int, 可选): 字符嵌入的数量，默认为256。
    """
    def __init__(
        self,
        dim,
        num_embeds = 256,
    ):
        super().__init__()
        self.dim = dim
        # 定义字符嵌入层
        self.embed = nn.Embedding(num_embeds, dim)

        # 定义绝对位置的多层感知机（MLP）
        self.abs_pos_mlp = Sequential(
            Rearrange('... -> ... 1'),  # 重塑张量形状
            Linear(1, dim),  # 线性层，将1维输入映射到dim维
            nn.SiLU(),  # 使用 SiLU 作为激活函数
            Linear(dim, dim)  # 线性层，将dim维输入映射回dim维
        )

    def forward(
        self,
        text,
        max_seq_len: int,
        mask
    ):
        """
        前向传播方法，执行插值字符嵌入。

        参数:
            text (Tensor): 输入的字符序列。
            max_seq_len (int): 最大序列长度。
            mask (Tensor, 可选): 可选的掩码张量，用于掩码输入。

        返回:
            Tensor: 插值后的字符嵌入，形状为 (batch_size, max_seq_len, dim)。
        """
        device = text.device  # 获取输入张量的设备
        # 如果未提供掩码，则使用默认的掩码
        mask = default(mask, (None,))

        # 用于存储插值后的嵌入
        interp_embeds = []
        # 用于存储插值后的绝对位置
        interp_abs_positions = []

        # 遍历每个文本序列和对应的掩码
        for one_text, one_mask in zip_longest(text, mask):
            
            # 筛选出有效的字符（字符ID >= 0）
            valid_text = one_text >= 0
            one_text = one_text[valid_text]
            # 对字符序列进行嵌入
            one_text_embed = self.embed(one_text)

            # save the absolute positions
            # 保存绝对位置信息

            # 获取字符序列的长度
            text_seq_len = one_text.shape[0]

            # determine audio sequence length from mask
            # 根据掩码确定音频序列的长度
            audio_seq_len = max_seq_len
            if exists(one_mask):
                audio_seq_len = one_mask.sum().long().item()

            # interpolate text embedding to audio embedding length
            # 对字符嵌入进行插值，使其长度与音频序列长度一致
            interp_text_embed = interpolate_1d(one_text_embed, audio_seq_len)
            # 生成插值后的绝对位置
            interp_abs_pos = torch.linspace(0, text_seq_len, audio_seq_len, device = device)

            # 添加插值后的嵌入
            interp_embeds.append(interp_text_embed)
            # 添加插值后的绝对位置
            interp_abs_positions.append(interp_abs_pos)

        # 对嵌入和绝对位置进行填充，使其长度一致
        interp_embeds = pad_sequence(interp_embeds)
        interp_abs_positions = pad_sequence(interp_abs_positions)

        # 进一步填充嵌入，使其达到最大序列长度
        interp_embeds = F.pad(interp_embeds, (0, 0, 0, max_seq_len - interp_embeds.shape[-2]))
        # 对绝对位置进行填充，使其达到最大序列长度
        interp_abs_positions = pad_to_length(interp_abs_positions, max_seq_len)

        # pass interp absolute positions through mlp for implicit positions
        # 将插值后的绝对位置通过 MLP，生成隐式位置嵌入
        interp_embeds = interp_embeds + self.abs_pos_mlp(interp_abs_positions)
        # 如果提供了掩码，则根据掩码对嵌入进行掩码处理
        if exists(mask):
            interp_embeds = einx.where('b n, b n d, -> b n d', mask, interp_embeds, 0.)

        return interp_embeds


# text audio cross conditioning in multistream setup
# 文本与音频交叉条件模块，用于多流设置中的文本音频交叉条件

class TextAudioCrossCondition(Module):
    """
    TextAudioCrossCondition 类实现了一个文本与音频交叉条件模块。
    该模块用于多流设置中，通过线性变换将音频和文本特征进行交叉条件化。
    它支持从音频到文本的条件传递，也可以选择仅从文本到音频的条件传递。

    参数说明:
        dim (int): 音频特征的维度。
        dim_text (int): 文本特征的维度。
        cond_audio_to_text (bool, 可选): 是否启用从音频到文本的条件传递，默认为 True。
    """
    def __init__(
        self,
        dim,
        dim_text,
        cond_audio_to_text = True,
    ):
        super().__init__()
        # 定义从文本到音频的条件线性变换
        self.text_to_audio = nn.Linear(dim_text + dim, dim, bias = False)
        # 将文本到音频线性变换的权重初始化为零
        nn.init.zeros_(self.text_to_audio.weight)

        # 是否启用从音频到文本的条件传递
        self.cond_audio_to_text = cond_audio_to_text

        if cond_audio_to_text:
            # 如果启用从音频到文本的条件传递，则定义从音频到文本的条件线性变换
            self.audio_to_text = nn.Linear(dim + dim_text, dim_text, bias = False)
            # 将音频到文本线性变换的权重初始化为零
            nn.init.zeros_(self.audio_to_text.weight)

    def forward(
        self,
        audio,
        text
    ):
        """
        前向传播方法，执行文本与音频交叉条件化。

        参数:
            audio (Tensor): 输入的音频特征。
            text (Tensor): 输入的文本特征。

        返回:
            Tuple[Tensor, Tensor]: 返回一个元组，包含条件化后的音频和文本特征。
        """
        # 将音频和文本特征打包在一起，形状为 (batch_size, n, dim + dim_text)
        audio_text, _ = pack((audio, text), 'b n *')
        # 计算文本到音频的条件输出，形状为 (batch_size, n, dim)
        text_cond = self.text_to_audio(audio_text)
        # 如果启用从音频到文本的条件传递，则计算音频到文本的条件输出，形状为 (batch_size, n, dim_text)
        audio_cond = self.audio_to_text(audio_text) if self.cond_audio_to_text else 0.
        # 将条件输出与原始音频和文本特征相加，得到条件化后的音频和文本特征
        return audio + text_cond, text + audio_cond


# attention and transformer backbone
# for use in both e2tts as well as duration module

class Transformer(Module):
    """
    Transformer 类实现了一个多流 Transformer 模型，支持音频和文本的交叉条件化。
    该模型包含多个 Transformer 层，每个层包含音频相关的模块和可选的文本相关的模块。
    支持绝对位置嵌入、时间条件化以及各种自定义参数，如注意力头数、隐藏层维度等。

    参数说明:
        dim (int): 音频特征的维度。
        dim_text (int, 可选): 文本特征的维度。如果未指定，则默认为音频维度的一半。
        depth (int, 可选): Transformer 层的总数，默认为8。
        heads (int, 可选): 音频注意力头的数量，默认为8。
        dim_head (int, 可选): 每个音频注意力头的维度，默认为64。
        ff_mult (int, 可选): 前馈神经网络中隐藏层的维度乘数，默认为4。
        text_depth (int, 可选): 文本条件化的层数。如果未指定，则默认为总层数。
        text_heads (int, 可选): 文本注意力头的数量。如果未指定，则默认为音频注意力头的数量。
        text_dim_head (int, 可选): 每个文本注意力头的维度。如果未指定，则默认为音频注意力头的维度。
        text_ff_mult (int, 可选): 文本前馈神经网络中隐藏层的维度乘数。如果未指定，则默认为音频前馈神经网络的乘数。
        cond_on_time (bool, 可选): 是否启用时间条件化，默认为True。
        abs_pos_emb (bool, 可选): 是否使用绝对位置嵌入，默认为True。
        max_seq_len (int, 可选): 最大序列长度，默认为8192。
        kernel_size (int, 可选): 卷积核的大小，默认为31。
        dropout (float, 可选): Dropout 层的失活概率，默认为0.1。
        num_registers (int, 可选): 注册器的数量，默认为32。
        attn_laser (bool, 可选): 是否使用激光注意力机制，默认为False。
        attn_laser_softclamp_value (float, 可选): 激光注意力机制的软限制值，默认为15.0。
        attn_kwargs (dict, 可选): 注意力机制的其他关键字参数，默认为空字典。
        ff_kwargs (dict, 可选): 前馈神经网络的其他关键字参数，默认为空字典。
    """
    @beartype
    def __init__(
        self,
        *,
        dim,
        dim_text = None, # will default to half of audio dimension
        depth = 8,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        text_depth = None,
        text_heads = None,
        text_dim_head = None,
        text_ff_mult = None,
        cond_on_time = True,
        abs_pos_emb = True,
        max_seq_len = 8192,
        kernel_size = 31,
        dropout = 0.1,
        num_registers = 32,
        attn_laser = False,
        attn_laser_softclamp_value = 15.,
        attn_kwargs: dict = dict(
            gate_value_heads = True,
            softclamp_logits = True,
        ),
        ff_kwargs: dict = dict(),
    ):
        super().__init__()
        assert divisible_by(depth, 2), 'depth needs to be even'

        # absolute positional embedding
        # 绝对位置嵌入

        self.max_seq_len = max_seq_len
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if abs_pos_emb else None

        self.dim = dim

        dim_text = default(dim_text, dim // 2)
        self.dim_text = dim_text

        text_heads = default(text_heads, heads)
        text_dim_head = default(text_dim_head, dim_head)
        text_ff_mult = default(text_ff_mult, ff_mult)
        text_depth = default(text_depth, depth)

        assert 1 <= text_depth <= depth, 'must have at least 1 layer of text conditioning, but less than total number of speech layers'

        self.depth = depth
        self.layers = ModuleList([])

        # registers
        # 寄存器

        self.num_registers = num_registers
        self.registers = nn.Parameter(torch.zeros(num_registers, dim))
        nn.init.normal_(self.registers, std = 0.02)

        self.text_registers = nn.Parameter(torch.zeros(num_registers, dim_text))
        nn.init.normal_(self.text_registers, std = 0.02)

        # rotary embedding
        # 旋转嵌入

        self.rotary_emb = RotaryEmbedding(dim_head)
        self.text_rotary_emb = RotaryEmbedding(dim_head)

        # time conditioning
        # will use adaptive rmsnorm
        # 时间条件化
        # 将使用自适应 RMSNorm

        self.cond_on_time = cond_on_time
        rmsnorm_klass = RMSNorm if not cond_on_time else AdaptiveRMSNorm
        postbranch_klass = Identity if not cond_on_time else partial(AdaLNZero, dim = dim)

        self.time_cond_mlp = Identity()

        if cond_on_time:
            self.time_cond_mlp = Sequential(
                RandomFourierEmbed(dim),
                Linear(dim + 1, dim),
                nn.SiLU()
            )

        for ind in range(depth):
            is_first_block = ind == 0

            is_later_half = ind >= (depth // 2)
            has_text = ind < text_depth

            # speech related
            # 与语音相关的模块

            speech_conv = DepthwiseConv(dim, kernel_size = kernel_size)

            attn_norm = rmsnorm_klass(dim)
            attn = Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = dropout, learned_value_residual_mix = not is_first_block, laser = attn_laser, laser_softclamp_value = attn_laser_softclamp_value, **attn_kwargs)
            attn_adaln_zero = postbranch_klass()

            ff_norm = rmsnorm_klass(dim)
            ff = FeedForward(dim = dim, glu = True, mult = ff_mult, dropout = dropout, **ff_kwargs)
            ff_adaln_zero = postbranch_klass()

            skip_proj = Linear(dim * 2, dim, bias = False) if is_later_half else None

            speech_modules = ModuleList([
                skip_proj,
                speech_conv,
                attn_norm,
                attn,
                attn_adaln_zero,
                ff_norm,
                ff,
                ff_adaln_zero,
            ])

            text_modules = None

            if has_text:
                # text related
                # 与文本相关的模块

                text_conv = DepthwiseConv(dim_text, kernel_size = kernel_size)

                text_attn_norm = RMSNorm(dim_text)
                text_attn = Attention(dim = dim_text, heads = text_heads, dim_head = text_dim_head, dropout = dropout, learned_value_residual_mix = not is_first_block, laser = attn_laser, laser_softclamp_value = attn_laser_softclamp_value, **attn_kwargs)

                text_ff_norm = RMSNorm(dim_text)
                text_ff = FeedForward(dim = dim_text, glu = True, mult = text_ff_mult, dropout = dropout, **ff_kwargs)

                # cross condition
                # 交叉条件化

                is_last = ind == (text_depth - 1)

                cross_condition = TextAudioCrossCondition(dim = dim, dim_text = dim_text, cond_audio_to_text = not is_last)

                text_modules = ModuleList([
                    text_conv,
                    text_attn_norm,
                    text_attn,
                    text_ff_norm,
                    text_ff,
                    cross_condition
                ])

            self.layers.append(ModuleList([
                speech_modules,
                text_modules
            ]))

        self.final_norm = RMSNorm(dim)

    def forward(
        self,
        x,
        times,
        mask,
        text_embed
    ):
        """
        前向传播方法，执行多流 Transformer 模型的前向计算。

        参数:
            x (Tensor): 输入的音频特征。
            times (Tensor, 可选): 时间条件输入。
            mask (Tensor, 可选): 输入的掩码。
            text_embed (Tensor, 可选): 输入的文本嵌入。

        返回:
            Tensor: 输出张量，形状与输入相同。
        """
        # 获取批次大小、序列长度和设备
        batch, seq_len, device = *x.shape[:2], x.device

        assert not (exists(times) ^ self.cond_on_time), '`times` must be passed in if `cond_on_time` is set to `True` and vice versa'

        # handle absolute positions if needed
        # 处理绝对位置嵌入（如果需要）

        if exists(self.abs_pos_emb):
            assert seq_len <= self.max_seq_len, f'{seq_len} exceeds the set `max_seq_len` ({self.max_seq_len}) on Transformer'
            seq = torch.arange(seq_len, device = device)
            x = x + self.abs_pos_emb(seq)

        # handle adaptive rmsnorm kwargs
        # 处理自适应 RMSNorm 的关键字参数

        norm_kwargs = dict()

        if exists(times):
            if times.ndim == 0:
                times = repeat(times, ' -> b', b = batch)

            times = self.time_cond_mlp(times)
            norm_kwargs.update(condition = times)

        # register tokens
        # 注册 tokens

        registers = repeat(self.registers, 'r d -> b r d', b = batch)
        x, registers_packed_shape = pack((registers, x), 'b * d')

        if exists(mask):
            mask = F.pad(mask, (self.num_registers, 0), value = True)

        # rotary embedding
        # 旋转嵌入

        rotary_pos_emb = self.rotary_emb.forward_from_seq_len(x.shape[-2])

        # text related
        # 与文本相关的处理

        if exists(text_embed):
            text_rotary_pos_emb = self.text_rotary_emb.forward_from_seq_len(x.shape[-2])

            text_registers = repeat(self.text_registers, 'r d -> b r d', b = batch)
            text_embed, _ = pack((text_registers, text_embed), 'b * d')

        # skip connection related stuff
        # 处理跳跃连接相关的内容

        skips = []

        # value residual
        # 价值残差

        text_attn_first_values = None
        attn_first_values = None

        # go through the layers
        # 遍历每一层

        for ind, (speech_modules, text_modules) in enumerate(self.layers):
            layer = ind + 1

            (
                maybe_skip_proj,
                speech_conv,
                attn_norm,
                attn,
                maybe_attn_adaln_zero,
                ff_norm,
                ff,
                maybe_ff_adaln_zero
            ) = speech_modules

            # smaller text transformer
            # 更小的文本 Transformer

            if exists(text_embed) and exists(text_modules):

                (
                    text_conv,
                    text_attn_norm,
                    text_attn,
                    text_ff_norm,
                    text_ff,
                    cross_condition
                ) = text_modules

                text_embed = text_conv(text_embed, mask = mask) + text_embed

                text_attn_out, text_attn_inter = text_attn(text_attn_norm(text_embed), rotary_pos_emb = text_rotary_pos_emb, mask = mask, return_intermediates = True, value_residual = text_attn_first_values)
                text_embed = text_attn_out + text_embed

                text_attn_first_values = default(text_attn_first_values, text_attn_inter.values)

                text_embed = text_ff(text_ff_norm(text_embed)) + text_embed

                x, text_embed = cross_condition(x, text_embed)

            # skip connection logic
            # 跳跃连接逻辑

            is_first_half = layer <= (self.depth // 2)
            is_later_half = not is_first_half

            if is_first_half:
                skips.append(x)

            if is_later_half:
                skip = skips.pop()
                x = torch.cat((x, skip), dim = -1)
                x = maybe_skip_proj(x)

            # position generating convolution
            # 位置生成卷积

            x = speech_conv(x, mask = mask) + x

            # attention and feedforward blocks
            # 注意力机制和前馈神经网络模块

            attn_out, attn_inter = attn(attn_norm(x, **norm_kwargs), rotary_pos_emb = rotary_pos_emb, mask = mask, return_intermediates = True, value_residual = attn_first_values)

            attn_first_values = default(attn_first_values, attn_inter.values)

            x = x + maybe_attn_adaln_zero(attn_out, **norm_kwargs)

            ff_out = ff(ff_norm(x, **norm_kwargs))

            x = x + maybe_ff_adaln_zero(ff_out, **norm_kwargs)

        assert len(skips) == 0

        _, x = unpack(x, registers_packed_shape, 'b * d')

        return self.final_norm(x)


# main classes

class DurationPredictor(Module):
    """
    DurationPredictor 类用于预测音频时长。
    该模块使用 Transformer 模型对输入的音频特征进行处理，并结合文本特征进行时长预测。
    支持多种输入类型，包括原始波形、梅尔频谱等，并可选择是否返回损失。

    参数说明:
        transformer (dict 或 Transformer): Transformer 模型的配置字典或实例。如果传入字典，则根据配置初始化 Transformer 模型。
        num_channels (int, 可选): 输入音频的特征通道数。如果未指定，则使用 MelSpec 的 n_mel_channels。
        mel_spec_kwargs (dict, 可选): MelSpec 模块的关键字参数，默认为空字典。
        char_embed_kwargs (dict, 可选): CharacterEmbed 模块的关键字参数，默认为空字典。
        text_num_embeds (int, 可选): 文本嵌入的数量。如果未指定，则根据 tokenizer 自动确定。
        tokenizer (str 或 Callable, 可选): 分词器类型或自定义分词器函数。默认为 'char_utf8'，可选 'phoneme_en' 或自定义函数。
    """
    @beartype
    def __init__(
        self,
        transformer: dict | Transformer,
        num_channels = None,
        mel_spec_kwargs: dict = dict(),
        char_embed_kwargs: dict = dict(),
        text_num_embeds = None,
        tokenizer: (
            Literal['char_utf8', 'phoneme_en'] |
            Callable[[list[str]], Int['b nt']]
        ) = 'char_utf8'
    ):
        super().__init__()
        
        if isinstance(transformer, dict):
            # 如果传入的是字典，则根据配置初始化 Transformer 模型
            transformer = Transformer(
                **transformer,
                # 不启用时间条件化
                cond_on_time = False
            )

        # mel spec
        # Mel 频谱转换模块
        self.mel_spec = MelSpec(**mel_spec_kwargs)
        # 如果未指定 num_channels，则使用 MelSpec 的 n_mel_channels
        self.num_channels = default(num_channels, self.mel_spec.n_mel_channels)

        # Transformer 模型
        self.transformer = transformer

        # Transformer 模型
        dim = transformer.dim
        # Transformer 的文本特征维度
        dim_text = transformer.dim_text

        # 特征维度
        self.dim = dim

        # 线性投影层，将输入音频特征映射到 Transformer 的维度
        self.proj_in = Linear(self.num_channels, self.dim)

        # tokenizer and text embed
        # 分词器和文本嵌入
        if callable(tokenizer):
            assert exists(text_num_embeds), '`text_num_embeds` must be given if supplying your own tokenizer encode function'
            # 自定义分词器函数
            self.tokenizer = tokenizer 
        elif tokenizer == 'char_utf8':
            # 设置文本嵌入数量
            text_num_embeds = 256
            # 使用 UTF-8 字符级分词器
            self.tokenizer = list_str_to_tensor
        elif tokenizer == 'phoneme_en':
            # 使用英文音素分词器
            self.tokenizer, text_num_embeds = get_g2p_en_encode()
        else:
            # 如果分词器类型未知，则抛出异常
            raise ValueError(f'unknown tokenizer string {tokenizer}')

        # 字符嵌入层
        self.embed_text = CharacterEmbed(dim_text, num_embeds = text_num_embeds, **char_embed_kwargs)

        # to prediction
        # 预测层
        self.to_pred = Sequential(
            nn.Linear(dim, 1, bias=False),  # 线性层，将特征维度映射到1维
            nn.Softplus(),  # Softplus 激活函数
            Rearrange('... 1 -> ...')  # 重塑张量形状，去除最后一个维度
        )

    def forward(
        self,
        x: Float['b n d'] | Float['b nw'],
        *,
        text: Int['b nt'] | list[str] | None = None,
        lens: Int['b'] | None = None,
        return_loss = True
    ):
        """
        前向传播方法，执行时长预测。

        参数:
            x (Tensor 或 Float['b n d'] 或 Float['b nw']): 输入音频特征，可以是原始波形或梅尔频谱。
            text (Tensor 或 Int['b nt'] 或 list[str] 或 None, 可选): 输入文本，可以是文本列表或张量。
            lens (Tensor 或 Int['b'] 或 None, 可选): 音频长度，可以是张量或 None。
            return_loss (bool, 可选): 是否返回损失，默认为 True。

        返回:
            Union[Tensor, Tuple[Tensor, Tensor]]: 如果 return_loss 为 False，则返回预测结果；否则返回损失和预测结果。
        """
        # 原始波形
        if x.ndim == 2:
            # 将原始波形转换为梅尔频谱
            x = self.mel_spec(x)
            # 重塑张量形状
            x = rearrange(x, 'b d n -> b n d')
            assert x.shape[-1] == self.dim

        # 将梅尔频谱映射到 Transformer 的维度
        x = self.proj_in(x)

        # 获取批次大小、序列长度和设备
        batch, seq_len, device = *x.shape[:2], x.device

        # text
        # 文本
        text_embed = None

        if exists(text):
            if isinstance(text, list):
                # 如果文本是列表，则转换为张量
                text = list_str_to_tensor(text).to(device)
                assert text.shape[0] == batch

            # 对文本进行嵌入
            text_embed = self.embed_text(text, seq_len)

        # handle lengths (duration)
        # 处理长度（时长）
        if not exists(lens):
            # 如果未提供长度，则使用序列长度
            lens = torch.full((batch,), seq_len, device = device)

        # 根据长度生成掩码
        mask = lens_to_mask(lens, length = seq_len)

        # if returning a loss, mask out randomly from an index and have it predict the duration
        # 如果返回损失，则随机选择一个索引并掩码掉，预测该位置的时长
        if return_loss:
            # 生成随机数
            rand_frac_index = x.new_zeros(batch).uniform_(0, 1)
            # 计算随机索引
            rand_index = (rand_frac_index * lens).long()

            # 生成序列索引
            seq = torch.arange(seq_len, device = device)
            # 掩码掉随机索引之前的位置
            mask &= einx.less('n, b -> b n', seq, rand_index)

        # attending
        # 进行注意力机制处理
        x = self.transformer(
            x,
            mask = mask,
            text_embed = text_embed,
        )

        # 对掩码后的张量进行平均
        x = maybe_masked_mean(x, mask)
        # 进行预测
        pred = self.to_pred(x)

        # return the prediction if not returning loss
        # 如果不返回损失，则返回预测结果
        if not return_loss:
            return pred

        # loss
        # 计算损失
        return F.mse_loss(pred, lens.float())


class E2TTS(Module):
    """
    E2TTS 类实现了一个端到端的文本到语音合成模型（End-to-End Text-to-Speech Synthesis）。
    该模型使用 Transformer 模型处理输入的文本特征，并结合时长预测模块生成梅尔频谱。
    支持多种输入类型，包括字符和音素，并可选择是否使用速度一致性损失。

    参数说明:
        transformer (dict 或 Transformer, 可选): Transformer 模型的配置字典或实例。如果传入字典，则根据配置初始化 Transformer 模型。默认为 None。
        duration_predictor (dict 或 DurationPredictor, 可选): 时长预测器的配置字典或实例。如果传入字典，则根据配置初始化时长预测器。默认为 None。
        odeint_kwargs (dict, 可选): ODE 求解器的关键字参数，默认为 {'atol': 1e-5, 'rtol': 1e-5, 'method': 'midpoint'}。
        cond_drop_prob (float, 可选): 条件丢弃概率，用于随机丢弃文本条件，默认为0.25。
        num_channels (int, 可选): 输入音频的特征通道数。如果未指定，则使用 MelSpec 的 n_mel_channels。
        mel_spec_module (Module, 可选): 梅尔频谱模块。如果未指定，则使用 MelSpec 类并根据 mel_spec_kwargs 初始化。
        char_embed_kwargs (dict, 可选): 字符嵌入模块的关键字参数，默认为空字典。
        mel_spec_kwargs (dict, 可选): 梅尔频谱模块的关键字参数，默认为空字典。
        frac_lengths_mask (tuple[float, float], 可选): 时长掩码的分数范围，默认为 (0.7, 1.0)。
        concat_cond (bool, 可选): 是否将条件与输入连接而不是相加，默认为 False。
        interpolated_text (bool, 可选): 是否使用插值文本嵌入，默认为 False。
        text_num_embeds (int, 可选): 文本嵌入的数量。如果未指定，则根据 tokenizer 自动确定。
        tokenizer (str 或 Callable, 可选): 分词器类型或自定义分词器函数，默认为 'char_utf8'，可选 'phoneme_en' 或自定义函数。
        use_vocos (bool, 可选): 是否使用 vocos 模块进行梅尔频谱到音频的转换，默认为 True。
        pretrained_vocos_path (str, 可选): 预训练的 vocos 模型路径，默认为 'charactr/vocos-mel-24khz'。
        sampling_rate (int, 可选): 采样率。如果未指定，则使用 MelSpec 的采样率。
        velocity_consistency_weight (float, 可选): 速度一致性损失的权重，默认为0。
    """
    @beartype
    def __init__(
        self,
        transformer: dict | Transformer = None,
        duration_predictor: dict | DurationPredictor | None = None,
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
        cond_drop_prob = 0.25,
        num_channels = None,
        mel_spec_module: Module | None = None,
        char_embed_kwargs: dict = dict(),
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.),
        concat_cond = False,
        interpolated_text = False,
        text_num_embeds: int | None = None,
        tokenizer: (
            Literal['char_utf8', 'phoneme_en'] |
            Callable[[list[str]], Int['b nt']]
        ) = 'char_utf8',
        use_vocos = True,
        pretrained_vocos_path = 'charactr/vocos-mel-24khz',
        sampling_rate: int | None = None,
        velocity_consistency_weight = 0.,
    ):
        super().__init__()

        if isinstance(transformer, dict):
            # 如果传入的是字典，则根据配置初始化 Transformer 模型
            transformer = Transformer(
                **transformer,
                # 启用时间条件化
                cond_on_time = True
            )

        if isinstance(duration_predictor, dict):
            # 如果传入的是字典，则根据配置初始化时长预测器
            duration_predictor = DurationPredictor(**duration_predictor)

        # Transformer 模型
        self.transformer = transformer

        # Transformer 的特征维度
        dim = transformer.dim
        # Transformer 的文本特征维度
        dim_text = transformer.dim_text

        # 特征维度
        self.dim = dim
        # 文本特征维度
        self.dim_text = dim_text

        # 时长掩码的分数范围
        self.frac_lengths_mask = frac_lengths_mask

        # 时长预测器
        self.duration_predictor = duration_predictor

        # sampling
        # 采样参数
        # ODE 求解器的关键字参数
        self.odeint_kwargs = odeint_kwargs

        # mel spec
        # 梅尔频谱模块
        # 使用 MelSpec 模块
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        # 如果未指定 num_channels，则使用 MelSpec 的 n_mel_channels
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)

        # 输入音频的特征通道数
        self.num_channels = num_channels
        # 采样率
        self.sampling_rate = default(sampling_rate, getattr(self.mel_spec, 'sampling_rate', None))

        # whether to concat condition and project rather than project both and sum
        # 是否将条件连接并投影，而不是分别投影后相加
        self.concat_cond = concat_cond

        if concat_cond:
            # 连接并投影
            self.proj_in = nn.Linear(num_channels * 2, dim)
        else:
            # 分别投影
            self.proj_in = nn.Linear(num_channels, dim)
            # 条件投影
            self.cond_proj_in = nn.Linear(num_channels, dim)

        # to prediction
        # 预测层
        # 线性层，将特征维度映射到音频通道数
        self.to_pred = Linear(dim, num_channels)

        # tokenizer and text embed
        # 分词器和文本嵌入
        if callable(tokenizer):
            assert exists(text_num_embeds), '`text_num_embeds` must be given if supplying your own tokenizer encode function'
            # 自定义分词器函数
            self.tokenizer = tokenizer
        elif tokenizer == 'char_utf8':
            # 设置文本嵌入数量
            text_num_embeds = 256
            # 使用 UTF-8 字符级分词器
            self.tokenizer = list_str_to_tensor
        elif tokenizer == 'phoneme_en':
            # 使用英文音素分词器
            self.tokenizer, text_num_embeds = get_g2p_en_encode()
        else:
            # 如果分词器类型未知，则抛出异常
            raise ValueError(f'unknown tokenizer string {tokenizer}')

        # 条件丢弃概率
        self.cond_drop_prob = cond_drop_prob

        # text embedding
        # 文本嵌入
        text_embed_klass = CharacterEmbed if not interpolated_text else InterpolatedCharacterEmbed
        # 字符嵌入层
        self.embed_text = text_embed_klass(dim_text, num_embeds = text_num_embeds, **char_embed_kwargs)

        # weight for velocity consistency
        # 速度一致性权重
        # 注册一个零张量
        self.register_buffer('zero', torch.tensor(0.), persistent = False)
        # 速度一致性损失的权重
        self.velocity_consistency_weight = velocity_consistency_weight

        # default vocos for mel -> audio
        # 默认的 vocos 模块，用于梅尔频谱到音频的转换
        # 使用预训练的 vocos 模块
        self.vocos = Vocos.from_pretrained(pretrained_vocos_path) if use_vocos else None

    @property
    def device(self):
        return next(self.parameters()).device

    def transformer_with_pred_head(
        self,
        x: Float['b n d'],
        cond: Float['b n d'],
        times: Float['b'],
        mask: Bool['b n'] | None = None,
        text: Int['b nt'] | None = None,
        drop_text_cond: bool | None = None,
        return_drop_text_cond = False
    ):
        """
        使用 Transformer 模型进行前向传播，并添加预测头。

        参数:
            x (Float['b n d']): 输入张量。
            cond (Float['b n d']): 条件输入张量。
            times (Float['b']): 时间条件输入。
            mask (Bool['b n'] | None, 可选): 输入掩码。
            text (Int['b nt'] | None, 可选): 输入文本。
            drop_text_cond (bool | None, 可选): 是否丢弃文本条件。
            return_drop_text_cond (bool, 可选): 是否返回是否丢弃文本条件的标志。

        返回:
            Union[Tensor, Tuple[Tensor, bool]]: 返回预测结果或预测结果和是否丢弃文本条件的标志。
        """
        # 获取序列长度
        seq_len = x.shape[-2]
        # 确定是否丢弃文本条件
        drop_text_cond = default(drop_text_cond, self.training and random() < self.cond_drop_prob)

        if self.concat_cond:
            # concat condition, given as using voicebox-like scheme
            # 如果连接条件，则使用类似 voicebox 的方案
            x = torch.cat((cond, x), dim = -1)

        # 投影输入
        x = self.proj_in(x)

        if not self.concat_cond:
            # an alternative is to simply sum the condition
            # seems to work fine
            # 如果不连接条件，则将条件相加
            cond = self.cond_proj_in(cond)
            x = x + cond

        # whether to use a text embedding
        # 是否使用文本嵌入
        text_embed = None
        if exists(text) and not drop_text_cond:
            text_embed = self.embed_text(text, seq_len, mask = mask)

        # attend
        # 进行注意力机制处理
        attended = self.transformer(
            x,
            times = times,
            mask = mask,
            text_embed = text_embed
        )

        # 添加预测头
        pred =  self.to_pred(attended)

        if not return_drop_text_cond:
            return pred

        return pred, drop_text_cond

    def cfg_transformer_with_pred_head(
        self,
        *args,
        cfg_strength: float = 1.,
        remove_parallel_component: bool = True,
        keep_parallel_frac: float = 0.,
        **kwargs,
    ):
        """
        使用 Transformer 模型进行前向传播，并添加预测头，同时应用条件指导（CFG）。

        参数:
            *args: 位置参数。
            cfg_strength (float, 可选): 条件指导强度，默认为1。
            remove_parallel_component (bool, 可选): 是否移除平行分量，默认为True。
            keep_parallel_frac (float, 可选): 保留平行分量的比例，默认为0。
            **kwargs: 其他关键字参数。

        返回:
            Tensor: 条件指导后的预测结果。
        """
        # 进行 Transformer 处理，不丢弃文本条件
        pred = self.transformer_with_pred_head(*args, drop_text_cond = False, **kwargs)

        if cfg_strength < 1e-5:
            # 如果条件指导强度极小，则直接返回预测结果
            return pred
        # 进行 Transformer 处理，丢弃文本条件
        null_pred = self.transformer_with_pred_head(*args, drop_text_cond = True, **kwargs)

        # 计算条件指导更新
        cfg_update = pred - null_pred

        if remove_parallel_component:
            # 计算平行和正交分量
            # https://arxiv.org/abs/2410.02416
            parallel, orthogonal = project(cfg_update, pred)
            # 调整平行分量
            cfg_update = orthogonal + parallel * keep_parallel_frac
        # 应用条件指导更新
        return pred + cfg_update * cfg_strength

    @torch.no_grad()
    def sample(
        self,
        cond: Float['b n d'] | Float['b nw'],
        *,
        text: Int['b nt'] | list[str] | None = None,
        lens: Int['b'] | None = None,
        duration: int | Int['b'] | None = None,
        steps = 32,
        cfg_strength = 1.,   # they used a classifier free guidance strength of 1.
        max_duration = 4096, # in case the duration predictor goes haywire
        vocoder: Callable[[Float['b d n']], list[Float['_']]] | None = None,
        return_raw_output: bool | None = None,
        save_to_filename: str | None = None
    ):
        """
        对输入条件进行采样，生成音频。

        参数:
            cond (Tensor): 条件输入，可以是梅尔频谱或原始波形。
            text (Tensor 或 list[str], 可选): 输入文本，可以是文本列表或张量。
            lens (Tensor 或 int, 可选): 音频长度，可以是张量或整数。
            duration (int 或 Tensor, 可选): 时长，可以是整数或张量。
            steps (int, 可选): 采样步数，默认为32。
            cfg_strength (float, 可选): 条件指导强度，默认为1。
            max_duration (int, 可选): 最大时长，默认为4096。
            vocoder (Callable, 可选): 自定义 vocoder 函数。
            return_raw_output (bool, 可选): 是否返回原始输出。
            save_to_filename (str, 可选): 保存输出到文件。

        返回:
            List[Tensor]: 生成的音频列表。
        """
        self.eval()

        # raw wave
        # 处理原始波形
        if cond.ndim == 2:
            # 将原始波形转换为梅尔频谱
            cond = self.mel_spec(cond)
            # 重塑张量形状
            cond = rearrange(cond, 'b d n -> b n d')
            assert cond.shape[-1] == self.num_channels

        # 获取批次大小、序列长度和设备
        batch, cond_seq_len, device = *cond.shape[:2], cond.device

        if not exists(lens):
            # 如果未提供长度，则使用序列长度
            lens = torch.full((batch,), cond_seq_len, device = device, dtype = torch.long)

        # text
        # 处理文本
        if isinstance(text, list):
            # 如果文本是列表，则转换为张量
            text = self.tokenizer(text).to(device)
            assert text.shape[0] == batch

        if exists(text):
            # 计算文本长度
            text_lens = (text != -1).sum(dim = -1)
            # 确保长度至少与文本长度相同
            lens = torch.maximum(text_lens, lens) # make sure lengths are at least those of the text characters

        # duration
        # 处理时长
        # 生成条件掩码
        cond_mask = lens_to_mask(lens)

        if exists(duration):
            if isinstance(duration, int):
                # 如果时长是整数，则转换为张量
                duration = torch.full((batch,), duration, device = device, dtype = torch.long)

        elif exists(self.duration_predictor):
            # 使用时长预测器预测时长
            duration = self.duration_predictor(cond, text = text, lens = lens, return_loss = False).long()

        # 增加一个时长，以确保生成内容
        duration = torch.maximum(lens + 1, duration) # just add one token so something is generated
        # 限制最大时长
        duration = duration.clamp(max = max_duration)

        assert duration.shape[0] == batch

        # 获取最大时长
        max_duration = duration.amax()

        # 对条件输入进行填充
        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value = 0.)
        # 对条件掩码进行填充
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value = False)
        # 重塑条件掩码形状
        cond_mask = rearrange(cond_mask, '... -> ... 1')

        # 生成时长掩码
        mask = lens_to_mask(duration)

        # neural ode
        # 神经常微分方程（Neural ODE）

        def fn(t, x):
            # at each step, conditioning is fixed
            # 在每一步中，条件是固定的

            # 根据条件掩码选择条件输入
            step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow
            # 预测流
            return self.cfg_transformer_with_pred_head(
                x,
                step_cond,
                times = t,
                text = text,
                mask = mask,
                cfg_strength = cfg_strength
            )

        # 初始化输入
        y0 = torch.randn_like(cond)
        # 生成时间步
        t = torch.linspace(0, 1, steps, device = self.device)
        
        # 进行常微分方程求解
        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        # 获取最终采样结果
        sampled = trajectory[-1]

        out = sampled
        # 根据条件掩码替换输出
        out = torch.where(cond_mask, cond, out)

        # able to return raw untransformed output, if not using mel rep
        # 如果需要返回原始未转换的输出，如果未使用梅尔表示，则返回

        if exists(return_raw_output) and return_raw_output:
            return out

        # take care of transforming mel to audio if `vocoder` is passed in, or if `use_vocos` is turned on
        # 如果传入 `vocoder`，或者 `use_vocos` 被打开，则将梅尔转换为音频

        if exists(vocoder):
            assert not exists(self.vocos), '`use_vocos` should not be turned on if you are passing in a custom `vocoder` on sampling'
            out = rearrange(out, 'b n d -> b d n') # 重塑张量形状
            out = vocoder(out) # 使用自定义 vocoder 进行转换

        elif exists(self.vocos):
            
            # 初始化音频列表
            audio = []
            for mel, one_mask in zip(out, mask):
                # 将梅尔频谱转换为幅度
                one_out = DB_to_amplitude(mel[one_mask], ref = 1., power = 0.5)

                # 重塑张量形状
                one_out = rearrange(one_out, 'n d -> 1 d n')
                # 使用 vocos 进行解码
                one_audio = self.vocos.decode(one_out)
                # 重塑音频形状
                one_audio = rearrange(one_audio, '1 nw -> nw')
                # 添加到音频列表
                audio.append(one_audio)

            out = audio

        if exists(save_to_filename):
            # 确保存在 vocoder 或 vocos
            assert exists(vocoder) or exists(self.vocos)
            # 确保存在采样率
            assert exists(self.sampling_rate)

            # 创建路径对象
            path = Path(save_to_filename)
            parent_path = path.parents[0]
            parent_path.mkdir(exist_ok = True, parents = True)

            for ind, one_audio in enumerate(out):
                # 重塑音频形状
                one_audio = rearrange(one_audio, 'nw -> 1 nw')
                # 生成保存路径
                save_path = str(parent_path / f'{ind + 1}.{path.name}')
                torchaudio.save(save_path, one_audio.detach().cpu(), sample_rate = self.sampling_rate)

        return out

    def forward(
        self,
        inp: Float['b n d'] | Float['b nw'], # mel or raw wave
        *,
        text: Int['b nt'] | list[str] | None = None,
        times: Int['b'] | None = None,
        lens: Int['b'] | None = None,
        velocity_consistency_model: E2TTS | None = None,
        velocity_consistency_delta = 1e-5
    ):
        """
        前向传播方法，执行模型的前向计算。

        参数:
            inp (Tensor): 输入可以是梅尔频谱或原始波形。
            text (Tensor 或 list[str], 可选): 输入文本，可以是文本列表或张量。
            times (Tensor 或 int, 可选): 时间条件输入。
            lens (Tensor 或 int, 可选): 音频长度，可以是张量或 None。
            velocity_consistency_model (E2TTS, 可选): 速度一致性模型。
            velocity_consistency_delta (float, 可选): 速度一致性损失参数。

        返回:
            E2TTSReturn: 返回包含总损失、条件输入、预测结果、生成流和损失分解的对象。
        """
        need_velocity_loss = exists(velocity_consistency_model) and self.velocity_consistency_weight > 0.

        # handle raw wave
        # 处理原始波形
        if inp.ndim == 2:
            # 将原始波形转换为梅尔频谱
            inp = self.mel_spec(inp)
            # 重塑张量形状
            inp = rearrange(inp, 'b d n -> b n d')
            assert inp.shape[-1] == self.num_channels

        # 获取批次大小、序列长度、数据类型和设备
        batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device

        # handle text as string
        # 处理文本
        if isinstance(text, list):
            # 如果文本是列表，则转换为张量
            text = self.tokenizer(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        # 处理长度和掩码
        if not exists(lens):
            # 如果未提供长度，则使用序列长度
            lens = torch.full((batch,), seq_len, device = device)

        # 生成掩码
        mask = lens_to_mask(lens, length = seq_len)

        # get a random span to mask out for training conditionally
        # 获取随机跨度掩码，用于训练时的填充
        # 生成随机长度
        frac_lengths = torch.zeros((batch,), device = self.device).float().uniform_(*self.frac_lengths_mask)
        # 生成随机跨度掩码
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths, max_length = seq_len)

        if exists(mask):
            # 结合掩码
            rand_span_mask &= mask

        # mel is x1
        # 输入梅尔频谱
        x1 = inp

        # main conditional flow training logic
        # just ~5 loc
        # 主条件流训练逻辑

        # x0 is gaussian noise
        # x0 是高斯噪声
        # 生成高斯噪声
        x0 = torch.randn_like(x1)

        # t is random times from above
        # t 是随机时间
        # 生成随机时间
        times = torch.rand((batch,), dtype = dtype, device = self.device)
        # 重塑张量形状
        t = rearrange(times, 'b -> b 1 1')

        # if need velocity consistency, make sure time does not exceed 1.
        # 如果需要速度一致性，确保时间不超过1
        if need_velocity_loss:
            # 调整时间
            t = t * (1. - velocity_consistency_delta)

        # sample xt (w in the paper)
        # 计算样本
        w = (1. - t) * x0 + t * x1
        # 计算流
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        # 仅预测随机掩码跨度内的内容，用于填充
        cond = einx.where(
            'b n, b n d, b n d -> b n d',
            rand_span_mask,
            torch.zeros_like(x1), x1 # 根据掩码选择条件输入
        )

        # transformer and prediction head
        # Transformer 和预测头
        pred, did_drop_text_cond = self.transformer_with_pred_head(
            w,
            cond,
            times = times,
            text = text,
            mask = mask,
            return_drop_text_cond = True # 返回是否丢弃文本条件
        )

        # maybe velocity consistency loss
        # 可能的速度一致性损失
        velocity_loss = self.zero

        if need_velocity_loss:
            # 调整时间
            t_with_delta = t + velocity_consistency_delta
            # 计算样本
            w_with_delta = (1. - t_with_delta) * x0 + t_with_delta * x1

            with torch.no_grad():
                ema_pred = velocity_consistency_model.transformer_with_pred_head(
                    w_with_delta,
                    cond,
                    times = times + velocity_consistency_delta,
                    text = text,
                    mask = mask,
                    drop_text_cond = did_drop_text_cond # 丢弃文本条件
                )

            # 计算均方误差损失
            velocity_loss = F.mse_loss(pred, ema_pred, reduction = 'none')
            # 计算平均损失
            velocity_loss = velocity_loss[rand_span_mask].mean()

        # flow matching loss
        # 流匹配损失
        # 计算均方误差损失
        loss = F.mse_loss(pred, flow, reduction = 'none')
        # 计算平均损失
        loss = loss[rand_span_mask].mean()

        # total loss and get breakdown
        # 总损失和分解损失
        total_loss = (
            loss +
            velocity_loss * self.velocity_consistency_weight # 速度一致性损失
        )
        # 分解损失
        breakdown = LossBreakdown(loss, velocity_loss)

        # return total loss and bunch of intermediates
        # 返回总损失和中间结果
        return E2TTSReturn(total_loss, cond, pred, x0 + pred, breakdown)
