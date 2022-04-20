import numpy as np
import torch
from torch.distributions import Normal
from torch import nn
from torch.nn import functional as F
import clip
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F 

class Rasterizer(nn.Module):
  """
  (N, num_strokes, num_control_points*2) -> (N, 3, 224, 224)
  """
  def __init__(self, num_control_points, num_strokes, device):
    super(Rasterizer, self).__init__()
    self.num_control_points = num_control_points
    self.num_strokes = num_strokes
    self.fc = nn.Linear(num_strokes*num_control_points*2, 3*224*224)
  def forward(self, figures):
    return self.fc(figures.view(-1, self.num_storokes*self.num_control_points*2)).view(-1, 3, 224, 224)

#class Rasterizer(nn.Module):
#  """
#  (N, num_strokes, num_control_points*2) -> (N, 3, 224, 224)
#  """
#  def __init__(self, num_control_points, num_strokes, device):
#    super(Rasterizer, self).__init__()
#    self.num_control_points = num_control_points
#    self.num_strokes = num_strokes
#    self.renderer = NeuralRenderer(self.num_control_points)
#    self.renderer.load_state_dict(torch.load("params.pkl"))
#    self.renderer.to(device)
#    self.renderer.eval()
#  def forward(self, figures):
#    return self.sum_imgs(self.renderer(figures.view(-1, self.num_control_points*2)).view(-1, self.num_strokes, 3, 224, 224))
#  def sum_imgs(self, imgs):
#    return 1 - torch.sigmoid( (torch.sum(1-imgs, dim=1) - 0.5) * 10)
#  def zero_grad(self):
#    self.renderer.zero_grad()

class NeuralRenderer(nn.Module):
    """
    (N, num_controlpoints*2) -> (N, 3, 224, 224)
    """
    def __init__(self, num_control_points):
        super(NeuralRenderer, self).__init__()
        self.fc1 = nn.Linear(num_control_points*2, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 4096)
        self.fc5 = nn.Linear(4096, 6272)
        self.conv1 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv6 = nn.Conv2d(32, 32, 3, 1, 1)
        self.norm3 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv8 = nn.Conv2d(16, 12, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = x.view(-1, 32, 14, 14)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.norm1(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.norm2(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.norm3(self.conv6(x)))
        x = F.relu(self.conv7(x))
        x = self.pixel_shuffle(self.conv8(x))
        x = torch.sigmoid(x)
        return x

class CLIP(torch.nn.Module):
    def __init__(self, device):
        super(CLIP, self).__init__()
        self.device = device
        self.model, clip_preprocess = clip.load("RN101", self.device, jit=False)
        self.visual_model = self.model.visual
        layers = list(self.model.visual.children())
        init_layers = torch.nn.Sequential(*layers)[:8]
        self.layer1 = layers[8]
        self.layer2 = layers[9]
        self.layer3 = layers[10]
        self.layer4 = layers[11]
        self.att_pool2d = layers[12]

        self.normalize_transform = transforms.Compose([
            clip_preprocess.transforms[0],  # Resize
            clip_preprocess.transforms[1],  # CenterCrop
            clip_preprocess.transforms[-1],  # Normalize
        ])

        self.model.eval()

    def forward(self, sketch):
        x = sketch.to(self.device)
        x = self.normalize_transform(x).unsqueeze(0).contiguous()
        def stem(m, x):
            for conv, bn in [(m.conv1, m.bn1), (m.conv2, m.bn2), (m.conv3, m.bn3)]:
                x = m.relu(bn(conv(x)))
            x = m.avgpool(x)
            return x
        x = x.type(self.visual_model.conv1.weight.dtype)
        x = stem(self.visual_model, x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        y = self.att_pool2d(x4)
        return y/torch.norm(y), [x2, x3]

class CLIPConvLoss(torch.nn.Module):
    def __init__(self, device):
        super(CLIPConvLoss, self).__init__()
        self.device = device
        self.model, clip_preprocess = clip.load("RN101", self.device, jit=False)

        self.visual_model = self.model.visual
        layers = list(self.model.visual.children())
        init_layers = torch.nn.Sequential(*layers)[:8]
        self.layer1 = layers[8]
        self.layer2 = layers[9]
        self.layer3 = layers[10]
        self.layer4 = layers[11]
        self.att_pool2d = layers[12]

        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
        ])  # clip normalisation
        self.normalize_transform = transforms.Compose([
            clip_preprocess.transforms[0],  # Resize
            clip_preprocess.transforms[1],  # CenterCrop
            clip_preprocess.transforms[-1],  # Normalize
        ])

        self.model.eval()
        self.num_augs = 0 # for avoiding an adversarial sketch

        augemntations = []
        augemntations.append(transforms.RandomPerspective(
            fill=0, p=1.0, distortion_scale=0.5))
        augemntations.append(transforms.RandomResizedCrop(
            224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)

    def forward(self, sketch, target):
        x = sketch.to(self.device)
        y = target.to(self.device)
        sketch_augs, img_augs = [self.normalize_transform(x)], [
            self.normalize_transform(y)]
    
        for _ in range(self.num_augs):
            augmented_pair = self.augment_trans(torch.cat([x, y]))
            sketch_augs.append(augmented_pair[0].unsqueeze(0))
            img_augs.append(augmented_pair[1].unsqueeze(0))

        xs = torch.cat(sketch_augs, dim=0).to(self.device)
        ys = torch.cat(img_augs, dim=0).to(self.device)

        xs_fc_features, xs_conv_features = self.forward_inspection_clip_resnet(
            xs.contiguous())
        ys_fc_features, ys_conv_features = self.forward_inspection_clip_resnet(
            ys.detach())

        #TODO: compare mean or sum
        conv_loss = [torch.square(x_conv - y_conv).mean() for x_conv, y_conv in zip(xs_conv_features, ys_conv_features)]
        fc_loss = (1 - torch.cosine_similarity(xs_fc_features, ys_fc_features, dim=1)).mean()
            
        return sum(conv_loss) + 0.1 * fc_loss

    def forward_inspection_clip_resnet(self, x):
        def stem(m, x):
            for conv, bn in [(m.conv1, m.bn1), (m.conv2, m.bn2), (m.conv3, m.bn3)]:
                x = m.relu(bn(conv(x)))
            x = m.avgpool(x)
            return x
        x = x.type(self.visual_model.conv1.weight.dtype)
        x = stem(self.visual_model, x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        y = self.att_pool2d(x4)
        return y, [x2, x3]

class TransitionModel(nn.Module):
    """
    このクラスは複数の要素を含んでいます.
    決定的状態遷移 （RNN) : h_t+1 = f(h_t, s_t, a_t)
    確率的状態遷移による1ステップ予測として定義される "prior" : p(s_t+1 | h_t+1)
    観測の情報を取り込んで定義される "posterior": q(s_t+1 | h_t+1, e_t+1)
    """
    def __init__(self, state_dim, action_dim, rnn_hidden_dim, embedding_dim=1024,
                 hidden_dim=200, min_stddev=0.1, act=F.elu):
        super(TransitionModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc_state_action = nn.Linear(state_dim + action_dim, hidden_dim)
      
        self.fc_rnn_hidden = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.fc_state_mean_prior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_stddev_prior = nn.Linear(hidden_dim, state_dim)
        self.fc_rnn_hidden_embedded_obs = nn.Linear(rnn_hidden_dim + embedding_dim, hidden_dim)
        self.fc_state_mean_posterior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_stddev_posterior = nn.Linear(hidden_dim, state_dim)

        #next hidden stateを計算
        self.rnn = nn.GRUCell(hidden_dim, rnn_hidden_dim)
        self._min_stddev = min_stddev
        self.act = act
  

    def forward(self, state, action, rnn_hidden, embedded_next_obs):
        """
        h_t+1 = f(h_t, s_t, a_t)
        prior p(s_t+1 | h_t+1) と posterior q(s_t+1 | h_t+1, e_t+1) を返す
        この2つが近づくように学習する
        """
        next_state_prior, rnn_hidden = self.prior(self.reccurent(state, action, rnn_hidden))
        next_state_posterior = self.posterior(rnn_hidden, embedded_next_obs)
        return next_state_prior, next_state_posterior, rnn_hidden
      
    def reccurent(self, state, action, rnn_hidden):
        """
        h_t+1 = f(h_t, s_t, a_t)を計算する
        """
        hidden = self.act(self.fc_state_action(torch.cat([state, action], dim=1)))
        #h_t+1を求める
        rnn_hidden = self.rnn(hidden, rnn_hidden)
        return rnn_hidden

    def prior(self, rnn_hidden):
        """
        prior p(s_t+1 | h_t+1) を計算する
        """
        #h_t+1を求める
        hidden = self.act(self.fc_rnn_hidden(rnn_hidden))

        mean = self.fc_state_mean_prior(hidden)
        stddev = F.softplus(self.fc_state_stddev_prior(hidden)) + self._min_stddev
        return Normal(mean, stddev), rnn_hidden

    def posterior(self, rnn_hidden, embedded_obs):
        """
        posterior q(s_t+1 | h_t+1, e_t+1)  を計算する
        """
        # h_t+1, o_t+1を結合し, q(s_t+1 | h_t+1, e_t+1) を計算する
        hidden = self.act(self.fc_rnn_hidden_embedded_obs(
            torch.cat([rnn_hidden, embedded_obs], dim=1)))
        mean = self.fc_state_mean_posterior(hidden)
        stddev = F.softplus(self.fc_state_stddev_posterior(hidden)) + self._min_stddev
        return Normal(mean, stddev)

class ObservationModel(nn.Module):
    """
    p(o_t | s_t, h_t)
    低次元の状態表現からControl Pointsを再構成するデコーダ (batch_size, num_strokes*num_control_points)
    """
    def __init__(self, state_dim, rnn_hidden_dim, num_strokes, num_control_points):
        super(ObservationModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + rnn_hidden_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, num_strokes*num_control_points*2)

    def forward(self, state, rnn_hidden):
        hidden = F.relu(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = F.relu(self.fc2(hidden))
        hidden = F.relu(self.fc3(hidden))
        obs = F.sigmoid(self.fc4(hidden))
        return obs

class RewardModel(nn.Module):
    """
    p(r_t | s_t, h_t)
    低次元の状態表現から報酬を予測する
    """
    def __init__(self, state_dim, rnn_hidden_dim, hidden_dim=400, act=F.elu):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.act = act

    def forward(self, state, rnn_hidden):
        hidden = self.act(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = self.act(self.fc2(hidden))
        hidden = self.act(self.fc3(hidden))
        reward = self.fc4(hidden)
        return reward

class RSSM:
    def __init__(self, state_dim, action_dim, rnn_hidden_dim,  device, num_strokes=8, num_control_points=4,):
        self.transition = TransitionModel(state_dim, action_dim, rnn_hidden_dim).to(device)
        self.observation = ObservationModel(state_dim, rnn_hidden_dim, num_strokes, num_control_points).to(device)
        self.reward = RewardModel(state_dim, rnn_hidden_dim,).to(device)

class Encoder(nn.Module):
    """
    (3, 224, 224)の画像を(1024,)のベクトルに変換するエンコーダ
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.cv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)
        self.cv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.cv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.cv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        self.cv5 = nn.Conv2d(256, 256, kernel_size=4, stride=8)

    def forward(self, obs):
        hidden = F.relu(self.cv1(obs))
        hidden = F.relu(self.cv2(hidden))
        hidden = F.relu(self.cv3(hidden))
        hidden = F.relu(self.cv4(hidden))
        embedded_obs = F.relu(self.cv5(hidden)).reshape(hidden.size(0), -1)
        return embedded_obs

#　今回のReplayBuffer
class ReplayBuffer(object):
    """
    RNNを用いて訓練するのに適したリプレイバッファ
    """
    def __init__(self, capacity, observation_shape, action_dim):
        self.capacity = capacity

        self.observations = np.zeros((capacity, *observation_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.bool)

        self.index = 0
        self.is_filled = False

    def push(self, observation, action, reward, done):
        """
        リプレイバッファに経験を追加する
        """
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.done[self.index] = done

        # indexは巡回し, 最も古い経験を上書きする
        if self.index == self.capacity - 1:
            self.is_filled = True
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size, chunk_length):
        """
        経験をリプレイバッファからサンプルします. （ほぼ）一様なサンプルです
        結果として返ってくるのは観測(画像), 行動, 報酬, 終了シグナルについての(batch_size, chunk_length, 各要素の次元)の配列です
        各バッチは連続した経験になっています
        注意: chunk_lengthをあまり大きな値にすると問題が発生する場合があります
        """
        episode_borders = np.where(self.done)[0]
        sampled_indexes = []
        for _ in range(batch_size):
            cross_border = True
            while cross_border:
                initial_index = np.random.randint(len(self) - chunk_length + 1)
                final_index = initial_index + chunk_length - 1
                cross_border = np.logical_and(initial_index <= episode_borders,
                                              episode_borders < final_index).any()#論理積
            sampled_indexes += list(range(initial_index, final_index + 1))

        sampled_observations = self.observations[sampled_indexes].reshape(
            batch_size, chunk_length, *self.observations.shape[1:])
        sampled_actions = self.actions[sampled_indexes].reshape(
            batch_size, chunk_length, self.actions.shape[1])
        sampled_rewards = self.rewards[sampled_indexes].reshape(
            batch_size, chunk_length, 1)
        sampled_done = self.done[sampled_indexes].reshape(
            batch_size, chunk_length, 1)
        return sampled_observations, sampled_actions, sampled_rewards, sampled_done

    def __len__(self):
        return self.capacity if self.is_filled else self.index

class ValueModel(nn.Module):
    """
    低次元の状態表現(state_dim + rnn_hidden_dim)から状態価値を出力する
    """
    def __init__(self, state_dim, rnn_hidden_dim, hidden_dim=400, act=F.elu):
        super(ValueModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.act = act

    def forward(self, state, rnn_hidden):
        hidden = self.act(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = self.act(self.fc2(hidden))
        hidden = self.act(self.fc3(hidden))
        state_value = self.fc4(hidden)
        return state_value

class ActionModel(nn.Module):
    """
    低次元の状態表現(state_dim + rnn_hidden_dim)から行動を計算するクラス
    """
    def __init__(self, state_dim, rnn_hidden_dim, action_dim,
                 hidden_dim=400, act=F.elu, min_stddev=1e-4, init_stddev=5.0):
        super(ActionModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_stddev = nn.Linear(hidden_dim, action_dim)
        self.act = act
        self.min_stddev = min_stddev
        self.init_stddev = np.log(np.exp(init_stddev) - 1)

    def forward(self, state, rnn_hidden, training=True):
        """
        training=Trueなら, NNのパラメータに関して微分可能な形の行動のサンプル（Reparametrizationによる）を返します
        training=Falseなら, 行動の確率分布の平均値を返します
        """
        hidden = self.act(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = self.act(self.fc2(hidden))
        hidden = self.act(self.fc3(hidden))
        hidden = self.act(self.fc4(hidden))

        # Dreamerの実装に合わせて少し平均と分散に対する簡単な変換が入っています
        mean = self.fc_mean(hidden)
        mean = 5.0 * torch.tanh(mean / 5.0)
        stddev = self.fc_stddev(hidden)
        stddev = F.softplus(stddev + self.init_stddev) + self.min_stddev

        if training:
            action = torch.tanh(Normal(mean, stddev).rsample())#微分可能にするためrsample()
        else:
            action = torch.tanh(mean)
        return action
