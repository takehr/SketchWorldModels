import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions.kl import kl_divergence
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from env import make_env
from models import RSSM, ReplayBuffer, ValueModel, ActionModel, Rasterizer, CLIPConvLoss, CLIP, Encoder
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--buffer_capacity", type=int, default=200000//4,
                    help="capacity of replay buffer")
parser.add_argument("--num_strokes", type=int, default=8,
                    help="number of strokes used to generate the sketch, this defines the level of abstraction.")
parser.add_argument("--state_dim", type=int, default=30,
                    help="number of stochastic state dim")
parser.add_argument("--rnn_hidden_dim", type=int, default=200,
                    help="number of rnn hidden dim")
parser.add_argument("--num_control_points", type=int, default=4,
                    help="number of control points of each stroke")
parser.add_argument("--batch_size", type=int, default=50//10,
                    help="batch_size")
parser.add_argument("--chunk_length", type=int, default=50//10,
                    help="length of chunk to update rnn")
parser.add_argument("--model_lr", type=float, default=6e-4,
                    help="model_lr")
parser.add_argument("--value_lr", type=float, default=8e-5,
                    help="value_lr")
parser.add_argument("--action_lr", type=float, default=8e-5,
                    help="action_lr")
parser.add_argument("--eps", type=float, default=1e-4,
                    help="eps of optimizer")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_obs(obs):
    """
    画像の変換. [0, 255] -> [-0.5, 0.5]
    """
    obs = obs.astype(np.float32)
    normalized_obs = obs / 255.0
    return normalized_obs

def lambda_target(rewards, values, gamma, lambda_):
    """
    価値関数の学習のためのλ-returnを計算します
    """
    V_lambda = torch.zeros_like(rewards, device=rewards.device)

    H = rewards.shape[0] - 1
    V_n = torch.zeros_like(rewards, device=rewards.device)
    V_n[H] = values[H]
    for n in range(1, H+1):
        # まずn-step returnを計算します
        # 注意: 系列が途中で終わってしまったら, 可能な中で最大のnを用いたn-stepを使います
        V_n[:-n] = (gamma ** n) * values[n:]
        for k in range(1, n+1):
            if k == n:
                V_n[:-n] += (gamma ** (n-1)) * rewards[k:]
            else:
                V_n[:-n] += (gamma ** (k-1)) * rewards[k:-n+k]

        # lambda_でn-step returnを重みづけてλ-returnを計算します
        if n == H:
            V_lambda += (lambda_ ** (H-1)) * V_n
        else:
            V_lambda += (1 - lambda_) * (lambda_ ** (n-1)) * V_n

    return V_lambda

class Agent:
    """
    ActionModelに基づき行動を決定する. そのためにRSSMを用いて状態表現をリアルタイムで推論して維持するクラス
    """
    def __init__(self, encoder, rssm, action_model):
        self.encoder = encoder
        self.rssm = rssm
        self.action_model = action_model

        self.device = next(self.action_model.parameters()).device
        self.rnn_hidden = torch.zeros(1, rssm.rnn_hidden_dim, device=self.device)

    def __call__(self, obs, training=True):
        # preprocessを適用, PyTorchのためにChannel-Firstに変換
        obs = preprocess_obs(obs)
        obs = torch.as_tensor(obs, device=self.device)
        obs = obs.transpose(1, 2).transpose(0, 1).unsqueeze(0)

        with torch.no_grad():
            # 観測を低次元の表現に変換し, posteriorからのサンプルをActionModelに入力して行動を決定する
            embedded_obs = self.encoder(obs)
            state_posterior = self.rssm.posterior(self.rnn_hidden, embedded_obs)
            state = state_posterior.sample()
            action = self.action_model(state, self.rnn_hidden, training=training)

            # 次のステップのためにRNNの隠れ状態を更新しておく
            _, self.rnn_hidden = self.rssm.prior(self.rssm.reccurent(state, action, self.rnn_hidden))

        return action.squeeze().cpu().numpy()

    #RNNの隠れ状態をリセット
    def reset(self):
        self.rnn_hidden = torch.zeros(1, self.rssm.rnn_hidden_dim, device=self.device)

env = make_env()
# リプレイバッファの宣言
buffer_capacity = args.buffer_capacity  # Colabのメモリの都合上, 元の実装より小さめにとっています
replay_buffer = ReplayBuffer(capacity=buffer_capacity,
                              observation_shape=env.observation_space.shape,
                              action_dim=env.action_space.shape[0])

# モデルの宣言
state_dim = args.state_dim  # 確率的状態の次元
rnn_hidden_dim = args.rnn_hidden_dim  # 決定的状態（RNNの隠れ状態）の次元
num_control_points = args.num_control_points
num_strokes = args.num_strokes
encoder = Encoder().to(device)#確率的状態の次元と決定的状態（RNNの隠れ状態）の次元は一致しなくて良い
rasterizer = Rasterizer(num_control_points, num_strokes, device)
clip_loss = CLIPConvLoss(device)
rssm = RSSM(state_dim,env.action_space.shape[0],rnn_hidden_dim, device, num_strokes, num_control_points)
value_model = ValueModel(state_dim, rnn_hidden_dim).to(device)
action_model = ActionModel(state_dim, rnn_hidden_dim,
                             env.action_space.shape[0]).to(device)

# オプティマイザの宣言
model_lr = args.model_lr  # rssm, obs_model, reward_modelの学習率
value_lr = args.value_lr
action_lr = args.action_lr
eps = args.eps
model_params = (list(rssm.transition.parameters()) +
                  list(rssm.observation.parameters()) +
                  list(rssm.reward.parameters()))
model_optimizer = torch.optim.Adam(model_params, lr=model_lr, eps=eps)
value_optimizer = torch.optim.Adam(value_model.parameters(), lr=value_lr, eps=eps)
action_optimizer = torch.optim.Adam(action_model.parameters(), lr=action_lr, eps=eps)

# その他ハイパーパラメータ
seed_episodes = 5  # 最初にランダム行動で探索するエピソード数
all_episodes = 100  # 学習全体のエピソード数（300ほどで, ある程度収束します）
test_interval = 10  # 何エピソードごとに探索ノイズなしのテストを行うか
model_save_interval = 20  # NNの重みを何エピソードごとに保存するか
collect_interval = 100  # 何回のNNの更新ごとに経験を集めるか（＝1エピソード経験を集めるごとに何回更新するか）

action_noise_var = 0.3  # 探索ノイズの強さ

batch_size = args.batch_size
chunk_length = args.chunk_length  # 1回の更新で用いる系列の長さ
imagination_horizon = 15  # Actor-Criticの更新のために, Dreamerで何ステップ先までの想像上の軌道を生成するか


gamma = 0.9  # 割引率
lambda_ = 0.95  # λ-returnのパラメータ
clip_grad_norm = 100  # gradient clippingの値
free_nats = 3  # KL誤差（RSSMのTransitionModelにおけるpriorとposteriorの間の誤差）がこの値以下の場合, 無視する
for episode in range(seed_episodes):
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.push(obs, action, reward, done)
        obs = next_obs

log_dir = 'logs'
writer = SummaryWriter(log_dir)

for episode in range(seed_episodes, all_episodes):
    # -----------------------------
    #      経験を集める
    # -----------------------------
    start = time.time()
    # 行動を決定するためのエージェントを宣言
    policy = Agent(encoder, rssm.transition, action_model)

    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = policy(obs)
        # 探索のためにガウス分布に従うノイズを加える(explaration noise)
        action += np.random.normal(0, np.sqrt(action_noise_var),
                                     env.action_space.shape[0])
        next_obs, reward, done, _ = env.step(action)
        
        #リプレイバッファに観測, 行動, 報酬, doneを格納
        replay_buffer.push(obs, action, reward, done)
        
        obs = next_obs
        total_reward += reward
    # 訓練時の報酬と経過時間をログとして表示
    writer.add_scalar('total reward at train', total_reward, episode)
    print('episode [%4d/%4d] is collected. Total reward is %f' %
            (episode+1, all_episodes, total_reward))
    print('elasped time for interaction: %.2fs' % (time.time() - start))

    # NNのパラメータを更新する
    start = time.time()
    for update_step in range(collect_interval):
        # -------------------------------------------------------------------------------------
        #  RSSM(trainsition_model, obs_model, reward_model)の更新 - Dynamics learning
        # -------------------------------------------------------------------------------------
        observations, actions, rewards, _ = \
            replay_buffer.sample(batch_size, chunk_length)
        # 観測を前処理し, RNNを用いたPyTorchでの学習のためにTensorの次元を調整
        observations = preprocess_obs(observations)
        observations = torch.as_tensor(observations, device=device)
        observations = observations.permute(1,0,4,2,3).contiguous()#(N,chunks,H,W,C) -> (chunks, N, C, H, W)
        actions = torch.as_tensor(actions, device=device).transpose(0, 1)
        rewards = torch.as_tensor(rewards, device=device).transpose(0, 1)

        # 観測をエンコーダで低次元のベクトルに変換
        embedded_observations = encoder(
            observations.view(-1, 3, 224, 224)).view(chunk_length, batch_size, -1)


        # 低次元の状態表現を保持しておくためのTensorを定義
        states = torch.zeros(chunk_length, batch_size, state_dim, device=device)
        rnn_hiddens = torch.zeros(chunk_length, batch_size, rnn_hidden_dim, device=device)

        # 低次元の状態表現は最初はゼロ初期化（timestep１つ分）
        state = torch.zeros(batch_size, state_dim, device=device)
        rnn_hidden = torch.zeros(batch_size, rnn_hidden_dim, device=device)

        # 状態s_tの予測を行ってそのロスを計算する（priorとposteriorの間のKLダイバージェンス）
        kl_loss = 0
        for l in range(chunk_length-1):
            next_state_prior, next_state_posterior, rnn_hidden = \
                rssm.transition(state, actions[l], rnn_hidden, embedded_observations[l+1])
            state = next_state_posterior.rsample()
            states[l+1] = state
            rnn_hiddens[l+1] = rnn_hidden
            kl = kl_divergence(next_state_prior, next_state_posterior).sum(dim=1)
            kl_loss += kl.clamp(min=free_nats).mean()  # 原論文通り, KL誤差がfree_nats以下の時は無視

        kl_loss /= (chunk_length - 1)

        # states[0] and rnn_hiddens[0]はゼロ初期化なので以降では使わない
        # states, rnn_hiddensは低次元の状態表現
        states = states[1:]
        rnn_hiddens = rnn_hiddens[1:]

        # 観測を再構成, また, 報酬を予測
        flatten_states = states.view(-1, state_dim)
        flatten_rnn_hiddens = rnn_hiddens.view(-1, rnn_hidden_dim)
        recon_observations = rasterizer(rssm.observation(flatten_states, flatten_rnn_hiddens).view(
            -1, num_strokes, num_control_points*2))
        predicted_rewards = rssm.reward(flatten_states, flatten_rnn_hiddens).view(chunk_length-1, batch_size, 1)

        # 観測と報酬の予測誤差を計算
        obs_loss = 0.5 * clip_loss(recon_observations.view(-1, 3, 224, 224), observations.view(-1, 3, 224, 224)[batch_size:])
        reward_loss = 0.5 * F.mse_loss(predicted_rewards, rewards[:-1])

        # 以上のロスを合わせて勾配降下で更新する
        model_loss = kl_loss + obs_loss + reward_loss
        model_optimizer.zero_grad()
        model_loss.backward()
        clip_grad_norm_(model_params, clip_grad_norm)
        model_optimizer.step()

        # --------------------------------------------------
        #  Action Model, Value　Modelの更新　- Behavior leaning
        # --------------------------------------------------
        # Actor-Criticのロスで他のモデルを更新することはないので勾配の流れを一度遮断
        # flatten_states, flatten_rnn_hiddensは RSSMから得られた低次元の状態表現を平坦化した値
        flatten_states = flatten_states.detach()
        flatten_rnn_hiddens = flatten_rnn_hiddens.detach()

        # DreamerにおけるActor-Criticの更新のために, 現在のモデルを用いた
        # 数ステップ先の未来の状態予測を保持するためのTensorを用意
        imaginated_states = torch.zeros(imagination_horizon + 1,
                                         *flatten_states.shape,
                                          device=flatten_states.device)
        imaginated_rnn_hiddens = torch.zeros(imagination_horizon + 1,
                                                *flatten_rnn_hiddens.shape,
                                                device=flatten_rnn_hiddens.device)

        #　未来予測をして想像上の軌道を作る前に, 最初の状態としては先ほどモデルの更新で使っていた
        # リプレイバッファからサンプルされた観測データを取り込んだ上で推論した状態表現を使う
        imaginated_states[0] = flatten_states
        imaginated_rnn_hiddens[0] = flatten_rnn_hiddens
        
        # open-loopで未来の状態予測を使い, 想像上の軌道を作る
        for h in range(1, imagination_horizon + 1):
            # 行動はActionModelで決定. この行動はモデルのパラメータに対して微分可能で,
            #　これを介してActionModelは更新される
            actions = action_model(flatten_states, flatten_rnn_hiddens)
            flatten_states_prior, flatten_rnn_hiddens = rssm.transition.prior(rssm.transition.reccurent(flatten_states,
                                                                   actions,
                                                                   flatten_rnn_hiddens))
            flatten_states = flatten_states_prior.rsample()
            imaginated_states[h] = flatten_states
            imaginated_rnn_hiddens[h] = flatten_rnn_hiddens

        # RSSMのreward_modelにより予測された架空の軌道に対する報酬を計算
        flatten_imaginated_states = imaginated_states.view(-1, state_dim)
        flatten_imaginated_rnn_hiddens = imaginated_rnn_hiddens.view(-1, rnn_hidden_dim)
        imaginated_rewards = \
            rssm.reward(flatten_imaginated_states,
                        flatten_imaginated_rnn_hiddens).view(imagination_horizon + 1, -1)
        imaginated_values = \
            value_model(flatten_imaginated_states,
                        flatten_imaginated_rnn_hiddens).view(imagination_horizon + 1, -1)
        # λ-returnのターゲットを計算(V_{\lambda}(s_{\tau})
        lambda_target_values = lambda_target(imaginated_rewards, imaginated_values, gamma, lambda_)

        # 価値関数の予測した価値が大きくなるようにActionModelを更新
        # PyTorchの基本は勾配降下だが, 今回は大きくしたいので-1をかける
        action_loss = -lambda_target_values.mean()
        action_optimizer.zero_grad()
        action_loss.backward()
        clip_grad_norm_(action_model.parameters(), clip_grad_norm)
        action_optimizer.step()
        # TD(λ)ベースの目的関数で価値関数を更新（価値関数のみを学習するため，学習しない変数のグラフは切っている. )
        imaginated_values = value_model(flatten_imaginated_states.detach(), flatten_imaginated_rnn_hiddens.detach()).view(imagination_horizon + 1, -1)        
        value_loss = 0.5 * F.mse_loss(imaginated_values, lambda_target_values.detach())
        value_optimizer.zero_grad()
        value_loss.backward()
        clip_grad_norm_(value_model.parameters(), clip_grad_norm)
        value_optimizer.step()
        # ログをTensorBoardに出力
        print('update_step: %3d model loss: %.5f, kl_loss: %.5f, '
             'obs_loss: %.5f, reward_loss: %.5f, '
             'value_loss: %.5f action_loss: %.5f'
                % (update_step + 1, model_loss.item(), kl_loss.item(),
                    obs_loss.item(), reward_loss.item(),
                    value_loss.item(), action_loss.item()))
        total_update_step = episode * collect_interval + update_step
        writer.add_scalar('model loss', model_loss.item(), total_update_step)
        writer.add_scalar('kl loss', kl_loss.item(), total_update_step)
        writer.add_scalar('obs loss', obs_loss.item(), total_update_step)
        writer.add_scalar('reward loss', reward_loss.item(), total_update_step)
        writer.add_scalar('value loss', value_loss.item(), total_update_step)
        writer.add_scalar('action loss', action_loss.item(), total_update_step)
    print('elasped time for update: %.2fs' % (time.time() - start))

    # --------------------------------------------------------------
    #    テストフェーズ. 探索ノイズなしでの性能を評価する
    # --------------------------------------------------------------
    if (episode + 1) % test_interval == 0:
        policy = Agent(encoder, rssm.transition, action_model)
        start = time.time()
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = policy(obs, training=False)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

        writer.add_scalar('total reward at test', total_reward, episode)
        print('Total test reward at episode [%4d/%4d] is %f' %
                (episode+1, all_episodes, total_reward))
        print('elasped time for test: %.2fs' % (time.time() - start))

    if (episode + 1) % model_save_interval == 0:
        # 定期的に学習済みモデルのパラメータを保存する
        model_log_dir = os.path.join(log_dir, 'episode_%04d' % (episode + 1))
        os.makedirs(model_log_dir)
        torch.save(encoder.state_dict(), os.path.join(model_log_dir, 'encoder.pth'))
        torch.save(rssm.transition.state_dict(), os.path.join(model_log_dir, 'rssm.pth'))
        torch.save(rssm.observation.state_dict(), os.path.join(model_log_dir, 'obs_model.pth'))
        torch.save(rssm.reward.state_dict(), os.path.join(model_log_dir, 'reward_model.pth'))
        torch.save(value_model.state_dict(), os.path.join(model_log_dir, 'value_model.pth'))
        torch.save(action_model.state_dict(), os.path.join(model_log_dir, 'action_model.pth'))

writer.close()
