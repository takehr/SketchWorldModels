import numpy as np
import gym
import pybullet_envs  # PyBulletの環境をgymに登録する
class GymWrapper_PyBullet(object):
    """
    PyBullet環境のためのラッパー
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    reward_range = (-np.inf, np.inf)

    # __init__でカメラ位置に関するパラメータ( cam_dist:カメラ距離, cam_yaw：カメラの水平面での回転, cam_pitch:カメラの縦方向での回転)を受け取り, カメラの位置を調整できるようにします.
    #　同時に画像の大きさも変更できるようにします
    def __init__(self, env, cam_dist=3, cam_yaw=0, cam_pitch=-30, render_width=320, render_height=240):
        self._env = env
        self._env.env._cam_dist = cam_dist
        self._env.env._cam_yaw = cam_yaw
        self._env.env._cam_pitch = cam_pitch
        self._env.env._render_width = render_width
        self._env.env._render_height = render_height

    def __getattr(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        width = self._env.env._render_width
        height = self._env.env._render_height
        return gym.spaces.Box(0, 255, (height, width, 3), dtype=np.uint8)

    @property
    def action_space(self):
        return self._env.action_space

    #　元の観測（低次元の状態）は今回は捨てて, env.render()で取得した画像を観測とします.
    #  画像, 報酬, 終了シグナルが得られます.
    def step(self, action):
        _, reward, done, info = self._env.step(action)
        obs = self._env.render(mode="rgb_array")
        return obs, reward, done, info

    def reset(self):
        self._env.reset()
        obs = self._env.render(mode="rgb_array")
        return obs

    def render(self, mode='human', **kwargs):
        return self._env.render(mode, **kwargs)

    def close(self):
        self._env.close()
        env = gym.make('HalfCheetahBulletEnv-v0')
class RepeatAction(gym.Wrapper):
    """
    同じ行動を指定された回数自動的に繰り返すラッパー. 観測は最後の行動に対応するものになる
    """
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self._skip = skip

    def reset(self):
        return self.env.reset()

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
def make_env():
    env = gym.make('HalfCheetahBulletEnv-v0')  # 環境を読み込む. 今回はHalfCheetah
    # Dreamerでは観測は64x64のRGB画像
    env = GymWrapper_PyBullet(env, cam_dist=2, cam_pitch=0, render_width=224, render_height=224)
    env = RepeatAction(env, skip=2)  # DreamerではActionRepeatは2
    return env
