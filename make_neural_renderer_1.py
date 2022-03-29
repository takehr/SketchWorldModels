import math 
import copy
import os 
import json 
import time 
import datetime
import glob 
import natsort 
import cv2
import matplotlib.pyplot as plt 
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torchinfo import summary



# ファイル関連の定数
FILE_NAME = os.path.basename(__file__)
FILE_NAME_WITHOUT_DOT = os.path.splitext(FILE_NAME)[0]
FILE_DIR_NAME = os.path.dirname(__file__)
STR_NOW = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ディレクトリ作成
log_dir = "./log_{}/log_{}".format(FILE_NAME_WITHOUT_DOT, STR_NOW)
os.makedirs(log_dir, exist_ok=True)

parser = ArgumentParser(allow_abbrev=False)
parser.add_argument("-t", "--TOTAL_TRAINING_STEPS", default=100, type=int)
parser.add_argument("--RADIUS", default=5, type=int)
parser.add_argument("--NUM_T_SPLIT", default=500, type=int)
parser.add_argument("--NUM_CONTROL_POINTS", default=4, type=int)
parser.add_argument("--NUM_STROKES", default=5, type=int)
parser.add_argument("--DEVICE", default=None, type=str)
args, unknown_args = parser.parse_known_args()
with open("{}/args.json".format(log_dir), mode="w") as f:
    json.dump(args.__dict__, f, indent=4)

    
# mainの定数
CANVAS_SIZE = 224
CANVAS_H = CANVAS_SIZE
CANVAS_W = CANVAS_SIZE
CANVAS_C = 3
NUM_CONTROL_POINTS = args.NUM_CONTROL_POINTS
NUM_STROKES = args.NUM_STROKES
BATCH_SIZE = 256
TOTAL_TRAINING_STEPS = args.TOTAL_TRAINING_STEPS


# 1ステップ分、円を描く
def draw_circle_step(img_np, tmp_x, tmp_y, tmp_r, tmp_color_array):
    ret_img_np = cv2.circle(img_np, (tmp_x, tmp_y), radius=tmp_r, color=tmp_color_array, thickness=-1)
    return ret_img_np

# pltで表示。
def show_img(img_np):
    plt.imshow(img_np, vmin=0, vmax=255)
def show_img_gray(img_np):
    plt.imshow(img_np, cmap="gray", vmin=0, vmax=255)

# <<<Bezier curve>>>
# n_C_k
def binomial_coefficient(n, k):
    assert n >= k, "n must be larger than k"
    return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))

# B_{n,i}(t), tは媒介変数
def bernstein_basis_function(n, i, t):
    return binomial_coefficient(n, i) * np.power(t, i) * np.power((1 - t), n - i)
    
# tにおけるベジェ曲線のある一点
# list_control_points: control_pointsのタプルのリスト
# num_control_points: 制御点の数．この数引く１がベジェ曲線の次数．
def bernstein_polynomials(list_control_points, t):
    num_control_points = len(list_control_points)
    # print("num_control_points = {}".format(num_control_points))
    tmp_x = 0
    tmp_y = 0
    for i in range(num_control_points):
        tmp_x += bernstein_basis_function(num_control_points - 1, i, t) * list_control_points[i][0]
        tmp_y += bernstein_basis_function(num_control_points - 1, i, t) * list_control_points[i][1]
    return tmp_x, tmp_y
    
# len(list_control_points)-1次のベジェ曲線を描画する関数．
# list_control_points: 制御点のリスト    
# num_t_split: 曲線の描画処理を何ステップ分割するか
def draw_bezier_curve(tmp_canvas_np_cv, list_control_points, num_t_split, color=(0,0,0), radius=5):
    for i in range(num_t_split):
        t = i / num_t_split
        tmp_x, tmp_y = bernstein_polynomials(list_control_points, t)
        tmp_canvas_np_cv = draw_circle_step(tmp_canvas_np_cv, int(tmp_x), int(tmp_y), radius, color)
    return tmp_canvas_np_cv


# neural rendererの定義
class NeuralRenderer(nn.Module):
    def __init__(self, num_control_points):
        super().__init__()
        self.fc1 = nn.Linear(num_control_points*2, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 4096)
        self.conv1 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(8, 49, 3, 1, 1)
        self.conv4 = nn.Conv2d(49, 49*3, 3, 1, 1)
        # self.conv5 = nn.Conv2d(4, 8, 3, 1, 1)
        # self.conv6 = nn.Conv2d(8, 4, 3, 1, 1)
        self.pixel_shuffle1 = nn.PixelShuffle(2)
        self.pixel_shuffle2 = nn.PixelShuffle(7)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle1(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle2(self.conv4(x))
        # x = F.relu(self.conv5(x))
        # x = self.pixel_shuffle(self.conv6(x))
        x = torch.sigmoid(x)
        return x.view(-1, CANVAS_W, CANVAS_H, CANVAS_C)

# 描画してみる
image_bezier_np_cv = np.ones((CANVAS_H, CANVAS_W, CANVAS_C), dtype=np.uint8)*255 #  OpenCV形式の画像データ。つまり、BGRの順
for j in range(NUM_STROKES):
    list_cp_x = np.random.randint(0, CANVAS_W+1, (NUM_CONTROL_POINTS,))
    list_cp_y = np.random.randint(0, CANVAS_H+1, (NUM_CONTROL_POINTS,))
    list_cp_for_bezier = [(list_cp_x[i], list_cp_y[i]) for i in range(NUM_CONTROL_POINTS)]

    image_bezier_np_cv = draw_bezier_curve(image_bezier_np_cv, list_cp_for_bezier, num_t_split=args.NUM_T_SPLIT, radius=args.RADIUS)
    # show_img(tmp_canvas_np_cv)
    # plt.show()

    # normalized_tmp_canvas_np_cv = tmp_canvas_np_cv / 255.0
    # normalized_tmp_canvas_np_cv = normalized_tmp_canvas_np_cv.astype(np.float32)
    # show_img(normalized_tmp_canvas_np_cv) # 0~1の範囲でも描画してくれる．
    # plt.show() 
# print(image_bezier_np_cv[150, :, 0])
# print(image_bezier_np_cv.dtype)
image_bezier_np_plt = cv2.cvtColor(image_bezier_np_cv, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
plt.imsave(log_dir+"/bezier_curve_try.jpg", image_bezier_np_plt)




# 訓練部分
# image_bezier_np_cvでbezier curveを描画する．np.uint8, 0~255, BGR
# image_bezier_np_pltをtarget_imageとしてbatch_imagesに格納する．np.float32, 0~1, RGB
criterion = nn.MSELoss()
neural_renderer = NeuralRenderer(NUM_CONTROL_POINTS)
summary(neural_renderer, input_size=(BATCH_SIZE,NUM_CONTROL_POINTS*2), col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1)
optimizer = torch.optim.Adam(neural_renderer.parameters(), lr=3e-6)
batch_size = BATCH_SIZE 
if args.DEVICE is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device = {}".format(device))
else:
    device = torch.device(args.DEVICE)
    print("device is set as {}.".format(device))
if device.type == "cuda": # device == "cuda" では，Trueになりえないので注意．deviceはstrではない．
    neural_renderer.cuda()

val_loss_min = np.inf
cnt = 0
cnt_increse_agein = 0 # loss_minよりもlossが上回ったエピソードの回数
total_training_steps = TOTAL_TRAINING_STEPS 
time_at_start = time.time()

while cnt <= total_training_steps:
    neural_renderer.train() 

    batch_points = [] # 入力データ
    batch_images = [] # ラベルデータ
    for i in range(batch_size):
        list_cp_x = np.random.randint(0, CANVAS_W+1, (NUM_CONTROL_POINTS,))
        list_cp_y = np.random.randint(0, CANVAS_H+1, (NUM_CONTROL_POINTS,))
        list_cp = []
        for i in range(NUM_CONTROL_POINTS):
            list_cp.append(list_cp_x[i]/CANVAS_W)
            list_cp.append(list_cp_y[i]/CANVAS_H)
        list_cp_for_bezier = [(list_cp_x[i], list_cp_y[i]) for i in range(NUM_CONTROL_POINTS)]
        list_cp = np.array(list_cp, dtype=np.float32)
        batch_points.append(list_cp)

        image_bezier_np_cv = np.ones((CANVAS_H, CANVAS_W, CANVAS_C), dtype=np.uint8)*255 #  OpenCV形式の画像データ。つまり、BGRの順
        image_bezier_np_cv = draw_bezier_curve(image_bezier_np_cv, list_cp_for_bezier, num_t_split=args.NUM_T_SPLIT, color=(0,0,0), radius=args.RADIUS)
        image_bezier_np_plt = cv2.cvtColor(image_bezier_np_cv, cv2.COLOR_BGR2RGB)
        image_bezier_np_plt = image_bezier_np_plt.astype(np.float32) / 255.0
        batch_images.append(image_bezier_np_plt)

    batch_points = torch.tensor(np.array(batch_points)).float()
    batch_images = torch.tensor(np.array(batch_images)).float()

    if device.type == "cuda":
        neural_renderer.cuda() # nn.Moduleに対するcuda()はin-place処理
        batch_points = batch_points.cuda() # tensorに対するcuda()はin-placeではない
        batch_images = batch_images.cuda()
    # print("neural_renderer.device = {}".format(neural_renderer.device)) # netにはdevice属性はない\
    # print("batch_points.device = {}".format(batch_points.device))
    # print("batch_images.device = {}".format(batch_images.device))

    pred_images = neural_renderer(batch_points) # ここに.cuda()はいらないらしい．
    optimizer.zero_grad()
    loss = criterion(pred_images, batch_images)
    loss.backward()
    optimizer.step()

    # 学習率の調整
    if cnt < 50000:
        lr = 1e-4
    elif cnt < 100000:
        lr = 1e-5
    else:
        lr = 1e-6
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr 
    
    # validation
    if cnt % 500 == 0:
        tmp_time = time.time()
        elapsed_time = tmp_time - time_at_start
        neural_renderer.eval()
        with torch.no_grad():
            pred_images = neural_renderer(batch_points)
            val_loss = criterion(pred_images, batch_images)
        print("time = {}: cnt = {}, val_loss = {}, val_loss per sample = {}".format(elapsed_time, cnt, val_loss, val_loss/batch_size))
        target_image = batch_images[0].cpu().detach().numpy().copy()
        pred_image = pred_images[0].cpu().detach().numpy().copy()
        plt.imsave(log_dir+"/target_image_cnt={}.jpg".format(cnt), target_image)
        plt.imsave(log_dir+"/pred_image_cnt={}.jpg".format(cnt), pred_image)
        if val_loss < val_loss_min:
            cnt_increse_agein = 0
            val_loss_min = val_loss
            if device.type == "cuda":
                neural_renderer.cpu()
            torch.save(neural_renderer.state_dict(), log_dir+"/neural_renderer_cnt={}_valLoss={}.pkl".format(cnt, val_loss))
            if device.type == "cuda":
                neural_renderer.cuda()
        if val_loss >= val_loss_min:
            cnt_increse_agein += 1 
        if cnt_increse_agein >= 1000: 
            print("training finished.")
            break 
                
    cnt += 1
    
# 訓練結果の可視化


# 訓練済みモデルをロードして，NUM_STROKES分回し，所望のスケッチ画像を得る
# TODO: neural_rendererとしてインスタンス１つ分しか使用していない．勾配が機能するのか？
del(neural_renderer)
path_neural_renderer_state_dict = natsort.natsorted(glob.glob(log_dir+"/neural_renderer_cnt=*.pkl"))[-1]
print("path_neural_renderer_state_dict = {}".format(path_neural_renderer_state_dict))
neural_renderer = NeuralRenderer(NUM_CONTROL_POINTS)
neural_renderer.load_state_dict(torch.load(path_neural_renderer_state_dict))
neural_renderer.eval() 

target_image_np_cv = np.ones((CANVAS_H, CANVAS_W, CANVAS_C), dtype=np.uint8)*255 # bezier描画関数による描画用．
dict_pred_images = {} # neural_renderer描画用．

for j in range(NUM_STROKES):
    list_cp_x = np.random.randint(0, CANVAS_W+1, (NUM_CONTROL_POINTS,))
    list_cp_y = np.random.randint(0, CANVAS_H+1, (NUM_CONTROL_POINTS,))
    # bezier 描画用
    list_cp_for_bezier = [(list_cp_x[i], list_cp_y[i]) for i in range(NUM_CONTROL_POINTS)]
    target_image_np_cv = draw_bezier_curve(target_image_np_cv, list_cp_for_bezier, num_t_split=args.NUM_T_SPLIT, radius=args.RADIUS)
    # neural_rendererによる描画用
    list_cp = []
    for i in range(NUM_CONTROL_POINTS):
        list_cp.append(list_cp_x[i]/CANVAS_W)
        list_cp.append(list_cp_y[i]/CANVAS_H)
    list_cp = np.array(list_cp, dtype=np.float32)
    dict_pred_images["stroke_{}".format(j)] = neural_renderer(torch.tensor(np.array([list_cp,])).float())

# neural_rendererによる複数ストロークの描画結果
tmp_sum_of_strokes = torch.tensor(np.zeros((1, CANVAS_W, CANVAS_H, CANVAS_C), dtype=np.float32))
for j in range(NUM_STROKES):
    print("stroke_{}: {}".format(j, dict_pred_images["stroke_{}".format(j)][0, 150, :3, 0]))
    tmp_sum_of_strokes += (1 - dict_pred_images["stroke_{}".format(j)]) # ストローク部分を抽出し，足し合わせるために，0~1を反転させ，blackを1とする．  
    print("tmp_sum_of_strokes = {}".format(tmp_sum_of_strokes.detach().numpy().copy()[0, 150, :3, 0]))
tmp_sum_of_strokes = torch.sigmoid((tmp_sum_of_strokes - 0.5) * 2)
print("tmp_sum_of_strokes = {}".format(tmp_sum_of_strokes.detach().numpy().copy()[0, 150, :3, 0]))
pred_image_sum_of_strokes = (1- tmp_sum_of_strokes).detach().numpy().copy()
print("pred_image_sum_of_strokes.shape = {}".format(pred_image_sum_of_strokes.shape))
# show_img(image_sum_of_strokes.detach().numpy().copy()[0])
# plt.show()
plt.imsave(log_dir+"/pred_image_sum_of_strokes.jpg", pred_image_sum_of_strokes[0])

# bezier描画結果
# print(target_image_np_cv[150, :, 0])
# print(target_image_np_cv.dtype)
target_image_np_plt = cv2.cvtColor(target_image_np_cv, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
plt.imsave(log_dir+"/target_image_sum_of_strokes.jpg", target_image_np_plt)
