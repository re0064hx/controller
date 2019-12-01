#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import scipy.integrate as sp
import numpy as np
import random

# SMC制御系
def system(t, x):
    q = 5          # 定常項ゲインq
    k = 1          # 比例項のゲインk
    ka, a = 5, 0.8 # 加速率到達則のパラメータ 
    s = [5.0, 1.0] # 切換超平面S

    # 戻り値用のリスト
    y = [0]*2

    # 切換関数の計算
    sigma = s[0]*x[0] + s[0]*x[1]

    # 制御入力u
    # u = -(q * np.sign(sigma) - x[0] + 4*x[1]) # 定常到達則
    u = -(q * np.sign(sigma) + k*sigma - x[0] + 4*x[1]) # 比例到達則
    #u = -(k * (np.abs(sigma)**a) * np.sign(sigma) - x[0] + 4*x[1]) # 加速率到達則
    
    # 外乱値を乱数で生成(値域は-1～1)
    d = random.uniform(-1, 1)

    # 常微分方程式（状態方程式）の計算
    dx1 = x[1]                     
    dx2 = -x[0] -x[1] + u + d
    
    # 計算結果を返す
    y[0] = dx1
    y[1] = dx2
    
    return y

def simulation(x0, end, step):
    x1 = []
    x2 = []
    t = []
    ode =  sp.ode(system)
    ode.set_integrator('dopri5', method='bdf', atol=1.e-2)
    ode.set_initial_value(x0, 0)
    t.append(0)
    x1.append(x0[0])
    x2.append(x0[1])

    while ode.successful() and ode.t < end - step:
        ode.integrate(ode.t + step)
        t.append(ode.t)
        x1.append(ode.y[0])
        x2.append(ode.y[1])

    return x1, x2, t

def draw_graph(x1, x2, t):
    plt.figure()
    # 左
    plt.subplot(121)
    plt.plot(t, x1, label='x_1')
    plt.plot(t, x2, label='x_2')
    plt.xlabel("Time")
    plt.ylabel("State variable")
    plt.legend()# 凡例表示
    plt.grid()  # グリッド表示

    # 右
    plt.subplot(122)
    plt.plot(x1, x2, label='x')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()# 凡例表示
    plt.grid()  # グリッド表示

    plt.show()


def main():
    # パラメータ
    x0 = [3.0, 1.0] # 初期値
    end = 10        # シミュレーション時間
    step = 0.01     # 時間の刻み幅

    # シミュレーション
    x1, x2, t = simulation(x0, end, step)
    
    # 結果をグラフ表示
    draw_graph(x1, x2, t)

    
if __name__ == "__main__":
    main()