#!/usr/bin/env python3

# Copyright 2023 TIER IV, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ing_theme_matplotlib import mpl_style
from pandas.core.frame import nv
from scipy.sparse import data
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as R

mpl_style(dark=True)
#plt.style.use("dark_background")
plt.rcParams["font.size"] = 12

ground_truth_pose = [49806.28958228331, 36616.997065172305, 106.44803576807921]
ground_truth_orientation = [0.005357401930847109, 0.0008296983102710916, -0.1469357742378811, 0.9891311784057661]
#ground_truth_pose = [49806.83115361129, 36616.710762250455, 106.54302514911168]
#ground_truth_orientation = [0.007844379324393247, -0.001474780089717242, -0.14283532811368882, 0.9897142818911713]
#ground_truth_pose = [61388.6129952297, 56179.59135545102, -7.544952608349622]
#ground_truth_orientation = [-0.0020080550268571805, -0.012445017151604279, 0.8875133664335579, 0.460609502360599]

rotation = R.from_quat(ground_truth_orientation)
euler_angles = rotation.as_euler('xyz', degrees=False)
print(euler_angles)

def draw_circle(radius: float, ax, color, linestyle, label, alpha, origin) -> None:
    theta = np.linspace(0, 2 * np.pi, 100)
    x = radius * np.cos(theta) + origin[0]
    y = radius * np.sin(theta) + origin[1]
    ax.plot(x, y, c=color, linestyle=linestyle, label=label, alpha=alpha)

class ParseCsv:
  def __init__(self, input_file):
    df = pd.read_csv(input_file)
    data_size = len(df)

    initial_pose_x  = df['initial_pose.x'].values
    initial_pose_y  = df['initial_pose.y'].values
    initial_pose_yaw = df['initial_pose.yaw'].values
    ndt_pose_x  = df['ndt_pose.x'].values
    ndt_pose_y  = df['ndt_pose.y'].values
    ndt_pose_yaw  = df['ndt_pose.yaw'].values
    tp = df['tp'].values
    nvtl = df['nvtl'].values

    fig = plt.figure(figsize=(18, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    #ax1.set_aspect("equal", adjustable="box")
    #ax1.scatter(initial_pose_x, initial_pose_y, c="b", label="particle pose")
    #ax1.scatter(ndt_pose_x, ndt_pose_y, c="r", label="ndt pose")
    #ax1.plot([initial_pose_x[:], ndt_pose_x[:]], [initial_pose_y[:], ndt_pose_y[:]], linestyle="dashed", c="g", linewidth=0.5)
    #ax1.legend()

    centroid_x = 0.0
    centroid_y = 0.0
    for i in range(data_size):
      centroid_x = centroid_x + ndt_pose_x[i]
      centroid_y = centroid_y + ndt_pose_y[i]
    centroid_x = centroid_x / data_size
    centroid_y = centroid_y / data_size

    in_tp_pos = []
    for i in range(data_size):
      if 3.0 < tp[i]:
      #if 2.3 < nvtl[i]:
        in_tp_pos.append([ndt_pose_x[i], ndt_pose_y[i], ndt_pose_yaw[i], tp[i]])
    in_tp_pos = np.array(in_tp_pos)

    in_nvtl_pos = []
    for i in range(len(ndt_pose_x)):
      #if 3.0 < tp[i]:
      if 2.3 < nvtl[i]:
        in_nvtl_pos.append([ndt_pose_x[i], ndt_pose_y[i], ndt_pose_yaw[i], nvtl[i]])
    in_nvtl_pos = np.array(in_nvtl_pos)

    tp_distances = [math.hypot(pos[0] - centroid_x, pos[1] - centroid_y) for pos in in_tp_pos]
    nvtl_distances = [math.hypot(pos[0] - centroid_x, pos[1] - centroid_y) for pos in in_nvtl_pos]

    ax1.set_aspect("equal", adjustable="box")
    ax1.scatter(ground_truth_pose[0], ground_truth_pose[1], s=200,c="w", marker="*", label="ground truth")
    sc1 = ax1.scatter(ndt_pose_x, ndt_pose_y, alpha=0.5, vmin=min(nvtl), vmax=max(nvtl), c=nvtl, cmap=cm.jet, label="ndt pose")
    fig.colorbar(sc1, ax=ax1)
    if len(nvtl_distances) != 0:
      draw_circle(max(nvtl_distances), ax1, color="cyan", linestyle="dashed", label=None, alpha=0.0, origin=(centroid_x, centroid_y))
    ax1.legend()
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title("Particle with NVTL")

    ax2.set_aspect("equal", adjustable="box")
    end_x = ground_truth_pose[0] + 0.1 * np.cos(euler_angles[2])
    end_y = ground_truth_pose[1] + 0.1 * np.sin(euler_angles[2])
    ax2.scatter(ground_truth_pose[0], ground_truth_pose[1], s=200, marker="*", color="white", label="ground truth")
    #ax2.quiver(ground_truth_pose[0], ground_truth_pose[1], end_x-ground_truth_pose[0], end_y-ground_truth_pose[1], color='white', angles='xy', scale_units='xy', scale=1)
    if len(nvtl_distances) != 0:
      best_pose = []
      min_score = -1.0
      for i in range(len(in_nvtl_pos)):
        if min_score < in_nvtl_pos[i, 3]:
          best_pose = in_nvtl_pos[i, :]
          min_score = in_nvtl_pos[i, 3]
      ax2.plot([ground_truth_pose[0], best_pose[0]], [ground_truth_pose[1], best_pose[1]], c="g", linewidth=1.0)
      draw_circle(max(nvtl_distances), ax2, color="cyan", linestyle="dashed", label=None, alpha=0.0, origin=(centroid_x, centroid_y))
      length = 0.25
      start_x_list = []
      start_y_list = []
      end_x_list = []
      end_y_list = []
      for i in range(len(in_nvtl_pos)):
        start_x_list.append(in_nvtl_pos[i, 0])
        start_y_list.append(in_nvtl_pos[i, 1])
        end_x_list.append((in_nvtl_pos[i, 0] + length * np.cos(in_nvtl_pos[i, 2]) - in_nvtl_pos[i, 0]))
        end_y_list.append((in_nvtl_pos[i, 1] + length * np.sin(in_nvtl_pos[i, 2]) - in_nvtl_pos[i, 1]))

      #ax2.plot(in_nvtl_pos[:, 0], in_nvtl_pos[:, 1], c="w", marker="o")
      #ax2.quiver(in_nvtl_pos[i, 0], in_nvtl_pos[i, 1], end_x-in_nvtl_pos[i, 0], end_y-in_nvtl_pos[i, 1], color='red', angles='xy', scale_units='xy', scale=1)
      #ax2.arrow(in_nvtl_pos[i, 0], in_nvtl_pos[i, 1], end_x - in_nvtl_pos[i, 0], end_y - in_nvtl_pos[i, 1], head_width=width, head_length=length, fc="white", ec='black')
    #ax2.scatter(in_nvtl_pos[:, 0], in_nvtl_pos[:, 1], s=100, c="w", marker="o")
      quiv2 = ax2.quiver(start_x_list, start_y_list, end_x_list, end_y_list, in_nvtl_pos[:, 3], cmap=cm.jet, color='red', angles='xy', scale_units='xy', scale=1, label="Position above NVTL Threshold")
    #ax2.set_aspect("equal", adjustable="box")
    #ax2.scatter(ground_truth_pose[0], ground_truth_pose[1], s=200,c="w", marker="*", label="ground truth")
    #ax2.scatter(in_tp_pos[:, 0], in_tp_pos[:, 1], s=100,c="w", marker="x")
    #sc2 = ax2.scatter(ndt_pose_x, ndt_pose_y, alpha=0.5, vmin=min(tp), vmax=max(tp), c=tp, cmap=cm.jet, label="ndt pose")
      fig.colorbar(quiv2, ax=ax2)
      error = math.hypot(ground_truth_pose[0] - best_pose[0], ground_truth_pose[1] - best_pose[1])
      ax2.set_title("Initial Position Candidate above NVTL Threshold \n NVTL: {} \n Error: {} m".format(best_pose[3], error))
    else:
      ax2.set_title("Pose Initialization is Failed")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.legend(loc='upper right')
    fig.savefig("nvtl.png")

    fig1 = plt.figure(figsize=(18, 10))
    ax3 = fig1.add_subplot(121)
    ax4 = fig1.add_subplot(122)

    #ax1.set_aspect("equal", adjustable="box")
    #ax1.scatter(initial_pose_x, initial_pose_y, c="b", label="particle pose")
    #ax1.scatter(ndt_pose_x, ndt_pose_y, c="r", label="ndt pose")
    #ax1.plot([initial_pose_x[:], ndt_pose_x[:]], [initial_pose_y[:], ndt_pose_y[:]], linestyle="dashed", c="g", linewidth=0.5)
    #ax1.legend()

    ax3.set_aspect("equal", adjustable="box")
    ax3.scatter(ground_truth_pose[0], ground_truth_pose[1], s=200,c="w", marker="*", label="ground truth")
    sc1 = ax3.scatter(ndt_pose_x, ndt_pose_y, alpha=0.5, vmin=min(tp), vmax=max(tp), c=tp, cmap=cm.jet, label="ndt pose")
    fig1.colorbar(sc1, ax=ax3)
    draw_circle(max(tp_distances), ax3, color="cyan", linestyle="dashed", label=None, alpha=0.0, origin=(centroid_x, centroid_y))
    ax3.legend()
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("y [m]")
    ax3.set_title("Particle with TP")

    ax4.set_aspect("equal", adjustable="box")
    #end_x = ground_truth_pose[0] + 0.1 * np.cos()
    #end_y = ground_truth_pose[1] + 0.1 * np.sin()
    best_pose = []
    ax4.scatter(ground_truth_pose[0], ground_truth_pose[1], s=200, marker="*", color="white", label="ground truth")
    #ax4.quiver(ground_truth_pose[0], ground_truth_pose[1], end_x-ground_truth_pose[0], end_y - ground_truth_pose[1], color='white', angles='xy', scale_units='xy', scale=1)
    if len(tp_distances) != 0:
      min_score = -1.0
      for i in range(len(in_tp_pos)):
        if min_score < in_tp_pos[i, 3]:
          best_pose = in_tp_pos[i, :]
          min_score = in_tp_pos[i, 3]
      ax4.plot([ground_truth_pose[0], best_pose[0]], [ground_truth_pose[1], best_pose[1]], c="g", linewidth=1.0)

      draw_circle(max(tp_distances), ax4, color="cyan", linestyle="dashed", label=None, alpha=0.0, origin=(centroid_x, centroid_y))
      length = 0.25
      start_x_list = []
      start_y_list = []
      end_x_list = []
      end_y_list = []
      for i in range(len(in_tp_pos)):
        start_x_list.append(in_tp_pos[i, 0])
        start_y_list.append(in_tp_pos[i, 1])
        end_x_list.append((in_tp_pos[i, 0] + length * np.cos(in_tp_pos[i, 2]) - in_tp_pos[i, 0]))
        end_y_list.append((in_tp_pos[i, 1] + length * np.sin(in_tp_pos[i, 2]) - in_tp_pos[i, 1]))
        #ax2.plot(in_nvtl_pos[:, 0], in_nvtl_pos[:, 1], c="w", marker="o")
        #ax4.quiver(in_tp_pos[i, 0], in_tp_pos[i, 1], end_x-in_tp_pos[i, 0], end_y-in_tp_pos[i, 1], color='red', angles='xy', scale_units='xy', scale=1)
      quiv4 = ax4.quiver(start_x_list, start_y_list, end_x_list, end_y_list, in_tp_pos[:, 3], cmap=cm.jet, color='red', angles='xy', scale_units='xy', scale=1, label="Position above TP Threshold")
        #ax2.arrow(in_nvtl_pos[i, 0], in_nvtl_pos[i, 1], end_x - in_nvtl_pos[i, 0], end_y - in_nvtl_pos[i, 1], head_width=width, head_length=length, fc="white", ec='black')
      #ax2.scatter(in_nvtl_pos[:, 0], in_nvtl_pos[:, 1], s=100, c="w", marker="o")
      #ax2.set_aspect("equal", adjustable="box")
      #ax2.scatter(ground_truth_pose[0], ground_truth_pose[1], s=200,c="w", marker="*", label="ground truth")
      #ax2.scatter(in_tp_pos[:, 0], in_tp_pos[:, 1], s=100,c="w", marker="x")
      #sc2 = ax2.scatter(ndt_pose_x, ndt_pose_y, alpha=0.5, vmin=min(tp), vmax=max(tp), c=tp, cmap=cm.jet, label="ndt pose")
      fig.colorbar(quiv4, ax=ax4)
      error = math.hypot(ground_truth_pose[0] - best_pose[0], ground_truth_pose[1] - best_pose[1])
      ax4.set_title("Initial Position Candidate above TP Threshold \n TP: {} \n Error: {} m".format(best_pose[3], error))
    else:
      ax4.set_title("Pose Initialization is Failed")
    ax4.legend(loc='upper right')
    ax4.set_xlabel("x [m]")
    ax4.set_ylabel("y [m]")
    fig1.savefig("tp.png")

    fig2 = plt.figure(figsize=(15, 15))
    ax5 = fig2.add_subplot(111)

    ax5.set_aspect("equal", adjustable="box")
    ax5.scatter(ground_truth_pose[0], ground_truth_pose[1], s=500,c="w", marker="*", label="ground truth")
    ax5.scatter(best_pose[0], best_pose[1], s=500,c="m", marker="*", label="convergence pose")
    
    start_x_list = []
    start_y_list = []
    end_x_list = []
    end_y_list = []
    p_start_x_list = []
    p_start_y_list = []
    p_end_x_list = []
    p_end_y_list = []
    length = 0.1
    for i in range(data_size):
      start_x_list.append(ndt_pose_x[i])
      start_y_list.append(ndt_pose_y[i])
      end_x_list.append((ndt_pose_x[i] + length * np.cos(ndt_pose_yaw[i]) - ndt_pose_x[i]))
      end_y_list.append((ndt_pose_y[i] + length * np.sin(ndt_pose_yaw[i]) - ndt_pose_y[i]))
      p_start_x_list.append(initial_pose_x[i])
      p_start_y_list.append(initial_pose_y[i])
      p_end_x_list.append((initial_pose_x[i] + length * np.cos(initial_pose_yaw[i]) - initial_pose_x[i]))
      p_end_y_list.append((initial_pose_y[i] + length * np.sin(initial_pose_yaw[i]) - initial_pose_y[i]))

    quiv5 = ax5.quiver(start_x_list, start_y_list, end_x_list, end_y_list, color='red', angles='xy', scale_units='xy', scale=1, label="ndt pose")
    quiv6 = ax5.quiver(p_start_x_list, p_start_y_list, p_end_x_list, p_end_y_list, color='blue', angles='xy', scale_units='xy', scale=1, label="particle pose")
    #ax5.scatter(initial_pose_x, initial_pose_y, c="b", label="particle pose")
    #ax5.scatter(ndt_pose_x, ndt_pose_y, c="r", label="ndt pose")
    ax5.plot([initial_pose_x[:], ndt_pose_x[:]], [initial_pose_y[:], ndt_pose_y[:]], linestyle="dashed", c="g", linewidth=0.5)
    ax5.legend()
    ax5.set_xlabel("x [m]")
    ax5.set_ylabel("y [m]")
    ax5.set_title("Initial pose and Convergence Pose")
    fig2.savefig("particle.png")


    plt.show()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input")
  args = parser.parse_args()

  parse_csv = ParseCsv(args.input)
