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

import argparse
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("dark_background")
plt.rcParams["font.size"] = 12

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

    line = []
    line = [
      [initial_pose_x[:], ndt_pose_x[:]],
      [initial_pose_y[:], ndt_pose_y[:]],
    ]

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_aspect("equal", adjustable="box")
    ax1.scatter(initial_pose_x, initial_pose_y, c="b", label="particle pose")
    ax1.scatter(ndt_pose_x, ndt_pose_y, c="r", label="ndt pose")
    ax1.plot([initial_pose_x[:], ndt_pose_x[:]], [initial_pose_y[:], ndt_pose_y[:]], linestyle="dashed", c="g", linewidth=0.5)
    ax1.legend()

    ax2.set_aspect("equal", adjustable="box")
    ax2.scatter(ndt_pose_x, ndt_pose_y, c="r", label="ndt pose")
    plt.show()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input")
  args = parser.parse_args()

  parse_csv = ParseCsv(args.input)
