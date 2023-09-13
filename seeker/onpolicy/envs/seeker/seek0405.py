# -*- coding:utf-8  -*-
# rjq 主要改动：增加stay动作、修改状态、增加弹回、得到有效动作、myagent动作mask选择、修改bug、
# rjq 渲染加上陷阱[后续可以改成掉入陷阱奖励﹣inf]；
from .gridgame import GridGame
from .observation import *
from PIL import ImageDraw, Image
import numpy as np
import operator

#import cv2
import time
from .get_logger import get_logger

import os, sys
import json
import pdb
abs_path = os.path.abspath(__file__)
dirname, filename = os.path.split(abs_path)

### player position
player_position = [
    [[0, 0], [0, 0]], # 0
    [[0, 0], [0, 0]], # 1
    [[0, 0], [0, 1]], # 2
    [[0, 0], [0, 1], [0, 3]], # 3
    [[0, 0], [0, 1], [0, 3], [0, 4]], # 4
    [[0, 0], [0, 1], [0, 3], [0, 4], [1, 1]], # 5
    [[0, 0], [0, 1], [0, 3], [0, 4], [1, 1], [1, 3]], # 6
    [[0, 0], [0, 1], [0, 3], [0, 4], [1, 1], [1, 3], [2, 1]], # 7
    [[0, 0], [0, 1], [0, 3], [0, 4], [1, 1], [1, 3], [2, 1], [2, 3]], # 8
    [[0, 0], [0, 1], [0, 3], [0, 4], [1, 1], [1, 3], [2, 1], [2, 3], [4, 0]], # 9
    [[0, 0], [0, 1], [0, 3], [0, 4], [1, 1], [1, 3], [2, 1], [2, 3], [4, 0], [3, 4]], # 10
    [[0, 0], [0, 1], [0, 3], [0, 4], [1, 1], [1, 3], [2, 1], [2, 3], [4, 0], [3, 4], [0, 2]], # 11
    [[0, 0], [0, 1], [0, 3], [0, 4], [1, 1], [1, 3], [2, 1], [2, 3], [4, 0], [3, 4], [0, 2], [1, 2]] # 12
]

class Seek(GridGame, GridObservation):
    def __init__(self, conf, number):
        # print("---------------------------------------seek0405--------------------------------------")
        colors = conf.get("colors", [(255, 255, 255), (0, 0, 0), (255, 69, 0), (222, 184, 135)])
        super().__init__(conf, colors)

        self.n_player = number

        self.board_width = 5
        self.board_height = 5

        self.player_move = False

        self.gridtype = 3  # 除player之外的格子类型
        self.n_barrier = 4
        self.barrier = [[2, 0], [2, 4], [3, 1], [3, 3]]  ## 障碍物的位置
        self.targets = [[4, 2]]  # rjq
        self.n_cell_type = self.gridtype + self.n_player  # rjq: 0：是否为普通空格点； 1：是否为目标点； 2至n_player+2-1：是否为玩家; n_player+2-1: 是否为障碍点；
        self.collision = 0


        # 方向[0, 1, 2, 3, 4]表示[上，下，左，右, 停]
        self.actions_name = {0: "up", 1: "down", 2: "left", 3: "right", 4: "stay"}  # rjq

        # init state
        self.players = []
        self.player_pos = player_position[self.n_player]
        for i in range(len(player_position[self.n_player])):
            player_pos_i = self.player_pos[i]
            self.players.append(
                Player(i, player_pos_i[0], player_pos_i[1], -1, self.board_height, self.board_width))  # rjq

        self.init_state()

        # images
        self.images = {"targets": [Bitmap(Image.open(dirname + "/images/seek/star.png"), self.grid_unit, (255, 0, 0)) for _ in
                                   range(len(self.targets))],
                       "players": [Bitmap(Image.open(dirname + "/images/seek/player.png"), self.grid_unit, (0, 0, 255)) for
                                   _ in range(self.n_player)],
                       "barrier": [Bitmap(Image.open(dirname + "/images/seek/barrier.png"), self.grid_unit, (255, 0, 0)) for
                                   _ in range(self.n_barrier)]   # rjq
                       }

    def init_state(self):
        self.step_cnt = 1
        self.reward = [0] * self.n_player
        # 最内层维度 [i][j][0]：是否为普通空格点； [i][j][1]：是否为目标点；
        # [i][j][2]至[i][j][n_player+2-1]：是否为玩家; [i][j][-1]: 是否为障碍点；
        self.current_state = [[[0] * (self.gridtype + self.n_player) for _ in range(self.board_width)] for _ in  # rjq
                              range(self.board_height)]
        ### 第一步 初始化
        for i in range(self.board_height):
            for j in range(self.board_width):
                self.current_state[i][j][0] = 1  # 初始化成普通空格点
        # target positions  # rjq 目标点 :(4,2)
        self.current_state[4][2][1] = 1
        self.current_state[4][2][0] = 0
        # 障碍物位置
        for i in self.barrier:
            self.current_state[i[0]][i[1]][-1] = 1  # position of barrier  # rjq
            self.current_state[i[0]][i[1]][0] = 0  # 更新，不为普通空格点  # rjq
        # player 位置
        for i in range(self.n_player):
            pos_i, pos_j = self.player_pos[i][0], self.player_pos[i][1]
            self.current_state[pos_i][pos_j][2 + i] = 1  ## 从2开始到2+10-1 位置初始化为1
            self.current_state[pos_i][pos_j][0] = 0  # 更新，不为普通空格点  # rjq

    
    
    
    def get_final_goal(self):
        # 最内层维度 [i][j][0]：是否为普通空格点； [i][j][1]：是否为目标点；
        # [i][j][2]至[i][j][n_player+2-1]：是否为玩家; [i][j][-1]: 是否为障碍点；
        current_state = [[[0] * (self.gridtype + self.n_player) for _ in range(self.board_width)] for _ in  # rjq
                              range(self.board_height)]
        ### 第一步 初始化
        for i in range(self.board_height):
            for j in range(self.board_width):
                current_state[i][j][0] = 1  # 初始化成普通空格点
        # target positions  # rjq 目标点 :(2,1)
        current_state[4][2][1] = 1
        current_state[4][2][0] = 0
        # 障碍物位置
        for i in self.barrier:
            current_state[i[0]][i[1]][-1] = 1  # position of barrier  # rjq
            current_state[i[0]][i[1]][0] = 0  # 更新，不为普通空格点  # rjq
        # player 位置
        for i in range(self.n_player):
            pos_i, pos_j = 4, 2 ##### 目标点的位置
            current_state[pos_i][pos_j][2 + i] = 1  ## 从2开始到2+10-1 位置初始化为1
            current_state[pos_i][pos_j][0] = 0  # 更新，不为普通空格点  # rjq
        return current_state
    
    
    
    def check_win(self):  # rjq0330
        for pos in self.targets:
            for player in range(self.n_player):
                player_pos = [self.players[player].row, self.players[player].col]
                if player_pos == pos:
                    continue
                else:
                    return False
        return True

    def get_grid_observation(self, current_state, player_id):
        return current_state

    def trans_action(self, joint_action):
        # joint_action = np.transpose(np.nonzero(np.array(joint_action)))[:,1]

        joint_action = np.expand_dims(joint_action, axis=1)

        joint_action = joint_action.astype(int).tolist()

        return joint_action

    def get_next_state(self, joint_action):
        joint_action = self.trans_action(joint_action)
        # print(joint_action)

        info_after = {}
        # new for next state
        next_state = [[[0] * (self.gridtype + self.n_player) for _ in range(self.board_width)] for _ in
                      range(self.board_height)]   # rjq

        for i in range(self.board_height):
            for j in range(self.board_width):
                next_state[i][j][0] = 1   # 这儿有一个bug，不是self.current_state

        # target positions  # rjq 目标点 :(2,1)
        next_state[4][2][1] = 1
        next_state[4][2][0] = 0
        # 障碍物位置
        for i in self.barrier:
            next_state[i[0]][i[1]][-1] = 1  # position of barrier  # rjq
            next_state[i[0]][i[1]][0] = 0  # 更新，不为普通空格点  # rjq

        not_valid = self.is_not_valid_action(joint_action)
        if not not_valid and self.step_cnt != 0:   ## 增加弹回动作；
            cur_pos_lists = []
            next_pos_lists = []
            for player_id in range(self.n_player):
                cur_pos_lists.append(self.players[player_id].get_cur_pos())  # rjq 记录当前两个player的位置
                # cur_pos = [self.players[player_id].row, self.players[player_id].col]
                # self.current_state[cur_pos[0]][cur_pos[1]][player_id + 2] = 0
                action = joint_action[player_id][0].index(1)
                self.players[player_id].direction = action
                self.players[player_id].move()
                next_pos = [self.players[player_id].row, self.players[player_id].col]
                next_pos_lists.append(next_pos)  # rjq 记录两个player下一步的位置

                ### 去重 list(set([tuple(t) for t in next_pos_lists]))

            # next_pos_lists_del_repeat = list(set([tuple(t) for t in next_pos_lists]))
            update_flag = [next_pos_lists.count(next_pos_lists[i]) == 1 for i in range(self.n_player)]
            self.collision = self.n_player - np.sum(update_flag)
            for player_id in range(self.n_player):
                if update_flag[player_id] == True:
                    next_state[next_pos_lists[player_id][0]][next_pos_lists[player_id][1]][player_id + 2] = 1
                    next_state[next_pos_lists[player_id][0]][next_pos_lists[player_id][1]][0] = 0
                    info_after[player_id] = next_pos_lists[player_id]
                else:
                    next_state[cur_pos_lists[player_id][0]][cur_pos_lists[player_id][1]][player_id + 2] = 1
                    next_state[cur_pos_lists[player_id][0]][cur_pos_lists[player_id][1]][0] = 0
                    info_after[player_id] = cur_pos_lists[player_id]
                    self.players[player_id].row = cur_pos_lists[player_id][0]  # rjq player 移回原位置
                    self.players[player_id].col = cur_pos_lists[player_id][1]

        info_after_list = []
        for player_id in range(self.n_player):
            info_after_list.append(info_after[player_id])

        self.player_move = self.check_player_move(cur_pos_lists, info_after_list)

        self.step_cnt += 1

        return next_state, str(info_after)


    def check_player_move(self, cur_pos_lists, info_after):
        if cur_pos_lists == info_after:
            return False
        else:
            return True

    def get_distance_target(self):
        sum_oushi = 0
        for i in range(self.n_player):
            sum_oushi += np.linalg.norm(np.array(self.players[i].get_cur_pos())-np.array(self.targets[0]))
        sum_oushi = sum_oushi / self.n_player

        return sum_oushi

    def check_getin_barrier(self):
        count = 0
        for i in range(self.n_player):
            if self.players[i].get_cur_pos() in self.barrier:
                count += 1
            # if self.players[i].get_cur_pos() in self.targets:
            #     print("step:::-------i am here", self.step_cnt, i)
        return count


    def get_hand_rewards(self):
        rewards_list = []
        for i in range(self.n_player):
            if self.get_distance_target() < 2:
                dis_rew = -self.get_distance_target() + 1
            elif self.get_distance_target() < 3:
                dis_rew = -self.get_distance_target()
            elif self.get_distance_target() < 4:
                dis_rew = -self.get_distance_target() - 1
            else:
                dis_rew = -self.get_distance_target()  ## 距离的奖励
            ###### 在目标的个数
            target_rew = 0
            if self.players[i].get_cur_pos() in self.targets:
                target_rew = 3.0
            rew_total = dis_rew + target_rew - self.collision * 1.0 - self.check_getin_barrier() * 5.0
            rewards_list.append(rew_total)
        return np.mean(rewards_list)

    def get_reward(self, joint_action):
        if self.check_win():
            for i in range(self.n_player):
                self.reward[i] = 100
        else:
            reward_shape = self.get_hand_rewards()
            for i in range(self.n_player):
                self.reward[i] = reward_shape
        # print("reward is {}".format(self.reward))


        return self.reward

    def get_terminal_actions(self):
        pass

    def is_in_destination(self, player_pos):
        # if player_pos == self.targets[0] or player_pos == self.target[1]:
        if player_pos == self.targets[0]:  # rjq
            return True
        return False

    def get_valid_action(self, player):  # rjq0330  得到有效的动作空间
        pos = player.get_cur_pos()
        if pos == [0, 0]:
            return [0, 1, 0, 1, 1]  # 右 下
        elif pos == [1, 0] or pos == [2, 0] or pos == [3, 0]:
            return [1, 1, 0, 1, 1]  # 右 上 下
        elif pos == [4, 0]:
            return [1, 0, 0, 1, 1]  # 右 上
        elif pos == [0, 1] or pos == [0, 2] or pos == [0, 3]:
            return [0, 1, 1, 1, 1]  # 下 左 右
        elif pos == [0, 4]:
            return [0, 1, 1, 0, 1]  # 下 左
        elif pos == [1, 4] or pos == [2, 4] or pos == [3, 4]:  
            return [1, 1, 1, 0, 1]  # 左 上 下
        elif pos == [4, 4]:
            return [1, 0, 1, 0, 1] # 左 上
        elif pos == [4, 1] or pos == [4, 3]:
            return [1, 0, 1, 1, 1]  # 左 右 上
        elif pos == [4, 2]:
            return [0, 0, 0, 0, 1]
        else:
            return [1, 1, 1, 1, 1]


    #
    # def get_valid_action(self, player):  # rjq0330  得到有效的动作空间
    #     pos = player.get_cur_pos()
    #     if pos == [0, 0]:
    #         return [0, 1, 0, 1, 1] # 右 下
    #     elif pos == [0, 2] or pos == [0, 1] or pos == [0, 3]:
    #         return [0, 1, 1, 1, 1] # 下 左 右
    #     elif pos == [0, 4]:
    #         return [0, 1, 1, 0, 1] # 下 左
    #     elif pos == [1, 0] or pos == [2, 1] or pos == [4, 0]:
    #         return [1, 0, 0, 1, 1] # 右 上
    #     elif pos == [1, 4] or pos == [2, 3] or pos == [4, 4]: # 左 上
    #         return [1, 0, 1, 0, 1]
    #     elif pos == [3, 0] or pos == [3, 4]: # 下
    #         return [0, 1, 0, 0, 1]
    #     elif pos == [3, 2]: # 上 下
    #         return [1, 1, 0, 0, 1]
    #     elif pos == [4, 1] or pos == [4, 3]: # 左 右
    #         return [1, 1, 0, 0, 1]
    #     else:
    #         return [1, 1, 1, 1, 1]


    def is_not_valid_action(self, joint_action):  # rjq
        not_valid = 0

        if len(joint_action) != self.n_player:
            raise Exception("joint action 维度不正确！", len(joint_action))

        for i in range(len(joint_action)):
            if len(joint_action[i][0]) != len(self.actions_name):
                raise Exception("玩家%d joint action维度不正确！" % i, joint_action[i])

        return not_valid

    def is_terminal(self):
        if self.check_win() or self.step_cnt > self.max_step:
            return True
        return False

    def reset(self):
        self.init_state()

    def set_action_space(self):
        action_space = [[5] for _ in range(self.n_player)]  # rjq 改成5
        return action_space

    @staticmethod
    def _render_board(state, board, colors, unit, fix, images, extra_info):
        im = GridGame._render_board(state, board, colors, unit, fix)
        draw = ImageDraw.Draw(im)
        targets = [[4, 2]]  # rjq

        for image in images["targets"]:
            draw.bitmap((image.y * unit + unit // fix - 4, image.x * unit + unit // fix - 4), image.bitmap,
                        image.color)

        for image in images["barrier"]:
            draw.bitmap((image.y * unit + unit // fix - 4, image.x * unit + unit // fix - 4), image.bitmap,
                        image.color)

        image_id = 0
        for i in extra_info.keys():
            x = i[0]
            y = i[1]

            value = extra_info[(x, y)]
            values = value.split("\n")

            for v in values:
                if v[0] == 'T':
                    continue
                if v[0] == 'P':
                    image = images["players"][image_id]
                    draw.bitmap((image.y * unit + unit // fix - 4, image.x * unit + unit // fix - 4), image.bitmap,
                                image.color)
                    image_id += 1

            # if [x, y] not in targets:
            #     for image in images["players"]:
            #         draw.bitmap((image.y * unit + unit // fix - 4, image.x * unit + unit // fix - 4), image.bitmap,
            #                     image.color)

            draw.text(((y + 1.0 / 8) * unit, (x + 1.0 / 20) * unit), extra_info[i], fill=(0, 0, 0))
        return im

    def render_board(self):
        extra_info = {}
        for i in range(len(self.targets)):
            pos = self.targets[i]
            x = pos[0]
            y = pos[1]
            self.images["targets"][i].set_pos(x, y)
            # print("position in render {}, {}".format(pos, i))
            if (x, y) not in extra_info.keys():
                extra_info[(x, y)] = 'T_' + str(i)
            else:
                extra_info[(x, y)] += '\n' + 'T_' + str(i)

        for i in range(self.n_player):
            x = self.players[i].row
            y = self.players[i].col
            self.images["players"][i].set_pos(x, y)
            if (x, y) not in extra_info.keys():
                extra_info[(x, y)] = 'P_' + str(i)
            else:
                extra_info[(x, y)] += '\n' + 'P_' + str(i)

        for i in range(self.n_barrier):
            x = self.barrier[i][0]
            y = self.barrier[i][1]
            self.images["barrier"][i].set_pos(x, y)
            if (x, y) not in extra_info.keys():
                extra_info[(x, y)] = 'B_' + str(i)
            else:
                extra_info[(x, y)] += '\n' + 'B_' + str(i)

        print("extra info is {}".format(extra_info))

        current_state = [[[0] * self.cell_dim for _ in range(self.board_width)] for _ in range(self.board_height)]
        im_data = np.array(Seek._render_board(self.get_render_data(current_state), self.grid, self.colors,
                                              self.grid_unit, self.grid_unit_fix, self.images, extra_info))
        self.game_tape.append(im_data)
        return im_data

    def printAll(self, pygame, screen, episode, step, num_steps):
        # ..........pygame 刷新一帧程序块，之后加入这一段
        if step == 0:  # 初次表示，取消初始化video
            fps = 10
            size = self.grid.size
            # size = (3, 3)
            file_path = r'../../logs/' + str(episode) + ".mp4v"  # 导出路径
            print("file_path", file_path)
            #fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
            #self.video = cv2.VideoWriter(file_path, fourcc, fps, size)
        else:  # 之后每一帧都写入video
            imagestring = pygame.image.tostring(screen.subsurface(0, 0, 3, 3), "RGB")
            pilImage = Image.frombytes("RGB", (3, 3), imagestring)
            #img = cv2.cvtColor(np.asarray(pilImage), cv2.COLOR_RGB2BGR)
            print("--------Begin to save the video--------")
            self.video.write(img)  # 把图片写进视频
            # if step == num_steps:
            # self.video.release()  # 释放

    def replay(self, n_epoch, state_set, action_set, episode_limit):
        '''

        :param state_set: list (num_episode, episode_limit, 2, 2)
        :return: im_data
        '''
        # self.reset()
        import pygame
        pygame.init()
        screen = pygame.display.set_mode(self.grid.size)
        pygame.display.set_caption(self.game_name)

        # video =cv2.VideoCapture("Youtube.mp4")
        # fps = video.get(cv2.CAP_PROP_FPS)
        # print(fps)
        # video.set(cv2.CAP_PROP_FPS, 60)
        # vid = []

        clock = pygame.time.Clock()
        env_type = "seek_2p"
        log_path = os.getcwd() + '/logs/'
        logger = get_logger(log_path, env_type, save_file=True)

        st = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        game_info = dict(game_name=env_type, n_player=self.n_player, board_height=self.board_height,
                         board_width=self.board_width,
                         init_state=str(self.get_render_data(self.current_state)), init_info=str(self.init_info), start_time=st,
                         mode="window",
                         render_info={"color": self.colors, "grid_unit": self.grid_unit, "fix": self.grid_unit_fix})
        # while not self.is_terminal():
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             pygame.quit()
        print('--------Replay One Episode Starting--------')
        for epoch in range(n_epoch):
            # pdb.set_trace()
            num_steps = np.nonzero(action_set[epoch, :, 0, 0])[0][-1]  # 最后一个index
            print("num_steps", num_steps)
            print("episode", epoch)
            # Resert players
            self.players = [Player(0, 0, 0, -1, self.board_height, self.board_width),  # rjq
                            Player(1, 0, 2, -1, self.board_height, self.board_width)]  # rjq
            if num_steps <= episode_limit:  # no-padding and lose finally
                for step in range(len(action_set[epoch])):
                    # For replay video
                    # s = "step%d" % self.step_cnt
                    game_info[step] = {}
                    game_info[step]["time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

                    self.current_state = state_set[epoch][step][0]
                    joint_action = action_set[epoch][step]
                    # not_valid = self.is_not_valid_action(joint_action)
                    game_info[step]["joint_action"] = str(joint_action)

                    ##################
                    # 根据状态判断两个agent的row和col
                    state_agent_step = state_set[epoch][step][0]
                    state_agent_step = state_agent_step.reshape(25, -1)
                    # pdb.set_trace()
                    for i in range(state_agent_step.shape[0]):
                        if (state_agent_step[i][2] == 1):
                            self.players[0].row = i // 5
                            self.players[0].col = i % 5
                        elif (state_agent_step[i][3] == 1):
                            self.players[1].row = i // 5
                            self.players[1].col = i % 5
                    # # if not not_valid:
                    # for player_id in range(self.n_player):
                    #     # cur_pos = [self.players[player_id].row, self.players[player_id].col]
                    #     # self.current_state[cur_pos[0]][cur_pos[1]][player_id + 2] = 0
                    #     action = joint_action[player_id][0]
                    #     self.players[player_id].direction = action
                    #     self.players[player_id].move()
                    ##################

                    game_info[step]["state"] = str(self.get_render_data(state_set))
                    pygame.surfarray.blit_array(screen, self.render_board().transpose(1, 0, 2))
                    pygame.display.flip()
                    clock.tick(10)
                    # self.printAll(pygame, screen, epoch, step, num_steps)
                # self.video.release()  # 释放
                json_object = json.dumps(game_info, indent=4, ensure_ascii=False)
                logger.info(json_object)
                # from games.render_from_log import replay_video
                # replay_video("logs/202012280941_seek_2p.log")
                print("--------End the video saving--------")
            if epoch == n_epoch - 1:
                pygame.quit()

        print('--------Replay One Episode Ending--------')



class Player:
    def __init__(self, player_id, row, col, direction, board_height, board_width):
        self.player_id = player_id
        self.row = row
        self.col = col
        self.direction = direction
        self.board_height = board_height
        self.board_width = board_width

    def move(self):
        next_pos = self.get_next_pos(self.row, self.col)
        self.row = next_pos[0]
        self.col = next_pos[1]



    def get_cur_pos(self):
        return [self.row, self.col]

    def get_next_pos(self, cur_row, cur_col):

        next_row = cur_row
        next_col = cur_col

        if self.direction == 0:  # up
            if cur_row - 1 >= 0:
                next_row = cur_row - 1
        elif self.direction == 1:  # down
            if cur_row + 1 < self.board_height:
                next_row = cur_row + 1
        elif self.direction == 2:  # left
            if cur_col - 1 >= 0:
                next_col = cur_col - 1
        elif self.direction == 3:  # right
            if cur_col + 1 < self.board_width:
                next_col = cur_col + 1

        return [next_row, next_col]


class Bitmap:
    def __init__(self, bitmap, unit, color):
        self.bitmap = bitmap
        self.x = 0
        self.y = 0
        self.unit = unit
        self.reshape()
        self.color = color

    def set_pos(self, x, y):
        self.x = x
        self.y = y

    def reshape(self):
        self.bitmap = self.bitmap.resize((self.unit, self.unit))
