# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023.3.8 15:35
# @Author  : James.T
# @File    : task_manager.py

import argparse
import csv
import time
import os
import json
from json.decoder import JSONDecodeError
import sys

task_config_template = {
    'task_id': 'None',
    'create_time': 'None',
    'description': 'None',
    'is_config': False,  # a flag determining the task is configured or not

    'train_dataset_config': {
        'train_dataset_name': 'COCO2014',  # what dataset to use
        'crop_size': 96,  # cropped HR size
        'scaling_factor': 4,  # up_scaling
    },
    'generator_config': {
        'large_kernel_size_g': 9,  # 第一层卷积和最后一层卷积的核大小
        'small_kernel_size_g': 3,  # 中间层卷积的核大小
        'n_channels_g': 64,  # 中间层通道数
        'n_blocks_g': 16,  # 残差模块数量
        'generator_weight_initial': None
    },
    'discriminator_config': {
        'kernel_size_d': 3,  # 中间层卷积的核大小
        'n_channels_d': 64,  # the first conv block channel
        'n_blocks_d': 8,
        'fc_size_d': 1024,
    },
    'perceptual_config': {
        'vgg19_i': 5,
        'vgg19_j': 4,
        'beta': 1e-3,
    },
    'hyper_params': {
        'total_epochs': None,
        'batch_size': None,
        'lr_initial': None,
        'lr_decay_gamma': None,
        'lr_milestones': [],
    },
    'others': {
        'n_gpu': 1,
        'worker': 4,
    },
}


class TaskManager:
    """
    provide functions to manage a whole program
    """
    def __init__(self):
        """
        initialization
        """
        self.__register_list_path = './task_record/register_list.csv'
        self.__register_list_fieldnames = ['create_time','task_id']
        self.__register_list = []

        try:
            self.__create_register_list()
            print('TaskManager: tanew project created...')
        except FileExistsError:
            print('TaskManager: current project exists, loading...')
            self.__get_register_list()

    def new_task(self, new_task_id):
        """
        create a new task

        :param new_task_id: a id for a new task
        :returns: None
        """
        # check if task id exists
        if self.__is_task_registered(new_task_id):
            print(f'TaskManager: task id \'{new_task_id}\' already exists, choose another one')
            sys.exit()

        # create directory, you can change dir arragement here
        task_brief = {'create_time': time.strftime('%Y%m%d_%H%M_%A', time.localtime()),
                      'task_id': new_task_id}
        task_path = self.__get_task_path(task_brief)
        try:
            os.mkdir(task_path)
            os.mkdir(os.path.join(task_path, 'checkpoint'))
            os.mkdir(os.path.join(task_path, 'record'))
        except FileExistsError:
            print(f'TaskManager: {task_path} or sth in it already exist')
            sys.exit()

        # create config file
        config_file = os.path.join(task_path, f'{new_task_id}_config.json')
        new_task_config = task_config_template
        new_task_config['task_id'] = new_task_id
        new_task_config['create_time'] = time.strftime(f'%Y.%m.%d %H:%M:%S %A', time.localtime())
        try:
            self.__write_task_config(new_task_config, config_file)
        except FileExistsError:
            print(f'TaskManager: {config_file} already exist')
            sys.exit()

        # register new task
        self.__add_register_list(task_brief)

    def show_register_list(self):
        """
        show register list
        """
        for task_brief in self.__register_list:
            print(task_brief)

    def display_task_config(self, task_id):
        """
        display task_config of a task named task_id

        :param task_id: the task you want to show
        :returns: None
        """
        config = self.get_task_config(task_id)
        print(json.dumps(config, indent='\t'))

    def get_task_config(self, task_id):
        """
        config

        :param task_id: the task you want to get
        :returns: dict, task config
        """
        if not self.__is_task_registered(task_id):
            print(f'TaskManager: no task called \'{task_id}\'')
            sys.exit()
        task_path = self.__get_task_path(self.__get_task_brief(task_id))
        task_config = self.__get_task_config(
            os.path.join(task_path, f'{task_id}_config.json')
        )
        if not task_config['is_config']:
            print('WARNING: task not configured yet')
        return task_config

    def get_task_path(self, task_id):
        """
        get path according to task_id

        :param task_id:
        :returns: a set of task path
        """
        if not self.__is_task_registered(task_id):
            print(f'TaskManager: no task called \'{task_id}\'')
            sys.exit()
        task_path = self.__get_task_path(self.__get_task_brief(task_id))
        task_path_dict = {
            'task_path': task_path,
            'log_path': os.path.join(task_path, 'log.log'),
            'checkpoint_path': os.path.join(task_path, 'checkpoint'),
            'record_path': os.path.join(task_path, 'record'),
        }
        return task_path_dict

    # def delete_task(self, task_id):
    #     """
    #     never recommend to use
    #     """
    #     pass

    def __get_task_brief(self, task_id):
        """
        get task brief from reg list
        """
        try:
            index = [item.get('task_id') for item in self.__register_list].index(task_id)
        except ValueError as err:
            print(f'TaskManager: task not found, {err}')
            sys.exit()
        return self.__register_list[index]

    def __get_task_path(self, task_brief):
        """
        get path from {'create_time', 'task_id'}
        """
        task_dirname = task_brief.get('create_time') + '_' + task_brief.get('task_id')
        task_path = os.path.join('task_record', task_dirname)
        return task_path

    def __is_task_registered(self, task_id):
        """
        check if task id exists
        """
        if task_id in [id.get('task_id') for id in self.__register_list]:
            return True
        else:
            return False

    def __create_register_list(self):
        """
        create a csvfile at __register_list_path

        :raises FileExistsError
        """
        # 也许可以换 pandas
        with open(self.__register_list_path, 'x', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.__register_list_fieldnames)
            writer.writeheader()

    def __get_register_list(self):
        """
        get register_list from file
        """
        with open(self.__register_list_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            self.__register_list = list(reader)

    def __add_register_list(self, task_brief):
        """
        append one task to register_list, and save to file
        """
        with open(self.__register_list_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.__register_list_fieldnames)
            writer.writerow(task_brief)
            self.__register_list.append(task_brief)

    def __write_register_list(self):
        pass

    def __write_task_config(self, config, config_path):
        """
        write config file
        """
        with open(config_path, 'w') as jsonfile:
            json.dump(config, jsonfile, indent='\t')

    def __get_task_config(self, config_path):
        """
        load config
        """
        with open(config_path, 'r') as jsonfile:
            try:
                config = json.load(jsonfile)
            except JSONDecodeError:
                print('TaskManager: Not valid json doc!')
                sys.exit()
        return config


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='task manager cmd')
    parser.add_argument("--new", "-n", action="store_true", help="create a new task")
    parser.add_argument("--list", "-l", action="store_true", help="show register list")
    parser.add_argument("--showconfig", "-s", action="store_true", help="show config of a task")
    parser.add_argument("--taskid", "-id", type=str, default='', help="an id of a task")
    args = parser.parse_args()

    tm = TaskManager()

    # new task
    if args.new:
        if args.taskid == '':
            print(f'TaskManager: task id required')
        else:
            tm.new_task(args.taskid)
            print(f'\'{args.taskid}\' created')

    # show task config
    elif args.showconfig:
        if args.taskid == '':
            print(f'TaskManager: taskid required')
        else:
            tm.display_task_config(args.taskid)

    # show register list
    elif args.list:
        tm.show_register_list()
            

# 改进其他任务通用性，比如文件夹结构配置