# -*- coding:utf-8  -*-
# 作者：zruizhi   
# 创建时间： 2020/10/30 4:15 下午   
# 描述：
import logging
import time
import os


def get_logger(log_path, name, save_file=True, console_out=True):
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 每分钟建一个文件
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_name = log_path + rq + '_' + name+  '.log'
    logfile = log_name
    if save_file:
        fh = logging.FileHandler(logfile, mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    # 输出到控制台
    if console_out:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(console)

    return logger