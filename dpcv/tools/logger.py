import os
import logging
from datetime import datetime


class Logger(object):
    def __init__(self, path_log):
        log_name = os.path.basename(path_log)
        self.log_name = log_name if log_name else "root"
        self.out_path = path_log

        log_dir = os.path.dirname(self.out_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def init_logger(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=logging.INFO)

        # file config handler
        file_handler = logging.FileHandler(self.out_path, 'w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # screen config handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # adding handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


def make_logger(out_dir):
    """
    directories under out_dir named by data_time
    :param out_dir: str
    :return:
    """
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(out_dir, time_str)  # use config time as directory name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # creat logger
    path_log = os.path.join(log_dir, "log.log")
    logger = Logger(path_log)
    logger = logger.init_logger()
    return logger, log_dir


if __name__ == "__main__":

    # setup_seed(2)
    # print(np.random.randint(0, 10, 1))

    logger = Logger('./logtest.log')
    logger = logger.init_logger()
    for i in range(10):
        logger.info('test:' + str(i))