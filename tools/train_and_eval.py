import os
import argparse
import logging
import time

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='paddle-rec run')
    parser.add_argument("-m", "--config_yaml", type=str, help="config file")
    parser.add_argument("-e", "--epochs", default=3, type=int, help="epochs")
    parser.add_argument("-n", "--model_name", type=str, help="model name")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    for epoch in range(args.epochs):
        if epoch == 0:
            cmd_str = "python -u tools/trainer.py -m {} -e {}".format(args.config_yaml, 1)
        else:
            cmd_str = "python -u tools/trainer.py -m {} -i {} -e {}".format(args.config_yaml,
                                                                            "output_model_{}/{}".format(args.model_name,
                                                                                                        0),
                                                                            1)
        # step 1, train model
        try:
            logger.info("#" * 10 + "train model, {} epoch".format(epoch) + "#" * 10)
            os.system(cmd_str)
        except:
            logger.info("train model error...")

        time.sleep(1)

        # step 2, eval model
        try:
            logger.info("#" * 10 + "eval model, {} epoch".format(epoch) + "#" * 10)
            cmd_str = "python -u tools/infer.py -m {} -s {} -e {}".format(args.config_yaml, 0, 1)
            os.system(cmd_str)
        except:
            logger.info("eval model error...")
# python3.7 -u tools/train_and_eval.py -m models/rank/fm/config.yaml -e 3 -n fm
