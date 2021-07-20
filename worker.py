import rpc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log", default=True, type=bool, help="save trace log")

if __name__ == '__main__':
    args = parser.parse_args()
    rpc.Cohort(save_trace_log=args.log).slave_forever()
