import argparse
import sys


def auto_convert(value):
    try:
        float_value = float(value)
        if float_value.is_integer():
            return int(float_value)
        else:
            return float_value
    except ValueError:
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        else:
            return value

# Define the command-line argument parser for usermode
parser = argparse.ArgumentParser()
parser.add_argument('--usermode', type=str, default="")

args, unknown_args = parser.parse_known_args()

# Set the user mode based on the command-line argument
usermodestr = args.usermode
if unknown_args:
    sys.argv = [sys.argv[0]] + unknown_args

usermode = {e.split('=')[0] : auto_convert(e.split('=')[1]) if len(e.split('=')) > 1 else None for e in (usermodestr[0].split(',') if type(usermodestr) is not str else usermodestr.split(',')) }