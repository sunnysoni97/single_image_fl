import argparse
from arg_handler import parse_bool

parser = argparse.ArgumentParser(description="test parser")

parser.add_argument("--flag2",type=parse_bool, default=False)


if __name__ == "__main__":
    args=parser.parse_args()
    print(f"Value of flag 2 (direct): {args.flag2}, type {type(args.flag2)}")
    if(args.flag2):
        print("Flag2 was true!")
    else:
        print("Flag2 was false!")
