import multiprocessing as mp
import sys

def catch_key():
    
    sys.stdin = open(0)
    sys.stdin.flush()
    while(True):
        a = sys.stdin.read(1)
        if(a=='e'):
            break
        else:
            print('unrecognised')
            sys.stdin.read(1)
    
    return True

if __name__ == "__main__":
    print("Starting the 5 minute sleep")
    print("Enter e to exit at anytime")
    a = mp.Process(target=catch_key)
    a.start()
    a.join(timeout=300)
    if a.is_alive():
        a.terminate()
    