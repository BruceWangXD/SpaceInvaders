import os
from message import create_msg


###
from random import choice
import random


###

if __name__ == "__main__":
    IPC_FIFO_NAME = "hello_ipc"

    fifo = os.open(IPC_FIFO_NAME, os.O_WRONLY)
    try:
        while True:
            
            name=bool(random.getrandbits(1))
            
            # name = input("Enter a name: ")
            
            content = f"{name}".encode("utf8")
            msg = create_msg(content)
            os.write(fifo, msg)
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        os.close(fifo)
