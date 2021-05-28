import os
from message import create_msg

if __name__ == "__main__":
    IPC_FIFO_NAME = "hello_ipc"

    fifo = os.open(IPC_FIFO_NAME, os.O_WRONLY)
    try:
        while True:
            name = input("Enter a name: ")
            content = f"Hello {name}!".encode("utf8")
            msg = create_msg(content)
            os.write(fifo, msg)
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        os.close(fifo)
