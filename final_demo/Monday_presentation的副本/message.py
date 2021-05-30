import struct

def encode_msg_size(size: int) -> bytes:
    return struct.pack("<I", size)

def decode_msg_size(size_bytes: bytes) -> int:
    return struct.unpack("<I", size_bytes)[0]

def create_msg(content: bytes) -> bytes:
    size = len(content)
    return encode_msg_size(size) + content
    
if __name__ == "__main__":
    print(encode_msg_size(12)) #=> b'\x0c\x00\x00\x00'
    print(decode_msg_size(b'\x0c\x00\x00\x00')) #=> 12
