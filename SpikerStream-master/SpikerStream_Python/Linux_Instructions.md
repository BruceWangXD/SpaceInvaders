Streaming SpikerStream in Python on Linux
===============================================
(Written by Tom Schwarz: contact me at github.com/tomiam8 if you need)

The files SpikerStream_Linux.py is based off SpikearStream_Python3.ipynb, but as a script instead of a notebook, and with a few changes for Linux.

To use:
1. Set line 31 (`cport = '/dev/ttyACM0'`) to the correct port.
You can find the correct port by running `ls /dev` to see all your ports, It will probably be something of the form 'ttyACMXX' (you need to have it plugged in so it shows up), try plugging and unplugging to see what changes if you need.

2. Run with `sudo python3 SpikerStream_Linux.py`. You need sudo so it can access the port (ideally check the code is safe before running it as sudo - it theoretically could have `os.run('rm -rf /') in there).
It will probably give you a ModuleNotFoundError as you don't have serial/numpy/matplotlib. Solve this with `sudo pip install pyserial`, run again, and repeat (`sudo pip install numpy`, `sudo pip install matplotlib`)

The code will run for a few seconds and then plot it in matplotlib.

SpikerStream_Linux_Live.py continuously updates the plot so it's 'live'. It only runs for ~10 seconds, you can make it run forever by changing line 54. You'll again need to change the port, this time it's on line 30.

That should work!
If it doesn't, try figuring out what's wrong and once you fix it, fork and pull request your changes so you can help other people too.


