# SpikerStream

This repository contains Matlab and python code for recording real-time data from the Backyard Brains [Spiker Box](https://backyardbrains.com/products/heartAndBrainSpikerBox)!

## Downloading

There are two ways that you can work with this code:

### 1. Download .zip

For users unfamiliar with git, the latest version of the repository can be downloaded by simply clicking the green "Code" button above, and then clicking "Download ZIP".

### 2. Forking the repo

If you are familiar with using git, please [make a fork](https://help.github.com/articles/fork-a-repo/) of this repository to your own Github account, then clone that fork to your local machine.
If you are working in a group, you can all work from your group's fork.

You may optionally add an [upstream remote](https://help.github.com/articles/fork-a-repo/#step-3-configure-git-to-sync-your-fork-with-the-original-spoon-knife-repository) to keep it synchronized with the main repository.

## Requirements

### Matlab

Make sure your Matlab installation includes the _Signal Processing & Communications_ and _Curve Fitting_ Toolboxes.

### Python

Make sure you install the `pyserial` package, which allows you to read from the input communications (COM) port.
For example, using the `pip` package manager:

```
pip install pyserial
```
