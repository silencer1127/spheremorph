import numpy as np
import subprocess as sp


# adapted from freesurfer deeplearn
def norm_curvature(curv, which_norm='Median', norm_percentile=97, std_thresh=3):
    if which_norm == 'Percentile':
        normed = np.clip(curv / np.percentile(curv, norm_percentile), 0, 1)
    elif which_norm == 'Median':
        min_clip = np.percentile(curv, 100 - norm_percentile)
        max_clip = np.percentile(curv, norm_percentile)
        st = np.std(np.clip(curv, min_clip, max_clip))
        normed = np.clip(((curv - np.median(curv)) / st), -std_thresh, std_thresh)
    else:
        normed = (curv - curv.mean()) / np.std(curv)

    return normed

# copied from fsmodule
def run(command, silent=False, background=False, executable='/bin/bash', logfile=None):
    '''Runs a shell command and returns the exit code.

    Note:
        A command run in the background will always return `None`.
    Args:
        command: Command to run.
        silent: Send output to devnull. Defaults to False.
        background: Run command as a background process. Defaults to False.
        executable: Shell executable. Defaults to `/bin/bash`.
    Returns:
        Command exit code.
    '''

    # redirect the standard output appropriately
    if silent:
        std = {'stdout': sp.DEVNULL, 'stderr': sp.DEVNULL}
    elif not background:
        std = {'stdout': sp.PIPE, 'stderr': sp.STDOUT}
    else:
        std = {}  # do not redirect

    # run the command
    process = sp.Popen(command, **std, shell=True, executable=executable)
    if not background:
        # write the standard output stream
        if process.stdout:
            for line in process.stdout:
                decoded = line.decode('utf-8')
                if logfile is not None:
                    with open(logfile, 'a') as file:
                        file.write(decoded)
                sys.stdout.write(decoded)
        # wait for process to finish
        process.wait()

    return process.returncode
