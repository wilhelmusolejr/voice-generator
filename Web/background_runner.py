import threading
import subprocess
import uuid
import os

def run_generator_async(args):
    subprocess.run(args, check=True)