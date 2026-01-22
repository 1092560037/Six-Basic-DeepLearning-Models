import time
import sys

def now():
    return time.strftime('%H:%M:%S')

print(now(), 'start')

try:
    print(now(), 'importing torch')
    import torch
    print(now(), 'imported torch', getattr(torch, '__version__', None))
except Exception as e:
    print(now(), 'torch import error:', repr(e))
    sys.exit(1)

try:
    print(now(), 'importing torchvision')
    import torchvision
    print(now(), 'imported torchvision', getattr(torchvision, '__version__', None))
except Exception as e:
    print(now(), 'torchvision import error:', repr(e))
    sys.exit(1)

try:
    print(now(), 'importing torchvision.transforms')
    from torchvision import transforms
    print(now(), 'imported transforms')
except Exception as e:
    print(now(), 'transforms import error:', repr(e))
    sys.exit(1)

print(now(), 'done')
