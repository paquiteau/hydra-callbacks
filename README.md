# Hydra Callbacks 


[![style](https://img.shields.io/badge/style-black-black)](https://github.com/psf/black)
[![framework](https://img.shields.io/badge/framework-hydra-blue)](https://hydra.cc)
[![codecov](https://codecov.io/gh/paquiteau/hydra-callbacks/branch/master/graph/badge.svg?token=NEV7SY24YB)](https://codecov.io/gh/paquiteau/hydra-callbacks)
[![CD](https://github.com/paquiteau/hydra-callbacks/actions/workflows/master-cd.yml/badge.svg)](https://github.com/paquiteau/hydra-callbacks/actions/workflows/master-cd.yml)
[![CI](https://github.com/paquiteau/hydra-callbacks/actions/workflows/test-ci.yml/badge.svg)](https://github.com/paquiteau/hydra-callbacks/actions/workflows/test-ci.yml)
[![Release](https://github.com/paquiteau/hydra-callbacks/releases/latest)](https://img.shields.io/github/v/release/paquiteau/hydra-callbacks)

A collection of usefulls callbacks for the [https://hydra.cc/](hydra) configuration framework.


## Installation 
``` shell 
pip install hydra-callbacks
```

Development version 
``` shell
pip install git+https://github.com/paquiteau/hydra-callbacks
```

## Usage 

In your hydra root config file add the following: 

``` yaml
hydra: 
  # ... 
  callbacks: 
    git_info:
      _target_: hydra_callbacks.GitInfo 
      clean: true
    latest_link:
      _target_: hydra_callbacks.LatestRunLink
```


## Available Callbacks 

| Name               | Action                                     |
|:-------------------|:-------------------------------------------|
| GitInfo            | Check status of Repository                 |
| LatestRunLink      | Get a link to the latest run               |
| MultiRunGatherer   | Gather results json file in a single table |
| RuntimePerformance | Get Execution time for each run            |

And more to come ! 

## Also Available 
  
  - `PerfLogger` : A simple to use performance logger
  
```python
  
from hydra_callbacks import PerfLogger 
import logging

log = logging.getLogger(__name__)
def main_app(cfg):
    with PerfLogger(log, "step1"):
        sleep(1)

    with PerfLogger(log, "step2"):
        sleep(2)
    log.info(PerfLogger.recap())

```

## You too, have cool Callbacks, or idea for one ? 

Open a PR or an issue !

### Possible Ideas
- [In progress] A Ressource Monitoring Callback 
- A callback that summarize log from multiple runs
