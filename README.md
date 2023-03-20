# Hydra Callbacks 

A collection of usefulls callbacks for the [https://hydra.cc/](hydra) configuration framework.


## Installation 


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

| Name               | Action                                     | Author                                    |
|:-------------------|:-------------------------------------------|-------------------------------------------|
| GitInfo            | Check status of Repository                 | [paquiteau](https://github.com/paquiteau) |
| LatestRunLink      | Get a link to the latest run               | [paquiteau](https://github.com/paquiteau) |
| MultiRunGatherer   | Gather results json file in a single table | [paquiteau](https://github.com/paquiteau) |
| RuntimePerformance | Get Execution time for each run            | [paquiteau](https://github.com/paquiteau) |

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

