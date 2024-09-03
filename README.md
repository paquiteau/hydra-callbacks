# Hydra Callbacks 

<p align="center">
<img src="https://raw.githubusercontent.com/paquiteau/hydra-callbacks/master/docs/img/hydra_callbacks.jpeg" width="400px" />
</p>

[![style](https://img.shields.io/badge/style-black-black)](https://github.com/psf/black)
[![framework](https://img.shields.io/badge/framework-hydra-blue)](https://hydra.cc)
[![codecov](https://codecov.io/gh/paquiteau/hydra-callbacks/branch/master/graph/badge.svg?token=NEV7SY24YB)](https://codecov.io/gh/paquiteau/hydra-callbacks)
[![CD](https://github.com/paquiteau/hydra-callbacks/actions/workflows/master-cd.yml/badge.svg)](https://github.com/paquiteau/hydra-callbacks/actions/workflows/master-cd.yml)
[![CI](https://github.com/paquiteau/hydra-callbacks/actions/workflows/test-ci.yml/badge.svg)](https://github.com/paquiteau/hydra-callbacks/actions/workflows/test-ci.yml)
[![Release](https://img.shields.io/github/v/release/paquiteau/hydra-callbacks)](https://github.com/paquiteau/hydra-callbacks/releases/latest)



A collection of usefulls and simple to use callbacks for the [hydra](https://hydra.cc/) configuration framework.


## Installation 
``` shell 
pip install hydra-callbacks
```

Development version 
``` shell
pip install git+https://github.com/paquiteau/hydra-callbacks
```

## Usage 

In your hydra root config file add the following, or analoguous:

``` yaml
hydra:
  callbacks:
    git_infos:
      _target_: hydra_callbacks.GitInfo
      clean: true
    latest_run:
      _target_: hydra_callbacks.LatestRunLink
    resource_monitor:
      _target_: hydra_callbacks.ResourceMonitor
      sample_interval: 0.5
    runtime_perf:
      _target_: hydra_callbacks.RuntimePerformance      
```

This will enrich your script output with: 

```console
paquiteau@laptop$ python my_app.py
[hydra] Git sha: 844b9ca1a74d8307ef5331351897cebb18f71b88, dirty: False

## All your app log and outputs ##

[hydra][INFO] - Total runtime: 0.51 seconds
[hydra][INFO] - Writing monitoring data to [...]/outputs/2023-04-06/16-02-46/resource_monitoring.csv
[hydra][INFO] - Latest run is at: [...]/outputs/latest
```


Detailled configuration for each callback is available in the `tests/test_app/` folder.

## Available Callbacks 

| Name               | Action                                             |
|:-------------------|:---------------------------------------------------|
| GitInfo            | Check status of Repository                         |
| LatestRunLink      | Get a link to the latest run                       |
| MultiRunGatherer   | Gather results json file in a single table         |
| RuntimePerformance | Get Execution time for each run                    |
| ResourceMonitor    | Monitor resources of running jobs (CPU and Memory) |

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
    PerfLogger.recap(log)

```
 - `RessourceMonitorService` : A simple CPU and GPU usage and memory sampler. It launches an extra process to monitor everything.
 
 ```python
from hydra_callbacks.monitor import RessourceMonitorService
import os 
 
monitor = RessourceMonitorService(interval=0.2, gpu_monit=True)
 
monitor.start()
# Launch heavy stuff 
metrics = monitor.stop()

# Or use it as a context manager
with RessourceMonitorService(interval=0.2, gpu_monit=True) as monitor: 
    # launch heavy stuff
    
metrics_values = monitor.get_values()
```


## You too, have cool Callbacks, or idea for one ? 

Open a PR or an issue !

### Possible Ideas
- A callback that summarize log from multiple runs
- Monitoring of GPU  using nvitop
  
<p align=center> :star2: If you like this work, don't forget to star it and share it 🌟 </p>

