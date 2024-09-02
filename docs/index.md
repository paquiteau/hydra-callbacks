# Hydra Callbacks

<center>
<img src=img/hydra_callbacks.jpeg width=400px/>
</center>

Add callback to your [hydra](https://hydra.cc/docs/intro) application.

Hydra Callbacks is a simple library that allows you to add callbacks to your hydra application. It is a simple way to add hooks to your application, without reinventing the wheel. 
You can collect results from multiple jobs, Monitor resources consumptions, setup extra script to run after the job is done, and even more. 

## Installation

```bash
pip install hydra-callbacks
```

## Usage

In you hydra config you can add a callback to your application, to run at the start and/or at the end of single or multirun job.

For instance, to automatically create a `latest` link to your job, use the `LatestRunLink` callback:


```yaml
# all your hydra config ...

hydra:
    callbacks:
        latest_run:
        _target_: hydra_callbacks.LatestRunLink
        run_base_dir:  ${result_dir}/outputs
        multirun_base_dir:  ${result_dir}/multirun
```

You can have as many callbacks as you want. 

Note that callback are "wrapping around" the hydra job, so at the start of the job they are executed top to bottom, and at the end of the job they are executed bottom to top.


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

Then you can run your hydra job as usual:

```term
paquiteau@laptop$ python my_app.py
[hydra] Git sha: 844b9ca1a74d8307ef5331351897cebb18f71b88, dirty: False

## All your app log and outputs ##

[hydra][INFO] - Total runtime: 0.51 seconds
[hydra][INFO] - Writing monitoring data to [...]/outputs/2023-04-06/16-02-46/resource_monitoring.csv
[hydra][INFO] - Latest run is at: [...]/outputs/latest
```
