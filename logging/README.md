# Logging
This is where we conduct all our experiment logging to generate plots for
training loss, validation loss, test accuracy, etc. Currently the default 
logger uses `tensorboard`. 

## Visualizing Experiment Information
Logs are automatically updated to the relevant version folders. If you want to
visualize the logs of experiment version x in `tensorboard` do the
following:
```python
tensorboard --logdir <version_x>
```

