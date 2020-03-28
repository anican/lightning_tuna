# callbacks
Here is a place for all your custom callbacks for deep learning experiments.  
Lightning has a callback system to execute arbitrary code. 

Callbacks should capture NON-ESSENTIAL logic that is NOT required for your 
LightningModule to run. An overall Lightning system should have:

1. Trainer for all engineering
2. LightningModule for all research code.
3. Callbacks for non-essential code.
