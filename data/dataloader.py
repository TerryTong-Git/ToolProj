
import json 

#TODO: Finish this dataloader, then do SFT. 
class NumGlueDataLoader:
    
    def __init__(self):
        # Interface is tokenized data
        self.data = []
        for split in ["dev", "test", "train"]:
            with open(f"NumGlue_{split}_1_4_8.json", "r") as f:
                self.data.append(json.load(f))
        
