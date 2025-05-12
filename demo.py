from ModuLM import *
import sys

if __name__ == "__main__":
    config = ModuLMConfig()
    
    args = config.parse_from_json("ModuLM_config.json")  
    model_runner = ModuLM(args)
    model_runner.run()