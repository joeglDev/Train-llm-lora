from LoraTrainingPipeline import LoraTrainingPipeline

# hf_model = # "mistralai/Mistral-7B-Instruct-v0.3"
runner = LoraTrainingPipeline(device="0", hf_model="pankajmathur/orca_mini_3b")
runner.run()
