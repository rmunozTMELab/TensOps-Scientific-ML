from config import DATASET_CONFIG
from create_data import DataGenerator

if __name__ == "__main__":
    
    # Load dataset config
    config = DATASET_CONFIG
    generator = DataGenerator(config)
    
    # Generate and save dataset
    dataset = generator.generate_dataset()
    generator.save_data(dataset)