from config import LAMBDA2_VALUES, dataset_config
from create_data import DataGenerator

if __name__ == "__main__":

    # One dataset per fixed transverse stretch lambda2. save_data() refuses to
    # overwrite (it appends _1, _2, ... instead), so delete the old pickles
    # before regenerating if the traction convention changed.
    for lambda2 in LAMBDA2_VALUES:
        config = dataset_config(lambda2)
        generator = DataGenerator(config)

        # Generate and save dataset
        dataset = generator.generate_dataset()
        generator.save_data(dataset)
