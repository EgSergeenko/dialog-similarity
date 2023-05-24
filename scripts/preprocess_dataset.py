import json
import os

import click
import pandas as pd


@click.command()
@click.option(
    '--input_dir',
    'input_dir',
    required=False,
    default='dstc8-schema-guided-dialogue',
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    '--output_dir',
    'output_dir',
    required=False,
    default='conversation-similarity',
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    '--metadata_path',
    'metadata_path',
    required=False,
    default='conversation-similarity/conved.csv',
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
def preprocess_dataset(
    input_dir: str, output_dir: str, metadata_path: str,
) -> None:
    sub_dirs = ['train']

    os.makedirs(os.path.join(output_dir, 'dialogs'), exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    dialog_ids = set()
    for column in ['anchor_conv', 'conv_1', 'conv_2']:
        dialog_ids.update(metadata[column].tolist())

    dialogs_filepaths = []
    for sub_dir in sub_dirs:
        dialogs_dir = os.path.join(input_dir, sub_dir)
        dialogs_filenames = os.listdir(dialogs_dir)
        dialogs_filenames.remove('schema.json')
        for dialogs_filename in dialogs_filenames:
            dialogs_filepaths.append(
                os.path.join(dialogs_dir, dialogs_filename),
            )

    for dialogs_filepath in dialogs_filepaths:
        with open(dialogs_filepath) as dialogs_file:
            dialogs_data = json.load(dialogs_file)
        for dialog in dialogs_data:
            dialog_id = dialog['dialogue_id']
            if dialog_id in dialog_ids:
                dialog_ids.remove(dialog_id)
                output_dialog_filepath = os.path.join(
                    output_dir, 'dialogs', '{0}.json'.format(dialog_id),
                )
                with open(output_dialog_filepath, 'w') as output_dialog_file:
                    json.dump(dialog, output_dialog_file)


if __name__ == '__main__':
    preprocess_dataset()
