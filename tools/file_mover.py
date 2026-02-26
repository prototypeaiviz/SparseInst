from loguru import logger
from pathlib import Path
import shutil
import json


def frame_mover_based_json(json_list, src_dirs, dest_base_dir):
    src_paths = [Path(s) for s in src_dirs]
    dest_base = Path(dest_base_dir)

    for json_file_path in json_list:
        json_file = Path(json_file_path)

        current_dest_dir = dest_base / json_file.stem / "imgs"
        current_dest_dir.mkdir(parents=True, exist_ok=True)

        if not json_file.exists():
            logger.error(f"JSON file not found: {json_file}")
            continue

        with open(json_file, 'r') as f:
            data = json.load(f)

        filenames_to_copy = [img['file_name'] for img in data.get('images', [])]
        logger.info(f"Processing {json_file.name}: Found {len(filenames_to_copy)} images.")

        copied_count = 0
        for file_name in filenames_to_copy:
            found = False
            for source in src_paths:
                source_file = source / file_name

                if source_file.exists():
                    shutil.copy2(source_file, current_dest_dir / file_name)
                    copied_count += 1
                    found = True
                    break

            if not found:
                logger.warning(f"File {file_name} (from {json_file.name}) not found in any source.")

        logger.success(f"Done! Copied {copied_count} images to {current_dest_dir}")


if __name__ == '__main__':

    json_paths = [
        '/home/mehran/Desktop/SparseInst_MONO_labelings/train.json',
        '/home/mehran/Desktop/SparseInst_MONO_labelings/val.json',
        '/home/mehran/Desktop/SparseInst_MONO_labelings/test.json'
    ]
    sources = ['/home/mehran/Desktop/Detectron/train/images', '/home/mehran/Desktop/Detectron/val/images', '/home/mehran/Desktop/Detectron/test/images']
    destination = '/home/mehran/Desktop/SparseInst_MONO_Images'

    frame_mover_based_json(json_paths, sources, destination)