from pathlib import Path

from mmseg.core.evaluation import get_palette

from mmseg.core import get_classes


def read_lines_from(filename):
    path = Path('coarse-classes') / filename
    with open(path) as f:
        lines = filter(lambda s: len(s) > 0, map(lambda s: s.strip(), f.readlines()))
    return set(lines)


def cocostuff_crude_palette():
    PERSON = 0
    BOAT = 1
    NATURAL = 2
    ANIMALS = 3
    EQUIPMENT = 4
    ARTIFACTS = 5
    FOOD = 6
    BUILDINGS = 7
    SKY = 8
    WATER = 9

    coarse_labels = {
        PERSON: ["person"],
        BOAT: ["boat"],
        NATURAL: read_lines_from('natural.txt'),
        ANIMALS: read_lines_from('animals.txt'),
        EQUIPMENT: read_lines_from('equipment.txt'),
        ARTIFACTS: read_lines_from('artifacts.txt'),
        FOOD: read_lines_from('food.txt'),
        BUILDINGS: read_lines_from('buildings.txt'),
        SKY: ["clouds", "sky-other"],
        WATER: ["river", "sea", "water-other", "waterdrops"]
    }

    coarse_palette = {
        PERSON: [192, 192, 0],
        BOAT: [192, 100, 0],
        NATURAL: [0, 192, 0],
        ANIMALS: [50, 100, 0],
        EQUIPMENT: [192, 0, 192],
        ARTIFACTS: [100, 100, 100],
        FOOD: [192, 0, 0],
        BUILDINGS: [100, 50, 0],
        SKY: [0, 255, 255],
        WATER: [0, 0, 255],
    }

    def fine_to_coarse(label:str):
        for idx, synonims in coarse_labels.items():
            if label in synonims:
                return coarse_palette[idx]
        raise ValueError("Unknown class", label)

    my_palette = get_palette('cocostuff')

    for idx, cls_name in enumerate(get_classes('cocostuff')):
        my_palette[idx] = fine_to_coarse(cls_name)

    return my_palette

