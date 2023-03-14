import numpy

BG_MOSS:    tuple = (67, 77, 0, 255)
BG_GRASS_1: tuple = (89, 102, 0, 255)
BG_GRASS_2: tuple = (111, 127, 0, 255)
BG_GRASS_3: tuple = (133, 153, 0, 255)
BG_GRASS_4: tuple = (155, 179, 0, 255)
BG_SAND:    tuple = (178, 204, 0, 255)
BG_PATH:    tuple = (200, 229, 0, 255)

BG_WATERFALL:  tuple = (3, 0, 77, 255)
BG_WHIRLPOOL:  tuple = (4, 0, 102, 255)
BG_CURRENT_NS: tuple = (5, 0, 127, 255)
BG_CURRENT_EW: tuple = (6, 0, 153, 255)
BG_DEEP_WATER: tuple = (7, 0, 179, 255)
BG_WATER_1:    tuple = (8, 0, 204, 255)
BG_WATER_2:    tuple = (9, 0, 229, 255)
BG_LAKE:       tuple = (10, 0, 255, 255)

BG_ROCK_PATH:  tuple = (77, 0, 0, 255)
BG_STAIRS:     tuple = (102, 0, 0, 255)
BG_BRIDGE:     tuple = (127, 0, 0, 255)
BG_PAVED_PATH: tuple = (153, 0, 0, 255)
BG_EMPTY_PATH: tuple = (179, 0, 0, 255)
BG_RUG_PATH:   tuple = (204, 0, 0, 255)
BG_SANDY_PATH: tuple = (229, 0, 0, 255)
BG_ROCK_EDGE:  tuple = (255, 0, 0, 255)

BG_PUDDLE:     tuple = (11, 30, 77, 255)
BG_TIDAL_EDGE: tuple = (14, 40, 102, 255)

BG_CLIFF:         tuple = (77, 50, 0, 255)
BG_BOULDER:       tuple = (102, 67, 0, 255)
BG_BLOCKER_1:     tuple = (127, 83, 0, 255)
BG_BLOCKER_2:     tuple = (153, 100, 0, 255)
BG_LARGE_BLOCKER: tuple = (204, 134, 0, 255)
BG_LARGE_TREE:    tuple = (229, 150, 0, 255)
BG_TREE:          tuple = (255, 167, 0, 255)

BG_FENCE:       tuple = (127, 37, 18, 255)
BG_FURNITURE:   tuple = (179, 53, 25, 255)
BG_BUILDING:    tuple = (204, 60, 29, 255)
BG_TOP_LAYER:   tuple = (229, 67, 32, 255)
BG_SECRET_BASE: tuple = (255, 75, 36, 255)

BG_COLOR_TO_DATA: map(tuple, numpy.int8) = {
    BG_MOSS:    1,
    BG_GRASS_1: 2,
    BG_GRASS_2: 3,
    BG_GRASS_3: 4,
    BG_GRASS_4: 5,
    BG_SAND:    6,
    BG_PATH:    7,

    BG_WATERFALL:  8,
    BG_WHIRLPOOL:  9,
    BG_CURRENT_NS: 10,
    BG_CURRENT_EW: 11,
    BG_DEEP_WATER: 12,
    BG_WATER_1:    13,
    BG_WATER_2:    14,
    BG_LAKE:       15,

    BG_ROCK_PATH:  16,
    BG_STAIRS:     17,
    BG_BRIDGE:     18,
    BG_PAVED_PATH: 19,
    BG_EMPTY_PATH: 20,
    BG_RUG_PATH:   21,
    BG_SANDY_PATH: 22,
    BG_ROCK_EDGE:  23,

    BG_PUDDLE:     24,
    BG_TIDAL_EDGE: 25,

    BG_CLIFF:         26,
    BG_BOULDER:       27,
    BG_BLOCKER_1:     28,
    BG_BLOCKER_2:     29,
    BG_LARGE_BLOCKER: 30,
    BG_LARGE_TREE:    31,
    BG_TREE:          32,

    BG_FENCE:       33,
    BG_FURNITURE:   33,
    BG_BUILDING:    33,
    BG_TOP_LAYER:   33,
    BG_SECRET_BASE: 33,
}

###############################################

GRASS_1_RGB:    tuple = (0, 153, 53, 255)
GRASS_2_RGB:    tuple = (0, 179, 62, 255)
GRASS_TALL_RGB: tuple = (0, 77, 26, 255)

GRASS_COLOR_TO_DATA: map(tuple, numpy.int8) = {
    # TODO: Does using disparate numbers here matter? Should I be using 1, 2, 3
    # or is having more distance between the numbers helpful for training?
    # Answer: Probably not. Simply bounding the possible output values via
    # the activation function could be ok?
    GRASS_1_RGB:    1,
    GRASS_2_RGB:    100,
    GRASS_TALL_RGB: 200
}

###############################################

SURFACE_BIKE_PATH:  tuple = (39, 0, 77, 255)
SURFACE_BIKE_SLOPE: tuple = (52, 0, 102, 255)
SURFACE_FLOWER_1:   tuple = (65, 0, 127, 255)
SURFACE_FLOWER_2:   tuple = (79, 0, 153, 255)

SURFACE_LEDGE:            tuple = (0, 77, 77, 255)
SURFACE_BERRY:            tuple = (0, 102, 102, 255)
SURFACE_ITEM:             tuple = (0, 127, 127, 255)
SURFACE_CUT_TREE:         tuple = (0, 153, 153, 255)
SURFACE_SIGN:             tuple = (0, 179, 179, 255)
SURFACE_SMASH_ROCK:       tuple = (0, 204, 204, 255)
SURFACE_BLOCKING_POKEMON: tuple = (0, 229, 229, 255)
SURFACE_POKEBALL_TREE:    tuple = (0, 255, 255, 255)

SURFACE_HIDDEN_ITEM: tuple = (213, 0, 229, 255)
SURFACE_WARP_ZONE:   tuple = (237, 0, 255)

SURFACE_COLOR_TO_DATA:  map(tuple, numpy.int8) = {
    SURFACE_BIKE_PATH:  1,
    SURFACE_BIKE_SLOPE: 2,
    SURFACE_FLOWER_1:   3,
    SURFACE_FLOWER_2:   4,

    SURFACE_LEDGE:             5,
    SURFACE_BERRY:             6,
    SURFACE_ITEM:              7,
    SURFACE_CUT_TREE:          8,
    SURFACE_SIGN:              9,
    SURFACE_SMASH_ROCK:       10,
    SURFACE_BLOCKING_POKEMON: 11,
    SURFACE_POKEBALL_TREE:    12,

    SURFACE_HIDDEN_ITEM: 13,
    SURFACE_WARP_ZONE:   14,
}

###############################################

TRAINER_WALK_AROUND:    tuple = (255, 161, 192, 255)
TRAINER_RUN_UP_DOWN:    tuple = (229, 145, 173, 255)
TRAINER_RUN_LEFT_RIGHT: tuple = (229, 145, 173, 255)
TRAINER_RUN_CCW:        tuple = (179, 113, 134, 255)
TRAINER_RUN_CW:         tuple = (153, 97, 115, 255)
TRAINER_RUN_UD: tuple = TRAINER_RUN_UP_DOWN
TRAINER_RUN_LR: tuple = TRAINER_RUN_LEFT_RIGHT
TRAINER_RUN_COUNTER_CLOCKWISE: tuple = TRAINER_RUN_CCW
TRAINER_RUN_CLOCKWISE: tuple = TRAINER_RUN_CW

TRAINER_HIDDEN:  tuple = (222, 156, 255, 255)
TRAINER_GENERIC: tuple = (67, 47, 77, 255)

TRAINER_LOOK_AROUND:     tuple = (67, 47, 77, 255)
TRAINER_LOOK_UP_DOWN:    tuple = (192, 229, 126, 255)
TRAINER_LOOK_LEFT_RIGHT: tuple = (170, 204, 112, 255)
TRAINER_LOOK_CCW:        tuple = (149, 179, 98, 255)
TRAINER_LOOK_CW:         tuple = (128, 153, 84, 255)
TRAINER_LOOK_UD: tuple = TRAINER_LOOK_UP_DOWN
TRAINER_LOOK_LR: tuple = TRAINER_LOOK_LEFT_RIGHT
TRAINER_LOOK_COUNTER_CLOCKWISE: tuple = TRAINER_LOOK_CCW
TRAINER_LOOK_CLOCKWISE: tuple = TRAINER_LOOK_CW

TRAINER_LOOK_DOWN:       tuple = (255, 156, 108, 255)
TRAINER_LOOK_UP:         tuple = (229, 140, 97, 255)
TRAINER_LOOK_LEFT:       tuple = (204, 125, 860, 255)
TRAINER_LOOK_RIGHT:      tuple = (179, 109, 76, 255)
TRAINER_LOOK_UP_RIGHT:   tuple = (153, 94, 65, 255)
TRAINER_LOOK_UP_LEFT:    tuple = (127, 78, 54, 255)
TRAINER_LOOK_DOWN_RIGHT: tuple = (102, 62, 43, 255)
TRAINER_LOOK_DOWN_LEFT:  tuple = (77, 47, 32, 255)

TRAINER_COLOR_TO_DATA:  map(tuple, numpy.int8) = {
    TRAINER_WALK_AROUND:    1,
    TRAINER_RUN_UP_DOWN:    2,
    TRAINER_RUN_LEFT_RIGHT: 3,
    TRAINER_RUN_CCW:        4,
    TRAINER_RUN_CW:         5,

    TRAINER_HIDDEN:  6,
    TRAINER_GENERIC: 7,

    TRAINER_LOOK_AROUND:      8,
    TRAINER_LOOK_UP_DOWN:     9,
    TRAINER_LOOK_LEFT_RIGHT: 10,
    TRAINER_LOOK_CCW:        11,
    TRAINER_LOOK_CW:         12,

    TRAINER_LOOK_DOWN:       13,
    TRAINER_LOOK_UP:         14,
    TRAINER_LOOK_LEFT:       15,
    TRAINER_LOOK_RIGHT:      16,
    TRAINER_LOOK_UP_RIGHT:   17,
    TRAINER_LOOK_UP_LEFT:    18,
    TRAINER_LOOK_DOWN_RIGHT: 18,
    TRAINER_LOOK_DOWN_LEFT:  20,
}

###############################################

WALKABLE:     tuple = (3, 255, 0, 255)
SWIMMABLE:    tuple = (0, 179, 255, 255)
IMPASSABLE:   tuple = (255, 0, 0, 255)
SEMIPASSABLE: tuple = (255, 255, 0, 255)

PASSABLE_COLOR_TO_DATA:  map(tuple, numpy.int8) = {
    WALKABLE:     1,
    SWIMMABLE:    2,
    IMPASSABLE:   3,
    SEMIPASSABLE: 4,
}
