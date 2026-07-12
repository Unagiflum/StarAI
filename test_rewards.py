REWARD_VALUES = tuple(
    [-40.96, -20.48, -10.24, -5.12, -2.56, -1.28, -0.64, -0.32, -0.16, -0.08, -0.04, -0.02, -0.01]
    + [0.0]
    + [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96]
)
defaults = {
    "Kill enemy": 40.96,
    "Enemy loses crew": 5.12,
    "Get debuffed": -1.28,
    "Lose crew": -2.56,
    "Die": -20.48,
}
for k, v in defaults.items():
    print(f"{k}: {v} in REWARD_VALUES -> {v in REWARD_VALUES}")
