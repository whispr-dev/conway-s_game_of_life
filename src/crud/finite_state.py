import golly as g

# Helper functions to draw basic components
def place_glider(x, y, direction="SE"):
    """Places a glider at the specified location facing the given direction."""
    glider = {
        "SE": [(0, 0), (1, 1), (1, 2), (2, 0), (2, 1)],
        "SW": [(0, 2), (1, 1), (1, 2), (2, 0), (2, 1)],
        "NE": [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)],
        "NW": [(0, 1), (0, 2), (1, 0), (1, 2), (2, 1)],
    }
    for dx, dy in glider[direction]:
        g.setcell(x + dx, y + dy, 1)

def place_block(x, y):
    """Places a 2x2 block at the specified location."""
    for dx in range(2):
        for dy in range(2):
            g.setcell(x + dx, y + dy, 1)

def place_glider_gun(x, y):
    """Places a Gosper Glider Gun at the specified location."""
    gun_pattern = [
        (0, 4), (0, 5), (1, 4), (1, 5),
        (10, 4), (10, 5), (10, 6), (11, 3), (11, 7), (12, 2),
        (12, 8), (13, 2), (13, 8), (14, 5), (15, 3), (15, 7),
        (16, 4), (16, 5), (16, 6), (17, 5), (20, 2), (20, 3),
        (20, 4), (21, 2), (21, 3), (21, 4), (22, 1), (22, 5),
        (24, 0), (24, 1), (24, 5), (24, 6), (34, 2), (34, 3),
        (35, 2), (35, 3)
    ]
    for dx, dy in gun_pattern:
        g.setcell(x + dx, y + dy, 1)

def create_fsm_with_counters():
    """Creates an FSM connected to two counters."""
    g.new("FSM with Counters")

    # Place glider guns as state machines
    place_glider_gun(10, 10)  # State 1 glider gun
    place_glider_gun(50, 10)  # State 2 glider gun

    # Place counters (blocks to count gliders)
    place_block(80, 20)  # Counter 1
    place_block(80, 50)  # Counter 2

    # Add glider streams connecting the FSM to counters
    place_glider(30, 20, direction="SE")  # Connect State 1 to Counter 1
    place_glider(30, 50, direction="SE")  # Connect State 2 to Counter 2

    # Display the created pattern
    g.fit()
    g.show("FSM with two counters created!")

if __name__ == "__main__":
    create_fsm_with_counters()
