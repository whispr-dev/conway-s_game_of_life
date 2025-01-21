import tkinter as tk
from tkinter import ttk
import random

class LifeVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Conway's Game of Life")

        # Variables
        self.cell_size = 8  # Size of each cell in pixels
        self.width = 75  # Number of cells horizontally
        self.height = 50  # Number of cells vertically
        self.grid = [[0] * self.width for _ in range(self.height)]
        self.running = False

        # Create controls
        self.create_controls()

        # Create canvas for grid display
        self.canvas = tk.Canvas(
            root,
            width=self.width * self.cell_size,
            height=self.height * self.cell_size,
            bg='white'
        )
        self.canvas.pack(padx=10, pady=10)

        # Initialize random pattern
        self.randomize_grid()

    def create_controls(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(padx=10, pady=5)

        # Start button
        ttk.Button(
            control_frame,
            text="Start",
            command=self.start_simulation
        ).pack(side=tk.LEFT, padx=5)

        # Stop button
        ttk.Button(
            control_frame,
            text="Stop",
            command=self.stop_simulation
        ).pack(side=tk.LEFT, padx=5)

        # Randomize button
        ttk.Button(
            control_frame,
            text="Randomize",
            command=self.randomize_grid
        ).pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_label = ttk.Label(control_frame, text="Alive Cells: 0")
        self.status_label.pack(side=tk.LEFT, padx=5)

    def randomize_grid(self):
        """Randomize the initial grid."""
        self.grid = [[random.choice([0, 1]) for _ in range(self.width)] for _ in range(self.height)]
        self.update_canvas()

    def start_simulation(self):
        """Start the Game of Life simulation."""
        self.running = True
        self.run_simulation()

    def stop_simulation(self):
        """Stop the Game of Life simulation."""
        self.running = False

    def run_simulation(self):
        """Run the simulation step by step."""
        if not self.running:
            return

        self.grid = self.next_generation()
        self.update_canvas()

        # Schedule next step
        self.root.after(100, self.run_simulation)

    def next_generation(self):
        """Calculate the next generation of the grid."""
        new_grid = [[0] * self.width for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                alive_neighbors = self.count_alive_neighbors(x, y)

                if self.grid[y][x] == 1:  # Alive cell
                    if alive_neighbors in [2, 3]:
                        new_grid[y][x] = 1
                else:  # Dead cell
                    if alive_neighbors == 3:
                        new_grid[y][x] = 1

        return new_grid

    def count_alive_neighbors(self, x, y):
        """Count the number of alive neighbors for a cell."""
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),         (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        count = 0
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                count += self.grid[ny][nx]
        return count

    def update_canvas(self):
        """Update the canvas to reflect the current grid."""
        self.canvas.delete("all")
        alive_cells = 0
        for y in range(self.height):
            for x in range(self.width):
                alive_neighbors = self.count_alive_neighbors(x, y)
                if self.grid[y][x] == 1:
                    alive_cells += 1
                    self.draw_cell(x, y, alive_neighbors)

        self.status_label.config(text=f"Alive Cells: {alive_cells}")

    def draw_cell(self, x, y, alive_neighbors):
        """Draw a single cell on the canvas with color based on surrounding cells."""
        if alive_neighbors > 5:
            color = "blue"
        elif alive_neighbors > 3:
            color = "white"
        elif alive_neighbors > 1:
            color = "green"
        elif alive_neighbors == 1:
            color = "yellow"
        else:
            color = "red"

        self.canvas.create_rectangle(
            x * self.cell_size,
            y * self.cell_size,
            (x + 1) * self.cell_size,
            (y + 1) * self.cell_size,
            fill=color,
            outline=''
        )

def main():
    root = tk.Tk()
    app = LifeVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
