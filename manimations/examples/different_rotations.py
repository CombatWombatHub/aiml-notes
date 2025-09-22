"""from https://docs.manim.community/en/stable/tutorials/quickstart.html
draw two squares, then show the quirks of .animate.rotate vs Rotate
.animate.rotate makes one get small and big, 
Rotate makes the other one actually rotate
"""

from manim import Scene, Square, Rotate, BLUE, GREEN, PI, LEFT, RIGHT, tempconfig #type: ignore

class DifferentRotations(Scene):
    def construct(self):
        # create square and move it left
        left_square = Square(color=BLUE, fill_opacity=0.7).shift(2 * LEFT)
        # create square and move it right
        right_square = Square(color=GREEN, fill_opacity=0.7).shift(2 * RIGHT)
        # animate the squares - show the quirks of the .animate.rotate vs Rotate
        self.play(
            left_square.animate.rotate(PI), # makes it get small and then big again
            Rotate(right_square, angle=PI), # makes it rotate 360 degrees CCW
            run_time=2 # take 2 seconds to do the animation
        )
        self.wait()

# manim -pqh different_rotations.py DifferentRotations
if __name__== "__main__":
    with tempconfig({"quality": "high_quality", "preview": True}):
        scene = DifferentRotations()
        scene.render()
