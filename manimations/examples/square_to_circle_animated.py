"""from https://docs.manim.community/en/stable/tutorials/quickstart.html
draws a square, rotates it, turns into a circle, then fills it with color
shows animation with the .animate syntax
"""

from manim import Scene, Circle, Create, Square, PI, Transform, PINK, tempconfig #type: ignore

class AnimatedSquareToCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        square = Square()  # create a square

        self.play(Create(square))  # show the square on screen
        self.play(square.animate.rotate(PI / 4))  # rotate the square
        self.play(Transform(square, circle))  # transform the square into a circle
        self.play(square.animate.set_fill(PINK, opacity=0.5))  # color the circle on screen

# high quality takes longer to render, probably larger file size, 
# but has a much higher framerate
# manim -pqh square_to_circle_animated.py AnimatedSquareToCircle
if __name__== "__main__":
    with tempconfig({"quality": "high_quality", "preview": True}):
        scene = AnimatedSquareToCircle()
        scene.render()