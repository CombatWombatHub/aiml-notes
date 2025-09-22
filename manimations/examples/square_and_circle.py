"""from https://docs.manim.community/en/stable/tutorials/quickstart.html
draws a square and a circle next to each other
shows positioning with next_to
"""

from manim import Scene, Circle, PINK, Square, BLUE, RIGHT, Create, tempconfig #type: ignore

class SquareAndCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set the color and transparency

        square = Square()  # create a square
        square.set_fill(BLUE, opacity=0.5)  # set the color and transparency

        square.next_to(circle, RIGHT, buff=0.5)  # move the circle to the right
        self.play(Create(circle), Create(square))  # show the shapes on screen

# manim -pqh square_and_circle.py SquareAndCircle
if __name__== "__main__":
    with tempconfig({"quality": "high_quality", "preview": True}):
        scene = SquareAndCircle()
        scene.render()
