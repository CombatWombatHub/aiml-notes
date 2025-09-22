"""from https://docs.manim.community/en/stable/tutorials/quickstart.html
draws a square, morphs it into a circle, then fades out
shows rotation, morphing with Transform, fading with FadeOut
"""

from manim import Scene, Circle, PINK, Square, PI, Transform, FadeOut, Create, tempconfig #type: ignore

class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set color and transparency

        square = Square()  # create a square
        square.rotate(PI / 4)  # rotate the square by 45 degrees (pi/4 radians)

        self.play(Create(square))  # animate the creation of the square
        self.play(Transform(square, circle))  # interpolate the square into the circle
        self.play(FadeOut(square))  # fade out animation

# manim -pqh square_to_circle.py SquareToCircle
if __name__== "__main__":
    with tempconfig({"quality": "high_quality", "preview": True}):
        scene = SquareToCircle()
        scene.render()
