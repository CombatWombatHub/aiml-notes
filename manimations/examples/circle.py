"""from https://docs.manim.community/en/stable/tutorials/quickstart.html
animates drawing a circle, saves as video file, previews in default movie viewer
"""

from manim import Scene, Circle, PINK, Create #type: ignore

# all animations go within the construct method of a Scene subclass
# other code like auxiliary or mathematical functions can live outside the class
class CreateCircle(Scene): # create a subclass of Scene
    def construct(self): # define the construct method
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set its color and transparency
        self.play(Create(circle))  # show the circle on screen

# create video file in the media directory, preview in default movie viewer
# could also run command: 
# manim -pqh circle.py CreateCircle
if __name__== "__main__":
    from manim import tempconfig
    with tempconfig({"quality": "high_quality", "preview": True}):
        scene = CreateCircle()
        scene.render()
