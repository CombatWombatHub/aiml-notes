from manim import Scene, Circle, PINK, Create #type: ignore

# from https://docs.manim.community/en/stable/tutorials/quickstart.html
# all animations go within the construct method of a Scene subclass
# other code like auxiliary or mathematical functions can live outside the class
class CreateCircle(Scene): # create a subclass of Scene
    def construct(self): # define the construct method
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set its color and transparency
        self.play(Create(circle))  # show the circle on screen


# the code can be run from the command line by running
# manim -pql example.py CreateCircle
# but also like this:
if __name__== "__main__":
    from manim import tempconfig
    # set low quality and disable preview window for faster rendering
    with tempconfig({"quality": "low_quality", "preview": False}):
        scene = CreateCircle()
        scene.render()
# this will create a video file in the media directory