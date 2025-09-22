"""from https://docs.manim.community/en/stable/tutorials/quickstart.html
draws a circle, transforms it into a square, then into a triangle
shows how to use Transform to cycle through transformations
without needing to keep a reference to each mobject in the transform cycle. 
"""


from manim import Scene, Circle, Square, Triangle, Transform, tempconfig #type: ignore

class TransformCycle(Scene):
    def construct(self):
        # creat the 3 shapes to transform between
        a = Circle()
        t1 = Square()
        t2 = Triangle()
        # do the initial cration
        self.add(a) # show the first shape right off the bat
        self.wait() # wait 1 second before first transformation
        # cycle through the transformations
        # can just keep transforming "a" since Transform changes the original
        for t in [t1,t2]:
            self.play(Transform(a,t))

# manim -pqh transform_cycle.py TransformCycle
if __name__== "__main__":
    with tempconfig({"quality": "high_quality", "preview": True}):
        scene = TransformCycle()
        scene.render()