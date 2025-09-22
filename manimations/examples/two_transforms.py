"""from https://docs.manim.community/en/stable/tutorials/quickstart.html
transform a circle into a square into a triangle, then fade out

show the difference between Transform and ReplacementTransform
Transform(mob1, mob2) transforms the points and some attributes of mob1 into mob2
ReplacementTransform(mob1, mob2) replaces mob1 with mob2

altered so that the two transformations happen side-by-side simultaneously
showing that the end product is the same
apparently which one you use is mostly up to personal preference
"""

from manim import Scene, Circle, Square, Triangle, Transform, ReplacementTransform, FadeOut, LEFT, RIGHT, tempconfig #type: ignore

class TwoTransforms(Scene):
    def construct(self):
        # create the 3 shapes for Transform, move left
        a = Circle().shift(2 * LEFT)
        b = Square().shift(2 * LEFT)
        c = Triangle().shift(2 * LEFT)

        # create the 3 shapes for ReplacementTransform, move right
        d = Circle().shift(2 * RIGHT)
        e = Square().shift(2 * RIGHT)
        f = Triangle().shift(2 * RIGHT)

        # do the corresponding transformations simultaneously
        self.play(Transform(a, b), ReplacementTransform(d, e))
        self.play(Transform(a, c), ReplacementTransform(e, f))
        self.play(FadeOut(a), FadeOut(f))

# manim -pqh two_transforms.py TwoTransforms
if __name__== "__main__":
    with tempconfig({"quality": "high_quality", "preview": True}):
        scene = TwoTransforms()
        scene.render()
