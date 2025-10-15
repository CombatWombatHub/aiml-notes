# LaTeX Equations
- Some notes and examples of complex LaTeX equations for reference

## Markdown and Equation-Writing Reference Material
- Markdown
    - A review of [Markdown Content](https://aimlabs.us.lmco.com/content/new) such as tables, etc
    - suggests you don't use the first level of heading? may just be for online content
    - Note - you can use &#x2611;, &#x2610;, and &#x2612; to indicate completion or failure etc
- Markdown Equations
    - AI Factory notebooks recognize KaTeX. Some LaTeX symbols work, some don't
    - These websites let you sketch a symbol and figure out the matching [LaTeX symbol](http://detexify.kirelabs.org/classify.html) or [Unicode Character](http://shapecatcher.com/)
- KaTeX formulas
    - here's a really good reference for [what to type to get something in a KaTeX formula](https://katex.org/docs/supported.html)
    - here's a list of symbols and functions that are [available and not available in KaTeX](https://utensil-site.github.io/available-in-katex/)
    - and KaTeX's own [list of what they do and don't support](https://katex.org/docs/support_table.html)
- LaTeX symbols
    - [Finding LaTex Mathematical Symbols](https://tex.stackexchange.com/questions/14/how-to-look-up-a-symbol-or-identify-a-letter-from-a-math-alphabet-or-other-chara) ($\varphi$ = \varphi etc)
    - Good PDF for some of them [Latex Symbol Glossary](https://www.cmor-faculty.rice.edu/~heinken/latex/symbols.pdf)
    - Huge [Comprehensive LaTex Symbols List](http://mirrors.ctan.org/info/symbols/comprehensive/symbols-a4.pdf)

## Some cool fancy equations

- where $\delta_{ij}=$ Kronecker delta
    - $\delta_{ij}=\begin{vmatrix}1&0&0\\0&1&0\\0&1&1\end{vmatrix}=\begin{cases}\delta_{ij}=1&\text{if}&i=j\\\delta_{ij}=0&\text{if}&i\ne j\end{cases}$
- you can expand the general stress strain relation into
    - $ $
$\begin{array}{ll}
\sigma_{11}=2\mu\epsilon_{11}+\lambda(\epsilon_{11}+\epsilon_{22}+\epsilon_{33}) & \delta_{11}=1 \\
\sigma_{22}=2\mu\epsilon_{22}+\lambda(\epsilon_{11}+\epsilon_{22}+\epsilon_{33}) & \delta_{22}=1 \\
\sigma_{33}=2\mu\epsilon_{33}+\lambda(\epsilon_{11}+\epsilon_{33}+\epsilon_{33}) & \delta_{33}=1 \\
\sigma_{12}=2\mu\epsilon_{12}                                                    & \delta_{12}=0 \\
\sigma_{23}=2\mu\epsilon_{23}                                                    & \delta_{23}=0 \\
\sigma_{31}=2\mu\epsilon_{31}                                                    & \delta_{31}=0
\end{array}$
- if I understand the above correctly, then my Galerkin Condition would be
    - $\int\limits_{-a/2}^{a/2}\int\limits_{-b/2}^{b/2} \underbrace{\left[ D \left( {\partial^4 w\over\partial x^4} + {\partial^4 w\over\partial y^4} + 2{\partial^4 w\over\partial x^2\partial y^2} \right) + N_x{\partial^2 w\over\partial x^2} + N_y{\partial^2 w\over\partial y^2} + 2N_{xy}{\partial^2 w\over\partial x\partial y} \right]}_{= R(\boldsymbol{x}) \textsf{= residual function = governing equation}} \overbrace{\underbrace{\quad\quad w \quad\quad}_{\begin{matrix}=\scriptsize\text{deflection}\\\scriptsize\text{function}\end{matrix}}}^{\begin{matrix}=\scriptsize\text{weighting}\\\scriptsize\text{function}\end{matrix}} ~dx~dy = 0$
    - comparison of the Loss Equations written in the Modulus Documentation:
- $ $
$
\underbrace{
\overbrace{
\begin{array}{lllllr}
L=\Big|\textcolor{pink}{\big\lgroup}\int_{\Omega}         & u_{net}(x,y,z)                   & -\phi\textcolor{pink}{\big\rgroup} & \Big|^p=\Big|\textcolor{pink}{\Big\lgroup}\frac{V}{B}\sum_{i}u_{net}(x_i, y_i, z_i)                  & -\phi & \textcolor{pink}{\Big\rgroup}\Big|^p \\
L=\Big|\textcolor{pink}{\big\lgroup}\int_{\partial\Omega} & u_{net}(x,y,z)\textcolor{pink}{\big\rgroup} & -\phi                   & \Big|^p=\Big|\textcolor{pink}{\Big\lgroup}\frac{S}{B}\sum_{i}u_{net}(x_i, y_i, z_i)\textcolor{pink}{\Big\rgroup} & -\phi &                  \Big|^p
\end{array}
}^{\textsf{PointwiseInteriorConstraint - network prediction at each point is compared to the true value at each point}}
}_{\textsf{IntegralBoundaryConstraint  - integral of network predictions compared to true integral value}}
$
- starting from the three groups of equations for the plate bending problem from Lecture 2,3,4:
    - $\begin{array}{ll} \textsf{Geometry:} & k_{\alpha\beta}=-w_{,\alpha\beta} \\ \textsf{Equilibrium:} & M_{\alpha\beta,\alpha\beta}+p=0 \\ \textsf{Elasticity:} & M_{\alpha\beta}=D\big[(1-\nu)k_{\alpha\beta}+\nu k_{\gamma\gamma}\delta_{\alpha\beta}\big] \end{array}$
- it's not about finding the equilibrium point itself so much as figuring out how stable that equilibrium is
    - $\delta^2\Pi\begin{cases} >0\to\textsf{Stable Equilibrium} \\ =0\to\textsf{Neutral Equilibrium} \\ <0\to\textsf{Unstable Equilibrium} \end{cases}$
- take the second variation to figure out stability
        - $\delta^2\Pi=(K-Pl)\delta\theta^2\xrightarrow[\textsf{to be stable}]{\textsf{must be positive}}\begin{cases} P&<&{k \over l} \\ |\delta\theta|&>&0 \end{cases}$
- Will need more derivatives
        - $\begin{aligned} w''(x) & =\phi''(x)  & = &&-({\pi\over l})^2sin{\pi x\over l} \\ w'''(x) & =\phi'''(x)  & = &&-({\pi\over l})^3cos{\pi x\over l} \\ w''''(x) & =\phi''''(x)  & =&& ({\pi\over l})^4sin{\pi x\over l} \end{aligned}$
        - plug all that into the equilibrium equation
            - $ $
$\begin{aligned}
& EIw'''' && +N && w'' & =0 \\
& EI\left(({\pi\over l})^4sin{\pi x\over l}\right) && +\left({\pi^2EI \over l^2}\right) && \left(-({\pi\over l})^2sin{\pi x\over l}\right) & =0 \\
& EI\left(({\pi\over l})^4sin{\pi x\over l}\right) && -EI({\pi\over l})^2 && ({\pi\over l})^2sin{\pi x\over l} & =0
\end{aligned}$